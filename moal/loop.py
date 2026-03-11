"""ActiveLearningLoop: orchestrates the full m-iteration acquisition campaign."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from moal.acquisition import CostAwareGreedyAcquisition
from moal.evaluation import ModelMetric, PipelineEvaluator
from moal.logging_config import suppress_noisy_loggers, temporary_log_level
from moal.model import ChemPropLightningModule, NoisyOracleModel
from moal.oracle import CostAwareOracle
from moal.preprocessing import SMILESPreprocessor
from moal.types import IterationResults, LoopResults, QueryType

if TYPE_CHECKING:
    from moal.dashboard import LiveDashboard

logger = logging.getLogger(__name__)

# Rich console used for status messages; writes to stderr so it doesn't mix
# with any stdout output and plays nicely with the progress bar.
_console = Console(stderr=True)


class ActiveLearningLoop:
    """Orchestrates a cost-aware active learning campaign.

    Each iteration executes three discrete steps (tracked individually in the
    progress bar):

    1. **Query oracle** — the acquisition function sends ``k`` queries.
    2. **Refit model** — ChemProp is fine-tuned on the growing labeled pool.
    3. **Select compounds** — model inference + acquisition scoring over the
       entire unlabeled pool to prepare the *next* iteration's queries.

    This separation makes it possible to pre-compute the next iteration's
    candidate list while the user inspects the dashboard.

    Parameters
    ----------
    oracle : CostAwareOracle
        Oracle wrapping the ground-truth dataset.
    model : ChemPropLightningModule or NoisyOracleModel
        Model to be iteratively fine-tuned.
    acquisition : CostAwareGreedyAcquisition
        Acquisition function for query selection.
    evaluator : PipelineEvaluator
        Evaluator for metric computation.
    preprocessor : SMILESPreprocessor, optional
        Used for pre-flight SMILES checks. A default instance is created if
        None is provided.
    trainer_kwargs : dict[str, Any], optional
        Forwarded to ``lightning.Trainer`` at each refit.
    datamodule_kwargs : dict[str, Any], optional
        Forwarded to ``MixedFidelityDataModule`` at each refit (e.g.,
        ``val_fraction``, ``seed``).
    dashboard : LiveDashboard, optional
        If provided, updated after every iteration.
    test_set : tuple[list[str], np.ndarray], optional
        ``(smiles_list, pec50_array)`` held-out scaffold split for model
        performance tracking. If provided, the model metric is computed after
        every refit and shown in the dashboard.
    model_metric : ModelMetric, optional
        Which metric to track for the model performance curve. Default is MAE.
    initial_error : float, optional
        Starting noise magnitude (pEC50 log-units) for the error ramp when
        using ``NoisyOracleModel`` in fast mode. Ignored for
        ``ChemPropLightningModule``. Default is 0.7.
    final_error : float, optional
        Ending noise magnitude (pEC50 log-units) for the error ramp. A linear
        schedule from ``initial_error`` to ``final_error`` is applied over all
        iterations. Set equal to ``initial_error`` for constant noise.
        Default is 0.5.
    reset_weights_on_refit : bool, optional
        When True, pass ``reset_weights=True`` to ``model.refit()`` at every
        active learning iteration. Default is False, which continues
        fine-tuning from the current model weights.
    output_dir : str or Path, optional
        Directory for Lightning default logs (``lightning_logs/``). Forwarded
        as ``output_dir`` to every ``model.refit()`` call. When None (default),
        Lightning writes to the current working directory.
    """

    def __init__(
        self,
        oracle: CostAwareOracle,
        model: ChemPropLightningModule | NoisyOracleModel,
        acquisition: CostAwareGreedyAcquisition,
        evaluator: PipelineEvaluator,
        preprocessor: SMILESPreprocessor | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
        datamodule_kwargs: dict[str, Any] | None = None,
        dashboard: "LiveDashboard | None" = None,
        test_set: tuple[list[str], np.ndarray] | None = None,
        model_metric: ModelMetric = ModelMetric.MAE,
        initial_error: float = 0.7,
        final_error: float = 0.5,
        reset_weights_on_refit: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        self.oracle = oracle
        self.model = model
        self.acquisition = acquisition
        self.evaluator = evaluator
        self.preprocessor = preprocessor or SMILESPreprocessor()
        self.trainer_kwargs = trainer_kwargs or {}
        self.datamodule_kwargs = datamodule_kwargs or {}
        self.dashboard = dashboard
        self.test_set = test_set
        self.model_metric = model_metric
        self.initial_error = initial_error
        self.final_error = final_error
        self.reset_weights_on_refit = reset_weights_on_refit
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, n_iterations: int, k_per_iteration: int) -> LoopResults:
        """Execute the full active learning campaign.

        Parameters
        ----------
        n_iterations : int
            Total number of AL iterations (m).
        k_per_iteration : int
            Number of oracle queries per iteration (k).

        Returns
        -------
        LoopResults
            Per-iteration history and final metrics.
        """
        suppress_noisy_loggers()

        results = LoopResults()
        n_total = len(self.oracle)
        n_true_actives = self.oracle.n_true_actives(self.evaluator.activity_threshold)

        # Build a per-iteration noise schedule for NoisyOracleModel fast mode;
        # linspace(a, a, n) naturally handles the constant-error case.
        noise_schedule: np.ndarray | None = None
        if isinstance(self.model, NoisyOracleModel):
            noise_schedule = np.linspace(
                self.initial_error, self.final_error, n_iterations
            )

        _console.print(
            f"[bold]moal[/bold] campaign starting — "
            f"[cyan]{n_iterations}[/cyan] iterations × "
            f"[cyan]{k_per_iteration}[/cyan] queries | "
            f"[cyan]{n_total}[/cyan] compounds | "
            f"[cyan]{n_true_actives}[/cyan] true actives"
        )

        # Pre-compute first iteration's candidate queries before entering the
        # progress bar so Step 3 of iteration i prepares for iteration i+1.
        # Both the unqueried pool and PS-labeled INTERVAL hits are scorable.
        unlabeled = self.oracle.get_unlabeled_smiles()
        ps_labeled = self.oracle.get_ps_labeled_smiles()
        all_scorable = unlabeled + ps_labeled
        if all_scorable:
            all_preds = self._predict(
                all_scorable, noise_schedule[0] if noise_schedule is not None else None
            )
            unlabeled_preds = all_preds[: len(unlabeled)]
            ps_labeled_preds = all_preds[len(unlabeled) :]
        else:
            unlabeled_preds = ps_labeled_preds = np.array([])
        pending_queries = (
            self.acquisition.select(
                unlabeled,
                unlabeled_preds,
                k_per_iteration,
                ps_labeled_smiles=ps_labeled,
                ps_labeled_predictions=ps_labeled_preds if ps_labeled else None,
            )
            if all_scorable
            else []
        )

        total_steps = n_iterations * 3
        with temporary_log_level(logging.WARNING, ["moal"]):
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=_console,
                transient=False,
            ) as progress:
                task = progress.add_task("Starting…", total=total_steps)

                for iteration in range(n_iterations):
                    queries = pending_queries  # prepared at end of previous iter

                    # --- Query oracle -------------------------------------
                    # Snapshot the PS-labeled pool before querying so planned DRC
                    # queries can be classified as upgrades vs. first-pass in the
                    # progress display without waiting for oracle results.
                    ps_labeled_before = set(self.oracle.get_ps_labeled_smiles())
                    n_drc = sum(1 for _, qt in queries if qt == QueryType.DOSE_RESPONSE)
                    n_ps = sum(1 for _, qt in queries if qt == QueryType.PRIMARY_SCREEN)
                    n_drc_upgrade_planned = sum(
                        1
                        for smi, qt in queries
                        if qt == QueryType.DOSE_RESPONSE and smi in ps_labeled_before
                    )
                    n_drc_new_planned = n_drc - n_drc_upgrade_planned
                    upgrade_drc_label = (
                        f" ([magenta]{n_drc_upgrade_planned} upgrades[/magenta])"
                        if n_drc_upgrade_planned > 0
                        else ""
                    )
                    progress.update(
                        task,
                        description=(
                            f"[cyan]Iter {iteration + 1}/{n_iterations}[/cyan]  "
                            f"Querying oracle — [orange1]{n_drc_new_planned} DRC[/orange1]"
                            f"{upgrade_drc_label}"
                            f", [steel_blue1]{n_ps} PS[/steel_blue1]"
                        ),
                    )
                    # Forward is_canonical so query_batch uses the same key strategy
                    # that was used when building the ground truth dict; omitting it
                    # would cause re-canonicalization to produce keys that don't exist
                    # when the oracle was initialised with is_canonical=True.
                    new_records = self.oracle.query_batch(
                        queries, iteration, is_canonical=self.oracle.is_canonical
                    )
                    # Derive actual per-fidelity costs from records returned by the
                    # oracle — not from the pre-query candidate list, which may
                    # differ if the oracle skips invalid or already-labeled compounds.
                    iter_drc_cost = sum(
                        r.cost
                        for r in new_records
                        if r.fidelity == QueryType.DOSE_RESPONSE
                    )
                    iter_ps_cost = sum(
                        r.cost
                        for r in new_records
                        if r.fidelity == QueryType.PRIMARY_SCREEN
                    )
                    # Upgrades are DRC queries for compounds already in the PS pool;
                    # ps_labeled_before was captured before this iteration's queries.
                    iter_upgrade_cost = sum(
                        r.cost
                        for r in new_records
                        if r.fidelity == QueryType.DOSE_RESPONSE
                        and r.canonical_smiles in ps_labeled_before
                    )
                    progress.advance(task)

                    # --- Refit model --------------------------------------
                    all_labeled = self.oracle.labeled_records
                    n_labeled = len(all_labeled)
                    n_labeled_drc = sum(
                        1 for r in all_labeled if r.fidelity == QueryType.DOSE_RESPONSE
                    )
                    n_labeled_ps = sum(
                        1 for r in all_labeled if r.fidelity == QueryType.PRIMARY_SCREEN
                    )
                    # Count cumulative upgrades from the evaluator breakdown so the
                    # refit message is consistent with the metrics that get logged.
                    n_cumulative_upgrades = int(
                        self.evaluator.fidelity_breakdown(all_labeled).get(
                            "upgrades", 0
                        )
                    )
                    upgrade_suffix = (
                        f", [magenta]{n_cumulative_upgrades} upgrades[/magenta]"
                        if n_cumulative_upgrades > 0
                        else ""
                    )
                    progress.update(
                        task,
                        description=(
                            f"[yellow]Iter {iteration + 1}/{n_iterations}[/yellow]  "
                            f"Retraining model — {n_labeled} records "
                            f"([orange1]{n_labeled_drc} DRC[/orange1], "
                            f"[steel_blue1]{n_labeled_ps} PS[/steel_blue1]"
                            f"{upgrade_suffix})"
                        ),
                    )
                    if new_records:
                        self.model.refit(
                            records=self.oracle.training_records,
                            trainer_kwargs=self.trainer_kwargs,
                            datamodule_kwargs=self.datamodule_kwargs,
                            reset_weights=self.reset_weights_on_refit,
                            output_dir=self.output_dir,
                        )

                    # Evaluate model metric on held-out test set.
                    model_metric_value: float | None = None
                    if self.test_set is not None:
                        test_smiles, test_pec50 = self.test_set
                        model_metric_value = self.evaluator.evaluate_model(
                            self.model,
                            test_smiles,
                            test_pec50,
                            self.model_metric,
                            noise_scale=noise_schedule[iteration]
                            if noise_schedule is not None
                            else None,
                        )
                    progress.advance(task)

                    # --- Select next compounds ----------------------------
                    remaining_unlabeled = self.oracle.get_unlabeled_smiles()
                    remaining_ps_labeled = self.oracle.get_ps_labeled_smiles()
                    all_remaining = remaining_unlabeled + remaining_ps_labeled
                    progress.update(
                        task,
                        description=(
                            f"[green]Iter {iteration + 1}/{n_iterations}[/green]  "
                            f"Selecting next {k_per_iteration} — "
                            f"[white]{len(remaining_unlabeled)} unqueried[/white], "
                            f"[magenta]{len(remaining_ps_labeled)} PS hits[/magenta] eligible for upgrade"
                        ),
                    )
                    if all_remaining:
                        all_preds = self._predict(
                            all_remaining,
                            noise_schedule[iteration]
                            if noise_schedule is not None
                            else None,
                        )
                        unlabeled_preds = all_preds[: len(remaining_unlabeled)]
                        ps_labeled_preds = all_preds[len(remaining_unlabeled) :]
                        pending_queries = self.acquisition.select(
                            remaining_unlabeled,
                            unlabeled_preds,
                            k_per_iteration,
                            ps_labeled_smiles=remaining_ps_labeled,
                            ps_labeled_predictions=ps_labeled_preds
                            if remaining_ps_labeled
                            else None,
                        )
                    else:
                        pending_queries = []
                    progress.advance(task)

                    # ---- Metrics & dashboard update -------------------------
                    metrics = self.evaluator.evaluate(
                        labeled=self.oracle.labeled_records,
                        n_total=n_total,
                        n_true_actives=n_true_actives,
                        iteration=iteration,
                    )
                    if model_metric_value is not None:
                        metrics[f"model_{self.model_metric.value}"] = model_metric_value

                    iter_result = IterationResults(
                        iteration=iteration,
                        queries=queries,
                        new_records=new_records,
                        metrics=metrics,
                        cumulative_cost=self.oracle.total_cost,
                        cumulative_labeled=n_labeled,
                        model_metric_value=model_metric_value,
                    )
                    results.iterations.append(iter_result)
                    results.total_cost = self.oracle.total_cost
                    results.total_labeled = n_labeled

                    if self.dashboard is not None:
                        self.dashboard.update(
                            labeled_records=self.oracle.labeled_records,
                            activity_threshold=self.evaluator.activity_threshold,
                            iter_drc_cost=iter_drc_cost,
                            iter_ps_cost=iter_ps_cost,
                            iter_upgrade_cost=iter_upgrade_cost,
                            model_metric_value=model_metric_value,
                        )

                    if not all_remaining:
                        progress.update(
                            task,
                            description="[green]All compounds queried — stopping early.",
                        )
                        break

        results.final_metrics = self.evaluator.evaluate(
            labeled=self.oracle.labeled_records,
            n_total=n_total,
            n_true_actives=n_true_actives,
            iteration=len(results.iterations) - 1,
        )
        n_final_upgrades = int(results.final_metrics.get("n_ps_to_drc_upgrades", 0))
        n_final_drc = int(results.final_metrics.get("n_drc_queries", 0))
        n_final_ps = int(results.final_metrics.get("n_ps_queries", 0))
        drc_label = (
            f"[orange1]{n_final_drc} DRC[/orange1] "
            f"([magenta]{n_final_upgrades} upgrades[/magenta])"
            if n_final_upgrades > 0
            else f"[orange1]{n_final_drc} DRC[/orange1]"
        )
        _console.print(
            f"[bold green]Campaign complete.[/bold green]  "
            f"[bold]Total cost:[/bold] [bold]${results.total_cost:.2f}[/bold]  |  "
            f"[steel_blue1]{n_final_ps} PS[/steel_blue1]  |  "
            f"{drc_label}  |  "
            f"Confirmed actives: [green]{int(results.final_metrics.get('n_confirmed_actives', 0))}[/green]"
            f" [dim](of {n_true_actives})[/dim]"
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict(
        self, smiles_list: list[str], noise_scale: float | None = None
    ) -> np.ndarray:
        """Route inference to the appropriate model backend.

        Passes ``noise_scale`` to ``NoisyOracleModel`` to support per-iteration
        error scheduling; ignored for ``ChemPropLightningModule``.

        Parameters
        ----------
        smiles_list : list[str]
            Canonical SMILES strings to score.
        noise_scale : float, optional
            Noise half-width for ``NoisyOracleModel``. Must be set when the
            active model is a ``NoisyOracleModel``; ignored otherwise.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with pEC50 point estimates.
        """
        if isinstance(self.model, NoisyOracleModel):
            # noise_scale is always set from the schedule when NoisyOracleModel is active
            return self.model.predict_smiles(smiles_list, noise_scale)  # type: ignore[arg-type]
        return self.model.predict_smiles(smiles_list)

    def _empty_result(self, iteration: int) -> IterationResults:
        return IterationResults(
            iteration=iteration,
            queries=[],
            new_records=[],
            metrics={},
            cumulative_cost=self.oracle.total_cost,
            cumulative_labeled=len(self.oracle.labeled_records),
        )
