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
from moal.planning import training_records_for_refit, validate_training_records
from moal.preprocessing import SMILESPreprocessor
from moal.types import IterationResults, LabelRecord, LoopResults, QueryType

if TYPE_CHECKING:
    from moal.dashboard import LiveDashboard

logger = logging.getLogger(__name__)

# Rich console used for status messages; writes to stderr so it doesn't mix
# with any stdout output and plays nicely with the progress bar
_console = Console(stderr=True)


def _merge_pretrain_with_oracle(
    pretrain: list[LabelRecord],
    oracle_records: list[LabelRecord],
    superseded_tracker: set[str],
) -> list[LabelRecord]:
    """Combine pretrain records with oracle-acquired records for model refit.

    Oracle records take unconditional precedence: if the oracle has any record
    for a compound (at *any* fidelity), **all** pretrain records for that
    compound are dropped and the canonical SMILES is added to
    ``superseded_tracker`` (caller emits a consolidated warning at end of
    campaign).  This prevents contradictory label combinations such as a
    pretrain DRC record surviving alongside an oracle PS query on the same
    compound.

    The combined list is passed through :func:`~moal.planning.training_records_for_refit`
    so that pretrain PS INTERVAL records are dropped when the oracle has
    acquired a DRC record for the same compound.  :func:`~moal.planning.validate_training_records`
    is then called as a safety net against any remaining inconsistencies in the
    pretrain data itself.

    Parameters
    ----------
    pretrain : list[LabelRecord]
        Records loaded from the optional pretrain CSV.  May be empty.
    oracle_records : list[LabelRecord]
        Records from ``oracle.training_records`` for the current iteration.
    superseded_tracker : set[str]
        Mutable set updated with canonical SMILES of any newly superseded
        pretrain records.  The caller uses this to emit a single end-of-loop
        warning.

    Returns
    -------
    list[LabelRecord]
        Merged and deduplicated records ready for ``model.refit()``.

    Raises
    ------
    ValueError
        If the pretrain records themselves contain inconsistencies (e.g. two
        DRC records for the same compound).
    """
    if not pretrain:
        return oracle_records

    oracle_smiles = {r.canonical_smiles for r in oracle_records}
    filtered: list[LabelRecord] = []
    for rec in pretrain:
        if rec.canonical_smiles in oracle_smiles:
            if rec.canonical_smiles not in superseded_tracker:
                superseded_tracker.add(rec.canonical_smiles)
        else:
            filtered.append(rec)

    merged = training_records_for_refit(filtered + oracle_records)
    validate_training_records(merged)
    return merged


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
    preprocessor : SMILESPreprocessor or None, optional
        Used for pre-flight SMILES checks. A default instance is created if
        None is provided.
    trainer_kwargs : dict[str, Any] or None, optional
        Forwarded to ``lightning.Trainer`` at each refit.
    datamodule_kwargs : dict[str, Any] or None, optional
        Forwarded to ``MixedFidelityDataModule`` at each refit (e.g.,
        ``val_fraction``, ``seed``).
    dashboard : LiveDashboard or None, optional
        If provided, updated after every iteration.
    test_set : tuple[list[str], numpy.ndarray] or None, optional
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
    pretrain_records : list[LabelRecord], optional
        Pre-labeled records loaded from an optional pretrain CSV (same
        mixed-fidelity format as the ``moal plan`` campaign state).  These
        are combined with oracle-acquired records before every
        ``model.refit()`` call.  Oracle records take precedence when both
        sources label the same compound at the same fidelity.  Default is
        an empty list, which reproduces the no-pretrain behaviour.
    output_dir : str or Path or None, optional
        Directory for Lightning default logs (``lightning_logs/``). Forwarded
        as ``output_dir`` to every ``model.refit()`` call. When None (default),
        Lightning writes to the current working directory.
    stop_when_all_actives_found : bool, optional
        When True, terminate the campaign as soon as every true active in the
        pool has been confirmed (``n_confirmed_actives`` reaches the pool's
        ``n_true_actives``), in addition to the existing stop on pool
        exhaustion. Default is False, which runs until ``n_iterations`` or pool
        exhaustion regardless of how many actives remain. Use with a generous
        ``n_iterations`` to let convergence, rather than the iteration count,
        end the run.

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
        dashboard: LiveDashboard | None = None,
        test_set: tuple[list[str], np.ndarray] | None = None,
        model_metric: ModelMetric = ModelMetric.MAE,
        initial_error: float = 0.7,
        final_error: float = 0.5,
        reset_weights_on_refit: bool = False,
        pretrain_records: list[LabelRecord] | None = None,
        output_dir: str | Path | None = None,
        stop_when_all_actives_found: bool = False,
    ) -> None:
        """Initialise the loop and store all campaign components."""
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
        self.pretrain_records: list[LabelRecord] = pretrain_records or []
        self.output_dir = output_dir
        self.stop_when_all_actives_found = stop_when_all_actives_found

    # ------------------------------------------------------------------
    # Seed evaluation
    # ------------------------------------------------------------------

    def _record_seed_iteration(
        self,
        results: LoopResults,
        n_true_actives: int,
        noise_scale: float | None,
        superseded_tracker: set[str],
    ) -> bool:
        """Evaluate and record the seed/initial state as iteration 0.

        Triggered automatically whenever the campaign starts from any labeled
        data: a costed warm-start already queried into the oracle and/or free
        pretrain records. The model is refit on that seed and evaluated, and the
        result is appended as iteration 0 at the seed's cumulative cost. Training
        here also makes the first acquisition model-informed rather than selected
        from an untrained model.

        Parameters
        ----------
        results : LoopResults
            Results accumulator; the seed iteration is appended when present.
        n_true_actives : int
            Active count in the pool, for the recall/confirmed-active metrics.
        noise_scale : float or None
            Noise scale for :class:`NoisyOracleModel` evaluation; ``None`` for
            real models.
        superseded_tracker : set[str]
            Shared set recording pretrain records superseded by oracle queries.

        Returns
        -------
        bool
            ``True`` when a seed state existed and was recorded as iteration 0,
            so the caller offsets subsequent acquisition indices by one.
        """
        seed_records = _merge_pretrain_with_oracle(
            self.pretrain_records, self.oracle.training_records, superseded_tracker
        )
        if not seed_records:
            return False

        self.model.refit(
            records=seed_records,
            trainer_kwargs=self.trainer_kwargs,
            datamodule_kwargs=self.datamodule_kwargs,
            reset_weights=self.reset_weights_on_refit,
            output_dir=self.output_dir,
        )

        model_metric_value: float | None = None
        if self.test_set is not None:
            test_smiles, test_pec50 = self.test_set
            model_metric_value = self.evaluator.evaluate_model(
                self.model, test_smiles, test_pec50, self.model_metric, noise_scale=noise_scale
            )

        metrics = self.evaluator.evaluate(
            labeled=self.oracle.labeled_records,
            n_true_actives=n_true_actives,
            iteration=0,
        )
        if model_metric_value is not None:
            metrics[f"model_{self.model_metric.value}"] = model_metric_value

        results.iterations.append(
            IterationResults(
                iteration=0,
                queries=[],
                new_records=[],
                metrics=metrics,
                cumulative_cost=self.oracle.total_cost,
                cumulative_labeled=len(self.oracle.labeled_records),
                model_metric_value=model_metric_value,
            )
        )
        results.total_cost = self.oracle.total_cost
        results.total_labeled = len(self.oracle.labeled_records)
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        n_iterations: int,
        plate_size: int,
        wells_per_ps: int,
        wells_per_drc: int,
    ) -> LoopResults:
        """Execute the full active learning campaign.

        Each iteration completes three sequential steps tracked in the Rich
        progress bar:

        1. **Query oracle** — issue queries from the pre-computed candidate
           list assembled at the end of the previous iteration.
        2. **Refit model** — fine-tune the model on the growing labeled pool.
        3. **Select compounds** — run model inference and acquisition scoring
           over the remaining pool to prepare the next iteration's queries.

        Parameters
        ----------
        n_iterations : int
            Total number of active learning iterations to run.
        plate_size : int
            Maximum total wells available per plate (i.e., per iteration).
            The acquisition greedily selects ranked candidates in score order,
            stopping as soon as the next candidate would push the total well
            count over this limit.
        wells_per_ps : int
            Number of wells consumed by a single PS query.
        wells_per_drc : int
            Number of wells consumed by a single DRC query.

        Returns
        -------
        LoopResults
            Aggregated per-iteration history, final evaluation metrics,
            total cost, and total number of labeled compounds at campaign end.

        Notes
        -----
        The first iteration's candidate queries are pre-computed before the
        main loop begins so that the acquisition step of iteration *i* can be
        treated as preparation for iteration *i+1*, keeping latency off the
        critical path.

        When ``model`` is a :class:`~moal.model.NoisyOracleModel`, the noise
        scale decreases linearly from ``initial_error`` to ``final_error``
        across all ``n_iterations`` via a ``numpy.linspace`` schedule.

        The loop terminates early (before reaching ``n_iterations``) if the
        unlabeled pool is exhausted.  ``LoopResults.iterations`` will then
        contain fewer than ``n_iterations`` entries.
        """
        suppress_noisy_loggers()

        results = LoopResults()
        n_total = len(self.oracle)
        n_true_actives = self.oracle.n_true_actives(self.evaluator.activity_threshold)

        # Build a per-iteration noise schedule for NoisyOracleModel fast mode;
        # linspace(a, a, n) naturally handles the constant-error case
        noise_schedule: np.ndarray | None = None
        if isinstance(self.model, NoisyOracleModel):
            noise_schedule = np.linspace(self.initial_error, self.final_error, n_iterations)

        _console.print(
            f"[bold]moal[/bold] campaign starting — "
            f"[cyan]{n_iterations}[/cyan] iterations | "
            f"plate: [cyan]{plate_size}[/cyan] wells "
            f"([cyan]{wells_per_ps}[/cyan] PS / [cyan]{wells_per_drc}[/cyan] DRC) | "
            f"[cyan]{n_total}[/cyan] compounds | "
            f"[cyan]{n_true_actives}[/cyan] true actives"
        )
        if self.pretrain_records:
            _console.print(
                f"  [dim]Pretrain pool: [bold]{len(self.pretrain_records)}[/bold] records[/dim]"
            )

        # Evaluate the initial state as iteration 0 whenever the campaign starts
        # from labeled data (a costed warm-start and/or free pretrain records).
        # Acquisition iterations are then offset to start at 1. The seed refit
        # also trains the model before the first candidate selection below.
        _superseded_tracker: set[str] = set()
        seed_recorded = self._record_seed_iteration(
            results,
            n_true_actives,
            noise_schedule[0] if noise_schedule is not None else None,
            _superseded_tracker,
        )
        iteration_offset = 1 if seed_recorded else 0

        # Pre-compute first iteration's candidate queries before entering the
        # progress bar so Step 3 of iteration i prepares for iteration i+1.
        # Both the unqueried pool and PS-labeled INTERVAL hits are scorable
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
                plate_size,
                wells_per_ps,
                wells_per_drc,
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
                    # progress display without waiting for oracle results
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
                    # when the oracle was initialised with is_canonical=True
                    new_records = self.oracle.query_batch(
                        queries, iteration, is_canonical=self.oracle.is_canonical
                    )
                    # Derive actual per-fidelity costs from records returned by the
                    # oracle — not from the pre-query candidate list, which may
                    # differ if the oracle skips invalid or already-labeled compounds
                    iter_drc_cost = sum(
                        r.cost for r in new_records if r.fidelity == QueryType.DOSE_RESPONSE
                    )
                    iter_ps_cost = sum(
                        r.cost for r in new_records if r.fidelity == QueryType.PRIMARY_SCREEN
                    )
                    # Upgrades are DRC queries for compounds already in the PS pool;
                    # ps_labeled_before was captured before this iteration's queries
                    iter_upgrade_cost = sum(
                        r.cost
                        for r in new_records
                        if r.fidelity == QueryType.DOSE_RESPONSE
                        and r.canonical_smiles in ps_labeled_before
                    )
                    iter_n_ps = sum(
                        1 for r in new_records if r.fidelity == QueryType.PRIMARY_SCREEN
                    )
                    iter_n_drc_upgrade = sum(
                        1
                        for r in new_records
                        if r.fidelity == QueryType.DOSE_RESPONSE
                        and r.canonical_smiles in ps_labeled_before
                    )
                    iter_n_drc_new = sum(
                        1
                        for r in new_records
                        if r.fidelity == QueryType.DOSE_RESPONSE
                        and r.canonical_smiles not in ps_labeled_before
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
                    # refit message is consistent with the metrics that get logged
                    n_cumulative_upgrades = int(
                        self.evaluator.fidelity_breakdown(all_labeled).get("upgrades", 0)
                    )
                    upgrade_suffix = (
                        f", [magenta]{n_cumulative_upgrades} upgrades[/magenta]"
                        if n_cumulative_upgrades > 0
                        else ""
                    )
                    refit_records = _merge_pretrain_with_oracle(
                        self.pretrain_records,
                        self.oracle.training_records,
                        _superseded_tracker,
                    )
                    n_pretrain_active = len(refit_records) - len(self.oracle.training_records)
                    pretrain_suffix = (
                        f" + [dim]{n_pretrain_active} pretrain[/dim]"
                        if n_pretrain_active > 0
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
                            f"{pretrain_suffix}"
                        ),
                    )
                    if new_records:
                        self.model.refit(
                            records=refit_records,
                            trainer_kwargs=self.trainer_kwargs,
                            datamodule_kwargs=self.datamodule_kwargs,
                            reset_weights=self.reset_weights_on_refit,
                            output_dir=self.output_dir,
                        )

                    # Evaluate model metric on held-out test set
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
                            f"Selecting (plate={plate_size}) — "
                            f"[white]{len(remaining_unlabeled)} unqueried[/white], "
                            f"[magenta]{len(remaining_ps_labeled)} PS hits[/magenta]"
                            " eligible for upgrade"
                        ),
                    )
                    if all_remaining:
                        all_preds = self._predict(
                            all_remaining,
                            noise_schedule[iteration] if noise_schedule is not None else None,
                        )
                        unlabeled_preds = all_preds[: len(remaining_unlabeled)]
                        ps_labeled_preds = all_preds[len(remaining_unlabeled) :]
                        pending_queries = self.acquisition.select(
                            remaining_unlabeled,
                            unlabeled_preds,
                            plate_size,
                            wells_per_ps,
                            wells_per_drc,
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
                        n_true_actives=n_true_actives,
                        iteration=iteration + iteration_offset,
                    )
                    if model_metric_value is not None:
                        metrics[f"model_{self.model_metric.value}"] = model_metric_value

                    iter_result = IterationResults(
                        iteration=iteration + iteration_offset,
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
                            iter_n_drc_new=iter_n_drc_new,
                            iter_n_upgrades=iter_n_drc_upgrade,
                            iter_n_ps=iter_n_ps,
                            model_metric_value=model_metric_value,
                        )

                    if not all_remaining:
                        progress.update(
                            task,
                            description="[green]All compounds queried — stopping early.",
                        )
                        break

                    # Optional convergence stop: every true active confirmed
                    if (
                        self.stop_when_all_actives_found
                        and metrics.get("n_confirmed_actives", 0) >= n_true_actives
                    ):
                        progress.update(
                            task,
                            description="[green]All actives confirmed — stopping early.",
                        )
                        break

        results.final_metrics = self.evaluator.evaluate(
            labeled=self.oracle.labeled_records,
            n_true_actives=n_true_actives,
            iteration=len(results.iterations) - 1,
        )
        n_final_upgrades = int(results.final_metrics.get("n_ps_to_drc_upgrades", 0))
        n_final_drc = int(results.final_metrics.get("n_drc_queries", 0))
        n_final_ps = int(results.final_metrics.get("n_ps_queries", 0))
        drc_label = (
            f"[orange1]{n_final_drc} DRC[/orange1] ([magenta]{n_final_upgrades} upgrades[/magenta])"
            if n_final_upgrades > 0
            else f"[orange1]{n_final_drc} DRC[/orange1]"
        )
        _console.print(
            f"[bold green]Campaign complete.[/bold green]  "
            f"[bold]Total cost:[/bold] [bold]${results.total_cost:.2f}[/bold]  |  "
            f"[steel_blue1]{n_final_ps} PS[/steel_blue1]  |  "
            f"{drc_label}  |  "
            f"Confirmed actives: [green]{int(results.final_metrics.get('n_confirmed_actives', 0))}"
            f"[/green] [dim](of {n_true_actives})[/dim]"
        )
        if _superseded_tracker:
            logger.warning(
                "%d pretrain record(s) were superseded by oracle queries during the "
                "campaign (oracle records used in their place): %s",
                len(_superseded_tracker),
                ", ".join(sorted(_superseded_tracker)),
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict(self, smiles_list: list[str], noise_scale: float | None = None) -> np.ndarray:
        """Route inference to the appropriate model backend.

        Dispatches to :meth:`~moal.model.NoisyOracleModel.predict_smiles` or
        :meth:`~moal.model.ChemPropLightningModule.predict_smiles` depending on
        the runtime type of ``self.model``, forwarding ``noise_scale`` only to
        the noisy surrogate.

        Parameters
        ----------
        smiles_list : list[str]
            Canonical SMILES strings to score.
        noise_scale : float, optional
            Uniform noise half-width (pEC50 log-units) passed to
            :class:`~moal.model.NoisyOracleModel`.  Must be provided when
            ``self.model`` is a :class:`~moal.model.NoisyOracleModel`; the
            value is ignored for :class:`~moal.model.ChemPropLightningModule`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N,)`` containing pEC50 point estimates, where
            ``N = len(smiles_list)``.

        Notes
        -----
        When the active model is :class:`~moal.model.NoisyOracleModel`,
        ``noise_scale`` is always populated from the per-iteration schedule
        constructed in :meth:`run`.  Passing ``None`` in that case will
        propagate to the surrogate model and may cause unexpected behaviour.
        """
        if isinstance(self.model, NoisyOracleModel):
            # noise_scale is always set from the schedule when NoisyOracleModel is active
            return self.model.predict_smiles(smiles_list, noise_scale)  # type: ignore[arg-type]
        return self.model.predict_smiles(smiles_list)

    def _empty_result(self, iteration: int) -> IterationResults:
        """Build a placeholder :class:`~moal.types.IterationResults` for a skipped iteration.

        Parameters
        ----------
        iteration : int
            Zero-based index of the iteration that produced no queries or
            new records.

        Returns
        -------
        IterationResults
            An :class:`~moal.types.IterationResults` instance with empty
            ``queries``, ``new_records``, and ``metrics`` collections.
            ``cumulative_cost`` and ``cumulative_labeled`` reflect the oracle
            state at the time of the call.

        Notes
        -----
        Intended as a safe sentinel for iterations where the unlabeled pool
        is already exhausted or no valid queries could be constructed.  The
        returned object keeps ``LoopResults.iterations`` length-consistent
        with the requested ``n_iterations``.
        """
        return IterationResults(
            iteration=iteration,
            queries=[],
            new_records=[],
            metrics={},
            cumulative_cost=self.oracle.total_cost,
            cumulative_labeled=len(self.oracle.labeled_records),
        )
