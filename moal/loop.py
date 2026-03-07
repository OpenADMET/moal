"""ActiveLearningLoop: orchestrates the full m-iteration acquisition campaign."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from moal.acquisition import CostAwareGreedyAcquisition
from moal.evaluation import ModelMetric, PipelineEvaluator
from moal.logging_config import suppress_noisy_loggers
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

    Args:
        oracle: CostAwareOracle wrapping the ground-truth dataset.
        model: ChemPropLightningModule to be iteratively fine-tuned.
        acquisition: CostAwareGreedyAcquisition for query selection.
        evaluator: PipelineEvaluator for metric computation.
        preprocessor: SMILESPreprocessor (used for pre-flight checks).
        trainer_kwargs: Forwarded to ``lightning.Trainer`` at each refit.
        dashboard: Optional LiveDashboard; updated after every iteration.
        test_set: Optional ``(smiles_list, pec50_array)`` held-out scaffold
            split for model performance tracking.  If provided, the model
            metric is computed after every refit and shown in the dashboard.
        model_metric: Which metric to track for the model performance curve.
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, n_iterations: int, k_per_iteration: int) -> LoopResults:
        """Execute the full active learning campaign.

        Args:
            n_iterations: Total number of AL iterations (m).
            k_per_iteration: Number of oracle queries per iteration (k).

        Returns:
            LoopResults containing per-iteration history and final metrics.
        """
        suppress_noisy_loggers()

        results = LoopResults()
        n_total = len(self.oracle)
        n_true_actives = self.oracle.n_true_actives(self.evaluator.activity_threshold)

        _console.print(
            f"[bold]moal[/bold] campaign starting — "
            f"[cyan]{n_iterations}[/cyan] iterations × "
            f"[cyan]{k_per_iteration}[/cyan] queries | "
            f"[cyan]{n_total}[/cyan] compounds | "
            f"[cyan]{n_true_actives}[/cyan] true actives"
        )

        # Pre-compute first iteration's candidate queries before entering the
        # progress bar so Step 3 of iteration i prepares for iteration i+1.
        unlabeled = self.oracle.get_unlabeled_smiles()
        predictions = self.model.predict_smiles(unlabeled) if unlabeled else np.array([])
        pending_queries = (
            self.acquisition.select(unlabeled, predictions, k_per_iteration)
            if unlabeled else []
        )

        total_steps = n_iterations * 3
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task("Starting…", total=total_steps)

            for iteration in range(n_iterations):
                queries = pending_queries  # prepared at end of previous iter

                # ---- Step 1: Query oracle --------------------------------
                n_drc = sum(1 for _, qt in queries if qt == QueryType.DOSE_RESPONSE)
                n_ps  = sum(1 for _, qt in queries if qt == QueryType.PRIMARY_SCREEN)
                progress.update(
                    task,
                    description=(
                        f"[cyan]Iter {iteration + 1}/{n_iterations}[/cyan]  "
                        f"Querying oracle — [orange1]{n_drc} DRCs[/orange1], "
                        f"[steel_blue1]{n_ps} primary screens[/steel_blue1]"
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
                    r.cost for r in new_records if r.fidelity == QueryType.DOSE_RESPONSE
                )
                iter_ps_cost = sum(
                    r.cost for r in new_records if r.fidelity == QueryType.PRIMARY_SCREEN
                )
                progress.advance(task)

                # ---- Step 2: Refit model ---------------------------------
                n_labeled = len(self.oracle.labeled_records)
                n_labeled_drc = sum(
                    1 for r in self.oracle.labeled_records
                    if r.fidelity == QueryType.DOSE_RESPONSE
                )
                n_labeled_ps = n_labeled - n_labeled_drc
                progress.update(
                    task,
                    description=(
                        f"[yellow]Iter {iteration + 1}/{n_iterations}[/yellow]  "
                        f"Retraining model — {n_labeled} labeled "
                        f"([orange1]{n_labeled_drc} DRC[/orange1], "
                        f"[steel_blue1]{n_labeled_ps} PS[/steel_blue1])"
                    ),
                )
                if new_records:
                    self.model.refit(
                        records=self.oracle.labeled_records,
                        trainer_kwargs=self.trainer_kwargs,
                        datamodule_kwargs=self.datamodule_kwargs,
                    )

                # Evaluate model metric on held-out test set.
                model_metric_value: float | None = None
                if self.test_set is not None:
                    test_smiles, test_pec50 = self.test_set
                    model_metric_value = self.evaluator.evaluate_model(
                        self.model, test_smiles, test_pec50, self.model_metric
                    )
                progress.advance(task)

                # ---- Step 3: Select next compounds ----------------------
                progress.update(
                    task,
                    description=(
                        f"[green]Iter {iteration + 1}/{n_iterations}[/green]  "
                        f"Selecting next {k_per_iteration} compounds…"
                    ),
                )
                remaining_unlabeled = self.oracle.get_unlabeled_smiles()
                if remaining_unlabeled:
                    next_preds = self.model.predict_smiles(remaining_unlabeled)
                    pending_queries = self.acquisition.select(
                        remaining_unlabeled, next_preds, k_per_iteration
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
                        model_metric_value=model_metric_value,
                    )

                if not remaining_unlabeled:
                    progress.update(task, description="[green]All compounds labeled — stopping early.")
                    break

        results.final_metrics = self.evaluator.evaluate(
            labeled=self.oracle.labeled_records,
            n_total=n_total,
            n_true_actives=n_true_actives,
            iteration=len(results.iterations) - 1,
        )
        _console.print(
            f"[bold green]Campaign complete.[/bold green]  "
            f"Total cost: [cyan]${results.total_cost:.2f}[/cyan]  |  "
            f"Labeled: [cyan]{results.total_labeled}[/cyan]  |  "
            f"Confirmed actives: [cyan]{int(results.final_metrics.get('n_confirmed_actives', 0))}[/cyan]"
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, iteration: int) -> IterationResults:
        return IterationResults(
            iteration=iteration,
            queries=[],
            new_records=[],
            metrics={},
            cumulative_cost=self.oracle.total_cost,
            cumulative_labeled=len(self.oracle.labeled_records),
        )
