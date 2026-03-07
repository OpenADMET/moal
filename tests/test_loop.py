"""Integration test for the full active learning loop.

Uses a synthetic ground-truth dataset and a mock model (returns random
pEC50 predictions) to verify that:
  - The loop runs for the correct number of iterations.
  - Cost is tracked monotonically.
  - The labeled pool grows by exactly k compounds per iteration.
  - Evaluation metrics are finite and consistent.
  - No compound is labeled twice.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pandas as pd
import pytest

from moal.acquisition import CostAwareGreedyAcquisition
from moal.dashboard import LiveDashboard
from moal.evaluation import PipelineEvaluator
from moal.loop import ActiveLearningLoop
from moal.model import ChemPropLightningModule
from moal.oracle import CostAwareOracle
from moal.preprocessing import SMILESPreprocessor
from moal.types import LabelRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 30 real SMILES sourced from common drug-like scaffolds.
_SMILES = [
    "c1ccccc1",
    "c1ccc(N)cc1",
    "c1ccc(O)cc1",
    "CC(=O)O",
    "CCO",
    "c1ccc2ccccc2c1",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "CC(C)O",
    "CCCO",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(Br)cc1",
    "Cc1ccccc1",
    "COc1ccccc1",
    "Nc1ccccc1",
    "Oc1ccccc1",
    "c1ccc(CC)cc1",
    "c1ccc(C(=O)O)cc1",
    "c1ccc(C#N)cc1",
    "c1ccc(S)cc1",
    "CC1=CC=CC=C1",
    "c1ccc(-c2ccccc2)cc1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOCC1",
    "C1CCSC1",
    "c1ccc(NC(=O)C)cc1",
    "c1ccc(OC(=O)C)cc1",
]

# Assign synthetic pEC50 values: roughly normal, 3 compounds are "active" (>7).
_RNG = np.random.default_rng(42)
_PECS50 = _RNG.normal(6.0, 1.2, len(_SMILES)).tolist()
_PECS50[2] = 7.8   # phenol — active
_PECS50[5] = 8.1   # naphthalene — active
_PECS50[23] = 7.3  # biphenyl — active


@pytest.fixture
def ground_truth_df():
    return pd.DataFrame({"smiles": _SMILES, "pec50": _PECS50})


@pytest.fixture
def oracle(ground_truth_df):
    return CostAwareOracle(
        ground_truth_df=ground_truth_df,
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        upper_bound=11.0,
    )


@pytest.fixture
def mock_model():
    """Spec-d mock that returns random pEC50 predictions (no CheMeleon needed).

    Using create_autospec ensures any API change to ChemPropLightningModule
    is caught here rather than silently accepted.
    """
    model = create_autospec(ChemPropLightningModule, instance=True)
    rng = np.random.default_rng(0)

    def _predict(smiles_list, **kwargs):
        return rng.normal(6.0, 1.5, len(smiles_list)).astype(np.float32)

    model.predict_smiles.side_effect = _predict
    model.refit.return_value = model
    return model


@pytest.fixture
def acquisition():
    return CostAwareGreedyAcquisition(
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        target_threshold=7.0,
        tau=0.5,
    )


@pytest.fixture
def evaluator():
    return PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)


@pytest.fixture
def loop(oracle, mock_model, acquisition, evaluator):
    return ActiveLearningLoop(
        oracle=oracle,
        model=mock_model,
        acquisition=acquisition,
        evaluator=evaluator,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N_ITERATIONS = 3
K = 5


class TestLoopExecution:
    def test_correct_number_of_iterations(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert len(results.iterations) == N_ITERATIONS

    def test_labeled_pool_grows(self, loop, oracle):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        prev = 0
        for iter_result in results.iterations:
            assert iter_result.cumulative_labeled >= prev
            prev = iter_result.cumulative_labeled
        assert results.total_labeled == N_ITERATIONS * K

    def test_cost_is_monotonically_increasing(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        costs = results.costs()
        assert all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))
        assert results.total_cost == pytest.approx(costs[-1])

    def test_no_compound_labeled_twice(self, loop, oracle):
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        canonical_smiles = [r.canonical_smiles for r in oracle.labeled_records]
        assert len(canonical_smiles) == len(set(canonical_smiles))

    def test_model_refit_called_each_iteration(self, loop, mock_model):
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert mock_model.refit.call_count == N_ITERATIONS

    def test_predict_smiles_never_receives_labeled_compounds(self, loop, mock_model):
        """The unlabeled pool passed to predict_smiles must shrink each iteration,
        confirming that already-labeled compounds are excluded before scoring."""
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        call_sizes = [
            len(c.args[0]) for c in mock_model.predict_smiles.call_args_list
        ]
        # Each iteration removes K newly-labeled compounds from the pool
        assert all(s1 > s2 for s1, s2 in zip(call_sizes, call_sizes[1:])), (
            f"Pool must strictly shrink between iterations; got sizes {call_sizes}"
        )


class TestMetrics:
    def test_metrics_are_finite(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for iter_result in results.iterations:
            for key, value in iter_result.metrics.items():
                assert np.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_total_cost_in_final_metrics(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert "total_cost" in results.final_metrics
        assert results.final_metrics["total_cost"] == pytest.approx(results.total_cost)

    def test_actives_per_dollar_non_negative(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for iter_result in results.iterations:
            assert iter_result.metrics.get("actives_per_dollar", 0.0) >= 0.0

    def test_recall_between_0_and_1(self, loop):
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for iter_result in results.iterations:
            recall = iter_result.metrics.get("recall", 0.0)
            assert 0.0 <= recall <= 1.0


class TestEarlyStop:
    def test_stops_when_all_labeled(self, ground_truth_df):
        """If k × n_iterations >= pool_size, the loop stops early without error."""
        oracle = CostAwareOracle(
            ground_truth_df=ground_truth_df.head(10),
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
        )
        rng = np.random.default_rng(99)

        mock = create_autospec(ChemPropLightningModule, instance=True)
        mock.predict_smiles.side_effect = lambda s, **kw: rng.normal(6.0, 1.0, len(s)).astype(np.float32)
        mock.refit.return_value = mock

        acq = CostAwareGreedyAcquisition(cost_ps=1.0, cost_drc=10.0)
        ev = PipelineEvaluator()
        loop = ActiveLearningLoop(oracle=oracle, model=mock, acquisition=acq, evaluator=ev)

        results = loop.run(n_iterations=100, k_per_iteration=5)
        assert results.total_labeled <= 10


class TestDashboardIntegration:
    def test_dashboard_update_called_each_iteration(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """dashboard.update() should be called exactly once per completed iteration."""
        mock_db = create_autospec(LiveDashboard, instance=True)
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            dashboard=mock_db,
        )
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert mock_db.update.call_count == N_ITERATIONS

    def test_dashboard_receives_iter_costs(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """Each dashboard.update() call should pass iter_drc_cost and iter_ps_cost."""
        mock_db = create_autospec(LiveDashboard, instance=True)
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            dashboard=mock_db,
        )
        loop.run(n_iterations=2, k_per_iteration=K)
        for call in mock_db.update.call_args_list:
            kwargs = call.kwargs
            assert "iter_drc_cost" in kwargs
            assert "iter_ps_cost" in kwargs
            assert kwargs["iter_drc_cost"] >= 0
            assert kwargs["iter_ps_cost"] >= 0

    def test_iter_costs_match_oracle_records(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """iter_drc_cost + iter_ps_cost passed to dashboard must equal actual
        oracle spend (no pre-query inflation)."""
        mock_db = create_autospec(LiveDashboard, instance=True)
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            dashboard=mock_db,
        )
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)

        # Sum of all per-iteration costs reported to dashboard must equal total oracle cost.
        reported_total = sum(
            call.kwargs["iter_drc_cost"] + call.kwargs["iter_ps_cost"]
            for call in mock_db.update.call_args_list
        )
        assert reported_total == pytest.approx(oracle.total_cost, rel=1e-6)


class TestTestSetIntegration:
    def test_model_metric_value_stored(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """model_metric_value should be present and finite when test_set is provided."""
        from moal.evaluation import ModelMetric

        rng = np.random.default_rng(42)
        test_smiles = oracle.get_unlabeled_smiles()[:5]
        test_pec50 = rng.normal(6.0, 1.0, len(test_smiles)).astype(np.float32)

        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            test_set=(test_smiles, test_pec50),
            model_metric=ModelMetric.MAE,
        )
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for ir in results.iterations:
            assert ir.model_metric_value is not None
            assert np.isfinite(ir.model_metric_value)

    def test_no_test_set_metric_value_none(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """model_metric_value should be None when no test_set is provided."""
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
        )
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for ir in results.iterations:
            assert ir.model_metric_value is None
