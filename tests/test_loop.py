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
from moal.types import LabelRecord, QueryType


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

    def test_no_compound_labeled_twice_same_fidelity(self, loop, oracle):
        """Each (smiles, fidelity) pair may appear at most once in labeled_records.

        A compound may have two records if it was upgraded from PS to DRC, but
        must never have two records with the same fidelity.
        """
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        pairs = [(r.canonical_smiles, r.fidelity) for r in oracle.labeled_records]
        assert len(pairs) == len(set(pairs))

    def test_model_refit_called_each_iteration(self, loop, mock_model):
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert mock_model.refit.call_count == N_ITERATIONS

    def test_predict_smiles_pool_never_grows(self, loop, mock_model):
        """The combined unlabeled + ps-labeled pool sent to predict_smiles must
        be non-increasing across iterations.

        The pool can stay flat in an iteration where every query is a new PS
        (compound moves from unlabeled to ps-labeled, pool size unchanged).
        It can only strictly shrink when a DRC query is made (compound leaves
        both pools entirely).  It must never grow.
        """
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        call_sizes = [
            len(c.args[0]) for c in mock_model.predict_smiles.call_args_list
        ]
        assert all(s1 >= s2 for s1, s2 in zip(call_sizes, call_sizes[1:])), (
            f"Scorable pool grew between iterations; sizes: {call_sizes}"
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
        # At most 10 unique compounds could be queried; each may contribute 2 records
        # (PS + DRC upgrade), so total_labeled can exceed pool_size.  Check unique compounds.
        unique_labeled = len(oracle._labeled)
        assert unique_labeled <= 10
        # Loop must have stopped early (well before 100 × 5 iterations)
        assert results.total_cost < 10 * 11  # upper bound: 10 DRCs


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


class TestIsCanonicalForwarding:
    """Regression test for the is_canonical key-forwarding bug.

    When is_canonical=True the oracle's _ground_truth keys are the raw CSV
    SMILES (e.g. Kekulé notation).  Before the fix, loop.py called
    query_batch() without forwarding is_canonical=True, causing the oracle to
    re-canonicalize those keys on lookup.  This produced a mismatch (Kekulé
    key → canonical lookup key) and emitted "Compound not found" warnings
    while silently skipping every query.
    """

    def test_queries_succeed_with_kekule_smiles_and_is_canonical_true(self):
        """Every compound selected by the acquisition must be successfully queried
        when the oracle holds Kekulé SMILES keys and is_canonical=True."""
        from moal.model import NoisyOracleModel

        # Kekulé SMILES that RDKit would canonicalize to lowercase aromatic
        # equivalents — these are the raw CSV keys when is_canonical=True.
        kekule_smiles = [
            "C1=CC=CC=C1",       # benzene (canonical: c1ccccc1)
            "C1=CC=C(O)C=C1",    # phenol (canonical: Oc1ccccc1)
            "C1=CC=C(N)C=C1",    # aniline (canonical: Nc1ccccc1)
            "C1=CC=NC=C1",       # pyridine (canonical: c1ccncc1)
            "C1=CC=CO1",         # furan (canonical: c1ccoc1)
            "CC(=O)O",           # acetic acid (already canonical)
        ]
        pec50s = [5.0, 7.8, 6.2, 5.5, 4.9, 6.0]
        df = pd.DataFrame({"smiles": kekule_smiles, "pec50": pec50s})

        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            upper_bound=11.0,
            is_canonical=True,   # keys stored verbatim, no RDKit canonicalization
        )

        # NoisyOracleModel gets the same dict, so lookups in predict_smiles match.
        model = NoisyOracleModel(oracle._ground_truth, noise_scale=0.0, seed=0)
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0,
            target_threshold=7.0, tau=0.5,
        )
        evaluator = PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)
        loop = ActiveLearningLoop(
            oracle=oracle, model=model, acquisition=acquisition, evaluator=evaluator
        )

        results = loop.run(n_iterations=2, k_per_iteration=2)

        # The oracle must have labeled compounds — if the bug were present,
        # every query would have been silently skipped and the pool would be empty.
        assert oracle.labeled_records, "No compounds were labeled — query_batch likely skipped all"
        # Each (SMILES, fidelity) pair must be unique — a compound may have both
        # PS and DRC records after an upgrade, but never two of the same fidelity.
        key_fidelity_pairs = [(r.canonical_smiles, r.fidelity) for r in oracle.labeled_records]
        assert len(key_fidelity_pairs) == len(set(key_fidelity_pairs))
        # Total cost must be positive (at least one successful query).
        assert results.total_cost > 0


class TestPSUpgradeInLoop:
    """Integration tests: PS → DRC upgrade path fires correctly inside the loop."""

    @pytest.fixture
    def upgrade_oracle(self):
        """Small pool where several compounds will get INTERVAL PS labels."""
        # pEC50 values chosen so: phenol (7.8) and naphthalene (8.1) are
        # above ps_threshold=5.0 → INTERVAL censoring → eligible for DRC upgrade.
        data = pd.DataFrame({
            "smiles": ["c1ccccc1", "c1ccc(O)cc1", "c1ccc2ccccc2c1",
                       "c1ccc(N)cc1", "CCO", "CC(=O)O"],
            "pec50": [4.0, 7.8, 8.1, 5.5, 6.0, 3.0],
        })
        return CostAwareOracle(
            ground_truth_df=data,
            cost_ps=1.0, cost_drc=10.0,
            ps_threshold=5.0, upper_bound=11.0,
        )

    @pytest.fixture
    def upgrade_loop(self, upgrade_oracle):
        from moal.model import NoisyOracleModel
        model = NoisyOracleModel(upgrade_oracle._ground_truth, noise_scale=0.0, seed=0)
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0,
            target_threshold=7.0, tau=0.5,
        )
        evaluator = PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)
        return ActiveLearningLoop(
            oracle=upgrade_oracle, model=model,
            acquisition=acquisition, evaluator=evaluator,
        )

    def test_upgrade_produces_both_records(self, upgrade_loop, upgrade_oracle):
        """Running enough iterations must produce at least one compound with
        both a PS and a DRC record (confirming the upgrade path fires)."""
        upgrade_loop.run(n_iterations=6, k_per_iteration=2)
        records = upgrade_oracle.labeled_records
        # Group by canonical SMILES
        from collections import defaultdict
        by_smiles: dict = defaultdict(list)
        for r in records:
            by_smiles[r.canonical_smiles].append(r.fidelity)
        upgraded = {
            smi for smi, fids in by_smiles.items()
            if QueryType.PRIMARY_SCREEN in fids and QueryType.DOSE_RESPONSE in fids
        }
        assert upgraded, "Expected at least one compound to have both PS and DRC records"

    def test_no_duplicate_fidelity_pairs(self, upgrade_loop, upgrade_oracle):
        """Each (smiles, fidelity) pair must appear at most once in labeled_records."""
        upgrade_loop.run(n_iterations=4, k_per_iteration=2)
        pairs = [(r.canonical_smiles, r.fidelity) for r in upgrade_oracle.labeled_records]
        assert len(pairs) == len(set(pairs))

    def test_cost_includes_both_ps_and_drc(self, upgrade_loop, upgrade_oracle):
        """Total cost must reflect both PS and DRC assays when upgrades occur."""
        upgrade_loop.run(n_iterations=6, k_per_iteration=2)
        manual_cost = sum(r.cost for r in upgrade_oracle.labeled_records)
        assert upgrade_oracle.total_cost == pytest.approx(manual_cost)

    def test_ps_labeled_pool_shrinks_as_upgrades_happen(self, upgrade_loop, upgrade_oracle):
        """After enough iterations, the PS-labeled pool should shrink to zero
        as all INTERVAL-censored hits are upgraded to DRC."""
        # Run enough iterations to exhaust the whole pool
        upgrade_loop.run(n_iterations=10, k_per_iteration=2)
        # All compounds labeled; none remain eligible for PS→DRC upgrade
        assert upgrade_oracle.get_ps_labeled_smiles() == []
