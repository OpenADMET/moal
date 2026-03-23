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

from collections import defaultdict
from unittest.mock import create_autospec, patch

import numpy as np
import pandas as pd
import pytest

from moal.acquisition import CostAwareGreedyAcquisition
from moal.dashboard import LiveDashboard
from moal.evaluation import ModelMetric, PipelineEvaluator
from moal.loop import ActiveLearningLoop
from moal.model import ChemPropLightningModule, NoisyOracleModel
from moal.oracle import CostAwareOracle
from moal.types import QueryType

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
_PECS50[2] = 7.8  # phenol — active
_PECS50[5] = 8.1  # naphthalene — active
_PECS50[23] = 7.3  # biphenyl — active


@pytest.fixture
def ground_truth_df():
    """Ground-truth DataFrame with 30 synthetic compounds, 3 of which are true actives."""
    return pd.DataFrame({"smiles": _SMILES, "pec50": _PECS50})


@pytest.fixture
def oracle(ground_truth_df):
    """CostAwareOracle built from the 30-compound ground-truth DataFrame."""
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
    """Standard acquisition object with cost_ps=1, cost_drc=10, and target_threshold=7.0."""
    return CostAwareGreedyAcquisition(
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        target_threshold=7.0,
        tau=0.5,
    )


@pytest.fixture
def evaluator():
    """PipelineEvaluator with activity_threshold=7.0 for loop-level recall and enrichment metrics."""
    return PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)


@pytest.fixture
def loop(oracle, mock_model, acquisition, evaluator):
    """Fully assembled ActiveLearningLoop using the mock model and shared oracle/acquisition/evaluator."""
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
    """Integration tests verifying that the loop runs correctly and manages the labeled pool and cost."""

    def test_correct_number_of_iterations(self, loop):
        """The results list must contain exactly n_iterations entries, confirming the loop ran the requested number of times."""
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert len(results.iterations) == N_ITERATIONS

    def test_labeled_pool_grows(self, loop, oracle):
        """The cumulative labeled count must be non-decreasing and total k × n_iterations compounds at the end."""
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        prev = 0
        for iter_result in results.iterations:
            assert iter_result.cumulative_labeled >= prev
            prev = iter_result.cumulative_labeled
        assert results.total_labeled == N_ITERATIONS * K

    def test_cost_is_monotonically_increasing(self, loop):
        """Cumulative cost must be non-decreasing, since assays can only add cost, not remove it."""
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
        """model.refit must be called exactly once per iteration to ensure the model is updated with new labels."""
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert mock_model.refit.call_count == N_ITERATIONS

    def test_reset_weights_flag_forwarded_to_refit(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """The configured refit reset policy must be forwarded to model.refit()."""
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            reset_weights_on_refit=True,
        )

        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)

        assert mock_model.refit.call_count == N_ITERATIONS
        assert all(call.kwargs["reset_weights"] is True for call in mock_model.refit.call_args_list)

    def test_predict_smiles_pool_never_grows(self, loop, mock_model):
        """The combined unlabeled + ps-labeled pool sent to predict_smiles must
        be non-increasing across iterations.

        The pool can stay flat in an iteration where every query is a new PS
        (compound moves from unlabeled to ps-labeled, pool size unchanged).
        It can only strictly shrink when a DRC query is made (compound leaves
        both pools entirely).  It must never grow.
        """
        loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        call_sizes = [len(c.args[0]) for c in mock_model.predict_smiles.call_args_list]
        assert all(s1 >= s2 for s1, s2 in zip(call_sizes, call_sizes[1:], strict=False)), (
            f"Scorable pool grew between iterations; sizes: {call_sizes}"
        )


class TestMetrics:
    """Tests that iteration-level metrics are finite, consistent, and within expected bounds."""

    def test_metrics_are_finite(self, loop):
        """All numeric metrics must be finite after every iteration; nan or inf would indicate a data pipeline bug."""
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for iter_result in results.iterations:
            for key, value in iter_result.metrics.items():
                assert np.isfinite(value), f"Metric {key} is not finite: {value}"

    def test_total_cost_in_final_metrics(self, loop):
        """final_metrics must contain total_cost matching oracle.total_cost so downstream reporting is consistent."""
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        assert "total_cost" in results.final_metrics
        assert results.final_metrics["total_cost"] == pytest.approx(results.total_cost)

    @pytest.mark.parametrize(
        "key,lo,hi",
        [
            ("actives_per_dollar", 0.0, None),
            ("recall", 0.0, 1.0),
        ],
    )
    def test_metric_in_bounds(self, loop, key, lo, hi):
        """Recall must lie in [0, 1] and actives_per_dollar must be non-negative across all iterations."""
        results = loop.run(n_iterations=N_ITERATIONS, k_per_iteration=K)
        for iter_result in results.iterations:
            assert key in iter_result.metrics, (
                f"Expected metric '{key}' in iter_result.metrics; "
                f"got keys: {list(iter_result.metrics.keys())}"
            )
            value = iter_result.metrics[key]
            assert value >= lo
            if hi is not None:
                assert value <= hi


class TestEarlyStop:
    """Tests that the loop terminates gracefully when the compound pool is exhausted."""

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
        mock.predict_smiles.side_effect = lambda s, **kw: rng.normal(6.0, 1.0, len(s)).astype(
            np.float32
        )
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
    """Tests that the dashboard receives the correct calls and cost breakdowns from the loop."""

    def test_dashboard_update_called_and_costs_correct(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """dashboard.update() should be called exactly once per completed iteration,
        with iter_drc_cost and iter_ps_cost that sum to the actual oracle spend.
        """
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

        for call in mock_db.update.call_args_list:
            kwargs = call.kwargs
            assert "iter_drc_cost" in kwargs
            assert "iter_ps_cost" in kwargs
            assert kwargs["iter_drc_cost"] >= 0
            assert kwargs["iter_ps_cost"] >= 0

        reported_total = sum(
            call.kwargs["iter_drc_cost"] + call.kwargs["iter_ps_cost"]
            for call in mock_db.update.call_args_list
        )
        assert reported_total == pytest.approx(oracle.total_cost, rel=1e-6)


class TestTestSetIntegration:
    """Tests that the per-iteration model evaluation against a held-out test set works correctly."""

    def test_model_metric_value_stored(self, oracle, mock_model, acquisition, evaluator):
        """model_metric_value should be present and finite when test_set is provided."""
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

    def test_no_test_set_metric_value_none(self, oracle, mock_model, acquisition, evaluator):
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
        when the oracle holds Kekulé SMILES keys and is_canonical=True.
        """
        # Kekulé SMILES that RDKit would canonicalize to lowercase aromatic
        # equivalents — these are the raw CSV keys when is_canonical=True.
        kekule_smiles = [
            "C1=CC=CC=C1",  # benzene (canonical: c1ccccc1)
            "C1=CC=C(O)C=C1",  # phenol (canonical: Oc1ccccc1)
            "C1=CC=C(N)C=C1",  # aniline (canonical: Nc1ccccc1)
            "C1=CC=NC=C1",  # pyridine (canonical: c1ccncc1)
            "C1=CC=CO1",  # furan (canonical: c1ccoc1)
            "CC(=O)O",  # acetic acid (already canonical)
        ]
        pec50s = [5.0, 7.8, 6.2, 5.5, 4.9, 6.0]
        df = pd.DataFrame({"smiles": kekule_smiles, "pec50": pec50s})

        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            upper_bound=11.0,
            is_canonical=True,  # keys stored verbatim, no RDKit canonicalization
        )

        # NoisyOracleModel gets the same dict, so lookups in predict_smiles match.
        model = NoisyOracleModel(oracle._ground_truth, seed=0)
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )
        evaluator = PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)
        loop = ActiveLearningLoop(
            oracle=oracle,
            model=model,
            acquisition=acquisition,
            evaluator=evaluator,
            initial_error=0.0,
            final_error=0.0,
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
        data = pd.DataFrame(
            {
                "smiles": [
                    "c1ccccc1",
                    "c1ccc(O)cc1",
                    "c1ccc2ccccc2c1",
                    "c1ccc(N)cc1",
                    "CCO",
                    "CC(=O)O",
                ],
                "pec50": [4.0, 7.8, 8.1, 5.5, 6.0, 3.0],
            }
        )
        return CostAwareOracle(
            ground_truth_df=data,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            upper_bound=11.0,
        )

    @pytest.fixture
    def upgrade_loop(self, upgrade_oracle):
        """ActiveLearningLoop wired to upgrade_oracle using NoisyOracleModel with zero noise for deterministic upgrades."""
        model = NoisyOracleModel(upgrade_oracle._ground_truth, seed=0)
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )
        evaluator = PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)
        return ActiveLearningLoop(
            oracle=upgrade_oracle,
            model=model,
            acquisition=acquisition,
            evaluator=evaluator,
            initial_error=0.0,
            final_error=0.0,
        )

    def test_upgrade_produces_both_records(self, upgrade_loop, upgrade_oracle):
        """Running enough iterations must produce at least one compound with
        both a PS and a DRC record (confirming the upgrade path fires).
        """
        upgrade_loop.run(n_iterations=6, k_per_iteration=2)
        records = upgrade_oracle.labeled_records
        # Group by canonical SMILES
        by_smiles: dict = defaultdict(list)
        for r in records:
            by_smiles[r.canonical_smiles].append(r.fidelity)
        upgraded = {
            smi
            for smi, fids in by_smiles.items()
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
        as all INTERVAL-censored hits are upgraded to DRC.
        """
        # Run enough iterations to exhaust the whole pool
        upgrade_loop.run(n_iterations=10, k_per_iteration=2)
        # All compounds labeled; none remain eligible for PS→DRC upgrade
        assert upgrade_oracle.get_ps_labeled_smiles() == []


# ---------------------------------------------------------------------------
# Error ramp dispatch tests
# ---------------------------------------------------------------------------

# Small pool used across ramp tests — enough compounds for several iterations
# without exhausting the pool on the first pass.
_RAMP_SMILES = [
    "c1ccccc1",
    "CCO",
    "c1ccc(O)cc1",
    "c1ccncc1",
    "c1ccoc1",
    "CC(C)O",
    "CCCO",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "Cc1ccccc1",
]
_RAMP_PECS50 = [4.0, 6.0, 7.5, 5.5, 4.9, 5.8, 6.2, 4.3, 5.1, 6.7]


@pytest.fixture
def ramp_oracle():
    """Small 10-compound oracle for noise-ramp dispatch tests; large enough to avoid early exhaustion."""
    df = pd.DataFrame({"smiles": _RAMP_SMILES, "pec50": _RAMP_PECS50})
    return CostAwareOracle(
        ground_truth_df=df,
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        upper_bound=11.0,
    )


class TestNoisyOracleErrorRamp:
    """Verify that the per-iteration noise ramp is correctly computed and dispatched."""

    N_ITER = 4
    K = 2

    def _make_loop(self, oracle, initial_error, final_error):
        model = NoisyOracleModel(oracle._ground_truth, seed=0)
        acq = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )
        ev = PipelineEvaluator(activity_threshold=7.0, upper_bound=11.0)
        return model, ActiveLearningLoop(
            oracle=oracle,
            model=model,
            acquisition=acq,
            evaluator=ev,
            initial_error=initial_error,
            final_error=final_error,
        )

    def test_ramp_dispatches_correct_noise_per_iteration(self, ramp_oracle):
        """Step 3 of each iteration must call predict_smiles with the scheduled noise_scale.

        The schedule is np.linspace(initial_error, final_error, n_iterations), so
        iteration i should receive schedule[i]. Spy on predict_smiles to capture
        all noise_scale arguments while allowing real predictions through.
        """
        initial, final = 0.8, 0.2
        expected_schedule = np.linspace(initial, final, self.N_ITER)

        model, loop = self._make_loop(ramp_oracle, initial, final)
        captured: list[float] = []
        real_predict = model.predict_smiles

        def _spy(smiles_list, noise_scale, batch_size=256):
            captured.append(noise_scale)
            return real_predict(smiles_list, noise_scale, batch_size)

        with patch.object(model, "predict_smiles", side_effect=_spy):
            loop.run(n_iterations=self.N_ITER, k_per_iteration=self.K)

        # The first call is the pre-loop seed; calls 1..N_ITER are the per-iteration
        # Step 3 selections. Slice off the seed call and check the iteration calls.
        assert len(captured) >= self.N_ITER + 1, (
            f"Expected at least {self.N_ITER + 1} predict_smiles calls, got {len(captured)}"
        )
        iter_noise_scales = captured[1 : self.N_ITER + 1]
        for i, (actual, expected) in enumerate(
            zip(iter_noise_scales, expected_schedule, strict=False)
        ):
            assert actual == pytest.approx(expected, abs=1e-7), (
                f"Iteration {i}: expected noise_scale={expected:.6f}, got {actual:.6f}"
            )

    def test_constant_ramp_when_initial_equals_final(self, ramp_oracle):
        """When initial_error == final_error, every predict_smiles call must use that value."""
        noise_val = 0.5
        model, loop = self._make_loop(ramp_oracle, noise_val, noise_val)
        captured: list[float] = []
        real_predict = model.predict_smiles

        def _spy(smiles_list, noise_scale, batch_size=256):
            captured.append(noise_scale)
            return real_predict(smiles_list, noise_scale, batch_size)

        with patch.object(model, "predict_smiles", side_effect=_spy):
            loop.run(n_iterations=self.N_ITER, k_per_iteration=self.K)

        assert len(captured) >= self.N_ITER + 1
        for i, ns in enumerate(captured):
            assert ns == pytest.approx(noise_val, abs=1e-7), (
                f"Call {i}: expected constant noise_scale={noise_val}, got {ns}"
            )

    def test_pre_loop_call_uses_initial_error(self, ramp_oracle):
        """The pre-loop seed call (before iteration 0) must use initial_error, not final_error."""
        initial, final = 0.9, 0.1
        model, loop = self._make_loop(ramp_oracle, initial, final)
        captured: list[float] = []
        real_predict = model.predict_smiles

        def _spy(smiles_list, noise_scale, batch_size=256):
            captured.append(noise_scale)
            return real_predict(smiles_list, noise_scale, batch_size)

        with patch.object(model, "predict_smiles", side_effect=_spy):
            loop.run(n_iterations=self.N_ITER, k_per_iteration=self.K)

        assert captured, "predict_smiles was never called"
        assert captured[0] == pytest.approx(initial, abs=1e-7), (
            f"Pre-loop call used noise_scale={captured[0]:.6f}, expected initial_error={initial}"
        )


class TestPretrainRecords:
    """Tests for ActiveLearningLoop behaviour when pretrain_records are provided."""

    N_ITER = 2
    K = 3

    @pytest.fixture
    def pretrain_loop(self, oracle, mock_model, acquisition, evaluator):
        """Loop with two pretrain records (one PS-INTERVAL, one DRC-EXACT).

        The PS-INTERVAL record is for an oracle compound (``unlabeled[0]``).
        PS-INTERVAL is used instead of PS-LEFT so that, if the oracle later
        acquires a DRC record for the same compound,
        ``training_records_for_refit`` silently drops the superseded PS
        record rather than ``validate_training_records`` raising ValueError.

        The DRC-EXACT record uses a SMILES (``"CCCC"``) that is **not** in
        the oracle's ground-truth pool.  This guarantees the oracle can never
        acquire a PS-LEFT record for it, avoiding the contradictory-fidelity
        conflict that ``validate_training_records`` is designed to catch.
        """
        from moal.types import CensoringType, LabelRecord, QueryType

        unlabeled = oracle.get_unlabeled_smiles()
        pretrain = [
            LabelRecord(
                smiles=unlabeled[0],
                canonical_smiles=unlabeled[0],
                value=5.0,
                upper_bound=11.0,
                censoring_type=CensoringType.INTERVAL,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=1.0,
                iteration=0,
            ),
            LabelRecord(
                smiles="CCCC",
                canonical_smiles="CCCC",
                value=7.5,
                upper_bound=7.5,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=0,
            ),
        ]
        return ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            pretrain_records=pretrain,
        )

    def test_pretrain_records_included_in_refit_call(self, pretrain_loop, mock_model):
        """model.refit must be called with the exact pretrain SMILES in the records list."""
        pretrain_loop.run(n_iterations=self.N_ITER, k_per_iteration=self.K)
        assert mock_model.refit.called
        # Every pretrain SMILES must appear among the records passed to the first refit
        first_call_records = mock_model.refit.call_args_list[0][1]["records"]
        pretrain_smiles = {r.canonical_smiles for r in pretrain_loop.pretrain_records}
        refit_smiles = {r.canonical_smiles for r in first_call_records}
        missing = pretrain_smiles - refit_smiles
        assert not missing, f"Pretrain SMILES not forwarded to refit: {missing}"

    def test_empty_pretrain_reproduces_no_pretrain_behaviour(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """With pretrain_records=[], loop behaviour must be identical to not passing the arg."""
        loop_no_pretrain = ActiveLearningLoop(
            oracle=oracle, model=mock_model, acquisition=acquisition, evaluator=evaluator
        )
        loop_empty = ActiveLearningLoop(
            oracle=oracle,
            model=mock_model,
            acquisition=acquisition,
            evaluator=evaluator,
            pretrain_records=[],
        )
        # Both should accept pretrain_records without error and produce the same refit count
        from unittest.mock import create_autospec

        m1 = create_autospec(ChemPropLightningModule, instance=True)
        m2 = create_autospec(ChemPropLightningModule, instance=True)
        rng = np.random.default_rng(0)
        m1.predict_smiles.side_effect = lambda s, **k: rng.normal(6.0, 1.5, len(s)).astype(
            np.float32
        )
        m1.refit.return_value = m1
        m2.predict_smiles.side_effect = lambda s, **k: rng.normal(6.0, 1.5, len(s)).astype(
            np.float32
        )
        m2.refit.return_value = m2

        loop_no_pretrain.model = m1
        loop_empty.model = m2

        loop_no_pretrain.run(n_iterations=self.N_ITER, k_per_iteration=self.K)
        loop_empty.run(n_iterations=self.N_ITER, k_per_iteration=self.K)

        assert m1.refit.call_count == m2.refit.call_count

    def test_oracle_supersedes_pretrain_same_fidelity(
        self, oracle, mock_model, acquisition, evaluator
    ):
        """When oracle acquires a compound at the same fidelity as a pretrain record, only oracle record is kept."""
        from moal.loop import _merge_pretrain_with_oracle
        from moal.types import CensoringType, LabelRecord, QueryType

        # Force the oracle to produce a DRC record for the first unlabeled compound
        unlabeled = oracle.get_unlabeled_smiles()
        target = unlabeled[0]

        # Pretrain DRC record for the same compound
        pretrain_drc = [
            LabelRecord(
                smiles=target,
                canonical_smiles=target,
                value=6.0,  # different value from oracle ground truth
                upper_bound=6.0,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=0,
            )
        ]
        # Simulate oracle having also acquired a DRC record for target
        oracle_drc = [
            LabelRecord(
                smiles=target,
                canonical_smiles=target,
                value=7.2,
                upper_bound=7.2,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=1,
            )
        ]
        tracker: set[str] = set()
        merged = _merge_pretrain_with_oracle(pretrain_drc, oracle_drc, tracker)

        # Oracle record survives, pretrain record is dropped
        drc_records = [r for r in merged if r.fidelity == QueryType.DOSE_RESPONSE]
        assert len(drc_records) == 1
        assert drc_records[0].value == pytest.approx(7.2)
        assert target in tracker

    def test_pretrain_ps_left_oracle_drc_raises_on_merge(self):
        """Merging a pretrain PS-LEFT record with an oracle DRC record for the same compound
        must raise ValueError — the same invariant enforced by validate_training_records in plan mode.

        A pretrain inactive label (pEC50 < threshold) combined with an oracle exact
        measurement produces contradictory LEFT and EXACT Tobit branches for the same compound.
        """
        from moal.loop import _merge_pretrain_with_oracle
        from moal.types import CensoringType, LabelRecord, QueryType

        smiles = "CCO"
        pretrain_ps_left = [
            LabelRecord(
                smiles=smiles,
                canonical_smiles=smiles,
                value=5.0,
                upper_bound=5.0,
                censoring_type=CensoringType.LEFT,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=1.0,
                iteration=0,
            )
        ]
        oracle_drc = [
            LabelRecord(
                smiles=smiles,
                canonical_smiles=smiles,
                value=7.8,
                upper_bound=7.8,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=1,
            )
        ]
        tracker: set[str] = set()
        with pytest.raises(ValueError, match="mixed-fidelity combination is unsupported"):
            _merge_pretrain_with_oracle(pretrain_ps_left, oracle_drc, tracker)

    def test_pretrain_ps_interval_deduped_when_oracle_upgrades(self):
        """Pretrain PS INTERVAL record must be dropped when oracle has a DRC record for the same compound."""
        from moal.loop import _merge_pretrain_with_oracle
        from moal.types import CensoringType, LabelRecord, QueryType

        smiles = "CCO"
        pretrain_ps = [
            LabelRecord(
                smiles=smiles,
                canonical_smiles=smiles,
                value=5.0,
                upper_bound=11.0,
                censoring_type=CensoringType.INTERVAL,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=1.0,
                iteration=0,
            )
        ]
        oracle_drc = [
            LabelRecord(
                smiles=smiles,
                canonical_smiles=smiles,
                value=7.8,
                upper_bound=7.8,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=2,
            )
        ]
        tracker: set[str] = set()
        merged = _merge_pretrain_with_oracle(pretrain_ps, oracle_drc, tracker)

        # PS INTERVAL for upgraded compound must be removed by training_records_for_refit
        ps_records = [r for r in merged if r.fidelity == QueryType.PRIMARY_SCREEN]
        assert len(ps_records) == 0
        drc_records = [r for r in merged if r.fidelity == QueryType.DOSE_RESPONSE]
        assert len(drc_records) == 1
