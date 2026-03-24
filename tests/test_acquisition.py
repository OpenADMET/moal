"""Tests for CostAwareGreedyAcquisition: score properties and selection logic."""

import numpy as np
import pytest

from moal.acquisition import CostAwareGreedyAcquisition, _binary_entropy, _sigmoid
from moal.types import QueryType


@pytest.fixture
def acq():
    """Standard acquisition object with asymmetric PS/DRC costs and a target threshold of 7.0."""
    return CostAwareGreedyAcquisition(
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        target_threshold=7.0,
        tau=0.5,
    )


class TestScoreHelpers:
    """Tests for the _sigmoid and _binary_entropy scoring utility functions."""

    def test_binary_entropy_max_at_half(self):
        """H(p) should be maximized at p=0.5."""
        p = np.array([0.1, 0.5, 0.9])
        h = _binary_entropy(p)
        assert h[1] > h[0]
        assert h[1] > h[2]

    @pytest.mark.parametrize("p", [1e-9, 1 - 1e-9])
    def test_binary_entropy_zero_at_extremes(self, p):
        """Binary entropy must approach 0 as p approaches 0 or 1, since the outcome is certain."""
        h = _binary_entropy(np.array([p]))
        assert h[0] == pytest.approx(0.0, abs=1e-3)

    def test_sigmoid_monotone(self):
        """The sigmoid must be strictly increasing over its domain so that higher predictions always score higher."""
        x = np.linspace(-5, 5, 20)
        s = _sigmoid(x, tau=0.5)
        assert np.all(np.diff(s) > 0)


class TestDRCScore:
    """Tests for _score_drc(): the exploitation score that rewards high predicted pEC50."""

    def test_higher_prediction_higher_drc_score(self, acq):
        """A compound with a higher predicted pEC50 must receive a higher DRC score, reflecting stronger exploitation incentive."""
        preds = np.array([4.0, 8.0])  # B is far more likely active
        scores_drc = acq._score_drc(preds)
        assert scores_drc[1] > scores_drc[0]

    def test_drc_score_normalized_by_cost(self, acq):
        """DRC score must equal the sigmoid output divided by cost_drc so that costly assays are appropriately penalized."""
        preds = np.array([7.5])
        score = acq._score_drc(preds)[0]
        assert score == pytest.approx(_sigmoid(np.array([7.5 - 7.0]), 0.5)[0] / 10.0, rel=1e-4)


class TestPSScore:
    """Tests for _score_ps(): the exploration score that rewards uncertainty near the PS threshold."""

    def test_ps_score_maximized_near_threshold(self, acq):
        """PS score should peak near ps_threshold (max entropy)."""
        preds = np.array([3.0, 5.0, 7.0])
        scores_ps = acq._score_ps(preds)
        # Score at ps_threshold (5.0) should be highest
        assert scores_ps[1] > scores_ps[0]
        assert scores_ps[1] > scores_ps[2]

    def test_ps_score_normalized_by_cost(self, acq):
        """PS score must equal binary entropy divided by cost_ps so that cheap screening is appropriately rewarded."""
        preds = np.array([5.0])
        score = acq._score_ps(preds)[0]
        p_cross = _sigmoid(np.array([5.0 - 5.0]), 0.5)[0]
        expected = float(_binary_entropy(np.array([p_cross]))[0]) / 1.0
        assert score == pytest.approx(expected, rel=1e-4)


class TestSelect:
    """Integration tests for CostAwareGreedyAcquisition.select()."""

    def test_returns_k_unique_queries(self, acq):
        """Select must return no repeated SMILES, since a compound should only be queried once."""
        smiles = [f"C{i}" for i in range(20)]
        preds = np.random.default_rng(0).normal(6.0, 1.5, 20).astype(np.float32)
        selected = acq.select(smiles, preds, plate_size=5, wells_per_ps=1, wells_per_drc=1)
        assert len(selected) == 5
        selected_smiles = [s for s, _ in selected]
        assert len(selected_smiles) == len(set(selected_smiles))

    def test_fidelity_types_are_valid(self, acq):
        """Every selection must be either PS or DRC — no other query type should ever be emitted."""
        smiles = [f"C{i}" for i in range(10)]
        preds = np.ones(10, dtype=np.float32) * 6.0
        selected = acq.select(smiles, preds, plate_size=8, wells_per_ps=1, wells_per_drc=1)
        for _, qt in selected:
            assert qt in (QueryType.PRIMARY_SCREEN, QueryType.DOSE_RESPONSE)

    def test_high_pec50_prefers_drc(self, acq):
        """Compounds with very high predicted pEC50 should prefer DRC."""
        smiles = ["high", "low"]
        preds = np.array([9.5, 3.0], dtype=np.float32)
        selected = acq.select(smiles, preds, plate_size=1, wells_per_ps=1, wells_per_drc=1)
        assert selected[0] == ("high", QueryType.DOSE_RESPONSE)

    def test_at_threshold_prefers_ps(self, acq):
        """Compounds at the PS threshold (max entropy) should prefer PS when cheap."""
        smiles = ["at_threshold"]
        preds = np.array([5.0], dtype=np.float32)
        selected = acq.select(smiles, preds, plate_size=1, wells_per_ps=1, wells_per_drc=1)
        assert selected[0][1] == QueryType.PRIMARY_SCREEN

    @pytest.mark.parametrize(
        "smiles,preds,plate_size",
        [
            ([], np.array([]), 5),
            ([f"C{i}" for i in range(10)], np.ones(10, dtype=np.float32) * 6.0, 0),
        ],
    )
    def test_empty_selection_returns_empty(self, acq, smiles, preds, plate_size):
        """An empty pool or plate_size=0 must return [] without error, as there is nothing to select."""
        assert (
            acq.select(smiles, preds, plate_size=plate_size, wells_per_ps=1, wells_per_drc=1) == []
        )

    def test_plate_larger_than_pool(self, acq):
        """When plate_size exceeds the pool's total well cost, all available compounds must be returned."""
        smiles = ["A", "B"]
        preds = np.array([5.0, 6.0], dtype=np.float32)
        selected = acq.select(smiles, preds, plate_size=100, wells_per_ps=1, wells_per_drc=1)
        assert len(selected) == 2  # limited by pool size

    def test_invalid_cost_raises(self):
        """Negative cost values must raise ValueError at construction time before any scoring occurs."""
        with pytest.raises(ValueError, match="positive"):
            CostAwareGreedyAcquisition(cost_ps=-1.0, cost_drc=10.0)

    def test_degenerate_thresholds_still_selects(self, acq):
        """When ps_threshold == target_threshold both scoring functions compete at the same
        point; the acquisition must still return valid (smiles, query_type) pairs.
        """
        degenerate = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=7.0,
            target_threshold=7.0,
            tau=0.5,
        )
        smiles = [f"C{i}" for i in range(5)]
        preds = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
        selected = degenerate.select(smiles, preds, plate_size=3, wells_per_ps=1, wells_per_drc=1)
        assert len(selected) == 3
        for _, qt in selected:
            assert qt in (QueryType.PRIMARY_SCREEN, QueryType.DOSE_RESPONSE)

    def test_select_with_nan_predictions(self, acq):
        """NaN in predictions must raise ValueError before scoring.

        Silently allowing NaN propagates undefined float comparisons into the
        sort step and produces non-deterministic selection results.
        """
        smiles = ["A", "B", "C"]
        preds = np.array([float("nan"), 6.0, 7.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            acq.select(smiles, preds, plate_size=2, wells_per_ps=1, wells_per_drc=1)

    def test_drc_cost_stops_at_plate_boundary(self, acq):
        """Selection must stop when the next DRC candidate would overflow the plate.

        With plate_size=14, wells_per_drc=13, and wells_per_ps=1, the top
        candidate (DRC, 13 wells) fills most of the plate.  The second candidate
        is also a DRC (13 wells), which would push total wells to 26 > 14, so
        the loop must hard-stop and return only the first compound.
        """
        acq_asymmetric = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )
        smiles = ["high1", "high2", "low"]
        # Both high compounds score heavily for DRC; low scores for PS
        preds = np.array([9.5, 9.0, 3.0], dtype=np.float32)
        selected = acq_asymmetric.select(
            smiles, preds, plate_size=14, wells_per_ps=1, wells_per_drc=13
        )
        # First candidate: DRC for "high1" (13 wells used; 1 well left)
        # Second candidate: DRC for "high2" would use 13 more → 26 > 14 → stop
        assert len(selected) == 1
        assert selected[0] == ("high1", QueryType.DOSE_RESPONSE)

    def test_wells_used_never_exceeds_plate_size(self):
        """Total wells consumed by the selection must never exceed plate_size."""
        acq = CostAwareGreedyAcquisition(cost_ps=1.0, cost_drc=10.0)
        rng = np.random.default_rng(7)
        smiles = [f"C{i}" for i in range(50)]
        preds = rng.normal(6.0, 1.5, 50).astype(np.float32)
        plate_size, wells_ps, wells_drc = 100, 1, 13
        selected = acq.select(
            smiles, preds, plate_size=plate_size, wells_per_ps=wells_ps, wells_per_drc=wells_drc
        )
        total_wells = sum(
            wells_drc if qt == QueryType.DOSE_RESPONSE else wells_ps for _, qt in selected
        )
        assert total_wells <= plate_size


class TestPSUpgradeCandidates:
    """Acquisition behaviour when ps_labeled_smiles pool is supplied."""

    @pytest.fixture
    def acq(self):
        """Acquisition fixture scoped to this class for upgrade-candidate tests."""
        return CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )

    def test_ps_labeled_compounds_only_generate_drc_candidates(self, acq):
        """Compounds in the PS-labeled pool must not be selectable as PS."""
        ps_smiles = ["A"]
        ps_preds = np.array([9.0], dtype=np.float32)
        selected = acq.select(
            [],
            np.array([]),
            plate_size=1,
            wells_per_ps=1,
            wells_per_drc=1,
            ps_labeled_smiles=ps_smiles,
            ps_labeled_predictions=ps_preds,
        )
        assert len(selected) == 1
        assert selected[0] == ("A", QueryType.DOSE_RESPONSE)

    def test_ps_labeled_and_unlabeled_compete_correctly(self, acq):
        """A high-scoring PS-labeled DRC candidate must beat a low-scoring unlabeled PS."""
        unlabeled = ["B"]
        unlabeled_preds = np.array([3.0], dtype=np.float32)  # low DRC score
        ps_labeled = ["A"]
        ps_labeled_preds = np.array([9.0], dtype=np.float32)  # high DRC score
        selected = acq.select(
            unlabeled,
            unlabeled_preds,
            plate_size=1,
            wells_per_ps=1,
            wells_per_drc=1,
            ps_labeled_smiles=ps_labeled,
            ps_labeled_predictions=ps_labeled_preds,
        )
        assert selected[0] == ("A", QueryType.DOSE_RESPONSE)

    def test_empty_ps_labeled_behaves_like_no_pool(self, acq):
        """Passing ps_labeled_smiles=[] must not change behaviour vs omitting the argument."""
        smiles = [f"C{i}" for i in range(5)]
        preds = np.ones(5, dtype=np.float32) * 6.0
        without_pool = acq.select(smiles, preds, plate_size=3, wells_per_ps=1, wells_per_drc=1)
        with_empty_pool = acq.select(
            smiles,
            preds,
            plate_size=3,
            wells_per_ps=1,
            wells_per_drc=1,
            ps_labeled_smiles=[],
            ps_labeled_predictions=None,
        )
        assert without_pool == with_empty_pool

    def test_no_smiles_length_mismatch_assertion(self, acq):
        """Mismatched ps_labeled_smiles and ps_labeled_predictions must raise ValueError."""
        with pytest.raises(ValueError):
            acq.select(
                [],
                np.array([]),
                plate_size=1,
                wells_per_ps=1,
                wells_per_drc=1,
                ps_labeled_smiles=["A", "B"],
                ps_labeled_predictions=np.array([1.0]),
            )
