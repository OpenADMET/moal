"""Tests for CostAwareGreedyAcquisition: score properties and selection logic."""

import math

import numpy as np
import pytest

from moal.acquisition import CostAwareGreedyAcquisition, _binary_entropy, _sigmoid
from moal.types import QueryType


@pytest.fixture
def acq():
    return CostAwareGreedyAcquisition(
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        target_threshold=7.0,
        tau=0.5,
    )


class TestScoreHelpers:
    def test_binary_entropy_max_at_half(self):
        """H(p) should be maximized at p=0.5."""
        p = np.array([0.1, 0.5, 0.9])
        h = _binary_entropy(p)
        assert h[1] > h[0]
        assert h[1] > h[2]

    def test_binary_entropy_zero_at_extremes(self):
        p = np.array([1e-9, 1 - 1e-9])
        h = _binary_entropy(p)
        assert h[0] == pytest.approx(0.0, abs=1e-3)
        assert h[1] == pytest.approx(0.0, abs=1e-3)

    def test_sigmoid_monotone(self):
        x = np.linspace(-5, 5, 20)
        s = _sigmoid(x, tau=0.5)
        assert np.all(np.diff(s) > 0)


class TestDRCScore:
    def test_higher_prediction_higher_drc_score(self, acq):
        smiles = ["A", "B"]
        preds = np.array([4.0, 8.0])  # B is far more likely active
        scores_drc = acq._score_drc(preds)
        assert scores_drc[1] > scores_drc[0]

    def test_drc_score_normalized_by_cost(self, acq):
        preds = np.array([7.5])
        score = acq._score_drc(preds)[0]
        assert score == pytest.approx(
            _sigmoid(np.array([7.5 - 7.0]), 0.5)[0] / 10.0, rel=1e-4
        )


class TestPSScore:
    def test_ps_score_maximized_near_threshold(self, acq):
        """PS score should peak near ps_threshold (max entropy)."""
        preds = np.array([3.0, 5.0, 7.0])
        scores_ps = acq._score_ps(preds)
        # Score at ps_threshold (5.0) should be highest
        assert scores_ps[1] > scores_ps[0]
        assert scores_ps[1] > scores_ps[2]

    def test_ps_score_normalized_by_cost(self, acq):
        preds = np.array([5.0])
        score = acq._score_ps(preds)[0]
        p_cross = _sigmoid(np.array([5.0 - 5.0]), 0.5)[0]
        expected = float(_binary_entropy(np.array([p_cross]))[0]) / 1.0
        assert score == pytest.approx(expected, rel=1e-4)


class TestSelect:
    def test_returns_k_queries(self, acq):
        smiles = [f"C{i}" for i in range(20)]
        preds = np.random.default_rng(0).normal(6.0, 1.5, 20).astype(np.float32)
        selected = acq.select(smiles, preds, k=5)
        assert len(selected) == 5

    def test_no_duplicate_compounds(self, acq):
        smiles = [f"C{i}" for i in range(20)]
        preds = np.random.default_rng(1).normal(6.0, 1.5, 20).astype(np.float32)
        selected = acq.select(smiles, preds, k=10)
        selected_smiles = [s for s, _ in selected]
        assert len(selected_smiles) == len(set(selected_smiles))

    def test_fidelity_types_are_valid(self, acq):
        smiles = [f"C{i}" for i in range(10)]
        preds = np.ones(10, dtype=np.float32) * 6.0
        selected = acq.select(smiles, preds, k=8)
        for _, qt in selected:
            assert qt in (QueryType.PRIMARY_SCREEN, QueryType.DOSE_RESPONSE)

    def test_high_pec50_prefers_drc(self, acq):
        """Compounds with very high predicted pEC50 should prefer DRC."""
        smiles = ["high", "low"]
        preds = np.array([9.5, 3.0], dtype=np.float32)
        selected = acq.select(smiles, preds, k=1)
        assert selected[0] == ("high", QueryType.DOSE_RESPONSE)

    def test_at_threshold_prefers_ps(self, acq):
        """Compounds at the PS threshold (max entropy) should prefer PS when cheap."""
        smiles = ["at_threshold"]
        preds = np.array([5.0], dtype=np.float32)
        selected = acq.select(smiles, preds, k=1)
        assert selected[0][1] == QueryType.PRIMARY_SCREEN

    def test_empty_unlabeled_returns_empty(self, acq):
        result = acq.select([], np.array([]), k=5)
        assert result == []

    def test_k_larger_than_pool(self, acq):
        smiles = ["A", "B"]
        preds = np.array([5.0, 6.0], dtype=np.float32)
        selected = acq.select(smiles, preds, k=100)
        assert len(selected) == 2  # limited by pool size

    def test_invalid_cost_raises(self):
        with pytest.raises(ValueError, match="positive"):
            CostAwareGreedyAcquisition(cost_ps=-1.0, cost_drc=10.0)

    def test_k_zero_returns_empty(self, acq):
        """k=0 must return an empty list without touching the pool."""
        smiles = [f"C{i}" for i in range(10)]
        preds = np.ones(10, dtype=np.float32) * 6.0
        assert acq.select(smiles, preds, k=0) == []

    def test_degenerate_thresholds_still_selects(self, acq):
        """When ps_threshold == target_threshold both scoring functions compete at the same
        point; the acquisition must still return valid (smiles, query_type) pairs."""
        degenerate = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=7.0,
            target_threshold=7.0,
            tau=0.5,
        )
        smiles = [f"C{i}" for i in range(5)]
        preds = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
        selected = degenerate.select(smiles, preds, k=3)
        assert len(selected) == 3
        for _, qt in selected:
            assert qt in (QueryType.PRIMARY_SCREEN, QueryType.DOSE_RESPONSE)
