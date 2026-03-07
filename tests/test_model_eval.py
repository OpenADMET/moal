"""Tests for PipelineEvaluator.evaluate_model and ModelMetric."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from moal.evaluation import ModelMetric, PipelineEvaluator
from moal.model import ChemPropLightningModule


@pytest.fixture
def evaluator():
    return PipelineEvaluator(activity_threshold=7.0)


def _mock_model(preds: np.ndarray):
    """Create a spec-d mock whose predict_smiles() returns preds."""
    model = create_autospec(ChemPropLightningModule, instance=True)
    model.predict_smiles.return_value = preds
    return model


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------

class TestPerfectPredictions:
    def test_mae_zero(self, evaluator):
        true = np.array([5.0, 6.0, 7.0, 8.0])
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, ["A", "B", "C", "D"], true, ModelMetric.MAE)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_rmse_zero(self, evaluator):
        true = np.array([5.0, 6.0, 7.0])
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, ["A", "B", "C"], true, ModelMetric.RMSE)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_kendall_tau_one(self, evaluator):
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, list("ABCDE"), true, ModelMetric.KENDALL_TAU)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_spearman_one(self, evaluator):
        true = np.array([1.0, 2.0, 3.0, 4.0])
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, list("ABCD"), true, ModelMetric.SPEARMAN_R)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_r2_one(self, evaluator):
        true = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, list("ABCDE"), true, ModelMetric.R2)
        assert result == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Known error values
# ---------------------------------------------------------------------------

class TestKnownErrors:
    def test_mae_constant_offset(self, evaluator):
        true = np.array([5.0, 6.0, 7.0])
        preds = true + 1.0  # constant offset of 1
        model = _mock_model(preds)
        result = evaluator.evaluate_model(model, ["A", "B", "C"], true, ModelMetric.MAE)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_rmse_known(self, evaluator):
        true = np.array([0.0, 0.0])
        preds = np.array([1.0, 1.0])  # error of 1 each → RMSE = 1
        model = _mock_model(preds)
        result = evaluator.evaluate_model(model, ["A", "B"], true, ModelMetric.RMSE)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_kendall_tau_inverted(self, evaluator):
        true  = np.array([1.0, 2.0, 3.0])
        preds = np.array([3.0, 2.0, 1.0])  # perfectly inverted → τ = -1
        model = _mock_model(preds)
        result = evaluator.evaluate_model(model, ["A", "B", "C"], true, ModelMetric.KENDALL_TAU)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_spearman_inverted(self, evaluator):
        true  = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([4.0, 3.0, 2.0, 1.0])
        model = _mock_model(preds)
        result = evaluator.evaluate_model(model, list("ABCD"), true, ModelMetric.SPEARMAN_R)
        assert result == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_smiles_returns_nan(self, evaluator):
        model = create_autospec(ChemPropLightningModule, instance=True)
        result = evaluator.evaluate_model(model, [], np.array([]), ModelMetric.MAE)
        assert np.isnan(result)
        # The early-return guard must short-circuit before any model call.
        model.predict_smiles.assert_not_called()

    def test_single_point_kendall(self, evaluator):
        """Single-point ranking is undefined → nan."""
        model = _mock_model(np.array([5.0]))
        result = evaluator.evaluate_model(model, ["A"], np.array([5.0]), ModelMetric.KENDALL_TAU)
        assert np.isnan(result)

    def test_single_point_spearman(self, evaluator):
        model = _mock_model(np.array([5.0]))
        result = evaluator.evaluate_model(model, ["A"], np.array([5.0]), ModelMetric.SPEARMAN_R)
        assert np.isnan(result)

    def test_r2_constant_truth(self, evaluator):
        """If all true values are the same, SS_tot = 0 → R² is nan."""
        true = np.array([5.0, 5.0, 5.0])
        model = _mock_model(np.array([4.0, 5.0, 6.0]))
        result = evaluator.evaluate_model(model, ["A", "B", "C"], true, ModelMetric.R2)
        assert np.isnan(result)

    def test_invalid_metric_raises(self, evaluator):
        with pytest.raises(ValueError):
            evaluator.evaluate_model(
                _mock_model(np.array([5.0])), ["A"], np.array([5.0]),
                "not_a_metric"  # type: ignore
            )
