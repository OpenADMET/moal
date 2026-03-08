"""Tests for PipelineEvaluator.evaluate_model and ModelMetric."""

from __future__ import annotations

from unittest.mock import create_autospec

import numpy as np
import pytest

from moal.evaluation import ModelMetric, PipelineEvaluator
from moal.model import ChemPropLightningModule


@pytest.fixture
def evaluator():
    """PipelineEvaluator configured with a standard activity threshold for metric tests."""
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
    """Tests that each metric returns its ideal value when predictions exactly match ground truth."""
    @pytest.mark.parametrize("metric,true,expected", [
        (ModelMetric.MAE,         np.array([5.0, 6.0, 7.0, 8.0]),          0.0),
        (ModelMetric.RMSE,        np.array([5.0, 6.0, 7.0]),                0.0),
        (ModelMetric.KENDALL_TAU, np.array([1.0, 2.0, 3.0, 4.0, 5.0]),     1.0),
        (ModelMetric.SPEARMAN_R,  np.array([1.0, 2.0, 3.0, 4.0]),          1.0),
        (ModelMetric.R2,          np.array([4.0, 5.0, 6.0, 7.0, 8.0]),     1.0),
    ])
    def test_perfect_prediction(self, evaluator, metric, true, expected):
        """When predictions equal ground truth, every metric must return its best possible value (0 for MAE/RMSE, 1 for rank/R2)."""
        model = _mock_model(true.copy())
        result = evaluator.evaluate_model(model, list("ABCDE"[:len(true)]), true, metric)
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Known error values
# ---------------------------------------------------------------------------

class TestKnownErrors:
    """Tests that each metric matches analytically-computed values for known prediction errors."""
    @pytest.mark.parametrize("metric,true,preds,expected", [
        (ModelMetric.MAE,         np.array([5.0, 6.0, 7.0]),  np.array([6.0, 7.0, 8.0]),  1.0),
        (ModelMetric.RMSE,        np.array([0.0, 0.0]),        np.array([1.0, 1.0]),        1.0),
        (ModelMetric.KENDALL_TAU, np.array([1.0, 2.0, 3.0]),  np.array([3.0, 2.0, 1.0]),  -1.0),
        (ModelMetric.SPEARMAN_R,  np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0]), -1.0),
    ])
    def test_known_error(self, evaluator, metric, true, preds, expected):
        """Metrics must match hand-calculated values, confirming the underlying formula is implemented correctly."""
        model = _mock_model(preds)
        result = evaluator.evaluate_model(model, list("ABCD"[:len(true)]), true, metric)
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests that degenerate inputs (empty sets, constant targets, single points) are handled gracefully."""
    def test_empty_smiles_returns_nan(self, evaluator):
        """With no test points the function must return nan and must not call predict_smiles, since there is nothing to evaluate."""
        model = create_autospec(ChemPropLightningModule, instance=True)
        result = evaluator.evaluate_model(model, [], np.array([]), ModelMetric.MAE)
        assert np.isnan(result)
        # The early-return guard must short-circuit before any model call.
        model.predict_smiles.assert_not_called()

    @pytest.mark.parametrize("metric", [ModelMetric.KENDALL_TAU, ModelMetric.SPEARMAN_R])
    def test_single_point_ranking_metric_is_nan(self, evaluator, metric):
        """Single-point ranking is undefined → nan."""
        model = _mock_model(np.array([5.0]))
        result = evaluator.evaluate_model(model, ["A"], np.array([5.0]), metric)
        assert np.isnan(result)

    def test_r2_constant_truth(self, evaluator):
        """If all true values are the same, SS_tot = 0 → R² is nan."""
        true = np.array([5.0, 5.0, 5.0])
        model = _mock_model(np.array([4.0, 5.0, 6.0]))
        result = evaluator.evaluate_model(model, ["A", "B", "C"], true, ModelMetric.R2)
        assert np.isnan(result)

    def test_invalid_metric_raises(self, evaluator):
        """Passing an unrecognized metric value must raise ValueError rather than silently returning a garbage result."""
        with pytest.raises(ValueError):
            evaluator.evaluate_model(
                _mock_model(np.array([5.0])), ["A"], np.array([5.0]),
                "not_a_metric"  # type: ignore
            )
