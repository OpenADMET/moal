"""Tests for CensoredRegressionLoss: gradient checks and branch correctness."""

import math

import pytest
import torch

from moal.loss import CensoredRegressionLoss, LossBreakdown
from moal.types import CensoringType, LabelRecord, QueryType


def _make_record(
    value: float,
    upper_bound: float,
    censoring_type: CensoringType,
    fidelity: QueryType = QueryType.DOSE_RESPONSE,
) -> LabelRecord:
    return LabelRecord(
        smiles="C",
        canonical_smiles="C",
        value=value,
        upper_bound=upper_bound,
        censoring_type=censoring_type,
        fidelity=fidelity,
        cost=1.0,
        iteration=0,
    )


class TestExactBranch:
    def test_zero_loss_at_truth(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([5.0])
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_increases_with_error(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss_small = loss_fn(torch.tensor([5.1]), [rec])
        loss_large = loss_fn(torch.tensor([6.0]), [rec])
        assert loss_large.item() > loss_small.item()

    def test_gradient_direction(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([7.0], requires_grad=True)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        loss.backward()
        # Prediction is above truth → gradient should be positive (push down)
        assert pred.grad.item() > 0


class TestLeftBranch:
    def test_loss_decreases_as_prediction_moves_below_threshold(self):
        """Lower predictions should incur lower LEFT-branch loss."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T = 5.0
        rec = _make_record(T, T, CensoringType.LEFT)
        loss_above = loss_fn(torch.tensor([6.0]), [rec])
        loss_at = loss_fn(torch.tensor([5.0]), [rec])
        loss_below = loss_fn(torch.tensor([3.0]), [rec])
        assert loss_above.item() > loss_at.item() > loss_below.item()

    def test_gradient_direction(self):
        """Predicting above the threshold should push gradient downward."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T = 5.0
        pred = torch.tensor([6.0], requires_grad=True)
        rec = _make_record(T, T, CensoringType.LEFT)
        loss = loss_fn(pred, [rec])
        loss.backward()
        assert pred.grad.item() > 0  # gradient > 0 → prediction should decrease


class TestIntervalBranch:
    def test_loss_minimized_inside_interval(self):
        """Predictions inside [T, U] should incur lower loss than outside."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss_inside = loss_fn(torch.tensor([7.0]), [rec])
        loss_below = loss_fn(torch.tensor([3.0]), [rec])
        loss_above = loss_fn(torch.tensor([13.0]), [rec])
        assert loss_inside.item() < loss_below.item()
        assert loss_inside.item() < loss_above.item()

    def test_gradient_direction_below_interval(self):
        """Predicting below T should have gradient pushing prediction upward."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        pred = torch.tensor([3.0], requires_grad=True)
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss = loss_fn(pred, [rec])
        loss.backward()
        assert pred.grad.item() < 0  # gradient < 0 → prediction should increase

    def test_interval_not_right_censored(self):
        """Interval branch must NOT penalise predictions of 8+ for [5, 11].

        This guards against the right-censoring bug: treating >= T as
        right-censored at T would produce increasing loss for ŷ >> T.
        """
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss_at_T = loss_fn(torch.tensor([5.0]), [rec])
        loss_at_8 = loss_fn(torch.tensor([8.0]), [rec])
        # Loss should be lower (or equal) for a prediction deeper inside [T, U]
        assert loss_at_8.item() <= loss_at_T.item() + 1e-4


class TestFidelityWeighting:
    def test_ps_loss_less_than_drc_for_same_error(self):
        """PS (inequality) samples should contribute less to total loss."""
        loss_fn = CensoredRegressionLoss(sigma=1.0, w_drc=1.0, w_ps=0.3)
        rec_drc = _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE)
        rec_ps = _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.PRIMARY_SCREEN)
        pred = torch.tensor([7.0])
        loss_drc = loss_fn(pred, [rec_drc])
        loss_ps = loss_fn(pred, [rec_ps])
        assert loss_ps.item() < loss_drc.item()


class TestLearnableSigma:
    def test_sigma_parameter_exists(self):
        loss_fn = CensoredRegressionLoss(sigma=0.5, learnable_sigma=True)
        params = list(loss_fn.parameters())
        assert len(params) == 1

    def test_sigma_bounded_from_below(self):
        loss_fn = CensoredRegressionLoss(sigma=0.5, learnable_sigma=True)
        # Force log_sigma very negative
        with torch.no_grad():
            loss_fn.log_sigma.fill_(-100.0)
        assert loss_fn.sigma.item() >= 0.05 - 1e-9


class TestLossBreakdown:
    def test_forward_consistent_with_breakdown(self):
        """forward() total must match forward_with_breakdown().total."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        preds = torch.tensor([5.0, 6.0, 7.0])
        recs = [
            _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
            _make_record(5.0, 11.0, CensoringType.INTERVAL, QueryType.PRIMARY_SCREEN),
            _make_record(5.0, 5.0, CensoringType.LEFT, QueryType.PRIMARY_SCREEN),
        ]
        total_scalar = loss_fn(preds, recs)
        breakdown = loss_fn.forward_with_breakdown(preds, recs)
        assert isinstance(breakdown, LossBreakdown)
        assert breakdown.total.item() == pytest.approx(total_scalar.item(), rel=1e-5)

    def test_breakdown_separates_fidelities(self):
        """drc_loss and ps_loss should differ when fidelity weights differ."""
        loss_fn = CensoredRegressionLoss(sigma=1.0, w_drc=1.0, w_ps=0.3)
        preds = torch.tensor([7.0, 7.0])
        recs = [
            _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
            _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.PRIMARY_SCREEN),
        ]
        bd = loss_fn.forward_with_breakdown(preds, recs)
        assert not bd.drc_loss.isnan()
        assert not bd.ps_loss.isnan()
        assert bd.drc_loss.item() > bd.ps_loss.item()

    def test_drc_loss_nan_when_no_drc_records(self):
        """drc_loss should be nan if batch contains only PS records."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        preds = torch.tensor([5.0, 6.0])
        recs = [
            _make_record(5.0, 5.0, CensoringType.LEFT, QueryType.PRIMARY_SCREEN),
            _make_record(5.0, 11.0, CensoringType.INTERVAL, QueryType.PRIMARY_SCREEN),
        ]
        bd = loss_fn.forward_with_breakdown(preds, recs)
        assert bd.drc_loss.isnan()
        assert not bd.ps_loss.isnan()
        # Total should still be finite (PS losses only)
        assert not bd.total.isnan()

    def test_ps_loss_nan_when_no_ps_records(self):
        """ps_loss should be nan if batch contains only DRC records."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        preds = torch.tensor([5.0, 6.0])
        recs = [
            _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
            _make_record(6.0, 6.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
        ]
        bd = loss_fn.forward_with_breakdown(preds, recs)
        assert not bd.drc_loss.isnan()
        assert bd.ps_loss.isnan()
        assert not bd.total.isnan()

    def test_breakdown_gradient_flows_through_total(self):
        """Gradients must flow through breakdown.total to the prediction."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([7.0], requires_grad=True)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE)
        bd = loss_fn.forward_with_breakdown(pred, [rec])
        bd.total.backward()
        assert pred.grad is not None
        assert pred.grad.item() != 0.0

    def test_zero_loss_at_truth(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([5.0])
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_increases_with_error(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss_small = loss_fn(torch.tensor([5.1]), [rec])
        loss_large = loss_fn(torch.tensor([6.0]), [rec])
        assert loss_large.item() > loss_small.item()

    def test_gradient_direction(self):
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([7.0], requires_grad=True)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        loss.backward()
        # Prediction is above truth → gradient should be positive (push down)
        assert pred.grad.item() > 0


class TestLeftBranch:
    def test_loss_decreases_as_prediction_moves_below_threshold(self):
        """Lower predictions should incur lower LEFT-branch loss."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T = 5.0
        rec = _make_record(T, T, CensoringType.LEFT)
        loss_above = loss_fn(torch.tensor([6.0]), [rec])
        loss_at = loss_fn(torch.tensor([5.0]), [rec])
        loss_below = loss_fn(torch.tensor([3.0]), [rec])
        assert loss_above.item() > loss_at.item() > loss_below.item()

    def test_gradient_direction(self):
        """Predicting above the threshold should push gradient downward."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T = 5.0
        pred = torch.tensor([6.0], requires_grad=True)
        rec = _make_record(T, T, CensoringType.LEFT)
        loss = loss_fn(pred, [rec])
        loss.backward()
        assert pred.grad.item() > 0  # gradient > 0 → prediction should decrease


class TestIntervalBranch:
    def test_loss_minimized_inside_interval(self):
        """Predictions inside [T, U] should incur lower loss than outside."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss_inside = loss_fn(torch.tensor([7.0]), [rec])
        loss_below = loss_fn(torch.tensor([3.0]), [rec])
        loss_above = loss_fn(torch.tensor([13.0]), [rec])
        assert loss_inside.item() < loss_below.item()
        assert loss_inside.item() < loss_above.item()

    def test_gradient_direction_below_interval(self):
        """Predicting below T should have gradient pushing prediction upward."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        pred = torch.tensor([3.0], requires_grad=True)
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss = loss_fn(pred, [rec])
        loss.backward()
        assert pred.grad.item() < 0  # gradient < 0 → prediction should increase

    def test_interval_not_right_censored(self):
        """Interval branch must NOT penalise predictions of 8+ for [5, 11].

        This guards against the right-censoring bug: treating >= T as
        right-censored at T would produce increasing loss for ŷ >> T.
        """
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        T, U = 5.0, 11.0
        rec = _make_record(T, U, CensoringType.INTERVAL)
        loss_at_T = loss_fn(torch.tensor([5.0]), [rec])
        loss_at_8 = loss_fn(torch.tensor([8.0]), [rec])
        # Loss should be lower (or equal) for a prediction deeper inside [T, U]
        assert loss_at_8.item() <= loss_at_T.item() + 1e-4


class TestFidelityWeighting:
    def test_ps_loss_less_than_drc_for_same_error(self):
        """PS (inequality) samples should contribute less to total loss."""
        loss_fn = CensoredRegressionLoss(sigma=1.0, w_drc=1.0, w_ps=0.3)
        rec_drc = _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE)
        rec_ps = _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.PRIMARY_SCREEN)
        pred = torch.tensor([7.0])
        loss_drc = loss_fn(pred, [rec_drc])
        loss_ps = loss_fn(pred, [rec_ps])
        assert loss_ps.item() < loss_drc.item()


class TestLearnableSigma:
    def test_sigma_parameter_exists(self):
        loss_fn = CensoredRegressionLoss(sigma=0.5, learnable_sigma=True)
        params = list(loss_fn.parameters())
        assert len(params) == 1

    def test_sigma_bounded_from_below(self):
        loss_fn = CensoredRegressionLoss(sigma=0.5, learnable_sigma=True)
        # Force log_sigma very negative
        with torch.no_grad():
            loss_fn.log_sigma.fill_(-100.0)
        assert loss_fn.sigma.item() >= 0.05 - 1e-9
