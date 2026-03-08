"""Tests for CensoredRegressionLoss: gradient checks and branch correctness."""

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
    """Helper that builds a minimal LabelRecord for loss function tests."""
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
    """Tests for the EXACT censoring branch of CensoredRegressionLoss (standard squared-error-like behaviour)."""

    def test_zero_loss_at_truth(self):
        """When prediction equals the true value, the EXACT branch must return ~0 loss."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([5.0])
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_increases_with_error(self):
        """Larger prediction errors must produce strictly larger loss, confirming monotonicity of the EXACT branch."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss_small = loss_fn(torch.tensor([5.1]), [rec])
        loss_large = loss_fn(torch.tensor([6.0]), [rec])
        assert loss_large.item() > loss_small.item()

    def test_gradient_direction(self):
        """When prediction overshoots truth, the gradient must be positive so that the optimizer pushes it back down."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        pred = torch.tensor([7.0], requires_grad=True)
        rec = _make_record(5.0, 5.0, CensoringType.EXACT)
        loss = loss_fn(pred, [rec])
        loss.backward()
        # Prediction is above truth → gradient should be positive (push down)
        assert pred.grad.item() > 0


class TestLeftBranch:
    """Tests for the LEFT censoring branch of CensoredRegressionLoss (confirmed inactive compounds)."""

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
    """Tests for the INTERVAL censoring branch of CensoredRegressionLoss (active compounds with pEC50 in [T, upper_bound])."""

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
    """Tests that per-fidelity loss weights (w_drc, w_ps) are correctly applied to DRC vs PS samples."""

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
    """Tests for the learnable sigma option: parameter existence and lower-bound enforcement."""

    def test_sigma_parameter_exists_and_bounded(self):
        """When learnable_sigma=True, the module must expose exactly one trainable parameter and enforce the minimum-sigma lower bound."""
        loss_fn = CensoredRegressionLoss(sigma=0.5, learnable_sigma=True)
        params = list(loss_fn.parameters())
        assert len(params) == 1
        # Force log_sigma very negative and confirm the lower bound holds.
        with torch.no_grad():
            loss_fn.log_sigma.fill_(-100.0)
        assert loss_fn.sigma.item() >= 0.05 - 1e-9


class TestLossBreakdown:
    """Tests for forward_with_breakdown(): correctness and gradient flow of the returned LossBreakdown NamedTuple."""

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

    @pytest.mark.parametrize(
        "absent_fidelity,nan_field,finite_field",
        [
            (QueryType.DOSE_RESPONSE, "drc_loss", "ps_loss"),
            (QueryType.PRIMARY_SCREEN, "ps_loss", "drc_loss"),
        ],
    )
    def test_absent_fidelity_loss_is_nan(
        self, absent_fidelity, nan_field, finite_field
    ):
        """The per-fidelity loss field must be nan when the batch contains no
        samples of that fidelity, while the other field and total remain finite."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        preds = torch.tensor([5.0, 6.0])
        if absent_fidelity == QueryType.DOSE_RESPONSE:
            # Batch with only PS records
            recs = [
                _make_record(5.0, 5.0, CensoringType.LEFT, QueryType.PRIMARY_SCREEN),
                _make_record(
                    5.0, 11.0, CensoringType.INTERVAL, QueryType.PRIMARY_SCREEN
                ),
            ]
        else:
            # Batch with only DRC records
            recs = [
                _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
                _make_record(6.0, 6.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE),
            ]
        bd = loss_fn.forward_with_breakdown(preds, recs)
        assert getattr(bd, nan_field).isnan()
        assert not getattr(bd, finite_field).isnan()
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

    def test_empty_records_raises(self):
        """An empty batch must raise rather than silently return nan or 0.

        torch.stack([]) raises RuntimeError, which is the expected contract —
        callers are responsible for not passing empty batches.
        """
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        with pytest.raises((RuntimeError, AssertionError)):
            loss_fn(torch.tensor([]), [])

    def test_mismatched_preds_records_length(self):
        """Mismatched predictions and records lengths must raise AssertionError."""
        loss_fn = CensoredRegressionLoss(sigma=1.0)
        preds = torch.tensor([5.0, 6.0, 7.0])  # 3 predictions
        recs = [
            _make_record(5.0, 5.0, CensoringType.EXACT, QueryType.DOSE_RESPONSE)
        ]  # 1 record
        with pytest.raises(AssertionError):
            loss_fn(preds, recs)
