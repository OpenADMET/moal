"""Censored regression loss (Tobit-style) for mixed-fidelity pEC50 data.

Three censoring branches are supported:

- EXACT:    standard squared error, (ŷ - y)² / σ²
- LEFT:     -log Φ((T - ŷ) / σ)            [true value is below threshold T]
- INTERVAL: -log[Φ((U - ŷ)/σ) - Φ((T - ŷ)/σ)]  [true value in [T, U]]

Per-fidelity loss weights prevent the high-volume, low-information primary
screen labels from dominating gradient updates over exact DRC measurements.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from moal.types import CensoringType, LabelRecord, QueryType

# Lower bound on σ to prevent loss collapse when σ is learned.
_SIGMA_MIN = 0.05
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def _normal_log_cdf(x: Tensor) -> Tensor:
    """Log of the standard normal CDF: log Φ(x)."""
    return torch.log(0.5 * (1.0 + torch.erf(x / math.sqrt(2))) + 1e-12)


class LossBreakdown(NamedTuple):
    """Per-fidelity loss components for diagnostic logging.

    All values are scalar tensors. ``drc_loss`` and ``ps_loss`` are
    ``nan`` when the batch contains no samples of that fidelity.
    """

    total: Tensor
    """Weighted mean loss over all samples in the batch."""
    drc_loss: Tensor
    """Mean weighted loss for DOSE_RESPONSE (EXACT) samples, or nan."""
    ps_loss: Tensor
    """Mean weighted loss for PRIMARY_SCREEN (LEFT/INTERVAL) samples, or nan."""


class CensoredRegressionLoss(nn.Module):
    """Mixed-fidelity Tobit regression loss.

    Args:
        sigma: Fixed noise scale in pEC50 log-units. A value of 0.5 is
            consistent with typical intra-assay pEC50 measurement variability
            (coefficient of variation ≈ 0.3–0.5 log units).
        w_drc: Loss weight for EXACT (DRC) samples.
        w_ps: Loss weight for LEFT/INTERVAL (Primary Screen) samples.
            Set < w_drc to prevent low-information inequality labels from
            dominating gradient updates.
        learnable_sigma: If True, σ is a learned scalar parameter (bounded
            from below by _SIGMA_MIN). Requires model output_size=2
            (mean, log_sigma); if False, output_size=1 (mean only).
    """

    def __init__(
        self,
        sigma: float = 0.5,
        w_drc: float = 1.0,
        w_ps: float = 0.3,
        learnable_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.w_drc = w_drc
        self.w_ps = w_ps
        self.learnable_sigma = learnable_sigma

        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        else:
            self.register_buffer("_sigma", torch.tensor(sigma))

    @property
    def sigma(self) -> Tensor:
        if self.learnable_sigma:
            return torch.clamp(self.log_sigma.exp(), min=_SIGMA_MIN)
        return self._sigma  # type: ignore[return-value]

    def _single_loss(self, pred: Tensor, rec: LabelRecord) -> Tensor:
        """Compute the weighted scalar loss for one (prediction, record) pair."""
        sigma = self.sigma
        ct = rec.censoring_type
        T = torch.tensor(rec.value, dtype=pred.dtype, device=pred.device)
        U = torch.tensor(rec.upper_bound, dtype=pred.dtype, device=pred.device)
        w = self.w_drc if rec.fidelity == QueryType.DOSE_RESPONSE else self.w_ps

        if ct == CensoringType.EXACT:
            return w * ((pred - T) / sigma) ** 2

        if ct == CensoringType.LEFT:
            # True value < T; penalise if model predicts above T.
            log_p = _normal_log_cdf((T - pred) / sigma)
            return w * (-log_p)

        if ct == CensoringType.INTERVAL:
            # True value in [T, U]; use log probability mass in the interval.
            log_p_upper = _normal_log_cdf((U - pred) / sigma)
            log_p_lower = _normal_log_cdf((T - pred) / sigma)
            # log(Φ(upper) - Φ(lower)) via log-sum-exp trick for stability.
            log_prob = torch.log(
                torch.clamp(log_p_upper.exp() - log_p_lower.exp(), min=1e-12)
            )
            return w * (-log_prob)

        raise ValueError(f"Unknown CensoringType: {ct}")

    def forward(self, predictions: Tensor, records: list[LabelRecord]) -> Tensor:
        """Compute the mean censored loss over a batch.

        Args:
            predictions: Shape ``(N,)`` — model's pEC50 point estimates.
                If ``learnable_sigma=True``, shape is ``(N, 2)`` where
                column 0 = mean, column 1 = log_sigma (per-sample σ is
                then ignored in favour of a single learned σ for now).
            records: List of N LabelRecord instances corresponding to
                each prediction.

        Returns:
            Scalar mean loss.
        """
        return self.forward_with_breakdown(predictions, records).total

    def forward_with_breakdown(
        self, predictions: Tensor, records: list[LabelRecord]
    ) -> LossBreakdown:
        """Compute per-fidelity loss breakdown for diagnostic logging.

        Args:
            predictions: Same contract as ``forward()``.
            records: List of N LabelRecord instances.

        Returns:
            :class:`LossBreakdown` with ``total``, ``drc_loss``, and
            ``ps_loss``.  ``drc_loss`` / ``ps_loss`` are ``nan`` when the
            batch contains no samples of that fidelity, so they can be
            logged without masking the aggregated loss.
        """
        if predictions.dim() == 2:
            preds = predictions[:, 0]
        else:
            preds = predictions

        assert len(records) == preds.shape[0], (
            f"predictions has {preds.shape[0]} rows but {len(records)} records were provided."
        )

        drc_losses: list[Tensor] = []
        ps_losses: list[Tensor] = []

        for pred, rec in zip(preds, records):
            l = self._single_loss(pred, rec)
            if rec.fidelity == QueryType.DOSE_RESPONSE:
                drc_losses.append(l)
            else:
                ps_losses.append(l)

        _nan = torch.tensor(float("nan"))
        drc_mean = torch.stack(drc_losses).mean() if drc_losses else _nan
        ps_mean  = torch.stack(ps_losses).mean()  if ps_losses  else _nan
        all_losses = drc_losses + ps_losses
        total = torch.stack(all_losses).mean()

        return LossBreakdown(total=total, drc_loss=drc_mean, ps_loss=ps_mean)
