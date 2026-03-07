"""Cost-aware greedy acquisition function.

Scoring logic (greedy, no ensemble uncertainty):

    DRC  score: p_active(x) / cost_DRC
    PS   score: H_binary(p_cross(x, T)) / cost_PS

Where:
    p_active(x)   = sigmoid((ŷ_x - target_threshold) / τ)
    p_cross(x, T) = sigmoid((ŷ_x - ps_threshold) / τ)
    H_binary(p)   = -p·log(p) - (1-p)·log(1-p)   [binary entropy, nats]

DRC and PS serve structurally different roles:
- DRC: exploitation — high score for compounds with high predicted activity.
  Rational when ŷ >> target; a DRC confirms and quantifies a likely active.
- PS: efficient exploration of the activity threshold — maximally informative
  when the model is most uncertain whether the compound clears the threshold.
  Cheaply resolves ambiguous cases without consuming DRC budget.

The two scores are on the same scale (information per dollar) and can be
compared directly. A compound contributes two candidates to the ranked list;
the greedy procedure picks top-k while ensuring each compound is selected at
most once.
"""

from __future__ import annotations

import math
import logging

import numpy as np

from moal.types import QueryType

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _sigmoid(x: np.ndarray, tau: float) -> np.ndarray:
    """Sigmoid with temperature τ."""
    return 1.0 / (1.0 + np.exp(-x / tau))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy H(p) in nats."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class CostAwareGreedyAcquisition:
    """Select k (compound, fidelity) query pairs per active-learning iteration.

    Args:
        cost_ps: Cost of a Primary Screen query (dollars).
        cost_drc: Cost of a Dose-Response Curve query (dollars).
        ps_threshold: The pEC50 threshold used by the primary screen.
            Drives the PS entropy score: maximum information at ŷ ≈ ps_threshold.
        target_threshold: The pEC50 definition of an "active" compound (e.g., 7.0).
            Drives the DRC exploitation score: maximum value at ŷ >> target_threshold.
        tau: Sigmoid temperature controlling exploitation sharpness.
            Smaller τ → more sharply exploit the highest-scoring compounds.
    """

    def __init__(
        self,
        cost_ps: float,
        cost_drc: float,
        ps_threshold: float = 5.0,
        target_threshold: float = 7.0,
        tau: float = 0.5,
    ) -> None:
        if cost_ps <= 0 or cost_drc <= 0:
            raise ValueError("Costs must be positive.")
        self.cost_ps = cost_ps
        self.cost_drc = cost_drc
        self.ps_threshold = ps_threshold
        self.target_threshold = target_threshold
        self.tau = tau

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_drc(self, predictions: np.ndarray) -> np.ndarray:
        """Exploitation score for DRC fidelity: p_active / cost_DRC."""
        p_active = _sigmoid(predictions - self.target_threshold, self.tau)
        return p_active / self.cost_drc

    def _score_ps(self, predictions: np.ndarray) -> np.ndarray:
        """Exploration score for PS fidelity: H_binary(p_cross) / cost_PS."""
        p_cross = _sigmoid(predictions - self.ps_threshold, self.tau)
        h = _binary_entropy(p_cross)
        return h / self.cost_ps

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        unlabeled_smiles: list[str],
        predictions: np.ndarray,
        k: int,
    ) -> list[tuple[str, QueryType]]:
        """Greedily select k (compound, fidelity) pairs.

        Args:
            unlabeled_smiles: List of canonical SMILES for all unlabeled compounds.
            predictions: Model pEC50 point estimates, shape (N,), aligned with
                unlabeled_smiles.
            k: Number of queries to select.

        Returns:
            Ordered list of (smiles, QueryType) pairs, highest-scoring first.
        """
        if len(unlabeled_smiles) == 0:
            logger.warning("No unlabeled compounds available for acquisition.")
            return []

        if k == 0:
            return []

        predictions = np.asarray(predictions, dtype=np.float32)
        assert len(unlabeled_smiles) == len(predictions), (
            f"SMILES list length ({len(unlabeled_smiles)}) must match "
            f"predictions length ({len(predictions)})."
        )

        scores_drc = self._score_drc(predictions)
        scores_ps = self._score_ps(predictions)

        # Build a unified candidate list: [(score, idx, QueryType)]
        candidates: list[tuple[float, int, QueryType]] = []
        for i in range(len(unlabeled_smiles)):
            candidates.append((float(scores_drc[i]), i, QueryType.DOSE_RESPONSE))
            candidates.append((float(scores_ps[i]), i, QueryType.PRIMARY_SCREEN))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected: list[tuple[str, QueryType]] = []
        selected_indices: set[int] = set()
        for score, idx, qt in candidates:
            if idx in selected_indices:
                continue
            selected.append((unlabeled_smiles[idx], qt))
            selected_indices.add(idx)
            if len(selected) >= k:
                break

        if len(selected) < k:
            logger.warning(
                "Could only select %d queries (requested %d); "
                "%d unlabeled compounds available.",
                len(selected), k, len(unlabeled_smiles),
            )
        return selected

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_summary(
        self, unlabeled_smiles: list[str], predictions: np.ndarray
    ) -> list[dict]:
        """Return per-compound score breakdown for inspection/logging."""
        predictions = np.asarray(predictions, dtype=np.float32)
        rows = []
        for smi, y_hat in zip(unlabeled_smiles, predictions):
            p_active = float(_sigmoid(np.array([y_hat - self.target_threshold]), self.tau)[0])
            p_cross = float(_sigmoid(np.array([y_hat - self.ps_threshold]), self.tau)[0])
            rows.append({
                "smiles": smi,
                "y_hat": float(y_hat),
                "p_active": p_active,
                "p_cross_threshold": p_cross,
                "score_drc": p_active / self.cost_drc,
                "score_ps": float(_binary_entropy(np.array([p_cross]))[0]) / self.cost_ps,
            })
        return rows
