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

Two compound pools feed into ``select()``:
- Unqueried compounds: eligible for either PS or DRC.
- PS-INTERVAL-labeled compounds: already screened; eligible for DRC upgrade
  only.  DRC-upgrade candidates compete on the same score as first-pass DRC
  candidates, so the acquisition naturally escalates promising hits.

The two scores are on the same scale (information per dollar) and can be
compared directly. A compound contributes at most two candidates to the
ranked list; the greedy procedure picks top-k while ensuring each compound
is selected at most once.
"""

from __future__ import annotations

import logging

import numpy as np

from moal.types import QueryType

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _sigmoid(x: np.ndarray, tau: float) -> np.ndarray:
    """Sigmoid function with temperature scaling.

    Computes ``1 / (1 + exp(-x / tau))``.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    tau : float
        Temperature parameter. Smaller values produce sharper transitions.

    Returns
    -------
    np.ndarray
        Sigmoid-transformed values, same shape as ``x``, in the range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x / tau))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats.

    Computes ``H(p) = -p·log(p) - (1-p)·log(1-p)``.  Input values are
    clipped to ``[_EPS, 1 - _EPS]`` after casting to float64 to avoid
    ``log(0)`` even when the input is float32.

    Parameters
    ----------
    p : np.ndarray
        Probability values.  Values outside ``[0, 1]`` are silently clipped.

    Returns
    -------
    np.ndarray
        Binary entropy in nats, same shape as ``p``, non-negative.
    """
    # Cast to float64 so that the clip bounds (1e-9, 1-1e-9) are representable;
    # float32 rounds 1-1e-9 to exactly 1.0, letting log(0) through despite clipping
    p = np.clip(p.astype(np.float64), _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))  # type: ignore[no-any-return]


class CostAwareGreedyAcquisition:
    """Select (compound, fidelity) query pairs that fit within a plate budget.

    Parameters
    ----------
    cost_ps : float
        Cost of a Primary Screen query (dollars).
    cost_drc : float
        Cost of a Dose-Response Curve query (dollars).
    ps_threshold : float, optional
        The pEC50 threshold used by the primary screen. Drives the PS entropy
        score: maximum information at ŷ ≈ ps_threshold. Default is 5.0.
    target_threshold : float, optional
        The pEC50 definition of an "active" compound (e.g., 7.0). Drives the
        DRC exploitation score: maximum value at ŷ >> target_threshold.
        Default is 7.0.
    tau : float, optional
        Sigmoid temperature controlling exploitation sharpness. Smaller τ
        means more sharply exploit the highest-scoring compounds. Default is 0.5.
    embedding_provenance_discount : float, optional
        Multiplicative discount applied to a candidate's score when its
        prediction is flagged as embedding-derived (see ``provenance``
        arguments on :meth:`select` and :meth:`score_summary`) — the
        concatenation architecture's (issue #36 Phase 2) fallback path for
        compounds never PS-screened, which rests on strictly more layers of
        inference than a prediction from an observed input. Must be in
        ``(0.0, 1.0]``. Default is 1.0 (no discount); callers that never pass
        a ``provenance`` array see no behavior change regardless of this
        value.

    Raises
    ------
    ValueError
        If ``cost_ps`` or ``cost_drc`` is not strictly positive, or if
        ``embedding_provenance_discount`` is not in ``(0.0, 1.0]``.
    """

    def __init__(
        self,
        cost_ps: float,
        cost_drc: float,
        ps_threshold: float = 5.0,
        target_threshold: float = 7.0,
        tau: float = 0.5,
        embedding_provenance_discount: float = 1.0,
    ) -> None:
        if cost_ps <= 0 or cost_drc <= 0:
            raise ValueError("Costs must be positive.")
        if not (0.0 < embedding_provenance_discount <= 1.0):
            raise ValueError(
                "embedding_provenance_discount must be in (0.0, 1.0], "
                f"got {embedding_provenance_discount}."
            )
        self.cost_ps = cost_ps
        self.cost_drc = cost_drc
        self.ps_threshold = ps_threshold
        self.target_threshold = target_threshold
        self.tau = tau
        self.embedding_provenance_discount = embedding_provenance_discount

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_drc(self, predictions: np.ndarray) -> np.ndarray:
        """Compute the DRC exploitation score for an array of predictions.

        Score is ``p_active(ŷ) / cost_DRC``, where
        ``p_active = sigmoid((ŷ - target_threshold) / τ)``.

        Parameters
        ----------
        predictions : np.ndarray
            Model pEC50 point estimates, shape ``(N,)``.

        Returns
        -------
        np.ndarray
            DRC acquisition scores, shape ``(N,)``, in the range
            ``(0, 1 / cost_DRC)``.
        """
        p_active = _sigmoid(predictions - self.target_threshold, self.tau)
        return p_active / self.cost_drc

    def _score_ps(self, predictions: np.ndarray) -> np.ndarray:
        """Compute the PS exploration score for an array of predictions.

        Score is ``H_binary(p_cross(ŷ, T)) / cost_PS``, where
        ``p_cross = sigmoid((ŷ - ps_threshold) / τ)`` and ``H_binary`` is
        binary entropy in nats.

        Parameters
        ----------
        predictions : np.ndarray
            Model pEC50 point estimates, shape ``(N,)``.

        Returns
        -------
        np.ndarray
            PS acquisition scores, shape ``(N,)``, in the range
            ``[0, log(2) / cost_PS]``.
        """
        p_cross = _sigmoid(predictions - self.ps_threshold, self.tau)
        h = _binary_entropy(p_cross)
        return h / self.cost_ps

    def _apply_provenance_discount(
        self, scores: np.ndarray, provenance: np.ndarray | None
    ) -> np.ndarray:
        """Apply ``embedding_provenance_discount`` to embedding-derived candidates.

        Parameters
        ----------
        scores : np.ndarray
            Raw acquisition scores, shape ``(N,)``.
        provenance : np.ndarray or None
            Boolean (or 0/1 float) array, shape ``(N,)``, True/1 where the
            prediction is embedding-derived (concatenation architecture,
            never-PS-screened fallback). ``None`` means no provenance
            information was supplied — every candidate is treated as
            observed-input, matching current behavior with no discount.

        Returns
        -------
        np.ndarray
            ``scores`` unchanged where ``provenance`` is False/0 or where
            ``provenance is None``; multiplied by
            ``embedding_provenance_discount`` where ``provenance`` is
            True/1.
        """
        if provenance is None:
            return scores
        provenance = np.asarray(provenance)
        if provenance.shape != scores.shape:
            raise ValueError(
                f"provenance shape {provenance.shape} must match scores shape {scores.shape}."
            )
        discount = np.where(provenance.astype(bool), self.embedding_provenance_discount, 1.0)
        return scores * discount

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        unlabeled_smiles: list[str],
        predictions: np.ndarray,
        plate_size: int,
        wells_per_ps: int,
        wells_per_drc: int,
        ps_labeled_smiles: list[str] | None = None,
        ps_labeled_predictions: np.ndarray | None = None,
        provenance: np.ndarray | None = None,
        ps_labeled_provenance: np.ndarray | None = None,
    ) -> list[tuple[str, QueryType]]:
        """Greedily select queries that fit within a plate well budget.

        Candidates are ranked by acquisition score (highest first) across two
        pools:

        - *Unqueried* compounds (``unlabeled_smiles``): eligible for either PS
          or DRC.  Both candidates enter the unified ranked list.
        - *PS-labeled* compounds (``ps_labeled_smiles``): already screened with
          PS; eligible for a DRC upgrade only.  Only a DRC candidate is
          generated for each.

        The loop walks the ranked list from highest to lowest score.  When the
        next candidate's well cost would push the running total above
        ``plate_size``, the loop stops and returns whatever has been selected so
        far.  No attempt is made to fill the remaining capacity with lower-ranked
        candidates — unused wells are deferred to the next iteration, where all
        candidates will be rescored on the updated labeled pool.

        A candidate whose unit well-cost on its own exceeds ``plate_size`` can
        never be placed and is skipped without halting the fill.  Setting a
        modality's wells-per-query above ``plate_size`` (e.g. ``wells_per_ps =
        plate_size + 1``) therefore disables that modality cleanly while leaving
        the other free to fill the plate.

        Parameters
        ----------
        unlabeled_smiles : list[str]
            Ground-truth keys for all unqueried compounds.
        predictions : np.ndarray
            Model pEC50 point estimates, shape ``(N,)``, aligned with
            ``unlabeled_smiles``.
        plate_size : int
            Maximum total wells available on the plate.  Selection stops as
            soon as adding the next candidate would exceed this limit.
        wells_per_ps : int
            Number of wells consumed by a single PS query.
        wells_per_drc : int
            Number of wells consumed by a single DRC query.
        ps_labeled_smiles : list[str], optional
            Ground-truth keys for compounds that have a PS label but no DRC
            label (i.e., INTERVAL-censored hits eligible for a full
            dose-response follow-up). Defaults to empty.
        ps_labeled_predictions : np.ndarray, optional
            Model pEC50 estimates, shape ``(M,)``, aligned with
            ``ps_labeled_smiles``. Required when ``ps_labeled_smiles`` is
            non-empty.
        provenance : np.ndarray, optional
            Boolean (or 0/1 float) array, shape ``(N,)``, aligned with
            ``unlabeled_smiles``. True/1 marks a prediction as
            embedding-derived (concatenation architecture, issue #36 Phase 2);
            its DRC and PS scores are multiplied by
            ``embedding_provenance_discount``. ``None`` (default) applies no
            discount, matching current behavior.
        ps_labeled_provenance : np.ndarray, optional
            Same semantics as ``provenance``, aligned with
            ``ps_labeled_smiles`` instead.

        Returns
        -------
        list[tuple[str, QueryType]]
            Ordered list of (smiles, QueryType) pairs, highest-scoring first,
            whose cumulative well cost does not exceed ``plate_size``.

        Raises
        ------
        ValueError
            If ``predictions`` contains non-finite values (NaN or inf).
        """
        ps_labeled_smiles = list(ps_labeled_smiles or [])

        if len(unlabeled_smiles) == 0 and len(ps_labeled_smiles) == 0:
            logger.warning("No unlabeled compounds available for acquisition.")
            return []

        if plate_size == 0:
            return []

        predictions = np.asarray(predictions, dtype=np.float32)
        if len(unlabeled_smiles) != len(predictions):
            raise ValueError(
                f"SMILES list length ({len(unlabeled_smiles)}) must match "
                f"predictions length ({len(predictions)})."
            )
        if len(predictions) > 0 and not np.all(np.isfinite(predictions)):
            raise ValueError(
                "predictions must contain only finite values; NaN or inf values "
                "produce undefined sort order and must be filtered before acquisition."
            )

        # Candidates are (score, smiles, QueryType); SMILES is the dedup key so
        # both pools can be merged without index-space collisions
        candidates: list[tuple[float, str, QueryType]] = []

        if unlabeled_smiles:
            scores_drc = self._apply_provenance_discount(self._score_drc(predictions), provenance)
            scores_ps = self._apply_provenance_discount(self._score_ps(predictions), provenance)
            for i, smi in enumerate(unlabeled_smiles):
                candidates.append((float(scores_drc[i]), smi, QueryType.DOSE_RESPONSE))
                candidates.append((float(scores_ps[i]), smi, QueryType.PRIMARY_SCREEN))

        # PS-labeled INTERVAL compounds contribute only a DRC-upgrade candidate
        if ps_labeled_smiles:
            psl_preds = np.asarray(ps_labeled_predictions, dtype=np.float32)
            if len(ps_labeled_smiles) != len(psl_preds):
                raise ValueError(
                    f"ps_labeled_smiles length ({len(ps_labeled_smiles)}) must match "
                    f"ps_labeled_predictions length ({len(psl_preds)})."
                )
            scores_drc_upgrade = self._apply_provenance_discount(
                self._score_drc(psl_preds), ps_labeled_provenance
            )
            for j, smi in enumerate(ps_labeled_smiles):
                candidates.append((float(scores_drc_upgrade[j]), smi, QueryType.DOSE_RESPONSE))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected: list[tuple[str, QueryType]] = []
        selected_smiles: set[str] = set()
        wells_used = 0
        for _score, smi, qt in candidates:
            if smi in selected_smiles:
                continue
            cost = wells_per_drc if qt == QueryType.DOSE_RESPONSE else wells_per_ps
            # A query type whose unit well-cost exceeds the entire plate can never
            # be placed; skip it rather than terminating the fill. This lets a
            # caller disable a modality by setting its wells-per-query above
            # plate_size without starving the other (still-placeable) modality.
            if cost > plate_size:
                continue
            if wells_used + cost > plate_size:
                break
            selected.append((smi, qt))
            selected_smiles.add(smi)
            wells_used += cost

        if wells_used == 0 and (unlabeled_smiles or ps_labeled_smiles):
            logger.warning(
                "No candidates fit within plate_size=%d "
                "(wells_per_ps=%d, wells_per_drc=%d). "
                "No queries selected for this iteration.",
                plate_size,
                wells_per_ps,
                wells_per_drc,
            )
        return selected

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_summary(
        self,
        unlabeled_smiles: list[str],
        predictions: np.ndarray,
        provenance: np.ndarray | None = None,
    ) -> list[dict]:
        """Return per-compound score breakdown for inspection and logging.

        Parameters
        ----------
        unlabeled_smiles : list[str]
            Ground-truth keys for the compounds to score.
        predictions : np.ndarray
            Model pEC50 point estimates, shape ``(N,)``, aligned with
            ``unlabeled_smiles``.
        provenance : np.ndarray, optional
            Boolean (or 0/1 float) array, shape ``(N,)``, aligned with
            ``unlabeled_smiles``. True/1 marks a prediction as
            embedding-derived (concatenation architecture, issue #36 Phase 2);
            ``score_drc``/``score_ps`` are multiplied by
            ``embedding_provenance_discount`` for that row, matching
            :meth:`select`'s ranking. ``None`` (default) applies no discount.

        Returns
        -------
        list[dict]
            One dict per compound with keys ``smiles``, ``y_hat``,
            ``p_active``, ``p_cross_threshold``, ``score_drc``, ``score_ps``,
            ``embedding_derived`` (bool, always present; False when
            ``provenance`` is None).
        """
        predictions = np.asarray(predictions, dtype=np.float32)
        provenance_arr = (
            np.zeros(len(predictions), dtype=bool)
            if provenance is None
            else np.asarray(provenance, dtype=bool)
        )
        if provenance_arr.shape != predictions.shape:
            raise ValueError(
                f"provenance shape {provenance_arr.shape} must match "
                f"predictions shape {predictions.shape}."
            )
        rows = []
        for smi, y_hat, is_embedding in zip(
            unlabeled_smiles, predictions, provenance_arr, strict=False
        ):
            p_active = float(_sigmoid(np.array([y_hat - self.target_threshold]), self.tau)[0])
            p_cross = float(_sigmoid(np.array([y_hat - self.ps_threshold]), self.tau)[0])
            discount = self.embedding_provenance_discount if is_embedding else 1.0
            rows.append(
                {
                    "smiles": smi,
                    "y_hat": float(y_hat),
                    "p_active": p_active,
                    "p_cross_threshold": p_cross,
                    "score_drc": (p_active / self.cost_drc) * discount,
                    "score_ps": (float(_binary_entropy(np.array([p_cross]))[0]) / self.cost_ps)
                    * discount,
                    "embedding_derived": bool(is_embedding),
                }
            )
        return rows
