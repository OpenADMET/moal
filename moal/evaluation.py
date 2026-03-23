"""Pipeline evaluation metrics with scaffold-aware splitting."""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scipy import stats

from moal.model import NoisyOracleModel
from moal.types import CensoringType, LabelRecord, QueryType

logger = logging.getLogger(__name__)


class ModelMetric(StrEnum):
    """Metric for evaluating model predictive performance on a held-out test set."""

    MAE = "mae"
    RMSE = "rmse"
    KENDALL_TAU = "kendall_tau"
    SPEARMAN_R = "spearman_r"
    R2 = "r2"


# ---------------------------------------------------------------------------
# Scaffold utilities
# ---------------------------------------------------------------------------


def _murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES for a compound.

    Falls back to the original SMILES when the scaffold is empty (e.g., for
    acyclic compounds) or when RDKit raises an exception during parsing.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string of the compound.

    Returns
    -------
    str
        Bemis-Murcko scaffold SMILES, or the original ``smiles`` if the
        scaffold cannot be computed.
    """
    try:
        scaffold = MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))
        return scaffold if scaffold else smiles
    except (ValueError, AttributeError) as exc:
        logger.warning("Could not compute Murcko scaffold for %r: %s", smiles, exc)
        return smiles


def scaffold_split(
    smiles_list: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split compound indices into train/test using Bemis-Murcko scaffold groups.

    Scaffold groups are sorted by size (largest first) and assigned to train
    or test such that no scaffold group is split across both sets. This
    prevents test-set compounds from sharing ring systems with train compounds,
    providing a realistic estimate of generalization to novel chemotypes.

    Parameters
    ----------
    smiles_list : list[str]
        List of canonical SMILES.
    test_size : float, optional
        Target fraction of compounds in the test set. Default is 0.2.
    seed : int, optional
        Random seed for deterministic scaffold group assignment when there
        are ties. Default is 42.

    Returns
    -------
    train_indices : list[int]
        Indices into ``smiles_list`` assigned to the training set.
    test_indices : list[int]
        Indices into ``smiles_list`` assigned to the test set.
    """
    rng = np.random.default_rng(seed)

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        sc = _murcko_scaffold(smi)
        scaffold_to_indices[sc].append(i)

    # Sort scaffold groups: largest first, break ties randomly.
    groups = list(scaffold_to_indices.values())
    order = np.argsort([-len(g) for g in groups])
    groups = [groups[i] for i in order]

    n_test_target = int(len(smiles_list) * test_size)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    # Shuffle within each split for reproducibility.
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PipelineEvaluator:
    """Computes efficiency metrics for a cost-aware active learning campaign.

    Parameters
    ----------
    activity_threshold : float, optional
        pEC50 threshold defining an "active" compound. Default is 7.0.
    upper_bound : float, optional
        Practical pEC50 ceiling used to estimate actives from INTERVAL labels
        when exact values are unavailable. Default is 11.0.
    """

    def __init__(
        self,
        activity_threshold: float = 7.0,
        upper_bound: float = 11.0,
    ) -> None:
        self.activity_threshold = activity_threshold
        self.upper_bound = upper_bound

    # ------------------------------------------------------------------
    # Active classification
    # ------------------------------------------------------------------

    def _is_confirmed_active(self, record: LabelRecord) -> bool:
        """Return True if the label definitively confirms activity.

        Parameters
        ----------
        record : LabelRecord
            A single labeled record with censoring type and value.

        Returns
        -------
        bool
            True when the record is confirmed active, False otherwise.

        Notes
        -----
        Classification rules by censoring type:

        - ``EXACT``: active when ``record.value >= activity_threshold``.
        - ``INTERVAL``: active when ``record.value >= activity_threshold``
          (the lower bound is at or above the threshold, so the compound
          is certainly active).
        - ``LEFT``: always False — the compound is below the PS threshold,
          which must be at or below the activity threshold.
        """
        if record.censoring_type == CensoringType.EXACT:
            return record.value >= self.activity_threshold
        if record.censoring_type == CensoringType.INTERVAL:
            return record.value >= self.activity_threshold
        return False  # LEFT: confirmed inactive (below ps_threshold ≤ target)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def actives_per_dollar(
        self, labeled: list[LabelRecord], n_true_actives: int | None = None
    ) -> float:
        """Return the number of confirmed actives found divided by total cost.

        Parameters
        ----------
        labeled : list[LabelRecord]
            All labeled records accumulated so far.
        n_true_actives : int, optional
            Total number of true actives in the pool (unused; retained for
            interface compatibility).

        Returns
        -------
        float
            Actives-per-dollar efficiency metric. Returns 0.0 when no cost has
            been incurred.
        """
        total_cost = sum(r.cost for r in labeled)
        if total_cost == 0:
            return 0.0
        n_actives = sum(1 for r in labeled if self._is_confirmed_active(r))
        return n_actives / total_cost

    def recall_at_budget(
        self,
        labeled: list[LabelRecord],
        n_true_actives: int,
        budget: float,
    ) -> float:
        """Return the fraction of all true actives found within a cost budget.

        Parameters
        ----------
        labeled : list[LabelRecord]
            All labeled records ordered by acquisition time (records[i] was
            acquired before records[j] when i < j).
        n_true_actives : int
            Total number of true actives in the full compound pool.
        budget : float
            Maximum cumulative dollar cost to consider.

        Returns
        -------
        float
            Recall in [0, 1]. Returns 1.0 when ``n_true_actives`` is 0.
        """
        if n_true_actives == 0:
            return 1.0
        cumulative_cost = 0.0
        n_found = 0
        for rec in labeled:
            if cumulative_cost >= budget:
                break
            cumulative_cost += rec.cost
            if self._is_confirmed_active(rec):
                n_found += 1
        return n_found / n_true_actives

    def enrichment_factor(
        self,
        labeled: list[LabelRecord],
        n_total: int,
        n_true_actives: int,
        fraction: float = 0.1,
    ) -> float:
        """Compute the enrichment factor at a given fraction of the compound pool.

        EF@fraction = (actives in top-fraction) / (expected actives by random).

        Parameters
        ----------
        labeled : list[LabelRecord]
            Labeled records in acquisition order.
        n_total : int
            Total number of compounds in the pool (labeled + unlabeled).
        n_true_actives : int
            Ground-truth count of actives in the full pool.
        fraction : float, optional
            Fraction of the pool to consider for the top set. Default is 0.1.

        Returns
        -------
        float
            Enrichment factor. Returns 0.0 when the pool or active count is 0.

        Notes
        -----
        The top-N subset is taken as the first ``int(n_total * fraction)``
        records in ``labeled`` (i.e., those acquired earliest). At least one
        compound is always included regardless of rounding.
        """
        if n_true_actives == 0 or n_total == 0:
            return 0.0
        n_top = max(1, int(n_total * fraction))
        actives_in_top = sum(1 for r in labeled[:n_top] if self._is_confirmed_active(r))
        expected_random = n_true_actives * fraction
        return actives_in_top / expected_random if expected_random > 0 else 0.0

    def cumulative_actives_curve(self, labeled: list[LabelRecord]) -> pd.DataFrame:
        """Build a DataFrame tracking actives and cost cumulatively.

        Parameters
        ----------
        labeled : list[LabelRecord]
            All labeled records in acquisition order.

        Returns
        -------
        pd.DataFrame
            One row per record with columns: ``iteration``, ``n_labeled``,
            ``cumulative_cost``, ``cumulative_actives``, ``actives_per_dollar``.
        """
        rows = []
        cumulative_cost = 0.0
        cumulative_actives = 0
        for i, rec in enumerate(labeled):
            cumulative_cost += rec.cost
            if self._is_confirmed_active(rec):
                cumulative_actives += 1
            apd = cumulative_actives / cumulative_cost if cumulative_cost > 0 else 0.0
            rows.append(
                {
                    "iteration": rec.iteration,
                    "n_labeled": i + 1,
                    "cumulative_cost": cumulative_cost,
                    "cumulative_actives": cumulative_actives,
                    "actives_per_dollar": apd,
                }
            )
        return pd.DataFrame(rows)

    def fidelity_breakdown(self, labeled: list[LabelRecord]) -> dict[str, int]:
        """Count DRC, PS, and PS→DRC upgrades in the labeled pool.

        A PS→DRC upgrade is a compound that has both a PS and a DRC record —
        the DRC was run as a follow-up to a primary screen hit rather than as
        a first-pass query.

        Parameters
        ----------
        labeled : list[LabelRecord]
            All labeled records to analyze.

        Returns
        -------
        dict[str, int]
            Dictionary with keys ``"DRC"``, ``"PS"``, and ``"upgrades"``.
        """
        counts: dict[str, int] = {"DRC": 0, "PS": 0, "upgrades": 0}
        fidelities_by_smiles: dict[str, set] = defaultdict(set)
        for rec in labeled:
            counts[rec.fidelity.value] += 1
            fidelities_by_smiles[rec.canonical_smiles].add(rec.fidelity)
        counts["upgrades"] = sum(
            1
            for fids in fidelities_by_smiles.values()
            if QueryType.PRIMARY_SCREEN in fids and QueryType.DOSE_RESPONSE in fids
        )
        return counts

    # ------------------------------------------------------------------
    # Evaluate checkpoint
    # ------------------------------------------------------------------

    def evaluate(
        self,
        labeled: list[LabelRecord],
        n_total: int,
        n_true_actives: int,
        iteration: int,
    ) -> dict[str, float]:
        """Return a flat dict of scalar metrics for logging.

        Parameters
        ----------
        labeled : list[LabelRecord]
            All labeled records accumulated so far.
        n_total : int
            Total number of compounds in the pool (labeled + unlabeled).
        n_true_actives : int
            Ground-truth count of actives in the full pool.
        iteration : int
            Current iteration index (0-based).

        Returns
        -------
        dict[str, float]
            Scalar metrics including ``iteration``, ``n_labeled``,
            ``n_confirmed_actives``, ``total_cost``, ``actives_per_dollar``,
            ``recall``, ``enrichment_factor_10pct``, ``n_drc_queries``,
            ``n_ps_queries``, and ``n_ps_to_drc_upgrades``.
        """
        total_cost = sum(r.cost for r in labeled)
        n_confirmed = sum(1 for r in labeled if self._is_confirmed_active(r))
        breakdown = self.fidelity_breakdown(labeled)

        metrics: dict[str, float] = {
            "iteration": float(iteration),
            "n_labeled": float(len(labeled)),
            "n_confirmed_actives": float(n_confirmed),
            "total_cost": total_cost,
            "actives_per_dollar": self.actives_per_dollar(labeled),
            "recall": n_confirmed / n_true_actives if n_true_actives > 0 else 0.0,
            "enrichment_factor_10pct": self.enrichment_factor(
                labeled, n_total, n_true_actives, fraction=0.1
            ),
            "n_drc_queries": float(breakdown["DRC"]),
            "n_ps_queries": float(breakdown["PS"]),
            "n_ps_to_drc_upgrades": float(breakdown["upgrades"]),
        }
        return metrics

    # ------------------------------------------------------------------
    # Model predictive performance
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Any,
        test_smiles: list[str],
        test_pec50: np.ndarray,
        metric: ModelMetric,
        noise_scale: float | None = None,
    ) -> float:
        """Evaluate model predictive performance on a held-out test set.

        Parameters
        ----------
        model : ChemPropLightningModule or NoisyOracleModel
            Any object with a compatible ``predict_smiles`` method.
        test_smiles : list[str]
            Canonical SMILES for held-out test compounds.
        test_pec50 : np.ndarray
            True pEC50 values aligned with ``test_smiles``.
        metric : ModelMetric
            Which scalar metric to compute.
        noise_scale : float, optional
            Noise half-width for ``NoisyOracleModel`` (fast mode). Must be
            provided when ``model`` is a ``NoisyOracleModel``; ignored otherwise.

        Returns
        -------
        float
            Scalar metric value. Returns ``nan`` when ``test_smiles`` is empty
            or when fewer than two samples are available for rank-correlation
            metrics.

        Raises
        ------
        ValueError
            If ``model`` is a ``NoisyOracleModel`` and ``noise_scale`` is None.
        ValueError
            If ``metric`` is not a recognised ``ModelMetric`` member.
        """
        if len(test_smiles) == 0:
            return float("nan")

        if isinstance(model, NoisyOracleModel):
            if noise_scale is None:
                raise ValueError(
                    "noise_scale must be provided when evaluating a NoisyOracleModel"
                )
            preds = model.predict_smiles(test_smiles, noise_scale)
        else:
            preds = model.predict_smiles(test_smiles)
        true = np.asarray(test_pec50, dtype=np.float32)

        if metric == ModelMetric.MAE:
            return float(np.mean(np.abs(preds - true)))

        if metric == ModelMetric.RMSE:
            return float(np.sqrt(np.mean((preds - true) ** 2)))

        if metric == ModelMetric.KENDALL_TAU:
            if len(preds) < 2:
                return float("nan")
            result = stats.kendalltau(preds, true)
            return float(result.statistic)

        if metric == ModelMetric.SPEARMAN_R:
            if len(preds) < 2:
                return float("nan")
            result = stats.spearmanr(preds, true)
            return float(result.statistic)

        if metric == ModelMetric.R2:
            ss_res = np.sum((true - preds) ** 2)
            ss_tot = np.sum((true - np.mean(true)) ** 2)
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        raise ValueError(f"Unknown ModelMetric: {metric}")
