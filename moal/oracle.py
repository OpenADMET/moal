"""CostAwareOracle: wraps the ground-truth dataset and dispenses labels at cost."""

from __future__ import annotations

import logging
import math

import pandas as pd

from moal.preprocessing import SMILESPreprocessor
from moal.types import CensoringType, LabelRecord, QueryType

logger = logging.getLogger(__name__)


class CostAwareOracle:
    """Simulates a wet-lab oracle over a fixed ground-truth dataset.

    This class is the **sole** interface between the active learning pipeline
    and true pEC50 values. It enforces:
    - Deduplication: each compound may only be queried once.
    - Cost tracking: every query increments the total cost.
    - Label correctness: PS queries return inequality labels; DRC queries return
      exact values.

    Args:
        ground_truth_df: DataFrame with columns ``smiles`` and ``pec50``.
        cost_ps: Cost in dollars for a Primary Screen query.
        cost_drc: Cost in dollars for a Dose-Response Curve query.
        ps_threshold: The pEC50 threshold used by the primary screen (e.g., 5.0).
            Compounds with true pEC50 < ps_threshold receive a LEFT label;
            others receive an INTERVAL label.
        upper_bound: Practical upper ceiling for the pEC50 scale (default 11.0),
            used as the upper boundary of INTERVAL labels.
        preprocessor: SMILESPreprocessor instance (or None to use defaults).
    """

    def __init__(
        self,
        ground_truth_df: pd.DataFrame,
        cost_ps: float,
        cost_drc: float,
        ps_threshold: float,
        upper_bound: float = 11.0,
        preprocessor: SMILESPreprocessor | None = None,
    ) -> None:
        self.cost_ps = cost_ps
        self.cost_drc = cost_drc
        self.ps_threshold = ps_threshold
        self.upper_bound = upper_bound

        self._preprocessor = preprocessor or SMILESPreprocessor()
        self._total_cost: float = 0.0
        self._labeled: dict[str, LabelRecord] = {}

        self._ground_truth = self._build_ground_truth(ground_truth_df)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_ground_truth(self, df: pd.DataFrame) -> dict[str, float]:
        required = {"smiles", "pec50"}
        if not required.issubset(df.columns):
            raise ValueError(f"ground_truth_df must contain columns {required}, got {set(df.columns)}")

        # Physically plausible pEC50 range: 0 (1 M IC50) to 14 (100 fM IC50).
        # Values outside this range almost always indicate data entry errors,
        # unit mismatches, or curve-fitting artefacts, and must not be passed
        # to the loss function where NaN/inf would corrupt the entire batch.
        _PECO50_MIN: float = 0.0
        _PECO50_MAX: float = 14.0

        ground_truth: dict[str, float] = {}
        failed_smiles: list[str] = []
        invalid_pec50: list[tuple[str, object]] = []

        for _, row in df.iterrows():
            raw_smiles = str(row["smiles"])
            raw_value = row["pec50"]

            # Validate pEC50 before storing.
            try:
                pec50 = float(raw_value)
            except (ValueError, TypeError):
                invalid_pec50.append((raw_smiles, raw_value))
                continue
            if not math.isfinite(pec50):
                invalid_pec50.append((raw_smiles, raw_value))
                continue
            if not (_PECO50_MIN <= pec50 <= _PECO50_MAX):
                invalid_pec50.append((raw_smiles, pec50))
                continue

            canonical = self._preprocessor.canonicalize(raw_smiles)
            if canonical is None:
                failed_smiles.append(raw_smiles)
                continue
            if canonical in ground_truth:
                logger.warning("Duplicate canonical SMILES in ground truth, keeping first: %s", canonical)
                continue
            ground_truth[canonical] = pec50

        if failed_smiles:
            logger.warning(
                "%d / %d SMILES failed preprocessing and were excluded from the oracle.",
                len(failed_smiles),
                len(df),
            )
        if invalid_pec50:
            logger.warning(
                "%d / %d compounds excluded due to invalid pEC50 values "
                "(NaN, inf, or outside [%.1f, %.1f]): first few: %s",
                len(invalid_pec50),
                len(df),
                _PECO50_MIN,
                _PECO50_MAX,
                [(s, v) for s, v in invalid_pec50[:3]],
            )
        logger.info("Oracle initialized with %d compounds.", len(ground_truth))
        return ground_truth

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, smiles: str, query_type: QueryType, iteration: int) -> LabelRecord:
        """Query the oracle for a single compound.

        Args:
            smiles: SMILES string (will be canonicalized internally).
            query_type: PRIMARY_SCREEN or DOSE_RESPONSE.
            iteration: Current AL iteration index (0-based).

        Returns:
            A LabelRecord with the appropriate label and cost.

        Raises:
            ValueError: If the compound has already been labeled.
            KeyError: If the compound is not in the ground truth dataset.
        """
        canonical = self._preprocessor.canonicalize(smiles)
        if canonical is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")
        if canonical in self._labeled:
            raise ValueError(
                f"Compound already labeled (canonical: {canonical!r}). "
                "Each compound may only be queried once."
            )
        if canonical not in self._ground_truth:
            raise KeyError(
                f"Compound not found in ground truth dataset (canonical: {canonical!r})."
            )

        true_pec50 = self._ground_truth[canonical]

        if query_type == QueryType.PRIMARY_SCREEN:
            record = self._make_ps_record(smiles, canonical, true_pec50, iteration)
        else:
            record = self._make_drc_record(smiles, canonical, true_pec50, iteration)

        self._labeled[canonical] = record
        self._total_cost += record.cost
        return record

    def query_batch(
        self, queries: list[tuple[str, QueryType]], iteration: int
    ) -> list[LabelRecord]:
        """Query the oracle for a batch of (smiles, query_type) pairs.

        Duplicates within the batch are detected before any queries are
        dispatched so that cost is never double-counted.
        """
        seen: set[str] = set()
        unique_queries: list[tuple[str, QueryType]] = []
        for smiles, qt in queries:
            canonical = self._preprocessor.canonicalize(smiles)
            if canonical is None:
                logger.warning("Skipping invalid SMILES in batch: %s", smiles)
                continue
            if canonical in seen:
                logger.warning("Duplicate within acquisition batch, skipping: %s", canonical)
                continue
            seen.add(canonical)
            unique_queries.append((smiles, qt))

        records: list[LabelRecord] = []
        for smiles, qt in unique_queries:
            try:
                records.append(self.query(smiles, qt, iteration))
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping query (%s, %s): %s", smiles, qt, exc)
        return records

    # ------------------------------------------------------------------
    # Label construction helpers
    # ------------------------------------------------------------------

    def _make_ps_record(
        self, smiles: str, canonical: str, true_pec50: float, iteration: int
    ) -> LabelRecord:
        if true_pec50 < self.ps_threshold:
            return LabelRecord(
                smiles=smiles,
                canonical_smiles=canonical,
                value=self.ps_threshold,
                upper_bound=self.ps_threshold,
                censoring_type=CensoringType.LEFT,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=self.cost_ps,
                iteration=iteration,
            )
        else:
            return LabelRecord(
                smiles=smiles,
                canonical_smiles=canonical,
                value=self.ps_threshold,
                upper_bound=self.upper_bound,
                censoring_type=CensoringType.INTERVAL,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=self.cost_ps,
                iteration=iteration,
            )

    def _make_drc_record(
        self, smiles: str, canonical: str, true_pec50: float, iteration: int
    ) -> LabelRecord:
        return LabelRecord(
            smiles=smiles,
            canonical_smiles=canonical,
            value=true_pec50,
            upper_bound=true_pec50,
            censoring_type=CensoringType.EXACT,
            fidelity=QueryType.DOSE_RESPONSE,
            cost=self.cost_drc,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def labeled_records(self) -> list[LabelRecord]:
        return list(self._labeled.values())

    def get_unlabeled_smiles(self) -> list[str]:
        """Return canonical SMILES for all compounds not yet labeled."""
        return [s for s in self._ground_truth if s not in self._labeled]

    def is_active(self, smiles: str, threshold: float = 7.0) -> bool:
        """Return True if the compound's true pEC50 meets the activity threshold."""
        canonical = self._preprocessor.canonicalize(smiles)
        if canonical is None or canonical not in self._ground_truth:
            raise KeyError(f"Compound not found or invalid SMILES: {smiles!r}")
        return self._ground_truth[canonical] >= threshold

    def n_true_actives(self, threshold: float = 7.0) -> int:
        """Total number of active compounds in the ground truth pool."""
        return sum(1 for v in self._ground_truth.values() if v >= threshold)

    def __len__(self) -> int:
        return len(self._ground_truth)
