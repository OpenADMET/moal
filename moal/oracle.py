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

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        DataFrame containing compound SMILES and pEC50 values.
    cost_ps : float
        Cost in dollars for a Primary Screen query.
    cost_drc : float
        Cost in dollars for a Dose-Response Curve query.
    ps_threshold : float
        The pEC50 threshold used by the primary screen (e.g., 5.0). Compounds
        with true pEC50 < ps_threshold receive a LEFT label; others receive an
        INTERVAL label.
    upper_bound : float, optional
        Practical upper ceiling for the pEC50 scale used as the upper boundary
        of INTERVAL labels. Default is 11.0.
    smiles_column : str, optional
        Name of the DataFrame column containing SMILES strings. Default is
        ``"smiles"``.
    pec50_column : str, optional
        Name of the DataFrame column containing pEC50 values. Default is
        ``"pec50"``.
    is_canonical : bool, optional
        When False (default), all SMILES in ``ground_truth_df`` are
        canonicalized up front via RDKit during initialization. When True, the
        input SMILES are assumed to already be in canonical form and the
        preprocessing step is skipped entirely. If ``is_canonical=True`` is
        used at init time, query call-sites should also pass
        ``is_canonical=True`` so that lookup keys remain consistent.
    preprocessor : SMILESPreprocessor, optional
        ``SMILESPreprocessor`` instance (or None to use defaults).
    """

    def __init__(
        self,
        ground_truth_df: pd.DataFrame,
        cost_ps: float,
        cost_drc: float,
        ps_threshold: float,
        upper_bound: float = 11.0,
        smiles_column: str = "smiles",
        pec50_column: str = "pec50",
        is_canonical: bool = False,
        preprocessor: SMILESPreprocessor | None = None,
    ) -> None:
        self.cost_ps = cost_ps
        self.cost_drc = cost_drc
        self.ps_threshold = ps_threshold
        self.upper_bound = upper_bound
        self.smiles_column = smiles_column
        self.pec50_column = pec50_column
        self.is_canonical = is_canonical

        self._preprocessor = preprocessor or SMILESPreprocessor()
        self._total_cost: float = 0.0
        # Each value is a list of records ordered chronologically.  A compound
        # may have at most one PS record followed by at most one DRC record.
        self._labeled: dict[str, list[LabelRecord]] = {}

        self._ground_truth = self._build_ground_truth(ground_truth_df)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_ground_truth(self, df: pd.DataFrame) -> dict[str, float]:
        """Build and validate the internal ground-truth mapping.

        Filters out rows with invalid or out-of-range pEC50 values, optionally
        canonicalizes SMILES, and deduplicates entries (first occurrence wins).
        Warnings are emitted for every rejected row.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame supplied by the caller. Must contain columns named
            ``self.smiles_column`` and ``self.pec50_column``.

        Returns
        -------
        dict[str, float]
            Mapping from (canonical) SMILES key to validated pEC50 value.
            Contains only physically plausible pEC50 values in ``[0.0, 14.0]``.

        Raises
        ------
        ValueError
            If ``df`` does not contain the expected SMILES and pEC50 columns.
        """
        required = {self.smiles_column, self.pec50_column}
        if not required.issubset(df.columns):
            raise ValueError(
                f"ground_truth_df must contain columns {required}, got {set(df.columns)}"
            )

        # Physically plausible pEC50 range: 0 (1 M IC50) to 14 (100 fM IC50).
        # Values outside this range almost always indicate data entry errors,
        # unit mismatches, or curve-fitting artefacts, and must not be passed
        # to the loss function where NaN/inf would corrupt the entire batch.
        _PECO50_MIN: float = 0.0
        _PECO50_MAX: float = 14.0

        # Validate pEC50 values and collect (smiles, pec50) pairs.
        valid_pairs: list[tuple[str, float]] = []
        invalid_pec50: list[tuple[str, object]] = []

        for _, row in df.iterrows():
            raw_smiles = str(row[self.smiles_column])
            raw_value = row[self.pec50_column]

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

            valid_pairs.append((raw_smiles, pec50))

        # Optionally canonicalize all valid SMILES up front, then build
        # the ground truth dict.  Separating validation and SMILES preprocessing
        # keeps them as distinct, auditable stages.
        failed_smiles: list[str] = []
        ground_truth: dict[str, float] = {}

        if self.is_canonical:
            # Trust the caller's assertion that SMILES are already canonical.
            keyed_pairs: list[tuple[str, float]] = valid_pairs
        else:
            keyed_pairs = []
            for raw_smiles, pec50 in valid_pairs:
                canonical = self._preprocessor.canonicalize(raw_smiles)
                if canonical is None:
                    failed_smiles.append(raw_smiles)
                    continue
                keyed_pairs.append((canonical, pec50))

        for key, pec50 in keyed_pairs:
            if key in ground_truth:
                logger.warning(
                    "Duplicate SMILES in ground truth, keeping first: %s", key
                )
                continue
            ground_truth[key] = pec50

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

    def query(
        self,
        smiles: str,
        query_type: QueryType,
        iteration: int,
        is_canonical: bool = False,
    ) -> LabelRecord:
        """Query the oracle for a single compound.

        Parameters
        ----------
        smiles : str
            SMILES string to query.
        query_type : QueryType
            ``PRIMARY_SCREEN`` or ``DOSE_RESPONSE``.
        iteration : int
            Current AL iteration index (0-based).
        is_canonical : bool, optional
            When False (default), the SMILES is canonicalized via RDKit before
            the ground truth lookup. When True, the SMILES is used as the
            lookup key directly — safe when the caller already holds keys
            returned by :meth:`get_unlabeled_smiles`. Must be consistent with
            the ``is_canonical`` setting used at oracle init, otherwise key
            mismatches will produce ``KeyError``.

        Returns
        -------
        LabelRecord
            A label record with the appropriate label and cost.

        Raises
        ------
        ValueError
            If ``is_canonical=False`` and the SMILES cannot be parsed, or if
            the compound has already been labeled.
        KeyError
            If the compound is not in the ground truth dataset.
        """
        if is_canonical:
            key = smiles
        else:
            key = self._preprocessor.canonicalize(smiles)
            if key is None:
                raise ValueError(f"Cannot parse SMILES: {smiles!r}")

        if key in self._labeled:
            existing = self._labeled[key]
            has_drc = any(r.fidelity == QueryType.DOSE_RESPONSE for r in existing)
            if has_drc:
                raise ValueError(
                    f"Compound already has a DRC label (key: {key!r}). "
                    "DRC is the highest-fidelity assay and cannot be repeated or downgraded."
                )
            if query_type == QueryType.PRIMARY_SCREEN:
                raise ValueError(
                    f"Compound already has a PS label (key: {key!r}). "
                    "Each compound may only receive one PS query."
                )
            # query_type is DRC and compound has only a PS record: upgrade is allowed
        if key not in self._ground_truth:
            raise KeyError(
                f"Compound not found in ground truth dataset (key: {key!r})."
            )

        true_pec50 = self._ground_truth[key]

        if query_type == QueryType.PRIMARY_SCREEN:
            record = self._make_ps_record(smiles, key, true_pec50, iteration)
        else:
            record = self._make_drc_record(smiles, key, true_pec50, iteration)

        self._labeled.setdefault(key, []).append(record)
        self._total_cost += record.cost
        return record

    def query_batch(
        self,
        queries: list[tuple[str, QueryType]],
        iteration: int,
        is_canonical: bool = False,
    ) -> list[LabelRecord]:
        """Query the oracle for a batch of (smiles, query_type) pairs.

        Duplicates within the batch are detected before any queries are
        dispatched so that cost is never double-counted.

        Parameters
        ----------
        queries : list[tuple[str, QueryType]]
            List of (smiles, query_type) pairs.
        iteration : int
            Current AL iteration index (0-based).
        is_canonical : bool, optional
            Forwarded to each :meth:`query` call. When True, SMILES are used
            as lookup keys directly without canonicalization. See :meth:`query`
            for key-consistency requirements.

        Returns
        -------
        list[LabelRecord]
            Successfully obtained label records (invalid or duplicate queries
            are skipped with a warning).
        """
        seen: set[str] = set()
        unique_queries: list[tuple[str, QueryType]] = []
        for smiles, qt in queries:
            if is_canonical:
                key = smiles
            else:
                key = self._preprocessor.canonicalize(smiles)
                if key is None:
                    logger.warning("Skipping invalid SMILES in batch: %s", smiles)
                    continue
            if key in seen:
                logger.warning("Duplicate within acquisition batch, skipping: %s", key)
                continue
            seen.add(key)
            unique_queries.append((smiles, qt))

        records: list[LabelRecord] = []
        for smiles, qt in unique_queries:
            try:
                records.append(
                    self.query(smiles, qt, iteration, is_canonical=is_canonical)
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping query (%s, %s): %s", smiles, qt, exc)
        return records

    # ------------------------------------------------------------------
    # Label construction helpers
    # ------------------------------------------------------------------

    def _make_ps_record(
        self, smiles: str, canonical: str, true_pec50: float, iteration: int
    ) -> LabelRecord:
        """Construct the appropriate PS ``LabelRecord`` for a compound.

        Compounds with ``true_pec50 < ps_threshold`` receive a LEFT-censored
        label at the threshold (confirmed inactive).  All other compounds
        receive an INTERVAL-censored label ``[ps_threshold, upper_bound]``
        (potentially active but exact value unknown from PS alone).

        Parameters
        ----------
        smiles : str
            Original SMILES string as supplied by the caller (stored verbatim
            in the record for downstream traceability).
        canonical : str
            Canonical SMILES key used for ground-truth lookup and deduplication.
        true_pec50 : float
            Ground-truth pEC50 value, already validated to be finite and within
            ``[0.0, 14.0]``.
        iteration : int
            Active learning iteration index (0-based) at which the query was
            issued.

        Returns
        -------
        LabelRecord
            A ``LEFT``-censored record when ``true_pec50 < ps_threshold``,
            or an ``INTERVAL``-censored record ``[ps_threshold, upper_bound]``
            otherwise.
        """
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
        """Construct an EXACT ``LabelRecord`` for a DRC query.

        DRC assays yield a precise pEC50 estimate, so the record is
        uncensored (``CensoringType.EXACT``) with ``value == upper_bound ==
        true_pec50``.

        Parameters
        ----------
        smiles : str
            Original SMILES string as supplied by the caller (stored verbatim
            in the record for downstream traceability).
        canonical : str
            Canonical SMILES key used for ground-truth lookup and deduplication.
        true_pec50 : float
            Ground-truth pEC50 value, already validated to be finite and within
            ``[0.0, 14.0]``.
        iteration : int
            Active learning iteration index (0-based) at which the query was
            issued.

        Returns
        -------
        LabelRecord
            An ``EXACT``-censored record whose ``value`` and ``upper_bound``
            are both set to ``true_pec50``.
        """
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
        """Cumulative cost of all queries issued so far, in dollars.

        Incremented atomically inside :meth:`query` after each successful
        label dispensation.  Includes both PS and DRC costs.

        Returns
        -------
        float
            Running sum of ``cost`` fields across all :class:`LabelRecord`
            instances stored in the oracle.
        """
        return self._total_cost

    @property
    def labeled_records(self) -> list[LabelRecord]:
        """All labeled records in chronological acquisition order.

        Within each iteration PS records precede DRC records, reflecting the
        order in which assays were physically run.  Chronological ordering is
        required for ``cumulative_actives_curve``, ``recall_at_budget``, and
        ``enrichment_factor`` to produce correct values when PS→DRC upgrades
        are present — without sorting, upgrade records would appear at the
        wrong position in the running totals.

        Returns
        -------
        list[LabelRecord]
            All records across every labeled compound, sorted by
            ``(iteration, fidelity_rank)`` where PS rank is 0 and DRC rank
            is 1.
        """
        flat = [r for records in self._labeled.values() for r in records]
        # Sort key: (iteration, fidelity rank) where PS=0 < DRC=1 so that
        # within the same iteration the PS record precedes the DRC upgrade.
        return sorted(
            flat,
            key=lambda r: (
                r.iteration,
                0 if r.fidelity == QueryType.PRIMARY_SCREEN else 1,
            ),
        )

    @property
    def training_records(self) -> list[LabelRecord]:
        """Labeled records suitable for model training.

        Identical to :attr:`labeled_records` except that PS INTERVAL records
        are excluded for any compound that also has a DRC record.  Keeping the
        redundant INTERVAL label alongside its EXACT counterpart adds no
        gradient information and inflates the compound's effective loss weight
        by ``w_ps`` (typically 0.3×), which biases gradient updates toward
        the most-upgraded scaffold clusters over many iterations.

        Returns
        -------
        list[LabelRecord]
            Chronologically ordered records with PS INTERVAL records removed
            for any compound that has been upgraded to DRC.

        Notes
        -----
        Always pass ``oracle.training_records`` (not ``oracle.labeled_records``)
        to :meth:`~moal.model.ChemPropLightningModule.refit`.
        """
        upgraded_smiles: set[str] = {
            key
            for key, records in self._labeled.items()
            if any(r.fidelity == QueryType.DOSE_RESPONSE for r in records)
        }
        return [
            r
            for r in self.labeled_records
            if not (
                r.fidelity == QueryType.PRIMARY_SCREEN
                and r.canonical_smiles in upgraded_smiles
            )
        ]

    def get_unlabeled_smiles(self) -> list[str]:
        """Return ground-truth keys for all compounds not yet queried with any assay.

        Keys are canonical SMILES when the oracle was initialised with
        ``is_canonical=False`` (the default), or the raw CSV SMILES when
        ``is_canonical=True``.  Callers that forward these keys to
        :meth:`query_batch` must pass the matching ``is_canonical`` flag.

        Returns
        -------
        list[str]
            SMILES keys present in the ground-truth pool that have not yet
            received any PS or DRC label.
        """
        return [s for s in self._ground_truth if s not in self._labeled]

    def get_ps_labeled_smiles(self) -> list[str]:
        """Return ground-truth keys for compounds that have a PS label but no DRC label.

        Only INTERVAL-censored PS records are returned.  LEFT-labeled compounds
        are confirmed inactive (pEC50 < ps_threshold) and are not useful DRC
        upgrade candidates.  Callers that forward these keys to
        :meth:`query_batch` must pass the matching ``is_canonical`` flag.

        Returns
        -------
        list[str]
            SMILES keys that hold exactly one PS INTERVAL record and no DRC
            record — i.e., compounds eligible for a DRC upgrade query.
        """
        result = []
        for key, records in self._labeled.items():
            if any(r.fidelity == QueryType.DOSE_RESPONSE for r in records):
                continue
            # Only promote INTERVAL PS labels; LEFT means confirmed inactive
            if any(
                r.fidelity == QueryType.PRIMARY_SCREEN
                and r.censoring_type == CensoringType.INTERVAL
                for r in records
            ):
                result.append(key)
        return result

    def is_active(self, smiles: str, threshold: float = 7.0) -> bool:
        """Return ``True`` if the compound's true pEC50 meets the activity threshold.

        Parameters
        ----------
        smiles : str
            SMILES string of the compound to look up. Canonicalized internally
            before the ground-truth lookup.
        threshold : float, optional
            Minimum pEC50 required to be considered active (inclusive).
            Default is ``7.0`` (IC50 ≤ 100 nM).

        Returns
        -------
        bool
            ``True`` when ``ground_truth[canonical] >= threshold``, ``False``
            otherwise.

        Raises
        ------
        KeyError
            If the SMILES cannot be parsed or is not present in the
            ground-truth pool.
        """
        canonical = self._preprocessor.canonicalize(smiles)
        if canonical is None or canonical not in self._ground_truth:
            raise KeyError(f"Compound not found or invalid SMILES: {smiles!r}")
        return self._ground_truth[canonical] >= threshold

    def n_true_actives(self, threshold: float = 7.0) -> int:
        """Return the total number of active compounds in the ground-truth pool.

        Parameters
        ----------
        threshold : float, optional
            Minimum pEC50 required to be considered active (inclusive).
            Default is ``7.0`` (IC50 ≤ 100 nM).

        Returns
        -------
        int
            Count of compounds in the ground-truth pool whose pEC50 is
            greater than or equal to ``threshold``.
        """
        return sum(1 for v in self._ground_truth.values() if v >= threshold)

    def __len__(self) -> int:
        """Return the total number of compounds in the ground-truth pool.

        Returns
        -------
        int
            Number of entries in the internal ground-truth mapping, after
            filtering invalid pEC50 values and deduplicating SMILES.
        """
        return len(self._ground_truth)
