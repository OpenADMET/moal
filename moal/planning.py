"""Helpers for one-shot acquisition planning from a unified campaign state CSV."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from moal.acquisition import CostAwareGreedyAcquisition
from moal.preprocessing import SMILESPreprocessor
from moal.types import CensoringType, LabelRecord, QueryType

logger = logging.getLogger(__name__)

_PECO50_MIN = 0.0
_PECO50_MAX = 14.0
_PLAN_MODE_ITERATION = 0
"""Iteration index assigned to externally supplied plan-mode training records."""


@dataclass
class CampaignState:
    """Parsed state of an active learning campaign from a unified state CSV.

    Attributes
    ----------
    training_records : list[LabelRecord]
        All labeled rows (``<``, ``>=``, ``==``), ready for model training.
    unqueried_rows : list[tuple[int, str]]
        ``(df_row_index, canonical_smiles)`` for rows with no label — eligible
        for either PS or DRC.
    ps_upgrade_rows : list[tuple[int, str]]
        ``(df_row_index, canonical_smiles)`` for PS-INTERVAL-labeled hits —
        eligible for DRC upgrade only.
    """

    training_records: list[LabelRecord]
    unqueried_rows: list[tuple[int, str]]
    ps_upgrade_rows: list[tuple[int, str]]


def parse_campaign_state(
    df: pd.DataFrame,
    *,
    cost_ps: float,
    cost_drc: float,
    upper_bound: float,
    preprocessor: SMILESPreprocessor,
    smiles_column: str = "smiles",
    relation_column: str = "relation",
    value_column: str = "value",
    weight_column: str | None = None,
    log2fc_columns: list[str] | None = None,
    is_canonical: bool = False,
    expected_ps_threshold: float | None = None,
) -> CampaignState:
    """Parse a unified campaign state CSV into training and inference partitions.

    Each row falls into one of four states based on its ``relation`` and
    ``value`` columns:

    - Both empty → unqueried inference target (eligible for PS or DRC)
    - ``<`` → inactive PS miss; training only
    - ``>=`` → PS hit; training **and** DRC-upgrade inference target
    - ``==`` → DRC / exact label; training only (terminal)

    Rows with exactly one of relation / value populated are rejected as
    inconsistent.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from the unified campaign state CSV.
    cost_ps, cost_drc : float
        Assay costs forwarded to each ``LabelRecord``.
    upper_bound : float
        pEC50 ceiling for INTERVAL labels.
    preprocessor : SMILESPreprocessor
        Used to canonicalize SMILES when ``is_canonical`` is False.
    smiles_column, relation_column, value_column : str
        Column names in ``df``.
    weight_column : str or None
        Optional column name for per-sample loss weights. When provided,
        each labeled row's weight is read from this column (NaN / empty cells
        default to 1.0). Must be a finite positive float when present. When
        None (default), all records receive ``weight=1.0``.
    log2fc_columns : list[str] or None
        Optional column names for the compound's observed continuous
        auxiliary readouts (e.g. log2FC at one or more primary-screen
        concentrations, a direct pIC50). When provided, each column's value
        is read into ``LabelRecord.raw_ps_readouts`` keyed by column name
        (NaN / empty cells are omitted rather than stored). When None
        (default), ``raw_ps_readouts`` is empty for every record.
    is_canonical : bool
        When True, skip RDKit canonicalization.
    expected_ps_threshold : float or None
        When set, every PS row's value must match this threshold exactly.

    Returns
    -------
    CampaignState
        Parsed partitions ready for training and inference.
    """
    if smiles_column not in df.columns:
        raise ValueError(
            f"state CSV must contain column {smiles_column!r}, got {sorted(df.columns)}"
        )
    if weight_column is not None and weight_column not in df.columns:
        raise ValueError(
            f"weight_column {weight_column!r} not found in state CSV, got {sorted(df.columns)}"
        )
    if log2fc_columns is not None:
        missing = [col for col in log2fc_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"log2fc_columns {missing!r} not found in state CSV, got {sorted(df.columns)}"
            )

    training_records: list[LabelRecord] = []
    unqueried_rows: list[tuple[int, str]] = []
    ps_upgrade_rows: list[tuple[int, str]] = []
    seen_unqueried: set[str] = set()
    n_unqueried_duplicates = 0

    for row_idx, row in df.iterrows():
        csv_row = cast(int, row_idx) + 2  # account for zero indexing + header row
        raw_smiles = str(row[smiles_column])
        relation_raw = row.get(relation_column, None)
        value_raw = row.get(value_column, None)

        relation_empty = pd.isna(relation_raw) or str(relation_raw).strip() == ""
        value_empty = pd.isna(value_raw) or str(value_raw).strip() == ""

        # Partial population is an inconsistent state
        if relation_empty != value_empty:
            raise ValueError(
                f"Row {csv_row}: relation and value must both be populated or both be empty."
            )

        canonical = raw_smiles if is_canonical else preprocessor.canonicalize(raw_smiles)
        if canonical is None:
            raise ValueError(f"Row {csv_row}: invalid SMILES {raw_smiles!r}.")

        if relation_empty:
            # Unqueried compound — deduplicate and warn rather than raise, mirroring
            # the original candidate pool parsing behavior
            if canonical in seen_unqueried:
                n_unqueried_duplicates += 1
                continue
            seen_unqueried.add(canonical)
            unqueried_rows.append((cast(int, row_idx), canonical))
            continue

        relation = str(relation_raw).strip()
        if relation not in {"<", ">=", "=="}:
            raise ValueError(
                f"Row {csv_row}: relation must be one of '<', '>=', or '==', got {relation!r}."
            )

        try:
            value = float(value_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {csv_row}: value must be a finite numeric pEC50 datum.") from exc

        if not math.isfinite(value):
            raise ValueError(f"Row {csv_row}: value must be finite, got {value_raw!r}.")
        if not (_PECO50_MIN <= value <= _PECO50_MAX):
            raise ValueError(
                f"Row {csv_row}: value must be within [{_PECO50_MIN:.1f}, {_PECO50_MAX:.1f}], "
                f"got {value}."
            )

        weight = 1.0
        if weight_column is not None:
            weight_raw = row.get(weight_column, None)
            if not (pd.isna(weight_raw) or str(weight_raw).strip() == ""):
                try:
                    weight = float(weight_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {csv_row}: weight must be a finite positive float,"
                        f" got {weight_raw!r}."
                    ) from exc
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        f"Row {csv_row}: weight must be finite and positive, got {weight}."
                    )

        raw_ps_readouts: dict[str, float] = {}
        for col in log2fc_columns or ():
            readout_raw = row.get(col, None)
            if pd.isna(readout_raw) or str(readout_raw).strip() == "":
                continue
            try:
                readout = float(readout_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row {csv_row}: column {col!r} must be a finite numeric readout,"
                    f" got {readout_raw!r}."
                ) from exc
            if not math.isfinite(readout):
                raise ValueError(
                    f"Row {csv_row}: column {col!r} must be finite, got {readout_raw!r}."
                )
            raw_ps_readouts[col] = readout

        if relation == "==":
            record = LabelRecord(
                smiles=raw_smiles,
                canonical_smiles=canonical,
                value=value,
                upper_bound=value,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=cost_drc,
                iteration=_PLAN_MODE_ITERATION,
                weight=weight,
                raw_ps_readouts=raw_ps_readouts,
            )
        else:
            if expected_ps_threshold is not None and not math.isclose(
                value, expected_ps_threshold, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"Row {csv_row}: PS threshold {value} does not match "
                    f"config oracle.ps_threshold={expected_ps_threshold}."
                )
            record = LabelRecord(
                smiles=raw_smiles,
                canonical_smiles=canonical,
                value=value,
                upper_bound=value if relation == "<" else upper_bound,
                censoring_type=(CensoringType.LEFT if relation == "<" else CensoringType.INTERVAL),
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=cost_ps,
                iteration=_PLAN_MODE_ITERATION,
                weight=weight,
                raw_ps_readouts=raw_ps_readouts,
            )
            # PS hits are DRC-upgrade inference targets in addition to training records
            if relation == ">=":
                ps_upgrade_rows.append((cast(int, row_idx), canonical))

        training_records.append(record)

    if n_unqueried_duplicates:
        logger.warning(
            "Skipped %d duplicate unqueried SMILES after canonicalization.",
            n_unqueried_duplicates,
        )

    # Reject cross-partition duplicates — a compound can't be both labeled and unqueried
    training_canonical = {rec.canonical_smiles for rec in training_records}
    cross_partition = [smi for _, smi in unqueried_rows if smi in training_canonical]
    if cross_partition:
        example = ", ".join(cross_partition[:3])
        raise ValueError(
            "State CSV contains compounds that appear as both labeled and unqueried; "
            f"first few: {example}"
        )

    validate_training_records(training_records)
    return CampaignState(
        training_records=training_records,
        unqueried_rows=unqueried_rows,
        ps_upgrade_rows=ps_upgrade_rows,
    )


def parse_pretrain_records(
    df: pd.DataFrame,
    *,
    cost_ps: float,
    cost_drc: float,
    upper_bound: float,
    preprocessor: SMILESPreprocessor,
    smiles_column: str = "smiles",
    relation_column: str = "relation",
    value_column: str = "value",
    weight_column: str | None = None,
    log2fc_columns: list[str] | None = None,
    is_canonical: bool = False,
    expected_ps_threshold: float | None = None,
) -> list[LabelRecord]:
    """Parse a pretrain CSV and return its labeled training records.

    Thin wrapper around :func:`parse_campaign_state` for use with
    ``moal simulate``.  The pretrain CSV uses the same mixed-fidelity
    format as the ``moal plan`` campaign state CSV:

    - ``<`` / ``>=`` / ``==`` rows → :class:`~moal.types.LabelRecord` objects
      returned as training records.
    - Rows with empty relation/value fields → unqueried entries; these provide
      no training signal and are skipped with a warning.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from the pretrain CSV.
    cost_ps, cost_drc : float
        Assay costs forwarded to each ``LabelRecord``.
    upper_bound : float
        pEC50 ceiling for INTERVAL labels.
    preprocessor : SMILESPreprocessor
        Used to canonicalize SMILES when ``is_canonical`` is False.
    smiles_column, relation_column, value_column : str
        Column names in ``df``.
    weight_column : str or None
        Optional column name for per-sample loss weights. Forwarded to
        :func:`parse_campaign_state`. When None (default), all records
        receive ``weight=1.0``.
    log2fc_columns : list[str] or None
        Optional column names for observed auxiliary readouts. Forwarded to
        :func:`parse_campaign_state`. When None (default),
        ``raw_ps_readouts`` is empty for every record.
    is_canonical : bool
        When True, skip RDKit canonicalization.
    expected_ps_threshold : float or None
        When set, every PS row's value must match this threshold exactly.
        Should be wired to ``oracle.ps_threshold`` from the campaign config.

    Returns
    -------
    list[LabelRecord]
        Labeled training records suitable for combining with oracle-acquired
        records before calling ``model.refit()``.
    """
    state = parse_campaign_state(
        df,
        cost_ps=cost_ps,
        cost_drc=cost_drc,
        upper_bound=upper_bound,
        preprocessor=preprocessor,
        smiles_column=smiles_column,
        relation_column=relation_column,
        value_column=value_column,
        weight_column=weight_column,
        log2fc_columns=log2fc_columns,
        is_canonical=is_canonical,
        expected_ps_threshold=expected_ps_threshold,
    )
    if state.unqueried_rows:
        logger.warning(
            "Pretrain CSV contains %d unqueried row(s) (empty relation/value); "
            "these provide no training signal and will be ignored.",
            len(state.unqueried_rows),
        )
    return state.training_records


def training_records_for_refit(records: list[LabelRecord]) -> list[LabelRecord]:
    """Return model-training records with upgraded PS hits de-duplicated.

    When a compound has both a PS INTERVAL record (``>=`` hit) and a DRC
    EXACT record, the PS record is excluded to prevent double-weighting
    during model training.  PS LEFT records (``<`` misses) are always
    retained regardless of DRC coverage.  Any ``raw_ps_readouts`` entries on
    the excluded PS record are merged onto the surviving DRC record (e.g. a
    DRC upgrade acquired directly through the oracle, with no readouts of its
    own) so they are not lost for the auxiliary encoder; a key already present
    on the DRC record's own readouts takes precedence.

    Parameters
    ----------
    records : list[LabelRecord]
        All labeled records from the campaign state, including both PS and
        DRC entries.

    Returns
    -------
    list[LabelRecord]
        Filtered list suitable for passing to ``model.refit()``.  At most
        one record per compound for upgraded compounds; both record types
        are retained for compounds that have only a PS miss.
    """
    upgraded_smiles = {
        rec.canonical_smiles for rec in records if rec.fidelity == QueryType.DOSE_RESPONSE
    }
    upgrade_readouts = {
        rec.canonical_smiles: rec.raw_ps_readouts
        for rec in records
        if rec.fidelity == QueryType.PRIMARY_SCREEN
        and rec.censoring_type == CensoringType.INTERVAL
        and rec.canonical_smiles in upgraded_smiles
        and rec.raw_ps_readouts
    }

    result = []
    for rec in records:
        if (
            rec.fidelity == QueryType.PRIMARY_SCREEN
            and rec.censoring_type == CensoringType.INTERVAL
            and rec.canonical_smiles in upgraded_smiles
        ):
            continue
        if rec.fidelity == QueryType.DOSE_RESPONSE and rec.canonical_smiles in upgrade_readouts:
            merged = {**upgrade_readouts[rec.canonical_smiles], **rec.raw_ps_readouts}
            if merged != rec.raw_ps_readouts:
                rec = replace(rec, raw_ps_readouts=merged)
        result.append(rec)
    return result


def annotate_campaign_state(
    df: pd.DataFrame,
    state: CampaignState,
    predictions: np.ndarray,
    acquisition: CostAwareGreedyAcquisition,
    provenance: np.ndarray | None = None,
) -> pd.DataFrame:
    """Annotate the campaign state DataFrame with acquisition scores.

    The ``predictions`` array must be aligned with the concatenation of
    ``state.unqueried_rows + state.ps_upgrade_rows`` in that order — the same
    ordering used when calling ``model.predict_smiles``.

    Five columns are appended to a copy of ``df``:

    - ``ps_score`` — PS exploration score; NaN for non-unqueried rows
    - ``drc_score`` — DRC exploitation score; NaN for training-only rows
    - ``overall_score`` — ``max(ps_score, drc_score)`` for unqueried rows,
      ``drc_score`` for PS upgrades, NaN for training-only rows
    - ``recommendation`` — ``"ps"`` or ``"drc"`` for inference targets; NaN for
      training-only rows
    - ``embedding_derived`` — True where ``provenance`` flagged the prediction
      as embedding-derived (see ``provenance`` below); NaN for training-only
      rows; always False when ``provenance`` is None
    - ``predicted_pec50`` — the raw model prediction from ``predictions``,
      unmodified by acquisition scoring; NaN for training-only rows

    Parameters
    ----------
    df : pd.DataFrame
        Original campaign state DataFrame (not mutated; a copy is returned).
    state : CampaignState
        Parsed campaign state from ``parse_campaign_state``.
    predictions : np.ndarray
        Model pEC50 predictions aligned with unqueried + ps_upgrade rows.
    acquisition : CostAwareGreedyAcquisition
        Acquisition function used to compute per-compound scores.
    provenance : np.ndarray, optional
        Boolean (or 0/1 float) array aligned with ``predictions``, forwarded
        to ``acquisition.score_summary`` so a discount (issue #36 Phase 3)
        applies to embedding-derived predictions from the concatenation
        architecture. ``None`` (default) applies no discount, matching
        current behavior.

    Returns
    -------
    pd.DataFrame
        Annotated copy with five new columns appended.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    n_inference = len(state.unqueried_rows) + len(state.ps_upgrade_rows)
    if len(predictions) != n_inference:
        raise ValueError(
            f"predictions length ({len(predictions)}) must match the number of "
            f"inference targets ({n_inference})."
        )
    if n_inference > 0 and not np.all(np.isfinite(predictions)):
        raise ValueError(
            "predictions must contain only finite values; NaN or inf values "
            "produce undefined acquisition scores."
        )
    provenance_arr = None if provenance is None else np.asarray(provenance)
    if provenance_arr is not None and len(provenance_arr) != n_inference:
        raise ValueError(
            f"provenance length ({len(provenance_arr)}) must match the number of "
            f"inference targets ({n_inference})."
        )

    result = df.copy()
    result["ps_score"] = np.nan
    result["drc_score"] = np.nan
    result["overall_score"] = np.nan
    result["recommendation"] = None  # Object dtype so string values can be assigned
    result["embedding_derived"] = None  # Object dtype so bool values can be assigned
    result["predicted_pec50"] = np.nan

    n_unqueried = len(state.unqueried_rows)
    unqueried_preds = predictions[:n_unqueried]
    upgrade_preds = predictions[n_unqueried:]
    unqueried_provenance = None if provenance_arr is None else provenance_arr[:n_unqueried]
    upgrade_provenance = None if provenance_arr is None else provenance_arr[n_unqueried:]

    # Score unqueried compounds — both PS and DRC are valid next actions
    if state.unqueried_rows:
        unqueried_canonical = [smi for _, smi in state.unqueried_rows]
        summaries = acquisition.score_summary(
            unqueried_canonical, unqueried_preds, provenance=unqueried_provenance
        )
        for (row_idx, _), summary, pred in zip(
            state.unqueried_rows, summaries, unqueried_preds, strict=False
        ):
            drc = float(summary["score_drc"])
            ps = float(summary["score_ps"])
            overall = max(drc, ps)
            rec = "drc" if drc >= ps else "ps"
            result.at[row_idx, "ps_score"] = ps
            result.at[row_idx, "drc_score"] = drc
            result.at[row_idx, "overall_score"] = overall
            result.at[row_idx, "recommendation"] = rec
            result.at[row_idx, "embedding_derived"] = summary["embedding_derived"]
            result.at[row_idx, "predicted_pec50"] = float(pred)

    # Score PS hits — only DRC upgrade is a valid next action; ps_score stays NaN
    if state.ps_upgrade_rows:
        upgrade_canonical = [smi for _, smi in state.ps_upgrade_rows]
        summaries = acquisition.score_summary(
            upgrade_canonical, upgrade_preds, provenance=upgrade_provenance
        )
        for (row_idx, _), summary, pred in zip(
            state.ps_upgrade_rows, summaries, upgrade_preds, strict=False
        ):
            drc = float(summary["score_drc"])
            result.at[row_idx, "drc_score"] = drc
            result.at[row_idx, "overall_score"] = drc
            result.at[row_idx, "recommendation"] = "drc"
            result.at[row_idx, "embedding_derived"] = summary["embedding_derived"]
            result.at[row_idx, "predicted_pec50"] = float(pred)

    return result


def validate_training_records(records: list[LabelRecord]) -> None:
    """Validate per-compound label consistency across training records.

    Each canonical SMILES may appear at most once per fidelity tier.  A
    compound that carries both a PS ``<`` LEFT record and a DRC EXACT record
    is rejected because the active/inactive disagreement between the two
    labels cannot be resolved during Tobit-loss training.

    Called by :func:`parse_campaign_state` and by
    :func:`~moal.loop._merge_pretrain_with_oracle` to enforce this invariant
    across both the ``moal plan`` and ``moal simulate`` (pretrain) workflows.

    Parameters
    ----------
    records : list[LabelRecord]
        Labeled records to validate.

    Raises
    ------
    ValueError
        If any canonical SMILES has more than one DRC record.
    ValueError
        If any canonical SMILES has more than one PS record.
    ValueError
        If any canonical SMILES has both a PS LEFT (``<``) record and a
        DRC record.
    """
    by_smiles: dict[str, list[LabelRecord]] = {}
    for record in records:
        by_smiles.setdefault(record.canonical_smiles, []).append(record)

    for canonical_smiles, grouped_records in by_smiles.items():
        drc_records = [rec for rec in grouped_records if rec.fidelity == QueryType.DOSE_RESPONSE]
        ps_records = [rec for rec in grouped_records if rec.fidelity == QueryType.PRIMARY_SCREEN]

        if len(drc_records) > 1:
            raise ValueError(
                f"Compound {canonical_smiles!r} has multiple DRC rows; "
                "expected at most one exact label per compound."
            )
        if len(ps_records) > 1:
            raise ValueError(
                f"Compound {canonical_smiles!r} has multiple PS rows; "
                "expected at most one PS label per compound."
            )
        if drc_records and ps_records and ps_records[0].censoring_type == CensoringType.LEFT:
            raise ValueError(
                f"Compound {canonical_smiles!r} has both a PS '<' (inactive) row and "
                "a DRC row. This mixed-fidelity combination is unsupported."
            )


def normalize_record_weights(records: list[LabelRecord]) -> list[LabelRecord]:
    """Return records with ``weight`` normalized to mean=1.0 per fidelity class.

    Normalizes DRC and PS records independently so that the global ``w_drc``
    and ``w_ps`` scale relationship is preserved after per-sample weighting.
    When all records already have ``weight=1.0`` (the default), normalization
    is a no-op.

    Parameters
    ----------
    records : list[LabelRecord]
        Training records whose weights will be normalized.  The original
        records are not modified — new :class:`~moal.types.LabelRecord`
        objects are returned.

    Returns
    -------
    list[LabelRecord]
        New records with normalized weights.  Ordering is preserved.

    Raises
    ------
    ValueError
        If all records in a fidelity class have weight=0 (mean is zero or
        negative), which would produce a division-by-zero normalization.
    """
    drc_weights = [rec.weight for rec in records if rec.fidelity == QueryType.DOSE_RESPONSE]
    ps_weights = [rec.weight for rec in records if rec.fidelity == QueryType.PRIMARY_SCREEN]

    drc_mean = float(np.mean(drc_weights)) if drc_weights else 1.0
    ps_mean = float(np.mean(ps_weights)) if ps_weights else 1.0

    if drc_mean <= 0:
        raise ValueError(f"Mean DRC weight is {drc_mean}; all DRC weights must be positive.")
    if ps_mean <= 0:
        raise ValueError(f"Mean PS weight is {ps_mean}; all PS weights must be positive.")

    normalized = []
    for rec in records:
        mean = drc_mean if rec.fidelity == QueryType.DOSE_RESPONSE else ps_mean
        if mean == 1.0 and rec.weight == 1.0:
            normalized.append(rec)
        else:
            normalized.append(replace(rec, weight=rec.weight / mean))
    return normalized
