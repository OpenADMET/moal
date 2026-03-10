"""Helpers for one-shot acquisition planning from mixed-fidelity CSV inputs."""

from __future__ import annotations

import logging
import math

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


def parse_training_records(
    training_df: pd.DataFrame,
    *,
    cost_ps: float,
    cost_drc: float,
    upper_bound: float,
    preprocessor: SMILESPreprocessor,
    smiles_column: str = "smiles",
    relation_column: str = "relation",
    value_column: str = "value",
    is_canonical: bool = False,
    expected_ps_threshold: float | None = None,
) -> list[LabelRecord]:
    """Parse a mixed-fidelity training CSV into ``LabelRecord`` objects.

    Expected columns are the configured SMILES, relation (``<``, ``>=``, or
    ``==``), and value columns.

    All produced records are assigned ``iteration=0`` because plan mode trains
    on an externally supplied labeled set rather than on records acquired from
    the active learning loop itself.
    """
    required_columns = {smiles_column, relation_column, value_column}
    missing = required_columns.difference(training_df.columns)
    if missing:
        raise ValueError(
            "training CSV must contain columns "
            f"{sorted(required_columns)}, got {sorted(training_df.columns)}"
        )

    records: list[LabelRecord] = []
    for row_idx, row in training_df.iterrows():
        csv_row = row_idx + 2  # account for zero indexing + header row
        raw_smiles = str(row[smiles_column])
        relation = str(row[relation_column]).strip()

        if relation not in {"<", ">=", "=="}:
            raise ValueError(
                f"Row {csv_row}: relation must be one of '<', '>=', or '==', "
                f"got {relation!r}."
            )

        try:
            value = float(row[value_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Row {csv_row}: value must be a finite numeric pEC50 datum."
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"Row {csv_row}: value must be finite, got {row[value_column]!r}."
            )
        if not (_PECO50_MIN <= value <= _PECO50_MAX):
            raise ValueError(
                f"Row {csv_row}: value must be within [{_PECO50_MIN:.1f}, {_PECO50_MAX:.1f}], "
                f"got {value}."
            )

        canonical = raw_smiles if is_canonical else preprocessor.canonicalize(raw_smiles)
        if canonical is None:
            raise ValueError(f"Row {csv_row}: invalid SMILES {raw_smiles!r}.")

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
                censoring_type=(
                    CensoringType.LEFT
                    if relation == "<"
                    else CensoringType.INTERVAL
                ),
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=cost_ps,
                iteration=_PLAN_MODE_ITERATION,
            )

        records.append(record)

    _validate_training_records(records)
    return records


def training_records_for_refit(records: list[LabelRecord]) -> list[LabelRecord]:
    """Return model-training records with upgraded PS hits de-duplicated."""
    upgraded_smiles = {
        rec.canonical_smiles
        for rec in records
        if rec.fidelity == QueryType.DOSE_RESPONSE
    }
    return [
        rec
        for rec in records
        if not (
            rec.fidelity == QueryType.PRIMARY_SCREEN
            and rec.censoring_type == CensoringType.INTERVAL
            and rec.canonical_smiles in upgraded_smiles
        )
    ]


def parse_candidate_smiles(
    candidate_df: pd.DataFrame,
    *,
    smiles_column: str,
    preprocessor: SMILESPreprocessor,
    is_canonical: bool = False,
) -> list[str]:
    """Parse, canonicalize, and deduplicate candidate-pool SMILES."""
    if smiles_column not in candidate_df.columns:
        raise ValueError(
            f"candidate CSV must contain column {smiles_column!r}, got "
            f"{sorted(candidate_df.columns)}"
        )

    candidates: list[str] = []
    seen: set[str] = set()
    n_duplicates = 0
    for row_idx, row in candidate_df.iterrows():
        csv_row = row_idx + 2
        raw_smiles = str(row[smiles_column])
        canonical = raw_smiles if is_canonical else preprocessor.canonicalize(raw_smiles)
        if canonical is None:
            raise ValueError(f"Row {csv_row}: invalid candidate SMILES {raw_smiles!r}.")
        if canonical in seen:
            n_duplicates += 1
            continue
        seen.add(canonical)
        candidates.append(canonical)

    if n_duplicates:
        logger.warning(
            "Skipped %d duplicate candidate SMILES after canonicalization.",
            n_duplicates,
        )
    return candidates


def build_acquisition_plan_dataframe(
    candidate_smiles: list[str],
    predictions: np.ndarray,
    acquisition: CostAwareGreedyAcquisition,
) -> pd.DataFrame:
    """Build the ranked acquisition plan output DataFrame.

    ``Overall Score`` is defined as ``max(PS Score, DRC Score)``. ``Query type``
    is ``"DRC"`` when ``DRC Score >= PS Score`` and ``"PS"`` otherwise, so ties
    are resolved in favor of DRC. Rows are sorted by ``Overall Score``
    descending, with original candidate input order used as a stable tie-break.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    if len(candidate_smiles) != len(predictions):
        raise ValueError(
            f"candidate_smiles length ({len(candidate_smiles)}) must match "
            f"predictions length ({len(predictions)})."
        )
    if len(predictions) > 0 and not np.all(np.isfinite(predictions)):
        raise ValueError(
            "predictions must contain only finite values; NaN or inf values "
            "produce undefined acquisition scores and must be filtered before "
            "building the acquisition plan."
        )

    rows = []
    for input_rank, summary in enumerate(
        acquisition.score_summary(candidate_smiles, predictions)
    ):
        drc_score = float(summary["score_drc"])
        ps_score = float(summary["score_ps"])
        # Overall Score is max(DRC, PS); ties go to DRC because it is the more
        # informative follow-up assay when both actions look equally valuable.
        query_type = "DRC" if drc_score >= ps_score else "PS"
        overall_score = max(drc_score, ps_score)
        rows.append(
            {
                "_input_order": input_rank,
                "Compound (SMILES)": summary["smiles"],
                "Query type": query_type,
                "PS Score": ps_score,
                "DRC Score": drc_score,
                "Overall Score": overall_score,
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        by=["Overall Score", "_input_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked[
        [
            "Rank",
            "Compound (SMILES)",
            "Query type",
            "PS Score",
            "DRC Score",
            "Overall Score",
        ]
    ]


def _validate_training_records(records: list[LabelRecord]) -> None:
    by_smiles: dict[str, list[LabelRecord]] = {}
    for record in records:
        by_smiles.setdefault(record.canonical_smiles, []).append(record)

    for canonical_smiles, grouped_records in by_smiles.items():
        drc_records = [
            rec for rec in grouped_records if rec.fidelity == QueryType.DOSE_RESPONSE
        ]
        ps_records = [
            rec for rec in grouped_records if rec.fidelity == QueryType.PRIMARY_SCREEN
        ]

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
        if (
            drc_records
            and ps_records
            and ps_records[0].censoring_type == CensoringType.LEFT
        ):
            raise ValueError(
                f"Compound {canonical_smiles!r} has both a PS '<' row and a DRC row. "
                "This mixed-fidelity combination is unsupported in plan mode."
            )
