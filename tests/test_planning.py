"""Tests for offline planning helpers."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from moal.acquisition import CostAwareGreedyAcquisition
from moal.planning import (
    annotate_campaign_state,
    normalize_record_weights,
    parse_campaign_state,
    parse_pretrain_records,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor
from moal.types import CensoringType, LabelRecord, QueryType


@pytest.fixture
def preprocessor() -> SMILESPreprocessor:
    return SMILESPreprocessor()


@pytest.fixture
def acquisition() -> CostAwareGreedyAcquisition:
    return CostAwareGreedyAcquisition(
        cost_ps=1.0,
        cost_drc=10.0,
        ps_threshold=5.0,
        target_threshold=7.0,
        tau=0.5,
    )


def _state_df(*rows: dict) -> pd.DataFrame:
    """Build a campaign state DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


class TestParseCampaignState:
    """Tests for parse_campaign_state, covering row classification, validation, and error handling."""

    def test_all_four_row_states_are_classified_correctly(self, preprocessor):
        """All four row types (unqueried, PS-miss, PS-hit, DRC) must be routed to the correct partition."""
        df = _state_df(
            {"smiles": "CCO", "relation": "", "value": ""},
            {"smiles": "CCN", "relation": "<", "value": 5.0},
            {"smiles": "CCC", "relation": ">=", "value": 5.0},
            {"smiles": "c1ccccc1", "relation": "==", "value": 7.4},
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        assert len(state.training_records) == 3
        assert len(state.unqueried_rows) == 1
        assert len(state.ps_upgrade_rows) == 1
        assert state.unqueried_rows[0][0] == 0
        assert state.ps_upgrade_rows[0][0] == 2

    def test_unqueried_rows_have_correct_row_indices(self, preprocessor):
        """Unqueried row indices must match the original DataFrame row positions so annotation writes to the right rows."""
        df = _state_df(
            {"smiles": "CCO", "relation": "==", "value": 7.0},
            {"smiles": "CCN", "relation": "", "value": ""},
            {"smiles": "CCC", "relation": "", "value": ""},
        )

        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        assert [idx for idx, _ in state.unqueried_rows] == [1, 2]

    def test_ps_hits_appear_in_both_training_and_ps_upgrade_rows(self, preprocessor):
        """PS-INTERVAL rows must appear as training records and also as DRC-upgrade inference targets."""
        df = _state_df(
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        assert len(state.training_records) == 1
        assert state.training_records[0].fidelity == QueryType.PRIMARY_SCREEN
        assert state.training_records[0].censoring_type == CensoringType.INTERVAL
        assert len(state.ps_upgrade_rows) == 1
        assert state.ps_upgrade_rows[0][1] == state.training_records[0].canonical_smiles

    def test_ps_misses_are_training_only(self, preprocessor):
        """PS-LEFT rows are confirmed inactive and must appear in training only, never as DRC-upgrade candidates."""
        df = _state_df({"smiles": "CCO", "relation": "<", "value": 5.0})

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        assert len(state.training_records) == 1
        assert state.training_records[0].censoring_type == CensoringType.LEFT
        assert len(state.unqueried_rows) == 0
        assert len(state.ps_upgrade_rows) == 0

    def test_drc_rows_are_training_only(self, preprocessor):
        """DRC/exact rows are terminal and must appear only in training, not in upgrade or unqueried lists."""
        df = _state_df({"smiles": "CCO", "relation": "==", "value": 7.2})

        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        assert len(state.training_records) == 1
        assert state.training_records[0].censoring_type == CensoringType.EXACT
        assert len(state.unqueried_rows) == 0
        assert len(state.ps_upgrade_rows) == 0

    def test_label_types_match_relations(self, preprocessor):
        """Each relation symbol must produce the expected censoring type, fidelity, and bound on its LabelRecord."""
        df = _state_df(
            {"smiles": "CCO", "relation": "<", "value": 5.0},
            {"smiles": "CCN", "relation": ">=", "value": 5.0},
            {"smiles": "CCC", "relation": "==", "value": 7.4},
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        assert [r.censoring_type for r in state.training_records] == [
            CensoringType.LEFT,
            CensoringType.INTERVAL,
            CensoringType.EXACT,
        ]
        assert [r.fidelity for r in state.training_records] == [
            QueryType.PRIMARY_SCREEN,
            QueryType.PRIMARY_SCREEN,
            QueryType.DOSE_RESPONSE,
        ]
        assert state.training_records[1].upper_bound == 11.0
        assert state.training_records[2].upper_bound == pytest.approx(7.4)
        assert all(r.iteration == 0 for r in state.training_records)

    def test_partial_row_population_raises(self, preprocessor):
        """A row with relation populated but value empty (or vice versa) must raise ValueError immediately."""
        df = _state_df({"smiles": "CCO", "relation": ">=", "value": float("nan")})

        with pytest.raises(ValueError, match="both be populated or both be empty"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_missing_smiles_column_raises(self, preprocessor):
        """A DataFrame missing the smiles column must raise ValueError that names the expected column."""
        df = pd.DataFrame([{"compound": "CCO", "relation": "", "value": ""}])

        with pytest.raises(ValueError, match="state CSV must contain column 'smiles'"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_threshold_mismatch_raises(self, preprocessor):
        """A PS row whose value differs from expected_ps_threshold must raise ValueError to prevent silent misconfiguration."""
        df = _state_df({"smiles": "CCO", "relation": ">=", "value": 4.5})

        with pytest.raises(ValueError, match="does not match config oracle.ps_threshold"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_invalid_relation_raises(self, preprocessor):
        """An unrecognized relation symbol must raise ValueError with a message listing the accepted symbols."""
        df = _state_df({"smiles": "CCO", "relation": "??", "value": 5.0})

        with pytest.raises(ValueError, match="relation must be one of"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_invalid_smiles_raises(self, preprocessor):
        """An unparseable SMILES string must raise ValueError so the caller can report it rather than producing garbage records."""
        df = _state_df({"smiles": "not-a-smiles", "relation": "", "value": ""})

        with pytest.raises(ValueError, match="invalid SMILES"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_left_ps_plus_drc_combination_raises(self, preprocessor):
        """A compound with both a PS-LEFT and a DRC row is an unsupported fidelity mix and must raise ValueError."""
        df = _state_df(
            {"smiles": "CCO", "relation": "<", "value": 5.0},
            {"smiles": "CCO", "relation": "==", "value": 6.8},
        )

        with pytest.raises(ValueError, match="mixed-fidelity combination is unsupported"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_multiple_drc_rows_for_same_compound_raises(self, preprocessor):
        """Two DRC rows for the same compound is invalid since only one exact label is permitted per compound."""
        df = _state_df(
            {"smiles": "CCO", "relation": "==", "value": 6.8},
            {"smiles": "CCO", "relation": "==", "value": 7.1},
        )

        with pytest.raises(ValueError, match="multiple DRC rows"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_multiple_ps_rows_for_same_compound_raises(self, preprocessor):
        """Two PS rows for the same compound must raise ValueError because the result would be ambiguous."""
        df = _state_df(
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
        )

        with pytest.raises(ValueError, match="multiple PS rows"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_cross_partition_duplicate_raises(self, preprocessor):
        """A compound appearing as both labeled and unqueried is a data integrity error and must raise ValueError."""
        df = _state_df(
            {"smiles": "CCO", "relation": "==", "value": 7.0},
            {"smiles": "CCO", "relation": "", "value": ""},
        )

        with pytest.raises(ValueError, match="appear as both labeled and unqueried"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_duplicate_unqueried_rows_are_skipped_with_warning(self, preprocessor, caplog):
        """Duplicate unqueried SMILES after canonicalization must emit a WARNING and keep only the first occurrence."""
        df = _state_df(
            {
                "smiles": "C(C)O",
                "relation": "",
                "value": "",
            },  # same as CCO after canonicalization
            {"smiles": "CCO", "relation": "", "value": ""},
        )

        with caplog.at_level("WARNING", logger="moal.planning"):
            state = parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

        assert len(state.unqueried_rows) == 1
        assert "duplicate unqueried SMILES" in caplog.text

    def test_refit_records_drop_upgraded_interval_ps_rows(self, preprocessor):
        """When a compound has both PS-INTERVAL and DRC records, the PS record must be dropped from refit to avoid double-weighting."""
        df = _state_df(
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
            {"smiles": "CCO", "relation": "==", "value": 7.2},
            {"smiles": "CCN", "relation": "==", "value": 6.1},
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        fit_records = training_records_for_refit(state.training_records)

        assert len(state.training_records) == 3
        assert len(fit_records) == 2
        assert all(
            not (
                r.canonical_smiles == state.training_records[0].canonical_smiles
                and r.fidelity == QueryType.PRIMARY_SCREEN
            )
            for r in fit_records
        )

    def test_log2fc_columns_populate_raw_ps_readouts_on_ps_rows(self, preprocessor):
        """log2fc_columns values must land on LabelRecord.raw_ps_readouts keyed by column name."""
        df = _state_df(
            {"smiles": "CCO", "relation": "<", "value": 5.0, "log2fc_1um": -1.2, "pic50": ""},
            {"smiles": "CCN", "relation": ">=", "value": 5.0, "log2fc_1um": 3.4, "pic50": 6.8},
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            log2fc_columns=["log2fc_1um", "pic50"],
            expected_ps_threshold=5.0,
        )

        readouts = {r.canonical_smiles: r.raw_ps_readouts for r in state.training_records}
        assert readouts[preprocessor.canonicalize("CCO")] == {"log2fc_1um": -1.2}
        assert readouts[preprocessor.canonicalize("CCN")] == {"log2fc_1um": 3.4, "pic50": 6.8}

    def test_log2fc_columns_blank_cell_omits_key(self, preprocessor):
        """An empty log2fc cell must be omitted from raw_ps_readouts rather than raising."""
        df = _state_df({"smiles": "CCO", "relation": "<", "value": 5.0, "log2fc": ""})

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            log2fc_columns=["log2fc"],
            expected_ps_threshold=5.0,
        )

        assert state.training_records[0].raw_ps_readouts == {}

    def test_missing_log2fc_column_raises(self, preprocessor):
        """Requesting a log2fc_columns entry absent from the CSV must raise ValueError."""
        df = _state_df({"smiles": "CCO", "relation": "<", "value": 5.0})

        with pytest.raises(ValueError, match="log2fc_columns"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                log2fc_columns=["log2fc"],
            )

    def test_refit_records_merge_raw_ps_readouts_onto_surviving_drc_record(self, preprocessor):
        """When a DRC-upgrade record lacks readouts of its own, the dropped PS record's readouts must be merged onto it."""
        upgraded_smiles = preprocessor.canonicalize("CCO")
        ps_record = LabelRecord(
            smiles="CCO",
            canonical_smiles=upgraded_smiles,
            value=5.0,
            upper_bound=11.0,
            censoring_type=CensoringType.INTERVAL,
            fidelity=QueryType.PRIMARY_SCREEN,
            cost=1.0,
            iteration=0,
            raw_ps_readouts={"log2fc_1um": 3.4},
        )
        drc_record = LabelRecord(
            smiles="CCO",
            canonical_smiles=upgraded_smiles,
            value=7.2,
            upper_bound=7.2,
            censoring_type=CensoringType.EXACT,
            fidelity=QueryType.DOSE_RESPONSE,
            cost=10.0,
            iteration=1,
        )

        fit_records = training_records_for_refit([ps_record, drc_record])

        assert len(fit_records) == 1
        assert fit_records[0].fidelity == QueryType.DOSE_RESPONSE
        assert fit_records[0].raw_ps_readouts == {"log2fc_1um": 3.4}

    def test_custom_column_names_are_supported(self, preprocessor):
        """Non-default smiles, relation, and value column names must be mapped correctly throughout parsing."""
        df = pd.DataFrame(
            [
                {"compound": "CCO", "kind": "<", "potency": 5.0},
                {"compound": "CCN", "kind": "==", "potency": 7.4},
                {"compound": "CCC", "kind": "", "potency": ""},
            ]
        )

        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            smiles_column="compound",
            relation_column="kind",
            value_column="potency",
            expected_ps_threshold=5.0,
        )

        assert len(state.training_records) == 2
        assert len(state.unqueried_rows) == 1


class TestAnnotateCampaignState:
    """Tests for annotate_campaign_state, covering score assignment, recommendations, and error handling."""

    def test_unqueried_rows_get_all_four_score_columns(self, preprocessor, acquisition):
        """Unqueried compounds must receive ps_score, drc_score, overall_score, and recommendation after annotation."""
        df = _state_df(
            {"smiles": "CCO", "relation": "", "value": ""},
        )
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )
        # Prediction near the PS threshold maximizes PS entropy
        predictions = np.array([5.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert not pd.isna(result.at[0, "ps_score"])
        assert not pd.isna(result.at[0, "drc_score"])
        assert not pd.isna(result.at[0, "overall_score"])
        assert result.at[0, "recommendation"] in {"ps", "drc"}
        assert result.at[0, "overall_score"] == pytest.approx(
            max(result.at[0, "ps_score"], result.at[0, "drc_score"])
        )

    def test_ps_upgrade_rows_get_drc_only(self, preprocessor, acquisition):
        """PS-upgrade rows must receive only a DRC score; ps_score must stay NaN because PS re-query is not an option."""
        df = _state_df(
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
        )
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        predictions = np.array([8.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert pd.isna(result.at[0, "ps_score"])
        assert not pd.isna(result.at[0, "drc_score"])
        assert result.at[0, "overall_score"] == pytest.approx(result.at[0, "drc_score"])
        assert result.at[0, "recommendation"] == "drc"

    def test_training_only_rows_get_nan_scores(self, preprocessor, acquisition):
        """Training-only rows (PS-LEFT and DRC) must have NaN for all four score columns since no further query is needed."""
        df = _state_df(
            {"smiles": "CCO", "relation": "<", "value": 5.0},
            {"smiles": "CCN", "relation": "==", "value": 7.2},
        )
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        result = annotate_campaign_state(df, state, np.empty(0, dtype=np.float32), acquisition)

        for col in ("ps_score", "drc_score", "overall_score", "recommendation"):
            assert pd.isna(result.at[0, col])
            assert pd.isna(result.at[1, col])

    def test_recommendation_is_ps_when_ps_score_dominates(self, preprocessor, acquisition):
        """A prediction at the PS threshold maximizes entropy, so recommendation must be 'ps'."""
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )
        # y_hat at the PS threshold gives maximal PS entropy and minimal DRC score
        predictions = np.array([5.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert result.at[0, "recommendation"] == "ps"

    def test_recommendation_is_drc_when_drc_score_dominates(self, preprocessor, acquisition):
        """A high prediction makes DRC exploitation outweigh PS exploration, so recommendation must be 'drc'."""
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )
        # High y_hat gives large DRC exploitation score
        predictions = np.array([11.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert result.at[0, "recommendation"] == "drc"

    def test_ps_upgrade_recommendation_is_always_drc(self, preprocessor, acquisition):
        """PS-upgrade rows may only be queried via DRC, so recommendation must always be 'drc' regardless of prediction."""
        df = _state_df(
            {"smiles": "CCO", "relation": ">=", "value": 5.0},
            {"smiles": "CCN", "relation": ">=", "value": 5.0},
        )
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        # Both near the PS threshold, but DRC is the only valid action for upgrades
        predictions = np.array([5.0, 5.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert result.at[0, "recommendation"] == "drc"
        assert result.at[1, "recommendation"] == "drc"

    def test_rejects_length_mismatch(self, preprocessor, acquisition):
        """Predictions array length must match the number of inference targets or ValueError is raised."""
        df = _state_df(
            {"smiles": "CCO", "relation": "", "value": ""},
            {"smiles": "CCN", "relation": "", "value": ""},
        )
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        with pytest.raises(ValueError, match="must match the number of inference targets"):
            annotate_campaign_state(df, state, np.array([5.0], dtype=np.float32), acquisition)

    def test_rejects_non_finite_predictions(self, preprocessor, acquisition):
        """NaN or inf in predictions must raise ValueError before scoring to prevent silently writing invalid scores."""
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        with pytest.raises(ValueError, match="must contain only finite values"):
            annotate_campaign_state(df, state, np.array([np.nan], dtype=np.float32), acquisition)

    def test_empty_inference_targets_returns_all_nan_score_columns(self, preprocessor, acquisition):
        """When no inference targets exist, all four score columns must be present in the result but filled with NaN."""
        df = _state_df({"smiles": "CCO", "relation": "==", "value": 7.2})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        result = annotate_campaign_state(df, state, np.empty(0, dtype=np.float32), acquisition)

        assert all(
            c in result.columns
            for c in ("ps_score", "drc_score", "overall_score", "recommendation")
        )
        assert pd.isna(result.at[0, "ps_score"])

    def test_original_dataframe_is_not_mutated(self, preprocessor, acquisition):
        """annotate_campaign_state must return a new DataFrame and must not add columns to the input."""
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        annotate_campaign_state(df, state, np.array([5.0], dtype=np.float32), acquisition)

        assert "ps_score" not in df.columns

    def test_uses_acquisition_score_summary_for_scoring(self, preprocessor):
        """Scores must come from acquisition.score_summary() to ensure the acquisition strategy drives recommendations."""
        acquisition = Mock(spec_set=["score_summary"])
        acquisition.score_summary.return_value = [
            {"smiles": "CCO", "score_drc": 0.3, "score_ps": 0.7},
        ]
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
        )

        result = annotate_campaign_state(df, state, np.array([5.0], dtype=np.float32), acquisition)

        assert result.at[0, "ps_score"] == pytest.approx(0.7)
        assert result.at[0, "drc_score"] == pytest.approx(0.3)
        assert result.at[0, "overall_score"] == pytest.approx(0.7)
        assert result.at[0, "recommendation"] == "ps"


class TestParsePretrainRecords:
    """Tests for parse_pretrain_records() — the thin pretrain CSV wrapper."""

    def test_labeled_rows_returned_as_training_records(self, preprocessor):
        """All three labeled relation types must be returned as LabelRecord objects."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "<", "value": 5.0},
                {"smiles": "CCN", "relation": ">=", "value": 5.0},
                {"smiles": "CCC", "relation": "==", "value": 7.2},
            ]
        )
        records = parse_pretrain_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        assert len(records) == 3
        fidelities = {r.fidelity for r in records}
        assert QueryType.PRIMARY_SCREEN in fidelities
        assert QueryType.DOSE_RESPONSE in fidelities

    def test_unqueried_rows_are_excluded(self, preprocessor):
        """Unqueried rows (empty relation/value) must be excluded from the returned training records."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 7.5},
                {"smiles": "CCN", "relation": "", "value": ""},
                {"smiles": "CCC", "relation": "", "value": ""},
            ]
        )
        records = parse_pretrain_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
        )
        # Only the labeled row is returned
        assert len(records) == 1
        assert records[0].fidelity == QueryType.DOSE_RESPONSE

    def test_unqueried_rows_trigger_logger_warning(self, preprocessor, caplog):
        """A logger.warning must be emitted when unqueried rows are found in the pretrain CSV."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 7.5},
                {"smiles": "CCN", "relation": "", "value": ""},
            ]
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="moal.planning"):
            parse_pretrain_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )
        assert any("unqueried" in msg.lower() for msg in caplog.messages), (
            "Expected a warning about unqueried rows, got: " + str(caplog.messages)
        )

    def test_ps_threshold_mismatch_raises(self, preprocessor):
        """A PS value that doesn't match expected_ps_threshold must raise ValueError."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "<", "value": 6.0},  # wrong threshold
            ]
        )
        with pytest.raises(ValueError, match="PS threshold"):
            parse_pretrain_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_invalid_smiles_raises(self, preprocessor):
        """An unparseable SMILES string must raise ValueError."""
        df = pd.DataFrame(
            [
                {"smiles": "not_a_smiles", "relation": "==", "value": 7.0},
            ]
        )
        with pytest.raises(ValueError, match="invalid SMILES"):
            parse_pretrain_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
            )

    def test_empty_dataframe_returns_empty_list(self, preprocessor):
        """An empty DataFrame must return an empty list without error."""
        df = pd.DataFrame(columns=["smiles", "relation", "value"])
        records = parse_pretrain_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
        )
        assert records == []

    def test_returns_only_training_records_not_upgrade_rows(self, preprocessor):
        """ps_upgrade_rows from parse_campaign_state must not be exposed — only training records."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": ">=", "value": 5.0},
            ]
        )
        records = parse_pretrain_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        # PS INTERVAL is a training record; ps_upgrade_rows are internal to CampaignState
        assert len(records) == 1
        assert records[0].censoring_type == CensoringType.INTERVAL


# ---------------------------------------------------------------------------
# normalize_record_weights
# ---------------------------------------------------------------------------


def _make_label_record(
    smiles: str,
    fidelity: QueryType,
    weight: float = 1.0,
) -> LabelRecord:
    """Minimal LabelRecord for normalize_record_weights tests."""
    return LabelRecord(
        smiles=smiles,
        canonical_smiles=smiles,
        value=5.0,
        upper_bound=5.0,
        censoring_type=CensoringType.EXACT,
        fidelity=fidelity,
        cost=1.0,
        iteration=0,
        weight=weight,
    )


class TestNormalizeRecordWeights:
    """Tests for normalize_record_weights()."""

    def test_unit_weights_unchanged(self):
        """All weight=1.0 records must remain unchanged after normalization."""
        records = [
            _make_label_record("C", QueryType.DOSE_RESPONSE, weight=1.0),
            _make_label_record("CC", QueryType.DOSE_RESPONSE, weight=1.0),
            _make_label_record("CCC", QueryType.PRIMARY_SCREEN, weight=1.0),
        ]
        normalized = normalize_record_weights(records)
        assert [r.weight for r in normalized] == pytest.approx([1.0, 1.0, 1.0])

    def test_drc_and_ps_normalized_independently(self):
        """DRC and PS weights must be normalized to mean=1.0 independently."""
        records = [
            _make_label_record("C", QueryType.DOSE_RESPONSE, weight=1.0),
            _make_label_record("CC", QueryType.DOSE_RESPONSE, weight=3.0),
            _make_label_record("CCC", QueryType.PRIMARY_SCREEN, weight=2.0),
            _make_label_record("CCCC", QueryType.PRIMARY_SCREEN, weight=6.0),
        ]
        normalized = normalize_record_weights(records)
        drc_weights = [r.weight for r in normalized if r.fidelity == QueryType.DOSE_RESPONSE]
        ps_weights = [r.weight for r in normalized if r.fidelity == QueryType.PRIMARY_SCREEN]
        assert sum(drc_weights) / len(drc_weights) == pytest.approx(1.0, rel=1e-6)
        assert sum(ps_weights) / len(ps_weights) == pytest.approx(1.0, rel=1e-6)

    def test_preserves_relative_ordering(self):
        """Relative order of weights within a fidelity class must be preserved."""
        records = [
            _make_label_record("C", QueryType.DOSE_RESPONSE, weight=1.0),
            _make_label_record("CC", QueryType.DOSE_RESPONSE, weight=4.0),
            _make_label_record("CCC", QueryType.DOSE_RESPONSE, weight=7.0),
        ]
        normalized = normalize_record_weights(records)
        weights = [r.weight for r in normalized]
        assert weights[0] < weights[1] < weights[2]

    def test_zero_weight_raises(self):
        """A record with weight=0.0 produces a zero mean, which must raise ValueError."""
        records = [
            _make_label_record("C", QueryType.DOSE_RESPONSE, weight=0.0),
            _make_label_record("CC", QueryType.DOSE_RESPONSE, weight=0.0),
        ]
        with pytest.raises(ValueError, match="Mean DRC weight"):
            normalize_record_weights(records)


class TestWeightColumnParsing:
    """Tests for weight_column support in parse_campaign_state()."""

    @pytest.fixture
    def preprocessor(self):
        return SMILESPreprocessor()

    def test_weight_column_read_correctly(self, preprocessor):
        """Records must carry the weight from the weight_column."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 6.0, "w": 4.0},
                {"smiles": "CC", "relation": "<", "value": 5.0, "w": 0.5},
            ]
        )
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            weight_column="w",
        )
        weights = {rec.fidelity: rec.weight for rec in state.training_records}
        assert weights[QueryType.DOSE_RESPONSE] == pytest.approx(4.0)
        assert weights[QueryType.PRIMARY_SCREEN] == pytest.approx(0.5)

    def test_weight_column_nan_defaults_to_one(self, preprocessor):
        """NaN in weight_column must default to weight=1.0."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 6.0, "w": float("nan")},
            ]
        )
        state = parse_campaign_state(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            weight_column="w",
        )
        assert state.training_records[0].weight == pytest.approx(1.0)

    def test_weight_column_invalid_raises(self, preprocessor):
        """Zero weight in weight_column must raise ValueError."""
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 6.0, "w": 0.0},
            ]
        )
        with pytest.raises(ValueError, match="weight must be finite and positive"):
            parse_campaign_state(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                weight_column="w",
            )
