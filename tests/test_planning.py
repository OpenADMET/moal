"""Tests for offline planning helpers."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from moal.acquisition import CostAwareGreedyAcquisition
from moal.planning import (
    CampaignState,
    annotate_campaign_state,
    parse_campaign_state,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor
from moal.types import CensoringType, QueryType


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
    def test_all_four_row_states_are_classified_correctly(self, preprocessor):
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
        df = _state_df({"smiles": "CCO", "relation": "==", "value": 7.2})

        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        assert len(state.training_records) == 1
        assert state.training_records[0].censoring_type == CensoringType.EXACT
        assert len(state.unqueried_rows) == 0
        assert len(state.ps_upgrade_rows) == 0

    def test_label_types_match_relations(self, preprocessor):
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
        df = _state_df({"smiles": "CCO", "relation": ">=", "value": float("nan")})

        with pytest.raises(ValueError, match="both be populated or both be empty"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_missing_smiles_column_raises(self, preprocessor):
        df = pd.DataFrame([{"compound": "CCO", "relation": "", "value": ""}])

        with pytest.raises(ValueError, match="state CSV must contain column 'smiles'"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_threshold_mismatch_raises(self, preprocessor):
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
        df = _state_df({"smiles": "CCO", "relation": "??", "value": 5.0})

        with pytest.raises(ValueError, match="relation must be one of"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_invalid_smiles_raises(self, preprocessor):
        df = _state_df({"smiles": "not-a-smiles", "relation": "", "value": ""})

        with pytest.raises(ValueError, match="invalid SMILES"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_left_ps_plus_drc_combination_raises(self, preprocessor):
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
        df = _state_df(
            {"smiles": "CCO", "relation": "==", "value": 6.8},
            {"smiles": "CCO", "relation": "==", "value": 7.1},
        )

        with pytest.raises(ValueError, match="multiple DRC rows"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_multiple_ps_rows_for_same_compound_raises(self, preprocessor):
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
        df = _state_df(
            {"smiles": "CCO", "relation": "==", "value": 7.0},
            {"smiles": "CCO", "relation": "", "value": ""},
        )

        with pytest.raises(ValueError, match="appear as both labeled and unqueried"):
            parse_campaign_state(
                df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
            )

    def test_duplicate_unqueried_rows_are_skipped_with_warning(
        self, preprocessor, caplog
    ):
        df = _state_df(
            {"smiles": "C(C)O", "relation": "", "value": ""},  # same as CCO after canonicalization
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

    def test_custom_column_names_are_supported(self, preprocessor):
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
    def test_unqueried_rows_get_all_four_score_columns(self, preprocessor, acquisition):
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

        result = annotate_campaign_state(
            df, state, np.empty(0, dtype=np.float32), acquisition
        )

        for col in ("ps_score", "drc_score", "overall_score", "recommendation"):
            assert pd.isna(result.at[0, col])
            assert pd.isna(result.at[1, col])

    def test_recommendation_is_ps_when_ps_score_dominates(self, preprocessor, acquisition):
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )
        # y_hat at the PS threshold gives maximal PS entropy and minimal DRC score
        predictions = np.array([5.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert result.at[0, "recommendation"] == "ps"

    def test_recommendation_is_drc_when_drc_score_dominates(
        self, preprocessor, acquisition
    ):
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )
        # High y_hat gives large DRC exploitation score
        predictions = np.array([11.0], dtype=np.float32)

        result = annotate_campaign_state(df, state, predictions, acquisition)

        assert result.at[0, "recommendation"] == "drc"

    def test_ps_upgrade_recommendation_is_always_drc(self, preprocessor, acquisition):
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
        df = _state_df(
            {"smiles": "CCO", "relation": "", "value": ""},
            {"smiles": "CCN", "relation": "", "value": ""},
        )
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        with pytest.raises(ValueError, match="must match the number of inference targets"):
            annotate_campaign_state(
                df, state, np.array([5.0], dtype=np.float32), acquisition
            )

    def test_rejects_non_finite_predictions(self, preprocessor, acquisition):
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        with pytest.raises(ValueError, match="must contain only finite values"):
            annotate_campaign_state(
                df, state, np.array([np.nan], dtype=np.float32), acquisition
            )

    def test_empty_inference_targets_returns_all_nan_score_columns(
        self, preprocessor, acquisition
    ):
        df = _state_df({"smiles": "CCO", "relation": "==", "value": 7.2})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        result = annotate_campaign_state(
            df, state, np.empty(0, dtype=np.float32), acquisition
        )

        assert all(c in result.columns for c in ("ps_score", "drc_score", "overall_score", "recommendation"))
        assert pd.isna(result.at[0, "ps_score"])

    def test_original_dataframe_is_not_mutated(self, preprocessor, acquisition):
        df = _state_df({"smiles": "CCO", "relation": "", "value": ""})
        state = parse_campaign_state(
            df, cost_ps=1.0, cost_drc=10.0, upper_bound=11.0, preprocessor=preprocessor
        )

        annotate_campaign_state(df, state, np.array([5.0], dtype=np.float32), acquisition)

        assert "ps_score" not in df.columns

    def test_uses_acquisition_score_summary_for_scoring(self, preprocessor):
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

        result = annotate_campaign_state(
            df, state, np.array([5.0], dtype=np.float32), acquisition
        )

        assert result.at[0, "ps_score"] == pytest.approx(0.7)
        assert result.at[0, "drc_score"] == pytest.approx(0.3)
        assert result.at[0, "overall_score"] == pytest.approx(0.7)
        assert result.at[0, "recommendation"] == "ps"

