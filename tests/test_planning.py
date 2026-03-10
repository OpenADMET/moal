"""Tests for offline planning helpers."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from moal.acquisition import CostAwareGreedyAcquisition
from moal.planning import (
    build_acquisition_plan_dataframe,
    parse_candidate_smiles,
    parse_training_records,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor
from moal.types import CensoringType, QueryType


@pytest.fixture
def preprocessor() -> SMILESPreprocessor:
    return SMILESPreprocessor()


class TestParseTrainingRecords:
    def test_relations_map_to_expected_label_types(self, preprocessor):
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "<", "value": 5.0},
                {"smiles": "CCN", "relation": ">=", "value": 5.0},
                {"smiles": "CCC", "relation": "==", "value": 7.4},
            ]
        )

        records = parse_training_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )

        assert [record.fidelity for record in records] == [
            QueryType.PRIMARY_SCREEN,
            QueryType.PRIMARY_SCREEN,
            QueryType.DOSE_RESPONSE,
        ]
        assert [record.censoring_type for record in records] == [
            CensoringType.LEFT,
            CensoringType.INTERVAL,
            CensoringType.EXACT,
        ]
        assert records[1].upper_bound == 11.0
        assert records[2].upper_bound == pytest.approx(7.4)
        assert all(record.iteration == 0 for record in records)

    def test_threshold_mismatch_raises(self, preprocessor):
        df = pd.DataFrame([{"smiles": "CCO", "relation": ">=", "value": 4.5}])

        with pytest.raises(ValueError, match="does not match config oracle.ps_threshold"):
            parse_training_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_missing_required_columns_raises(self, preprocessor):
        df = pd.DataFrame([{"smiles": "CCO", "pec50": 6.2}])

        with pytest.raises(ValueError, match="training CSV must contain columns"):
            parse_training_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_refit_records_drop_upgraded_interval_ps_rows(self, preprocessor):
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": ">=", "value": 5.0},
                {"smiles": "CCO", "relation": "==", "value": 7.2},
                {"smiles": "CCN", "relation": "==", "value": 6.1},
            ]
        )

        records = parse_training_records(
            df,
            cost_ps=1.0,
            cost_drc=10.0,
            upper_bound=11.0,
            preprocessor=preprocessor,
            expected_ps_threshold=5.0,
        )
        fit_records = training_records_for_refit(records)

        assert len(records) == 3
        assert len(fit_records) == 2
        assert all(
            not (
                record.canonical_smiles == records[0].canonical_smiles
                and record.fidelity == QueryType.PRIMARY_SCREEN
            )
            for record in fit_records
        )

    def test_left_ps_plus_drc_combination_raises(self, preprocessor):
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "<", "value": 5.0},
                {"smiles": "CCO", "relation": "==", "value": 6.8},
            ]
        )

        with pytest.raises(
            ValueError, match="mixed-fidelity combination is unsupported in plan mode"
        ):
            parse_training_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_multiple_drc_rows_raise(self, preprocessor):
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": "==", "value": 6.8},
                {"smiles": "CCO", "relation": "==", "value": 7.1},
            ]
        )

        with pytest.raises(ValueError, match="multiple DRC rows"):
            parse_training_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )

    def test_multiple_ps_rows_raise(self, preprocessor):
        df = pd.DataFrame(
            [
                {"smiles": "CCO", "relation": ">=", "value": 5.0},
                {"smiles": "CCO", "relation": ">=", "value": 5.0},
            ]
        )

        with pytest.raises(ValueError, match="multiple PS rows"):
            parse_training_records(
                df,
                cost_ps=1.0,
                cost_drc=10.0,
                upper_bound=11.0,
                preprocessor=preprocessor,
                expected_ps_threshold=5.0,
            )


class TestParseCandidateSmiles:
    def test_deduplicates_after_canonicalization(self, preprocessor):
        df = pd.DataFrame({"smiles": ["C(C)O", "CCO", "CCN"]})

        candidates = parse_candidate_smiles(
            df,
            smiles_column="smiles",
            preprocessor=preprocessor,
        )

        assert candidates == ["CCO", "CCN"]

    def test_missing_smiles_column_raises(self, preprocessor):
        df = pd.DataFrame({"compound": ["CCO"]})

        with pytest.raises(ValueError, match="candidate CSV must contain column 'smiles'"):
            parse_candidate_smiles(
                df,
                smiles_column="smiles",
                preprocessor=preprocessor,
            )

    def test_invalid_candidate_smiles_raises(self, preprocessor):
        df = pd.DataFrame({"smiles": ["not-a-smiles"]})

        with pytest.raises(ValueError, match="invalid candidate SMILES"):
            parse_candidate_smiles(
                df,
                smiles_column="smiles",
                preprocessor=preprocessor,
            )


class TestBuildAcquisitionPlanDataFrame:
    def test_ranks_by_overall_score_and_formats_columns(self):
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )

        result = build_acquisition_plan_dataframe(
            candidate_smiles=["ambiguous", "high"],
            predictions=np.array([5.0, 8.0], dtype=np.float32),
            acquisition=acquisition,
        )

        assert list(result.columns) == [
            "Rank",
            "Compound (SMILES)",
            "Query type",
            "PS Score",
            "DRC Score",
            "Overall Score",
        ]
        assert result["Rank"].tolist() == [1, 2]
        assert result["Compound (SMILES)"].tolist() == ["ambiguous", "high"]
        assert result["Query type"].tolist() == ["PS", "DRC"]
        assert np.allclose(
            result["Overall Score"].to_numpy(),
            np.maximum(result["PS Score"].to_numpy(), result["DRC Score"].to_numpy()),
        )
        assert result["Overall Score"].is_monotonic_decreasing

    def test_ties_choose_drc_and_preserve_input_order(self):
        acquisition = Mock(spec_set=["score_summary"])
        acquisition.score_summary.return_value = [
            {
                "smiles": "first",
                "score_ps": 0.4,
                "score_drc": 0.4,
            },
            {
                "smiles": "second",
                "score_ps": 0.1,
                "score_drc": 0.4,
            },
        ]

        result = build_acquisition_plan_dataframe(
            candidate_smiles=["first", "second"],
            predictions=np.array([1.0, 2.0], dtype=np.float32),
            acquisition=acquisition,
        )

        assert result["Rank"].tolist() == [1, 2]
        assert result["Compound (SMILES)"].tolist() == ["first", "second"]
        assert result["Query type"].tolist() == ["DRC", "DRC"]
        assert result["Overall Score"].tolist() == pytest.approx([0.4, 0.4])

    def test_rejects_non_finite_predictions(self):
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )

        with pytest.raises(ValueError, match="must contain only finite values"):
            build_acquisition_plan_dataframe(
                candidate_smiles=["bad"],
                predictions=np.array([np.nan], dtype=np.float32),
                acquisition=acquisition,
            )

    def test_rejects_length_mismatch(self):
        acquisition = CostAwareGreedyAcquisition(
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            target_threshold=7.0,
            tau=0.5,
        )

        with pytest.raises(ValueError, match="must match predictions length"):
            build_acquisition_plan_dataframe(
                candidate_smiles=["a", "b"],
                predictions=np.array([1.0], dtype=np.float32),
                acquisition=acquisition,
            )
