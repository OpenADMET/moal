"""Tests for CostAwareOracle: cost tracking, deduplication, label dispatch."""

import pandas as pd
import pytest

from moal.oracle import CostAwareOracle
from moal.types import CensoringType, QueryType

# Minimal ground truth for testing: 6 compounds with known pEC50 values.
_GT_DATA = pd.DataFrame(
    {
        "smiles": [
            "c1ccccc1",        # benzene  — pEC50 4.0 (below ps_threshold 5.0)
            "c1ccc(N)cc1",     # aniline  — pEC50 5.5 (above ps_threshold, below 7)
            "c1ccc(O)cc1",     # phenol   — pEC50 7.5 (active)
            "CC(=O)O",         # acetic acid — pEC50 3.0
            "CCO",             # ethanol  — pEC50 6.0
            "c1ccc2ccccc2c1",  # naphthalene — pEC50 8.0 (active)
        ],
        "pec50": [4.0, 5.5, 7.5, 3.0, 6.0, 8.0],
    }
)
_PS_THRESHOLD = 5.0
_COST_PS = 1.0
_COST_DRC = 10.0


def _make_oracle(**kwargs) -> CostAwareOracle:
    defaults = dict(
        ground_truth_df=_GT_DATA,
        cost_ps=_COST_PS,
        cost_drc=_COST_DRC,
        ps_threshold=_PS_THRESHOLD,
    )
    defaults.update(kwargs)
    return CostAwareOracle(**defaults)


class TestOracleInit:
    def test_n_compounds(self):
        oracle = _make_oracle()
        assert len(oracle) == len(_GT_DATA)

    def test_n_true_actives(self):
        oracle = _make_oracle()
        assert oracle.n_true_actives(threshold=7.0) == 2  # phenol + naphthalene

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError, match="must contain columns"):
            CostAwareOracle(
                ground_truth_df=pd.DataFrame({"foo": [1]}),
                cost_ps=1.0,
                cost_drc=10.0,
                ps_threshold=5.0,
            )


class TestPrimaryScreenQueries:
    def test_below_threshold_gives_left_label(self):
        oracle = _make_oracle()
        # benzene has pEC50=4.0 < ps_threshold=5.0 → LEFT
        rec = oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert rec.censoring_type == CensoringType.LEFT
        assert rec.value == pytest.approx(_PS_THRESHOLD)
        assert rec.cost == pytest.approx(_COST_PS)

    def test_above_threshold_gives_interval_label(self):
        oracle = _make_oracle()
        # aniline has pEC50=5.5 >= ps_threshold=5.0 → INTERVAL
        rec = oracle.query("c1ccc(N)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert rec.censoring_type == CensoringType.INTERVAL
        assert rec.value == pytest.approx(_PS_THRESHOLD)
        assert rec.upper_bound == pytest.approx(11.0)  # default upper_bound

    def test_active_compound_gives_interval_label(self):
        oracle = _make_oracle()
        # naphthalene pEC50=8.0 >= ps_threshold → INTERVAL
        rec = oracle.query("c1ccc2ccccc2c1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert rec.censoring_type == CensoringType.INTERVAL


class TestDRCQueries:
    def test_exact_label(self):
        oracle = _make_oracle()
        rec = oracle.query("c1ccccc1", QueryType.DOSE_RESPONSE, iteration=0)
        assert rec.censoring_type == CensoringType.EXACT
        assert rec.value == pytest.approx(4.0)
        assert rec.cost == pytest.approx(_COST_DRC)


class TestCostTracking:
    def test_cost_accumulates(self):
        oracle = _make_oracle()
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("c1ccc(N)cc1", QueryType.DOSE_RESPONSE, iteration=0)
        assert oracle.total_cost == pytest.approx(_COST_PS + _COST_DRC)

    def test_zero_cost_before_queries(self):
        oracle = _make_oracle()
        assert oracle.total_cost == pytest.approx(0.0)


class TestDeduplication:
    def test_ps_after_ps_raises(self):
        oracle = _make_oracle()
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        with pytest.raises(ValueError, match="already has a PS label"):
            oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=1)

    def test_ps_after_drc_raises(self):
        """Downgrading from DRC to PS must be blocked; DRC supersedes PS."""
        oracle = _make_oracle()
        oracle.query("c1ccccc1", QueryType.DOSE_RESPONSE, iteration=0)
        with pytest.raises(ValueError, match="already has a DRC label"):
            oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=1)

    def test_drc_after_drc_raises(self):
        oracle = _make_oracle()
        oracle.query("c1ccccc1", QueryType.DOSE_RESPONSE, iteration=0)
        with pytest.raises(ValueError, match="already has a DRC label"):
            oracle.query("c1ccccc1", QueryType.DOSE_RESPONSE, iteration=1)

    def test_batch_dedup_within_batch(self):
        oracle = _make_oracle()
        queries = [
            ("c1ccccc1", QueryType.PRIMARY_SCREEN),
            ("c1ccccc1", QueryType.DOSE_RESPONSE),  # duplicate key in same batch
        ]
        records = oracle.query_batch(queries, iteration=0)
        assert len(records) == 1
        assert oracle.total_cost == pytest.approx(_COST_PS)


class TestUnlabeledPool:
    def test_all_unlabeled_initially(self):
        oracle = _make_oracle()
        assert len(oracle.get_unlabeled_smiles()) == len(_GT_DATA)

    def test_labeled_compound_removed_from_pool(self):
        oracle = _make_oracle()
        before = len(oracle.get_unlabeled_smiles())
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        after = len(oracle.get_unlabeled_smiles())
        assert after == before - 1


class TestPSUpgrade:
    """PS → DRC two-stage upgrade: confirm a PS hit with a full dose-response curve."""

    def test_drc_after_interval_ps_succeeds(self):
        """DRC on a compound that previously yielded an INTERVAL PS must succeed."""
        oracle = _make_oracle()
        # c1ccc(O)cc1 (phenol) has pEC50=7.5 >= ps_threshold=5.0 → INTERVAL
        ps_rec = oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert ps_rec.censoring_type == CensoringType.INTERVAL
        drc_rec = oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)
        assert drc_rec.censoring_type == CensoringType.EXACT

    def test_both_records_in_labeled_records(self):
        """After a PS→DRC upgrade, labeled_records must contain both records."""
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)
        records = oracle.labeled_records
        assert len(records) == 2
        fidelities = {r.fidelity for r in records}
        assert QueryType.PRIMARY_SCREEN in fidelities
        assert QueryType.DOSE_RESPONSE in fidelities

    def test_labeled_records_sorted_by_iteration(self):
        """labeled_records must be ordered by (iteration, PS-before-DRC) regardless of insertion order."""
        oracle = _make_oracle()
        # Query three distinct compounds across two iterations.
        oracle.query("CCO", QueryType.PRIMARY_SCREEN, iteration=0)         # iter 0 PS
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0) # iter 0 PS
        oracle.query("CCO", QueryType.DOSE_RESPONSE, iteration=1)          # iter 1 DRC (upgrade)
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)  # iter 1 DRC (upgrade)
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=2)    # iter 2 PS

        records = oracle.labeled_records
        assert len(records) == 5
        # All records must be non-decreasing in iteration.
        iterations = [r.iteration for r in records]
        assert iterations == sorted(iterations), "Records must be ordered by iteration"
        # Within iteration 1, both records are DRC (no PS in iter 1 here) — just verify
        # the two iteration-0 PS records both precede the iteration-1 DRC records.
        iter0 = [r for r in records if r.iteration == 0]
        iter1 = [r for r in records if r.iteration == 1]
        assert all(r.fidelity == QueryType.PRIMARY_SCREEN for r in iter0)
        assert all(r.fidelity == QueryType.DOSE_RESPONSE for r in iter1)

    def test_labeled_records_ps_before_drc_within_same_iteration(self):
        """When PS and DRC upgrades share the same iteration index, PS must come first."""
        oracle = _make_oracle()
        # Force both the PS and the DRC to carry iteration=0 to test intra-iteration ordering.
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=0)
        records = oracle.labeled_records
        assert len(records) == 2
        assert records[0].fidelity == QueryType.PRIMARY_SCREEN
        assert records[1].fidelity == QueryType.DOSE_RESPONSE

    def test_training_records_excludes_ps_when_drc_present(self):
        """training_records must omit the PS record for a compound that has been upgraded to DRC."""
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)
        records = oracle.training_records
        # Only the EXACT DRC record should survive.
        assert len(records) == 1
        assert records[0].fidelity == QueryType.DOSE_RESPONSE
        assert records[0].censoring_type == CensoringType.EXACT

    def test_training_records_keeps_ps_without_drc(self):
        """training_records must retain PS records for compounds that have not been upgraded."""
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        records = oracle.training_records
        assert len(records) == 1
        assert records[0].fidelity == QueryType.PRIMARY_SCREEN

    def test_training_records_identical_to_labeled_when_no_upgrades(self):
        """Without any upgrades, training_records must equal labeled_records."""
        oracle = _make_oracle()
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("CCO", QueryType.DOSE_RESPONSE, iteration=0)
        assert oracle.training_records == oracle.labeled_records

    def test_cost_accumulates_for_both_assays(self):
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)
        assert oracle.total_cost == pytest.approx(_COST_PS + _COST_DRC)

    def test_ps_labeled_smiles_initially_empty(self):
        oracle = _make_oracle()
        assert oracle.get_ps_labeled_smiles() == []

    def test_interval_ps_appears_in_ps_labeled_smiles(self):
        """An INTERVAL-censored PS record must appear in get_ps_labeled_smiles."""
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)  # INTERVAL
        assert len(oracle.get_ps_labeled_smiles()) == 1

    def test_left_ps_excluded_from_ps_labeled_smiles(self):
        """A LEFT-censored PS record (confirmed inactive) must NOT appear in get_ps_labeled_smiles."""
        oracle = _make_oracle()
        # c1ccccc1 (benzene) has pEC50=4.0 < ps_threshold=5.0 → LEFT
        ps_rec = oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert ps_rec.censoring_type == CensoringType.LEFT
        assert oracle.get_ps_labeled_smiles() == []

    def test_ps_labeled_smiles_removed_after_drc_upgrade(self):
        """After a DRC upgrade, the compound must leave the PS-labeled pool."""
        oracle = _make_oracle()
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0)
        assert len(oracle.get_ps_labeled_smiles()) == 1
        oracle.query("c1ccc(O)cc1", QueryType.DOSE_RESPONSE, iteration=1)
        assert oracle.get_ps_labeled_smiles() == []

    def test_ps_labeled_smiles_excludes_unlabeled(self):
        """get_ps_labeled_smiles must not include compounds with no label at all."""
        oracle = _make_oracle()
        assert oracle.get_ps_labeled_smiles() == []
        oracle.query("c1ccccc1", QueryType.PRIMARY_SCREEN, iteration=0)   # LEFT — excluded
        oracle.query("c1ccc(O)cc1", QueryType.PRIMARY_SCREEN, iteration=0) # INTERVAL — included
        oracle.query("CCO", QueryType.PRIMARY_SCREEN, iteration=0)          # ethanol pEC50=6.0 → INTERVAL
        assert len(oracle.get_ps_labeled_smiles()) == 2


class TestIsActive:
    def test_active_compound(self):
        oracle = _make_oracle()
        assert oracle.is_active("c1ccc(O)cc1", threshold=7.0) is True

    def test_inactive_compound(self):
        oracle = _make_oracle()
        assert oracle.is_active("c1ccccc1", threshold=7.0) is False


class TestUnknownCompound:
    def test_query_unknown_smiles_raises_key_error(self):
        """Querying a SMILES that is not in the ground truth must raise KeyError."""
        oracle = _make_oracle()
        with pytest.raises(KeyError):
            oracle.query("C1CC1", QueryType.DOSE_RESPONSE, iteration=0)

    def test_query_batch_unknown_smiles_skipped(self):
        """query_batch silently skips compounds that are not in the ground truth,
        consistent with the documented (ValueError, KeyError) catch block."""
        oracle = _make_oracle()
        queries = [
            ("c1ccccc1", QueryType.PRIMARY_SCREEN),  # valid
            ("C1CC1", QueryType.DOSE_RESPONSE),       # not in ground truth
        ]
        records = oracle.query_batch(queries, iteration=0)
        assert len(records) == 1


class TestCustomColumnNames:
    def test_custom_columns_accepted(self):
        """Oracle must work when the DataFrame uses non-default column names."""
        df = pd.DataFrame(
            {
                "compound_smiles": ["c1ccccc1", "CCO", "c1ccc(O)cc1"],
                "activity": [4.0, 6.0, 7.5],
            }
        )
        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            smiles_column="compound_smiles",
            pec50_column="activity",
        )
        assert len(oracle) == 3

    def test_custom_columns_query_works(self):
        """Querying an oracle built from custom-named columns must return correct labels."""
        df = pd.DataFrame(
            {
                "mol": ["c1ccccc1"],
                "potency": [4.0],
            }
        )
        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            smiles_column="mol",
            pec50_column="potency",
        )
        rec = oracle.query("c1ccccc1", QueryType.DOSE_RESPONSE, iteration=0)
        assert rec.value == pytest.approx(4.0)

    def test_wrong_smiles_column_raises(self):
        """Passing a smiles_column that does not exist must raise ValueError."""
        df = pd.DataFrame({"smiles": ["c1ccccc1"], "pec50": [5.0]})
        with pytest.raises(ValueError, match="must contain columns"):
            CostAwareOracle(
                ground_truth_df=df,
                cost_ps=1.0,
                cost_drc=10.0,
                ps_threshold=5.0,
                smiles_column="nonexistent",
            )

    def test_wrong_pec50_column_raises(self):
        """Passing a pec50_column that does not exist must raise ValueError."""
        df = pd.DataFrame({"smiles": ["c1ccccc1"], "pec50": [5.0]})
        with pytest.raises(ValueError, match="must contain columns"):
            CostAwareOracle(
                ground_truth_df=df,
                cost_ps=1.0,
                cost_drc=10.0,
                ps_threshold=5.0,
                pec50_column="nonexistent",
            )

    def test_error_message_names_configured_columns(self):
        """The ValueError message must name the configured column(s), not the defaults."""
        df = pd.DataFrame({"smiles": ["c1ccccc1"], "pec50": [5.0]})
        with pytest.raises(ValueError, match="my_smiles"):
            CostAwareOracle(
                ground_truth_df=df,
                cost_ps=1.0,
                cost_drc=10.0,
                ps_threshold=5.0,
                smiles_column="my_smiles",
            )


class TestIsCanonical:
    # RDKit re-encodes "OCC" to "CCO" — a reliable rewrite to verify skip behavior.
    _REWRITABLE = "OCC"
    _REWRITTEN = "CCO"

    def _make_canonical_oracle(self, **kwargs) -> CostAwareOracle:
        df = pd.DataFrame(
            {
                "smiles": [self._REWRITTEN, "c1ccc(O)cc1"],
                "pec50": [6.0, 7.5],
            }
        )
        defaults = dict(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            is_canonical=True,
        )
        defaults.update(kwargs)
        return CostAwareOracle(**defaults)

    def test_init_with_is_canonical_true_accepted(self):
        """Oracle must initialize without error when is_canonical=True."""
        oracle = self._make_canonical_oracle()
        assert len(oracle) == 2

    def test_is_canonical_true_skips_rewrite(self):
        """When is_canonical=True, keys in ground truth must equal the raw input strings."""
        oracle = self._make_canonical_oracle()
        # The raw key "CCO" must be stored as-is, not re-encoded by RDKit.
        assert self._REWRITTEN in oracle._ground_truth

    def test_is_canonical_false_rewrites_smiles(self):
        """When is_canonical=False, a non-canonical SMILES must be rewritten to its
        canonical form before storage."""
        df = pd.DataFrame({"smiles": [self._REWRITABLE], "pec50": [6.0]})
        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            is_canonical=False,
        )
        # "OCC" must have been rewritten to "CCO" at init time.
        assert self._REWRITTEN in oracle._ground_truth
        assert self._REWRITABLE not in oracle._ground_truth

    def test_query_with_is_canonical_true_returns_correct_label(self):
        """query() with is_canonical=True must return the correct label without
        calling canonicalize()."""
        oracle = self._make_canonical_oracle()
        rec = oracle.query(self._REWRITTEN, QueryType.DOSE_RESPONSE, iteration=0, is_canonical=True)
        assert rec.value == pytest.approx(6.0)

    def test_query_batch_with_is_canonical_true_works(self):
        """query_batch() with is_canonical=True must label all supplied compounds."""
        oracle = self._make_canonical_oracle()
        queries = [
            (self._REWRITTEN, QueryType.PRIMARY_SCREEN),
            ("c1ccc(O)cc1", QueryType.DOSE_RESPONSE),
        ]
        records = oracle.query_batch(queries, iteration=0, is_canonical=True)
        assert len(records) == 2

    def test_query_batch_dedup_with_is_canonical_true(self):
        """Duplicate detection in query_batch must still work when is_canonical=True."""
        oracle = self._make_canonical_oracle()
        queries = [
            (self._REWRITTEN, QueryType.PRIMARY_SCREEN),
            (self._REWRITTEN, QueryType.DOSE_RESPONSE),  # duplicate
        ]
        records = oracle.query_batch(queries, iteration=0, is_canonical=True)
        assert len(records) == 1

    def test_mismatched_canonical_flag_raises_key_error(self):
        """Querying with is_canonical=False against an oracle built with
        is_canonical=True produces a KeyError when RDKit rewrites the key."""
        # Build oracle with raw "OCC" key (is_canonical=True, no rewrite).
        df = pd.DataFrame({"smiles": [self._REWRITABLE], "pec50": [6.0]})
        oracle = CostAwareOracle(
            ground_truth_df=df,
            cost_ps=1.0,
            cost_drc=10.0,
            ps_threshold=5.0,
            is_canonical=True,
        )
        # Query with is_canonical=False: RDKit rewrites "OCC" → "CCO", which
        # is not in the ground truth, so a KeyError is expected.
        with pytest.raises(KeyError):
            oracle.query(self._REWRITABLE, QueryType.DOSE_RESPONSE, iteration=0, is_canonical=False)


class TestPec50Validation:
    def test_nan_pec50_excluded(self):
        import math
        df = pd.DataFrame({"smiles": ["c1ccccc1", "CCO"], "pec50": [float("nan"), 5.0]})
        oracle = CostAwareOracle(
            ground_truth_df=df, cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0
        )
        # Only CCO should survive.
        assert len(oracle) == 1
        # The surviving compound must have a finite stored value.
        assert all(math.isfinite(v) for v in oracle._ground_truth.values())

    def test_inf_pec50_excluded(self):
        """Positive and negative infinity must be excluded."""
        df = pd.DataFrame({
            "smiles": ["c1ccccc1", "CCO", "c1ccc(N)cc1"],
            "pec50": [float("inf"), float("-inf"), 5.0],
        })
        oracle = CostAwareOracle(
            ground_truth_df=df, cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0
        )
        assert len(oracle) == 1  # only aniline

    def test_out_of_range_pec50_excluded(self):
        """pEC50 outside [0, 14] (e.g., -50 or 999) must be excluded."""
        df = pd.DataFrame({
            "smiles": ["c1ccccc1", "CCO", "c1ccc(N)cc1"],
            "pec50": [-50.0, 999.0, 7.0],
        })
        oracle = CostAwareOracle(
            ground_truth_df=df, cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0
        )
        assert len(oracle) == 1  # only aniline with pEC50=7.0

    def test_boundary_values_accepted(self):
        """Boundary values 0.0 and 14.0 are physically plausible and must be kept."""
        df = pd.DataFrame({
            "smiles": ["c1ccccc1", "CCO"],
            "pec50": [0.0, 14.0],
        })
        oracle = CostAwareOracle(
            ground_truth_df=df, cost_ps=1.0, cost_drc=10.0, ps_threshold=5.0
        )
        assert len(oracle) == 2

    def test_valid_pec50_range_all_kept(self):
        """All 6 fixtures have valid pEC50 values — none should be excluded."""
        oracle = _make_oracle()
        assert len(oracle) == len(_GT_DATA)
