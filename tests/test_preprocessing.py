"""Tests for SMILESPreprocessor."""

import pytest

from moal.preprocessing import SMILESPreprocessor


@pytest.fixture
def pp():
    return SMILESPreprocessor(remove_salts=True)


class TestCanonicalize:
    def test_valid_smiles(self, pp):
        result = pp.canonicalize("c1ccccc1")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_smiles_returns_none(self, pp):
        result = pp.canonicalize("NOT_A_SMILES!!!")
        assert result is None

    def test_salt_stripping(self, pp):
        # sodium benzoate: [Na+].[O-]C(=O)c1ccccc1
        result = pp.canonicalize("[Na+].[O-]C(=O)c1ccccc1")
        assert result is not None
        assert "[Na+]" not in result

    def test_canonical_idempotent(self, pp):
        smi = "CCO"
        c1 = pp.canonicalize(smi)
        c2 = pp.canonicalize(c1)
        assert c1 == c2

    def test_different_inputs_same_molecule(self, pp):
        # Both represent ethanol
        c1 = pp.canonicalize("OCC")
        c2 = pp.canonicalize("CCO")
        assert c1 == c2


class TestProcessBatch:
    def test_valid_batch(self, pp):
        smiles = ["c1ccccc1", "CCO", "c1ccc(N)cc1"]
        canonical, failed = pp.process_batch(smiles)
        assert len(canonical) == 3
        assert len(failed) == 0

    def test_mixed_valid_invalid(self, pp):
        smiles = ["c1ccccc1", "INVALID", "CCO"]
        canonical, failed = pp.process_batch(smiles)
        assert len(canonical) == 2
        assert len(failed) == 1
        assert "INVALID" in failed

    def test_empty_batch(self, pp):
        canonical, failed = pp.process_batch([])
        assert canonical == []
        assert failed == []


class TestChiralityPreservation:
    def test_r_chirality_preserved(self, pp):
        """(R) stereocentre must survive canonicalization."""
        # (R)-2-bromobutane
        r_smi = "[C@@H](Br)(CC)C"
        result = pp.canonicalize(r_smi)
        assert result is not None
        assert "@" in result  # RDKit encodes chirality with @ notation

    def test_s_chirality_preserved(self, pp):
        """(S) stereocentre must survive canonicalization."""
        # (S)-2-bromobutane
        s_smi = "[C@H](Br)(CC)C"
        result = pp.canonicalize(s_smi)
        assert result is not None
        assert "@" in result

    def test_r_and_s_enantiomers_are_distinct(self, pp):
        """The canonical forms of (R) and (S) enantiomers must differ."""
        r_result = pp.canonicalize("[C@@H](Br)(CC)C")
        s_result = pp.canonicalize("[C@H](Br)(CC)C")
        assert r_result is not None
        assert s_result is not None
        assert r_result != s_result

    def test_achiral_molecule_unaffected(self, pp):
        """isomericSmiles=True must not corrupt non-chiral molecules."""
        result = pp.canonicalize("c1ccccc1")
        assert result is not None
        assert "@" not in result
