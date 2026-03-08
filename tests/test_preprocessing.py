"""Tests for SMILESPreprocessor."""

import pytest

from moal.preprocessing import SMILESPreprocessor


@pytest.fixture
def pp():
    """Preprocessor with salt stripping enabled for use across canonicalization tests."""
    return SMILESPreprocessor(remove_salts=True)


class TestCanonicalize:
    """Tests for SMILESPreprocessor.canonicalize()."""
    def test_valid_smiles(self, pp):
        """Valid SMILES must return a non-empty canonical string, not None."""
        result = pp.canonicalize("c1ccccc1")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_smiles_returns_none(self, pp):
        """Invalid SMILES input must return None rather than raise an exception."""
        result = pp.canonicalize("NOT_A_SMILES!!!")
        assert result is None

    def test_salt_stripping(self, pp):
        # sodium benzoate: [Na+].[O-]C(=O)c1ccccc1
        """Salt stripping must remove counter-ions from multi-component SMILES, leaving only the largest fragment."""
        result = pp.canonicalize("[Na+].[O-]C(=O)c1ccccc1")
        assert result is not None
        assert "[Na+]" not in result

    def test_canonical_idempotent(self, pp):
        """Applying canonicalize twice must return the same result as applying it once."""
        smi = "CCO"
        c1 = pp.canonicalize(smi)
        c2 = pp.canonicalize(c1)
        assert c1 == c2

    def test_different_inputs_same_molecule(self, pp):
        # Both represent ethanol
        """Different SMILES representations of the same molecule must yield the same canonical form."""
        c1 = pp.canonicalize("OCC")
        c2 = pp.canonicalize("CCO")
        assert c1 == c2


class TestProcessBatch:
    """Tests for SMILESPreprocessor.process_batch()."""
    def test_valid_batch(self, pp):
        """All valid SMILES must be returned in the canonical list with no failures."""
        smiles = ["c1ccccc1", "CCO", "c1ccc(N)cc1"]
        canonical, failed = pp.process_batch(smiles)
        assert len(canonical) == 3
        assert len(failed) == 0

    def test_mixed_valid_invalid(self, pp):
        """Invalid SMILES must be reported in the failed list and not contaminate the canonical output."""
        smiles = ["c1ccccc1", "INVALID", "CCO"]
        canonical, failed = pp.process_batch(smiles)
        assert len(canonical) == 2
        assert len(failed) == 1
        assert "INVALID" in failed

    def test_empty_batch(self, pp):
        """An empty input must return two empty lists without raising."""
        canonical, failed = pp.process_batch([])
        assert canonical == []
        assert failed == []


class TestChiralityPreservation:
    """Tests that chirality information survives the canonicalization pipeline."""

    @pytest.mark.parametrize("smi", [
        "[C@@H](Br)(CC)C",  # (R)-2-bromobutane
        "[C@H](Br)(CC)C",   # (S)-2-bromobutane
    ])
    def test_chirality_preserved(self, pp, smi):
        """Both R and S stereocentres must survive canonicalization."""
        result = pp.canonicalize(smi)
        assert result is not None
        assert "@" in result  # RDKit encodes chirality with @ notation

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
