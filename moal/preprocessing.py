"""SMILES preprocessing: canonicalization and salt stripping via RDKit."""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

logger = logging.getLogger(__name__)

_REMOVER = SaltRemover()


class SMILESPreprocessor:
    """Canonicalize SMILES and strip counterions/salts using RDKit.

    All SMILES must pass through this preprocessor before being stored in a
    LabelRecord or passed to any model. This ensures consistent graph
    construction regardless of input source.
    """

    def __init__(self, remove_salts: bool = True) -> None:
        self._remove_salts = remove_salts

    def canonicalize(self, smiles: str) -> str | None:
        """Return the RDKit-canonical, salt-stripped SMILES, or None on failure."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES (could not parse): %s", smiles)
            return None
        if self._remove_salts:
            mol = _REMOVER.StripMol(mol, dontRemoveEverything=True)
        if mol is None or mol.GetNumAtoms() == 0:
            logger.warning("SMILES reduced to empty molecule after salt stripping: %s", smiles)
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def process_batch(
        self, smiles_list: list[str]
    ) -> tuple[list[str], list[str]]:
        """Canonicalize a batch of SMILES.

        Returns:
            canonical: list of successfully canonicalized SMILES (same length as
                valid entries in smiles_list).
            failed: list of original SMILES strings that could not be processed.
        """
        canonical: list[str] = []
        failed: list[str] = []
        for smi in smiles_list:
            result = self.canonicalize(smi)
            if result is None:
                failed.append(smi)
            else:
                canonical.append(result)
        if failed:
            logger.warning(
                "%d / %d SMILES failed preprocessing.", len(failed), len(smiles_list)
            )
        return canonical, failed
