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
    ``LabelRecord`` or passed to any model. This ensures consistent graph
    construction regardless of input source.

    Parameters
    ----------
    remove_salts : bool, optional
        When ``True`` (default), strip counterions and salts via RDKit's
        ``SaltRemover`` before canonicalization.
    """

    def __init__(self, remove_salts: bool = True) -> None:
        self._remove_salts = remove_salts

    def canonicalize(self, smiles: str) -> str | None:
        """Return the RDKit-canonical, salt-stripped SMILES, or ``None`` on failure.

        Parameters
        ----------
        smiles : str
            Input SMILES string (may be non-canonical or salt-containing).

        Returns
        -------
        str or None
            Canonical SMILES string, or ``None`` if the molecule could not be
            parsed or was reduced to an empty structure after salt stripping.

        Notes
        -----
        Canonicalization uses ``isomericSmiles=True`` to preserve
        stereochemistry. This is set explicitly because RDKit's default
        behavior has changed across versions and chirality must be retained.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Invalid SMILES (could not parse): %s", smiles)
            return None
        if self._remove_salts:
            mol = _REMOVER.StripMol(mol, dontRemoveEverything=True)
        if mol is None or mol.GetNumAtoms() == 0:
            logger.warning(
                "SMILES reduced to empty molecule after salt stripping: %s", smiles
            )
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def process_batch(self, smiles_list: list[str]) -> tuple[list[str], list[str]]:
        """Canonicalize a batch of SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
            Input SMILES strings to process.

        Returns
        -------
        canonical : list of str
            Successfully canonicalized SMILES. Length equals
            ``len(smiles_list) - len(failed)``.
        failed : list of str
            Original SMILES strings that could not be canonicalized (unparseable
            or reduced to an empty structure after salt stripping).
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
