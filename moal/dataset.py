"""Mixed-fidelity dataset and PyTorch Lightning DataModule."""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
from torch.utils.data import DataLoader, Dataset, random_split

from moal.types import LabelRecord

logger = logging.getLogger(__name__)


class MixedFidelityDataset(Dataset):
    """Thin PyTorch Dataset wrapping a list of LabelRecords.

    Each item is a (MoleculeDatapoint, LabelRecord) pair. The MolGraph is built
    lazily and cached on first access.

    Parameters
    ----------
    records : list[LabelRecord]
        Labeled observations from the oracle.
    """

    def __init__(
        self,
        records: list[LabelRecord],
    ) -> None:
        self.records = records
        self._mol_graphs = MoleculeDataset(
            [MoleculeDatapoint.from_smi(r.canonical_smiles) for r in self.records]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Any, LabelRecord]:
        return self._mol_graphs[idx], self.records[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[Any, LabelRecord]],
    ) -> tuple[Any, list[LabelRecord]]:
        """Collate a list of (MoleculeDatapoint, LabelRecord) into a batch.

        Returns
        -------
        tuple[BatchMolGraph, list[LabelRecord]]
            Batched molecular graph and corresponding label records.
        """
        # Unzip the batch into separate tuples.
        datapoints, records = zip(*batch)

        # Extract the underlying MolGraph from each datapoint to build the batch.
        bmg = BatchMolGraph([dp.mg for dp in datapoints])

        return bmg, list(records)


class MixedFidelityDataModule(L.LightningDataModule):
    """LightningDataModule for mixed-fidelity labeled data.

    Parameters
    ----------
    records : list[LabelRecord]
        All labeled observations (train + val pool).
    batch_size : int, optional
        Number of samples per mini-batch. Default is 64.
    val_fraction : float, optional
        Fraction of records held out for validation. Default is 0.1.
    num_workers : int, optional
        DataLoader worker count (0 = main process). Default is 0.
    seed : int, optional
        Random seed for the train/val split. Default is 42.
    """

    def __init__(
        self,
        records: list[LabelRecord],
        batch_size: int = 64,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.records = records
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataset: MixedFidelityDataset | None = None
        self._val_dataset: MixedFidelityDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        n_val = max(1, int(len(self.records) * self.val_fraction))
        n_train = len(self.records) - n_val
        if n_train <= 0:
            logger.warning(
                "Too few records (%d) for a val split; using all for training.",
                len(self.records),
            )
            n_train, n_val = len(self.records), 0

        full = MixedFidelityDataset(self.records)
        if n_val > 0:
            train_subset, val_subset = random_split(
                full,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )
            # Wrap subsets so we can access .records for collate
            self._train_dataset = _SubsetWrapper(train_subset, full)
            self._val_dataset = _SubsetWrapper(val_subset, full)
        else:
            self._train_dataset = full
            self._val_dataset = None

    def transfer_batch_to_device(
        self,
        batch: tuple[Any, list[LabelRecord]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[Any, list[LabelRecord]]:
        """Move only the BatchMolGraph to the device; leave LabelRecords on CPU.

        Lightning's default ``apply_to_collection`` recurses into dataclasses
        and fails on frozen ones. We bypass that by handling the transfer
        manually for our (BatchMolGraph, list[LabelRecord]) batch shape.
        """
        mol_graph, records = batch
        mol_graph = super().transfer_batch_to_device(mol_graph, device, dataloader_idx)
        return mol_graph, records

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=MixedFidelityDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=MixedFidelityDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class _SubsetWrapper(Dataset):
    """Thin wrapper that preserves ``MixedFidelityDataset`` collate semantics
    when working with a ``random_split`` Subset.

    Parameters
    ----------
    subset : Subset
        A PyTorch ``random_split`` subset referencing the full dataset.
    full_dataset : MixedFidelityDataset
        The underlying dataset the subset was derived from.
    """

    def __init__(self, subset: Any, full_dataset: MixedFidelityDataset) -> None:
        self._subset = subset
        self._full = full_dataset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> tuple[Any, LabelRecord]:
        return self._subset[idx]
