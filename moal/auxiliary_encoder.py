"""Auxiliary encoder pretrained on primary-screen readouts (log2FC, pIC50, etc.).

Phase 1 of issue #36 (``moal plan``-only; excluded from ``moal simulate`` to
avoid an acquisition-endogeneity problem in the live active-learning loop).
Trains a small ChemProp encoder via masked multi-task regression over
``LabelRecord.raw_ps_readouts``, sharing the main model's backbone
construction (:func:`moal.model.build_mpnn`) rather than a bespoke
architecture, so its embeddings live in the same representation space as the
main pEC50 model. Readouts are used as-is: no per-plate/per-batch
normalization is applied (see :class:`~moal.config.AuxiliaryEncoderConfig`
for why).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
from chemprop.data.dataloader import build_dataloader
from chemprop.models import MPNN
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from moal.config import AuxiliaryEncoderConfig
from moal.model import build_mpnn
from moal.types import LabelRecord

logger = logging.getLogger(__name__)


def masked_mse_loss(preds: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error computed over only the masked (observed) task entries.

    Parameters
    ----------
    preds : Tensor
        Shape ``(batch, n_tasks)`` model predictions.
    targets : Tensor
        Shape ``(batch, n_tasks)`` targets. Values at positions where
        ``mask`` is False are ignored and may hold arbitrary placeholder
        values.
    mask : Tensor
        Boolean tensor of shape ``(batch, n_tasks)``; True where a compound
        had an observed readout for that task.

    Returns
    -------
    Tensor
        Scalar masked MSE, differentiable with respect to ``preds``. When
        ``mask`` has no True entries (e.g. a batch with no observed readouts
        for any task), returns a zero-valued tensor still connected to
        ``preds`` so the training step remains well-defined rather than
        raising a division-by-zero.
    """
    mask_f = mask.to(preds.dtype)
    denom = mask_f.sum()
    if denom == 0:
        return preds.sum() * 0.0
    return ((preds - targets) ** 2 * mask_f).sum() / denom


class _AuxiliaryDataset(Dataset):
    """Dataset pairing a molecular graph with a masked multi-task target vector.

    Parameters
    ----------
    records : list[LabelRecord]
        Records with a non-empty ``raw_ps_readouts``. Records lacking any
        readout carry no training signal and should be filtered out by the
        caller before construction.
    task_names : list[str]
        Fixed, ordered list of readout keys; determines target/mask column
        order and the auxiliary encoder's output dimensionality.
    """

    def __init__(self, records: list[LabelRecord], task_names: list[str]) -> None:
        self.records = records
        self.task_names = task_names
        self._mol_graphs = MoleculeDataset(
            [MoleculeDatapoint.from_smi(r.canonical_smiles) for r in records]  # pyright: ignore[reportArgumentType]
        )
        self._targets = torch.zeros(len(records), len(task_names), dtype=torch.float32)
        self._mask = torch.zeros(len(records), len(task_names), dtype=torch.bool)
        for i, rec in enumerate(records):
            for j, name in enumerate(task_names):
                if name in rec.raw_ps_readouts:
                    self._targets[i, j] = rec.raw_ps_readouts[name]
                    self._mask[i, j] = True

    def __len__(self) -> int:
        """Return the number of records in the dataset.

        Returns
        -------
        int
            Total number of readout-bearing records.
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Tensor]:
        """Return the (datapoint, target row, mask row) triple at ``idx``.

        Parameters
        ----------
        idx : int
            Zero-based index into the dataset.

        Returns
        -------
        tuple[Any, Tensor, Tensor]
            A ``(MoleculeDatapoint, targets, mask)`` triple; ``targets`` and
            ``mask`` each have shape ``(n_tasks,)``.
        """
        return self._mol_graphs[idx], self._targets[idx], self._mask[idx]

    @staticmethod
    def collate_fn(batch: list[tuple[Any, Tensor, Tensor]]) -> tuple[Any, Tensor, Tensor]:
        """Collate a list of (datapoint, target row, mask row) into a batch.

        Parameters
        ----------
        batch : list[tuple[Any, Tensor, Tensor]]
            Items as returned by :meth:`__getitem__`.

        Returns
        -------
        tuple[BatchMolGraph, Tensor, Tensor]
            Batched molecular graph, stacked targets ``(batch, n_tasks)``,
            and stacked mask ``(batch, n_tasks)``.
        """
        datapoints, targets, masks = zip(*batch, strict=False)
        bmg = BatchMolGraph([dp.mg for dp in datapoints])
        return bmg, torch.stack(list(targets)), torch.stack(list(masks))


class AuxiliaryDataModule(L.LightningDataModule):
    """LightningDataModule for masked multi-task auxiliary-encoder pretraining.

    Parameters
    ----------
    records : list[LabelRecord]
        Records with a non-empty ``raw_ps_readouts``.
    task_names : list[str]
        Fixed, ordered list of readout keys.
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
        task_names: list[str],
        batch_size: int = 64,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.records = records
        self.task_names = task_names
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create the train and validation dataset splits.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage identifier; unused, accepted for interface
            compatibility.
        """
        n_val = max(1, int(len(self.records) * self.val_fraction))
        n_train = len(self.records) - n_val
        if n_train <= 0:
            logger.warning(
                "Too few readout-bearing records (%d) for a val split; using all for training.",
                len(self.records),
            )
            n_train, n_val = len(self.records), 0

        full = _AuxiliaryDataset(self.records, self.task_names)
        if n_val > 0:
            self._train_dataset, self._val_dataset = random_split(
                full,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            self._train_dataset = full
            self._val_dataset = None

    def transfer_batch_to_device(
        self, batch: tuple[Any, Tensor, Tensor], device: torch.device, dataloader_idx: int
    ) -> tuple[Any, Tensor, Tensor]:
        """Move the batched mol graph and target/mask tensors to ``device``.

        Parameters
        ----------
        batch : tuple[Any, Tensor, Tensor]
            A ``(BatchMolGraph, targets, mask)`` triple.
        device : torch.device
            Target device.
        dataloader_idx : int
            Index of the dataloader (required by the Lightning interface).

        Returns
        -------
        tuple[Any, Tensor, Tensor]
            The same triple moved to ``device``.
        """
        mol_graph, targets, mask = batch
        mol_graph = super().transfer_batch_to_device(mol_graph, device, dataloader_idx)
        return mol_graph, targets.to(device), mask.to(device)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Returns
        -------
        DataLoader
            Shuffled DataLoader over the training split.
        """
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_AuxiliaryDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return the validation DataLoader, or ``None`` when no val split exists.

        Returns
        -------
        DataLoader or None
            Non-shuffled DataLoader over the validation split, or ``None``.
        """
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_AuxiliaryDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class AuxiliaryEncoderModule(L.LightningModule):
    """ChemProp MPNN trained via masked multi-task regression on auxiliary readouts.

    Parameters
    ----------
    task_names : list[str]
        Fixed, ordered list of readout keys the model was (or will be)
        trained against; determines the predictor head's output width.
    config : AuxiliaryEncoderConfig
        Backbone architecture, freeze schedule, and optimization
        hyperparameters.
    """

    def __init__(self, task_names: list[str], config: AuxiliaryEncoderConfig) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("task_names must be non-empty.")
        self.task_names = list(task_names)
        self._config = config
        self._encoder_frozen = True
        self.model = build_mpnn(
            from_foundation=config.from_foundation,
            ffn_hidden_dim=config.ffn_hidden_dim,
            ffn_num_layers=config.ffn_num_layers,
            message_hidden_dim=config.message_hidden_dim,
            depth=config.depth,
            n_tasks=len(self.task_names),
        )
        self._freeze_encoder()

    @property
    def embedding_dim(self) -> int:
        """Width of the backbone's pooled structural embedding.

        Returns
        -------
        int
            The message-passing encoder's native output width (CheMeleon's
            fixed width, or ``config.message_hidden_dim`` for a random-init
            encoder) — the row width :meth:`embed_smiles` returns, and the
            embedding block width the concatenation architecture (Phase 2)
            needs from :func:`moal.concatenation_model.concatenation_feature_dim`.
        """
        return cast(int, cast(Any, self.model).message_passing.output_dim)

    # ------------------------------------------------------------------
    # Freeze / unfreeze schedule
    # ------------------------------------------------------------------

    def _encoder_params(self) -> list[nn.Parameter]:
        """Return the trainable parameters of the message-passing encoder.

        Returns
        -------
        list[nn.Parameter]
            Parameters belonging to ``self.model.message_passing``.
        """
        return list(cast(nn.Module, self.model.message_passing).parameters())

    def _head_params(self) -> list[nn.Parameter]:
        """Return the trainable parameters of the aggregation layer and FFN head.

        Returns
        -------
        list[nn.Parameter]
            Parameters belonging to ``self.model.agg`` and
            ``self.model.predictor``, concatenated in that order.
        """
        return list(cast(nn.Module, self.model.agg).parameters()) + list(
            cast(nn.Module, self.model.predictor).parameters()
        )

    def _freeze_encoder(self) -> None:
        """Freeze all message-passing encoder parameters."""
        for p in self._encoder_params():
            p.requires_grad_(False)
        self._encoder_frozen = True

    def _unfreeze_encoder(self) -> None:
        """Unfreeze the message-passing encoder after the warm-up phase."""
        for p in self._encoder_params():
            p.requires_grad_(True)
        self._encoder_frozen = False

    def on_train_epoch_start(self) -> None:
        """Lightning hook: unfreeze the encoder once warm-up is complete."""
        if self._encoder_frozen and self.current_epoch >= self._config.freeze_epochs:
            self._unfreeze_encoder()
            self.trainer.strategy.setup_optimizers(self.trainer)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def forward(self, batch_mol_graph: Any) -> Tensor:
        """Run a forward pass and return multi-task predictions.

        Parameters
        ----------
        batch_mol_graph : Any
            A batched molecular graph (``chemprop.data.BatchMolGraph``).

        Returns
        -------
        Tensor
            Shape ``(batch, n_tasks)`` predictions.
        """
        return cast(Tensor, self.model(batch_mol_graph))

    def training_step(self, batch: tuple[Any, Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log the masked multi-task training loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, Tensor, Tensor]
            A ``(mol_graph, targets, mask)`` triple.
        batch_idx : int
            Index of the batch within the current epoch (unused).

        Returns
        -------
        Tensor
            Scalar training loss used for the backward pass.
        """
        mol_graph, targets, mask = batch
        preds = self(mol_graph)
        loss = masked_mse_loss(preds, targets, mask)
        self.log("aux_train_loss", loss, prog_bar=True, batch_size=targets.shape[0])
        return loss

    def validation_step(self, batch: tuple[Any, Tensor, Tensor], batch_idx: int) -> None:
        """Compute and log the masked multi-task validation loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, Tensor, Tensor]
            A ``(mol_graph, targets, mask)`` triple.
        batch_idx : int
            Index of the batch within the current validation epoch (unused).
        """
        mol_graph, targets, mask = batch
        preds = self(mol_graph)
        loss = masked_mse_loss(preds, targets, mask)
        self.log("aux_val_loss", loss, prog_bar=True, batch_size=targets.shape[0])

    def configure_optimizers(self) -> Adam:
        """Build and return the Adam optimizer for the current freeze state.

        Returns
        -------
        Adam
            When the encoder is frozen, a single-group Adam optimizer for the
            multi-task head at ``config.lr``. After the encoder is unfrozen,
            a second param group for the encoder is added at the same
            ``config.lr`` (no discriminative rate split, unlike the main
            model's ``mpnn_lr`` / ``ffn_lr``).
        """
        param_groups = [
            {
                "params": self._head_params(),
                "lr": self._config.lr,
                "weight_decay": self._config.weight_decay,
            }
        ]
        if not self._encoder_frozen:
            param_groups.append(
                {
                    "params": self._encoder_params(),
                    "lr": self._config.lr,
                    "weight_decay": self._config.weight_decay,
                }
            )
        return Adam(param_groups)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_smiles(self, smiles_list: list[str], batch_size: int = 256) -> np.ndarray:
        """Return pooled structural embeddings (pre-predictor) for a list of SMILES.

        Used by the concatenation architecture (Phase 2) to supply a
        structural fallback for compounds with no observed auxiliary
        readout. Uses ``chemprop.models.MPNN.fingerprint``, which applies
        message-passing, mean pooling, and batch-norm but stops short of the
        multi-task predictor head.

        Parameters
        ----------
        smiles_list : list[str]
            **Must be RDKit-canonical, salt-stripped SMILES**, matching
            :meth:`moal.model.ChemPropLightningModule.predict_smiles`'s
            contract.
        batch_size : int, optional
            Number of molecules processed per forward pass. Default is 256.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, embedding_dim)``, aligned with
            ``smiles_list``. ``embedding_dim`` is the backbone's native
            output width (CheMeleon's fixed width, or ``message_hidden_dim``
            for a random-init encoder), not
            ``AuxiliaryEncoderConfig.embedding_dim``.
        """
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles_list])  # pyright: ignore[reportArgumentType]
        dataloader = build_dataloader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        all_embeddings = []
        with torch.inference_mode():
            for batch in dataloader:
                batch.bmg.to(self.device)
                embedding = cast(MPNN, self.model).fingerprint(batch.bmg)
                all_embeddings.append(embedding.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def pretrain_auxiliary_encoder(
    records: list[LabelRecord],
    config: AuxiliaryEncoderConfig,
    trainer_kwargs: dict[str, Any] | None = None,
    datamodule_kwargs: dict[str, Any] | None = None,
) -> AuxiliaryEncoderModule:
    """Pretrain (or load) the auxiliary encoder from a campaign's labeled records.

    When ``config.checkpoint_path`` is set, pretraining is skipped entirely
    and the checkpoint is loaded instead — the explicit opt-in override for
    cases where retraining every ``moal plan`` invocation is too expensive.
    Otherwise the encoder is trained from scratch on every call, using
    whichever ``raw_ps_readouts`` exist in ``records`` at that moment.

    Parameters
    ----------
    records : list[LabelRecord]
        All labeled records from the campaign state. Records with an empty
        ``raw_ps_readouts`` are filtered out before training; they carry no
        auxiliary training signal.
    config : AuxiliaryEncoderConfig
        Backbone, freeze schedule, and optimization hyperparameters.
    trainer_kwargs : dict[str, Any], optional
        Additional keyword arguments forwarded to ``lightning.Trainer``.
        Ignored when loading from ``config.checkpoint_path``.
    datamodule_kwargs : dict[str, Any], optional
        Passed to :class:`AuxiliaryDataModule` (e.g. ``val_fraction``,
        ``seed``). Ignored when loading from ``config.checkpoint_path``.

    Returns
    -------
    AuxiliaryEncoderModule
        The trained (or loaded) auxiliary encoder.

    Raises
    ------
    ValueError
        If no record in ``records`` carries any auxiliary readout and
        ``config.checkpoint_path`` is unset.
    """
    if config.checkpoint_path is not None:
        return load_auxiliary_encoder_checkpoint(config.checkpoint_path, config)

    readout_records = [rec for rec in records if rec.raw_ps_readouts]
    if not readout_records:
        raise ValueError(
            "No records carry raw_ps_readouts; cannot pretrain the auxiliary encoder. "
            "Set config.checkpoint_path to load a cached checkpoint instead."
        )

    task_names = sorted({key for rec in readout_records for key in rec.raw_ps_readouts})
    logger.info(
        "Pretraining auxiliary encoder on %d readout-bearing record(s), tasks=%s",
        len(readout_records),
        task_names,
    )

    module = AuxiliaryEncoderModule(task_names=task_names, config=config)
    dm = AuxiliaryDataModule(readout_records, task_names, **(datamodule_kwargs or {}))
    dm.setup()

    kwargs: dict[str, Any] = {
        "max_epochs": config.max_epochs,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if trainer_kwargs:
        kwargs.update(trainer_kwargs)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    trainer = L.Trainer(**kwargs)
    trainer.fit(module, datamodule=dm)
    return module


def save_auxiliary_encoder_checkpoint(module: AuxiliaryEncoderModule, path: str | Path) -> None:
    """Save an auxiliary encoder to a checkpoint usable by ``config.checkpoint_path``.

    Parameters
    ----------
    module : AuxiliaryEncoderModule
        A trained auxiliary encoder.
    path : str or Path
        Destination file path.
    """
    torch.save({"task_names": module.task_names, "state_dict": module.state_dict()}, path)


def load_auxiliary_encoder_checkpoint(
    path: str | Path, config: AuxiliaryEncoderConfig
) -> AuxiliaryEncoderModule:
    """Load an auxiliary encoder checkpoint written by :func:`save_auxiliary_encoder_checkpoint`.

    Parameters
    ----------
    path : str or Path
        Path to the checkpoint file.
    config : AuxiliaryEncoderConfig
        Backbone architecture the checkpoint was trained with; must match
        the checkpoint's own architecture (``from_foundation``,
        ``ffn_hidden_dim``, etc.) or ``load_state_dict`` will raise.

    Returns
    -------
    AuxiliaryEncoderModule
        The restored auxiliary encoder, with the encoder still frozen
        (caller-visible state, not resumed training state).
    """
    logger.info("Loading cached auxiliary encoder checkpoint from %s (retraining skipped).", path)
    ckpt = cast(dict[str, Any], torch.load(path, weights_only=True))
    module = AuxiliaryEncoderModule(task_names=ckpt["task_names"], config=config)
    module.load_state_dict(ckpt["state_dict"])
    return module
