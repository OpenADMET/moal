"""Concatenation architecture for the auxiliary log2FC/pIC50 signal (#36 Phase 2).

Concatenates, per compound, either its own observed auxiliary readouts (when
PS-screened) or the pretrained :class:`~moal.auxiliary_encoder.AuxiliaryEncoderModule`'s
structural embedding (when never PS-screened), plus a provenance flag
distinguishing the two, onto the pooled graph embedding before the pEC50
predictor head. Graph-only prediction is the unconditional fallback: a
compound with neither an observed readout nor (obviously) a missing
embedding never occurs, since the embedding path always has a fallback
value.

Reuses chemprop's native ``MPNN.forward(bmg, X_d=...)`` concatenation point
(see :func:`moal.model.build_mpnn`'s ``extra_input_dim``) rather than a
bespoke predictor wrapper, and the same ``CensoredRegressionLoss``,
freeze/unfreeze schedule, and refit contract as
:class:`moal.model.ChemPropLightningModule`, so this is a second,
coexisting model path selectable per run rather than a replacement.

``moal plan``-only, matching :mod:`moal.auxiliary_encoder`.
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
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from moal.auxiliary_encoder import AuxiliaryEncoderModule
from moal.loss import CensoredRegressionLoss
from moal.model import _validate_from_foundation, build_mpnn
from moal.planning import normalize_record_weights
from moal.types import LabelRecord, QueryType

logger = logging.getLogger(__name__)


def concatenation_feature_dim(n_tasks: int, embedding_dim: int) -> int:
    """Return the width of the concatenation feature vector.

    Parameters
    ----------
    n_tasks : int
        Number of distinct auxiliary readout keys (``AuxiliaryEncoderModule.task_names``).
    embedding_dim : int
        Width of the auxiliary encoder's structural embedding (its
        backbone's native output width; see
        :meth:`~moal.auxiliary_encoder.AuxiliaryEncoderModule.embed_smiles`).

    Returns
    -------
    int
        ``2 * n_tasks + embedding_dim + 1``: observed-readout vector,
        readout mask, structural embedding, and a single provenance flag.
    """
    return 2 * n_tasks + embedding_dim + 1


def build_concatenation_features(
    canonical_smiles: list[str],
    readouts: list[dict[str, float]],
    aux_encoder: AuxiliaryEncoderModule,
    *,
    use_observed_readout: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """Build the per-compound concatenation feature matrix.

    The auxiliary encoder's structural embedding is always computed and
    included, for every compound, regardless of whether it has an observed
    readout — this keeps the main model's embedding-consuming input pathway
    exercised by every training row, so training and inference always share
    the same input distribution shape (a compound with an observed readout
    at training time looks input-wise like any other compound, differing
    only in whether its readout block is also populated).

    When ``use_observed_readout`` is True and ``readouts[i]`` is non-empty,
    the observed-readout block is additionally populated (per-task values
    where present, zero elsewhere), the mask block marks which tasks were
    observed, and the readout-used flag is 1. Otherwise the readout and mask
    blocks stay zero and the flag is 0 — the compound is scored from its
    structural embedding alone.

    Parameters
    ----------
    canonical_smiles : list[str]
        RDKit-canonical SMILES, one per compound.
    readouts : list[dict[str, float]]
        Per-compound ``LabelRecord.raw_ps_readouts``-shaped dict, aligned
        with ``canonical_smiles``. An empty dict means "never PS-screened".
    aux_encoder : AuxiliaryEncoderModule
        Pretrained auxiliary encoder; supplies both ``task_names`` (readout
        key order) and the structural embedding.
    use_observed_readout : bool, optional
        When True (default), compounds with a non-empty readout also get
        their raw value concatenated alongside the embedding. When False,
        the readout and mask blocks are always zero and every compound is
        scored from its embedding alone, matching
        ``AuxiliaryModelConfig.use_observed_readout``.
    batch_size : int, optional
        Batch size for the embedding forward pass. Default is 256.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, concatenation_feature_dim(...))``, aligned with
        ``canonical_smiles``.

    Raises
    ------
    ValueError
        If ``len(canonical_smiles) != len(readouts)``.
    """
    if len(canonical_smiles) != len(readouts):
        raise ValueError(
            f"canonical_smiles ({len(canonical_smiles)}) and readouts ({len(readouts)}) "
            "must be the same length."
        )

    task_names = aux_encoder.task_names
    n_tasks = len(task_names)
    n = len(canonical_smiles)

    readout_vec = np.zeros((n, n_tasks), dtype=np.float32)
    readout_mask = np.zeros((n, n_tasks), dtype=np.float32)
    readout_used = np.zeros((n, 1), dtype=np.float32)

    if use_observed_readout:
        for i, readout in enumerate(readouts):
            if not readout:
                continue
            for j, name in enumerate(task_names):
                if name in readout:
                    readout_vec[i, j] = readout[name]
                    readout_mask[i, j] = 1.0
            readout_used[i, 0] = 1.0

    embeddings = aux_encoder.embed_smiles(canonical_smiles, batch_size=batch_size)

    return np.concatenate([readout_vec, readout_mask, embeddings, readout_used], axis=1)


class _ConcatenatedDataset(Dataset):
    """Dataset pairing a molecular graph and LabelRecord with a precomputed feature row.

    Parameters
    ----------
    records : list[LabelRecord]
        Labeled observations.
    features : np.ndarray
        Precomputed concatenation features, shape ``(len(records), feature_dim)``,
        aligned with ``records`` (typically from :func:`build_concatenation_features`).
    """

    def __init__(self, records: list[LabelRecord], features: np.ndarray) -> None:
        if len(records) != len(features):
            raise ValueError(
                f"records ({len(records)}) and features ({len(features)}) must be the same length."
            )
        self.records = records
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._mol_graphs = MoleculeDataset(
            [MoleculeDatapoint.from_smi(r.canonical_smiles) for r in records]  # pyright: ignore[reportArgumentType]
        )

    def __len__(self) -> int:
        """Return the number of records in the dataset.

        Returns
        -------
        int
            Total number of labeled observations.
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, LabelRecord]:
        """Return the (datapoint, feature row, LabelRecord) triple at ``idx``.

        Parameters
        ----------
        idx : int
            Zero-based index into the dataset.

        Returns
        -------
        tuple[Any, Tensor, LabelRecord]
            A ``(MoleculeDatapoint, feature row, LabelRecord)`` triple.
        """
        return self._mol_graphs[idx], self._features[idx], self.records[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[Any, Tensor, LabelRecord]],
    ) -> tuple[Any, Tensor, list[LabelRecord]]:
        """Collate a list of (datapoint, feature row, LabelRecord) into a batch.

        Parameters
        ----------
        batch : list[tuple[Any, Tensor, LabelRecord]]
            Items as returned by :meth:`__getitem__`.

        Returns
        -------
        tuple[BatchMolGraph, Tensor, list[LabelRecord]]
            Batched molecular graph, stacked feature matrix
            ``(batch, feature_dim)``, and corresponding label records.
        """
        datapoints, features, records = zip(*batch, strict=False)
        bmg = BatchMolGraph([dp.mg for dp in datapoints])
        return bmg, torch.stack(list(features)), list(records)


class ConcatenationChemPropLightningModule(L.LightningModule):
    """ChemProp MPNN with a concatenated auxiliary-signal input before the pEC50 head.

    Parameters mirror :class:`moal.model.ChemPropLightningModule` exactly,
    plus ``concat_feature_dim``; see that class for the shared parameters'
    documentation.

    Parameters
    ----------
    concat_feature_dim : int
        Width of the concatenation feature vector (see
        :func:`concatenation_feature_dim`); determines the predictor head's
        input width alongside the backbone's own pooled-embedding width.
    use_observed_readout : bool, optional
        Fixed at construction and used by both :meth:`refit` and
        :meth:`predict_smiles` — this determines the input distribution the
        model's weights are actually fit against (e.g. when False, the
        readout/mask input dimensions are always exactly zero throughout
        training, so their weights never receive gradient signal; feeding
        them nonzero values at inference would exercise untrained weights).
        Deliberately not a per-call parameter on either method, so
        training-time and inference-time routing cannot drift apart. Default
        is True.
    ffn_hidden_dim, ffn_num_layers, message_hidden_dim, depth, freeze_epochs,
    mpnn_lr, ffn_lr, mpnn_weight_decay, ffn_weight_decay, sigma,
    learnable_sigma, from_foundation
        See :class:`moal.model.ChemPropLightningModule`. Note ``w_drc``/``w_ps``
        are deliberately absent here: :meth:`refit` requires every record to
        be DOSE_RESPONSE (see below), so the DRC-vs-PS fidelity weighting
        that parameter pair controls in
        :class:`moal.model.ChemPropLightningModule` has nothing to
        differentiate in this class and would be a redundant, no-op scalar
        confounded with ``sigma``.
    """

    def __init__(
        self,
        concat_feature_dim: int,
        use_observed_readout: bool = True,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 2,
        message_hidden_dim: int = 300,
        depth: int = 3,
        freeze_epochs: int = 10,
        mpnn_lr: float = 1e-5,
        ffn_lr: float = 1e-3,
        mpnn_weight_decay: float = 0.0,
        ffn_weight_decay: float = 0.0,
        sigma: float = 0.5,
        learnable_sigma: bool = False,
        from_foundation: str | bool = "chemeleon",
    ) -> None:
        super().__init__()
        _validate_from_foundation(from_foundation)
        self._from_foundation = from_foundation
        self.concat_feature_dim = concat_feature_dim
        self.use_observed_readout = use_observed_readout
        self.save_hyperparameters()

        self.freeze_epochs = freeze_epochs
        self.mpnn_lr = mpnn_lr
        self.ffn_lr = ffn_lr
        self.mpnn_weight_decay = mpnn_weight_decay
        self.ffn_weight_decay = ffn_weight_decay
        self._encoder_frozen = True

        self._epoch_losses: dict[str, list[Tensor]] = {
            "train_drc": [],
            "train_ps": [],
            "val_drc": [],
            "val_ps": [],
        }

        # w_drc/w_ps are fixed equal (not exposed as parameters): refit()
        # requires every record to be DOSE_RESPONSE, so the PS branch never
        # fires and the two weights would otherwise be a redundant, no-op
        # scalar confounded with sigma.
        self.loss_fn = CensoredRegressionLoss(
            sigma=sigma, w_drc=1.0, w_ps=1.0, learnable_sigma=learnable_sigma
        )

        self.model = build_mpnn(
            from_foundation=from_foundation,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            message_hidden_dim=message_hidden_dim,
            depth=depth,
            n_tasks=1,
            extra_input_dim=concat_feature_dim,
        )
        self._freeze_encoder()

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
        if self._encoder_frozen and self.current_epoch >= self.freeze_epochs:
            self._unfreeze_encoder()
            self.trainer.strategy.setup_optimizers(self.trainer)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def forward(self, batch_mol_graph: Any, x_d: Tensor) -> Tensor:
        """Run a forward pass and return scalar pEC50 predictions.

        Parameters
        ----------
        batch_mol_graph : Any
            A batched molecular graph (``chemprop.data.BatchMolGraph``).
        x_d : Tensor
            Concatenation features, shape ``(batch, concat_feature_dim)``.

        Returns
        -------
        Tensor
            1-D tensor of shape ``(N,)`` with predicted pEC50 values.
        """
        return cast(Tensor, self.model(batch_mol_graph, X_d=x_d).squeeze(-1))

    def training_step(self, batch: tuple[Any, Tensor, list[LabelRecord]], batch_idx: int) -> Tensor:
        """Compute and log the training loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, Tensor, list[LabelRecord]]
            A ``(mol_graph, x_d, records)`` triple.
        batch_idx : int
            Index of the batch within the current epoch (unused).

        Returns
        -------
        Tensor
            Scalar total training loss used for the backward pass.
        """
        mol_graph, x_d, records = batch
        predictions = self(mol_graph, x_d)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("train_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self._epoch_losses["train_drc"].append(breakdown.drc_loss.detach())
        if not breakdown.ps_loss.isnan():
            self._epoch_losses["train_ps"].append(breakdown.ps_loss.detach())
        return breakdown.total

    def validation_step(self, batch: tuple[Any, Tensor, list[LabelRecord]], batch_idx: int) -> None:
        """Compute and log the validation loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, Tensor, list[LabelRecord]]
            A ``(mol_graph, x_d, records)`` triple.
        batch_idx : int
            Index of the batch within the current validation epoch (unused).
        """
        mol_graph, x_d, records = batch
        predictions = self(mol_graph, x_d)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("val_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self._epoch_losses["val_drc"].append(breakdown.drc_loss.detach())
        if not breakdown.ps_loss.isnan():
            self._epoch_losses["val_ps"].append(breakdown.ps_loss.detach())

    def on_train_epoch_end(self) -> None:
        """Emit epoch-mean DRC and PS training losses with a fixed key set."""
        self._log_epoch_fidelity_means("train")

    def on_validation_epoch_end(self) -> None:
        """Emit epoch-mean DRC and PS validation losses with a fixed key set."""
        self._log_epoch_fidelity_means("val")

    def _log_epoch_fidelity_means(self, stage: str) -> None:
        """Log epoch-mean fidelity losses for ``stage`` and reset accumulators.

        Parameters
        ----------
        stage : str
            Either ``"train"`` or ``"val"``.
        """
        for fidelity in ("drc", "ps"):
            values = self._epoch_losses[f"{stage}_{fidelity}"]
            mean = torch.stack(values).mean() if values else torch.tensor(float("nan"))
            self.log(f"{stage}_{fidelity}_loss", mean)
            self._epoch_losses[f"{stage}_{fidelity}"] = []

    def configure_optimizers(self) -> Adam:
        """Build and return the Adam optimizer for the current freeze state.

        Returns
        -------
        Adam
            Same param-group structure as
            :meth:`moal.model.ChemPropLightningModule.configure_optimizers`.
        """
        param_groups = [
            {
                "params": self._head_params(),
                "lr": self.ffn_lr,
                "weight_decay": self.ffn_weight_decay,
            }
        ]
        if not self._encoder_frozen:
            param_groups.append(
                {
                    "params": self._encoder_params(),
                    "lr": self.mpnn_lr,
                    "weight_decay": self.mpnn_weight_decay,
                }
            )
        return Adam(param_groups)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_smiles(
        self,
        smiles_list: list[str],
        readouts: list[dict[str, float]],
        aux_encoder: AuxiliaryEncoderModule,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Run batch inference over a list of canonical SMILES with concatenated features.

        Parameters
        ----------
        smiles_list : list[str]
            **Must be RDKit-canonical, salt-stripped SMILES**; see
            :meth:`moal.model.ChemPropLightningModule.predict_smiles`.
        readouts : list[dict[str, float]]
            Per-compound observed readouts, aligned with ``smiles_list``.
            Forwarded to :func:`build_concatenation_features`.
        aux_encoder : AuxiliaryEncoderModule
            Pretrained auxiliary encoder supplying both the readout-key
            order and the structural embedding.
        batch_size : int, optional
            Number of molecules processed per forward pass. Default is 256.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with pEC50 point estimates, aligned with
            ``smiles_list``.
        """
        features = build_concatenation_features(
            smiles_list,
            readouts,
            aux_encoder,
            use_observed_readout=self.use_observed_readout,
            batch_size=batch_size,
        )
        x_d = torch.as_tensor(features, dtype=torch.float32)

        # Chunk manually (rather than via chemprop's build_dataloader) so each
        # chunk's x_d slice is trivially aligned with its BatchMolGraph by
        # construction, instead of depending on undocumented batch-boundary
        # behavior inside the dataloader.
        all_preds = []
        with torch.inference_mode():
            for start in range(0, len(smiles_list), batch_size):
                chunk_smiles = smiles_list[start : start + batch_size]
                chunk_dataset = MoleculeDataset(
                    [MoleculeDatapoint.from_smi(s) for s in chunk_smiles]  # pyright: ignore[reportArgumentType]
                )
                bmg = BatchMolGraph([chunk_dataset[i].mg for i in range(len(chunk_dataset))])
                bmg.to(self.device)
                chunk_x_d = x_d[start : start + len(chunk_smiles)].to(self.device)
                preds = self(bmg, chunk_x_d).cpu().numpy().tolist()
                all_preds.extend(preds)

        return np.array(all_preds, dtype=np.float32)

    def refit(
        self,
        records: list[LabelRecord],
        aux_encoder: AuxiliaryEncoderModule,
        max_epochs: int = 30,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        trainer_kwargs: dict[str, Any] | None = None,
        datamodule_kwargs: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> ConcatenationChemPropLightningModule:
        """Refit the model on a (growing) labeled pool, using concatenated features.

        Parameters
        ----------
        records : list[LabelRecord]
            All labeled records accumulated so far.
        aux_encoder : AuxiliaryEncoderModule
            Pretrained auxiliary encoder used to build each record's
            concatenation features via :func:`build_concatenation_features`.
        max_epochs : int, optional
            Number of training epochs. Default is 30.
        enable_progress_bar : bool, optional
            Whether to show the Lightning progress bar. Default is False.
        enable_model_summary : bool, optional
            Whether to print the model summary at the start of training.
            Default is False.
        trainer_kwargs : dict[str, Any], optional
            Additional keyword arguments forwarded directly to
            ``lightning.Trainer``.
        datamodule_kwargs : dict[str, Any], optional
            Passed to the underlying data module (e.g. ``val_fraction``,
            ``seed``).
        output_dir : str or Path, optional
            Directory used as Lightning's ``default_root_dir``.

        Returns
        -------
        ConcatenationChemPropLightningModule
            self (for chaining).

        Raises
        ------
        ValueError
            If any record's fidelity is not ``QueryType.DOSE_RESPONSE``. The
            loss weighting is fixed equal for DRC/PS (see class docstring),
            so PS records would be silently mis-weighted rather than
            differentiated; callers should train the concatenation
            architecture on DRC records only.
        """
        non_drc = [rec for rec in records if rec.fidelity != QueryType.DOSE_RESPONSE]
        if non_drc:
            raise ValueError(
                f"ConcatenationChemPropLightningModule.refit() received {len(non_drc)} "
                "non-DOSE_RESPONSE record(s); this class trains on DRC records only "
                "(see class docstring for why PS/DRC loss weighting is unsupported here)."
            )
        records = normalize_record_weights(records)
        features = build_concatenation_features(
            [rec.canonical_smiles for rec in records],
            [rec.raw_ps_readouts for rec in records],
            aux_encoder,
            use_observed_readout=self.use_observed_readout,
        )
        dm = _ConcatenatedDataModule(records, features, **(datamodule_kwargs or {}))
        dm.setup()

        kwargs: dict[str, Any] = {
            "max_epochs": max_epochs,
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": enable_model_summary,
        }
        if trainer_kwargs:
            kwargs.update(trainer_kwargs)
        if output_dir is not None and "default_root_dir" not in kwargs:
            kwargs["default_root_dir"] = str(output_dir)
        kwargs.setdefault("logger", False)
        kwargs.setdefault("enable_checkpointing", False)
        trainer = L.Trainer(**kwargs)
        trainer.fit(self, datamodule=dm)
        return self


class _ConcatenatedDataModule(L.LightningDataModule):
    """LightningDataModule for concatenation-architecture pretraining.

    Same train/val split and device-transfer shape as
    :class:`~moal.dataset.MixedFidelityDataModule`, extended to also bundle
    each record's precomputed concatenation feature row. Not a subclass of
    that class: nearly every method's batch shape differs (an added feature
    tensor), so subclassing would mean overriding almost everything anyway.

    Parameters
    ----------
    records : list[LabelRecord]
        All labeled observations (train + val pool).
    features : np.ndarray
        Precomputed concatenation features, aligned with ``records``.
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
        features: np.ndarray,
        batch_size: int = 64,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.records = records
        self._features = features
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create the train and validation dataset splits over (record, feature) pairs.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage identifier; unused, accepted for interface
            compatibility.
        """
        n_val = int(len(self.records) * self.val_fraction)
        if self.val_fraction > 0.0:
            n_val = max(1, n_val)
        n_train = len(self.records) - n_val
        if n_train <= 0:
            logger.warning(
                "Too few records (%d) for a val split; using all for training.",
                len(self.records),
            )
            n_train, n_val = len(self.records), 0

        full = _ConcatenatedDataset(self.records, self._features)
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
        self,
        batch: tuple[Any, Tensor, list[LabelRecord]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[Any, Tensor, list[LabelRecord]]:
        """Move the batched mol graph and feature tensor to ``device``.

        Parameters
        ----------
        batch : tuple[Any, Tensor, list[LabelRecord]]
            A ``(BatchMolGraph, x_d, records)`` triple.
        device : torch.device
            Target device.
        dataloader_idx : int
            Index of the dataloader (required by the Lightning interface).

        Returns
        -------
        tuple[Any, Tensor, list[LabelRecord]]
            The same triple with the graph and feature tensor moved to
            ``device``; the LabelRecord list is returned unchanged.
        """
        mol_graph, x_d, records = batch
        mol_graph = super().transfer_batch_to_device(mol_graph, device, dataloader_idx)
        return mol_graph, x_d.to(device), records

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader.

        Returns
        -------
        DataLoader
            Shuffled DataLoader over the training split using
            :meth:`_ConcatenatedDataset.collate_fn`.
        """
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_ConcatenatedDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader, empty when no val split exists.

        Lightning requires a real iterable from this hook (returning ``None``
        raises); an empty ``DataLoader`` yields zero validation batches,
        which is the correct behavior for ``val_fraction=0.0``.

        Returns
        -------
        DataLoader
            Non-shuffled DataLoader over the validation split, or an empty
            ``DataLoader`` if no split was formed.
        """
        if self._val_dataset is None:
            return DataLoader([], batch_size=self.batch_size)  # pyright: ignore[reportArgumentType]
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_ConcatenatedDataset.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
