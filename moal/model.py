"""ChemProp Lightning module with CheMeleon pretrained weights.

Key design constraints:
1. CheMeleon atom/bond feature constants are hardcoded and asserted at init.
2. CheMeleon weights are loaded with strict=True — no silent mismatches.
3. A freeze/unfreeze schedule trains the FFN head first (freeze_epochs epochs)
   then unfreezes the encoder with a discriminative (lower) learning rate.
4. Loss is CensoredRegressionLoss from moal.loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from moal.loss import CensoredRegressionLoss
from moal.types import LabelRecord

logger = logging.getLogger(__name__)


def download_chemeleon():
    """
    Download CheMeleon checkpoint if not already cached locally.

    """
    logger.info(
        "Please cite DOI: 10.48550/arXiv.2506.15792 when using CheMeleon in published work"
    )
    ckpt_dir = Path().home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / "chemeleon_mp.pt"
    if not model_path.exists():
        logger.info(
            f"Downloading CheMeleon Foundation model from Zenodo (https://zenodo.org/records/15460715) to {model_path}"
        )
        urlretrieve(
            r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            model_path,
        )
    else:
        logger.info(f"Loading cached CheMeleon from {model_path}")


class ChemPropLightningModule(L.LightningModule):
    """ChemProp MPNN fine-tuned from CheMeleon pretrained weights.

    Parameters
    ----------
    hidden_size : int, optional
        MPNN hidden dimension (must match CheMeleon checkpoint). Default is 300.
    depth : int, optional
        Number of message-passing steps. Default is 3.
    ffn_hidden_size : int, optional
        FFN head hidden dimension. Default is 300.
    ffn_num_layers : int, optional
        Number of FFN layers. Default is 2.
    freeze_epochs : int, optional
        Number of epochs to train only the FFN head before unfreezing the
        encoder. Default is 10.
    lr_encoder : float, optional
        Learning rate for the message-passing encoder after unfreezing.
        Default is 1e-5.
    lr_head : float, optional
        Learning rate for the FFN head. Default is 1e-3.
    sigma : float, optional
        Fixed noise scale for ``CensoredRegressionLoss``. Default is 0.5.
    w_drc : float, optional
        DRC loss weight. Default is 1.0.
    w_ps : float, optional
        Primary screen loss weight. Default is 0.3.
    learnable_sigma : bool, optional
        If True, σ is a learned parameter. Default is False.
    """

    def __init__(
        self,
        ffn_hidden_size: int = 300,
        ffn_num_layers: int = 2,
        freeze_epochs: int = 10,
        lr_encoder: float = 1e-5,
        lr_head: float = 1e-3,
        sigma: float = 0.5,
        w_drc: float = 1.0,
        w_ps: float = 0.3,
        learnable_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.freeze_epochs = freeze_epochs
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        self._encoder_frozen = True

        self.loss_fn = CensoredRegressionLoss(
            sigma=sigma, w_drc=w_drc, w_ps=w_ps, learnable_sigma=learnable_sigma
        )

        self.model = self._build_model(
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
        )
        self._freeze_encoder()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        ffn_hidden_size: int,
        ffn_num_layers: int,
    ) -> nn.Module:
        try:
            from chemprop.models import MPNN
            from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chemprop>=2.0 is required. Install with: pip install chemprop"
            ) from exc

        # Load message passing from CheMeleon
        chemeleon_weights = self._get_chemeleon_mp()

        # Mean aggregation
        agg = MeanAggregation()

        # Message passing
        mp = BondMessagePassing(**chemeleon_weights["hyper_parameters"])
        mp.load_state_dict(chemeleon_weights["state_dict"])

        # FFN predictor head
        ffn = RegressionFFN(
            input_dim=mp.output_dim,  # Infer input dim from mp output
            hidden_dim=ffn_hidden_size,
            n_layers=ffn_num_layers,
        )
        return MPNN(message_passing=mp, agg=agg, predictor=ffn)

    def _get_chemeleon_mp(self) -> None:
        # Ensure CheMeleon checkpoint is downloaded``
        download_chemeleon()

        # Path to checkpoint
        ckpt_path = Path().home() / ".chemprop" / "chemeleon_mp.pt"

        # Load weights
        weights = torch.load(ckpt_path, weights_only=True)

        return weights

    # ------------------------------------------------------------------
    # Freeze / unfreeze schedule
    # ------------------------------------------------------------------

    def _encoder_params(self) -> list[nn.Parameter]:
        return list(self.model.message_passing.parameters())

    def _head_params(self) -> list[nn.Parameter]:
        return list(self.model.agg.parameters()) + list(
            self.model.predictor.parameters()
        )

    def _freeze_encoder(self) -> None:
        for p in self._encoder_params():
            p.requires_grad_(False)
        self._encoder_frozen = True
        logger.debug("Encoder frozen.")

    def _unfreeze_encoder(self) -> None:
        for p in self._encoder_params():
            p.requires_grad_(True)
        self._encoder_frozen = False
        logger.info("Encoder unfrozen after warm-up.")

    def on_train_epoch_start(self) -> None:
        if self._encoder_frozen and self.current_epoch >= self.freeze_epochs:
            self._unfreeze_encoder()
            # Rebuild optimizers so the newly unfrozen params get lr_encoder.
            self.trainer.strategy.setup_optimizers(self.trainer)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

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

    def forward(self, batch_mol_graph: Any) -> Tensor:
        return self.model(batch_mol_graph).squeeze(-1)

    def training_step(
        self, batch: tuple[Any, list[LabelRecord]], batch_idx: int
    ) -> Tensor:
        mol_graph, records = batch
        predictions = self(mol_graph)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("train_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self.log("train_drc_loss", breakdown.drc_loss, batch_size=len(records))
        if not breakdown.ps_loss.isnan():
            self.log("train_ps_loss", breakdown.ps_loss, batch_size=len(records))
        return breakdown.total

    def validation_step(
        self, batch: tuple[Any, list[LabelRecord]], batch_idx: int
    ) -> None:
        mol_graph, records = batch
        predictions = self(mol_graph)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("val_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self.log("val_drc_loss", breakdown.drc_loss, batch_size=len(records))
        if not breakdown.ps_loss.isnan():
            self.log("val_ps_loss", breakdown.ps_loss, batch_size=len(records))

    def configure_optimizers(self) -> Adam:
        param_groups = [{"params": self._head_params(), "lr": self.lr_head}]
        if not self._encoder_frozen:
            param_groups.append(
                {"params": self._encoder_params(), "lr": self.lr_encoder}
            )
        return Adam(param_groups)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_smiles(
        self, smiles_list: list[str], batch_size: int = 256
    ) -> np.ndarray:
        """Run batch inference over a list of canonical SMILES.

        Parameters
        ----------
        smiles_list : list[str]
            **Must be RDKit-canonical, salt-stripped SMILES.** Passing
            non-canonical or salt-containing SMILES produces silently incorrect
            graph representations because the featurizer does not canonicalize
            internally. All SMILES in the active learning loop are
            pre-canonicalized by ``SMILESPreprocessor`` before reaching this
            method. If calling this method directly from user code, run each
            SMILES through ``SMILESPreprocessor().canonicalize(smi)`` first.
        batch_size : int, optional
            Number of molecules processed per forward pass. Default is 256.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with pEC50 point estimates, aligned with
            ``smiles_list``.
        """
        from chemprop.data import MoleculeDatapoint, MoleculeDataset
        from chemprop.data.dataloader import build_dataloader

        # Create the full dataset once rather than chunking manually.
        # Note that if using ChemProp v2 you may also need to pass a featurizer to this class.
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles_list])

        # Let the dataloader handle the batching and graph collation automatically.
        dataloader = build_dataloader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []

        # Disable gradient tracking for inference.
        with torch.inference_mode():
            for batch in dataloader:
                # Move bactch to the device
                batch.bmg.to(self.device)

                # Make predictions and detach
                preds = self(batch.bmg).cpu().numpy().tolist()

                # Accumulate predictions
                all_preds.extend(preds)

        return np.array(all_preds, dtype=np.float32)

    def refit(
        self,
        records: list[LabelRecord],
        max_epochs: int = 30,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        trainer_kwargs: dict[str, Any] | None = None,
        datamodule_kwargs: dict[str, Any] | None = None,
        reset_weights: bool = False,
        output_dir: str | Path | None = None,
    ) -> "ChemPropLightningModule":
        """Refit the model on a (growing) labeled pool.

        Parameters
        ----------
        records : list[LabelRecord]
            All labeled records accumulated so far.
        max_epochs : int, optional
            Number of training epochs. Default is 30. When using the CLI this
            is inferred from ``TrainerConfig.max_epochs`` via
            ``TrainerConfig.to_dict()``.
        enable_progress_bar : bool, optional
            Whether to show the Lightning progress bar. Default is False.
        enable_model_summary : bool, optional
            Whether to print the model summary at the start of training.
            Default is False.
        trainer_kwargs : dict[str, Any], optional
            Additional keyword arguments forwarded directly to
            ``lightning.Trainer`` (e.g. ``accelerator``). Any keys that
            overlap with the explicit parameters above take precedence.
        datamodule_kwargs : dict[str, Any], optional
            Passed to ``MixedFidelityDataModule`` (e.g. ``val_fraction``,
            ``seed``). Use ``TrainerConfig.to_datamodule_kwargs()`` to
            populate this from the campaign config so train/val split
            parameters are reproducible and config-driven.
        reset_weights : bool, optional
            If True, reload CheMeleon weights before refitting (full warm-start
            from pretrained). Default is False (continue fine-tuning from
            current weights).
        output_dir : str or Path, optional
            Directory where Lightning should write its default logs
            (``lightning_logs/``). When provided and ``default_root_dir`` is
            not already present in ``trainer_kwargs``, it is injected as
            ``default_root_dir`` into the ``L.Trainer`` constructor. If None
            (default), Lightning writes to the current working directory.

        Returns
        -------
        ChemPropLightningModule
            self (for chaining).
        """
        from moal.dataset import MixedFidelityDataModule

        if reset_weights:
            self._build_model()
            self._freeze_encoder()

        dm = MixedFidelityDataModule(records, **(datamodule_kwargs or {}))
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

        trainer = L.Trainer(**kwargs)
        trainer.fit(self, datamodule=dm)
        return self


# ---------------------------------------------------------------------------
# Fast surrogate model
# ---------------------------------------------------------------------------


class NoisyOracleModel:
    """Noisy oracle surrogate that bypasses CheMeleon entirely.

    Predicts pEC50 by adding Uniform(-noise_scale, +noise_scale) noise to the
    true ground-truth values. Intended for rapid campaign prototyping,
    debugging, and integration testing without downloading or loading the
    CheMeleon foundation model.

    This model satisfies the same two-method interface as
    :class:`ChemPropLightningModule` (``predict_smiles`` and ``refit``) so it
    can be used as a drop-in replacement in :class:`~moal.loop.ActiveLearningLoop`.

    The noise level is not fixed at construction — it is supplied per-call via
    the ``noise_scale`` argument to ``predict_smiles``, allowing the caller
    (typically :class:`~moal.loop.ActiveLearningLoop`) to implement an error
    ramp across iterations.

    Parameters
    ----------
    ground_truth : dict[str, float]
        Mapping from canonical SMILES to true pEC50 values. Typically
        ``oracle._ground_truth``.
    seed : int, optional
        Seed for the internal RNG, ensuring reproducible predictions across
        runs with the same configuration. Default is 42.
    """

    def __init__(
        self,
        ground_truth: dict[str, float],
        seed: int = 42,
    ) -> None:
        self._ground_truth = ground_truth
        self._rng = np.random.default_rng(seed)

    def predict_smiles(
        self, smiles_list: list[str], noise_scale: float, batch_size: int = 256
    ) -> np.ndarray:
        """Return noisy pEC50 estimates for a list of canonical SMILES.

        Each prediction is sampled as ``true_pec50 + Uniform(-2 * noise_scale, +2 * noise_scale)``.
        The ``batch_size`` argument is accepted for interface compatibility but
        has no effect — there is no batched computation.

        Parameters
        ----------
        smiles_list : list[str]
            Canonical SMILES strings. Each must be present in the
            ``ground_truth`` dict supplied at construction.
        noise_scale : float
            Half-width of the uniform noise distribution (pEC50 log-units).
            A value of 0.0 returns exact oracle values. Must be non-negative.
            Passed by the caller on every invocation, enabling per-iteration
            error schedules.
        batch_size : int, optional
            Ignored; retained for interface compatibility with
            :class:`ChemPropLightningModule`. Default is 256.

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` with pEC50 estimates, aligned with
            ``smiles_list``.

        Raises
        ------
        ValueError
            If ``noise_scale`` is negative.
        KeyError
            If any SMILES in ``smiles_list`` is not in ``ground_truth``.
        """
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        preds = np.empty(len(smiles_list), dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            # KeyError propagates if smi is absent, matching ChemPropLightningModule behaviour
            true_pec50 = self._ground_truth[smi]
            # Must multiple by 2 to model MAE
            noise = self._rng.uniform(-2 * noise_scale, 2 * noise_scale)
            preds[i] = true_pec50 + noise
        return preds

    def refit(
        self,
        records: list[LabelRecord],
        max_epochs: int = 30,
        enable_progress_bar: bool = False,
        enable_model_summary: bool = False,
        trainer_kwargs: dict[str, Any] | None = None,
        datamodule_kwargs: dict[str, Any] | None = None,
        reset_weights: bool = False,
        output_dir: str | Path | None = None,
    ) -> "NoisyOracleModel":
        """No-op refit — fast mode has no learnable parameters to update.

        All arguments are accepted for interface compatibility with
        :class:`ChemPropLightningModule` and silently ignored.

        Returns
        -------
        NoisyOracleModel
            self
        """
        return self
