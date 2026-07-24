"""ChemProp Lightning module with CheMeleon pretrained weights.

Key design constraints:
1. CheMeleon architecture hyperparameters are loaded from the checkpoint's
   ``hyper_parameters`` dict; no separate atom/bond dimension constants are required.
2. CheMeleon weights are loaded with strict=True — no silent mismatches.
3. A freeze/unfreeze schedule trains the FFN head first (freeze_epochs epochs)
   then unfreezes the encoder with a discriminative (lower) learning rate.
4. Loss is CensoredRegressionLoss from moal.loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from urllib.request import urlretrieve

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.dataloader import build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from torch import Tensor
from torch.optim import Adam

from moal.dataset import MixedFidelityDataModule
from moal.loss import CensoredRegressionLoss
from moal.planning import normalize_record_weights
from moal.types import LabelRecord

logger = logging.getLogger(__name__)

_KNOWN_FOUNDATION_MODELS: frozenset[str] = frozenset({"chemeleon"})


def _validate_from_foundation(value: str | bool) -> None:
    """Validate the ``from_foundation`` parameter value.

    Parameters
    ----------
    value : str or bool
        The value to validate.

    Raises
    ------
    ValueError
        If ``value`` is not ``False``, a known named model, or an existing
        file path.
    """
    if value is False:
        return
    if not isinstance(value, str):
        raise ValueError(
            f"from_foundation must be False, a known model name, or a filesystem path; "
            f"got {value!r}. Known names: {sorted(_KNOWN_FOUNDATION_MODELS)}"
        )
    if value in _KNOWN_FOUNDATION_MODELS:
        return
    if Path(value).exists():
        return
    raise ValueError(
        f"from_foundation={value!r} is not a recognised foundation model name "
        f"and does not resolve to an existing file path. "
        f"Known names: {sorted(_KNOWN_FOUNDATION_MODELS)}. "
        "Pass False to use random ChemProp weights."
    )


def load_foundation_weights(from_foundation: str | bool) -> dict:
    """Load pretrained message-passing weights from a named model or local path.

    Factored out of :class:`ChemPropLightningModule` so callers share an
    identical checkpoint-loading path.

    Parameters
    ----------
    from_foundation : str or bool
        ``"chemeleon"`` downloads (or reuses the cached copy of) the
        CheMeleon checkpoint from Zenodo. Any other string is treated as a
        local filesystem path. Must not be ``False``; validate with
        :func:`_validate_from_foundation` first.

    Returns
    -------
    dict
        Checkpoint dictionary with ``hyper_parameters`` and ``state_dict``
        keys.
    """
    if from_foundation == "chemeleon":
        download_chemeleon()
        ckpt_path = Path().home() / ".chemprop" / "chemeleon_mp.pt"
    else:
        ckpt_path = Path(str(from_foundation))
        logger.info("Loading foundation weights from local path: %s", ckpt_path)
    return cast(dict[str, Any], torch.load(ckpt_path, weights_only=True))


def build_mpnn(
    from_foundation: str | bool,
    ffn_hidden_dim: int,
    ffn_num_layers: int,
    message_hidden_dim: int,
    depth: int,
    n_tasks: int = 1,
) -> nn.Module:
    """Construct a ChemProp MPNN, dispatching on ``from_foundation``.

    Factored out of :class:`ChemPropLightningModule` so encoder construction
    lives in one place rather than being hand-rolled per caller.

    Parameters
    ----------
    from_foundation : str or bool
        ``False`` builds the message-passing encoder with random weights at
        ``message_hidden_dim`` / ``depth``. Any other value loads foundation
        weights via :func:`load_foundation_weights`, which also supplies the
        encoder's architecture (``message_hidden_dim`` and ``depth`` are
        ignored in that case).
    ffn_hidden_dim : int
        Hidden dimension of the FFN predictor head.
    ffn_num_layers : int
        Number of layers in the FFN predictor head.
    message_hidden_dim : int
        Message-passing hidden width (``d_h``) for the random-init encoder.
        Ignored when a foundation checkpoint supplies the architecture.
    depth : int
        Number of message-passing steps for the random-init encoder. Ignored
        when a foundation checkpoint supplies the architecture.
    n_tasks : int, optional
        Number of regression targets predicted per compound. Default is 1,
        the model's single pEC50 target.

    Returns
    -------
    nn.Module
        Fully assembled ``chemprop.models.MPNN``. Aggregation is always
        ``MeanAggregation``: CheMeleon's own pretraining used a mean readout,
        so any foundation-weights branch is constrained to match it; the
        random-init branch keeps the same readout for consistency between
        the two initialisation paths rather than introducing an
        undocumented behavioural difference.

    Notes
    -----
    ``message_hidden_dim`` and ``depth`` apply only on the
    ``from_foundation=False`` branch; for a foundation checkpoint the
    encoder architecture is read from the checkpoint's stored
    ``hyper_parameters`` so the pretrained weights load with ``strict=True``.
    """
    if from_foundation is False:
        logger.info(
            "Building ChemProp encoder with random weights "
            "(from_foundation=False, d_h=%d, depth=%d).",
            message_hidden_dim,
            depth,
        )
        mp: nn.Module = BondMessagePassing(  # pyright: ignore[reportAbstractUsage]
            d_h=message_hidden_dim, depth=depth
        )
    else:
        foundation_weights = load_foundation_weights(from_foundation)
        mp = BondMessagePassing(**foundation_weights["hyper_parameters"])  # pyright: ignore[reportAbstractUsage]
        mp.load_state_dict(foundation_weights["state_dict"])

    agg = MeanAggregation()
    ffn = RegressionFFN(  # pyright: ignore[reportAbstractUsage]
        n_tasks=n_tasks,
        input_dim=cast(BondMessagePassing, mp).output_dim,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
    )
    return cast(nn.Module, MPNN(message_passing=mp, agg=agg, predictor=ffn))


def download_chemeleon() -> None:
    """Download the CheMeleon checkpoint if not already cached locally.

    The file is stored at ``~/.chemprop/chemeleon_mp.pt``.  If the file
    already exists the download is skipped and the cached copy is used.

    Notes
    -----
    Checkpoint source: https://zenodo.org/records/15460715.
    Please cite DOI: 10.48550/arXiv.2506.15792 when using CheMeleon in
    published work.
    """
    ckpt_dir = Path().home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / "chemeleon_mp.pt"
    if not model_path.exists():
        logger.info(
            "Downloading CheMeleon Foundation model from Zenodo"
            " (https://zenodo.org/records/15460715) to %s",
            model_path,
        )
        urlretrieve(
            r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            model_path,
        )
    else:
        logger.info(f"Loading cached CheMeleon from {model_path}")

    logger.info("Please cite DOI: 10.48550/arXiv.2506.15792 when using CheMeleon in published work")


class ChemPropLightningModule(L.LightningModule):
    """ChemProp MPNN with configurable foundation-model encoder initialisation.

    Parameters
    ----------
    ffn_hidden_dim : int, optional
        FFN head hidden dimension. Default is 300.
    ffn_num_layers : int, optional
        Number of FFN layers. Default is 2.
    message_hidden_dim : int, optional
        Message-passing hidden width (``d_h``) for the random-init encoder.
        Applies only when ``from_foundation=False``; for a foundation
        checkpoint the width is read from the checkpoint. Default is 300
        (ChemProp's native default).
    depth : int, optional
        Number of message-passing steps for the random-init encoder. Applies
        only when ``from_foundation=False``; for a foundation checkpoint the
        depth is read from the checkpoint. Default is 3 (ChemProp's native
        default).
    freeze_epochs : int, optional
        Number of epochs to train only the FFN head before unfreezing the
        encoder. Default is 10.
    mpnn_lr : float, optional
        Learning rate for the message-passing encoder after unfreezing.
        Default is 1e-5.
    ffn_lr : float, optional
        Learning rate for the FFN head. Default is 1e-3.
    mpnn_weight_decay : float, optional
        L2 weight decay for the message-passing encoder param group. Default
        is 0.0 (no regularisation).
    ffn_weight_decay : float, optional
        L2 weight decay for the FFN head param group. Default is 0.0 (no
        regularisation).
    sigma : float, optional
        Fixed noise scale for ``CensoredRegressionLoss``. Default is 0.5.
    w_drc : float, optional
        DRC loss weight. Default is 1.0.
    w_ps : float, optional
        Primary screen loss weight. Default is 0.3.
    learnable_sigma : bool, optional
        If True, σ is a learned parameter. Default is False.
    from_foundation : str or bool, optional
        Controls encoder initialisation. ``"chemeleon"`` (default) downloads
        and loads CheMeleon pretrained weights. A filesystem path string loads
        a local checkpoint in ``{hyper_parameters, state_dict}`` format.
        ``False`` builds the encoder with default ChemProp architecture and
        random weights.
    """

    def __init__(
        self,
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
        w_drc: float = 1.0,
        w_ps: float = 0.3,
        learnable_sigma: bool = False,
        from_foundation: str | bool = "chemeleon",
    ) -> None:
        super().__init__()
        _validate_from_foundation(from_foundation)
        self._from_foundation = from_foundation
        self.save_hyperparameters()

        self.freeze_epochs = freeze_epochs
        self.mpnn_lr = mpnn_lr
        self.ffn_lr = ffn_lr
        self.mpnn_weight_decay = mpnn_weight_decay
        self.ffn_weight_decay = ffn_weight_decay
        self._encoder_frozen = True

        # Per-epoch accumulators for fidelity-resolved losses, reset each epoch.
        # Emitting both fidelity keys together at epoch end keeps the CSV logger
        # header fixed even when individual batches contain only one fidelity, so
        # the metrics columns never shift mid-run and trip the CSV writer
        self._epoch_losses: dict[str, list[Tensor]] = {
            "train_drc": [],
            "train_ps": [],
            "val_drc": [],
            "val_ps": [],
        }

        self.loss_fn = CensoredRegressionLoss(
            sigma=sigma, w_drc=w_drc, w_ps=w_ps, learnable_sigma=learnable_sigma
        )

        self.model = self._build_model(
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            message_hidden_dim=message_hidden_dim,
            depth=depth,
        )
        self._freeze_encoder()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        ffn_hidden_dim: int,
        ffn_num_layers: int,
        message_hidden_dim: int,
        depth: int,
    ) -> nn.Module:
        """Construct the MPNN, dispatching on ``self._from_foundation``.

        Thin wrapper around the shared :func:`build_mpnn`; see that function
        for the full construction contract.

        Parameters
        ----------
        ffn_hidden_dim : int
            Hidden dimension of the FFN predictor head.
        ffn_num_layers : int
            Number of layers in the FFN predictor head.
        message_hidden_dim : int
            Message-passing hidden width (``d_h``) for the random-init encoder.
            Ignored when a foundation checkpoint supplies the architecture.
        depth : int
            Number of message-passing steps for the random-init encoder.
            Ignored when a foundation checkpoint supplies the architecture.

        Returns
        -------
        nn.Module
            Fully assembled ``chemprop.models.MPNN`` with a single-task
            (``n_tasks=1``) predictor head.
        """
        return build_mpnn(
            from_foundation=self._from_foundation,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            message_hidden_dim=message_hidden_dim,
            depth=depth,
        )

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
        """Freeze all message-passing encoder parameters.

        Sets ``requires_grad=False`` on every encoder parameter and marks
        ``_encoder_frozen`` as ``True``.  Called once at construction and
        implicitly again whenever ``reset_weights=True`` is passed to
        :meth:`refit`.
        """
        for p in self._encoder_params():
            p.requires_grad_(False)
        self._encoder_frozen = True
        logger.debug("Encoder frozen.")

    def _unfreeze_encoder(self) -> None:
        """Unfreeze the message-passing encoder after the warm-up phase.

        Sets ``requires_grad=True`` on every encoder parameter and marks
        ``_encoder_frozen`` as ``False``.  Invoked automatically by
        :meth:`on_train_epoch_start` once ``current_epoch`` reaches
        ``freeze_epochs``.
        """
        for p in self._encoder_params():
            p.requires_grad_(True)
        self._encoder_frozen = False
        logger.debug("Encoder unfrozen after warm-up.")

    def on_train_epoch_start(self) -> None:
        """Lightning hook: unfreeze the encoder when warm-up is complete.

        When ``current_epoch`` first reaches ``freeze_epochs`` and the encoder
        is still frozen, :meth:`_unfreeze_encoder` is called and the
        optimizers are rebuilt so that the newly unfrozen parameters receive
        ``mpnn_lr``.

        Notes
        -----
        The epoch counter resets to zero on every :meth:`refit` call because
        a new ``L.Trainer`` is created each time.  This ensures the warm-up
        phase applies at the start of every active-learning iteration,
        regardless of how many training epochs elapsed in previous iterations.
        """
        if self._encoder_frozen and self.current_epoch >= self.freeze_epochs:
            self._unfreeze_encoder()
            # Rebuild optimizers so the newly unfrozen params get mpnn_lr
            self.trainer.strategy.setup_optimizers(self.trainer)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def forward(self, batch_mol_graph: Any) -> Tensor:
        """Run a forward pass and return scalar pEC50 predictions.

        Parameters
        ----------
        batch_mol_graph : Any
            A batched molecular graph (``chemprop.data.BatchMolGraph``)
            produced by the chemprop dataloader collate function.

        Returns
        -------
        Tensor
            1-D tensor of shape ``(N,)`` with predicted pEC50 values.
        """
        return cast(Tensor, self.model(batch_mol_graph).squeeze(-1))

    def training_step(self, batch: tuple[Any, list[LabelRecord]], batch_idx: int) -> Tensor:
        """Compute and log the training loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, list[LabelRecord]]
            A ``(mol_graph, records)`` pair returned by the training
            dataloader.  ``mol_graph`` is a batched ``BatchMolGraph``;
            ``records`` is the corresponding list of
            :class:`~moal.types.LabelRecord` objects carrying censoring
            metadata.
        batch_idx : int
            Index of the batch within the current epoch (unused).

        Returns
        -------
        Tensor
            Scalar total training loss used for the backward pass.

        Notes
        -----
        Fidelity-resolved losses are accumulated per epoch (see
        ``_epoch_losses``) and emitted in :meth:`on_train_epoch_end` rather than
        logged per step, so the metrics column set stays fixed across the run.
        """
        mol_graph, records = batch
        predictions = self(mol_graph)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("train_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self._epoch_losses["train_drc"].append(breakdown.drc_loss.detach())
        if not breakdown.ps_loss.isnan():
            self._epoch_losses["train_ps"].append(breakdown.ps_loss.detach())
        return breakdown.total

    def validation_step(self, batch: tuple[Any, list[LabelRecord]], batch_idx: int) -> None:
        """Compute and log the validation loss for one batch.

        Parameters
        ----------
        batch : tuple[Any, list[LabelRecord]]
            A ``(mol_graph, records)`` pair returned by the validation
            dataloader.
        batch_idx : int
            Index of the batch within the current validation epoch (unused).

        Notes
        -----
        Fidelity-resolved losses are accumulated per epoch (see
        ``_epoch_losses``) and emitted in :meth:`on_validation_epoch_end` rather
        than logged per step, so the metrics column set stays fixed across the
        run.
        """
        mol_graph, records = batch
        predictions = self(mol_graph)
        breakdown = self.loss_fn.forward_with_breakdown(predictions, records)
        self.log("val_loss", breakdown.total, prog_bar=True, batch_size=len(records))
        if not breakdown.drc_loss.isnan():
            self._epoch_losses["val_drc"].append(breakdown.drc_loss.detach())
        if not breakdown.ps_loss.isnan():
            self._epoch_losses["val_ps"].append(breakdown.ps_loss.detach())

    def on_train_epoch_end(self) -> None:
        """Emit epoch-mean DRC and PS training losses with a fixed key set.

        Both ``train_drc_loss`` and ``train_ps_loss`` are logged every epoch,
        using ``nan`` for a fidelity that appeared in no batch this epoch. Always
        emitting both keys keeps the CSV logger header stable, which the previous
        per-step conditional logging did not, causing intermittent
        ``dict contains fields not in fieldnames`` crashes on runs where a
        fidelity was sparse across batches.
        """
        self._log_epoch_fidelity_means("train")

    def on_validation_epoch_end(self) -> None:
        """Emit epoch-mean DRC and PS validation losses with a fixed key set.

        See :meth:`on_train_epoch_end`; the same fixed-key-set rationale applies
        to the validation metrics.
        """
        self._log_epoch_fidelity_means("val")

    def _log_epoch_fidelity_means(self, stage: str) -> None:
        """Log epoch-mean fidelity losses for ``stage`` and reset accumulators.

        Parameters
        ----------
        stage : str
            Either ``"train"`` or ``"val"``; selects which accumulators to
            reduce and the metric-key prefix to log under.
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
            When the encoder is frozen, a single-group Adam optimizer for
            the FFN head at ``ffn_lr`` with ``ffn_weight_decay``.  After the
            encoder is unfrozen, a two-group Adam with an additional encoder
            group at ``mpnn_lr`` with ``mpnn_weight_decay``.

        Notes
        -----
        This method is re-invoked by Lightning after :meth:`on_train_epoch_start`
        calls ``setup_optimizers`` so that newly unfrozen encoder parameters
        are registered at the correct learning rate.
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
    def predict_smiles(self, smiles_list: list[str], batch_size: int = 256) -> np.ndarray:
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
        # Create the full dataset once rather than chunking manually
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles_list])  # pyright: ignore[reportArgumentType]

        # Let the dataloader handle batching and graph collation automatically.
        # drop_last=False is explicit: chemprop defaults to dropping the last
        # batch when len(dataset) % batch_size == 1 to protect batch-norm
        # during training, but at inference that would silently omit a molecule
        # and misalign predictions with the input SMILES list.
        dataloader = build_dataloader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        all_preds = []

        # Disable gradient tracking for inference
        with torch.inference_mode():
            for batch in dataloader:
                # Move batch to device
                batch.bmg.to(self.device)

                # Make predictions
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
    ) -> ChemPropLightningModule:
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
            Directory used as Lightning's ``default_root_dir`` (checkpoints and
            any opt-in logger output). When provided and ``default_root_dir`` is
            not already present in ``trainer_kwargs``, it is injected into the
            ``L.Trainer`` constructor. If None (default), Lightning writes to the
            current working directory. Note the logger is disabled by default,
            so no ``lightning_logs/`` CSVs are written unless a caller opts in
            via ``trainer_kwargs["logger"]``.

        Returns
        -------
        ChemPropLightningModule
            self (for chaining).
        """
        if reset_weights:
            self.model = self._build_model(
                ffn_hidden_dim=self.hparams["ffn_hidden_dim"],
                ffn_num_layers=self.hparams["ffn_num_layers"],
                message_hidden_dim=self.hparams["message_hidden_dim"],
                depth=self.hparams["depth"],
            )
            self._freeze_encoder()

        records = normalize_record_weights(records)
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

        # Disable Lightning's logger by default: the campaign reads metrics from
        # LoopResults, never from the lightning_logs CSVs, and Lightning's
        # CSVLogger header-rewrite path crashes intermittently on the
        # fidelity-dependent metric keys (train_ps_loss/train_drc_loss appearing
        # or going all-nan across refits). A caller can still opt in by passing
        # logger in trainer_kwargs
        kwargs.setdefault("logger", False)
        # Disable checkpointing by default: refit trains a fixed number of epochs
        # and the campaign never reloads a checkpoint (each iteration refits from
        # scratch and evaluates in-process). Lightning's default ModelCheckpoint
        # writes via a temp file that it then renames onto the target directory,
        # which raises OSError(EXDEV) "Invalid cross-device link" on nodes whose
        # temp dir is a different filesystem from output_dir. A caller can still
        # opt in by passing enable_checkpointing or a ModelCheckpoint callback.
        kwargs.setdefault("enable_checkpointing", False)
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
            # Multiply by 2 so that the expected absolute error equals noise_scale
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
    ) -> NoisyOracleModel:
        """No-op refit — fast mode has no learnable parameters to update.

        All arguments are accepted for interface compatibility with
        :class:`ChemPropLightningModule` and silently ignored.

        Parameters
        ----------
        records : list[LabelRecord]
            Labeled records accumulated so far (ignored).
        max_epochs : int, optional
            Ignored. Default is 30.
        enable_progress_bar : bool, optional
            Ignored. Default is False.
        enable_model_summary : bool, optional
            Ignored. Default is False.
        trainer_kwargs : dict[str, Any], optional
            Ignored. Default is None.
        datamodule_kwargs : dict[str, Any], optional
            Ignored. Default is None.
        reset_weights : bool, optional
            Ignored. Default is False.
        output_dir : str or Path, optional
            Ignored. Default is None.

        Returns
        -------
        NoisyOracleModel
            self (unchanged).
        """
        return self
