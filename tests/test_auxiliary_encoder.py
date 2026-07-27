"""Tests for the auxiliary log2FC/pIC50 encoder (issue #36 Phase 1).

All tests use ``from_foundation=False`` (random-init ChemProp encoder) so
no network download or cached CheMeleon checkpoint is required.
"""

from __future__ import annotations

import pytest
import torch
from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset

from moal.auxiliary_encoder import (
    AuxiliaryDataModule,
    AuxiliaryEncoderModule,
    load_auxiliary_encoder_checkpoint,
    masked_mse_loss,
    pretrain_auxiliary_encoder,
    save_auxiliary_encoder_checkpoint,
)
from moal.config import AuxiliaryModelConfig
from moal.types import CensoringType, LabelRecord, QueryType

_SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CCCl", "CCBr", "CCOCC", "CCCC"]


def _records_with_readouts() -> list[LabelRecord]:
    """Build a small set of LabelRecords with mixed, partially-overlapping readouts."""
    records = []
    for i, smi in enumerate(_SMILES):
        readouts = {"log2fc_1um": float(i) - 3.0}
        if i % 2 == 0:
            readouts["pic50"] = 6.0 + i * 0.1
        records.append(
            LabelRecord(
                smiles=smi,
                canonical_smiles=smi,
                value=5.0,
                upper_bound=11.0,
                censoring_type=CensoringType.LEFT,
                fidelity=QueryType.PRIMARY_SCREEN,
                cost=1.0,
                iteration=0,
                raw_ps_readouts=readouts,
            )
        )
    return records


def _batch(smiles_list: list[str]) -> BatchMolGraph:
    """Build a BatchMolGraph for a list of SMILES, mirroring moal.dataset's featurization path."""
    dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles_list])
    return BatchMolGraph([dataset[i].mg for i in range(len(dataset))])


def _fast_config(**overrides) -> AuxiliaryModelConfig:
    defaults = {
        "from_foundation": False,
        "message_hidden_dim": 16,
        "ffn_hidden_dim": 16,
        "depth": 1,
        "freeze_epochs": 0,
    }
    defaults.update(overrides)
    return AuxiliaryModelConfig(**defaults)


class TestMaskedMSELoss:
    """Tests for masked_mse_loss: gradient flow and zero-mask handling."""

    def test_masked_entries_contribute_zero_gradient(self):
        """A task masked out for every sample in the batch must receive zero gradient on that task's predictions."""
        preds = torch.tensor([[1.0, 5.0], [2.0, 5.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 999.0], [0.0, 999.0]])
        mask = torch.tensor([[True, False], [True, False]])

        loss = masked_mse_loss(preds, targets, mask)
        loss.backward()

        assert preds.grad is not None
        assert torch.all(preds.grad[:, 1] == 0.0)
        assert torch.any(preds.grad[:, 0] != 0.0)

    def test_fully_masked_batch_returns_zero_without_raising(self):
        """A batch with no observed targets at all must return a differentiable zero loss, not raise or NaN."""
        preds = torch.zeros(3, 2, requires_grad=True)
        targets = torch.zeros(3, 2)
        mask = torch.zeros(3, 2, dtype=torch.bool)

        loss = masked_mse_loss(preds, targets, mask)

        assert loss.item() == 0.0
        loss.backward()
        assert preds.grad is not None


class TestAuxiliaryEncoderModule:
    """Tests for AuxiliaryEncoderModule construction and freeze/unfreeze schedule."""

    def test_freeze_epochs_zero_starts_unfrozen_after_epoch_start(self):
        """With freeze_epochs=0, the encoder must unfreeze at the very first epoch boundary."""
        config = _fast_config(freeze_epochs=0)
        module = AuxiliaryEncoderModule(task_names=["log2fc_1um"], config=config)
        assert module._encoder_frozen is True

    def test_output_width_matches_task_count(self):
        """A forward pass's prediction width must equal len(task_names)."""
        config = _fast_config()
        module = AuxiliaryEncoderModule(task_names=["log2fc_1um", "pic50"], config=config)

        preds = module(_batch(["CCO", "CCN"]))

        assert preds.shape == (2, 2)

    def test_empty_task_names_raises(self):
        """Constructing with an empty task_names list must raise ValueError."""
        with pytest.raises(ValueError, match="task_names"):
            AuxiliaryEncoderModule(task_names=[], config=_fast_config())

    def test_embed_smiles_returns_backbone_width_aligned_with_input(self):
        """embed_smiles must return one embedding row per input SMILES, at the backbone's native width."""
        config = _fast_config(message_hidden_dim=24)
        module = AuxiliaryEncoderModule(task_names=["log2fc_1um"], config=config)

        embeddings = module.embed_smiles(["CCO", "CCN", "CCC"])

        assert embeddings.shape == (3, 24)


class TestPretrainAuxiliaryEncoder:
    """Tests for pretrain_auxiliary_encoder: training end-to-end and checkpoint opt-in."""

    def test_trains_and_returns_module_with_expected_tasks(self):
        """Pretraining on mixed-readout records must produce a module whose task_names is the sorted union of observed keys."""
        records = _records_with_readouts()
        config = _fast_config()

        module = pretrain_auxiliary_encoder(records, config, max_epochs=1)

        assert module.task_names == ["log2fc_1um", "pic50"]

    def test_records_without_any_readout_raises(self):
        """Pretraining with no readout-bearing records and no checkpoint_path must raise ValueError."""
        bare_record = LabelRecord(
            smiles="CCO",
            canonical_smiles="CCO",
            value=5.0,
            upper_bound=11.0,
            censoring_type=CensoringType.EXACT,
            fidelity=QueryType.DOSE_RESPONSE,
            cost=10.0,
            iteration=0,
        )
        with pytest.raises(ValueError, match="raw_ps_readouts"):
            pretrain_auxiliary_encoder([bare_record], _fast_config())

    def test_checkpoint_path_skips_retraining(self, tmp_path, monkeypatch):
        """When checkpoint_path is set, pretrain_auxiliary_encoder must load the checkpoint rather than training."""
        records = _records_with_readouts()
        config = _fast_config()
        trained = pretrain_auxiliary_encoder(records, config, max_epochs=1)
        ckpt_path = tmp_path / "aux_encoder.pt"
        save_auxiliary_encoder_checkpoint(trained, ckpt_path)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("Trainer.fit must not be called when checkpoint_path is set")

        monkeypatch.setattr("lightning.Trainer.fit", _fail_if_called)

        loaded_config = _fast_config(checkpoint_path=str(ckpt_path))
        loaded = pretrain_auxiliary_encoder(records, loaded_config)

        assert loaded.task_names == trained.task_names


class TestAuxiliaryEncoderCheckpoint:
    """Tests for save/load round-tripping."""

    def test_round_trips_weights_and_task_names(self, tmp_path):
        """A saved-then-loaded checkpoint must reproduce identical predictions and task_names."""
        records = _records_with_readouts()
        config = _fast_config()
        trained = pretrain_auxiliary_encoder(records, config, max_epochs=1)
        trained.eval()

        path = tmp_path / "aux_encoder.pt"
        save_auxiliary_encoder_checkpoint(trained, path)
        loaded = load_auxiliary_encoder_checkpoint(path, config)
        loaded.eval()

        bmg = _batch(["CCO", "CCN"])
        with torch.no_grad():
            preds_trained = trained(bmg)
            preds_loaded = loaded(bmg)

        assert loaded.task_names == trained.task_names
        assert torch.allclose(preds_trained, preds_loaded)


class TestAuxiliaryDataModule:
    """Tests for AuxiliaryDataModule train/val splitting and dataloader batch shape."""

    def test_train_batch_shapes_match_task_count(self):
        """A training batch's targets/mask must have shape (batch, n_tasks)."""
        records = _records_with_readouts()
        task_names = ["log2fc_1um", "pic50"]
        dm = AuxiliaryDataModule(records, task_names, batch_size=4, val_fraction=0.25, seed=1)
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        _, targets, mask = batch

        assert targets.shape[1] == 2
        assert mask.shape[1] == 2
        assert mask.dtype == torch.bool

    def test_too_few_records_uses_all_for_training(self):
        """When the record pool is too small for a val split, val_dataloader must be empty."""
        records = _records_with_readouts()[:1]
        dm = AuxiliaryDataModule(records, ["log2fc_1um"], val_fraction=0.1)
        dm.setup()

        assert len(dm.val_dataloader()) == 0
