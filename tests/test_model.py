"""Tests for ChemPropLightningModule initialization.

All tests patch ``_load_chempeleon_weights`` to a no-op so that no real
checkpoint file is required. This isolates init logic (model construction,
freeze schedule, hyperparameter storage, optimizer configuration) from I/O.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from moal.loss import CensoredRegressionLoss
from moal.model import ChemPropLightningModule


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model(monkeypatch) -> ChemPropLightningModule:
    """Default-config module with checkpoint loading disabled."""
    monkeypatch.setattr(ChemPropLightningModule, "_load_chempeleon_weights", lambda *_: None)
    return ChemPropLightningModule(chempeleon_ckpt_path="dummy.ckpt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(monkeypatch, **kwargs) -> ChemPropLightningModule:
    """Instantiate with checkpoint loading disabled, forwarding kwargs."""
    monkeypatch.setattr(ChemPropLightningModule, "_load_chempeleon_weights", lambda *_: None)
    return ChemPropLightningModule(chempeleon_ckpt_path="dummy.ckpt", **kwargs)


# ---------------------------------------------------------------------------
# Default initialization
# ---------------------------------------------------------------------------


class TestDefaultInit:
    def test_model_attribute_is_nn_module(self, model):
        """The inner MPNN must be an nn.Module."""
        assert isinstance(model.model, nn.Module)

    def test_encoder_frozen_at_init(self, model):
        """Encoder must be frozen immediately after init."""
        assert model._encoder_frozen is True

    def test_encoder_params_require_no_grad(self, model):
        """All message-passing parameters must have requires_grad=False after init."""
        assert all(not p.requires_grad for p in model._encoder_params())

    def test_head_params_require_grad(self, model):
        """FFN head parameters must remain trainable after the encoder is frozen."""
        assert all(p.requires_grad for p in model._head_params())

    def test_loss_fn_type(self, model):
        """loss_fn must be a CensoredRegressionLoss instance."""
        assert isinstance(model.loss_fn, CensoredRegressionLoss)

    def test_hparams_contains_architecture_keys(self, model):
        """save_hyperparameters must record all architecture and training params."""
        expected = {
            "hidden_size", "depth", "ffn_hidden_size", "ffn_num_layers",
            "freeze_epochs", "lr_encoder", "lr_head",
            "sigma", "w_drc", "w_ps", "learnable_sigma",
        }
        assert expected.issubset(set(model.hparams.keys()))

    def test_hparams_default_values(self, model):
        """Default hyperparameters must match the documented defaults."""
        assert model.hparams["hidden_size"] == 300
        assert model.hparams["depth"] == 3
        assert model.hparams["ffn_hidden_size"] == 300
        assert model.hparams["ffn_num_layers"] == 2
        assert model.hparams["freeze_epochs"] == 10
        assert model.hparams["sigma"] == pytest.approx(0.5)
        assert model.hparams["w_drc"] == pytest.approx(1.0)
        assert model.hparams["w_ps"] == pytest.approx(0.3)
        assert model.hparams["learnable_sigma"] is False


# ---------------------------------------------------------------------------
# Architecture parametrization
# ---------------------------------------------------------------------------


class TestArchitectureParams:
    @pytest.mark.parametrize("hidden_size", [128, 256, 512])
    def test_hidden_size_sets_message_passing_width(self, monkeypatch, hidden_size):
        """W_h in the message-passing layer must reflect the configured hidden size."""
        m = _make_model(monkeypatch, hidden_size=hidden_size)
        assert m.model.message_passing.W_h.in_features == hidden_size
        assert m.model.message_passing.W_h.out_features == hidden_size

    @pytest.mark.parametrize("depth", [2, 3, 5])
    def test_depth_sets_message_passing_depth(self, monkeypatch, depth):
        """The message-passing depth attribute must match the configured depth."""
        m = _make_model(monkeypatch, depth=depth)
        assert m.model.message_passing.depth == depth

    @pytest.mark.parametrize("ffn_num_layers", [1, 2, 4])
    def test_ffn_num_layers_sets_predictor_depth(self, monkeypatch, ffn_num_layers):
        """The FFN predictor must contain ffn_num_layers + 1 sequential blocks
        (chemprop adds an extra input projection block)."""
        m = _make_model(monkeypatch, ffn_num_layers=ffn_num_layers)
        assert len(m.model.predictor.ffn) == ffn_num_layers + 1

    @pytest.mark.parametrize("ffn_hidden_size", [64, 128, 512])
    def test_ffn_hidden_size_sets_predictor_width(self, monkeypatch, ffn_hidden_size):
        """The hidden linear layers in the FFN predictor must match ffn_hidden_size."""
        m = _make_model(monkeypatch, ffn_hidden_size=ffn_hidden_size)
        # Block 1 is the first hidden layer; index [2] is the Linear within the Sequential.
        assert m.model.predictor.ffn[1][2].in_features == ffn_hidden_size

    @pytest.mark.parametrize(
        "hidden_size,depth,ffn_num_layers",
        [
            (128, 2, 1),
            (256, 4, 3),
            (512, 3, 2),
        ],
    )
    def test_combined_architecture_params(self, monkeypatch, hidden_size, depth, ffn_num_layers):
        """Combinations of architecture params must all be reflected in the built model."""
        m = _make_model(monkeypatch, hidden_size=hidden_size, depth=depth, ffn_num_layers=ffn_num_layers)
        assert m.model.message_passing.W_h.in_features == hidden_size
        assert m.model.message_passing.depth == depth
        assert len(m.model.predictor.ffn) == ffn_num_layers + 1


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------


class TestTrainingHyperparams:
    @pytest.mark.parametrize("freeze_epochs", [0, 5, 20])
    def test_freeze_epochs_stored(self, monkeypatch, freeze_epochs):
        """freeze_epochs must be stored as an instance attribute."""
        m = _make_model(monkeypatch, freeze_epochs=freeze_epochs)
        assert m.freeze_epochs == freeze_epochs

    @pytest.mark.parametrize("lr_head", [1e-4, 1e-3, 5e-3])
    def test_lr_head_in_optimizer(self, monkeypatch, lr_head):
        """The head optimizer param group must use the configured lr_head."""
        m = _make_model(monkeypatch, lr_head=lr_head)
        opt = m.configure_optimizers()
        # When frozen, only the head param group is present.
        assert opt.param_groups[0]["lr"] == pytest.approx(lr_head)

    @pytest.mark.parametrize("lr_encoder", [1e-6, 1e-5, 1e-4])
    def test_lr_encoder_in_optimizer_after_unfreeze(self, monkeypatch, lr_encoder):
        """After unfreezing, the encoder param group must use the configured lr_encoder."""
        m = _make_model(monkeypatch, lr_encoder=lr_encoder)
        m._unfreeze_encoder()
        opt = m.configure_optimizers()
        encoder_lrs = [g["lr"] for g in opt.param_groups if g["lr"] != m.lr_head]
        assert encoder_lrs == [pytest.approx(lr_encoder)]


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------


class TestLossConfig:
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0])
    def test_sigma_stored_in_loss_fn(self, monkeypatch, sigma):
        """loss_fn.sigma must reflect the configured sigma value."""
        m = _make_model(monkeypatch, sigma=sigma)
        assert float(m.loss_fn.sigma) == pytest.approx(sigma, abs=1e-4)

    @pytest.mark.parametrize("w_drc,w_ps", [(1.0, 0.3), (2.0, 1.0), (0.5, 0.5)])
    def test_loss_weights_stored(self, monkeypatch, w_drc, w_ps):
        """loss_fn must store the configured DRC and PS loss weights."""
        m = _make_model(monkeypatch, w_drc=w_drc, w_ps=w_ps)
        assert m.loss_fn.w_drc == pytest.approx(w_drc)
        assert m.loss_fn.w_ps == pytest.approx(w_ps)

    def test_learnable_sigma_false_has_no_parameters(self, monkeypatch):
        """With learnable_sigma=False, loss_fn must expose no learnable parameters."""
        m = _make_model(monkeypatch, learnable_sigma=False)
        assert list(m.loss_fn.parameters()) == []

    def test_learnable_sigma_true_has_log_sigma_parameter(self, monkeypatch):
        """With learnable_sigma=True, loss_fn must expose log_sigma as an nn.Parameter."""
        m = _make_model(monkeypatch, learnable_sigma=True)
        assert isinstance(m.loss_fn.log_sigma, nn.Parameter)
        assert m.loss_fn.log_sigma.requires_grad is True


# ---------------------------------------------------------------------------
# Freeze / unfreeze schedule
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    def test_configure_optimizers_frozen_has_one_param_group(self, model):
        """When the encoder is frozen, configure_optimizers must return one param group."""
        opt = model.configure_optimizers()
        assert len(opt.param_groups) == 1

    def test_configure_optimizers_unfrozen_has_two_param_groups(self, model):
        """After unfreezing, configure_optimizers must return two param groups."""
        model._unfreeze_encoder()
        opt = model.configure_optimizers()
        assert len(opt.param_groups) == 2

    def test_unfreeze_sets_requires_grad_true(self, model):
        """_unfreeze_encoder must set requires_grad=True on all encoder parameters."""
        model._unfreeze_encoder()
        assert all(p.requires_grad for p in model._encoder_params())

    def test_unfreeze_sets_frozen_flag_false(self, model):
        """_unfreeze_encoder must update the _encoder_frozen flag."""
        model._unfreeze_encoder()
        assert model._encoder_frozen is False

    def test_freeze_after_unfreeze_restores_no_grad(self, model):
        """Re-freezing after an unfreeze must restore requires_grad=False on encoder params."""
        model._unfreeze_encoder()
        model._freeze_encoder()
        assert all(not p.requires_grad for p in model._encoder_params())
        assert model._encoder_frozen is True

    def test_encoder_and_head_params_are_disjoint(self, model):
        """_encoder_params and _head_params must not share any tensors."""
        enc_ids = {id(p) for p in model._encoder_params()}
        head_ids = {id(p) for p in model._head_params()}
        assert enc_ids.isdisjoint(head_ids)

    def test_encoder_and_head_params_cover_all_model_params(self, model):
        """_encoder_params and _head_params together must account for all model parameters."""
        all_ids = {id(p) for p in model.model.parameters()}
        covered = {id(p) for p in model._encoder_params()} | {id(p) for p in model._head_params()}
        assert covered == all_ids


# ---------------------------------------------------------------------------
# Checkpoint errors
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    def test_missing_checkpoint_raises_file_not_found(self):
        """Passing a nonexistent checkpoint path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="CheMeleon checkpoint not found"):
            ChemPropLightningModule(chempeleon_ckpt_path="/no/such/file.ckpt")
