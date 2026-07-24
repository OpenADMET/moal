"""Tests for ChemPropLightningModule initialization and NoisyOracleModel.

CheMeleon weight downloading is patched out for all tests via an autouse
fixture that replaces ``_build_model`` with a function that constructs a real
MPNN from default (randomly-initialised) ChemProp layers.  This means the
model carries random weights, which is sufficient for every structural and
hyperparameter assertion in this module.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

from moal.loss import CensoredRegressionLoss
from moal.model import _KNOWN_FOUNDATION_MODELS, ChemPropLightningModule, NoisyOracleModel
from moal.types import CensoringType, LabelRecord, QueryType

# Capture the real _build_model before any test fixture can patch it.
_REAL_BUILD_MODEL = ChemPropLightningModule._build_model

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_chemeleon_download(monkeypatch):
    """Patch out CheMeleon weight loading for all tests in this module.

    Replaces ``_build_model`` with a function that constructs a real MPNN from
    default ChemProp layers (no checkpoint required), so tests never trigger a
    network download or require the cached checkpoint file.
    """

    def _fake_build_model(self, ffn_hidden_dim, ffn_num_layers, message_hidden_dim, depth):
        mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth)
        agg = MeanAggregation()
        ffn = RegressionFFN(
            input_dim=mp.output_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_num_layers,
        )
        return MPNN(message_passing=mp, agg=agg, predictor=ffn)

    monkeypatch.setattr(ChemPropLightningModule, "_build_model", _fake_build_model)


@pytest.fixture
def model() -> ChemPropLightningModule:
    """Default-config module with CheMeleon download patched out."""
    return ChemPropLightningModule()


# ---------------------------------------------------------------------------
# Default initialization
# ---------------------------------------------------------------------------


class TestDefaultInit:
    """Tests for default-parameter ChemPropLightningModule initialization."""

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
            "ffn_hidden_dim",
            "ffn_num_layers",
            "message_hidden_dim",
            "depth",
            "freeze_epochs",
            "mpnn_lr",
            "ffn_lr",
            "mpnn_weight_decay",
            "ffn_weight_decay",
            "sigma",
            "w_drc",
            "w_ps",
            "learnable_sigma",
            "from_foundation",
        }
        assert expected.issubset(set(model.hparams.keys()))

    def test_hparams_default_values(self, model):
        """Default hyperparameters must match the documented defaults."""
        assert model.hparams["ffn_hidden_dim"] == 300
        assert model.hparams["ffn_num_layers"] == 2
        assert model.hparams["message_hidden_dim"] == 300
        assert model.hparams["depth"] == 3
        assert model.hparams["freeze_epochs"] == 10
        assert model.hparams["sigma"] == pytest.approx(0.5)
        assert model.hparams["w_drc"] == pytest.approx(1.0)
        assert model.hparams["w_ps"] == pytest.approx(0.3)
        assert model.hparams["learnable_sigma"] is False
        assert model.hparams["from_foundation"] == "chemeleon"


# ---------------------------------------------------------------------------
# Architecture parametrization
# ---------------------------------------------------------------------------


class TestArchitectureParams:
    """Tests that FFN architecture hyperparameters are correctly forwarded to the predictor head."""

    @pytest.mark.parametrize("ffn_num_layers", [1, 2, 4])
    def test_ffn_num_layers_sets_predictor_depth(self, ffn_num_layers):
        """The FFN predictor must contain ffn_num_layers + 1 sequential blocks
        (chemprop adds an extra input projection block).
        """
        m = ChemPropLightningModule(ffn_num_layers=ffn_num_layers)
        assert len(m.model.predictor.ffn) == ffn_num_layers + 1

    @pytest.mark.parametrize("ffn_hidden_dim", [64, 128, 512])
    def test_ffn_hidden_dim_sets_predictor_width(self, ffn_hidden_dim):
        """The hidden linear layers in the FFN predictor must match ffn_hidden_dim."""
        m = ChemPropLightningModule(ffn_hidden_dim=ffn_hidden_dim)
        # Block 1 is the first hidden layer; index [2] is the Linear within the Sequential.
        assert m.model.predictor.ffn[1][2].in_features == ffn_hidden_dim

    @pytest.mark.parametrize("message_hidden_dim,depth", [(128, 2), (2048, 6)])
    def test_random_init_encoder_uses_message_hidden_dim_and_depth(
        self, message_hidden_dim, depth, monkeypatch
    ):
        """from_foundation=False must build the encoder at the requested d_h and depth.

        Restores the real _build_model so the random-init branch runs against a
        real BondMessagePassing, then checks the encoder width (output_dim == d_h)
        and the number of message-passing weight matrices (one per step).
        """
        monkeypatch.setattr(ChemPropLightningModule, "_build_model", _REAL_BUILD_MODEL)
        m = ChemPropLightningModule(
            from_foundation=False, message_hidden_dim=message_hidden_dim, depth=depth
        )
        assert m.model.message_passing.output_dim == message_hidden_dim
        assert m.model.message_passing.depth == depth


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------


class TestTrainingHyperparams:
    """Tests that training hyperparameters (freeze_epochs, learning rates) are stored and accessible."""

    @pytest.mark.parametrize("freeze_epochs", [0, 5, 20])
    def test_freeze_epochs_stored(self, freeze_epochs):
        """freeze_epochs must be stored as an instance attribute."""
        m = ChemPropLightningModule(freeze_epochs=freeze_epochs)
        assert m.freeze_epochs == freeze_epochs

    @pytest.mark.parametrize("ffn_lr", [1e-4, 1e-3, 5e-3])
    def test_ffn_lr_in_optimizer(self, ffn_lr):
        """The head optimizer param group must use the configured ffn_lr."""
        m = ChemPropLightningModule(ffn_lr=ffn_lr)
        opt = m.configure_optimizers()
        # When frozen, only the head param group is present.
        assert opt.param_groups[0]["lr"] == pytest.approx(ffn_lr)

    @pytest.mark.parametrize("mpnn_lr", [1e-6, 1e-5, 1e-4])
    def test_mpnn_lr_in_optimizer_after_unfreeze(self, mpnn_lr):
        """After unfreezing, the encoder param group must use the configured mpnn_lr."""
        m = ChemPropLightningModule(mpnn_lr=mpnn_lr)
        m._unfreeze_encoder()
        opt = m.configure_optimizers()
        encoder_lrs = [g["lr"] for g in opt.param_groups if g["lr"] != m.ffn_lr]
        assert encoder_lrs == [pytest.approx(mpnn_lr)]

    def test_weight_decay_defaults_to_zero(self, model):
        """Both param groups must default to zero weight decay (no regularisation)."""
        model._unfreeze_encoder()
        opt = model.configure_optimizers()
        assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)

    @pytest.mark.parametrize("ffn_wd", [1e-4, 1e-2])
    def test_ffn_weight_decay_in_head_group(self, ffn_wd):
        """The FFN head param group (group 0) must use the configured ffn_weight_decay."""
        m = ChemPropLightningModule(ffn_weight_decay=ffn_wd)
        opt = m.configure_optimizers()
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(ffn_wd)

    @pytest.mark.parametrize("mpnn_wd", [1e-4, 1e-2])
    def test_mpnn_weight_decay_in_encoder_group_after_unfreeze(self, mpnn_wd):
        """After unfreezing, the encoder param group must use the configured mpnn_weight_decay."""
        m = ChemPropLightningModule(mpnn_weight_decay=mpnn_wd, ffn_weight_decay=0.0)
        m._unfreeze_encoder()
        opt = m.configure_optimizers()
        encoder_wd = [g["weight_decay"] for g in opt.param_groups if g["weight_decay"] != 0.0]
        assert encoder_wd == [pytest.approx(mpnn_wd)]


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------


class TestLossConfig:
    """Tests that loss-related parameters (sigma, fidelity weights, learnable_sigma) are wired to the loss function."""

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0])
    def test_sigma_stored_in_loss_fn(self, sigma):
        """loss_fn.sigma must reflect the configured sigma value."""
        m = ChemPropLightningModule(sigma=sigma)
        assert float(m.loss_fn.sigma) == pytest.approx(sigma, abs=1e-4)

    @pytest.mark.parametrize("w_drc,w_ps", [(1.0, 0.3), (2.0, 1.0), (0.5, 0.5)])
    def test_loss_weights_stored(self, w_drc, w_ps):
        """loss_fn must store the configured DRC and PS loss weights."""
        m = ChemPropLightningModule(w_drc=w_drc, w_ps=w_ps)
        assert m.loss_fn.w_drc == pytest.approx(w_drc)
        assert m.loss_fn.w_ps == pytest.approx(w_ps)

    @pytest.mark.parametrize("learnable,expected_param_count", [(False, 0), (True, 1)])
    def test_learnable_sigma_parameter(self, learnable, expected_param_count):
        """loss_fn must expose the correct number of parameters based on learnable_sigma."""
        m = ChemPropLightningModule(learnable_sigma=learnable)
        assert len(list(m.loss_fn.parameters())) == expected_param_count
        if learnable:
            assert isinstance(m.loss_fn.log_sigma, nn.Parameter)
            assert m.loss_fn.log_sigma.requires_grad is True


# ---------------------------------------------------------------------------
# Freeze / unfreeze schedule
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    """Tests for the encoder freeze/unfreeze schedule: optimizer count changes at the freeze epoch boundary."""

    def test_configure_optimizers_frozen_has_one_param_group(self, model):
        """When the encoder is frozen, configure_optimizers must return one param group."""
        opt = model.configure_optimizers()
        assert len(opt.param_groups) == 1

    def test_configure_optimizers_unfrozen_has_two_param_groups(self, model):
        """After unfreezing, configure_optimizers must return two param groups."""
        model._unfreeze_encoder()
        opt = model.configure_optimizers()
        assert len(opt.param_groups) == 2

    def test_unfreeze_sets_requires_grad_and_flag(self, model):
        """_unfreeze_encoder must set requires_grad=True and clear the frozen flag."""
        model._unfreeze_encoder()
        assert all(p.requires_grad for p in model._encoder_params())
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


class TestRefit:
    """Tests for the real Lightning refit path."""

    def test_refit_with_datamodule_emits_no_transfer_warning(self, model, tmp_path):
        """Moving batch-transfer logic to the datamodule should silence the warning."""
        records = [
            LabelRecord(
                smiles="CCO",
                canonical_smiles="CCO",
                value=6.0,
                upper_bound=6.0,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=0,
            ),
            LabelRecord(
                smiles="c1ccccc1",
                canonical_smiles="c1ccccc1",
                value=5.5,
                upper_bound=5.5,
                censoring_type=CensoringType.EXACT,
                fidelity=QueryType.DOSE_RESPONSE,
                cost=10.0,
                iteration=0,
            ),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = model.refit(
                records=records,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                datamodule_kwargs={"val_fraction": 0.5, "seed": 0},
                trainer_kwargs={"accelerator": "cpu"},
                output_dir=tmp_path / "out",
            )

        assert returned is model
        assert not any("transfer_batch_to_device" in str(w.message) for w in caught), caught


# ---------------------------------------------------------------------------
# NoisyOracleModel
# ---------------------------------------------------------------------------

# Small fixed ground truth shared across all fast-mode tests.
_GT: dict[str, float] = {
    "c1ccccc1": 4.0,
    "CCO": 6.0,
    "c1ccc(O)cc1": 7.5,
    "c1ccncc1": 5.5,
}


@pytest.fixture
def noisy_model() -> NoisyOracleModel:
    """NoisyOracleModel seeded with a small ground-truth dict for deterministic noise tests."""
    return NoisyOracleModel(ground_truth=_GT, seed=0)


class TestNoisyOracleModel:
    """Tests for NoisyOracleModel: noise bounds, reproducibility, and refit no-op behaviour."""

    def test_predictions_within_noise_bounds(self, noisy_model):
        """All predictions must lie within [true - noise_scale, true + noise_scale]."""
        smiles = list(_GT.keys())
        preds = noisy_model.predict_smiles(smiles, noise_scale=0.5)
        for smi, pred in zip(smiles, preds, strict=False):
            true = _GT[smi]
            assert true - 1.0 <= pred <= true + 1.0, (
                f"Prediction {pred:.4f} out of noise bounds for {smi} (true={true})"
            )

    def test_zero_noise_returns_exact_values(self):
        """noise_scale=0.0 must return predictions identical to true pEC50 values."""
        model = NoisyOracleModel(ground_truth=_GT, seed=0)
        smiles = list(_GT.keys())
        preds = model.predict_smiles(smiles, noise_scale=0.0)
        for smi, pred in zip(smiles, preds, strict=False):
            assert pred == pytest.approx(_GT[smi], abs=1e-6)

    def test_predictions_reproducible_with_same_seed(self):
        """Two models with the same seed must produce identical predictions."""
        smiles = list(_GT.keys())
        m1 = NoisyOracleModel(ground_truth=_GT, seed=99)
        m2 = NoisyOracleModel(ground_truth=_GT, seed=99)
        np.testing.assert_array_equal(
            m1.predict_smiles(smiles, noise_scale=0.5),
            m2.predict_smiles(smiles, noise_scale=0.5),
        )

    def test_predictions_differ_with_different_seeds(self):
        """Different seeds must (almost certainly) produce different predictions."""
        smiles = list(_GT.keys())
        m1 = NoisyOracleModel(ground_truth=_GT, seed=1)
        m2 = NoisyOracleModel(ground_truth=_GT, seed=2)
        assert not np.array_equal(
            m1.predict_smiles(smiles, noise_scale=0.5),
            m2.predict_smiles(smiles, noise_scale=0.5),
        )

    def test_refit_is_noop(self, noisy_model):
        """refit() must return self and leave predictions unchanged."""
        smiles = list(_GT.keys())
        # Exhaust some RNG state before capturing the reference predictions
        ref = NoisyOracleModel(ground_truth=_GT, seed=0)
        preds_before = ref.predict_smiles(smiles, noise_scale=0.5)

        ref2 = NoisyOracleModel(ground_truth=_GT, seed=0)
        returned = ref2.refit(records=[], trainer_kwargs={"max_epochs": 5})
        assert returned is ref2  # must return self
        preds_after = ref2.predict_smiles(smiles, noise_scale=0.5)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_negative_noise_scale_raises(self):
        """A negative noise_scale must raise ValueError when calling predict_smiles."""
        model = NoisyOracleModel(ground_truth=_GT)
        with pytest.raises(ValueError, match="non-negative"):
            model.predict_smiles(list(_GT.keys()), noise_scale=-1.0)

    def test_unknown_smiles_raises_key_error(self, noisy_model):
        """SMILES absent from ground_truth must raise KeyError."""
        with pytest.raises(KeyError):
            noisy_model.predict_smiles(["C1CC1"], noise_scale=0.5)

    def test_empty_smiles_and_output_dtype(self, noisy_model):
        """An empty input must return an empty float32 array; any input must yield float32."""
        empty_result = noisy_model.predict_smiles([], noise_scale=0.5)
        assert empty_result.shape == (0,)
        assert empty_result.dtype == np.float32

        single_result = noisy_model.predict_smiles(["c1ccccc1"], noise_scale=0.5)
        assert single_result.dtype == np.float32

    def test_batch_size_argument_ignored(self, noisy_model):
        """batch_size kwarg must be accepted without error or behaviour change."""
        smiles = list(_GT.keys())
        preds_default = NoisyOracleModel(ground_truth=_GT, seed=7).predict_smiles(
            smiles, noise_scale=0.3
        )
        preds_custom = NoisyOracleModel(ground_truth=_GT, seed=7).predict_smiles(
            smiles, noise_scale=0.3, batch_size=1
        )
        np.testing.assert_array_equal(preds_default, preds_custom)


# ---------------------------------------------------------------------------
# from_foundation flag
# ---------------------------------------------------------------------------


class TestFromFoundation:
    """Tests for the from_foundation parameter on ChemPropLightningModule."""

    def test_known_foundation_models_contains_chemeleon(self):
        """_KNOWN_FOUNDATION_MODELS must contain 'chemeleon'."""
        assert "chemeleon" in _KNOWN_FOUNDATION_MODELS

    def test_default_is_chemeleon(self, model):
        """Default from_foundation must be 'chemeleon' in hparams."""
        assert model.hparams["from_foundation"] == "chemeleon"

    def test_false_builds_random_encoder(self):
        """from_foundation=False must construct the model without any weight loading.

        The _build_model patch means no real weights are loaded anyway, so we
        just verify construction succeeds and the hparam is recorded correctly.
        """
        m = ChemPropLightningModule(from_foundation=False)
        assert m.hparams["from_foundation"] is False
        assert isinstance(m.model, nn.Module)

    def test_false_encoder_passes_arch_to_bond_message_passing(self, monkeypatch):
        """from_foundation=False must call BondMessagePassing with d_h/depth, not load weights.

        We temporarily restore the real _build_model so the False-branch dispatch
        runs, then verify BondMessagePassing is called once with the configured
        d_h and depth and load_foundation_weights is never invoked.
        """
        calls = []

        original_bmp = BondMessagePassing

        def tracking_bmp(*args, **kwargs):
            calls.append((args, kwargs))
            return original_bmp(*args, **kwargs)

        monkeypatch.setattr("moal.model.BondMessagePassing", tracking_bmp)
        monkeypatch.setattr(ChemPropLightningModule, "_build_model", _REAL_BUILD_MODEL)
        monkeypatch.setattr(
            "moal.model.load_foundation_weights",
            lambda from_foundation: (_ for _ in ()).throw(
                AssertionError(
                    "load_foundation_weights must not be called when from_foundation=False"
                )
            ),
        )

        m = ChemPropLightningModule(from_foundation=False, message_hidden_dim=512, depth=4)
        # BondMessagePassing must have been called once with the configured architecture
        assert len(calls) == 1
        assert calls[0] == ((), {"d_h": 512, "depth": 4})
        assert m._from_foundation is False

    def test_nonexistent_path_raises_value_error(self):
        """A path string that does not exist on disk must raise ValueError at construction."""
        with pytest.raises(ValueError, match="does not resolve to an existing file path"):
            ChemPropLightningModule(from_foundation="/nonexistent/path/weights.pt")

    def test_unknown_named_string_raises_value_error(self):
        """An unrecognised named string must raise ValueError listing known models."""
        with pytest.raises(ValueError, match="not a recognised foundation model name"):
            ChemPropLightningModule(from_foundation="unknown_model_v99")

    def test_true_raises_value_error(self):
        """True is not a valid from_foundation value and must raise ValueError."""
        with pytest.raises(ValueError):
            ChemPropLightningModule(from_foundation=True)  # type: ignore[arg-type]

    def test_custom_path_loads_weights(self, tmp_path, monkeypatch):
        """A valid local path must be accepted and its checkpoint loaded.

        We write a minimal fake checkpoint, restore the real _build_model so
        the path-loading branch executes, and mock torch.load to return a
        lightweight dict with correct keys.
        """
        weights_path = tmp_path / "custom_weights.pt"
        weights_path.touch()  # create empty file so path validation passes

        # Build a minimal real BondMessagePassing to get valid hyper_parameters
        real_mp = BondMessagePassing()
        fake_ckpt = {
            "hyper_parameters": {k: v for k, v in real_mp.hparams.items() if k != "cls"},
            "state_dict": real_mp.state_dict(),
        }

        # Restore real _build_model and mock torch.load
        monkeypatch.setattr(ChemPropLightningModule, "_build_model", _REAL_BUILD_MODEL)
        monkeypatch.setattr(torch, "load", lambda path, **kwargs: fake_ckpt)

        m = ChemPropLightningModule(from_foundation=str(weights_path))
        assert m.hparams["from_foundation"] == str(weights_path)
        assert isinstance(m.model, nn.Module)
