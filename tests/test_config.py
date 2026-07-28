"""Tests for pipeline configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from lightning.pytorch.callbacks import EarlyStopping

from moal.config import AuxiliaryModelConfig, PipelineConfig, TrainerConfig


def _write_yaml(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(raw, f)
    return path


class TestAuxiliaryModelConfig:
    """Tests for AuxiliaryModelConfig's default-off behavior and round-trip through from_yaml."""

    def test_defaults_to_none_when_absent(self, tmp_path):
        """auxiliary_model must be None when the YAML has no auxiliary_model key."""
        path = _write_yaml(tmp_path, {"seed": 1})

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_model is None

    def test_round_trips_through_from_yaml(self, tmp_path):
        """An explicit auxiliary_model block must populate a matching AuxiliaryModelConfig."""
        path = _write_yaml(
            tmp_path,
            {
                "auxiliary_model": {
                    "freeze_epochs": 3,
                    "embedding_dim": 128,
                    "checkpoint_path": "aux_encoder.pt",
                }
            },
        )

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_model == AuxiliaryModelConfig(
            freeze_epochs=3, embedding_dim=128, checkpoint_path="aux_encoder.pt"
        )


class TestAuxiliaryTrainerConfig:
    """Tests for auxiliary_trainer's default and round-trip through from_yaml."""

    def test_defaults_to_trainer_config_defaults_when_absent(self, tmp_path):
        """auxiliary_trainer must default to TrainerConfig() when the YAML has no auxiliary_trainer key."""
        path = _write_yaml(tmp_path, {"seed": 1})

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_trainer == TrainerConfig()

    def test_round_trips_through_from_yaml(self, tmp_path):
        """An explicit auxiliary_trainer block must populate a matching TrainerConfig, independent of trainer."""
        path = _write_yaml(
            tmp_path,
            {
                "trainer": {"max_epochs": 30, "val_fraction": 0.0},
                "auxiliary_trainer": {"max_epochs": 15, "val_fraction": 0.2},
            },
        )

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_trainer == TrainerConfig(max_epochs=15, val_fraction=0.2)
        assert cfg.trainer == TrainerConfig(max_epochs=30, val_fraction=0.0)


class TestTrainerConfigEarlyStopping:
    """Tests for TrainerConfig.to_dict()'s early-stopping callback wiring."""

    def test_omits_callbacks_by_default(self):
        """to_dict() must not add a callbacks key when early_stopping is False."""
        kwargs = TrainerConfig().to_dict()

        assert "callbacks" not in kwargs

    def test_adds_early_stopping_callback_when_enabled(self):
        """to_dict() must add an EarlyStopping callback configured from the early_stopping_* fields."""
        kwargs = TrainerConfig(
            early_stopping=True,
            early_stopping_monitor="aux_val_loss",
            early_stopping_patience=3,
            early_stopping_mode="max",
            early_stopping_min_delta=0.01,
        ).to_dict()

        [callback] = kwargs["callbacks"]
        assert isinstance(callback, EarlyStopping)
        assert callback.monitor == "aux_val_loss"
        assert callback.patience == 3
        assert callback.mode == "max"
        assert callback.min_delta == 0.01
