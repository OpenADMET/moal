"""Tests for pipeline configuration."""

from __future__ import annotations

from lightning.pytorch.callbacks import EarlyStopping

from moal.config import TrainerConfig


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
            early_stopping_monitor="val_loss",
            early_stopping_patience=3,
            early_stopping_mode="max",
            early_stopping_min_delta=0.01,
        ).to_dict()

        [callback] = kwargs["callbacks"]
        assert isinstance(callback, EarlyStopping)
        assert callback.monitor == "val_loss"
        assert callback.patience == 3
        assert callback.mode == "max"
        assert callback.min_delta == 0.01
