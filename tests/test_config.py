"""Tests for pipeline configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from moal.config import AuxiliaryEncoderConfig, PipelineConfig


def _write_yaml(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(raw, f)
    return path


class TestAuxiliaryEncoderConfig:
    """Tests for AuxiliaryEncoderConfig's default-off behavior and round-trip through from_yaml."""

    def test_defaults_to_none_when_absent(self, tmp_path):
        """auxiliary_encoder must be None when the YAML has no auxiliary_encoder key."""
        path = _write_yaml(tmp_path, {"seed": 1})

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_encoder is None

    def test_round_trips_through_from_yaml(self, tmp_path):
        """An explicit auxiliary_encoder block must populate a matching AuxiliaryEncoderConfig."""
        path = _write_yaml(
            tmp_path,
            {
                "auxiliary_encoder": {
                    "freeze_epochs": 3,
                    "embedding_dim": 128,
                    "checkpoint_path": "aux_encoder.pt",
                }
            },
        )

        cfg = PipelineConfig.from_yaml(path)

        assert cfg.auxiliary_encoder == AuxiliaryEncoderConfig(
            freeze_epochs=3, embedding_dim=128, checkpoint_path="aux_encoder.pt"
        )
