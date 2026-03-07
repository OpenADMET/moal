"""Tests for moal.cli — the installed ``moal`` command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from moal.cli import main


class TestCLIHelp:
    def test_help_shows_usage_and_options(self):
        """--help must exit 0 and document the command purpose and all three options."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # Verify application-level content from the command docstring.
        assert "campaign" in result.output.lower()
        # All three user-facing options must be listed.
        for flag in ("--config", "--output-dir", "--verbose"):
            assert flag in result.output


class TestCLIMissingConfig:
    def test_missing_config_exits_nonzero(self):
        """Invoking without --config must fail (required option)."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_nonexistent_config_exits_nonzero(self):
        """A --config path that does not exist must fail."""
        runner = CliRunner()
        result = runner.invoke(main, ["--config", "/nonexistent/path/config.yaml"])
        assert result.exit_code != 0


class TestCLINoGroundTruth:
    def test_empty_ground_truth_csv_exits_one(self, tmp_path):
        """A config with data.ground_truth_csv='' must exit with code 1.

        Uses --output-dir to redirect all file output to tmp_path so the test
        does not leak a results/ directory into the pytest working directory.
        """
        cfg = tmp_path / "config.yaml"
        cfg.write_text("data:\n  ground_truth_csv: ''\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1


class TestCLIBadCSV:
    def test_missing_csv_file_exits_one(self, tmp_path):
        """A config pointing at a non-existent CSV must exit with code 1."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("data:\n  ground_truth_csv: /no/such/file.csv\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1

    def test_malformed_csv_exits_one(self, tmp_path):
        """A config pointing at a file that is not a valid CSV must exit with code 1."""
        bad_csv = tmp_path / "bad.csv"
        # Write something that will trip pandas' parser (unbalanced quotes).
        bad_csv.write_text('smiles,pec50\n"unclosed quote,5.0\n')
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"data:\n  ground_truth_csv: {bad_csv}\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1


class TestCLICustomColumns:
    def test_custom_column_names_accepted(self, tmp_path):
        """A config with smiles_column/pec50_column matching the CSV headers must not exit 1
        at the oracle-init stage (it will fail later at model init, which is fine here)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("mol,potency\nc1ccccc1,5.0\nCCO,7.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "data:\n"
            f"  ground_truth_csv: {csv_file}\n"
            "  smiles_column: mol\n"
            "  pec50_column: potency\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        # The oracle init succeeds with custom columns; any later failure (e.g.,
        # missing model checkpoint) must not be a column-mismatch error.
        assert "must contain columns" not in (result.output or "")
        if result.exception:
            assert "must contain columns" not in str(result.exception)

    def test_mismatched_column_names_exits_one(self, tmp_path):
        """A config whose smiles_column doesn't match the CSV headers must exit with code 1."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("smiles,pec50\nc1ccccc1,5.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "data:\n"
            f"  ground_truth_csv: {csv_file}\n"
            "  smiles_column: nonexistent_col\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1


class TestExampleConfig:
    def test_example_config_is_valid_yaml(self):
        """examples/default_config.yaml must parse as valid YAML."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        assert config_path.exists(), f"examples/default_config.yaml not found at {config_path}"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_example_config_has_required_sections(self):
        """The example config must contain all top-level PipelineConfig sections."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        for section in ("oracle", "model", "acquisition", "trainer", "dashboard", "data", "active_learning_loop"):
            assert section in data, f"Missing section '{section}' in default_config.yaml"

    def test_example_config_loads_as_pipeline_config(self):
        """PipelineConfig.from_yaml must accept the example config without errors."""
        from pathlib import Path
        from moal.config import PipelineConfig

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        cfg = PipelineConfig.from_yaml(config_path)
        assert cfg.oracle.cost_ps == 1.0
        assert cfg.oracle.cost_drc == 10.0
        assert cfg.oracle.ps_threshold == 5.0
        assert cfg.oracle.activity_threshold == 7.0
        assert cfg.model.freeze_epochs == 10
        assert cfg.acquisition.tau == 0.5
        assert cfg.active_learning_loop.n_iterations == 20
        assert cfg.active_learning_loop.k_per_iteration == 10
        assert cfg.test_set_size == 0.15
        assert cfg.trainer.val_fraction == 0.1
        assert cfg.trainer.split_seed == 42
        assert cfg.data.smiles_column == "smiles"
        assert cfg.data.pec50_column == "pec50"
        assert cfg.data.is_canonical is False
        # Fast mode defaults must be present in the example config.
        assert cfg.model.fast is False
        assert cfg.model.initial_error == pytest.approx(0.7)
        assert cfg.model.final_error == pytest.approx(0.5)

    def test_fast_mode_config_does_not_require_checkpoint(self, tmp_path):
        """A config with fast=true must not fail at the checkpoint-loading stage.

        The oracle is initialised successfully; the run will eventually fail
        (or succeed) at model instantiation, but it must not exit=1 with a
        'checkpoint not found' or 'chempeleon' error message.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "smiles,pec50\n"
            "c1ccccc1,4.0\n"
            "CCO,6.0\n"
            "c1ccc(O)cc1,7.5\n"
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "data:\n"
            f"  ground_truth_csv: {csv_file}\n"
            "model:\n"
            "  fast: true\n"
            "  initial_error: 0.7\n"
            "  final_error: 0.3\n"
            "dashboard:\n"
            "  enabled: false\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        # A checkpoint-related error is the failure mode we guard against.
        error_text = (result.output or "") + str(result.exception or "")
        assert "checkpoint" not in error_text.lower()
        assert "chempeleon" not in error_text.lower()
