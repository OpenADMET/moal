"""Tests for moal.cli — the installed ``moal`` command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from moal.cli import main


class TestCLIHelp:
    """Tests that the --help flag exits cleanly and exposes all expected options."""
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
    """Tests that missing or nonexistent --config paths produce a non-zero exit code."""
    @pytest.mark.parametrize("args", [
        [],
        ["--config", "/nonexistent/path/config.yaml"],
    ])
    def test_bad_config_exits_nonzero(self, args):
        """Invoking without --config or with a non-existent path must fail."""
        runner = CliRunner()
        result = runner.invoke(main, args)
        assert result.exit_code != 0


class TestCLINoGroundTruth:
    """Tests that an empty ground_truth_csv path triggers a clean exit-1 error."""
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
    """Tests that a missing or malformed CSV file triggers a clean exit-1 error."""
    @pytest.mark.parametrize("csv_content", [
        None,  # missing file (config points at a non-existent path)
        'smiles,pec50\n"unclosed quote,5.0\n',  # malformed CSV
    ])
    def test_bad_csv_exits_one(self, tmp_path, csv_content):
        """A config pointing at a missing or malformed CSV must exit with code 1."""
        if csv_content is None:
            csv_path = tmp_path / "no_such_file.csv"
        else:
            csv_path = tmp_path / "bad.csv"
            csv_path.write_text(csv_content)
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"data:\n  ground_truth_csv: {csv_path}\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1


class TestCLICustomColumns:
    """Tests that custom smiles_column/pec50_column config keys are correctly forwarded to the oracle."""
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
    """Tests that the bundled examples/default_config.yaml is valid and loadable end-to-end."""
    def test_example_config_is_valid_yaml_with_required_sections(self):
        """examples/default_config.yaml must parse as valid YAML and contain all
        top-level PipelineConfig sections."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        assert config_path.exists(), f"examples/default_config.yaml not found at {config_path}"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
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
        assert cfg.active_learning_loop.n_iterations == 20
        assert cfg.active_learning_loop.k_per_iteration == 10
        assert cfg.model.fast is False

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
