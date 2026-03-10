"""Tests for moal.cli — the installed ``moal`` command."""

from __future__ import annotations

from unittest.mock import ANY, Mock

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import moal.cli as cli
from moal.cli import main


def _result_text(result) -> str:
    return f"{result.output}\n{result.exception or ''}"


def _simulate_config(
    *,
    input_csv: str = "",
    smiles_column: str = "smiles",
    pec50_column: str = "pec50",
    is_canonical: bool = False,
    extra: str = "",
) -> str:
    return (
        "data:\n"
        "  simulate:\n"
        f"    input_csv: {input_csv}\n"
        f"    smiles_column: {smiles_column}\n"
        f"    pec50_column: {pec50_column}\n"
        f"    is_canonical: {'true' if is_canonical else 'false'}\n" + extra
    )


def _plan_config(
    *,
    input_csv: str = "",
    output_csv: str = "campaign_state.csv",
    smiles_column: str = "smiles",
    relation_column: str = "relation",
    value_column: str = "value",
    is_canonical: bool = False,
    extra: str = "",
) -> str:
    return (
        "data:\n"
        "  plan:\n"
        f"    input_csv: {input_csv}\n"
        f"    output_csv: {output_csv}\n"
        f"    smiles_column: {smiles_column}\n"
        f"    relation_column: {relation_column}\n"
        f"    value_column: {value_column}\n"
        f"    is_canonical: {'true' if is_canonical else 'false'}\n"
        + extra
    )


class TestCLIHelp:
    """Tests for top-level and subcommand help text."""

    def test_root_help_shows_subcommands(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "simulate" in result.output
        assert "plan" in result.output

    @pytest.mark.parametrize("subcommand", ["simulate", "plan"])
    def test_subcommand_help_exits_cleanly(self, subcommand):
        runner = CliRunner()
        result = runner.invoke(main, [subcommand, "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--output-dir" in result.output
        assert "--verbose" in result.output

    def test_plan_help_does_not_show_removed_csv_flags(self):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0
        for flag in ("--training-csv", "--candidate-csv", "--output-csv"):
            assert flag not in result.output

    def test_missing_banner_asset_is_non_fatal(self, monkeypatch, caplog):
        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("assets/terminal.txt"):
                raise FileNotFoundError("missing banner")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        with caplog.at_level("DEBUG"):
            cli._print_banner()

        assert "Banner asset not found" in caplog.text


class TestCLIMissingConfig:
    """Tests explicit-subcommand config validation."""

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ([], "Usage: main [OPTIONS] COMMAND [ARGS]..."),
            (["simulate"], "Missing option '--config' / '-c'."),
            (
                ["simulate", "--config", "/nonexistent/path/config.yaml"],
                "File '/nonexistent/path/config.yaml' does not exist.",
            ),
        ],
    )
    def test_bad_config_exits_nonzero(self, args, message):
        runner = CliRunner()
        result = runner.invoke(main, args)
        assert result.exit_code != 0
        assert message in _result_text(result)


class TestSimulateCommand:
    """Tests for the simulation command."""

    def test_root_command_does_not_accept_simulate_options(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_simulate_config())
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0
        assert "No such option: --config" in result.output

    def test_empty_ground_truth_csv_exits_one(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_simulate_config())
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "input_csv must be set" in _result_text(result)

    @pytest.mark.parametrize(
        "csv_content",
        [
            None,
            'smiles,pec50\n"unclosed quote,5.0\n',
        ],
    )
    def test_bad_csv_exits_one(self, tmp_path, csv_content):
        if csv_content is None:
            csv_path = tmp_path / "no_such_file.csv"
        else:
            csv_path = tmp_path / "bad.csv"
            csv_path.write_text(csv_content)
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_simulate_config(input_csv=str(csv_path)))
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        if csv_content is None:
            assert "data.simulate.input_csv not found" in _result_text(result)
        else:
            assert "Failed to read data.simulate.input_csv" in _result_text(result)

    def test_custom_column_names_accepted(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("mol,potency\nc1ccccc1,5.0\nCCO,7.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _simulate_config(
                input_csv=str(csv_file),
                smiles_column="mol",
                pec50_column="potency",
            )
            + "model:\n"
            "  fast: true\n"
            "dashboard:\n"
            "  enabled: false\n"
            "active_learning_loop:\n"
            "  n_iterations: 1\n"
            "  k_per_iteration: 1\n"
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "iteration_metrics.csv").exists()
        assert (tmp_path / "out" / "cumulative_actives_curve.csv").exists()

    def test_mismatched_column_names_exits_one(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("smiles,pec50\nc1ccccc1,5.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _simulate_config(input_csv=str(csv_file), smiles_column="nonexistent_col")
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "must contain columns" in _result_text(result)


class TestPlanCommand:
    """Tests for the one-shot acquisition planning subcommand."""

    def test_plan_requires_configured_input_path(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(_plan_config() + "model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "data.plan.input_csv" in _result_text(result)

    def test_plan_writes_annotated_state_csv(self, tmp_path, monkeypatch):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text(
            "smiles,relation,value\n"
            "CCO,>=,5.0\n"
            "c1ccccc1,==,8.1\n"
            "CCN,,\n"
            "CCCC,,\n"
        )
        output_csv = tmp_path / "state_out.csv"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "oracle:\n"
            "  cost_ps: 1.0\n"
            "  cost_drc: 10.0\n"
            "  ps_threshold: 5.0\n"
            "acquisition:\n"
            "  ps_threshold: 5.0\n"
            "  target_threshold: 7.0\n"
            "  tau: 0.5\n"
            + _plan_config(
                input_csv=str(state_csv),
                output_csv=str(output_csv),
            )
            + "model:\n"
            "  fast: false\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "dashboard:\n"
            "  enabled: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        # unqueried: CCN, CCCC (2); ps upgrade: CCO (1) → 3 total predictions
        model.predict_smiles.return_value = np.array([5.0, 8.0, 6.5], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        assert output_csv.exists()
        written = pd.read_csv(output_csv)

        # Original columns must be preserved
        assert "smiles" in written.columns
        assert "relation" in written.columns
        assert "value" in written.columns

        # Four new score columns must be appended
        for col in ("ps_score", "drc_score", "overall_score", "recommendation"):
            assert col in written.columns

        # Unqueried rows should have scores; DRC terminal row should be NaN
        unqueried_mask = written["relation"].isna() | (written["relation"] == "")
        assert not written.loc[unqueried_mask, "drc_score"].isna().any()
        drc_terminal_mask = written["relation"] == "=="
        assert written.loc[drc_terminal_mask, "ps_score"].isna().all()
        assert written.loc[drc_terminal_mask, "drc_score"].isna().all()

        # PS hit row gets DRC upgrade score but no PS score
        ps_hit_mask = written["relation"] == ">="
        assert written.loc[ps_hit_mask, "ps_score"].isna().all()
        assert not written.loc[ps_hit_mask, "drc_score"].isna().any()
        assert (written.loc[ps_hit_mask, "recommendation"] == "drc").all()

        # model.refit called with deduplicated PS records (PS hit + DRC terminal)
        model.refit.assert_called_once_with(
            records=ANY,
            trainer_kwargs=ANY,
            datamodule_kwargs=ANY,
            reset_weights=False,
            output_dir=tmp_path / "out",
        )
        # inference_smiles = [unqueried...] + [ps_upgrade...]
        model.predict_smiles.assert_called_once_with(["CCN", "CCCC", "CCO"])

    def test_plan_rejects_empty_state_csv_with_no_training_data(self, tmp_path):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "did not contain any labeled records" in result.output

    def test_plan_rejects_fast_mode(self, tmp_path):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,6.0\nCCN,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "model:\n  fast: true\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "does not support model.fast=true" in result.output

    def test_plan_invalid_relation_schema_exits_one(self, tmp_path):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,??,6.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "dashboard:\n  enabled: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "relation must be one of" in result.output

    def test_plan_rejects_invalid_smiles(self, tmp_path):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,6.0\nnot-a-smiles,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "invalid SMILES" in result.output

    def test_plan_rejects_unsupported_left_ps_plus_drc_mix(self, tmp_path):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,<,5.0\nCCO,==,6.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "mixed-fidelity combination is unsupported in plan mode" in result.output

    def test_plan_surfaces_non_finite_prediction_failure(self, tmp_path, monkeypatch):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,6.0\nCCN,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv)) + "model:\n  fast: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([np.nan], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert "predictions must contain only finite values" in _result_text(result)

    def test_plan_accepts_custom_column_names(self, tmp_path, monkeypatch):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text(
            "compound,kind,potency\n"
            "CCO,>=,5.0\n"
            "c1ccccc1,==,8.1\n"
            "CCN,,\n"
        )
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "oracle:\n"
            "  cost_ps: 1.0\n"
            "  cost_drc: 10.0\n"
            "  ps_threshold: 5.0\n"
            "acquisition:\n"
            "  ps_threshold: 5.0\n"
            "  target_threshold: 7.0\n"
            "  tau: 0.5\n"
            + _plan_config(
                input_csv=str(state_csv),
                output_csv="out.csv",
                smiles_column="compound",
                relation_column="kind",
                value_column="potency",
            )
            + "model:\n"
            "  fast: false\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "dashboard:\n"
            "  enabled: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([5.0, 6.5], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        # inference_smiles = [CCN (unqueried)] + [CCO (ps upgrade)]
        model.predict_smiles.assert_called_once_with(["CCN", "CCO"])

    def test_plan_handles_all_terminal_compounds_writes_nan_scores(
        self, tmp_path, monkeypatch
    ):
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,7.2\nCCN,<,5.0\n")
        output_csv = tmp_path / "out.csv"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv), output_csv=str(output_csv))
            + "model:\n  fast: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        assert output_csv.exists()
        written = pd.read_csv(output_csv)
        # No inference targets — skip model training and prediction entirely
        model.refit.assert_not_called()
        model.predict_smiles.assert_not_called()
        assert written["ps_score"].isna().all()
        assert written["drc_score"].isna().all()


class TestExampleConfig:
    """Tests that the bundled examples/default_config.yaml remains valid."""

    def test_example_config_is_valid_yaml_with_required_sections(self):
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        assert config_path.exists(), (
            f"examples/default_config.yaml not found at {config_path}"
        )
        with open(config_path) as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict)
        for section in (
            "oracle",
            "model",
            "acquisition",
            "trainer",
            "dashboard",
            "data",
            "active_learning_loop",
        ):
            assert section in data, (
                f"Missing section '{section}' in default_config.yaml"
            )

    def test_example_config_loads_as_pipeline_config(self):
        from pathlib import Path

        from moal.config import PipelineConfig

        config_path = Path(__file__).parent.parent / "examples" / "default_config.yaml"
        cfg = PipelineConfig.from_yaml(config_path)
        assert cfg.oracle.cost_ps == 1.0
        assert cfg.oracle.cost_drc == 10.0
        assert cfg.active_learning_loop.n_iterations == 20
        assert cfg.active_learning_loop.k_per_iteration == 100
        assert cfg.model.fast is True
        assert cfg.model.reset_weights_on_refit is False
        assert cfg.data.simulate.input_csv == ""
        assert cfg.data.plan.output_csv == "campaign_state.csv"
        assert cfg.data.plan.input_csv == ""
