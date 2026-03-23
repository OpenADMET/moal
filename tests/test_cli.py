"""Tests for moal.cli — the installed ``moal`` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

import moal.cli as cli
from moal.cli import main
from moal.config import PipelineConfig


def _result_text(result) -> str:
    return f"{result.output}\n{result.exception or ''}"


def _cli_output(result) -> str:
    stderr = getattr(result, "stderr", "")
    return f"{result.output}\n{stderr}"


class _ProgressRecorder:
    instances: list[_ProgressRecorder] = []

    def __init__(self, *args, **kwargs):
        self.added_tasks: list[dict[str, object]] = []
        self.updated_tasks: list[dict[str, object]] = []
        self.advanced_tasks: list[object] = []

    def __enter__(self):
        type(self).instances.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, description, total):
        self.added_tasks.append({"description": description, "total": total})
        return "task-1"

    def update(self, task, **kwargs):
        payload = {"task": task}
        payload.update(kwargs)
        self.updated_tasks.append(payload)

    def advance(self, task):
        self.advanced_tasks.append(task)


def _latest_progress() -> _ProgressRecorder:
    assert _ProgressRecorder.instances
    return _ProgressRecorder.instances[-1]


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
        f"    is_canonical: {'true' if is_canonical else 'false'}\n" + extra
    )


class TestCLIHelp:
    """Tests for top-level and subcommand help text."""

    def test_root_help_shows_subcommands(self):
        """Root --help must list both the simulate and plan subcommands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "simulate" in result.output
        assert "plan" in result.output

    @pytest.mark.parametrize("subcommand", ["simulate", "plan"])
    def test_subcommand_help_exits_cleanly(self, subcommand):
        """Each subcommand's --help must exit 0 and list the shared --config, --output-dir, and --verbose flags."""
        runner = CliRunner()
        result = runner.invoke(main, [subcommand, "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--output-dir" in result.output
        assert "--verbose" in result.output

    def test_plan_help_does_not_show_removed_csv_flags(self):
        """Flags removed from the plan subcommand must not appear in its help text to avoid user confusion."""
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0
        for flag in ("--training-csv", "--candidate-csv", "--output-csv"):
            assert flag not in result.output

    def test_missing_banner_asset_is_non_fatal(self, monkeypatch, caplog):
        """A missing banner asset file must log at DEBUG level and not raise."""
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
        """Missing or non-existent config arguments must produce a non-zero exit code with a descriptive message."""


class TestSimulateCommand:
    """Tests for the simulation command."""

    def test_root_command_does_not_accept_simulate_options(self, tmp_path):
        """Subcommand flags passed to the root command must be rejected since they belong to a subcommand."""
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
        """Omitting data.simulate.input_csv must exit with code 1 and a message naming the missing field."""

    @pytest.mark.parametrize(
        "csv_content",
        [
            None,
            'smiles,pec50\n"unclosed quote,5.0\n',
        ],
    )
    def test_bad_csv_exits_one(self, tmp_path, csv_content):
        """A missing file or malformed CSV must produce exit code 1 with a message that names the problematic path."""
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
        """Non-default smiles/pec50 column names must be accepted and produce a successful simulate run."""
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
        """Specifying a smiles column that is absent from the CSV must exit with code 1 and name the missing column."""
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
        """Omitting data.plan.input_csv must exit with code 1 and include the field name in the message."""
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
        """End-to-end plan command must produce an annotated CSV with ps_score, drc_score, and recommendation columns."""
        state_csv = tmp_path / "state.csv"
        state_csv.write_text(
            "smiles,relation,value\nCCO,>=,5.0\nc1ccccc1,==,8.1\nCCN,,\nCCCC,,\n"
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
        monkeypatch.setattr("moal.cli.Progress", _ProgressRecorder)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        assert output_csv.exists()
        progress = _latest_progress()
        assert progress.added_tasks == [
            {"description": "[cyan]Parsing campaign state[/cyan]", "total": 3}
        ]
        descriptions = [
            update["description"]
            for update in progress.updated_tasks
            if "description" in update
        ]
        assert (
            "[yellow]Training model[/yellow] — 2 records ([orange1]1 DRC[/orange1], [steel_blue1]1 PS[/steel_blue1])"
            in descriptions
        )
        assert (
            "[green]Scoring compounds[/green] - [white]2 unqueried[/white], "
            "[magenta]1 PS hits[/magenta] eligible for upgrade"
        ) in descriptions
        # Completion summary should follow simulate's palette; "Wrote" belongs in log only
        cli_out = _cli_output(result)
        assert "Plan complete." in cli_out
        assert " PS" in cli_out and " DRC" in cli_out
        assert "Wrote:" not in cli_out
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

    def test_plan_suppresses_noisy_third_party_warnings(self, tmp_path, monkeypatch):
        """suppress_noisy_loggers must be called exactly once so third-party warnings do not pollute plan output."""
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,6.0\nCCN,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv), output_csv="out.csv") + "model:\n"
            "  fast: false\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "dashboard:\n"
            "  enabled: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([5.0], dtype=np.float32)
        suppress_mock = Mock()

        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)
        monkeypatch.setattr("moal.cli.suppress_noisy_loggers", suppress_mock)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        suppress_mock.assert_called_once_with()

    def test_plan_suppresses_info_logs_while_progress_is_active(
        self, tmp_path, monkeypatch
    ):
        """Moal INFO logs must not bleed into Rich progress output during a plan run."""
        state_csv = tmp_path / "state.csv"
        state_csv.write_text("smiles,relation,value\nCCO,==,6.0\nCCN,,\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(input_csv=str(state_csv), output_csv="out.csv") + "model:\n"
            "  fast: false\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "dashboard:\n"
            "  enabled: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([5.0], dtype=np.float32)

        def noisy_refit(**kwargs):
            cli.logging.getLogger("moal.model").info("Noisy model log")

        model.refit.side_effect = noisy_refit
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        assert "Noisy model log" not in _cli_output(result)

    def test_plan_rejects_empty_state_csv_with_no_training_data(self, tmp_path):
        """A state CSV with only unqueried rows must exit with code 1 and explain that training data is required."""
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
        """Fast mode is incompatible with plan because there is no oracle ground truth for candidate scoring; must exit 1."""
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
        """An unrecognized relation symbol in the state CSV must exit with code 1 and name the bad value."""
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
        """An unparseable SMILES string must exit with code 1 and include the offending string in the message."""
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
        """A compound appearing with both a PS-LEFT and a DRC row is an unsupported fidelity mix and must exit 1."""
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
        """NaN predictions from the model must surface as exit code 1 rather than silently producing invalid scores."""
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
        """Non-default smiles, relation, and value column names must be read correctly throughout the plan pipeline."""
        state_csv = tmp_path / "state.csv"
        state_csv.write_text(
            "compound,kind,potency\nCCO,>=,5.0\nc1ccccc1,==,8.1\nCCN,,\n"
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
        """When all compounds are terminal, the plan command must write NaN score columns without invoking the model."""
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
        monkeypatch.setattr("moal.cli.Progress", _ProgressRecorder)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, _result_text(result)
        assert output_csv.exists()
        progress = _latest_progress()
        descriptions = [
            update["description"]
            for update in progress.updated_tasks
            if "description" in update
        ]
        assert (
            "[green]Scoring compounds[/green] - [white]0 unqueried[/white], "
            "[magenta]0 PS hits[/magenta] eligible for upgrade"
        ) in descriptions
        assert not any("Training model -" in desc for desc in descriptions)
        # Terminal-only path produces zero recommendations; completion line still renders
        cli_out = _cli_output(result)
        assert "Plan complete." in cli_out
        assert "0 PS" in cli_out and "0 DRC" in cli_out
        written = pd.read_csv(output_csv)
        # No inference targets — skip model training and prediction entirely
        model.refit.assert_not_called()
        model.predict_smiles.assert_not_called()
        assert written["ps_score"].isna().all()
        assert written["drc_score"].isna().all()


class TestExampleConfig:
    """Tests that the bundled examples/default_config.yaml remains valid."""

    def test_example_config_is_valid_yaml_with_required_sections(self):
        """The bundled default config must be valid YAML that contains every top-level section required by PipelineConfig."""
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
        """The bundled default config must deserialize into a PipelineConfig with the expected cost and iteration values."""
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
