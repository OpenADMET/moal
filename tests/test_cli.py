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
    training_input_csv: str = "",
    candidate_pool_input_csv: str = "",
    output_csv: str = "acquisition_plan.csv",
    training_smiles_column: str = "smiles",
    training_relation_column: str = "relation",
    training_value_column: str = "value",
    training_is_canonical: bool = False,
    candidate_pool_smiles_column: str = "smiles",
    candidate_pool_is_canonical: bool = False,
    extra: str = "",
) -> str:
    return (
        "data:\n"
        "  plan:\n"
        f"    output_csv: {output_csv}\n"
        "    training:\n"
        f"      input_csv: {training_input_csv}\n"
        f"      smiles_column: {training_smiles_column}\n"
        f"      relation_column: {training_relation_column}\n"
        f"      value_column: {training_value_column}\n"
        f"      is_canonical: {'true' if training_is_canonical else 'false'}\n"
        "    candidate_pool:\n"
        f"      input_csv: {candidate_pool_input_csv}\n"
        f"      smiles_column: {candidate_pool_smiles_column}\n"
        f"      is_canonical: {'true' if candidate_pool_is_canonical else 'false'}\n"
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

    @pytest.mark.parametrize(
        ("config_text", "message"),
        [
            (
                _plan_config(candidate_pool_input_csv="candidates.csv"),
                "data.plan.training.input_csv",
            ),
            (
                _plan_config(training_input_csv="train.csv"),
                "data.plan.candidate_pool.input_csv",
            ),
        ],
    )
    def test_plan_requires_configured_input_paths(self, tmp_path, config_text, message):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(config_text + "model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 1
        assert message in _result_text(result)

    def test_plan_writes_ranked_csv(self, tmp_path, monkeypatch):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,>=,5.0\nc1ccccc1,==,8.1\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\nCCCC\n")
        output_csv = tmp_path / "plan.csv"
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
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
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
        model.predict_smiles.return_value = np.array([5.0, 8.0], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_csv.exists()
        written = pd.read_csv(output_csv)
        # With the pinned acquisition settings above, CCN at y_hat=5.0 has
        # maximal PS entropy (~0.693) and negligible DRC score, while CCCC at
        # y_hat=8.0 has a DRC score of sigmoid((8-7)/0.5)/10 (~0.088), so CCN
        # must rank first as a PS query and CCCC second as a DRC query.
        assert list(written.columns) == [
            "Rank",
            "Compound (SMILES)",
            "Query type",
            "PS Score",
            "DRC Score",
            "Overall Score",
        ]
        assert written["Rank"].tolist() == [1, 2]
        assert written["Compound (SMILES)"].tolist() == ["CCN", "CCCC"]
        assert written["Query type"].tolist() == ["PS", "DRC"]
        assert np.allclose(
            written["Overall Score"].to_numpy(),
            np.maximum(written["PS Score"].to_numpy(), written["DRC Score"].to_numpy()),
        )
        model.refit.assert_called_once_with(
            records=ANY,
            trainer_kwargs=ANY,
            datamodule_kwargs=ANY,
            reset_weights=False,
            output_dir=tmp_path / "out",
        )
        fit_records = model.refit.call_args.kwargs["records"]
        assert len(fit_records) == 2
        model.predict_smiles.assert_called_once_with(["CCN", "CCCC"])

    def test_plan_rejects_empty_training_csv(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "training CSV did not contain any labeled records" in result.output

    def test_plan_rejects_fast_mode(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: true\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "does not support model.fast=true" in result.output

    def test_plan_invalid_training_schema_exits_one(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,??,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "dashboard:\n  enabled: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "relation must be one of" in result.output

    def test_plan_rejects_overlap_between_training_and_candidates(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCO\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "already appear in the training set" in result.output

    def test_plan_rejects_empty_candidate_pool(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "candidate CSV did not contain any candidate SMILES" in result.output

    def test_plan_rejects_missing_candidate_smiles_column(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("compound\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
                candidate_pool_smiles_column="smiles",
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "candidate CSV must contain column 'smiles'" in result.output

    def test_plan_rejects_invalid_candidate_smiles(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nnot-a-smiles\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "invalid candidate SMILES" in result.output

    def test_plan_rejects_unsupported_left_ps_plus_drc_training_mix(self, tmp_path):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,<,5.0\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "mixed-fidelity combination is unsupported in plan mode" in result.output

    def test_plan_surfaces_non_finite_prediction_failure(self, tmp_path, monkeypatch):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
            )
            + "model:\n  fast: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([np.nan], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "predictions must contain only finite values" in _result_text(result)

    def test_plan_accepts_custom_training_and_candidate_columns(
        self, tmp_path, monkeypatch
    ):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("compound,kind,potency\nCCO,>=,5.0\nc1ccccc1,==,8.1\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("compound_id\nCCN\nCCCC\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            _plan_config(
                training_input_csv=str(training_csv),
                candidate_pool_input_csv=str(candidate_csv),
                output_csv="plan.csv",
                training_smiles_column="compound",
                training_relation_column="kind",
                training_value_column="potency",
                candidate_pool_smiles_column="compound_id",
            )
            + "oracle:\n"
            "  cost_ps: 1.0\n"
            "  cost_drc: 10.0\n"
            "  ps_threshold: 5.0\n"
            "acquisition:\n"
            "  ps_threshold: 5.0\n"
            "  target_threshold: 7.0\n"
            "  tau: 0.5\n"
            "model:\n"
            "  fast: false\n"
            "trainer:\n"
            "  max_epochs: 1\n"
            "dashboard:\n"
            "  enabled: false\n"
        )

        model = Mock(spec_set=["refit", "predict_smiles"])
        model.predict_smiles.return_value = np.array([5.0, 8.0], dtype=np.float32)
        monkeypatch.setattr("moal.cli._build_plan_model", lambda cfg: model)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["plan", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "plan.csv").exists()
        model.predict_smiles.assert_called_once_with(["CCN", "CCCC"])


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
        assert cfg.data.plan.output_csv == "acquisition_plan.csv"
        assert cfg.data.plan.training.input_csv == ""
        assert cfg.data.plan.candidate_pool.input_csv == ""
