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

    def test_plan_help_shows_plan_specific_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["plan", "--help"])
        assert result.exit_code == 0
        for flag in ("--training-csv", "--candidate-csv", "--output-csv"):
            assert flag in result.output

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
        cfg.write_text("data:\n  ground_truth_csv: ''\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code != 0
        assert "No such option: --config" in result.output

    def test_empty_ground_truth_csv_exits_one(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("data:\n  ground_truth_csv: ''\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        assert "ground_truth_csv must be set" in _result_text(result)

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
        cfg.write_text(f"data:\n  ground_truth_csv: {csv_path}\n")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["simulate", "--config", str(cfg), "--output-dir", str(tmp_path / "out")],
        )
        assert result.exit_code == 1
        if csv_content is None:
            assert "data.ground_truth_csv not found" in _result_text(result)
        else:
            assert "Failed to read data.ground_truth_csv" in _result_text(result)

    def test_custom_column_names_accepted(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("mol,potency\nc1ccccc1,5.0\nCCO,7.0\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "data:\n"
            f"  ground_truth_csv: {csv_file}\n"
            "  smiles_column: mol\n"
            "  pec50_column: potency\n"
            "model:\n"
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
            f"data:\n  ground_truth_csv: {csv_file}\n  smiles_column: nonexistent_col\n"
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

    def test_plan_writes_ranked_csv(self, tmp_path, monkeypatch):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text(
            "smiles,relation,value\n"
            "CCO,>=,5.0\n"
            "c1ccccc1,==,8.1\n"
        )
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\nCCCC\n")
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
            "data:\n"
            "  smiles_column: smiles\n"
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

        output_csv = tmp_path / "plan.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
                "--output-csv",
                str(output_csv),
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
            np.maximum(
                written["PS Score"].to_numpy(), written["DRC Score"].to_numpy()
            ),
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
        cfg.write_text("model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("model:\n  fast: true\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("dashboard:\n  enabled: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("data:\n  smiles_column: smiles\nmodel:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "invalid candidate SMILES" in result.output

    def test_plan_rejects_unsupported_left_ps_plus_drc_training_mix(
        self, tmp_path
    ):
        training_csv = tmp_path / "training.csv"
        training_csv.write_text("smiles,relation,value\nCCO,<,5.0\nCCO,==,6.0\n")
        candidate_csv = tmp_path / "candidates.csv"
        candidate_csv.write_text("smiles\nCCN\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  fast: false\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "--config",
                str(cfg),
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
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
        cfg.write_text("model:\n  fast: false\n")

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
                "--training-csv",
                str(training_csv),
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 1
        assert "predictions must contain only finite values" in _result_text(result)


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
        assert cfg.active_learning_loop.k_per_iteration == 10
        assert cfg.model.fast is False
        assert cfg.model.reset_weights_on_refit is False
