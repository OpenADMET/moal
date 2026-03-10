"""CLI entry point for running simulations and acquisition planning.

Install the package (``pip install -e .``) to get the ``moal`` command.
The explicit subcommands are::

    moal simulate --config examples/default_config.yaml
    moal plan --config examples/default_config.yaml --training-csv train.csv --candidate-csv pool.csv

"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable

import click
import pandas as pd

from moal.config import PipelineConfig
from moal.planning import (
    build_acquisition_plan_dataframe,
    parse_candidate_smiles,
    parse_training_records,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor

logger = logging.getLogger(__name__)


def _common_cli_options(*, required: bool) -> Callable:
    def decorator(func: Callable) -> Callable:
        wrapped = click.option(
            "--config",
            "-c",
            required=required,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Path to a YAML campaign configuration file.",
        )(func)
        wrapped = click.option(
            "--output-dir",
            "-o",
            default=None,
            type=click.Path(file_okay=False, path_type=Path),
            help="Override output directory from config.",
        )(wrapped)
        wrapped = click.option(
            "--verbose", "-v", is_flag=True, help="Enable DEBUG logging."
        )(wrapped)
        return wrapped

    return decorator


@click.group()
def main() -> None:
    """moal CLI with simulation and one-shot planning subcommands."""


@main.command()
@_common_cli_options(required=True)
def simulate(config: Path, output_dir: Path | None, verbose: bool) -> None:
    """Run a cost-aware synthetic active learning campaign."""
    _configure_logging(verbose)
    _print_banner()
    _print_welcome()

    from moal.evaluation import ModelMetric, PipelineEvaluator, scaffold_split
    from moal.loop import ActiveLearningLoop
    from moal.oracle import CostAwareOracle

    cfg = PipelineConfig.from_yaml(config)
    logger.info("Loaded config from %s", config)

    out_dir = _prepare_output_dir(cfg, output_dir)

    if not cfg.data.ground_truth_csv:
        raise click.ClickException("data.ground_truth_csv must be set in the config.")

    ground_truth_df = _read_csv(
        Path(cfg.data.ground_truth_csv), label="data.ground_truth_csv"
    )
    logger.info(
        "Loaded %d compounds from %s", len(ground_truth_df), cfg.data.ground_truth_csv
    )

    preprocessor = SMILESPreprocessor()
    oracle = CostAwareOracle(
        ground_truth_df=ground_truth_df,
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        ps_threshold=cfg.oracle.ps_threshold,
        upper_bound=cfg.oracle.upper_bound,
        smiles_column=cfg.data.smiles_column,
        pec50_column=cfg.data.pec50_column,
        is_canonical=cfg.data.is_canonical,
        preprocessor=preprocessor,
    )

    model = _build_simulation_model(cfg, oracle._ground_truth)
    acquisition = _build_acquisition(cfg)

    evaluator = PipelineEvaluator(
        activity_threshold=cfg.oracle.activity_threshold,
        upper_bound=cfg.oracle.upper_bound,
    )

    import numpy as np

    all_smiles = oracle.get_unlabeled_smiles()
    test_set = None
    if all_smiles and cfg.data.test_set_size > 0:
        _, test_idx = scaffold_split(
            all_smiles, test_size=cfg.data.test_set_size, seed=cfg.seed
        )
        if test_idx:
            test_smiles = [all_smiles[i] for i in test_idx]
            test_pec50 = [oracle._ground_truth[s] for s in test_smiles]
            test_set = (test_smiles, np.array(test_pec50, dtype=np.float32))

    dashboard = None
    if cfg.dashboard.enabled:
        from moal.dashboard import LiveDashboard
        from moal.evaluation import ModelMetric

        dashboard = LiveDashboard(
            n_iterations=cfg.active_learning_loop.n_iterations,
            n_compounds=len(oracle),
            model_metric=ModelMetric(cfg.dashboard.model_metric),
            figsize=cfg.dashboard.figsize,
            show=cfg.dashboard.show,
        )

    from moal.evaluation import ModelMetric

    loop = ActiveLearningLoop(
        oracle=oracle,
        model=model,
        acquisition=acquisition,
        evaluator=evaluator,
        preprocessor=preprocessor,
        trainer_kwargs=cfg.trainer.to_dict(),
        datamodule_kwargs=cfg.trainer.to_datamodule_kwargs(),
        dashboard=dashboard,
        test_set=test_set,
        model_metric=ModelMetric(cfg.dashboard.model_metric),
        initial_error=cfg.model.initial_error,
        final_error=cfg.model.final_error,
        reset_weights_on_refit=cfg.model.reset_weights_on_refit,
        output_dir=out_dir,
    )

    results = loop.run(
        n_iterations=cfg.active_learning_loop.n_iterations,
        k_per_iteration=cfg.active_learning_loop.k_per_iteration,
    )

    if dashboard is not None:
        final_path = out_dir / "dashboard_final.png"
        dashboard.save(final_path)
        logger.info("Final dashboard saved to %s", final_path)

        gif_path = out_dir / "dashboard_animation.gif"
        dashboard.save_gif(gif_path)
        dashboard.close()

    metrics_df = pd.DataFrame([r.metrics for r in results.iterations])
    metrics_path = out_dir / "iteration_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics written to %s", metrics_path)

    curve_df = evaluator.cumulative_actives_curve(oracle.labeled_records)
    curve_path = out_dir / "cumulative_actives_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info("Cumulative actives curve written to %s", curve_path)

    n_final_upgrades = int(results.final_metrics.get("n_ps_to_drc_upgrades", 0))
    n_final_drc = int(results.final_metrics.get("n_drc_queries", 0))
    n_final_ps = int(results.final_metrics.get("n_ps_queries", 0))
    n_true_actives = oracle.n_true_actives(evaluator.activity_threshold)
    upgrade_detail = f" ({n_final_upgrades} upgrades)" if n_final_upgrades > 0 else ""
    logger.info(
        "Campaign complete. Total cost: $%.2f | PS: %d | DRC: %d%s | Confirmed actives: %d (of %d)",
        results.total_cost,
        n_final_ps,
        n_final_drc,
        upgrade_detail,
        int(results.final_metrics.get("n_confirmed_actives", 0)),
        n_true_actives,
    )


@main.command()
@click.option(
    "--training-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV of labeled training records with columns: smiles, relation, value.",
)
@click.option(
    "--candidate-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV of unlabeled candidate compounds.",
)
@click.option(
    "--output-csv",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination for the ranked acquisition plan CSV.",
)
@_common_cli_options(required=True)
def plan(
    training_csv: Path,
    candidate_csv: Path,
    output_csv: Path | None,
    config: Path,
    output_dir: Path | None,
    verbose: bool,
) -> None:
    """Train on mixed-fidelity records and rank the next acquisition batch."""
    _configure_logging(verbose)
    _print_banner()
    _print_welcome()

    cfg = PipelineConfig.from_yaml(config)
    logger.info("Loaded config from %s", config)

    out_dir = _prepare_output_dir(cfg, output_dir)
    plan_path = output_csv or (out_dir / "acquisition_plan.csv")
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    training_df = _read_csv(training_csv, label="training CSV")
    candidate_df = _read_csv(candidate_csv, label="candidate CSV")

    preprocessor = SMILESPreprocessor()
    try:
        records = parse_training_records(
            training_df,
            cost_ps=cfg.oracle.cost_ps,
            cost_drc=cfg.oracle.cost_drc,
            upper_bound=cfg.oracle.upper_bound,
            preprocessor=preprocessor,
            is_canonical=cfg.data.is_canonical,
            expected_ps_threshold=cfg.oracle.ps_threshold,
        )
        candidate_smiles = parse_candidate_smiles(
            candidate_df,
            smiles_column=cfg.data.smiles_column,
            preprocessor=preprocessor,
            is_canonical=cfg.data.is_canonical,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not records:
        raise click.ClickException("training CSV did not contain any labeled records.")

    fit_records = training_records_for_refit(records)
    if not candidate_smiles:
        raise click.ClickException("candidate CSV did not contain any candidate SMILES.")

    labeled_smiles = {record.canonical_smiles for record in records}
    overlapping = [smiles for smiles in candidate_smiles if smiles in labeled_smiles]
    if overlapping:
        example = ", ".join(overlapping[:3])
        raise click.ClickException(
            "candidate CSV contains compounds that already appear in the training set; "
            f"first few overlaps: {example}"
        )

    model = _build_plan_model(cfg)
    model.refit(
        records=fit_records,
        trainer_kwargs=cfg.trainer.to_dict(),
        datamodule_kwargs=cfg.trainer.to_datamodule_kwargs(),
        reset_weights=cfg.model.reset_weights_on_refit,
        output_dir=out_dir,
    )
    predictions = model.predict_smiles(candidate_smiles)

    acquisition = _build_acquisition(cfg)
    plan_df = build_acquisition_plan_dataframe(
        candidate_smiles=candidate_smiles,
        predictions=predictions,
        acquisition=acquisition,
    )
    plan_df.to_csv(plan_path, index=False)
    logger.info("Acquisition plan written to %s", plan_path)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _print_banner() -> None:
    banner_path = Path(__file__).parent.parent / "assets" / "terminal.txt"
    try:
        with open(banner_path) as handle:
            print(handle.read() + "\n")
    except FileNotFoundError:
        logger.debug("Banner asset not found at %s; skipping terminal banner.", banner_path)


def _print_welcome() -> None:
    version = importlib.metadata.version("moal")
    print(
        f"Welcome to moal-v{version}: multi-objective active learning for drug discovery!\n"
    )


def _prepare_output_dir(cfg: PipelineConfig, output_dir: Path | None) -> Path:
    out_dir = Path(output_dir or cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config_used.yaml")
    return out_dir


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise click.ClickException(f"{label} not found: {path}") from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise click.ClickException(f"Failed to read {label} {path}: {exc}") from exc


def _build_acquisition(cfg: PipelineConfig):
    from moal.acquisition import CostAwareGreedyAcquisition

    return CostAwareGreedyAcquisition(
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        ps_threshold=cfg.acquisition.ps_threshold,
        target_threshold=cfg.acquisition.target_threshold,
        tau=cfg.acquisition.tau,
    )


def _build_simulation_model(
    cfg: PipelineConfig, ground_truth: dict[str, float]
):
    from moal.model import ChemPropLightningModule, NoisyOracleModel

    if cfg.model.fast:
        logger.info(
            "Fast mode enabled — using NoisyOracleModel with error ramp %.3f → %.3f over %d iterations",
            cfg.model.initial_error,
            cfg.model.final_error,
            cfg.active_learning_loop.n_iterations,
        )
        return NoisyOracleModel(ground_truth=ground_truth, seed=cfg.seed)

    return ChemPropLightningModule(
        ffn_hidden_size=cfg.model.ffn_hidden_size,
        ffn_num_layers=cfg.model.ffn_num_layers,
        freeze_epochs=cfg.model.freeze_epochs,
        lr_encoder=cfg.model.lr_encoder,
        lr_head=cfg.model.lr_head,
        sigma=cfg.model.sigma,
        w_drc=cfg.model.w_drc,
        w_ps=cfg.model.w_ps,
        learnable_sigma=cfg.model.learnable_sigma,
    )


def _build_plan_model(cfg: PipelineConfig):
    from moal.model import ChemPropLightningModule

    if cfg.model.fast:
        raise click.ClickException(
            "moal plan does not support model.fast=true because offline planning "
            "has no oracle ground truth for candidate compounds."
        )

    return ChemPropLightningModule(
        ffn_hidden_size=cfg.model.ffn_hidden_size,
        ffn_num_layers=cfg.model.ffn_num_layers,
        freeze_epochs=cfg.model.freeze_epochs,
        lr_encoder=cfg.model.lr_encoder,
        lr_head=cfg.model.lr_head,
        sigma=cfg.model.sigma,
        w_drc=cfg.model.w_drc,
        w_ps=cfg.model.w_ps,
        learnable_sigma=cfg.model.learnable_sigma,
    )


if __name__ == "__main__":
    main()
