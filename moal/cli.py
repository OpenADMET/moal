"""CLI entry point for running simulations and acquisition planning.

Install the package (``pip install -e .``) to get the ``moal`` command.
The explicit subcommands are::

    moal simulate --config examples/default_config.yaml
    moal plan --config examples/default_config.yaml

"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from pathlib import Path
from typing import Callable

import click
import lightning as L
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from moal.config import PipelineConfig
from moal.logging_config import suppress_noisy_loggers, temporary_log_level
from moal.planning import (
    annotate_campaign_state,
    parse_campaign_state,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor
from moal.types import QueryType

logger = logging.getLogger(__name__)
_console = Console(stderr=True)


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

    if not cfg.data.simulate.input_csv:
        raise click.ClickException("data.simulate.input_csv must be set in the config.")

    ground_truth_df = _read_csv(
        Path(cfg.data.simulate.input_csv),
        label="data.simulate.input_csv",
    )
    logger.info(
        "Loaded %d compounds from %s",
        len(ground_truth_df),
        cfg.data.simulate.input_csv,
    )

    preprocessor = SMILESPreprocessor()
    oracle = CostAwareOracle(
        ground_truth_df=ground_truth_df,
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        ps_threshold=cfg.oracle.ps_threshold,
        upper_bound=cfg.oracle.upper_bound,
        smiles_column=cfg.data.simulate.smiles_column,
        pec50_column=cfg.data.simulate.pec50_column,
        is_canonical=cfg.data.simulate.is_canonical,
        preprocessor=preprocessor,
    )

    # Setting all seeds
    L.seed_everything(cfg.seed, workers=True, verbose=False)

    # Build model
    model = _build_simulation_model(cfg, oracle._ground_truth)

    # Build acquisition function
    acquisition = _build_acquisition(cfg)

    evaluator = PipelineEvaluator(
        activity_threshold=cfg.oracle.activity_threshold,
        upper_bound=cfg.oracle.upper_bound,
    )

    import numpy as np

    all_smiles = oracle.get_unlabeled_smiles()
    test_set = None
    if all_smiles and cfg.data.simulate.test_set_size > 0:
        _, test_idx = scaffold_split(
            all_smiles, test_size=cfg.data.simulate.test_set_size, seed=cfg.seed
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
            port=cfg.dashboard.port,
            export_width=cfg.dashboard.export_width,
            export_height=cfg.dashboard.export_height,
            theme=cfg.dashboard.theme,
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

    # Write output CSVs immediately so results are available as soon as
    # computation finishes, independent of any dashboard export latency.
    metrics_df = pd.DataFrame([r.metrics for r in results.iterations])
    metrics_path = out_dir / "iteration_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics written to %s", metrics_path)

    curve_df = evaluator.cumulative_actives_curve(oracle.labeled_records)
    curve_path = out_dir / "cumulative_actives_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info("Cumulative actives curve written to %s", curve_path)

    if dashboard is not None:
        html_path = out_dir / "dashboard_animation.html"
        dashboard.save_html(html_path)

        # GIF export is O(N) kaleido renders — run in a background thread so
        # the Werkzeug server can be shut down without blocking on it.  The
        # thread is non-daemon so the process waits for it to finish cleanly.
        gif_path = out_dir / "dashboard_animation.gif"
        gif_thread = threading.Thread(
            target=dashboard.save_gif, args=(gif_path,), daemon=False
        )
        gif_thread.start()
        logger.info(
            "GIF animation rendering started in background",
        )

        dashboard.close()
        gif_thread.join()


@main.command()
@_common_cli_options(required=True)
def plan(config: Path, output_dir: Path | None, verbose: bool) -> None:
    """Train on mixed-fidelity records and score the next acquisition batch."""
    _configure_logging(verbose)
    _print_banner()
    _print_welcome()
    suppress_noisy_loggers()

    cfg = PipelineConfig.from_yaml(config)
    logger.info("Loaded config from %s", config)

    out_dir = _prepare_output_dir(cfg, output_dir)
    if not cfg.data.plan.input_csv:
        raise click.ClickException("data.plan.input_csv must be set in the config.")

    state_csv = Path(cfg.data.plan.input_csv)
    plan_path = _resolve_plan_output_path(cfg, out_dir)
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    state_df = _read_csv(state_csv, label="data.plan.input_csv")

    preprocessor = SMILESPreprocessor()
    import numpy as np

    _console.print("[bold]moal[/bold] plan starting")
    parse_description = "[cyan]Parsing campaign state[/cyan]"
    warning_message: str | None = None

    with temporary_log_level(logging.WARNING, ["moal"]):
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task(parse_description, total=3)

            try:
                state = parse_campaign_state(
                    state_df,
                    cost_ps=cfg.oracle.cost_ps,
                    cost_drc=cfg.oracle.cost_drc,
                    upper_bound=cfg.oracle.upper_bound,
                    preprocessor=preprocessor,
                    smiles_column=cfg.data.plan.smiles_column,
                    relation_column=cfg.data.plan.relation_column,
                    value_column=cfg.data.plan.value_column,
                    is_canonical=cfg.data.plan.is_canonical,
                    expected_ps_threshold=cfg.oracle.ps_threshold,
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            progress.advance(task)

            if not state.training_records:
                raise click.ClickException(
                    "state CSV did not contain any labeled records."
                )

            fit_records = training_records_for_refit(state.training_records)
            acquisition = _build_acquisition(cfg)
            n_inference = len(state.unqueried_rows) + len(state.ps_upgrade_rows)

            scoring_description = (
                "[green]Scoring compounds[/green] - "
                f"[white]{len(state.unqueried_rows)} unqueried[/white], "
                f"[magenta]{len(state.ps_upgrade_rows)} PS hits[/magenta] eligible for upgrade"
            )

            if n_inference == 0:
                warning_message = (
                    "No inference targets found; all compounds are in a terminal or inactive "
                    "state. Writing state CSV with blank score columns."
                )
                progress.update(task, total=2, description=scoring_description)
                annotated_df = annotate_campaign_state(
                    state_df, state, np.empty(0, dtype=np.float32), acquisition
                )
                annotated_df.to_csv(plan_path, index=False)
                progress.advance(task)
            else:
                n_labeled = len(state.training_records)
                n_labeled_drc = sum(
                    1
                    for r in state.training_records
                    if r.fidelity == QueryType.DOSE_RESPONSE
                )
                n_labeled_ps = sum(
                    1
                    for r in state.training_records
                    if r.fidelity == QueryType.PRIMARY_SCREEN
                )
                # Upgrades are PS-INTERVAL records that also have a DRC record;
                # training_records_for_refit removes them, so the difference is the count.
                n_upgrades = n_labeled - len(fit_records)
                upgrade_suffix = (
                    f", [magenta]{n_upgrades} upgrades[/magenta]"
                    if n_upgrades > 0
                    else ""
                )
                retraining_description = (
                    f"[yellow]Training model[/yellow] — {n_labeled} records "
                    f"([orange1]{n_labeled_drc} DRC[/orange1], "
                    f"[steel_blue1]{n_labeled_ps} PS[/steel_blue1]"
                    f"{upgrade_suffix})"
                )
                progress.update(task, description=retraining_description)

                # Setting all seeds
                L.seed_everything(cfg.seed, workers=True, verbose=False)

                # Build model
                model = _build_plan_model(cfg)

                # Train model
                model.refit(
                    records=fit_records,
                    trainer_kwargs=cfg.trainer.to_dict(),
                    datamodule_kwargs=cfg.trainer.to_datamodule_kwargs(),
                    reset_weights=cfg.model.reset_weights_on_refit,
                    output_dir=out_dir,
                )
                progress.advance(task)

                progress.update(task, description=scoring_description)

                # Collect SMILES for inference: unqueried compounds and PS hits eligible for upgrade
                inference_smiles = [smi for _, smi in state.unqueried_rows] + [
                    smi for _, smi in state.ps_upgrade_rows
                ]

                # Make predictions
                predictions = model.predict_smiles(inference_smiles)

                try:
                    annotated_df = annotate_campaign_state(
                        state_df, state, predictions, acquisition
                    )
                except ValueError as exc:
                    raise click.ClickException(str(exc)) from exc

                # Sort by score
                annotated_df = annotated_df.sort_values(
                    by="overall_score", ascending=False
                ).reset_index(drop=True)

                # Save to CSV
                annotated_df.to_csv(plan_path, index=False)

                # Advance progress bar (to completion)
                progress.advance(task)

    if warning_message is not None:
        logger.warning(warning_message)

    n_ps_rec = int((annotated_df["recommendation"] == "ps").sum())
    n_drc_rec = int((annotated_df["recommendation"] == "drc").sum())
    n_drc_upgrades = len(state.ps_upgrade_rows)
    drc_label = (
        f"[orange1]{n_drc_rec} DRC[/orange1] ([magenta]{n_drc_upgrades} upgrades[/magenta])"
        if n_drc_upgrades > 0
        else f"[orange1]{n_drc_rec} DRC[/orange1]"
    )

    _console.print(
        f"[bold green]Plan complete.[/bold green]  "
        f"[bold]Recommendation:[/bold] [steel_blue1]{n_ps_rec} PS[/steel_blue1]  |  "
        f"{drc_label}"
    )

    logger.info(
        "Annotated state CSV written to %s ",
        plan_path,
    )


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
        logger.debug(
            "Banner asset not found at %s; skipping terminal banner.", banner_path
        )


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


def _resolve_plan_output_path(cfg: PipelineConfig, out_dir: Path) -> Path:
    configured = Path(cfg.data.plan.output_csv)
    if configured.is_absolute():
        return configured
    return out_dir / configured


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


def _build_simulation_model(cfg: PipelineConfig, ground_truth: dict[str, float]):
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
