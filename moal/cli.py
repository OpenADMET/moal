"""CLI entry point for running simulations and acquisition planning.

Install the package (``pip install -e .``) to get the ``moal`` command.
The explicit subcommands are::

    moal simulate --config examples/default_config.yaml
    moal plan --config examples/default_config.yaml

"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from pathlib import Path

import click
import lightning as L
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from moal.acquisition import CostAwareGreedyAcquisition
from moal.config import PipelineConfig
from moal.dashboard import LiveDashboard
from moal.evaluation import ModelMetric, PipelineEvaluator, scaffold_split
from moal.logging_config import suppress_noisy_loggers, temporary_log_level
from moal.loop import ActiveLearningLoop
from moal.model import ChemPropLightningModule, NoisyOracleModel
from moal.oracle import CostAwareOracle
from moal.planning import (
    annotate_campaign_state,
    parse_campaign_state,
    parse_pretrain_records,
    training_records_for_refit,
)
from moal.preprocessing import SMILESPreprocessor
from moal.types import LabelRecord, QueryType

logger = logging.getLogger(__name__)
_console = Console(stderr=True)


def _common_cli_options(*, required: bool) -> Callable:
    """Decorator factory that attaches common CLI options to a Click command.

    Parameters
    ----------
    required : bool
        When True, the ``--config`` option is required.

    Returns
    -------
    Callable
        A decorator that wraps a Click command function, attaching
        ``--config``, ``--output-dir``, and ``--verbose`` options.
    """

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
        wrapped = click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")(
            wrapped
        )
        return wrapped

    return decorator


@click.group()
def main() -> None:
    """Moal CLI with simulation and one-shot planning subcommands."""


@main.command()
@_common_cli_options(required=True)
def simulate(config: Path, output_dir: Path | None, verbose: bool) -> None:
    """Run a cost-aware synthetic active learning campaign.

    Parameters
    ----------
    config : Path
        Path to a YAML campaign configuration file.
    output_dir : Path or None
        Override for the output directory specified in the config. If None,
        uses ``cfg.data.output_dir``.
    verbose : bool
        When True, sets the logging level to DEBUG.

    Raises
    ------
    click.ClickException
        If ``data.simulate.input_csv`` is not set in the config or the
        CSV cannot be read.
    """
    _configure_logging(verbose)
    _print_banner()
    _print_welcome()

    cfg = PipelineConfig.from_yaml(config)
    logger.info("Loaded config from %s", config)

    out_dir = _prepare_output_dir(cfg, output_dir)

    if not cfg.data.simulate.input_csv:
        raise click.ClickException("data.simulate.input_csv must be set in the config.")

    simulate_df = _read_csv(
        Path(cfg.data.simulate.input_csv),
        label="data.simulate.input_csv",
    )
    logger.info(
        "Loaded %d rows from %s",
        len(simulate_df),
        cfg.data.simulate.input_csv,
    )

    relation_col = cfg.data.simulate.relation_column
    if relation_col not in simulate_df.columns:
        raise click.ClickException(
            f"simulate input CSV must contain relation column {relation_col!r} for filtering "
            f"to DRC rows, got {sorted(simulate_df.columns)}"
        )

    # Only == (exact DRC) rows define the oracle ground truth pool
    drc_mask = simulate_df[relation_col] == "=="
    n_skipped = int((~drc_mask).sum())
    if n_skipped > 0:
        logger.info(
            "Skipped %d non-DRC row(s) (blank or PS) from simulate input CSV — "
            "only '==' rows are used as oracle ground truth",
            n_skipped,
        )
    ground_truth_df = simulate_df[drc_mask].copy()
    if ground_truth_df.empty:
        raise click.ClickException(
            "simulate input CSV contains no DRC ('==') rows; "
            "at least one row with relation '==' is required to build the oracle pool."
        )

    preprocessor = SMILESPreprocessor()
    try:
        oracle = CostAwareOracle(
            ground_truth_df=ground_truth_df,
            cost_ps=cfg.oracle.cost_ps,
            cost_drc=cfg.oracle.cost_drc,
            ps_threshold=cfg.oracle.ps_threshold,
            upper_bound=cfg.oracle.upper_bound,
            smiles_column=cfg.data.simulate.smiles_column,
            pec50_column=cfg.data.simulate.value_column,
            is_canonical=cfg.data.simulate.is_canonical,
            preprocessor=preprocessor,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

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
        dashboard = LiveDashboard(
            n_iterations=cfg.active_learning_loop.n_iterations,
            n_compounds=len(oracle),
            model_metric=ModelMetric(cfg.dashboard.model_metric),
            port=cfg.dashboard.port,
            export_width=cfg.dashboard.export_width,
            export_height=cfg.dashboard.export_height,
            theme=cfg.dashboard.theme,
        )

    # Load optional pretrain data (same campaign state format as moal plan)
    pretrain_records = _load_pretrain_records(cfg, preprocessor, test_set)

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
        pretrain_records=pretrain_records,
        output_dir=out_dir,
    )

    try:
        results = loop.run(
            n_iterations=cfg.active_learning_loop.n_iterations,
            plate_size=cfg.active_learning_loop.plate_size,
            wells_per_ps=cfg.active_learning_loop.wells_per_ps,
            wells_per_drc=cfg.active_learning_loop.wells_per_drc,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # Write output CSVs immediately so results are available as soon as
    # computation finishes, independent of any dashboard export latency
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

        html_cdn_path = out_dir / "dashboard_animation_cdn.html"
        dashboard.save_html(html_cdn_path, use_cdn=True)

        gif_path = out_dir / "dashboard_animation.gif"
        dashboard.save_gif(gif_path)
        dashboard.close()


@main.command()
@_common_cli_options(required=True)
def plan(config: Path, output_dir: Path | None, verbose: bool) -> None:
    """Train on mixed-fidelity records and score the next acquisition batch.

    Parameters
    ----------
    config : Path
        Path to a YAML campaign configuration file.
    output_dir : Path or None
        Override for the output directory specified in the config. If None,
        uses ``cfg.data.output_dir``.
    verbose : bool
        When True, sets the logging level to DEBUG.

    Raises
    ------
    click.ClickException
        If ``data.plan.input_csv`` is not set, the CSV cannot be read,
        the state CSV contains no labeled records, or campaign state
        parsing fails.
    """
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
                    weight_column=cfg.data.plan.weight_column,
                    log2fc_column=cfg.data.plan.log2fc_column,
                    is_canonical=cfg.data.plan.is_canonical,
                    expected_ps_threshold=cfg.oracle.ps_threshold,
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            progress.advance(task)

            if not state.training_records:
                raise click.ClickException("state CSV did not contain any labeled records.")

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
                    1 for r in state.training_records if r.fidelity == QueryType.DOSE_RESPONSE
                )
                n_labeled_ps = sum(
                    1 for r in state.training_records if r.fidelity == QueryType.PRIMARY_SCREEN
                )
                # Upgrades are PS-INTERVAL records that also have a DRC record;
                # training_records_for_refit removes them, so the difference is the count
                n_upgrades = n_labeled - len(fit_records)
                upgrade_suffix = (
                    f", [magenta]{n_upgrades} upgrades[/magenta]" if n_upgrades > 0 else ""
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


def _load_pretrain_records(
    cfg: PipelineConfig,
    preprocessor: SMILESPreprocessor,
    test_set: tuple[list[str], np.ndarray] | None,
) -> list[LabelRecord]:
    """Load and validate optional pretrain records for ``moal simulate``.

    Returns an empty list when ``data.simulate.pretrain.input_csv`` is not
    set.  When the pretrain CSV is configured, parses it as a mixed-fidelity
    campaign state CSV (same format as ``moal plan``), validates PS threshold
    consistency, and warns about any overlap with the held-out test set.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration.
    preprocessor : SMILESPreprocessor
        Used to canonicalize SMILES from the pretrain CSV.
    test_set : tuple or None
        ``(smiles_list, pec50_array)`` held-out scaffold split, or None if no
        test set is configured.  Used solely for overlap detection.

    Returns
    -------
    list[LabelRecord]
        Parsed pretrain records, or an empty list if no pretrain CSV is set.
    """
    pretrain_cfg = cfg.data.simulate.pretrain
    if not pretrain_cfg.input_csv:
        return []

    pretrain_df = _read_csv(
        Path(pretrain_cfg.input_csv),
        label="data.simulate.pretrain.input_csv",
    )
    logger.info(
        "Loaded %d rows from pretrain CSV %s",
        len(pretrain_df),
        pretrain_cfg.input_csv,
    )

    try:
        records: list[LabelRecord] = parse_pretrain_records(
            pretrain_df,
            cost_ps=cfg.oracle.cost_ps,
            cost_drc=cfg.oracle.cost_drc,
            upper_bound=cfg.oracle.upper_bound,
            preprocessor=preprocessor,
            smiles_column=pretrain_cfg.smiles_column,
            relation_column=pretrain_cfg.relation_column,
            value_column=pretrain_cfg.value_column,
            weight_column=pretrain_cfg.weight_column,
            log2fc_column=pretrain_cfg.log2fc_column,
            is_canonical=pretrain_cfg.is_canonical,
            expected_ps_threshold=cfg.oracle.ps_threshold,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    logger.info("Parsed %d pretrain training records.", len(records))

    # Warn about test-set overlap (potential evaluation metric inflation)
    if test_set is not None and records:
        test_smiles_set = set(test_set[0])
        pretrain_canonical = {r.canonical_smiles for r in records}
        overlap = sorted(pretrain_canonical & test_smiles_set)
        if overlap:
            logger.warning(
                "%d pretrain compound(s) overlap with the held-out test set. "
                "Model performance metrics may be inflated for these compounds. "
                "Overlapping SMILES:\n%s",
                len(overlap),
                "\n".join(f"  {s}" for s in overlap),
            )

    return records


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI process.

    Parameters
    ----------
    verbose : bool
        When True, sets the root logger level to DEBUG; otherwise INFO.

    Notes
    -----
    Uses ``force=True`` so that any previously installed root handlers are
    replaced. Must only be called inside CLI entry points, never at module
    scope, to avoid reconfiguring the root logger during test collection.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _print_banner() -> None:
    """Print the ASCII art banner from the assets directory.

    Notes
    -----
    Silently skips if the asset file ``assets/terminal.txt`` is not found.
    """
    banner_path = Path(__file__).parent.parent / "assets" / "terminal.txt"
    try:
        with open(banner_path) as handle:
            print(handle.read() + "\n")
    except FileNotFoundError:
        logger.debug("Banner asset not found at %s; skipping terminal banner.", banner_path)


def _print_welcome() -> None:
    """Print the versioned welcome message to stdout."""
    version = importlib.metadata.version("moal")
    print(f"Welcome to moal-v{version}: multi-objective active learning for drug discovery!\n")


def _prepare_output_dir(cfg: PipelineConfig, output_dir: Path | None) -> Path:
    """Resolve and create the output directory, then snapshot the config.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration. Used as the fallback output directory
        (``cfg.data.output_dir``) and serialized to ``config_used.yaml``.
    output_dir : Path or None
        Explicit output directory override. When None, falls back to
        ``cfg.data.output_dir``.

    Returns
    -------
    Path
        Resolved and created output directory path.
    """
    out_dir = Path(output_dir or cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config_used.yaml")
    return out_dir


def _resolve_plan_output_path(cfg: PipelineConfig, out_dir: Path) -> Path:
    """Resolve the output CSV path for the ``plan`` subcommand.

    Absolute paths are returned unchanged. Relative paths are resolved
    relative to ``out_dir``.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration supplying ``data.plan.output_csv``.
    out_dir : Path
        Base output directory used to resolve relative paths.

    Returns
    -------
    Path
        Resolved output CSV path.
    """
    configured = Path(cfg.data.plan.output_csv)
    if configured.is_absolute():
        return configured
    return out_dir / configured


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    """Read a CSV file, raising a user-friendly ``ClickException`` on failure.

    Parameters
    ----------
    path : Path
        Filesystem path of the CSV file to read.
    label : str
        Human-readable config key shown in error messages (e.g.,
        ``"data.simulate.input_csv"``).

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame.

    Raises
    ------
    click.ClickException
        If the file is not found, cannot be opened, or cannot be parsed.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise click.ClickException(f"{label} not found: {path}") from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise click.ClickException(f"Failed to read {label} {path}: {exc}") from exc


def _build_acquisition(cfg: PipelineConfig) -> CostAwareGreedyAcquisition:
    """Instantiate the acquisition function from pipeline config.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration.

    Returns
    -------
    CostAwareGreedyAcquisition
        Configured acquisition function instance.
    """
    return CostAwareGreedyAcquisition(
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        ps_threshold=cfg.acquisition.ps_threshold,
        target_threshold=cfg.acquisition.target_threshold,
        tau=cfg.acquisition.tau,
    )


def _build_simulation_model(
    cfg: PipelineConfig, ground_truth: dict[str, float]
) -> ChemPropLightningModule | NoisyOracleModel:
    """Instantiate the predictive model for a simulation campaign.

    When ``cfg.model.fast`` is True, returns a ``NoisyOracleModel`` that
    adds noise to oracle ground-truth values. Otherwise, returns a
    ``ChemPropLightningModule``.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration.
    ground_truth : dict[str, float]
        Mapping from canonical SMILES to true pEC50 values. Only used when
        ``cfg.model.fast`` is True.

    Returns
    -------
    ChemPropLightningModule or NoisyOracleModel
        Configured model ready for ``refit()`` and ``predict_smiles()``.
    """
    if cfg.model.fast:
        logger.info(
            "Fast mode enabled — using NoisyOracleModel with error ramp"
            " %.3f → %.3f over %d iterations",
            cfg.model.initial_error,
            cfg.model.final_error,
            cfg.active_learning_loop.n_iterations,
        )
        return NoisyOracleModel(ground_truth=ground_truth, seed=cfg.seed)

    return ChemPropLightningModule(
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        ffn_num_layers=cfg.model.ffn_num_layers,
        message_hidden_dim=cfg.model.message_hidden_dim,
        depth=cfg.model.depth,
        freeze_epochs=cfg.model.freeze_epochs,
        mpnn_lr=cfg.model.mpnn_lr,
        ffn_lr=cfg.model.ffn_lr,
        mpnn_weight_decay=cfg.model.mpnn_weight_decay,
        ffn_weight_decay=cfg.model.ffn_weight_decay,
        sigma=cfg.model.sigma,
        w_drc=cfg.model.w_drc,
        w_ps=cfg.model.w_ps,
        learnable_sigma=cfg.model.learnable_sigma,
        from_foundation=cfg.model.from_foundation,
    )


def _build_plan_model(cfg: PipelineConfig) -> ChemPropLightningModule:
    """Instantiate a ``ChemPropLightningModule`` for offline planning.

    Parameters
    ----------
    cfg : PipelineConfig
        Active campaign configuration.

    Returns
    -------
    ChemPropLightningModule
        Configured model ready for ``refit()`` and ``predict_smiles()``.

    Raises
    ------
    click.ClickException
        If ``cfg.model.fast`` is True, since offline planning has no oracle
        ground truth for candidate compounds.
    """
    if cfg.model.fast:
        raise click.ClickException(
            "moal plan does not support model.fast=true because offline planning "
            "has no oracle ground truth for candidate compounds."
        )

    return ChemPropLightningModule(
        ffn_hidden_dim=cfg.model.ffn_hidden_dim,
        ffn_num_layers=cfg.model.ffn_num_layers,
        message_hidden_dim=cfg.model.message_hidden_dim,
        depth=cfg.model.depth,
        freeze_epochs=cfg.model.freeze_epochs,
        mpnn_lr=cfg.model.mpnn_lr,
        ffn_lr=cfg.model.ffn_lr,
        mpnn_weight_decay=cfg.model.mpnn_weight_decay,
        ffn_weight_decay=cfg.model.ffn_weight_decay,
        sigma=cfg.model.sigma,
        w_drc=cfg.model.w_drc,
        w_ps=cfg.model.w_ps,
        learnable_sigma=cfg.model.learnable_sigma,
        from_foundation=cfg.model.from_foundation,
    )


if __name__ == "__main__":
    main()
