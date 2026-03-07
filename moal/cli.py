"""CLI entry point for running a cost-aware active learning campaign.

Install the package (``pip install -e .``) to get the ``moal`` command::

    moal --config examples/default_config.yaml --output-dir results/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to a YAML campaign configuration file.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    type=click.Path(),
    help="Override output directory from config.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def main(config: str, output_dir: str | None, verbose: bool) -> None:
    """Run a cost-aware synthetic active learning campaign."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from moal.acquisition import CostAwareGreedyAcquisition
    from moal.config import PipelineConfig
    from moal.evaluation import ModelMetric, PipelineEvaluator, scaffold_split
    from moal.loop import ActiveLearningLoop
    from moal.model import ChemPropLightningModule, NoisyOracleModel
    from moal.oracle import CostAwareOracle
    from moal.preprocessing import SMILESPreprocessor

    cfg = PipelineConfig.from_yaml(config)
    logger.info("Loaded config from %s", config)

    out_dir = Path(output_dir or cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config_used.yaml")

    if not cfg.data.ground_truth_csv:
        logger.error("data.ground_truth_csv must be set in the config.")
        sys.exit(1)

    try:
        ground_truth_df = pd.read_csv(cfg.data.ground_truth_csv)
    except FileNotFoundError:
        logger.error("data.ground_truth_csv not found: %s", cfg.data.ground_truth_csv)
        sys.exit(1)
    except (OSError, pd.errors.ParserError) as exc:
        logger.error("Failed to read data.ground_truth_csv %s: %s", cfg.data.ground_truth_csv, exc)
        sys.exit(1)
    logger.info("Loaded %d compounds from %s", len(ground_truth_df), cfg.data.ground_truth_csv)

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

    if cfg.model.fast:
        model = NoisyOracleModel(
            ground_truth=oracle._ground_truth,
            noise_scale=cfg.model.noise_scale,
            seed=cfg.seed,
        )
        logger.info(
            "Fast mode enabled — using NoisyOracleModel with noise_scale=%.3f",
            cfg.model.noise_scale,
        )
    else:
        model = ChemPropLightningModule(
            chempeleon_ckpt_path=cfg.model.chempeleon_ckpt_path,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth,
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

    acquisition = CostAwareGreedyAcquisition(
        cost_ps=cfg.oracle.cost_ps,
        cost_drc=cfg.oracle.cost_drc,
        ps_threshold=cfg.acquisition.ps_threshold,
        target_threshold=cfg.acquisition.target_threshold,
        tau=cfg.acquisition.tau,
    )

    evaluator = PipelineEvaluator(
        activity_threshold=cfg.oracle.activity_threshold,
        upper_bound=cfg.oracle.upper_bound,
    )

    # Build a scaffold-stratified held-out test set for model performance tracking.
    import numpy as np

    all_smiles = oracle.get_unlabeled_smiles()
    test_set = None
    if all_smiles and cfg.test_set_size > 0:
        _, test_idx = scaffold_split(all_smiles, test_size=cfg.test_set_size, seed=cfg.seed)
        if test_idx:
            test_smiles = [all_smiles[i] for i in test_idx]
            test_pec50 = [oracle._ground_truth[s] for s in test_smiles]
            test_set = (test_smiles, np.array(test_pec50, dtype=np.float32))

    dashboard = None
    if cfg.dashboard.enabled:
        from moal.dashboard import LiveDashboard

        dashboard = LiveDashboard(
            n_iterations=cfg.active_learning_loop.n_iterations,
            model_metric=ModelMetric(cfg.dashboard.model_metric),
            save_dir=cfg.dashboard.save_dir or None,
            figsize=cfg.dashboard.figsize,
            show=cfg.dashboard.show,
        )

    model_metric = ModelMetric(cfg.dashboard.model_metric)

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
        model_metric=model_metric,
    )

    results = loop.run(
        n_iterations=cfg.active_learning_loop.n_iterations,
        k_per_iteration=cfg.active_learning_loop.k_per_iteration,
    )

    if dashboard is not None:
        final_path = out_dir / "dashboard_final.png"
        dashboard.save(final_path)
        logger.info("Final dashboard saved to %s", final_path)

        # Assemble per-iteration snapshots into a looping GIF when save_dir is set.
        if dashboard.save_dir:
            gif_path = dashboard.save_dir / "dashboard_animation.gif"
            dashboard.save_gif(gif_path)

        dashboard.close()

    # Write per-iteration metrics to CSV.
    metrics_df = pd.DataFrame([r.metrics for r in results.iterations])
    metrics_path = out_dir / "iteration_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics written to %s", metrics_path)

    # Write cumulative actives curve.
    curve_df = evaluator.cumulative_actives_curve(oracle.labeled_records)
    curve_path = out_dir / "cumulative_actives_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info("Cumulative actives curve written to %s", curve_path)

    logger.info(
        "Campaign complete. Total cost: $%.2f | Labeled: %d | Confirmed actives: %d"
        " | PS→DRC upgrades: %d",
        results.total_cost,
        results.total_labeled,
        int(results.final_metrics.get("n_confirmed_actives", 0)),
        int(results.final_metrics.get("n_ps_to_drc_upgrades", 0)),
    )


if __name__ == "__main__":
    main()
