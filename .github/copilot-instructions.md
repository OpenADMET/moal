# moal — Copilot Instructions

## Build, test, and lint commands

```bash
# Install (editable)
pip install -e .

# Run full test suite
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_loss.py

# Run a single test class or test
python -m pytest tests/test_loss.py::TestLossBreakdown
python -m pytest tests/test_loss.py::TestLossBreakdown::test_forward_consistent_with_breakdown

# CLI entry points
moal simulate --config examples/default_config.yaml
moal simulate --config examples/default_config.yaml --output-dir results/
moal simulate --config examples/default_config.yaml --verbose

moal plan --config examples/default_config.yaml
moal plan --config examples/default_config.yaml --output-dir results/
```

No linter is configured. `pytest` is the only test runner; config lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## Architecture

The pipeline has two modes: **`moal simulate`** runs a synthetic active learning campaign over a fixed compound pool (ground-truth CSV with `smiles` and `pec50`); **`moal plan`** trains on a real mixed-fidelity campaign state CSV and scores the next acquisition batch for wet-lab prioritization.

### Data flow — `moal simulate`

```
PipelineConfig (YAML)
    │
    ▼
CostAwareOracle ──────── ground_truth_df (CSV, pec50 validated [0, 14])
    │  query_batch()
    ▼
LabelRecord (frozen dataclass)
    │   fidelity: PS → LEFT (inactive) or INTERVAL (active) label
    │   fidelity: DRC → EXACT label
    │   PS→DRC upgrade: oracle.get_ps_labeled_smiles() feeds upgrade candidates
    ▼
oracle.training_records ── excludes INTERVAL PS for compounds with DRC
    │   (prevents double-weighting of upgraded compounds)
    ▼
MixedFidelityDataModule ── train/val split (scaffold-unaware random split)
    │
    ▼
ChemPropLightningModule.refit()   ← or NoisyOracleModel (fast=True)
    │  CensoredRegressionLoss (Tobit: EXACT / LEFT / INTERVAL branches)
    │  Freeze/unfreeze schedule: FFN head only for freeze_epochs, then encoder
    ▼
model.predict_smiles(unlabeled + ps_labeled)  →  pEC50 point estimates
    │
    ▼
CostAwareGreedyAcquisition.select(unlabeled, predictions, k,
                                  ps_labeled_smiles, ps_labeled_predictions)
    │  DRC score = p_active(ŷ) / cost_DRC                [exploitation]
    │  PS  score = H_binary(p_cross(ŷ, T)) / cost_PS     [exploration]
    │  PS-labeled compounds: DRC-upgrade candidates only (no PS re-query)
    │  Greedy top-k, one query per compound
    ▼
ActiveLearningLoop.run()  →  LoopResults
    │  3 Rich progress steps per iteration: query → refit → select
    │  NoisyOracleModel: noise_scale ramps linearly initial_error → final_error
    ▼
LiveDashboard (Plotly + Dash, 4 panels: actives curve, cost stacks, model metric, compound status)
    │  Live browser at 127.0.0.1:8050 (1s refresh); PNG frames via matplotlib Agg
    │  save_html() → interactive HTML with iteration slider
    │  save_gif()  → animated GIF from matplotlib PNG frames
    ▼
output_dir/
    ├── config_used.yaml
    ├── dashboard_animation.html          # interactive Plotly with slider
    ├── dashboard_animation.gif           # animated GIF
    ├── iteration_metrics.csv
    ├── cumulative_actives_curve.csv
    └── lightning_logs/
```

### Data flow — `moal plan`

```
PipelineConfig (YAML)
    │
    ▼
parse_campaign_state(df)  ──── unified campaign state CSV
    │  relation="" / value=""  → unqueried (eligible for PS or DRC)
    │  relation="<"            → PS miss (LEFT label, train only)
    │  relation=">="           → PS hit  (INTERVAL label, train + upgrade candidate)
    │  relation="=="           → DRC exact label (EXACT, train only)
    ▼
CampaignState
    │  .training_records  → MixedFidelityDataModule → ChemPropLightningModule.refit()
    │  .unqueried_rows + .ps_upgrade_rows → model.predict_smiles() → predictions
    ▼
CostAwareGreedyAcquisition.score_summary()
    ▼
annotate_campaign_state(df, state, predictions, acquisition)
    │  appends: ps_score, drc_score, overall_score, recommendation ("ps"/"drc")
    │  sorted by overall_score descending
    ▼
output_dir/campaign_state.csv   (configurable via data.plan.output_csv)
```

### Three distinct threshold parameters

These are separate and must not be conflated:

| Parameter | Location | Role |
|---|---|---|
| `oracle.ps_threshold` | `OracleConfig` | Where the PS assay draws the inequality (e.g., 5.0 → "< 5" or ">= 5") |
| `acquisition.ps_threshold` | `AcquisitionConfig` | Mirror of the above, drives PS entropy score. Must match oracle value. |
| `acquisition.target_threshold` / `oracle.activity_threshold` | Both configs | What "active" means for DRC scoring and evaluation (e.g., 7.0 = 100 nM) |

### Key correctness invariants

- **PS `>= T` labels are INTERVAL-censored `[T, upper_bound]`**, not right-censored at T. Right-censoring inverts gradients for active compounds. See `CensoringType.INTERVAL` in `types.py` and the `INTERVAL` branch in `loss.py`.
- **pEC50 values outside `[0.0, 14.0]`** (and NaN/inf) are excluded by `oracle._build_ground_truth()` before they reach the loss function. Silently passing NaN would corrupt entire training batches.
- **`Chem.MolToSmiles(mol, isomericSmiles=True)`** is explicit in `preprocessing.py`. Do not remove this — RDKit's default has changed historically and chirality must be preserved.
- **`TrainerConfig.to_dict()`** returns only `L.Trainer`-compatible keys (`max_epochs`, `accelerator`, `enable_progress_bar`, `enable_model_summary`, `log_every_n_steps`). `val_fraction`, `split_seed`, and `num_workers` are consumed by `MixedFidelityDataModule` via `to_datamodule_kwargs()` which returns `{val_fraction, seed, num_workers}`. Passing them to `L.Trainer` raises `TypeError`.
- **`logging.basicConfig`** must only be called inside `cli.main()`, never at module scope. Module-level calls fire on import and reconfigure the root logger during test collection.
- **`oracle.training_records`** (not `oracle.labeled_records`) must be passed to `model.refit()`. It excludes PS INTERVAL records for compounds that have a DRC record, preventing double-weighting of upgraded compounds.

---

## Conventions

### Config hierarchy

All campaign parameters live in `moal/config.py` as frozen dataclasses. The YAML config maps directly to the nested hierarchy below. `PipelineConfig.from_yaml()` is the single entry point for deserialization; `PipelineConfig.to_yaml()` serializes back. When adding a new parameter, add it to the appropriate `@dataclass(frozen=True)` class and to `from_yaml()`.

**YAML sections and their dataclasses:**

| YAML key | Dataclass | Notable fields |
|---|---|---|
| `oracle:` | `OracleConfig` | `cost_ps`, `cost_drc`, `ps_threshold`, `upper_bound`, `activity_threshold` |
| `model:` | `ModelConfig` | `hidden_size`, `depth`, `ffn_hidden_size`, `ffn_num_layers`, `freeze_epochs`, `lr_encoder`, `lr_head`, `sigma`, `w_drc`, `w_ps`, `learnable_sigma`, `reset_weights_on_refit`, **`fast`**, **`initial_error`**, **`final_error`** |
| `acquisition:` | `AcquisitionConfig` | `ps_threshold`, `target_threshold`, **`tau`** |
| `trainer:` | `TrainerConfig` | `max_epochs`, `accelerator`, `enable_progress_bar`, `enable_model_summary`, `val_fraction`, `split_seed`, `num_workers`, `log_every_n_steps` |
| `dashboard:` | `DashboardConfig` | `enabled`, `model_metric`, `port`, `export_width`, `export_height`, `theme` |
| `data:` | `DataConfig` | `output_dir`; nested `simulate:` → `SimulationDataConfig`; nested `plan:` → `PlanDataConfig` |
| `data.simulate:` | `SimulationDataConfig` | `input_csv`, `smiles_column`, `pec50_column`, `is_canonical`, `test_set_size` |
| `data.plan:` | `PlanDataConfig` | `input_csv`, `output_csv`, `smiles_column`, `relation_column`, `value_column`, `is_canonical` |
| `active_learning_loop:` | `ActiveLearningLoopConfig` | `n_iterations`, `k_per_iteration` |
| *(top-level)* | `PipelineConfig` | `seed` |

### Fast mode (NoisyOracleModel)

When `model.fast = true` in the config, `ActiveLearningLoop` uses `NoisyOracleModel` instead of `ChemPropLightningModule`. This surrogate looks up true pEC50 values from the oracle's ground truth and adds uniform noise, skipping all neural network training. The noise scale ramps linearly from `model.initial_error` down to `model.final_error` across iterations. Use fast mode for rapid campaign prototyping and integration tests; never for real experiments.

### Planning module

`moal/planning.py` powers the `moal plan` command. Key components:

- **`CampaignState`** (dataclass) — holds `training_records: list[LabelRecord]`, `unqueried_rows: list[tuple[int, str]]`, and `ps_upgrade_rows: list[tuple[int, str]]` (DRC upgrade candidates).
- **`parse_campaign_state(df, *, cost_ps, cost_drc, upper_bound, preprocessor, ...)`** — converts the unified campaign state CSV into a `CampaignState`. Validates that no compound appears in multiple partitions and that relation/value fields are consistent.
- **`training_records_for_refit(records)`** — mirrors `oracle.training_records` deduplication logic: drops PS INTERVAL rows for compounds that also have a DRC record.
- **`annotate_campaign_state(df, state, predictions, acquisition)`** — appends `ps_score`, `drc_score`, `overall_score`, and `recommendation` columns; sorts by `overall_score` descending. `predictions` must align with `state.unqueried_rows + state.ps_upgrade_rows`.

### Dashboard (Plotly + Dash)

`LiveDashboard` in `moal/dashboard.py` replaced the old matplotlib-only dashboard. Key notes:

- **Live browser view** served at `127.0.0.1:<port>` (default 8050) with 1-second Dash callback refresh during `moal simulate`.
- **4-panel 2×2 subplot layout:** (1) cumulative actives curve, (2) per-iteration cost breakdown with secondary-y cumulative cost line, (3) model performance metric over iterations, (4) compound status bars (PS-only, DRC, upgrades, unqueried).
- **`save_html(path)`** exports a standalone Plotly HTML file with an iteration slider and play/pause controls.
- **`save_gif(path)`** assembles matplotlib PNG frames (rendered via the Agg backend) into an animated GIF — no external renderer required.
- **`DashboardConfig`** fields: `enabled`, `model_metric`, `port`, `export_width`, `export_height`, `theme` (Plotly template name, e.g. `"plotly_dark"`). The old `figsize` and `show` fields no longer exist.

### Label records

`LabelRecord` (frozen dataclass in `types.py`) is the canonical unit of labeled data throughout the pipeline. It carries the censoring type, fidelity, cost, and iteration alongside the SMILES and bounds. All training, evaluation, and dashboard code operates on `list[LabelRecord]` — do not pass raw floats between components.

`IterationResults` and `LoopResults` are also defined in `types.py`. `IterationResults` carries an optional `model_metric_value` (None when no test set is configured).

### Logging

All modules use `logger = logging.getLogger(__name__)`. The `suppress_noisy_loggers()` function in `logging_config.py` must be called once at the start of `loop.run()` to silence PyTorch Lightning, RDKit, and matplotlib. Do not suppress moal's own loggers.

### Loss breakdown

`CensoredRegressionLoss.forward_with_breakdown()` returns a `LossBreakdown` NamedTuple with `total`, `drc_loss`, and `ps_loss`. The per-fidelity fields are `nan` (not zero) when no samples of that fidelity are in the batch — Lightning skips `nan` log values, which is intentional. `forward()` delegates to `forward_with_breakdown()`. The INTERVAL branch uses direct CDF subtraction clamped to `1e-12` to avoid `log(0)`.

### Freeze/unfreeze schedule

`ChemPropLightningModule` freezes the CheMeleon encoder for the first `freeze_epochs` training epochs, then unfreezes and adds a second optimizer for the encoder at `lr_encoder`. The epoch counter resets on every `trainer.fit()` call (every AL iteration). This is intentional — early iterations have tiny labeled pools where encoder fine-tuning would overfit.

### Scaffold split

`scaffold_split()` in `evaluation.py` uses Bemis-Murcko scaffolds, assigns groups largest-first to the test set, and may slightly exceed the requested `test_size` if the first scaffold group is large. This is acceptable — scaffold groups cannot be split. This split is used only for the held-out model evaluation test set (built in `cli.main()` when `cfg.data.test_set_size > 0`); the train/val split inside `MixedFidelityDataModule` is a plain random split.

### Model evaluation metrics

`PipelineEvaluator` in `evaluation.py` supports five model metrics via the `ModelMetric` enum: `MAE`, `RMSE`, `KENDALL_TAU`, `SPEARMAN_R`, `R2`. The metric is configured via `dashboard.model_metric` in the YAML and displayed in the third dashboard panel. `evaluate_model()` accepts an optional `noise_scale` argument for use with `NoisyOracleModel`.

### CLI filesystem isolation in tests

`CliRunner.invoke()` does **not** sandbox file I/O by default. Any test that triggers CLI output-directory creation must pass `--output-dir str(tmp_path / "out")` (or use `runner.isolated_filesystem()`) to avoid leaking `results/` into the pytest CWD.

### CheMeleon feature dimensions

`_CHEMPELEON_ATOM_FDIM = 72` and `_CHEMPELEON_BOND_FDIM = 14` in `model.py` are hardcoded to match the CheMeleon pretraining feature spec. These are verified at model initialization. Do not change them without updating the checkpoint.
