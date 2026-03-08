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

# CLI entry point
moal --config examples/default_config.yaml
moal --config examples/default_config.yaml --output-dir results/
moal --config examples/default_config.yaml --verbose
```

No linter is configured. `pytest` is the only test runner; config lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

---

## Architecture

The pipeline is a **cost-aware active learning loop** over a fixed compound pool. The user provides a CSV with `smiles` and `pec50` columns; the oracle simulates a wet-lab screen by dispensing labels at cost.

### Data flow

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
LiveDashboard (matplotlib, 3 panels: actives curve, cost stacks, model metric)
    │  PNG snapshot after every update; save_gif() exports animated GIF
    ▼
output_dir/
    ├── config_used.yaml
    ├── dashboard_final.png
    ├── dashboard_animation.gif
    ├── iteration_metrics.csv
    └── cumulative_actives_curve.csv
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
- **`TrainerConfig.to_dict()`** returns only `L.Trainer`-compatible keys (`max_epochs`, `accelerator`, `enable_progress_bar`, `enable_model_summary`). `val_fraction` and `split_seed` are consumed by `MixedFidelityDataModule` via `to_datamodule_kwargs()` which returns `{val_fraction, seed}`. Passing them to `L.Trainer` raises `TypeError`.
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
| `model:` | `ModelConfig` | `hidden_size`, `depth`, `ffn_hidden_size`, `ffn_num_layers`, `freeze_epochs`, `lr_encoder`, `lr_head`, `sigma`, `w_drc`, `w_ps`, `learnable_sigma`, **`fast`**, **`initial_error`**, **`final_error`** |
| `acquisition:` | `AcquisitionConfig` | `ps_threshold`, `target_threshold`, **`tau`** |
| `trainer:` | `TrainerConfig` | `max_epochs`, `accelerator`, `enable_progress_bar`, `enable_model_summary`, `val_fraction`, `split_seed` |
| `dashboard:` | `DashboardConfig` | `enabled`, `model_metric`, `figsize`, `show` |
| `data:` | `DataConfig` | `ground_truth_csv`, `smiles_column`, `pec50_column`, `is_canonical`, `output_dir` |
| `active_learning_loop:` | `ActiveLearningLoopConfig` | `n_iterations`, `k_per_iteration` |
| *(top-level)* | `PipelineConfig` | `test_set_size`, `seed` |

### Fast mode (NoisyOracleModel)

When `model.fast = true` in the config, `ActiveLearningLoop` uses `NoisyOracleModel` instead of `ChemPropLightningModule`. This surrogate looks up true pEC50 values from the oracle's ground truth and adds uniform noise, skipping all neural network training. The noise scale ramps linearly from `model.initial_error` down to `model.final_error` across iterations. Use fast mode for rapid campaign prototyping and integration tests; never for real experiments.

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

`scaffold_split()` in `evaluation.py` uses Bemis-Murcko scaffolds, assigns groups largest-first to the test set, and may slightly exceed the requested `test_size` if the first scaffold group is large. This is acceptable — scaffold groups cannot be split. This split is used only for the held-out model evaluation test set (built in `cli.main()` when `test_set_size > 0`); the train/val split inside `MixedFidelityDataModule` is a plain random split.

### Model evaluation metrics

`PipelineEvaluator` in `evaluation.py` supports five model metrics via the `ModelMetric` enum: `MAE`, `RMSE`, `KENDALL_TAU`, `SPEARMAN_R`, `R2`. The metric is configured via `dashboard.model_metric` in the YAML and displayed in the third dashboard panel. `evaluate_model()` accepts an optional `noise_scale` argument for use with `NoisyOracleModel`.

### CLI filesystem isolation in tests

`CliRunner.invoke()` does **not** sandbox file I/O by default. Any test that triggers CLI output-directory creation must pass `--output-dir str(tmp_path / "out")` (or use `runner.isolated_filesystem()`) to avoid leaking `results/` into the pytest CWD.

### CheMeleon feature dimensions

`_CHEMPELEON_ATOM_FDIM = 72` and `_CHEMPELEON_BOND_FDIM = 14` in `model.py` are hardcoded to match the CheMeleon pretraining feature spec. These are verified at model initialization. Do not change them without updating the checkpoint.
