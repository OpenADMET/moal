# moal — Cost-Aware Synthetic Active Learning

A Python pipeline for maximizing the discovery of **active compounds** (pEC50 > 7) from an unrevealed dataset while strictly minimizing labeling cost. The oracle offers two query fidelities:

- **Primary Screen (PS):** Returns an inequality label (`< T` or `>= T`) at a configurable threshold. Cheap.
- **Dose-Response Curve (DRC):** Returns the exact continuous pEC50 value. Expensive.

The underlying predictive model is **ChemProp** initialized with **CheMeleon** pretrained weights, trained with a **Tobit (censored regression) loss** that correctly handles both label types. The framework is **PyTorch Lightning**.

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, `chemprop>=2.0`, `lightning>=2.0`, `rdkit`, `pytorch`.

## Quick Start

```bash
moal --config examples/default_config.yaml
```

Copy and edit the example config to point at your data:

```bash
cp examples/default_config.yaml my_campaign.yaml
# edit my_campaign.yaml: set ground_truth_csv and chempeleon_ckpt_path
moal --config my_campaign.yaml --output-dir results/
```

The full set of options (with defaults and documentation) is in `examples/default_config.yaml`. A minimal config looks like:

```yaml
oracle:
  cost_ps: 1.0
  cost_drc: 10.0
  ps_threshold: 5.0
  activity_threshold: 7.0

model:
  chempeleon_ckpt_path: /path/to/chempeleon.pt
  freeze_epochs: 10
  lr_encoder: 1.0e-5
  lr_head: 1.0e-3

acquisition:
  ps_threshold: 5.0
  target_threshold: 7.0
  tau: 0.5

n_iterations: 20
k_per_iteration: 10
ground_truth_csv: data/compounds.csv   # columns: smiles, pec50
output_dir: results/
```

## Running Tests

```bash
# All tests
pytest

# Single test file
pytest tests/test_loss.py -v

# Single test
pytest tests/test_oracle.py::TestDeduplication::test_requery_raises -v
```

## Architecture

```
moal/
├── cli.py              moal entry point (Click command; installed as `moal`)
├── types.py            QueryType, CensoringType, LabelRecord, LoopResults, IterationResults
├── preprocessing.py    SMILESPreprocessor (RDKit canonicalization + salt stripping + isomeric SMILES)
├── oracle.py           CostAwareOracle (ground truth wrapper, dedup, cost tracking, pEC50 validation)
├── loss.py             CensoredRegressionLoss (Tobit: EXACT / LEFT / INTERVAL; per-fidelity breakdown)
├── dataset.py          MixedFidelityDataset, MixedFidelityDataModule (configurable seed + val_fraction)
├── model.py            ChemPropLightningModule (CheMeleon weights, freeze schedule, per-fidelity logging)
├── acquisition.py      CostAwareGreedyAcquisition
├── loop.py             ActiveLearningLoop (rich progress bar, dashboard wiring)
├── evaluation.py       PipelineEvaluator (scaffold split, APD, recall@budget, EF, evaluate_model)
├── dashboard.py        LiveDashboard (3-panel matplotlib, live-updating or file-only)
├── logging_config.py   suppress_noisy_loggers() — silences Lightning, RDKit, etc.
└── config.py           PipelineConfig hierarchy (YAML-serializable)

examples/
└── default_config.yaml  Fully-documented config with all defaults (start here)
```

## Live Dashboard

Three panels update in real time after every iteration:

| Panel | x-axis | y-axis |
|---|---|---|
| Cumulative Actives | Cost ($) | Confirmed actives found |
| Cost Breakdown | Iteration | Per-iter DRC (orange) + PS (blue) cost, cumulative total line |
| Model Performance | Iteration | Configurable metric (MAE / RMSE / Kendall's τ / Spearman's ρ / R²) |

The model performance panel requires a held-out test set (scaffold-split) — the CLI creates one automatically. In headless/server environments, set `dashboard.show: false` and `dashboard.save_dir: results/` in your YAML config to write PNG snapshots.

## Progress Bar

The campaign emits a rich progress bar with `n_iterations × 3` discrete steps:

```
 ⠹  Iter 3/20  Querying oracle — 7 DRCs, 3 primary screens          ████░░  9/60  0:00:12
 ⠹  Iter 3/20  Retraining model — 28 labeled (14 DRC, 14 PS)        ████░░ 10/60  0:00:45
 ⠹  Iter 3/20  Selecting next 10 compounds…                         ████░░ 11/60  0:00:46
```

## Key Design Notes

**Interval censoring:** Primary screen hits (`>= T`) are modeled as interval-censored `[T, 11.0]`, not right-censored at T. Using right-censoring would invert gradient signals for active compounds.

**Three distinct thresholds:**
- Assay detection limit (left-censoring boundary): pEC50 ≈ 4.0
- Primary screen hit threshold T (`ps_threshold`): pEC50 ≈ 5.0
- Optimization target (`activity_threshold`): pEC50 = 7.0

**Acquisition strategy (greedy):**
- `score(x, DRC) = sigmoid((ŷ - 7.0) / τ) / cost_DRC` — exploits likely actives
- `score(x, PS) = H_binary(sigmoid((ŷ - T) / τ)) / cost_PS` — cheaply resolves threshold ambiguity

**CheMeleon weight loading:** Uses `strict=True`. Atom feature constants are hardcoded in `model.py` and asserted at initialization to catch silent feature mismatches.

**Freeze schedule:** FFN head is trained alone for `freeze_epochs` epochs; the message-passing encoder is then unfrozen with a discriminative (lower) learning rate to avoid catastrophic forgetting.

**Chirality preservation:** `SMILESPreprocessor` passes `isomericSmiles=True` to RDKit `MolToSmiles`, ensuring stereocentres survive canonicalization regardless of RDKit version defaults.

**pEC50 input validation:** `CostAwareOracle` rejects NaN, ±inf, and values outside `[0.0, 14.0]` at ingestion with a warning. This prevents a single bad CSV row from producing NaN gradients that would corrupt an entire training batch.

**Per-fidelity loss monitoring:** `training_step` and `validation_step` log `train_drc_loss`, `train_ps_loss`, `val_drc_loss`, and `val_ps_loss` separately (in addition to the aggregate `train_loss` / `val_loss`), making it possible to detect if DRC regression degrades while PS labels keep the total loss deceptively low.

## Output Files

| File | Description |
|---|---|
| `results/iteration_metrics.csv` | Per-iteration scalar metrics |
| `results/cumulative_actives_curve.csv` | Cumulative actives vs. cost curve |
| `results/dashboard_final.png` | Final dashboard snapshot |
| `results/config_used.yaml` | Exact config used for reproducibility |
