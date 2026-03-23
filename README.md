# `moal`: multi-objective active learning

A Python pipeline for maximizing the discovery of **active compounds** (pEC50 > 7) from an unrevealed dataset while strictly minimizing labeling cost. The oracle offers two query fidelities:

- **Primary Screen (PS):** Returns an inequality label (`< T` or `>= T`) at a configurable threshold. Cheap. A hit (`>= T`) is an INTERVAL-censored label — eligible for a DRC upgrade in a later iteration.
- **Dose-Response Curve (DRC):** Returns the exact continuous pEC50 value. Expensive. Can be run as a first-pass query *or* as a follow-up upgrade on a PS hit.

The underlying predictive model is **ChemProp** initialized with **CheMeleon** pretrained weights, trained with a **Tobit (censored regression) loss** that correctly handles both label types.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
moal simulate --config examples/default_config.yaml
```

Copy and edit the example config to point at your data:

```bash
cp examples/default_config.yaml my_campaign.yaml

# Edit my_campaign.yaml: set data.simulate.input_csv
moal simulate --config my_campaign.yaml --output-dir results/
```

CheMeleon pretrained weights are downloaded automatically from Zenodo on first
run and cached at `~/.chemprop/chemeleon_mp.pt` for subsequent use.

## Commands

### `simulate`

Runs the full cost-aware active learning loop:

```bash
moal simulate --config examples/default_config.yaml
moal simulate --config examples/default_config.yaml --output-dir results/
moal simulate --config examples/default_config.yaml --verbose
```

### `plan`

Trains the model on labeled records from a unified campaign state CSV and scores
all inference targets to produce a ranked acquisition recommendation:

```bash
moal plan \
  --config examples/default_config.yaml
```

`plan` does **not** run the active learning loop or dashboard; it is a one-shot
train-and-score workflow designed for iterative real-world campaigns.

#### Unified state CSV format

A single CSV drives both the training set and the inference target list. The
expected base columns are `smiles`, `relation`, and `value` (all configurable
via `data.plan.*`):

| `relation` | `value` | Row state | Action |
|---|---|---|---|
| empty | empty | Unqueried | Model scores for PS **or** DRC; `recommendation` = `"ps"` or `"drc"` |
| `<` | numeric | PS miss (inactive) | Training only; score columns are NaN |
| `>=` | numeric | PS hit | Training **and** DRC-upgrade inference target; `recommendation` = `"drc"` |
| `==` | numeric | DRC result (terminal) | Training only; score columns are NaN |

#### Output format

`moal plan` **annotates and re-exports the same CSV**, appending four columns:

| Column | Description |
|---|---|
| `ps_score` | PS exploration score (`H_binary(p_cross) / cost_PS`); NaN for non-unqueried rows |
| `drc_score` | DRC exploitation score (`p_active / cost_DRC`); NaN for training-only rows |
| `overall_score` | `max(ps_score, drc_score)` for unqueried rows; `drc_score` for PS upgrades; NaN otherwise |
| `recommendation` | `"ps"` or `"drc"` for inference targets; NaN for training-only rows |

The annotated file preserves all original columns so it can be **re-ingested
unmodified** in the next iteration after new experimental results are filled in.

`plan` does not support `model.fast = true`, because offline planning has no
oracle ground truth for unseen compounds.

## Live Dashboard

Four panels update in real time after every iteration:

| Panel | x-axis | y-axis |
|---|---|---|
| Cumulative Actives | Cost ($) | Confirmed actives found |
| Cost Breakdown | Iteration | Per-iter DRC (orange) + PS (blue) cost, cumulative total line |
| Model Performance | Iteration | Configurable metric (MAE / RMSE / Kendall's τ / Spearman's ρ / R²) |
| Compound Status | Unqueried, PS, DRC | Number of compounds |

The model performance panel requires a held-out test set (scaffold-split) — the CLI creates one automatically. In headless/server environments, set `dashboard.show: false` in your YAML config.

## Progress Bar

The campaign emits a rich progress bar with `n_iterations × 3` discrete steps:

```
 ⠹  Iter 3/20  Querying oracle — 5 DRC (2 upgrades), 3 PS             ████░░  15%  0:00:12
 ⠹  Iter 3/20  Retraining model — 28 records (14 DRC, 12 PS, 2 upgrades)  ████░░  17%  0:00:45
 ⠹  Iter 3/20  Selecting next 10 — 18 unqueried, 3 PS hits eligible for upgrade  ████░░  18%  0:00:46
```

## Key Design Notes

**Interval censoring:** Primary screen hits (`>= T`) are modeled as interval-censored `[T, 11.0]`, not right-censored at T. Using right-censoring would invert gradient signals for active compounds.

**Three distinct thresholds:**
- Assay detection limit (left-censoring boundary): pEC50 ≈ 4.0
- Primary screen hit threshold T (`ps_threshold`): pEC50 ≈ 5.0
- Optimization target (`activity_threshold`): pEC50 = 7.0

**Acquisition strategy (greedy):** Each iteration scores two pools — unqueried compounds (eligible for PS or DRC) and PS-INTERVAL-labeled hits (eligible for DRC upgrade only) — on the same cost-normalised scale:
- `score(x, DRC) = sigmoid((ŷ - 7.0) / τ) / cost_DRC` — exploits likely actives; applies equally to first-pass DRC and upgrade-DRC candidates
- `score(x, PS) = H_binary(sigmoid((ŷ - T) / τ)) / cost_PS` — cheaply resolves threshold ambiguity; only generated for unqueried compounds

**Per-fidelity loss monitoring:** `training_step` and `validation_step` log `train_drc_loss`, `train_ps_loss`, `val_drc_loss`, and `val_ps_loss` separately (in addition to the aggregate `train_loss` / `val_loss`), making it possible to detect if DRC regression degrades while PS labels keep the total loss deceptively low.

## Output Files

### `simulate`

| File | Description |
|---|---|
| `results/iteration_metrics.csv` | Per-iteration scalar metrics |
| `results/cumulative_actives_curve.csv` | Cumulative actives vs. cost curve |
| `results/dashboard_animation.html` | Campaign dashboard animation as interactive HTML |
| `results/dashboard_animation.gif` | Campaign dashboard animation as portable GIF |
| `results/config_used.yaml` | Exact config used for reproducibility |

### `plan`

| File | Description |
|---|---|
| `results/campaign_state.csv` | Original state CSV annotated with `ps_score`, `drc_score`, `overall_score`, `recommendation` |
| `results/config_used.yaml` | Exact config used for reproducibility |

## Architecture

```
moal/
├── cli.py              moal CLI group (`simulate`, `plan`; installed as `moal`)
├── types.py            QueryType, CensoringType, LabelRecord, LoopResults, IterationResults
├── preprocessing.py    SMILESPreprocessor (RDKit canonicalization + salt stripping + isomeric SMILES)
├── oracle.py           CostAwareOracle (ground truth wrapper, dedup, cost tracking, pEC50 validation)
├── loss.py             CensoredRegressionLoss (Tobit: EXACT / LEFT / INTERVAL; per-fidelity breakdown)
├── dataset.py          MixedFidelityDataset, MixedFidelityDataModule (configurable seed + val_fraction)
├── model.py            ChemPropLightningModule (CheMeleon weights, freeze schedule, per-fidelity logging)
├── acquisition.py      CostAwareGreedyAcquisition
├── planning.py         Mixed-fidelity CSV parsing + one-shot acquisition ranking helpers
├── loop.py             ActiveLearningLoop (rich progress bar, dashboard wiring)
├── evaluation.py       PipelineEvaluator (scaffold split, APD, recall@budget, EF, evaluate_model)
├── dashboard.py        LiveDashboard (3-panel matplotlib, live-updating or file-only)
├── logging_config.py   suppress_noisy_loggers() — silences Lightning, RDKit, etc.
└── config.py           PipelineConfig hierarchy (YAML-serializable)

examples/
└── default_config.yaml  Fully-documented config with all defaults (start here)
```

## Running Tests

```bash
# All tests
pytest

# Single test file
pytest tests/test_loss.py -v

# Single test
pytest tests/test_oracle.py::TestDeduplication::test_ps_after_ps_raises -v
```
