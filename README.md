![Banner image](assets/banner.jpg)

# `moal`: multi-objective active learning

A Python pipeline for maximizing the discovery of **active compounds** (pEC50 > 7) from an unrevealed dataset while strictly minimizing labeling cost. The oracle offers two query fidelities:

- **Primary Screen (PS):** Returns an inequality label (`< T` or `>= T`) at a configurable threshold. Cheap. A hit (`>= T`) is an INTERVAL-censored label, eligible for a DRC upgrade in a later iteration.
- **Dose-Response Curve (DRC):** Returns the exact continuous pEC50 value. Expensive. Can be run as a first-pass query *or* as a follow-up upgrade on a PS hit.

The underlying predictive model is **ChemProp** fine-tuned with a **Tobit (censored regression) loss** that correctly handles both label types. By default the ChemProp encoder is initialised with **CheMeleon** pretrained weights (see `model.from_foundation` below).

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

# Edit my_campaign.yaml: set data.simulate.input_csv to a campaign state CSV
# where rows with relation "==" define the oracle ground truth pool
moal simulate --config my_campaign.yaml --output-dir results/
```

The simulate input CSV uses the **same unified campaign state format** as `moal plan`
(columns `smiles`, `relation`, `value`).  Only rows with `relation == "=="` (exact
DRC results) are loaded as oracle ground truth; primary screen rows (`<`, `>=`) and
unqueried rows (empty) are skipped.

If you have prior experimental data in the same mixed-fidelity format, you can
supply it as a pretrain pool, and the model will train on it
at every iteration alongside whatever the oracle acquires:

```yaml
data:
  simulate:
    input_csv: path/to/ground_truth.csv
    pretrain:
      input_csv: path/to/prior_data.csv   # same <, >=, == format as moal plan
```

CheMeleon pretrained weights are downloaded automatically from Zenodo on first
run and cached at `~/.chemprop/chemeleon_mp.pt` for subsequent use (only when
`model.from_foundation: chemeleon`, the default).

## Commands

### `simulate`

Runs the full cost-aware active learning loop:

```bash
moal simulate --config examples/default_config.yaml
moal simulate --config examples/default_config.yaml --output-dir results/
moal simulate --config examples/default_config.yaml --verbose
```

#### Optional pretrain data

If you have prior labeled data (from a previous campaign, a public assay, or
any source in the `moal plan` campaign state format), you can warm-start the
model at every iteration by pointing `data.simulate.pretrain.input_csv` at it.
The pretrain records are parsed with the same `<` / `>=` / `==` logic used by
`moal plan`, combined with oracle-acquired records before each `model.refit()`,
and deduplicated so that:

- Oracle records **always take precedence** over pretrain records at the same
  fidelity (the oracle has ground truth; the pretrain source may not).
- Pretrain PS INTERVAL records for compounds that the oracle later upgrades to
  DRC are **automatically dropped**: the exact label supersedes the censored one.
- Unqueried rows (empty `relation`/`value`) in the pretrain CSV are **skipped with
  a warning**, they carry no training signal.
- Compounds that overlap with the held-out test set trigger a **warning listing
  each SMILES**: training on them inflates model evaluation metrics.

```yaml
data:
  simulate:
    input_csv: path/to/ground_truth.csv
    pretrain:
      input_csv: path/to/prior_data.csv
      smiles_column: smiles       # column names default to the moal plan format
      relation_column: relation
      value_column: value
      is_canonical: false
```

Omitting `pretrain.input_csv` (the default) leaves the workflow completely
unchanged. `pretrain` is not supported with `model.fast = true` because
`NoisyOracleModel.refit()` ignores its records argument; the noisy oracle
already has access to ground truth and does not benefit from pretrain data.

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

A single CSV schema is shared by `moal simulate`, `moal plan`, and the optional
pretrain sub-input.  The expected base columns are `smiles`, `relation`, and
`value` (all configurable via `data.simulate.*`, `data.plan.*`, and
`data.simulate.pretrain.*` respectively).

| `relation` | `value` | Row state | Action in `moal simulate` | Action in `moal plan` |
|---|---|---|---|---|
| `==` | numeric | DRC result (exact) | **Oracle ground truth** | Training only; score columns are NaN |
| `<` | numeric | PS miss (inactive) | Skipped | Training only; score columns are NaN |
| `>=` | numeric | PS hit | Skipped | Training **and** DRC-upgrade inference target |
| empty | empty | Unqueried | Skipped | Model scores for PS **or** DRC |

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

The model performance panel requires a held-out test set (scaffold-split); the CLI creates one automatically. In headless/server environments, set `dashboard.show: false` in your YAML config.

## Progress Bar

The campaign emits a rich progress bar with `n_iterations × 3` discrete steps:

```
 ⠹  Iter 3/20  Querying oracle — 5 DRC (2 upgrades), 3 PS             ████░░  15%  0:00:12
 ⠹  Iter 3/20  Retraining model — 28 oracle + 60 pretrain records (DRC / PS)  ████░░  17%  0:00:45
 ⠹  Iter 3/20  Selecting (plate=1536) — 18 unqueried, 3 PS hits eligible for upgrade  ████░░  18%  0:00:46
```

## Key Design Notes

**Foundation model (`model.from_foundation`):** Controls which weights initialise the ChemProp message-passing encoder. Three values are accepted:

- `"chemeleon"` (default): downloads the CheMeleon checkpoint from Zenodo and loads it.
- A filesystem path string: loads a local checkpoint in the same `{hyper_parameters, state_dict}` format as CheMeleon.
- `false`: builds the encoder with default ChemProp architecture and random weights; no checkpoint required. Useful for ablation studies or environments without network access.

Unknown named strings and non-existent paths raise `ValueError` at model construction. The `from_foundation` value is recorded in Lightning checkpoints alongside all other hyperparameters.

**Unified input format:** All three CSV inputs (`data.simulate.input_csv`, `data.simulate.pretrain.input_csv`, and `data.plan.input_csv`) use the same campaign state schema (`smiles`, `relation`, `value`).  For `moal simulate`, only `==` rows are loaded as oracle ground truth; PS and blank rows are skipped.  For `moal plan` and the pretrain input, all labeled rows (`<`, `>=`, `==`) become training records; unqueried rows (empty) are inference targets or skipped with a warning, respectively.

**Pretrain warm-starting:** `moal simulate` accepts a pretrain CSV (`data.simulate.pretrain.input_csv`) in the same mixed-fidelity format. Pretrain records are merged with oracle-acquired records before each `model.refit()` call. Oracle records always win on a same-fidelity duplicate; pretrain PS INTERVAL records are automatically dropped when the oracle upgrades that compound to DRC. See `data.simulate.pretrain.*` in the config reference for all fields.

**Interval censoring:** Primary screen hits (`>= T`) are modeled as interval-censored `[T, 11.0]`, not right-censored at T. Using right-censoring would invert gradient signals for active compounds.

**Three distinct thresholds:**
- Assay detection limit (left-censoring boundary): pEC50 ≈ 4.0
- Primary screen hit threshold T (`ps_threshold`): pEC50 ≈ 5.0
- Optimization target (`activity_threshold`): pEC50 = 7.0

**Acquisition strategy (plate-budget greedy):** Each iteration scores two pools, unqueried compounds (eligible for PS or DRC) and PS-INTERVAL-labeled hits (eligible for DRC upgrade only), on the same cost-normalised scale:
- `score(x, DRC) = sigmoid((ŷ - 7.0) / τ) / cost_DRC`: exploits likely actives; applies equally to first-pass DRC and upgrade-DRC candidates
- `score(x, PS) = H_binary(sigmoid((ŷ - T) / τ)) / cost_PS`: cheaply resolves threshold ambiguity; only generated for unqueried compounds

Candidates are selected in score order until adding the next would exceed `active_learning_loop.plate_size` wells (each PS query costs `wells_per_ps`, each DRC costs `wells_per_drc`). When the next candidate overflows the plate the loop hard-stops; remaining candidates are deferred to the next iteration and rescored on the updated model. To replicate a flat query count of k, set `plate_size=k`, `wells_per_ps=1`, `wells_per_drc=1`.

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
│                       shared by `moal plan` and `moal simulate` pretrain loading
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
