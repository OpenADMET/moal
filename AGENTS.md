# Project Context: moal

`moal` is a cost-aware synthetic active-learning pipeline for cheminformatics. It runs mixed-fidelity campaigns over pEC50 data where the model decides both which compounds to query and at what assay cost, competing a dose-response (DRC, exploitation) score against a primary-screen (PS, threshold-exploration) score on a shared expected-value-per-dollar unit. It has two CLI modes: `moal simulate` runs a synthetic campaign over a ground-truth pool, and `moal plan` scores the next acquisition batch from a real campaign-state CSV for wet-lab prioritization.

## Environment

Python 3.11+ (the gate targets 3.12). Install editable with the dev extras; there is no uv project or lockfile.

```bash
pip install -e .[dev]
pre-commit install
```

Runtime dependencies (chemprop, lightning, torch, rdkit, pandas, scipy, click, plotly/dash) are declared with ranged bounds in `pyproject.toml`; add new ones there, never with ad-hoc installs.

## Commands

```bash
# Run a campaign / score a batch
moal simulate --config examples/default_config.yaml --output-dir results/
moal plan --config examples/default_config.yaml --output-dir results/

# Tests
python -m pytest tests/
python -m pytest tests/test_loss.py::TestLossBreakdown::test_forward_consistent_with_breakdown

# Lint, format, and type gate (also wired as pre-commit)
pre-commit run --all-files
```

The gate is ruff (`E,W,F,I,N,UP,B,S,D`, numpy docstrings, line length 100), ruff-format, and pyright (basic, over `moal/`).

## Architecture

A campaign assembles these components, all configured from one YAML mapped onto frozen dataclasses in `moal/config.py` (`PipelineConfig.from_yaml()` is the single deserialization entry point):

- **`CostAwareOracle`** (`oracle.py`) answers queries against the ground truth, emitting `LabelRecord`s. A PS query yields a LEFT-censored (miss) or INTERVAL-censored (hit) label; a DRC query yields an EXACT label; PS hits can later be upgraded to DRC.
- **`LabelRecord`** (`types.py`) is the canonical unit of labeled data: censoring type, fidelity, cost, bounds, and SMILES. All training, evaluation, and dashboard code operates on `list[LabelRecord]`.
- **`ChemPropLightningModule`** (`model.py`) refits each iteration under `CensoredRegressionLoss` (`loss.py`), a Tobit loss with EXACT/LEFT/INTERVAL branches. The encoder is a ChemProp message-passing network initialized from a foundation model (`from_foundation`, default the CheMeleon checkpoint downloaded from Zenodo) and frozen for `freeze_epochs` before fine-tuning. `NoisyOracleModel` (fast mode) substitutes noisy ground-truth lookups for training, for prototyping and tests only.
- **`CostAwareGreedyAcquisition`** (`acquisition.py`) ranks compounds by the better of the DRC and PS per-dollar scores and fills the plate greedily.
- **`ActiveLearningLoop`** (`loop.py`) drives query → refit → select per iteration; `LiveDashboard` (`dashboard.py`, Plotly/Dash) visualizes progress and exports HTML/GIF.

`moal plan` reuses the model and acquisition over a parsed `CampaignState` (`planning.py`) instead of the oracle, annotating the input CSV with PS/DRC/overall scores and a recommendation.

## Conventions

- Configuration is immutable: parameters live as frozen-dataclass fields in `config.py`; add a parameter to the dataclass and to `from_yaml()`, never read a raw dict downstream.
- Pass typed `LabelRecord`s between components, never bare floats.
- Three threshold parameters are distinct and must not be conflated: `oracle.ps_threshold`, `acquisition.ps_threshold` (must match the oracle value), and `target_threshold`/`activity_threshold`.
- Every module uses `logging.getLogger(__name__)`; `logging.basicConfig` is called only inside `cli.main()`.

## Review personas

Invoke the adversarial reviewer matching the change under review for a domain critique:

- **Machine Learning Expert**: splits, the active-learning loop, the censored loss, evaluation metrics (data leakage, train/serve skew).
- **Medicinal Chemist**: pEC50 handling, potency thresholds, and SAR interpretation (units, log space, censored labels).
- **Chemoinformatician**: SMILES handling, featurization, and scaffold-aware splitting (sanitization, stereochemistry preservation).
- **Biologist**: pEC50 as assay data, primary-screen vs dose-response semantics, and activity thresholds (affinity vs potency, assay mechanism).
