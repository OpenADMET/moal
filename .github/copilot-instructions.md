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
    │   fidelity: PS → LEFT/INTERVAL label
    │   fidelity: DRC → EXACT label
    ▼
MixedFidelityDataModule ── train/val split (scaffold-unaware random split)
    │
    ▼
ChemPropLightningModule.refit()
    │  CensoredRegressionLoss (Tobit: EXACT / LEFT / INTERVAL branches)
    │  Freeze/unfreeze schedule: FFN head only for freeze_epochs, then encoder
    ▼
model.predict_smiles(unlabeled)  →  pEC50 point estimates (no uncertainty)
    │
    ▼
CostAwareGreedyAcquisition.select(unlabeled, predictions, k)
    │  DRC score = p_active(ŷ) / cost_DRC   [exploitation]
    │  PS  score = H(p_cross(ŷ, T)) / cost_PS  [threshold exploration]
    │  Greedy top-k, one query per compound
    ▼
ActiveLearningLoop.run()  →  LoopResults
    │  3 Rich progress steps per iteration: query → refit → select
    ▼
LiveDashboard (matplotlib, 3 panels: actives curve, cost stacks, model metric)
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
- **`TrainerConfig.to_dict()`** returns only `L.Trainer`-compatible keys. `val_fraction` and `split_seed` are consumed by `MixedFidelityDataModule` via `to_datamodule_kwargs()`. Passing them to `L.Trainer` raises `TypeError`.
- **`logging.basicConfig`** must only be called inside `cli.main()`, never at module scope. Module-level calls fire on import and reconfigure the root logger during test collection.

---

## Conventions

### Config hierarchy

All campaign parameters live in `moal/config.py` as frozen dataclasses. The YAML config maps directly to the nested hierarchy (`oracle:`, `model:`, `acquisition:`, `trainer:`, `dashboard:`, and top-level fields). `PipelineConfig.from_yaml()` is the single entry point for deserialization. When adding a new parameter, add it to the appropriate `@dataclass(frozen=True)` class and to `from_yaml()`.

### Label records

`LabelRecord` (frozen dataclass in `types.py`) is the canonical unit of labeled data throughout the pipeline. It carries the censoring type, fidelity, cost, and iteration alongside the SMILES. All training, evaluation, and dashboard code operates on `list[LabelRecord]` — do not pass raw floats between components.

### Logging

All modules use `logger = logging.getLogger(__name__)`. The `suppress_noisy_loggers()` function in `logging_config.py` must be called once at the start of `loop.run()` to silence PyTorch Lightning, RDKit, and matplotlib. Do not suppress moal's own loggers.

### Loss breakdown

`CensoredRegressionLoss.forward_with_breakdown()` returns a `LossBreakdown` NamedTuple with `total`, `drc_loss`, and `ps_loss`. The per-fidelity fields are `nan` (not zero) when no samples of that fidelity are in the batch — Lightning skips `nan` log values, which is intentional. `forward()` delegates to `forward_with_breakdown()`.

### Freeze/unfreeze schedule

`ChemPropLightningModule` freezes the CheMeleon encoder for the first `freeze_epochs` training epochs, then unfreezes and adds a second optimizer for the encoder at `lr_encoder`. The epoch counter resets on every `trainer.fit()` call (every AL iteration). This is intentional — early iterations have tiny labeled pools where encoder fine-tuning would overfit.

### Scaffold split

`scaffold_split()` in `evaluation.py` uses Bemis-Murcko scaffolds, assigns groups largest-first to the test set, and may slightly exceed the requested `test_size` if the first scaffold group is large. This is acceptable — scaffold groups cannot be split. This split is used only for the held-out model evaluation test set; the train/val split inside `MixedFidelityDataModule` is a plain random split.

### CLI filesystem isolation in tests

`CliRunner.invoke()` does **not** sandbox file I/O by default. Any test that triggers CLI output-directory creation must pass `--output-dir str(tmp_path / "out")` (or use `runner.isolated_filesystem()`) to avoid leaking `results/` into the pytest CWD.

### CheMeleon feature dimensions

`_CHEMPELEON_ATOM_FDIM = 72` and `_CHEMPELEON_BOND_FDIM = 14` in `model.py` are hardcoded to match the CheMeleon pretraining feature spec. These are verified at model initialization. Do not change them without updating the checkpoint.
