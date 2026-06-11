---
description: "moal pipeline conventions: frozen-dataclass config, LabelRecord as the canonical data unit, interval-censored PS labels and the Tobit loss, pEC50 validation, chirality preservation, the trainer/datamodule kwarg split, and the three distinct threshold parameters. Apply when editing the moal package or its tests."
paths:
- 'moal/**/*.py'
- 'tests/**/*.py'
---

You are an expert contributor to moal, a cost-aware synthetic active-learning pipeline for cheminformatics. It runs two CLI modes: `moal simulate` (synthetic campaign over a ground-truth pool) and `moal plan` (score the next acquisition batch from a real mixed-fidelity campaign state). For general guidance see `python-core`, `python-docs`, `python-testing`, `python-packaging`, `python-security`, `machine-learning`, `chemoinformatics`, `medicinal-chemistry`, and `biology`.

## Principles

1. Configuration is immutable and centralized; behavior is reproducible from a config plus a seed.
2. Labeled data flows as typed records, never as bare floats passed between components.
3. Censoring is part of the label's meaning; losing it corrupts the gradient, not just a value.

## Configuration

- All campaign parameters live in `moal/config.py` as `@dataclass(frozen=True)` classes mirroring the YAML hierarchy. `PipelineConfig.from_yaml()` is the single deserialization entry point and `to_yaml()` the serializer; a new parameter is added to the right dataclass and to `from_yaml()`, never read ad hoc from a raw dict elsewhere.
- `TrainerConfig.to_dict()` returns only keys `L.Trainer` accepts (`max_epochs`, `accelerator`, `enable_progress_bar`, `enable_model_summary`, `log_every_n_steps`). The datamodule-only fields (`val_fraction`, `split_seed`, `num_workers`) go through `to_datamodule_kwargs()`; passing them to `L.Trainer` raises `TypeError`.

## Labeled data and censoring

- `LabelRecord` (frozen dataclass in `types.py`) is the canonical unit of labeled data; training, evaluation, and dashboard code operate on `list[LabelRecord]`, never raw floats between components.
- A PS hit (`>= T`) is INTERVAL-censored `[T, upper_bound]`, not right-censored at `T`; right-censoring inverts gradients for active compounds. Preserve the `CensoringType.INTERVAL` branch in `loss.py` (`CensoredRegressionLoss`, Tobit), which uses clamped CDF subtraction to avoid `log(0)`.
- Pass `oracle.training_records` (not `oracle.labeled_records`) to refit: it drops PS INTERVAL records for compounds that also have a DRC record, preventing double-weighting of upgraded compounds. The same dedup runs over the combined pool in `_merge_pretrain_with_oracle()`.

## Chemistry and data validation

- Keep `Chem.MolToSmiles(mol, isomericSmiles=True)` explicit in `preprocessing.py`; RDKit's default has shifted historically and chirality must be preserved.
- pEC50 values outside `[0.0, 14.0]`, plus NaN/inf, are excluded by the oracle before they reach the loss; never relax this, a NaN corrupts the whole training batch.

## Thresholds, splitting, and modes

- Three threshold parameters are distinct and must not be conflated: `oracle.ps_threshold` (where the PS assay draws its inequality), `acquisition.ps_threshold` (drives the PS entropy score, must equal the oracle value), and `target_threshold`/`activity_threshold` (what "active" means for DRC scoring and evaluation).
- `scaffold_split()` in `evaluation.py` (Bemis-Murcko, largest group to test first) is only for the held-out evaluation test set; the train/val split inside `MixedFidelityDataModule` is a plain random split. Do not swap one for the other.
- Fast mode (`model.fast = true`, `NoisyOracleModel`) is for prototyping and integration tests only, never real experiments; its `refit()` ignores its records argument by design.

## Logging and tests

- Every module uses `logging.getLogger(__name__)`; call `logging.basicConfig` only inside `cli.main()`, never at module scope (a module-level call reconfigures the root logger on import, during test collection). `suppress_noisy_loggers()` runs once at the start of `loop.run()` and must not silence moal's own loggers.
- `CliRunner.invoke()` does not sandbox file I/O; any test that triggers output-directory creation passes `--output-dir str(tmp_path / "out")` (or uses `runner.isolated_filesystem()`) so `results/` does not leak into the pytest CWD.
