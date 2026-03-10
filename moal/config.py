"""Pipeline configuration dataclasses.

All configuration for a campaign is expressed as a nested hierarchy of
frozen dataclasses, making configs serializable, inspectable, and hashable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OracleConfig:
    """Oracle and query cost parameters."""

    cost_ps: float = 1.0
    """Cost in dollars for a Primary Screen query."""

    cost_drc: float = 10.0
    """Cost in dollars for a Dose-Response Curve query."""

    ps_threshold: float = 5.0
    """pEC50 threshold used by the primary screen (e.g., 5.0 = 10 μM IC50).
    Compounds with true pEC50 < ps_threshold receive a LEFT label;
    compounds with pEC50 >= ps_threshold receive an INTERVAL label."""

    upper_bound: float = 11.0
    """Practical upper ceiling of the pEC50 scale for INTERVAL labels."""

    activity_threshold: float = 7.0
    """pEC50 threshold defining an 'active' compound for evaluation."""


@dataclass(frozen=True)
class ModelConfig:
    """ChemProp + CheMeleon model parameters."""

    ffn_hidden_size: int = 300
    ffn_num_layers: int = 2

    freeze_epochs: int = 10
    """Number of warm-up epochs to train only the FFN head."""

    lr_encoder: float = 1e-5
    """Learning rate for the message-passing encoder after unfreezing."""

    lr_head: float = 1e-3
    """Learning rate for the FFN head throughout training."""

    sigma: float = 0.5
    """Fixed noise scale for the Tobit loss (pEC50 log-units)."""

    w_drc: float = 1.0
    w_ps: float = 0.3
    learnable_sigma: bool = False

    reset_weights_on_refit: bool = False
    """When True, reload pretrained CheMeleon weights before each refit.
    Default is False, which warm-starts each iteration from the current model
    weights."""

    fast: bool = False
    """When True, bypass CheMeleon and use NoisyOracleModel instead.
    No checkpoint is required. Intended for rapid experimentation and testing."""

    initial_error: float = 0.7
    """Starting noise magnitude (pEC50 log-units) for the error ramp in fast mode.
    Uniform(-initial_error, +initial_error) noise is applied at iteration 0."""

    final_error: float = 0.5
    """Ending noise magnitude (pEC50 log-units) for the error ramp in fast mode.
    The ramp linearly interpolates from initial_error to final_error over all iterations.
    Set equal to initial_error for a constant noise level."""


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition function hyper-parameters."""

    ps_threshold: float = 5.0
    """PS threshold used by the entropy score. Should match OracleConfig.ps_threshold."""

    target_threshold: float = 7.0
    """Optimization target threshold used by the DRC exploitation score."""

    tau: float = 0.5
    """Sigmoid temperature. Lower = more exploitative."""


@dataclass(frozen=True)
class TrainerConfig:
    """Keyword arguments forwarded to lightning.Trainer during refit."""

    max_epochs: int = 30
    accelerator: str = "auto"
    enable_progress_bar: bool = False
    enable_model_summary: bool = False

    val_fraction: float = 0.1
    """Fraction of labeled records held out for validation during refit.
    Exposed here so it is visible and reproducible from the YAML config."""

    split_seed: int = 42
    """Random seed for the train/val split inside MixedFidelityDataModule.
    Changing this produces a different val set without touching the oracle."""

    num_workers: int = 1
    """Number of worker processes for train and val DataLoaders.
    0 = load data in the main process (slowest, avoids multiprocessing overhead).
    1 = one background worker per DataLoader (recommended default).
    Increase for large datasets where data loading is the bottleneck."""

    log_every_n_steps: int = 1
    """How often (in training steps) Lightning logs metrics.

    Default is 1 rather than Lightning's built-in default of 50 because AL
    labeled pools are small: even late in a campaign the number of training
    batches per epoch is typically well below 50, which would trigger a
    ``UserWarning`` from Lightning and suppress all step-level logs.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return only the kwargs that ``lightning.Trainer`` accepts.

        ``val_fraction``, ``split_seed``, and ``num_workers`` are consumed by
        ``MixedFidelityDataModule`` and must not be forwarded to
        ``L.Trainer``.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``max_epochs``, ``accelerator``,
            ``enable_progress_bar``, ``enable_model_summary``, and
            ``log_every_n_steps``.
        """
        return {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "enable_progress_bar": self.enable_progress_bar,
            "enable_model_summary": self.enable_model_summary,
            "log_every_n_steps": self.log_every_n_steps,
        }

    def to_datamodule_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``MixedFidelityDataModule``.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``val_fraction``, ``seed``, and ``num_workers``.
        """
        return {
            "val_fraction": self.val_fraction,
            "seed": self.split_seed,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the live matplotlib campaign dashboard."""

    enabled: bool = True
    """Whether to show the live dashboard at all."""

    model_metric: str = "mae"
    """Metric to display in the model performance panel.
    Valid values: ``mae``, ``rmse``, ``kendall_tau``, ``spearman_r``, ``r2``."""

    figsize: tuple[int, int] = (15, 4)
    """Overall figure size (width, height) in inches."""

    show: bool = True
    """If True, attempt interactive ``plt.ion()`` display.
    Set to False for headless/server environments."""


@dataclass(frozen=True)
class SimulationDataConfig:
    """Dataset settings for the synthetic active-learning simulation workflow."""

    input_csv: str = ""
    """Path to CSV containing compound SMILES and pEC50 values."""

    smiles_column: str = "smiles"
    """Name of the column in ``input_csv`` that contains SMILES strings."""

    pec50_column: str = "pec50"
    """Name of the column in ``input_csv`` that contains pEC50 values."""

    is_canonical: bool = False
    """When False (default), input SMILES are canonicalized via RDKit during
    oracle initialization. Set to True when the CSV already stores canonical
    SMILES and preprocessing can be skipped."""

    test_set_size: float = 0.15
    """Fraction of the compound pool held out as a scaffold-split test set for
    model performance tracking. Set to 0.0 to disable test-set evaluation."""


@dataclass(frozen=True)
class PlanTrainingDataConfig:
    """Training-set settings for offline plan mode."""

    input_csv: str = ""
    """Path to CSV containing labeled mixed-fidelity training records."""

    smiles_column: str = "smiles"
    """Name of the SMILES column in ``input_csv``."""

    relation_column: str = "relation"
    """Name of the relation column in ``input_csv`` (`<`, `>=`, `==`)."""

    value_column: str = "value"
    """Name of the pEC50 / threshold value column in ``input_csv``."""

    is_canonical: bool = False
    """When False (default), training-set SMILES are canonicalized via RDKit
    before validation and scoring."""


@dataclass(frozen=True)
class PlanCandidatePoolDataConfig:
    """Candidate-pool settings for offline plan mode."""

    input_csv: str = ""
    """Path to CSV containing unlabeled candidate compounds."""

    smiles_column: str = "smiles"
    """Name of the SMILES column in ``input_csv``."""

    is_canonical: bool = False
    """When False (default), candidate-pool SMILES are canonicalized via RDKit
    before validation and scoring."""


@dataclass(frozen=True)
class PlanDataConfig:
    """Input/output settings for offline train-and-rank planning."""

    output_csv: str = "acquisition_plan.csv"
    """Destination for the ranked acquisition plan CSV.

    Relative paths are resolved under ``data.output_dir`` so both subcommands can
    be redirected together with ``--output-dir``.
    """

    training: PlanTrainingDataConfig = field(default_factory=PlanTrainingDataConfig)
    """Training-set settings used by ``moal plan``."""

    candidate_pool: PlanCandidatePoolDataConfig = field(
        default_factory=PlanCandidatePoolDataConfig
    )
    """Candidate-pool settings used by ``moal plan``."""


@dataclass(frozen=True)
class DataConfig:
    """Command-specific dataset and I/O settings."""

    output_dir: str = "results"
    """Directory to write campaign outputs and config snapshots."""

    simulate: SimulationDataConfig = field(default_factory=SimulationDataConfig)
    """Dataset settings used by ``moal simulate``."""

    plan: PlanDataConfig = field(default_factory=PlanDataConfig)
    """Dataset settings used by ``moal plan``."""


@dataclass(frozen=True)
class ActiveLearningLoopConfig:
    """Parameters controlling the active learning iteration loop."""

    n_iterations: int = 20
    """Number of active learning iterations (m)."""

    k_per_iteration: int = 10
    """Number of oracle queries issued per iteration (k)."""


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a full active learning campaign."""

    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    active_learning_loop: ActiveLearningLoopConfig = field(
        default_factory=ActiveLearningLoopConfig
    )

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a ``PipelineConfig`` from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        PipelineConfig
            Fully populated pipeline configuration instance.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        data_raw = raw.get("data", {})
        return cls(
            oracle=OracleConfig(**raw.get("oracle", {})),
            model=ModelConfig(**raw.get("model", {})),
            acquisition=AcquisitionConfig(**raw.get("acquisition", {})),
            trainer=TrainerConfig(**raw.get("trainer", {})),
            dashboard=DashboardConfig(**raw.get("dashboard", {})),
            data=DataConfig(
                output_dir=data_raw.get("output_dir", "results"),
                simulate=SimulationDataConfig(**data_raw.get("simulate", {})),
                plan=PlanDataConfig(
                    output_csv=data_raw.get("plan", {}).get(
                        "output_csv", "acquisition_plan.csv"
                    ),
                    training=PlanTrainingDataConfig(
                        **data_raw.get("plan", {}).get("training", {})
                    ),
                    candidate_pool=PlanCandidatePoolDataConfig(
                        **data_raw.get("plan", {}).get("candidate_pool", {})
                    ),
                ),
            ),
            active_learning_loop=ActiveLearningLoopConfig(
                **raw.get("active_learning_loop", {})
            ),
            seed=raw.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path for the serialized YAML.
        """
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)
