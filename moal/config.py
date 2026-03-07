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

    chempeleon_ckpt_path: str = ""
    """Path to the CheMeleon pretrained checkpoint file."""

    hidden_size: int = 300
    depth: int = 3
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

    def to_dict(self) -> dict[str, Any]:
        """Return only the kwargs that lightning.Trainer accepts.

        ``val_fraction`` and ``split_seed`` are consumed by
        ``MixedFidelityDataModule`` and must not be forwarded to
        ``L.Trainer``.
        """
        return {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "enable_progress_bar": self.enable_progress_bar,
            "enable_model_summary": self.enable_model_summary,
        }

    def to_datamodule_kwargs(self) -> dict[str, Any]:
        """Return kwargs for MixedFidelityDataModule."""
        return {
            "val_fraction": self.val_fraction,
            "seed": self.split_seed,
        }


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the live matplotlib campaign dashboard."""

    enabled: bool = True
    """Whether to show the live dashboard at all."""

    model_metric: str = "mae"
    """Metric to display in the model performance panel.
    Valid values: ``mae``, ``rmse``, ``kendall_tau``, ``spearman_r``, ``r2``."""

    save_dir: str = ""
    """Directory to write PNG snapshots after each iteration. Empty = don't save."""

    figsize: tuple[int, int] = (15, 4)
    """Overall figure size (width, height) in inches."""

    show: bool = True
    """If True, attempt interactive ``plt.ion()`` display.
    Set to False for headless/server environments."""


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a full active learning campaign."""

    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    n_iterations: int = 20
    """Number of active learning iterations (m)."""

    k_per_iteration: int = 10
    """Number of queries per iteration (k)."""

    test_set_size: float = 0.15
    """Fraction of the compound pool held out as a scaffold-split test set for
    model performance tracking. Set to 0.0 to disable test-set evaluation."""

    ground_truth_csv: str = ""
    """Path to CSV containing compound SMILES and pEC50 values."""

    smiles_column: str = "smiles"
    """Name of the column in ``ground_truth_csv`` that contains SMILES strings."""

    pec50_column: str = "pec50"
    """Name of the column in ``ground_truth_csv`` that contains pEC50 values."""

    is_canonical: bool = False
    """When False (default), all SMILES in ``ground_truth_csv`` are canonicalized
    via RDKit during oracle initialization.  Set to True if the input SMILES are
    already in canonical form to skip that preprocessing step."""

    output_dir: str = "results"
    """Directory to write campaign outputs (metrics CSV, model checkpoint)."""

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            oracle=OracleConfig(**raw.get("oracle", {})),
            model=ModelConfig(**raw.get("model", {})),
            acquisition=AcquisitionConfig(**raw.get("acquisition", {})),
            trainer=TrainerConfig(**raw.get("trainer", {})),
            dashboard=DashboardConfig(**raw.get("dashboard", {})),
            n_iterations=raw.get("n_iterations", 20),
            k_per_iteration=raw.get("k_per_iteration", 10),
            test_set_size=raw.get("test_set_size", 0.15),
            ground_truth_csv=raw.get("ground_truth_csv", ""),
            smiles_column=raw.get("smiles_column", "smiles"),
            pec50_column=raw.get("pec50_column", "pec50"),
            is_canonical=raw.get("is_canonical", False),
            output_dir=raw.get("output_dir", "results"),
            seed=raw.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file."""
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)
