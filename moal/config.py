"""Pipeline configuration dataclasses.

All configuration for a campaign is expressed as a nested hierarchy of
frozen dataclasses, making configs serializable, inspectable, and hashable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OracleConfig:
    """Oracle and query cost parameters.

    Attributes
    ----------
    cost_ps : float
        Cost in dollars for a Primary Screen query.
    cost_drc : float
        Cost in dollars for a Dose-Response Curve query.
    ps_threshold : float
        pEC50 threshold used by the primary screen (e.g., 5.0 = 10 μM IC50).
        Compounds with true pEC50 < ps_threshold receive a LEFT label;
        compounds with pEC50 >= ps_threshold receive an INTERVAL label.
    upper_bound : float
        Practical upper ceiling of the pEC50 scale for INTERVAL labels.
    activity_threshold : float
        pEC50 threshold defining an 'active' compound for evaluation.
    """

    cost_ps: float = 1.0
    cost_drc: float = 10.0
    ps_threshold: float = 5.0
    upper_bound: float = 11.0
    activity_threshold: float = 7.0


@dataclass(frozen=True)
class ModelConfig:
    """ChemProp + CheMeleon model parameters.

    Attributes
    ----------
    ffn_hidden_dim : int
        Hidden layer size of the FFN prediction head.
    ffn_num_layers : int
        Number of layers in the FFN prediction head.
    message_hidden_dim : int
        Message-passing hidden width (``d_h``) for the random-init encoder.
        Used only when ``from_foundation=False``; a foundation checkpoint
        supplies its own width.
    depth : int
        Number of message-passing steps for the random-init encoder. Used only
        when ``from_foundation=False``; a foundation checkpoint supplies its own
        depth.
    freeze_epochs : int
        Number of warm-up epochs to train only the FFN head.
    mpnn_lr : float
        Learning rate for the message-passing encoder after unfreezing.
    ffn_lr : float
        Learning rate for the FFN head throughout training.
    mpnn_weight_decay : float
        L2 weight decay for the message-passing encoder param group.
    ffn_weight_decay : float
        L2 weight decay for the FFN head param group.
    sigma : float
        Fixed noise scale for the Tobit loss (pEC50 log-units).
    w_drc : float
        Loss weight applied to DRC (EXACT-censored) samples.
    w_ps : float
        Loss weight applied to PS (LEFT/INTERVAL-censored) samples.
    learnable_sigma : bool
        When True, sigma is a learnable parameter rather than fixed.
    reset_weights_on_refit : bool
        When True, reload pretrained CheMeleon weights before each refit.
        Default is False, which warm-starts each iteration from the current
        model weights.
    fast : bool
        When True, bypass CheMeleon and use NoisyOracleModel instead.
        No checkpoint is required. Intended for rapid experimentation and
        testing.
    initial_error : float
        Starting noise magnitude (pEC50 log-units) for the error ramp in fast
        mode. Uniform(-initial_error, +initial_error) noise is applied at
        iteration 0.
    final_error : float
        Ending noise magnitude (pEC50 log-units) for the error ramp in fast
        mode. The ramp linearly interpolates from initial_error to final_error
        over all iterations. Set equal to initial_error for a constant noise
        level.
    from_foundation : str or bool
        Controls encoder initialisation. ``"chemeleon"`` (default) downloads
        and loads CheMeleon pretrained weights. A filesystem path string loads
        a local checkpoint in the same ``{hyper_parameters, state_dict}``
        format. ``False`` builds the encoder with default ChemProp architecture
        and random weights (no checkpoint required).
    """

    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    message_hidden_dim: int = 300
    depth: int = 3
    freeze_epochs: int = 10
    mpnn_lr: float = 1e-5
    ffn_lr: float = 1e-3
    mpnn_weight_decay: float = 0.0
    ffn_weight_decay: float = 0.0
    sigma: float = 0.5
    w_drc: float = 1.0
    w_ps: float = 0.3
    learnable_sigma: bool = False
    reset_weights_on_refit: bool = False
    fast: bool = False
    initial_error: float = 0.7
    final_error: float = 0.5
    from_foundation: str | bool = "chemeleon"


@dataclass(frozen=True)
class AuxiliaryModelConfig:
    """Auxiliary encoder architecture for the ``LabelRecord.raw_ps_readouts`` signal.

    ``moal plan``-only (see the ``moal simulate`` exclusion in the module
    docstring reference, issue #36). Off by default: ``moal plan`` behaves
    exactly as it does today unless this config is explicitly set.

    Shares the main model's ChemProp/CheMeleon backbone construction
    (``ModelConfig.from_foundation``, mean-pooling readout) rather than a
    bespoke architecture, so its embeddings live in the same representation
    space. When ``from_foundation="chemeleon"``, the readout is constrained
    to mean aggregation to match CheMeleon's own pretraining; the paper's
    recommended attentive readout is only reachable with
    ``from_foundation=False``.

    Trains a masked multi-task regression head, one output per distinct key
    observed across ``raw_ps_readouts`` (e.g. one head per log2FC
    concentration, plus a head for a direct pIC50 column when present).
    Compounds missing a given key contribute no gradient to that head.

    Readouts are used as-is, with no per-plate/per-batch normalization step.
    The design this config implements (issue #36) specified that step as a
    named, non-optional prerequisite for pretraining; it is not implemented
    here and is a known, documented limitation until `moal`'s campaign-state
    schema gains a plate/batch identifier.

    Attributes
    ----------
    from_foundation : str or bool
        Encoder initialisation, forwarded to :func:`moal.model.build_mpnn`.
        Same semantics as ``ModelConfig.from_foundation``. Default
        ``"chemeleon"`` shares the main model's foundation checkpoint.
    ffn_hidden_dim : int
        Hidden dimension of the multi-task FFN predictor head.
    ffn_num_layers : int
        Number of layers in the multi-task FFN predictor head.
    message_hidden_dim : int
        Message-passing hidden width (``d_h``) for the random-init encoder.
        Used only when ``from_foundation=False``.
    depth : int
        Number of message-passing steps for the random-init encoder. Used
        only when ``from_foundation=False``.
    freeze_epochs : int
        Number of warm-up epochs to train only the multi-task FFN head,
        analogous to ``ModelConfig.freeze_epochs`` but scheduled
        independently for the auxiliary encoder.
    lr : float
        Learning rate for the multi-task FFN head, and for the message-passing
        encoder after unfreezing (no separate discriminative rate, unlike the
        main model's ``mpnn_lr`` / ``ffn_lr`` split).
    weight_decay : float
        L2 weight decay applied to all trainable parameters.
    embedding_dim : int
        Dimensionality of the pooled molecular embedding exposed to the main
        model's concatenation architecture (Phase 2). Ignored by the
        retrained-encoder architecture, which has no separate embedding
        output at inference.
    checkpoint_path : str or None
        Explicit opt-in path to a cached auxiliary-encoder checkpoint. When
        set, pretraining is skipped and this checkpoint is loaded instead.
        When None (default), the auxiliary encoder is retrained from scratch
        on every ``moal plan`` invocation using the current campaign-state
        CSV's ``raw_ps_readouts``, so newly accumulated readouts improve the
        next run automatically.
    use_observed_readout : bool
        Controls the concatenation architecture's main-model input, not the
        auxiliary encoder's own pretraining (which always uses whatever
        ``raw_ps_readouts`` exist, regardless of this flag). The auxiliary
        encoder's structural embedding is always concatenated into the main
        model for every compound. When True (default), a compound with an
        observed readout *additionally* gets its raw value concatenated
        alongside the embedding. When False, every compound is scored from
        its embedding alone and the readout/mask blocks stay zero for all
        compounds; a constant-zero input column is a mathematical no-op for
        a plain linear layer (zero gradient, zero forward contribution), so
        this does not degrade model capacity.
    """

    from_foundation: str | bool = "chemeleon"
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    message_hidden_dim: int = 300
    depth: int = 3
    freeze_epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.0
    embedding_dim: int = 300
    checkpoint_path: str | None = None
    use_observed_readout: bool = True


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition function hyper-parameters.

    Attributes
    ----------
    ps_threshold : float
        PS threshold used by the entropy score. Should match
        OracleConfig.ps_threshold.
    target_threshold : float
        Optimization target threshold used by the DRC exploitation score.
    tau : float
        Sigmoid temperature. Lower = more exploitative.
    embedding_provenance_discount : float
        Multiplicative discount applied to a candidate's acquisition score
        when its prediction rests on the concatenation architecture's
        (issue #36 Phase 2) auxiliary-embedding path rather than an observed
        readout — i.e. a compound never PS-screened, scored through one more
        layer of inference than an observed-input prediction. Must be in
        ``(0.0, 1.0]``; 1.0 (default) is a no-op, so acquisition behavior is
        unchanged unless a caller explicitly passes per-candidate provenance
        to :meth:`~moal.acquisition.CostAwareGreedyAcquisition.select` or
        :meth:`~moal.acquisition.CostAwareGreedyAcquisition.score_summary`
        *and* sets this below 1.0.
    """

    ps_threshold: float = 5.0
    target_threshold: float = 7.0
    tau: float = 0.5
    embedding_provenance_discount: float = 1.0


@dataclass(frozen=True)
class TrainerConfig:
    """Keyword arguments forwarded to lightning.Trainer during refit.

    Attributes
    ----------
    max_epochs : int
        Maximum number of training epochs per refit call.
    accelerator : str
        Hardware accelerator passed to ``lightning.Trainer`` (e.g., ``"auto"``,
        ``"cpu"``, ``"gpu"``).
    enable_progress_bar : bool
        Whether to display a Lightning progress bar during training.
    enable_model_summary : bool
        Whether to print the Lightning model summary before training.
    val_fraction : float
        Fraction of labeled records held out for validation during refit.
        Exposed here so it is visible and reproducible from the YAML config.
    split_seed : int
        Random seed for the train/val split inside MixedFidelityDataModule.
        Changing this produces a different val set without touching the oracle.
    num_workers : int
        Number of worker processes for train and val DataLoaders.
        0 = load data in the main process (slowest, avoids multiprocessing
        overhead). 1 = one background worker per DataLoader (recommended
        default). Increase for large datasets where data loading is the
        bottleneck.
    log_every_n_steps : int
        How often (in training steps) Lightning logs metrics. Default is 1
        rather than Lightning's built-in default of 50 because AL labeled
        pools are small: even late in a campaign the number of training
        batches per epoch is typically well below 50, which would trigger a
        ``UserWarning`` from Lightning and suppress all step-level logs.
    gradient_clip_val : float or None
        Gradient clipping threshold passed to ``lightning.Trainer``. None
        (default) disables clipping. Useful for stabilising training when the
        censored mixed-fidelity loss produces large gradients.
    gradient_clip_algorithm : str
        Clipping algorithm passed to ``lightning.Trainer`` when
        ``gradient_clip_val`` is set: ``"norm"`` (default) or ``"value"``.
        Ignored when ``gradient_clip_val`` is None.
    """

    max_epochs: int = 30
    accelerator: str = "auto"
    enable_progress_bar: bool = False
    enable_model_summary: bool = False
    val_fraction: float = 0.1
    split_seed: int = 42
    num_workers: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: float | None = None
    gradient_clip_algorithm: str = "norm"

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
            ``log_every_n_steps``.  ``gradient_clip_val`` (and
            ``gradient_clip_algorithm``) are added only when clipping is
            enabled, so the default (None) leaves Lightning's clipping off.
        """
        kwargs: dict[str, Any] = {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "enable_progress_bar": self.enable_progress_bar,
            "enable_model_summary": self.enable_model_summary,
            "log_every_n_steps": self.log_every_n_steps,
        }
        # Forward clipping only when enabled; passing gradient_clip_algorithm
        # with gradient_clip_val=None raises in lightning.Trainer
        if self.gradient_clip_val is not None:
            kwargs["gradient_clip_val"] = self.gradient_clip_val
            kwargs["gradient_clip_algorithm"] = self.gradient_clip_algorithm
        return kwargs

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
    """Configuration for the Plotly + Dash live campaign dashboard.

    Attributes
    ----------
    enabled : bool
        Whether to run the live dashboard server at all.
    model_metric : str
        Metric to display in the model performance panel. Valid values:
        ``mae``, ``rmse``, ``kendall_tau``, ``spearman_r``, ``r2``.
    port : int
        Local port for the Dash server (bound to 127.0.0.1 only).
    export_width : int
        Pixel width used when exporting static PNG frames (requires kaleido).
    export_height : int
        Pixel height used when exporting static PNG frames (requires kaleido).
    theme : str
        Plotly template applied to all figure renders (live browser, HTML, and
        GIF). Any valid Plotly template name is accepted, e.g. ``plotly``,
        ``plotly_white``, ``ggplot2``, ``seaborn``, ``simple_white``.
    """

    enabled: bool = True
    model_metric: str = "mae"
    port: int = 8050
    export_width: int = 1400
    export_height: int = 800
    theme: str = "plotly_dark"


@dataclass(frozen=True)
class PretrainDataConfig:
    """Settings for an optional pretraining dataset supplied to ``moal simulate``.

    The pretraining CSV uses the same mixed-fidelity campaign state format as
    ``moal plan``: each row carries a SMILES string, a relation symbol
    (``<``, ``>=``, or ``==``), and a numeric pEC50 value.  Rows with empty
    relation/value fields are skipped with a warning — they provide no
    training signal.

    When ``input_csv`` is empty (the default), pretraining is disabled and
    the simulation workflow is identical to its previous behaviour.

    Attributes
    ----------
    input_csv : str
        Path to the pretrain CSV.  Leave empty to disable pretraining.
    smiles_column : str
        Name of the SMILES column in ``input_csv``.
    relation_column : str
        Name of the relation column (``<``, ``>=``, ``==``, or empty).
    value_column : str
        Name of the pEC50 / threshold value column.
    weight_column : str or None
        Optional column name for per-sample loss weights. When set, each
        labeled row's weight is read from this column (NaN / missing cells
        default to 1.0). When None (default), all records receive weight=1.0.
    log2fc_columns : list[str] or None
        Optional column names for observed continuous auxiliary readouts
        (e.g. log2FC at one or more primary-screen concentrations, a direct
        pIC50). When set, populates ``LabelRecord.raw_ps_readouts`` keyed by
        column name. When None (default), ``raw_ps_readouts`` is empty for
        every record.
    is_canonical : bool
        When False (default), SMILES are canonicalized via RDKit during
        parsing.
    """

    input_csv: str = ""
    smiles_column: str = "smiles"
    relation_column: str = "relation"
    value_column: str = "value"
    weight_column: str | None = None
    log2fc_columns: list[str] | None = None
    is_canonical: bool = False


@dataclass(frozen=True)
class SimulationDataConfig:
    """Dataset settings for the synthetic active-learning simulation workflow.

    The input CSV uses the same unified campaign state format as ``moal plan``
    and the pretrain sub-input: each row carries a SMILES string, a relation
    symbol (``<``, ``>=``, ``==``, or empty), and a numeric pEC50 value.
    Only rows with ``relation == "=="`` (exact DRC results) are used to build
    the oracle ground truth pool.  Primary screen rows (``<``, ``>=``) and
    unqueried rows (empty relation/value) are skipped with a log message.

    Attributes
    ----------
    input_csv : str
        Path to the campaign state CSV.  Must contain at least one row with
        ``relation == "=="``; all other rows are skipped.
    smiles_column : str
        Name of the SMILES column in ``input_csv``.
    relation_column : str
        Name of the relation column in ``input_csv`` (``<``, ``>=``, ``==``,
        or empty).
    value_column : str
        Name of the pEC50 / threshold value column in ``input_csv``.
    is_canonical : bool
        When False (default), input SMILES are canonicalized via RDKit during
        oracle initialization. Set to True when the CSV already stores
        canonical SMILES and preprocessing can be skipped.
    test_set_size : float
        Fraction of the compound pool held out as a scaffold-split test set
        for model performance tracking. Set to 0.0 to disable test-set
        evaluation.
    pretrain : PretrainDataConfig
        Optional pretraining dataset in campaign state CSV format.  Pretrain
        records are combined with oracle-acquired records at every
        ``model.refit()`` call.  When ``pretrain.input_csv`` is empty
        (default), behaviour is unchanged from the no-pretrain workflow.
    """

    input_csv: str = ""
    smiles_column: str = "smiles"
    relation_column: str = "relation"
    value_column: str = "value"
    is_canonical: bool = False
    test_set_size: float = 0.15
    pretrain: PretrainDataConfig = field(default_factory=PretrainDataConfig)


@dataclass(frozen=True)
class PlanDataConfig:
    """Input/output settings for offline train-and-rank planning.

    Attributes
    ----------
    input_csv : str
        Path to the unified campaign state CSV. Expected columns: SMILES,
        relation (``<``, ``>=``, ``==``, or empty), and value (numeric pEC50
        or empty). Rows with populated relation and value are treated as the
        labeled training set; rows with both fields empty are treated as
        unqueried inference targets; PS-INTERVAL rows (``>=``) are treated as
        both training records and DRC-upgrade inference targets.
    output_csv : str
        Destination for the annotated state CSV. Relative paths are resolved
        under ``data.output_dir``. The output preserves the original columns
        and appends ``ps_score``, ``drc_score``, ``overall_score``, and
        ``recommendation`` so the file can be re-ingested in the next
        iteration.
    smiles_column : str
        Name of the SMILES column in ``input_csv``.
    relation_column : str
        Name of the relation column in ``input_csv`` (``<``, ``>=``, ``==``,
        or empty).
    value_column : str
        Name of the pEC50 / threshold value column in ``input_csv``.
    weight_column : str or None
        Optional column name for per-sample loss weights. When set, each
        labeled row's weight is read from this column (NaN / missing cells
        default to 1.0). When None (default), all records receive weight=1.0.
    log2fc_columns : list[str] or None
        Optional column names for observed continuous auxiliary readouts
        (e.g. log2FC at one or more primary-screen concentrations, a direct
        pIC50). When set, populates ``LabelRecord.raw_ps_readouts`` keyed by
        column name for PS rows (and DRC rows for upgraded compounds, when
        the CSV carries the readout on that row). When None (default),
        ``raw_ps_readouts`` is empty for every record.
    is_canonical : bool
        When False (default), SMILES are canonicalized via RDKit during
        parsing.
    """

    input_csv: str = ""
    output_csv: str = "campaign_state.csv"
    smiles_column: str = "smiles"
    relation_column: str = "relation"
    value_column: str = "value"
    weight_column: str | None = None
    log2fc_columns: list[str] | None = None
    is_canonical: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Command-specific dataset and I/O settings.

    Attributes
    ----------
    output_dir : str
        Directory to write campaign outputs and config snapshots.
    simulate : SimulationDataConfig
        Dataset settings used by ``moal simulate``.
    plan : PlanDataConfig
        Dataset settings used by ``moal plan``.
    """

    output_dir: str = "results"
    simulate: SimulationDataConfig = field(default_factory=SimulationDataConfig)
    plan: PlanDataConfig = field(default_factory=PlanDataConfig)


@dataclass(frozen=True)
class ActiveLearningLoopConfig:
    """Parameters controlling the active learning iteration loop.

    Attributes
    ----------
    n_iterations : int
        Number of active learning iterations (m).
    plate_size : int
        Maximum number of wells available per plate (i.e., per iteration).
        The acquisition greedily selects ranked candidates in score order,
        stopping as soon as the next candidate would push the total well
        count over this limit.  Any remaining plate capacity is accepted
        and the unused candidates are deferred to the next iteration, where
        the model will be re-scored on the updated labeled pool.
    wells_per_ps : int
        Number of wells consumed by a single Primary Screen (PS) query.
        Typically 1 for a singlet primary screen.
    wells_per_drc : int
        Number of wells consumed by a single Dose-Response Curve (DRC) query.
        For example, a 13-point DRC in duplicate consumes 26 wells; a compact
        8-point singlet DRC consumes 8.
    """

    n_iterations: int = 20
    plate_size: int = 1536
    wells_per_ps: int = 1
    wells_per_drc: int = 13


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a full active learning campaign.

    Attributes
    ----------
    oracle : OracleConfig
        Oracle and query cost parameters.
    model : ModelConfig
        ChemProp + CheMeleon model parameters.
    acquisition : AcquisitionConfig
        Acquisition function hyper-parameters.
    trainer : TrainerConfig
        Keyword arguments forwarded to lightning.Trainer during refit.
    dashboard : DashboardConfig
        Live campaign dashboard configuration.
    data : DataConfig
        Command-specific dataset and I/O settings.
    active_learning_loop : ActiveLearningLoopConfig
        Parameters controlling the active learning iteration loop.
    auxiliary_model : AuxiliaryModelConfig or None
        Optional auxiliary log2FC/pIC50 encoder architecture for ``moal plan``
        (issue #36). ``None`` (default) disables the feature entirely;
        ``moal plan`` behaves exactly as it does without this config.
    auxiliary_trainer : TrainerConfig
        Keyword arguments forwarded to ``lightning.Trainer`` during auxiliary
        encoder pretraining, scheduled independently from the main model's
        ``trainer``. Unused when ``auxiliary_model`` is None.
    seed : int
        Global random seed for the campaign.
    """

    oracle: OracleConfig = field(default_factory=OracleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    active_learning_loop: ActiveLearningLoopConfig = field(default_factory=ActiveLearningLoopConfig)
    auxiliary_model: AuxiliaryModelConfig | None = None
    auxiliary_trainer: TrainerConfig = field(default_factory=TrainerConfig)

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
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
        simulate_raw = data_raw.get("simulate", {})
        pretrain_raw = simulate_raw.pop("pretrain", {}) if isinstance(simulate_raw, dict) else {}
        auxiliary_model_raw = raw.get("auxiliary_model", None)
        return cls(
            oracle=OracleConfig(**raw.get("oracle", {})),
            model=ModelConfig(**raw.get("model", {})),
            acquisition=AcquisitionConfig(**raw.get("acquisition", {})),
            trainer=TrainerConfig(**raw.get("trainer", {})),
            dashboard=DashboardConfig(**raw.get("dashboard", {})),
            data=DataConfig(
                output_dir=data_raw.get("output_dir", "results"),
                simulate=SimulationDataConfig(
                    **simulate_raw,
                    pretrain=PretrainDataConfig(**pretrain_raw),
                ),
                plan=PlanDataConfig(**data_raw.get("plan", {})),
            ),
            active_learning_loop=ActiveLearningLoopConfig(**raw.get("active_learning_loop", {})),
            auxiliary_model=(
                AuxiliaryModelConfig(**auxiliary_model_raw)
                if auxiliary_model_raw is not None
                else None
            ),
            auxiliary_trainer=TrainerConfig(**raw.get("auxiliary_trainer", {})),
            seed=raw.get("seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path for the serialized YAML.
        """

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}  # type: ignore[arg-type]
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)
