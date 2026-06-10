"""Core types: enums, dataclasses, and result containers for the moal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class QueryType(StrEnum):
    """Fidelity level of an oracle query.

    Attributes
    ----------
    PRIMARY_SCREEN : str
        Low-fidelity primary screen (PS) assay, yielding a censored
        inequality label (``<`` miss or ``>=`` hit).
    DOSE_RESPONSE : str
        High-fidelity dose-response curve (DRC) assay, yielding an exact
        pEC50 measurement.
    """

    PRIMARY_SCREEN = "PS"
    DOSE_RESPONSE = "DRC"


class CensoringType(StrEnum):
    """How a label is censored relative to the true continuous pEC50.

    Attributes
    ----------
    EXACT : str
        The observed value equals the true pEC50 (up to assay noise).
        Produced by DRC assays.
    LEFT : str
        The true pEC50 lies below the censoring threshold T; label is
        ``"< T"``.  Produced by PS misses.
    INTERVAL : str
        The true pEC50 lies in ``[T, upper_bound]``; label is ``">= T"``.
        Produced by PS hits.  Encoded as interval-censored rather than
        right-censored at T to avoid inverting gradients for active
        compounds in the Tobit loss.
    """

    EXACT = "exact"
    LEFT = "left"
    INTERVAL = "interval"


@dataclass(frozen=True)
class LabelRecord:
    """A single labeled observation from the oracle.

    Attributes
    ----------
    smiles : str
        Original (possibly non-canonical) SMILES string as supplied to the
        oracle.
    canonical_smiles : str
        RDKit-canonical, salt-stripped SMILES used as the ground-truth lookup
        key.
    value : float
        For EXACT labels, the true pEC50. For LEFT labels, the censoring
        threshold T (true pEC50 < T). For INTERVAL labels, the lower bound T
        (true pEC50 in [T, upper_bound]).
    upper_bound : float
        For EXACT labels, equals ``value``. For INTERVAL labels, the practical
        pEC50 ceiling (e.g., 11.0). Unused for LEFT labels.
    censoring_type : CensoringType
        How the label is censored relative to the true continuous pEC50.
    fidelity : QueryType
        Whether this record came from a Primary Screen or Dose-Response Curve
        assay.
    cost : float
        Dollar cost of the oracle query that produced this record.
    iteration : int
        Zero-based active learning iteration index at which this record was
        acquired.
    weight : float
        Per-sample loss weight used by :class:`~moal.loss.CensoredRegressionLoss`.
        Normalized to mean=1.0 within each fidelity class by
        :func:`~moal.planning.normalize_record_weights` before training.
        Defaults to 1.0 (uniform weighting).
    """

    smiles: str
    canonical_smiles: str
    value: float
    upper_bound: float
    censoring_type: CensoringType
    fidelity: QueryType
    cost: float
    iteration: int
    weight: float = 1.0
    """Per-sample loss weight. Normalized to mean=1.0 per fidelity class by
    :func:`~moal.planning.normalize_record_weights` before training so the
    global ``w_drc`` / ``w_ps`` scale relationship is preserved. Default 1.0
    is a no-op and preserves backward compatibility."""


@dataclass
class IterationResults:
    """Results produced by a single active learning iteration.

    Attributes
    ----------
    iteration : int
        Zero-based index of this iteration within the campaign.
    queries : list[tuple[str, QueryType]]
        Ordered list of ``(smiles, query_type)`` pairs selected by the
        acquisition function in this iteration.
    new_records : list[LabelRecord]
        Label records returned by the oracle for ``queries``.
    metrics : dict[str, float]
        Per-iteration evaluation metrics (e.g., ``"n_actives_found"``).
    cumulative_cost : float
        Total oracle spend (in cost units) up to and including this
        iteration.
    cumulative_labeled : int
        Total number of labeled compounds up to and including this
        iteration.
    model_metric_value : float or None
        Value of the configured model evaluation metric on the held-out
        test set, or ``None`` if no test set was provided.
    """

    iteration: int
    queries: list[tuple[str, QueryType]]
    new_records: list[LabelRecord]
    metrics: dict[str, float]
    cumulative_cost: float
    cumulative_labeled: int
    model_metric_value: float | None = None


@dataclass
class LoopResults:
    """Aggregate results from a complete active learning campaign.

    Attributes
    ----------
    iterations : list[IterationResults]
        Ordered results from each completed AL iteration.
    final_metrics : dict[str, float]
        Campaign-level summary metrics computed after the final iteration.
    total_cost : float
        Total oracle spend across all iterations.
    total_labeled : int
        Total number of unique compounds labeled across all iterations.
    """

    iterations: list[IterationResults] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_labeled: int = 0

    def costs(self) -> list[float]:
        """Return cumulative cost at the end of each iteration.

        Returns
        -------
        list[float]
            Cumulative cost values, one per completed iteration.
        """
        return [r.cumulative_cost for r in self.iterations]

    def metric_history(self, key: str) -> list[float]:
        """Return the value of a named metric across all iterations.

        Parameters
        ----------
        key : str
            Metric name (as stored in ``IterationResults.metrics``).

        Returns
        -------
        list[float]
            Metric values, one per completed iteration. Missing values are
            represented as ``float("nan")``.
        """
        return [r.metrics.get(key, float("nan")) for r in self.iterations]
