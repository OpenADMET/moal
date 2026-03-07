"""Core types: enums, dataclasses, and result containers for the moal pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Fidelity level of an oracle query."""

    PRIMARY_SCREEN = "PS"
    DOSE_RESPONSE = "DRC"


class CensoringType(str, Enum):
    """How a label is censored relative to the true continuous pEC50."""

    EXACT = "exact"
    LEFT = "left"        # True pEC50 is in (-inf, value); label: "< value"
    INTERVAL = "interval"  # True pEC50 is in [value, upper_bound]; label: ">= value"


@dataclass(frozen=True)
class LabelRecord:
    """A single labeled observation from the oracle.

    For EXACT labels, value = true pEC50 and upper_bound = value.
    For LEFT labels, value = censoring threshold T; true pEC50 < T.
    For INTERVAL labels, value = lower bound T, upper_bound = practical ceiling
    (e.g., 11.0); true pEC50 in [T, upper_bound].
    """

    smiles: str
    canonical_smiles: str
    value: float
    upper_bound: float
    censoring_type: CensoringType
    fidelity: QueryType
    cost: float
    iteration: int


@dataclass
class IterationResults:
    """Results produced by a single active learning iteration."""

    iteration: int
    queries: list[tuple[str, QueryType]]
    new_records: list[LabelRecord]
    metrics: dict[str, float]
    cumulative_cost: float
    cumulative_labeled: int
    model_metric_value: float | None = None
    """Value of the configured model evaluation metric on the held-out test set,
    or None if no test set was provided."""


@dataclass
class LoopResults:
    """Aggregate results from a complete active learning campaign."""

    iterations: list[IterationResults] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_labeled: int = 0

    def costs(self) -> list[float]:
        return [r.cumulative_cost for r in self.iterations]

    def metric_history(self, key: str) -> list[float]:
        return [r.metrics.get(key, float("nan")) for r in self.iterations]
