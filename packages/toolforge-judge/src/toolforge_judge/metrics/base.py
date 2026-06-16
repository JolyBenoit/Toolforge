"""Metric result model, base class and small shared helpers.

A metric maps a :class:`MetricWindow` to one or more :class:`MetricValue`. A
*tool*-scoped metric yields one value per ``tool_id``; a *pipeline*-scoped
metric yields a single value for the whole window.

Judge-dependent metrics (families 2 and 5, and parts of 3/6) are real metrics
too: they emit a :class:`MetricValue` with ``value=None`` and
``status="requires_judge"`` until the Judge has filled the backing column
(``span.contribution``, per-node verdicts, …). This lets the engine list the
full metric catalogue today and have the values populate "naturally" once the
Judges land.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .data import MetricWindow
from .env import MetricEnv

# Status constants for a MetricValue.
OK = "ok"                       # computed from telemetry
REQUIRES_JUDGE = "requires_judge"   # needs a Judge verdict first
INSUFFICIENT_DATA = "insufficient_data"  # below env.min_samples
NOT_COMPUTABLE = "not_computable"   # needs replay / counterfactual machinery


@dataclass
class MetricValue:
    metric: str
    family: str
    scope: str               # "tool" | "pipeline"
    window: str              # "short" | "long" | "version_delta"
    value: float | None
    n: int                   # sample size backing the value
    breached: bool = False
    tool_id: str | None = None
    status: str = OK
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Metric:
    """Base class for all metrics.

    Subclasses set the class attributes and implement :meth:`compute`.
    """

    name: str = ""
    family: str = ""
    scope: str = "tool"          # "tool" | "pipeline"
    requires_judge: bool = False

    def compute(self, window: MetricWindow) -> list[MetricValue]:  # pragma: no cover
        raise NotImplementedError

    # -- helpers available to every metric ---------------------------------

    def _value(
        self,
        window: MetricWindow,
        value: float | None,
        n: int,
        *,
        win: str = "short",
        tool_id: str | None = None,
        status: str = OK,
        breached: bool = False,
        **detail: Any,
    ) -> MetricValue:
        # Guard: never let a metric breach on a sample below min_samples.
        if breached and n < window.env.min_samples:
            breached = False
            status = INSUFFICIENT_DATA
        return MetricValue(
            metric=self.name,
            family=self.family,
            scope=self.scope,
            window=win,
            value=value,
            n=n,
            breached=breached,
            tool_id=tool_id,
            status=status,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Shared numeric helpers
# ---------------------------------------------------------------------------


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """Ratio in [0, 1], or None when there is no denominator."""
    if denominator <= 0:
        return None
    return numerator / denominator


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (pct in [0, 100]). None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def variance(values: list[float]) -> float | None:
    """Population variance, or None if fewer than 2 samples."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy (base 2) of a multiset given its bucket counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def divergence_breach(short: float | None, long: float | None, env: MetricEnv) -> bool:
    """True when the short-window value exceeds the long baseline by the factor."""
    if short is None or long is None or long <= 0:
        return False
    return short >= long * env.divergence_factor
