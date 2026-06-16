"""Metric engine — run the full catalogue over a window and report.

The engine is the single entry point the Judges will call. It owns the metric
catalogue, computes every metric over a :class:`MetricWindow`, and packages the
results into a :class:`MetricReport` that is trivially serialisable (for storing
the notes/percentages in Postgres at the Judge stage) and diffable across
pipeline versions via :func:`version_delta`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .base import OK, Metric, MetricValue
from .contribution import CONTRIBUTION_METRICS
from .data import MetricWindow, TaskRecord, TelemetryReader
from .env import MetricEnv
from .judge_scored import INVOCATION_METRICS, OUTPUT_QUALITY_METRICS
from .meta import META_METRICS
from .reliability import RELIABILITY_METRICS
from .structural import STRUCTURAL_METRICS


def default_catalogue() -> list[Metric]:
    """Every metric across the six families, in spec order."""
    return [
        *RELIABILITY_METRICS,      # 1
        *INVOCATION_METRICS,       # 2  (judge-scored)
        *CONTRIBUTION_METRICS,     # 3
        *STRUCTURAL_METRICS,       # 4
        *OUTPUT_QUALITY_METRICS,   # 5  (judge-scored)
        *META_METRICS,             # 6
    ]


@dataclass
class MetricReport:
    """All metric values for one window, plus light provenance."""

    usecase_id: str
    run_id: str | None
    computed_at: str
    short_window: int
    long_window: int
    values: list[MetricValue] = field(default_factory=list)

    # -- queries -----------------------------------------------------------

    @property
    def breaches(self) -> list[MetricValue]:
        return [v for v in self.values if v.breached]

    def for_tool(self, tool_id: str) -> list[MetricValue]:
        return [v for v in self.values if v.tool_id == tool_id]

    def for_metric(self, name: str) -> list[MetricValue]:
        return [v for v in self.values if v.metric == name]

    def get(self, metric: str, tool_id: str | None = None) -> MetricValue | None:
        for v in self.values:
            if v.metric == metric and v.tool_id == tool_id:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "run_id": self.run_id,
            "computed_at": self.computed_at,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "values": [v.to_dict() for v in self.values],
        }


class MetricEngine:
    def __init__(self, catalogue: list[Metric] | None = None) -> None:
        self._catalogue = catalogue if catalogue is not None else default_catalogue()

    @property
    def catalogue(self) -> list[Metric]:
        return list(self._catalogue)

    def compute(self, window: MetricWindow) -> MetricReport:
        values: list[MetricValue] = []
        for metric in self._catalogue:
            values.extend(metric.compute(window))
        return MetricReport(
            usecase_id=window.usecase_id,
            run_id=window.run_id,
            computed_at=datetime.now(UTC).isoformat(),
            short_window=window.env.short_window,
            long_window=window.env.long_window,
            values=values,
        )

    # -- convenience constructors -----------------------------------------

    def compute_from_tasks(
        self,
        usecase_id: str,
        tasks: Iterable[TaskRecord],
        env: MetricEnv | None = None,
        *,
        run_id: str | None = None,
    ) -> MetricReport:
        env = env or MetricEnv()
        window = MetricWindow.from_tasks(usecase_id, tasks, env, run_id=run_id)
        return self.compute(window)

    def compute_from_postgres(
        self,
        dsn: str,
        usecase_id: str,
        env: MetricEnv | None = None,
        *,
        run_id: str | None = None,
    ) -> MetricReport:
        env = env or MetricEnv()
        reader = TelemetryReader(dsn)
        window = reader.load_window(usecase_id, env, run_id=run_id)
        return self.compute(window)


# ---------------------------------------------------------------------------
# Cross-version comparison (family 6: version_delta)
# ---------------------------------------------------------------------------


@dataclass
class MetricDelta:
    metric: str
    tool_id: str | None
    before: float | None
    after: float | None
    delta: float | None       # after - before
    regressed: bool


def version_delta(
    after: MetricReport,
    before: MetricReport,
    env: MetricEnv | None = None,
    *,
    higher_is_better: set[str] | None = None,
) -> list[MetricDelta]:
    """Per-metric delta between two pipeline versions (N vs N-1).

    ``regressed`` is True when the metric moved in the worse direction by more
    than ``env.version_delta_regression``. By default every metric is treated as
    "lower is better" (error/retry/dead/redundancy/…); pass the names that are
    "higher is better" (e.g. criticality, precision, recall) in
    ``higher_is_better``.
    """
    env = env or MetricEnv()
    higher = higher_is_better or {
        "criticality_score", "selection_precision", "selection_recall",
        "param_extraction_accuracy", "node_assertion_pass_rate", "dag_diversity",
    }
    index: dict[tuple[str, str | None], MetricValue] = {
        (v.metric, v.tool_id): v for v in before.values
    }
    deltas: list[MetricDelta] = []
    for v in after.values:
        if v.status != OK:
            continue
        prev = index.get((v.metric, v.tool_id))
        if prev is None or prev.status != OK or v.value is None or prev.value is None:
            continue
        diff = v.value - prev.value
        worse = -diff if v.metric in higher else diff
        deltas.append(MetricDelta(
            metric=v.metric, tool_id=v.tool_id,
            before=prev.value, after=v.value, delta=diff,
            regressed=worse > env.version_delta_regression,
        ))
    return deltas
