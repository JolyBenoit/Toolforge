"""Family 1 — execution reliability.

Pure-telemetry metrics, computed per tool over the short window (with the long
window as the SPC baseline where relevant). Implemented first because they need
no Judge verdict: every input is already in ``tf_prod_spans``.
"""
from __future__ import annotations

from collections import Counter

from .base import (
    Metric,
    MetricValue,
    percentile,
    safe_ratio,
)
from .data import MetricWindow, SpanRecord
from .taxonomy import TIMEOUT, classify_error

FAMILY = "reliability"


def _tool_calls(tasks: list, tool_id: str) -> list[SpanRecord]:
    calls: list[SpanRecord] = []
    for task in tasks:
        calls.extend(task.tool_calls_for(tool_id))
    return calls


def _error_classes(calls: list[SpanRecord]) -> list[str]:
    """All classified error strings across the failed attempts of these calls."""
    classes: list[str] = []
    for call in calls:
        for retry in call.retries:
            classes.append(classify_error(retry.get("error")))
    return classes


class ErrorRate(Metric):
    """% of calls ending in ``status='error'`` after retries. Absolute threshold."""

    name = "error_rate"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            calls = _tool_calls(window.short, tool_id)
            n = len(calls)
            errors = sum(1 for c in calls if c.status == "error")
            value = safe_ratio(errors, n)
            breached = value is not None and value > window.env.error_rate_abs
            out.append(
                self._value(
                    window, value, n, tool_id=tool_id, breached=breached,
                    errors=errors, threshold=window.env.error_rate_abs,
                )
            )
        return out


class RetryRate(Metric):
    """% of calls needing >= 1 retry. High here = ill-specified input schema."""

    name = "retry_rate"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            calls = _tool_calls(window.short, tool_id)
            n = len(calls)
            retried = sum(1 for c in calls if c.retried)
            value = safe_ratio(retried, n)
            breached = value is not None and value > window.env.retry_rate_abs
            # Evidence for the creator instruction: the retry error messages.
            sample_errors = [
                r.get("error")
                for c in calls
                for r in c.retries
            ][:5]
            out.append(
                self._value(
                    window, value, n, tool_id=tool_id, breached=breached,
                    retried=retried, threshold=window.env.retry_rate_abs,
                    sample_errors=sample_errors,
                )
            )
        return out


class ErrorTaxonomyDistribution(Metric):
    """Distribution of error classes; breach fires on *concentration*.

    The interesting trigger is not the overall rate but whether one class
    dominates a tool's failures — that makes the diagnosis near-automatic.
    ``value`` is the top class's share.
    """

    name = "error_taxonomy_distribution"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            calls = _tool_calls(window.short, tool_id)
            classes = _error_classes(calls)
            n = len(classes)
            if n == 0:
                out.append(
                    self._value(window, None, 0, tool_id=tool_id, status="ok",
                                distribution={})
                )
                continue
            counts = Counter(classes)
            top_class, top_count = counts.most_common(1)[0]
            value = top_count / n
            breached = value >= window.env.taxonomy_concentration
            out.append(
                self._value(
                    window, value, n, tool_id=tool_id, breached=breached,
                    distribution=dict(counts), top_class=top_class,
                    threshold=window.env.taxonomy_concentration,
                )
            )
        return out


class TimeoutRate(Metric):
    """% of calls whose failures are timeouts. Absolute threshold."""

    name = "timeout_rate"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            calls = _tool_calls(window.short, tool_id)
            n = len(calls)
            timeouts = 0
            for c in calls:
                attempt_classes = {classify_error(r.get("error")) for r in c.retries}
                if TIMEOUT in attempt_classes:
                    timeouts += 1
            value = safe_ratio(timeouts, n)
            breached = value is not None and value > window.env.timeout_rate_abs
            out.append(
                self._value(
                    window, value, n, tool_id=tool_id, breached=breached,
                    timeouts=timeouts, threshold=window.env.timeout_rate_abs,
                )
            )
        return out


class LatencyP95(Metric):
    """p95 latency with SPC drift: short p95 vs long p95.

    A rising p95 signals a mis-dimensioned tool or inputs that keep growing.
    ``value`` is the short-window p95; breach fires on the drift factor.
    """

    name = "latency_p95"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            short_durs = [
                c.duration_ms for c in _tool_calls(window.short, tool_id)
                if c.duration_ms is not None
            ]
            long_durs = [
                c.duration_ms for c in _tool_calls(window.long, tool_id)
                if c.duration_ms is not None
            ]
            short_p95 = percentile(short_durs, 95)
            long_p95 = percentile(long_durs, 95)
            breached = (
                short_p95 is not None
                and long_p95 is not None
                and long_p95 > 0
                and short_p95 >= long_p95 * window.env.latency_p95_drift_factor
            )
            out.append(
                self._value(
                    window, short_p95, len(short_durs), tool_id=tool_id,
                    breached=breached, long_p95=long_p95,
                    drift_factor=window.env.latency_p95_drift_factor,
                )
            )
        return out


RELIABILITY_METRICS: list[Metric] = [
    ErrorRate(),
    RetryRate(),
    ErrorTaxonomyDistribution(),
    TimeoutRate(),
    LatencyP95(),
]
