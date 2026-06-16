"""Family 4 — structural stability (the DAG dimension).

``position_variance`` and ``premature_call_rate`` are computed from telemetry.
``order_sensitivity`` requires counterfactual replay (re-running the pipeline on
permuted timelines) which lives outside the metric layer, so it is reported as
``not_computable`` here with the machinery hook documented.
"""
from __future__ import annotations

import hashlib
import json

from .base import (
    NOT_COMPUTABLE,
    Metric,
    MetricValue,
    safe_ratio,
    variance,
)
from .dag import depths
from .data import MetricWindow, SpanRecord

FAMILY = "structural"


def _output_hash(span: SpanRecord) -> str:
    blob = json.dumps(span.output, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _input_fingerprint(span: SpanRecord) -> tuple[frozenset[str], int]:
    """(non-null keys, count of non-null values) — used to detect enrichment."""
    data = span.input or {}
    non_null = {k for k, v in data.items() if v is not None}
    return frozenset(non_null), len(non_null)


class PositionVariance(Metric):
    """Variance of the tool's DAG depth across tasks, paired with output spread.

    High variance + equivalent outputs = robustness (good). High variance +
    divergent outputs = fragility to input ordering (bad). ``value`` is the
    position variance; ``detail.output_divergence`` carries the second axis so
    the breach only fires on the bad quadrant.
    """

    name = "position_variance"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            positions: list[float] = []
            out_hashes: list[str] = []
            for task in window.long:
                depth = depths(task)
                for call in task.tool_calls_for(tool_id):
                    if call.span_id in depth:
                        positions.append(float(depth[call.span_id]))
                    out_hashes.append(_output_hash(call))
            var = variance(positions)
            n = len(positions)
            divergence = safe_ratio(len(set(out_hashes)), len(out_hashes)) or 0.0
            high_var = var is not None and var >= window.env.position_variance_high
            # Bad quadrant: position moves around AND outputs disagree.
            breached = high_var and divergence > 0.5
            out.append(self._value(
                window, var, n, win="long", tool_id=tool_id, breached=breached,
                output_divergence=round(divergence, 3),
                threshold=window.env.position_variance_high,
            ))
        return out


class PrematureCallRate(Metric):
    """% of calls made before all needed info had arrived.

    Detected from telemetry: the same tool is called again *later in the same
    task* with an enriched input (a strict superset of non-null fields). That is
    exactly the "user gives info piecemeal" case — a high rate means the pipeline
    doesn't wait / replan.
    """

    name = "premature_call_rate"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            total = 0
            premature = 0
            for task in window.short:
                calls = task.tool_calls_for(tool_id)
                fps = [(_input_fingerprint(c)) for c in calls]
                for i, (keys_i, _) in enumerate(fps):
                    total += 1
                    # Any later call strictly enriches this one's inputs?
                    for keys_j, _ in fps[i + 1:]:
                        if keys_i < keys_j:  # strict subset => later call enriched
                            premature += 1
                            break
            value = safe_ratio(premature, total)
            breached = value is not None and value > window.env.premature_call_rate_abs
            out.append(self._value(window, value, total, tool_id=tool_id,
                                   breached=breached, premature=premature,
                                   threshold=window.env.premature_call_rate_abs))
        return out


class OrderSensitivity(Metric):
    """% of timeline permutations that change the downstream output.

    Requires counterfactual replay (re-execution on permuted input timelines),
    which is driven by the sandbox, not the read-only metric layer. Surfaced
    here as a catalogue entry so the engine lists it; populated once the replay
    harness writes its results back.
    """

    name = "order_sensitivity"
    family = FAMILY
    scope = "tool"
    requires_judge = False

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        return [
            self._value(window, None, 0, tool_id=tool_id, status=NOT_COMPUTABLE,
                        note="needs counterfactual replay harness")
            for tool_id in window.tool_ids
        ]


STRUCTURAL_METRICS: list[Metric] = [
    PositionVariance(),
    PrematureCallRate(),
    OrderSensitivity(),
]
