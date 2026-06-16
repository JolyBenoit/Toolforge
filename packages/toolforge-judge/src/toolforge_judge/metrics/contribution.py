"""Family 3 — contribution to the result (from the backward pass).

``dead_call_rate`` reads the ``span.contribution`` column the Judge fills during
its backward pass; until then it reports ``status=requires_judge``.
``redundancy_rate`` is computed deterministically from telemetry (a later call
with the same tool_id + identical input in the same task) and *augmented* by the
Judge's ``redundant`` verdict when present. ``criticality_score`` is structural,
over successful tasks, and needs no Judge.
"""
from __future__ import annotations

import json

from .base import (
    REQUIRES_JUDGE,
    Metric,
    MetricValue,
    safe_ratio,
)
from .dag import critical_path_nodes, span_to_tool
from .data import MetricWindow, SpanRecord

FAMILY = "contribution"


def _input_key(span: SpanRecord) -> str:
    return json.dumps(span.input or {}, sort_keys=True, ensure_ascii=False, default=str)


def _all_tool_calls(tasks: list, tool_id: str) -> list[SpanRecord]:
    calls: list[SpanRecord] = []
    for task in tasks:
        calls.extend(task.tool_calls_for(tool_id))
    return calls


class DeadCallRate(Metric):
    """% of calls marked ``contribution='dead'`` (output never reached result)."""

    name = "dead_call_rate"
    family = FAMILY
    scope = "tool"
    requires_judge = True

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            calls = _all_tool_calls(window.short, tool_id)
            scored = [c for c in calls if c.contribution is not None]
            if not scored:
                out.append(self._value(window, None, 0, tool_id=tool_id,
                                       status=REQUIRES_JUDGE))
                continue
            dead = sum(1 for c in scored if c.contribution == "dead")
            value = safe_ratio(dead, len(scored))
            breached = value is not None and value > window.env.dead_call_rate_abs
            out.append(self._value(window, value, len(scored), tool_id=tool_id,
                                   breached=breached, dead=dead,
                                   threshold=window.env.dead_call_rate_abs))
        return out


class RedundancyRate(Metric):
    """% of calls whose output was already available.

    Deterministic signal: a call whose (tool_id, input) already appeared earlier
    in the same task. Augmented by the Judge's ``contribution='redundant'`` when
    that column is populated.
    """

    name = "redundancy_rate"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            total = 0
            redundant = 0
            judge_redundant = 0
            for task in window.short:
                seen: set[str] = set()
                for call in task.tool_calls_for(tool_id):
                    total += 1
                    key = _input_key(call)
                    if key in seen:
                        redundant += 1
                    else:
                        seen.add(key)
                    if call.contribution == "redundant":
                        judge_redundant += 1
            # Union of the two signals, bounded by total.
            effective = min(total, redundant + judge_redundant) if total else 0
            value = safe_ratio(effective, total)
            breached = value is not None and value > window.env.redundancy_rate_abs
            out.append(self._value(window, value, total, tool_id=tool_id,
                                   breached=breached, duplicate_inputs=redundant,
                                   judge_redundant=judge_redundant,
                                   threshold=window.env.redundancy_rate_abs))
        return out


class CriticalityScore(Metric):
    """How often the tool sits on the DAG critical path of successful tasks.

    Long-window weight metric. A tool near zero over the long window is a
    candidate for removal (pipeline simplification). Breach = below the floor.
    """

    name = "criticality_score"
    family = FAMILY
    scope = "tool"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        successful = [t for t in window.long if t.succeeded]
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            present = 0
            on_critical = 0
            for task in successful:
                calls = task.tool_calls_for(tool_id)
                if not calls:
                    continue
                present += 1
                crit = critical_path_nodes(task)
                s2t = span_to_tool(task)
                if any(sid in crit and s2t.get(sid) == tool_id for sid in crit):
                    on_critical += 1
            value = safe_ratio(on_critical, present)
            breached = value is not None and value < window.env.criticality_floor
            out.append(self._value(window, value, present, win="long",
                                   tool_id=tool_id, breached=breached,
                                   on_critical=on_critical,
                                   floor=window.env.criticality_floor))
        return out


CONTRIBUTION_METRICS: list[Metric] = [
    DeadCallRate(),
    RedundancyRate(),
    CriticalityScore(),
]
