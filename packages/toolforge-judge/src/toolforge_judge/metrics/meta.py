"""Family 6 — loop / meta indicators (guard the improvement loop itself).

``dag_diversity`` is computable from telemetry now. ``version_delta`` is a
cross-version comparison driven by :func:`version_delta` over two metric
reports (see ``engine``). ``judge_agreement`` and ``oscillation_detector`` need
Judge outputs / instruction history and are surfaced as ``requires_judge``.
"""
from __future__ import annotations

from collections import Counter

from .base import (
    NOT_COMPUTABLE,
    REQUIRES_JUDGE,
    Metric,
    MetricValue,
    shannon_entropy,
)
from .dag import canonical_signature
from .data import MetricWindow

FAMILY = "meta"


class DagDiversity(Metric):
    """Entropy of canonical DAG signatures over the window.

    A collapse of short-window entropy versus the long baseline, while use cases
    stay varied, is the "refinement without exploration" failure mode (precision
    up, recall down). ``value`` is the short-window entropy.
    """

    name = "dag_diversity"
    family = FAMILY
    scope = "pipeline"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        short_sigs = [canonical_signature(t) for t in window.short]
        long_sigs = [canonical_signature(t) for t in window.long]
        short_ent = shannon_entropy(list(Counter(short_sigs).values()))
        long_ent = shannon_entropy(list(Counter(long_sigs).values()))
        breached = (
            long_ent > 0
            and short_ent / long_ent < window.env.dag_diversity_collapse_ratio
        )
        return [self._value(
            window, short_ent, len(short_sigs), breached=breached,
            long_entropy=round(long_ent, 4),
            distinct_short=len(set(short_sigs)),
            distinct_long=len(set(long_sigs)),
            collapse_ratio=window.env.dag_diversity_collapse_ratio,
        )]


class VersionDelta(Metric):
    """Per-metric delta between pipeline version N and N-1 on the regression set.

    This is THE self-improvement signal. It is a cross-version comparison, so it
    is produced by :func:`toolforge_judge.metrics.engine.version_delta` over two
    stored :class:`MetricReport`s rather than from a single window. Listed here
    so the catalogue is complete.
    """

    name = "version_delta"
    family = FAMILY
    scope = "pipeline"

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        return [self._value(window, None, 0, status=NOT_COMPUTABLE,
                            note="computed across two MetricReports, see engine.version_delta")]


class JudgeAgreement(Metric):
    """Agreement between the Judge and explicit user corrections.

    Below ``judge_agreement_min`` the Judge must be recalibrated before its
    instructions are allowed to change anything. Needs the Judge's per-task
    verdicts, so reported as ``requires_judge`` until those exist.
    """

    name = "judge_agreement"
    family = FAMILY
    scope = "pipeline"
    requires_judge = True

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        corrections = sum(
            1 for t in window.long
            if (t.user_feedback or {}).get("explicit") == "correction"
        )
        return [self._value(window, None, corrections, status=REQUIRES_JUDGE,
                            user_corrections=corrections,
                            threshold=window.env.judge_agreement_min)]


class OscillationDetector(Metric):
    """Flags a tool whose instructions flip direction across consecutive versions.

    Needs the Judge's instruction history per tool; surfaced as ``requires_judge``
    until that history is persisted. When it fires the tool is frozen and
    escalated to a human.
    """

    name = "oscillation_detector"
    family = FAMILY
    scope = "tool"
    requires_judge = True

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        return [self._value(window, None, 0, tool_id=tool_id, status=REQUIRES_JUDGE,
                            window_versions=window.env.oscillation_versions)
                for tool_id in window.tool_ids]


META_METRICS: list[Metric] = [
    DagDiversity(),
    VersionDelta(),
    JudgeAgreement(),
    OscillationDetector(),
]
