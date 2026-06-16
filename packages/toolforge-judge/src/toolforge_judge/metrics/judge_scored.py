"""Families 2 & 5 — invocation relevance and output quality (Judge-scored).

These cannot be computed from raw telemetry: they need the static Judge's
per-node verdict (was this the right tool? were the params faithful? did the
node's contract pass?). They are full catalogue entries that emit
``status=requires_judge`` with the threshold attached, so the engine reports the
complete metric set today and the values populate once the Judge writes its
per-node scores back into the store.

When the Judge layer lands it will supply a ``JudgeScores`` lookup and these
metrics will read it the same way the reliability metrics read ``retries``.
"""
from __future__ import annotations

from .base import REQUIRES_JUDGE, Metric, MetricValue
from .data import MetricWindow


class _JudgeScoredToolMetric(Metric):
    """Base for per-tool metrics whose value comes from a Judge verdict."""

    family = ""
    scope = "tool"
    requires_judge = True
    threshold_attr: str = ""

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        threshold = getattr(window.env, self.threshold_attr, None)
        return [
            self._value(window, None, 0, tool_id=tool_id, status=REQUIRES_JUDGE,
                        threshold=threshold)
            for tool_id in window.tool_ids
        ]


# -- family 2: invocation relevance ----------------------------------------


class SelectionPrecision(_JudgeScoredToolMetric):
    """% of this tool's calls the Judge deems an appropriate choice for the node."""

    name = "selection_precision"
    family = "invocation"
    threshold_attr = "selection_precision_min"


class SelectionRecall(_JudgeScoredToolMetric):
    """% of situations where the tool should have been called but wasn't.

    Low recall → change the tool *description*, not its implementation.
    """

    name = "selection_recall"
    family = "invocation"
    threshold_attr = "selection_recall_min"


class ParamExtractionAccuracy(_JudgeScoredToolMetric):
    """% of calls whose params faithfully reflect the timeline's info_units."""

    name = "param_extraction_accuracy"
    family = "invocation"
    threshold_attr = "param_extraction_min"


# -- family 5: output quality ----------------------------------------------


class NodeAssertionPassRate(_JudgeScoredToolMetric):
    """% of Contract assertions passing for this tool's node."""

    name = "node_assertion_pass_rate"
    family = "output_quality"
    threshold_attr = "node_assertion_pass_min"


class DownstreamCorrectionRate(_JudgeScoredToolMetric):
    """% of cases where a downstream LLM corrects/ignores this tool's output."""

    name = "downstream_correction_rate"
    family = "output_quality"
    threshold_attr = "downstream_correction_rate_abs"


class OutputSchemaDrift(_JudgeScoredToolMetric):
    """% of outputs not conforming to the tool's declared output schema."""

    name = "output_schema_drift"
    family = "output_quality"
    threshold_attr = "output_schema_drift_abs"


INVOCATION_METRICS: list[Metric] = [
    SelectionPrecision(),
    SelectionRecall(),
    ParamExtractionAccuracy(),
]

OUTPUT_QUALITY_METRICS: list[Metric] = [
    NodeAssertionPassRate(),
    DownstreamCorrectionRate(),
    OutputSchemaDrift(),
]
