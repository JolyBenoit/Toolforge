"""Families 2 & 5 — invocation relevance and output quality (Judge-scored).

These cannot be computed from raw telemetry: they need the static Judge's
per-node verdict (was this the right tool? were the params faithful? did the
node's contract pass?). Each is a full catalogue entry that reports one of three
states per tool, so the engine always lists the complete metric set:

* ``ok`` + value — the Judge produced a score for this metric (read from
  ``window.judge_scores``, the means of the static judge's per-task notes);
* ``judged`` — the Judge assessed the tool but returned no usable number for
  this particular dimension (measured, yet valueless);
* ``requires_judge`` — no Judge pass has covered this tool yet.

``window.judge_scores`` is supplied by the caller that holds the judge notes,
the same way the reliability metrics read ``retries`` off the spans.
"""
from __future__ import annotations

from .base import JUDGED, OK, REQUIRES_JUDGE, Metric, MetricValue
from .data import MetricWindow


class _JudgeScoredToolMetric(Metric):
    """Base for per-tool metrics whose value comes from a Judge verdict."""

    family = ""
    scope = "tool"
    requires_judge = True
    threshold_attr: str = ""
    # Key under which the static judge's note carries this metric's score
    # (empty = the judge does not score this dimension yet → always pending).
    score_key: str = ""
    higher_is_better: bool = True

    def compute(self, window: MetricWindow) -> list[MetricValue]:
        threshold = getattr(window.env, self.threshold_attr, None)
        scores = window.judge_scores or {}
        judged = window.judged_tools or set()
        out: list[MetricValue] = []
        for tool_id in window.tool_ids:
            entry = scores.get((tool_id, self.score_key)) if self.score_key else None
            if entry is not None:
                value, n = entry
                out.append(self._value(
                    window, value, n, tool_id=tool_id, status=OK,
                    breached=self._is_breach(value, threshold, window),
                    threshold=threshold,
                ))
            elif tool_id in judged and self.score_key:
                # The judge ran on this tool but gave no number for this score.
                out.append(self._value(window, None, 0, tool_id=tool_id,
                                       status=JUDGED, threshold=threshold))
            else:
                out.append(self._value(window, None, 0, tool_id=tool_id,
                                       status=REQUIRES_JUDGE, threshold=threshold))
        return out

    def _is_breach(
        self, value: float, threshold: float | None, window: MetricWindow
    ) -> bool:
        if threshold is None:
            return False
        return value < threshold if self.higher_is_better else value > threshold


# -- family 2: invocation relevance ----------------------------------------


class SelectionPrecision(_JudgeScoredToolMetric):
    """% of this tool's calls the Judge deems an appropriate choice for the node."""

    name = "selection_precision"
    family = "invocation"
    threshold_attr = "selection_precision_min"
    score_key = "selection_precision"


class SelectionRecall(_JudgeScoredToolMetric):
    """% of situations where the tool should have been called but wasn't.

    Low recall → change the tool *description*, not its implementation. The
    static judge does not score recall yet (it needs a counterfactual pass), so
    this stays pending until that machinery lands.
    """

    name = "selection_recall"
    family = "invocation"
    threshold_attr = "selection_recall_min"


class ParamExtractionAccuracy(_JudgeScoredToolMetric):
    """% of calls whose params faithfully reflect the timeline's info_units."""

    name = "param_extraction_accuracy"
    family = "invocation"
    threshold_attr = "param_extraction_min"
    score_key = "param_extraction"


# -- family 5: output quality ----------------------------------------------


class NodeAssertionPassRate(_JudgeScoredToolMetric):
    """% of Contract assertions passing for this tool's node.

    Backed by the static judge's ``output_quality`` score (was the tool's output
    correct / useful), the same mapping ``aggregate`` uses for its threshold.
    """

    name = "node_assertion_pass_rate"
    family = "output_quality"
    threshold_attr = "node_assertion_pass_min"
    score_key = "output_quality"


class DownstreamCorrectionRate(_JudgeScoredToolMetric):
    """% of cases where a downstream LLM corrects/ignores this tool's output.

    Backed by the static judge's ``downstream_correction`` rate (lower is
    better): a high value means later steps routinely had to fix this tool's
    output, so the fault is in the *implementation*.
    """

    name = "downstream_correction_rate"
    family = "output_quality"
    threshold_attr = "downstream_correction_rate_abs"
    score_key = "downstream_correction"
    higher_is_better = False


class OutputSchemaDrift(_JudgeScoredToolMetric):
    """% of outputs not conforming to the tool's declared output schema.

    Backed by the static judge's ``output_schema_drift`` rate (lower is better):
    a high value means the handler emits off-contract payloads.
    """

    name = "output_schema_drift"
    family = "output_quality"
    threshold_attr = "output_schema_drift_abs"
    score_key = "output_schema_drift"
    higher_is_better = False


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
