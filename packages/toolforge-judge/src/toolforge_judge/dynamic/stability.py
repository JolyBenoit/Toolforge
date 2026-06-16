"""Mean structural stability — the dynamic judge's headline dimension.

Derived from the family-4 / family-6 metrics already computed by the engine
(``position_variance`` + its output divergence, ``premature_call_rate``,
``order_sensitivity``) plus ``dag_diversity``. Each tool gets a stability score
in ``[0, 1]``; the pipeline score is their mean over the window.

Per the spec, high position variance is only bad when outputs *diverge* — so the
variance penalty is gated on output divergence, not applied blindly.
"""
from __future__ import annotations

from ..metrics.engine import MetricReport
from ..metrics.env import MetricEnv
from .models import StructuralStability, ToolStability


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tool_stability(
    report: MetricReport, tool_id: str, env: MetricEnv
) -> ToolStability:
    pv = report.get("position_variance", tool_id)
    prem = report.get("premature_call_rate", tool_id)
    order = report.get("order_sensitivity", tool_id)

    position_variance = pv.value if pv else None
    divergence = pv.detail.get("output_divergence") if pv else None
    premature = prem.value if prem and prem.value is not None else None
    # order_sensitivity is not_computable today (needs replay) -> treat as 0.
    order_sensitivity = order.value if order and order.value is not None else None

    penalty = premature or 0.0
    # Variance penalty gated on output divergence and a "high variance" position.
    if (
        position_variance is not None
        and position_variance >= env.position_variance_high
        and divergence is not None
    ):
        penalty += divergence
    penalty += order_sensitivity or 0.0

    return ToolStability(
        tool_id=tool_id,
        position_variance=position_variance,
        output_divergence=divergence,
        premature_call_rate=premature,
        order_sensitivity=order_sensitivity,
        stability_score=round(_clamp01(1.0 - penalty), 4),
    )


def compute_structural_stability(
    report: MetricReport, env: MetricEnv
) -> StructuralStability:
    """Roll the per-tool stability into the pipeline mean structural stability."""
    tool_ids = sorted({v.tool_id for v in report.values if v.tool_id is not None})
    per_tool = [_tool_stability(report, t, env) for t in tool_ids]

    scores = [t.stability_score for t in per_tool]
    mean_score = round(sum(scores) / len(scores), 4) if scores else None

    diversity = report.get("dag_diversity", None)
    dag_diversity = diversity.value if diversity else None

    breached = mean_score is not None and mean_score < env.structural_stability_min
    return StructuralStability(
        per_tool=per_tool,
        mean_structural_stability=mean_score,
        dag_diversity=dag_diversity,
        breached=breached,
    )
