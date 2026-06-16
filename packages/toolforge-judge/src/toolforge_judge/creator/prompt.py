"""Assemble the creator judge's two user messages, and parse its JSON replies.

Only *data* is assembled here; the instructions and the output contracts live in
the system-prompt files (``prompts/judge_creator_axes_system.md`` and
``prompts/judge_creator_system.md``), per the "no hardcoded prompts" rule.

The robust JSON extraction is shared with the static judge.
"""
from __future__ import annotations

import json
from typing import Any, get_args

from ..dynamic.models import DynamicJudgeReport, ToolGlobalNote
from ..metrics.engine import MetricReport
from ..static.models import UseCaseSpec
from ..static.prompt import _extract_json
from .models import (
    CreatorAction,
    CreatorInstruction,
    Priority,
    ToolImprovementAxes,
)

_VALID_ACTION: set[str] = set(get_args(CreatorAction))
_VALID_PRIORITY: set[str] = set(get_args(Priority))


# ---------------------------------------------------------------------------
# Stage 1 — per-tool improvement axes
# ---------------------------------------------------------------------------


def tool_breach_views(report: MetricReport, tool_id: str) -> list[dict[str, Any]]:
    """The breaching, tool-scoped metric values for one tool, prompt-ready."""
    views: list[dict[str, Any]] = []
    for v in report.for_tool(tool_id):
        if not v.breached:
            continue
        views.append(
            {
                "metric": v.metric,
                "family": v.family,
                "window": v.window,
                "value": v.value,
                "n": v.n,
                "detail": v.detail,
            }
        )
    return views


def build_axes_message(
    tool_id: str,
    breaches: list[dict[str, Any]],
    global_note: ToolGlobalNote | None,
) -> str:
    """Serialise one tool's breaches + run/run comments for the stage-1 pass."""
    payload = {
        "tool_id": tool_id,
        "breaching_metrics": breaches,
        "run_over_run_comments": global_note.to_dict() if global_note else None,
    }
    return (
        "Synthesise the improvement axes for this single tool from its breaching "
        "metrics and its run-over-run comments. Return the JSON object described "
        "in your instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def axes_evidence(
    breaches: list[dict[str, Any]], global_note: ToolGlobalNote | None
) -> dict[str, Any]:
    """The raw material a stage-1 synthesis was grounded on (kept for audit)."""
    return {
        "breaching_metrics": breaches,
        "run_over_run_comments": global_note.to_dict() if global_note else None,
    }


def parse_axes(tool_id: str, evidence: dict[str, Any], raw: str) -> ToolImprovementAxes:
    """Parse a stage-1 reply into :class:`ToolImprovementAxes`."""
    data = _extract_json(raw)
    raw_axes = data.get("axes") or []
    axes = [str(a) for a in raw_axes if str(a).strip()] if isinstance(raw_axes, list) else []
    return ToolImprovementAxes(
        tool_id=tool_id,
        summary=str(data.get("summary", "")).strip(),
        axes=axes,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Stage 2 — pipeline-aware corrective instructions
# ---------------------------------------------------------------------------


def _usecase_view(usecase: UseCaseSpec) -> dict[str, Any]:
    return {
        "usecase_id": usecase.usecase_id,
        "utility": usecase.utility,
        "rules": usecase.rules,
        "tools": [
            {"tool_id": t.tool_id, "description": t.description, "schema": t.schema}
            for t in usecase.tools
        ],
    }


def _dynamic_view(report: DynamicJudgeReport) -> dict[str, Any]:
    return {
        "structural_stability": report.structural_stability.to_dict(),
        "metric_breaches": report.breaching_metrics,
        "diagnosis": report.diagnosis,
    }


def build_instructions_message(
    tool_axes: list[ToolImprovementAxes],
    dynamic_report: DynamicJudgeReport,
    usecase: UseCaseSpec,
) -> str:
    """Serialise the stage-1 axes + dynamic picture + the *full* pipeline.

    The full tool catalogue (not just the problematic tools) is included on
    purpose: a corrective instruction must account for the whole use case so it
    neither breaks the pipeline nor introduces redundancy.
    """
    payload = {
        "tool_improvement_axes": [a.to_dict() for a in tool_axes],
        "dynamic_report": _dynamic_view(dynamic_report),
        "pipeline": _usecase_view(usecase),
    }
    return (
        "Here are the per-tool improvement axes, the cross-run dynamic picture, "
        "and the full pipeline description. Propose the corrective instructions "
        "for the Creator described in your instructions. Account for the whole "
        "use case: do not break it, do not introduce redundancy. Return the JSON "
        "object described in your instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def _priority(value: Any) -> Priority:
    return value if value in _VALID_PRIORITY else "medium"  # type: ignore[return-value]


def parse_instructions(raw: str) -> list[CreatorInstruction]:
    """Parse a stage-2 reply into a list of :class:`CreatorInstruction`.

    Entries with an unknown ``action`` or no target are dropped — the Creator
    only ever sees instructions it can actually act on.
    """
    data = _extract_json(raw)
    out: list[CreatorInstruction] = []
    for item in data.get("instructions") or []:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if action not in _VALID_ACTION:
            continue
        body = str(item.get("body", "")).strip()
        if not body:
            continue
        raw_targets = item.get("target_tools") or []
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        targets = [str(t) for t in raw_targets if str(t).strip()]
        if action != "create_tool" and not targets:
            # Every action but a fresh create must name the tool(s) it touches.
            continue
        out.append(
            CreatorInstruction(
                action=action,  # type: ignore[arg-type]
                target_tools=targets,
                body=body,
                rationale=str(item.get("rationale", "")).strip(),
                priority=_priority(item.get("priority")),
                expected_effect=str(item.get("expected_effect", "")).strip(),
            )
        )
    return out
