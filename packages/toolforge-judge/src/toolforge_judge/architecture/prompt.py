"""Assemble the architecture judge's two messages, and parse the JSON replies.

Only *data* is assembled here — the judging instructions live in the system
prompts (``prompts/judge_architecture_tool_system.md`` for pass 1,
``prompts/judge_architecture_system.md`` for pass 2), per the project's
"no hardcoded prompts" rule. The robust JSON extraction and the field trimmer
are shared with the static judge.
"""
from __future__ import annotations

import json
from typing import Any

from ..static.prompt import _extract_json, _trim
from .models import (
    ArchitectureFinding,
    ArchitectureSpec,
    RichToolSpec,
    ToolContract,
)

_VALID_CATEGORY: set[str] = {
    "over_simplification",
    "technical_constraint",
    "overkill",
    "redundant_step",
    "coverage_gap",
    "wiring",
    "ordering",
}
_VALID_SEVERITY: set[str] = {"info", "warning", "error"}
_VALID_ACTION: set[str] = {
    "modify_implementation",
    "modify_description",
    "modify_usage",
    "create_tool",
    "remove_tool",
    "merge_tools",
    "split_tool",
    "none",
}

# Handler sources can be long; keep the pass-1 prompt lean.
_MAX_SOURCE_CHARS = 6000


# ---------------------------------------------------------------------------
# Pass 1 — per-tool contract read
# ---------------------------------------------------------------------------


def build_contract_message(spec: ArchitectureSpec, tool: RichToolSpec) -> str:
    """Serialise the use-case context + one tool's source into the prompt."""
    source = tool.source or ""
    if len(source) > _MAX_SOURCE_CHARS:
        source = source[:_MAX_SOURCE_CHARS] + f"\n…[+{len(source) - _MAX_SOURCE_CHARS} chars]"
    payload = {
        "use_case": {"utility": spec.utility, "rules": spec.rules},
        "tool": {
            "tool_id": tool.tool_id,
            "description": tool.description,
            "input_schema": _trim(tool.input_schema),
            "requirements": tool.requirements,
            "handler_source": source,
        },
    }
    return (
        "Read this tool's handler and return the JSON contract described in your "
        "instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def parse_contract(tool_id: str, raw_response: str) -> ToolContract:
    """Parse a pass-1 reply into a :class:`ToolContract`."""
    try:
        data = _extract_json(raw_response)
    except ValueError as exc:
        raise ValueError(f"pass 1 (contract for tool {tool_id!r}): {exc}") from exc
    return ToolContract(
        tool_id=tool_id,
        output_contract=str(data.get("output_contract", "")),
        limits=_str_list(data.get("limits")),
        local_risks=_str_list(data.get("local_risks")),
        raw_response=raw_response,
    )


# ---------------------------------------------------------------------------
# Pass 2 — pipeline coherence
# ---------------------------------------------------------------------------


def build_findings_message(
    spec: ArchitectureSpec,
    contracts: list[ToolContract],
    telemetry_digest: dict[str, Any] | None = None,
) -> str:
    """Serialise the use case + every tool's compact contract (+ optional
    telemetry digest) into the pass-2 prompt. The handler source is *not*
    repeated here — only the derived contracts — to keep the prompt lean."""
    by_id = {c.tool_id: c for c in contracts}
    tools = []
    for t in spec.tools:
        c = by_id.get(t.tool_id)
        tools.append({
            "tool_id": t.tool_id,
            "description": t.description,
            "input_schema": _trim(t.input_schema),
            "output_contract": c.output_contract if c else "",
            "limits": c.limits if c else [],
            "local_risks": c.local_risks if c else [],
        })
    payload: dict[str, Any] = {
        "use_case": {
            "usecase_id": spec.usecase_id,
            "utility": spec.utility,
            "rules": spec.rules,
        },
        "tools": tools,
    }
    if telemetry_digest:
        payload["telemetry_digest"] = _trim(telemetry_digest)
    return (
        "Assess this pipeline as a designed system and return the JSON object of "
        "findings described in your instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def parse_findings(raw_response: str) -> list[ArchitectureFinding]:
    """Parse a pass-2 reply into validated :class:`ArchitectureFinding`s."""
    try:
        data = _extract_json(raw_response)
    except ValueError as exc:
        raise ValueError(f"pass 2 (pipeline findings): {exc}") from exc
    findings: list[ArchitectureFinding] = []
    for f in data.get("findings") or []:
        if not isinstance(f, dict) or "category" not in f or "body" not in f:
            continue
        category = f.get("category")
        if category not in _VALID_CATEGORY:
            continue
        severity = f.get("severity")
        severity = severity if severity in _VALID_SEVERITY else "warning"
        action = f.get("proposed_action")
        action = action if action in _VALID_ACTION else "none"
        findings.append(
            ArchitectureFinding(
                category=category,  # type: ignore[arg-type]
                severity=severity,  # type: ignore[arg-type]
                tools_involved=_str_list(f.get("tools_involved")),
                requirement_threatened=str(f.get("requirement_threatened", "")),
                body=str(f["body"]),
                evidence=str(f.get("evidence", "")),
                proposed_action=action,  # type: ignore[arg-type]
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Telemetry digest (post-run mode) — built duck-typed from a dynamic report
# ---------------------------------------------------------------------------


def digest_from_dynamic_report(report: Any) -> dict[str, Any]:
    """Compact a ``DynamicJudgeReport`` into the fields pass 2 reasons over.

    Duck-typed so the architecture package never hard-depends on the dynamic
    judge; pass-2 only needs the breaches, per-tool notes and the diagnosis.
    """
    if report is None:
        return {}
    ss = getattr(report, "structural_stability", None)
    metric_report = getattr(report, "metric_report", None)
    breaches = getattr(metric_report, "breaches", []) if metric_report else []
    notes = getattr(report, "tool_global_notes", []) or []
    return {
        "mean_structural_stability": getattr(ss, "mean_structural_stability", None),
        "metric_breaches": sorted({getattr(b, "metric", "") for b in breaches} - {""}),
        "tool_notes": [
            {
                "tool_id": n.tool_id,
                "mean_scores": n.mean_scores,
                "recommendation_rate": n.recommendation_rate,
                "breaches": n.breaches,
            }
            for n in notes
        ],
        "diagnosis": getattr(report, "diagnosis", None),
    }


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None and str(v).strip()]
