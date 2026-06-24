"""Assemble the judge's user message from telemetry, and parse its JSON reply.

Only *data* is assembled here — the judging instructions and the output
contract live in the system-prompt file (``prompts/judge_static_system.md``),
honouring the project's "no hardcoded prompts" constraint.
"""
from __future__ import annotations

import json
from typing import Any

from ..metrics.data import SpanRecord, TaskRecord
from .models import (
    Contribution,
    SpanVerdict,
    StaticJudgeResult,
    ToolNote,
    UseCaseSpec,
)

_VALID_CONTRIB: set[str] = {"necessary", "redundant", "dead"}
_VALID_TARGET: set[str] = {"none", "implementation", "description", "usage"}
_MAX_FIELD_CHARS = 2000  # trim oversized inputs/outputs to keep the prompt lean


def _trim(value: Any) -> Any:
    """Trim long string leaves so a giant output doesn't blow the context."""
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + f"…[+{len(value) - _MAX_FIELD_CHARS} chars]"
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim(v) for v in value]
    return value


def _span_view(span: SpanRecord) -> dict[str, Any]:
    base: dict[str, Any] = {
        "span_id": span.span_id,
        "type": span.type,
        "status": span.status,
    }
    if span.is_tool_call:
        base.update(
            tool_id=span.tool_id,
            tool_version=span.tool_version,
            input=_trim(span.input),
            output=_trim(span.output),
            retries=[r.get("error") for r in span.retries],
            duration_ms=span.duration_ms,
        )
    elif span.type == "user_wait":
        base.update(user_turn=span.user_turn, user_message=_trim(span.user_message))
    elif span.type == "llm_call":
        # The orchestrator's own turn — keep only its assistant text, trimmed.
        base.update(assistant=_trim(span.response) if span.response else None)
    return base


def _dag_view(task: TaskRecord) -> list[dict[str, str]]:
    dag = task.dag or {}
    return [
        {"from": e["from_span"], "to": e["to_span"], "via": e.get("via", "control")}
        for e in (dag.get("edges") or [])
    ]


def build_user_message(usecase: UseCaseSpec, task: TaskRecord) -> str:
    """Serialise the use case context + one task's telemetry into the prompt."""
    payload = {
        "use_case": {
            "usecase_id": usecase.usecase_id,
            "utility": usecase.utility,
            "rules": usecase.rules,
            "tools": [
                {"tool_id": t.tool_id, "description": t.description, "schema": t.schema}
                for t in usecase.tools
            ],
        },
        "task": {
            "task_id": task.task_id,
            "status": task.status,
            "input_timeline": _trim(task.input_timeline),
            "spans": [_span_view(s) for s in task.spans],
            "dag_edges": _dag_view(task),
            "final_output": _trim(task.final_output),
        },
    }
    return (
        "Evaluate this single execution trace and return the JSON object "
        "described in your instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _preview(text: str, *, limit: int = 300) -> str:
    """A compact, single-line preview of a model reply for error messages."""
    snippet = " ".join(text.split())  # collapse newlines/runs of whitespace
    if len(snippet) > limit:
        snippet = snippet[:limit] + "…"
    return snippet


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of the reply.

    On failure the raised ``ValueError`` carries a short preview of what the
    model actually returned, so an operator can tell an empty completion from a
    prose/refusal reply from malformed JSON without re-running.
    """
    raw = text
    text = text.strip()
    if not text:
        raise ValueError(
            "judge returned an empty response (no text streamed) — check the "
            "judge model, max_tokens, and provider for the [llm.judge] backend"
        )
    if text.startswith("```"):
        # strip a ```json … ``` fence if the model added one
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(
            "no JSON object found in judge response — the model replied with "
            f"prose instead of JSON: {_preview(raw)!r}"
        )
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"judge response held malformed JSON ({exc}): "
                        f"{_preview(blob)!r}"
                    ) from exc
    raise ValueError(
        "unbalanced JSON object in judge response (likely truncated — raise "
        f"max_tokens for [llm.judge]): {_preview(raw)!r}"
    )


def _clamp01(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _contribution(value: Any) -> Contribution:
    return value if value in _VALID_CONTRIB else "necessary"  # type: ignore[return-value]


def parse_result(
    task: TaskRecord, judge_model: str, raw_response: str
) -> StaticJudgeResult:
    """Parse a raw judge reply into a validated :class:`StaticJudgeResult`."""
    data = _extract_json(raw_response)

    verdicts: list[SpanVerdict] = []
    for v in data.get("span_verdicts") or []:
        if not isinstance(v, dict) or "span_id" not in v:
            continue
        verdicts.append(
            SpanVerdict(
                span_id=str(v["span_id"]),
                tool_id=str(v.get("tool_id", "")),
                contribution=_contribution(v.get("contribution")),
                selection_appropriate=bool(v.get("selection_appropriate", True)),
                param_fidelity=_clamp01(v.get("param_fidelity", 1.0)),
                rationale=str(v.get("rationale", "")),
            )
        )

    notes: list[ToolNote] = []
    for n in data.get("tool_notes") or []:
        if not isinstance(n, dict) or "tool_id" not in n:
            continue
        raw_scores = n.get("scores") or {}
        scores = {str(k): _clamp01(val) for k, val in raw_scores.items()}
        rec = n.get("recommendation")
        rec = str(rec) if rec else None
        target = n.get("recommendation_target")
        target = target if target in _VALID_TARGET else ("none" if rec is None else "usage")
        notes.append(
            ToolNote(
                tool_id=str(n["tool_id"]),
                scores=scores,
                recommendation=rec,
                recommendation_target=target,  # type: ignore[arg-type]
            )
        )

    return StaticJudgeResult(
        task_id=task.task_id,
        run_id=task.run_id,
        usecase_id=task.usecase_id,
        judge_model=judge_model,
        span_verdicts=verdicts,
        tool_notes=notes,
        raw_response=raw_response,
    )
