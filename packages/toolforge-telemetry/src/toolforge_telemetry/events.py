"""Typed telemetry event — serialises to the JSONL wire format."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TelemetryEvent:
    kind: str          # "creation" | "execution"
    event: str         # propose_tool | update_tool | sandbox_validated | promote |
                       # deprecate | call_tool | call_tool_error |
                       # task_start | task_complete | task_error |
                       # llm_call_complete
    tool: str
    ts: str = field(default_factory=_now)
    version: int | None = None
    duration_ms: float | None = None
    error: str | None = None
    # Consumer-side observability fields
    task_id: str | None = None          # correlates all events within one consumer task
    input_preview: str | None = None    # first 300 chars of tool input (JSON)
    output_preview: str | None = None   # first 300 chars of tool output
    error_kind: str | None = None       # "timeout" | "validation" | "runtime" | "api"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "event": self.event,
            "tool": self.tool,
            "ts": self.ts,
        }
        if self.version is not None:
            d["version"] = self.version
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.error is not None:
            d["error"] = self.error
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.input_preview is not None:
            d["input_preview"] = self.input_preview
        if self.output_preview is not None:
            d["output_preview"] = self.output_preview
        if self.error_kind is not None:
            d["error_kind"] = self.error_kind
        return d
