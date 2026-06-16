"""Append-only JSONL writer with optional OpenTelemetry span emission."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .events import TelemetryEvent


class TelemetryWriter:
    """Thread-safe, append-only writer for telemetry.jsonl."""

    def __init__(self, path: Path, *, otel_enabled: bool = False) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._otel = otel_enabled

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if self._otel:
            from ._otel import emit_span
            emit_span(event)

    def append_event(self, event: TelemetryEvent) -> None:
        self.append(event.to_dict())
