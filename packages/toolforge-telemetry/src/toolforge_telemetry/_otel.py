"""Optional OpenTelemetry span emission — no-op when OTel is not configured."""
from __future__ import annotations

from typing import Any

try:
    from opentelemetry import trace as _ot_trace

    _tracer = _ot_trace.get_tracer("toolforge.telemetry")

    def emit_span(event_dict: dict[str, Any]) -> None:
        with _tracer.start_as_current_span(event_dict.get("event", "unknown")) as span:
            for k, v in event_dict.items():
                if v is not None:
                    span.set_attribute(f"toolforge.{k}", str(v))

except ImportError:  # pragma: no cover
    def emit_span(event_dict: dict[str, Any]) -> None:  # type: ignore[misc]
        pass
