"""Static "Perf"-style timeline reconstruction from stored telemetry.

The consumer's live Perf view (``consumer._stream_task``) renders the run as it
streams, timing each step with ``time.monotonic``. That renderer can't be reused
for *past* runs — but every datum it shows is persisted per span
(``llm_turn_index`` / ``call_index_in_turn`` for order, ``duration_ms`` for
timings, ``tokens_in``/``tokens_out`` for usage). This module replays those spans
into the same visual language (``→``/``←``/``⚙`` arrows, yellow durations, token
counts) so the Judge's Runs tab shows a faithful, read-only timeline.

It works on :class:`~toolforge_judge.metrics.data.SpanRecord` so it stays a pure
formatting function, free of any DB or Textual dependency.
"""
from __future__ import annotations

from collections.abc import Iterable

from rich.markup import escape
from toolforge_judge.metrics.data import SpanRecord


def _dur(ms: float | None) -> str:
    return f"  [yellow]{ms / 1000:.3f}s[/]" if ms else ""


def _ordered(spans: Iterable[SpanRecord]) -> list[SpanRecord]:
    """Replay order: by LLM turn, then call index in turn, then wall clock."""
    return sorted(
        spans,
        key=lambda s: (
            s.llm_turn_index if s.llm_turn_index is not None else 0,
            s.call_index_in_turn if s.call_index_in_turn is not None else 0,
            s.started_at,
        ),
    )


def reconstruct_timeline(spans: Iterable[SpanRecord]) -> list[str]:
    """Return Rich-markup lines mirroring the live Perf view, for a RichLog."""
    lines: list[str] = []
    for sp in _ordered(spans):
        if sp.type == "llm_call":
            turn = sp.llm_turn_index if sp.llm_turn_index is not None else "?"
            lines.append(
                f"[bold cyan]→[/]  LLM   [dim](turn {turn})[/]{_dur(sp.duration_ms)}"
            )
            if sp.tokens_in or sp.tokens_out:
                lines.append(
                    f"       [dim]↳ {sp.tokens_in or 0:,} prompt / "
                    f"{sp.tokens_out or 0:,} completion tokens[/]"
                )
        elif sp.type == "tool_call":
            is_error = (sp.status or "").lower() in ("error", "failed")
            glyph = "[red]✗[/]" if is_error else "[bold yellow]⚙[/]"
            tool = escape(sp.tool_id or "?")
            lines.append(f"{glyph} {tool}{_dur(sp.duration_ms)}")
            if sp.nested_llm_calls:
                lines.append(
                    f"       [dim]↳ {len(sp.nested_llm_calls)} nested LLM call(s)[/]"
                )
            if sp.retried:
                lines.append(f"       [dim]↳ {len(sp.retries)} retry/-ies[/]")
            preview = _output_preview(sp.output)
            if preview:
                color = "red" if is_error else "dim"
                lines.append(f"       [{color}]↳ out {preview}[/]")
        elif sp.type == "user_wait":
            lines.append(f"[yellow]⏸[/]  user turn {sp.user_turn or '?'}")
            if sp.user_message:
                lines.append(f"       [dim]↳ {escape(sp.user_message[:120])}[/]")
    return lines


def _output_preview(output: object, max_len: int = 120) -> str:
    if output is None:
        return ""
    text = output if isinstance(output, str) else str(output)
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return escape(text)
