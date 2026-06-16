"""Tests for TelemetryEvent and TelemetryWriter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolforge_telemetry import TelemetryEvent, TelemetryWriter


# --- TelemetryEvent ---


def test_event_required_fields() -> None:
    ev = TelemetryEvent(kind="creation", event="propose_tool", tool="extract")
    d = ev.to_dict()
    assert d["kind"] == "creation"
    assert d["event"] == "propose_tool"
    assert d["tool"] == "extract"
    assert "ts" in d


def test_event_omits_none_fields() -> None:
    ev = TelemetryEvent(kind="creation", event="propose_tool", tool="t")
    d = ev.to_dict()
    assert "version" not in d
    assert "duration_ms" not in d
    assert "error" not in d


def test_event_includes_optional_fields_when_set() -> None:
    ev = TelemetryEvent(
        kind="execution",
        event="call_tool",
        tool="t",
        version=2,
        duration_ms=42.5,
        error=None,
    )
    d = ev.to_dict()
    assert d["version"] == 2
    assert d["duration_ms"] == 42.5
    assert "error" not in d


def test_event_includes_error_when_set() -> None:
    ev = TelemetryEvent(kind="execution", event="call_tool_error", tool="t", error="boom")
    d = ev.to_dict()
    assert d["error"] == "boom"


def test_event_ts_is_iso_string() -> None:
    ev = TelemetryEvent(kind="creation", event="promote", tool="t")
    ts = ev.to_dict()["ts"]
    assert "T" in ts  # basic ISO-8601 check
    assert "+" in ts or ts.endswith("Z") or "+00:00" in ts


# --- TelemetryWriter ---


def test_writer_creates_file_on_first_append(tmp_path: Path) -> None:
    p = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(p)
    assert not p.exists()
    writer.append({"kind": "creation", "event": "test", "tool": "t"})
    assert p.exists()


def test_writer_appends_valid_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "tel.jsonl"
    writer = TelemetryWriter(p)
    writer.append({"kind": "creation", "event": "propose_tool", "tool": "a"})
    writer.append({"kind": "creation", "event": "promote", "tool": "a"})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "propose_tool"
    assert json.loads(lines[1])["event"] == "promote"


def test_writer_append_event_serialises_correctly(tmp_path: Path) -> None:
    p = tmp_path / "tel.jsonl"
    writer = TelemetryWriter(p)
    ev = TelemetryEvent(kind="execution", event="call_tool", tool="extract", duration_ms=12.3)
    writer.append_event(ev)
    d = json.loads(p.read_text(encoding="utf-8").strip())
    assert d["kind"] == "execution"
    assert d["tool"] == "extract"
    assert d["duration_ms"] == 12.3


def test_writer_appends_not_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "tel.jsonl"
    writer = TelemetryWriter(p)
    for i in range(5):
        writer.append({"n": i})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert json.loads(lines[-1])["n"] == 4


def test_writer_unicode_safe(tmp_path: Path) -> None:
    p = tmp_path / "tel.jsonl"
    writer = TelemetryWriter(p)
    writer.append({"tool": "élève", "event": "test"})
    raw = p.read_text(encoding="utf-8")
    assert "élève" in raw
