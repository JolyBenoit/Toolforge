"""Tests for TelemetryStore implementations that don't require PostgreSQL."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolforge_telemetry import (
    JSONLTelemetryStore,
    NullTelemetryStore,
    TelemetryConfigError,
    get_store,
)


# ---------------------------------------------------------------------------
# NullTelemetryStore
# ---------------------------------------------------------------------------


def test_null_store_is_silent(tmp_path: Path) -> None:
    store = NullTelemetryStore()
    store.record_task_start("t1", "r1", "uc1")
    cid = store.record_tool_call(task_id="t1", run_id="r1", usecase_id="uc1", tool_name="x")
    assert isinstance(cid, str)
    store.complete_task("t1", status="success")


# ---------------------------------------------------------------------------
# JSONLTelemetryStore
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_store(tmp_path: Path) -> JSONLTelemetryStore:
    return JSONLTelemetryStore(tmp_path / "telemetry")


def _read_records(store: JSONLTelemetryStore) -> list[dict]:
    lines: list[dict] = []
    for p in sorted(store._dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append(json.loads(line))
    return lines


def test_jsonl_creates_dir_on_first_write(jsonl_store: JSONLTelemetryStore) -> None:
    assert not jsonl_store._dir.exists()
    jsonl_store.record_task_start("t1", "r1", "uc1")
    assert jsonl_store._dir.exists()


def test_jsonl_record_task_start(jsonl_store: JSONLTelemetryStore) -> None:
    jsonl_store.record_task_start("t1", "r1", "uc1", worker_id="w0")
    records = _read_records(jsonl_store)
    assert len(records) == 1
    r = records[0]
    assert r["record"] == "task_start"
    assert r["task_id"] == "t1"
    assert r["run_id"] == "r1"
    assert r["usecase_id"] == "uc1"
    assert r["worker_id"] == "w0"
    assert "ts" in r


def test_jsonl_record_tool_call_returns_call_id(jsonl_store: JSONLTelemetryStore) -> None:
    cid = jsonl_store.record_tool_call(
        task_id="t1", run_id="r1", usecase_id="uc1",
        tool_name="extract", tool_version=2,
        inputs={"path": "/tmp/x.pdf"}, output={"pages": 3},
        duration_ms=42.5,
    )
    assert isinstance(cid, str) and len(cid) > 0
    records = _read_records(jsonl_store)
    assert records[0]["call_id"] == cid
    assert records[0]["tool_name"] == "extract"
    assert records[0]["tool_version"] == 2
    assert records[0]["inputs"] == {"path": "/tmp/x.pdf"}
    assert records[0]["duration_ms"] == 42.5


def test_jsonl_record_tool_call_respects_provided_call_id(jsonl_store: JSONLTelemetryStore) -> None:
    cid = jsonl_store.record_tool_call(
        call_id="fixed-id", task_id="t1", run_id="r1", usecase_id="uc1", tool_name="x"
    )
    assert cid == "fixed-id"
    assert _read_records(jsonl_store)[0]["call_id"] == "fixed-id"


def test_jsonl_record_llm_call(jsonl_store: JSONLTelemetryStore) -> None:
    msgs = [{"role": "user", "content": "hello"}]
    cid = jsonl_store.record_llm_call(
        task_id="t1", run_id="r1", usecase_id="uc1",
        model="claude-sonnet-4-6",
        messages=msgs,
        response={"role": "assistant", "content": "hi"},
        input_tokens=10, output_tokens=5, duration_ms=800.0,
    )
    r = _read_records(jsonl_store)[0]
    assert r["record"] == "llm_call"
    assert r["call_id"] == cid
    assert r["model"] == "claude-sonnet-4-6"
    assert r["messages"] == msgs
    assert r["input_tokens"] == 10


def test_jsonl_complete_task(jsonl_store: JSONLTelemetryStore) -> None:
    jsonl_store.complete_task("t1", status="success", duration_ms=1234.5)
    r = _read_records(jsonl_store)[0]
    assert r["record"] == "task_complete"
    assert r["status"] == "success"
    assert r["duration_ms"] == 1234.5


def test_jsonl_multiple_records_all_appended(jsonl_store: JSONLTelemetryStore) -> None:
    jsonl_store.record_task_start("t1", "r1", "uc1")
    jsonl_store.record_tool_call(task_id="t1", run_id="r1", usecase_id="uc1", tool_name="a")
    jsonl_store.record_tool_call(task_id="t1", run_id="r1", usecase_id="uc1", tool_name="b")
    jsonl_store.complete_task("t1", status="success")
    assert len(_read_records(jsonl_store)) == 4


# ---------------------------------------------------------------------------
# get_store factory
# ---------------------------------------------------------------------------


def test_get_store_draft_returns_jsonl(tmp_path: Path) -> None:
    store = get_store("draft", telemetry_dir=tmp_path / "tel")
    assert isinstance(store, JSONLTelemetryStore)


def test_get_store_validated_returns_jsonl(tmp_path: Path) -> None:
    store = get_store("validated", telemetry_dir=tmp_path / "tel")
    assert isinstance(store, JSONLTelemetryStore)


def test_get_store_in_production_without_dsn_raises(tmp_path: Path) -> None:
    with pytest.raises(TelemetryConfigError, match="PostgreSQL DSN"):
        get_store("in_production", telemetry_dir=tmp_path / "tel", pg_dsn=None)
