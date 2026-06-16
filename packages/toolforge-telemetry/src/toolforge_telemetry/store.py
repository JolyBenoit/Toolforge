"""TelemetryStore — Protocol, factory, and built-in implementations."""
from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class TelemetryConfigError(Exception):
    pass


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TelemetryStore:
    """Interface for execution telemetry backends.

    draft / validated runs  → JSONLTelemetryStore  (no external service)
    in_production runs      → PostgresTelemetryStore (requires PostgreSQL)
    """

    def record_task_start(
        self,
        task_id: str,
        run_id: str,
        usecase_id: str,
        *,
        worker_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    def record_tool_call(
        self,
        *,
        call_id: str | None = None,
        task_id: str,
        run_id: str,
        usecase_id: str,
        tool_name: str,
        tool_version: int | None = None,
        inputs: dict[str, Any] | None = None,
        output: Any = None,
        duration_ms: float | None = None,
        error: str | None = None,
        error_kind: str | None = None,
    ) -> str:
        """Record a tool call. Returns the generated call_id."""
        raise NotImplementedError

    def record_llm_call(
        self,
        *,
        call_id: str | None = None,
        task_id: str,
        run_id: str,
        usecase_id: str,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        response: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: float | None = None,
    ) -> str:
        """Record an LLM call. Returns the generated call_id."""
        raise NotImplementedError

    def complete_task(
        self,
        task_id: str,
        *,
        status: str,
        duration_ms: float | None = None,
    ) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NullTelemetryStore — no-op, for tests and offline stubs
# ---------------------------------------------------------------------------


class NullTelemetryStore(TelemetryStore):
    def record_task_start(self, task_id: str, run_id: str, usecase_id: str, *, worker_id: str | None = None) -> None:
        pass

    def record_tool_call(self, *, call_id: str | None = None, task_id: str, run_id: str, usecase_id: str, tool_name: str, tool_version: int | None = None, inputs: dict[str, Any] | None = None, output: Any = None, duration_ms: float | None = None, error: str | None = None, error_kind: str | None = None) -> str:
        return call_id or secrets.token_hex(8)

    def record_llm_call(self, *, call_id: str | None = None, task_id: str, run_id: str, usecase_id: str, model: str | None = None, messages: list[dict[str, Any]] | None = None, response: Any = None, input_tokens: int | None = None, output_tokens: int | None = None, duration_ms: float | None = None) -> str:
        return call_id or secrets.token_hex(8)

    def complete_task(self, task_id: str, *, status: str, duration_ms: float | None = None) -> None:
        pass


# ---------------------------------------------------------------------------
# JSONLTelemetryStore — draft / validated runs, no external service
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JSONLTelemetryStore(TelemetryStore):
    """Append-only JSONL store with daily file rotation.

    Writes to: {telemetry_dir}/YYYY-MM-DD.jsonl
    Thread-safe via a per-instance lock.
    """

    def __init__(self, telemetry_dir: Path) -> None:
        self._dir = telemetry_dir
        self._lock = threading.Lock()

    def _append(self, record: dict[str, Any]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._dir / f"{date}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def record_task_start(self, task_id: str, run_id: str, usecase_id: str, *, worker_id: str | None = None) -> None:
        self._append({
            "record": "task_start",
            "task_id": task_id,
            "run_id": run_id,
            "usecase_id": usecase_id,
            "worker_id": worker_id,
            "ts": _now_iso(),
        })

    def record_tool_call(self, *, call_id: str | None = None, task_id: str, run_id: str, usecase_id: str, tool_name: str, tool_version: int | None = None, inputs: dict[str, Any] | None = None, output: Any = None, duration_ms: float | None = None, error: str | None = None, error_kind: str | None = None) -> str:
        cid = call_id or secrets.token_hex(8)
        self._append({
            "record": "tool_call",
            "call_id": cid,
            "task_id": task_id,
            "run_id": run_id,
            "usecase_id": usecase_id,
            "tool_name": tool_name,
            "tool_version": tool_version,
            "inputs": inputs,
            "output": output,
            "duration_ms": duration_ms,
            "error": error,
            "error_kind": error_kind,
            "ts": _now_iso(),
        })
        return cid

    def record_llm_call(self, *, call_id: str | None = None, task_id: str, run_id: str, usecase_id: str, model: str | None = None, messages: list[dict[str, Any]] | None = None, response: Any = None, input_tokens: int | None = None, output_tokens: int | None = None, duration_ms: float | None = None) -> str:
        cid = call_id or secrets.token_hex(8)
        self._append({
            "record": "llm_call",
            "call_id": cid,
            "task_id": task_id,
            "run_id": run_id,
            "usecase_id": usecase_id,
            "model": model,
            "messages": messages,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_ms": duration_ms,
            "ts": _now_iso(),
        })
        return cid

    def complete_task(self, task_id: str, *, status: str, duration_ms: float | None = None) -> None:
        self._append({
            "record": "task_complete",
            "task_id": task_id,
            "status": status,
            "duration_ms": duration_ms,
            "ts": _now_iso(),
        })


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_store(
    run_status: Literal["draft", "validated", "in_production"],
    *,
    telemetry_dir: Path,
    pg_dsn: str | None = None,
) -> TelemetryStore:
    """Return the appropriate TelemetryStore for the given run status.

    - draft / validated  → JSONLTelemetryStore  (no PostgreSQL required)
    - in_production      → PostgresTelemetryStore (pg_dsn required)
    """
    if run_status == "in_production":
        if not pg_dsn:
            raise TelemetryConfigError(
                "PostgreSQL DSN is required for in_production runs. "
                "Start PostgreSQL (docker compose up -d) and set TOOLFORGE_PG_DSN "
                "or configure [telemetry] dsn in toolforge.toml."
            )
        from ._pg_store import PostgresTelemetryStore
        return PostgresTelemetryStore(pg_dsn)
    return JSONLTelemetryStore(telemetry_dir)
