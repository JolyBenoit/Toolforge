"""PostgresTelemetryStore — production-grade store backed by PostgreSQL.

Requires psycopg[binary]>=3.1 (optional dependency).
Install: uv add 'toolforge-telemetry[postgres]'

One connection is opened per operation so concurrent tasks (threads/processes)
never share a connection and PostgreSQL handles parallelism natively via MVCC.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any

from .store import TelemetryConfigError, TelemetryStore

try:
    import psycopg  # type: ignore[import-untyped]
    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False


_DDL = """
CREATE TABLE IF NOT EXISTS tf_tasks (
    task_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    usecase_id   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',
    started_at   TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_ms  DOUBLE PRECISION,
    worker_id    TEXT
);

CREATE INDEX IF NOT EXISTS tf_tasks_run_idx       ON tf_tasks (run_id);
CREATE INDEX IF NOT EXISTS tf_tasks_usecase_ts    ON tf_tasks (usecase_id, started_at DESC);

CREATE TABLE IF NOT EXISTS tf_tool_calls (
    call_id      TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL REFERENCES tf_tasks(task_id),
    run_id       TEXT NOT NULL,
    usecase_id   TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    tool_version INTEGER,
    inputs       JSONB,
    output       JSONB,
    duration_ms  DOUBLE PRECISION,
    error        TEXT,
    error_kind   TEXT,
    ts           TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS tf_tool_calls_task     ON tf_tool_calls (task_id);
CREATE INDEX IF NOT EXISTS tf_tool_calls_run_ts   ON tf_tool_calls (run_id, ts DESC);
CREATE INDEX IF NOT EXISTS tf_tool_calls_usecase  ON tf_tool_calls (usecase_id, ts DESC);

CREATE TABLE IF NOT EXISTS tf_llm_calls (
    call_id       TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tf_tasks(task_id),
    run_id        TEXT NOT NULL,
    usecase_id    TEXT NOT NULL,
    model         TEXT,
    messages      JSONB,
    response      JSONB,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    duration_ms   DOUBLE PRECISION,
    ts            TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS tf_llm_calls_task      ON tf_llm_calls (task_id);
CREATE INDEX IF NOT EXISTS tf_llm_calls_run_ts    ON tf_llm_calls (run_id, ts DESC);
CREATE INDEX IF NOT EXISTS tf_llm_calls_usecase   ON tf_llm_calls (usecase_id, ts DESC);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonb(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


class PostgresTelemetryStore(TelemetryStore):
    """Telemetry store backed by PostgreSQL — for in_production runs.

    Each method opens its own connection so parallel tasks never contend on
    a shared connection object. PostgreSQL's MVCC handles concurrent inserts.
    """

    def __init__(self, dsn: str) -> None:
        if not _PSYCOPG_AVAILABLE:
            raise TelemetryConfigError(
                "psycopg is not installed. "
                "Add the postgres extra: uv add 'toolforge-telemetry[postgres]'"
            )
        self._dsn = dsn
        self._ensure_schema()

    def _connect(self) -> "psycopg.Connection[Any]":
        return psycopg.connect(self._dsn)  # type: ignore[return-value]

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_DDL)
            conn.commit()

    def record_task_start(
        self,
        task_id: str,
        run_id: str,
        usecase_id: str,
        *,
        worker_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_tasks (task_id, run_id, usecase_id, status, started_at, worker_id)
                VALUES (%s, %s, %s, 'running', %s, %s)
                ON CONFLICT (task_id) DO NOTHING
                """,
                (task_id, run_id, usecase_id, _now(), worker_id),
            )
            conn.commit()

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
        cid = call_id or secrets.token_hex(8)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_tool_calls
                  (call_id, task_id, run_id, usecase_id, tool_name, tool_version,
                   inputs, output, duration_ms, error, error_kind, ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    cid, task_id, run_id, usecase_id, tool_name, tool_version,
                    _jsonb(inputs), _jsonb(output),
                    duration_ms, error, error_kind, _now(),
                ),
            )
            conn.commit()
        return cid

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
        cid = call_id or secrets.token_hex(8)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_llm_calls
                  (call_id, task_id, run_id, usecase_id, model,
                   messages, response, input_tokens, output_tokens, duration_ms, ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    cid, task_id, run_id, usecase_id, model,
                    _jsonb(messages), _jsonb(response),
                    input_tokens, output_tokens, duration_ms, _now(),
                ),
            )
            conn.commit()
        return cid

    def complete_task(
        self,
        task_id: str,
        *,
        status: str,
        duration_ms: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tf_tasks
                SET status = %s, completed_at = %s, duration_ms = %s
                WHERE task_id = %s
                """,
                (status, _now(), duration_ms, task_id),
            )
            conn.commit()
