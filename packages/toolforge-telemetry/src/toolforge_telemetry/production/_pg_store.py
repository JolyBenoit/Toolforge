"""PostgresProductionTelemetryStore — Postgres backend for in_production runs.

Requires: uv add 'toolforge-telemetry[postgres]'

Design decisions
----------------
- One connection per operation: no shared state across concurrent tasks.
- JSONB for every structured field: enables JSON-path queries by the Judge
  without schema migrations when the payload evolves.
- ``tf_prod_spans`` is a single unified table for all span types
  (llm_call, tool_call, user_wait).  The ``type`` column is the discriminator.
  Unused columns for a given type are NULL — simpler for cross-type queries
  than a table-per-type design.
- ``input_timeline`` is appended incrementally via ``jsonb_insert`` so the
  caller never needs to buffer the full timeline in memory.
"""
from __future__ import annotations

import json
from typing import Any

from .models import (
    DAG,
    InputTimelineEntry,
    PipelineSpec,
    Span,
    TaskCost,
    TaskStatus,
    UserFeedback,
    to_dict,
)
from .store import ProductionTelemetryStore

try:
    import psycopg  # type: ignore[import-untyped]
    _PSYCOPG_AVAILABLE = True
except ImportError:
    _PSYCOPG_AVAILABLE = False


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
-- Immutable pipeline version snapshot written at promote-to-production.
CREATE TABLE IF NOT EXISTS tf_prod_pipeline_specs (
    run_id          TEXT PRIMARY KEY,
    usecase_id      TEXT NOT NULL,
    promoted_at     TIMESTAMPTZ NOT NULL,
    forked_from     TEXT,
    change_reason   TEXT,
    system_prompt   TEXT,
    tools_snapshot  JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS tf_prod_specs_usecase
    ON tf_prod_pipeline_specs (usecase_id);

-- One row per complete conversation session.
CREATE TABLE IF NOT EXISTS tf_prod_tasks (
    task_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    usecase_id      TEXT NOT NULL,
    user_session_id TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    input_timeline  JSONB NOT NULL DEFAULT '[]',
    final_output    TEXT,
    user_feedback   JSONB,
    cost            JSONB,
    dag             JSONB
);
CREATE INDEX IF NOT EXISTS tf_prod_tasks_run
    ON tf_prod_tasks (run_id);
CREATE INDEX IF NOT EXISTS tf_prod_tasks_uc_ts
    ON tf_prod_tasks (usecase_id, started_at DESC);

-- One row per span (llm_call | tool_call | user_wait).
-- Columns are typed per span type; unused columns are NULL.
CREATE TABLE IF NOT EXISTS tf_prod_spans (
    span_id             TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL REFERENCES tf_prod_tasks (task_id),
    run_id              TEXT NOT NULL,
    usecase_id          TEXT NOT NULL,
    type                TEXT NOT NULL,
    llm_turn_index      INTEGER,
    call_index_in_turn  INTEGER,
    parent_spans        JSONB NOT NULL DEFAULT '[]',
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    duration_ms         DOUBLE PRECISION,
    status              TEXT,

    -- tool_call --
    tool_id             TEXT,
    tool_version        INTEGER,
    input               JSONB,
    output              JSONB,
    retries             JSONB NOT NULL DEFAULT '[]',
    nested_llm_calls    JSONB NOT NULL DEFAULT '[]',

    -- llm_call --
    llm_model           TEXT,
    prompt_hash         TEXT,
    system_prompt       TEXT,
    messages            JSONB,
    response            JSONB,
    tokens_in           INTEGER,
    tokens_out          INTEGER,

    -- user_wait --
    user_turn           INTEGER,
    user_message        TEXT,

    -- judge (filled post-hoc, never at capture time) --
    contribution        TEXT
);
CREATE INDEX IF NOT EXISTS tf_prod_spans_task
    ON tf_prod_spans (task_id);
CREATE INDEX IF NOT EXISTS tf_prod_spans_run_ts
    ON tf_prod_spans (run_id, started_at DESC);
CREATE INDEX IF NOT EXISTS tf_prod_spans_type
    ON tf_prod_spans (type, started_at DESC);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _j(value: Any) -> str | None:
    """Serialise a value to a JSONB-compatible string, or None."""
    if value is None:
        return None
    return json.dumps(to_dict(value), ensure_ascii=False, default=str)


def _j_required(value: Any) -> str:
    return json.dumps(to_dict(value), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PostgresProductionTelemetryStore(ProductionTelemetryStore):
    """Production telemetry backend backed by PostgreSQL.

    Each public method opens and closes its own connection so concurrent
    tasks never share a connection object.
    """

    def __init__(self, dsn: str) -> None:
        if not _PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg is not installed. "
                "Add the postgres extra: uv add 'toolforge-telemetry[postgres]'"
            )
        self._dsn = dsn
        self._ensure_schema()

    # Keep this short: libpq applies it per resolved address (IPv4 + IPv6),
    # so the worst-case wait is roughly 2× this value. A small value means an
    # unreachable database degrades to "telemetry off" in a few seconds instead
    # of stalling a run launch for ~20s.
    _CONNECT_TIMEOUT_S = 3

    def _connect(self) -> "psycopg.Connection[Any]":
        # Bound the connection attempt so an unreachable database fails fast
        # with a clear error instead of hanging on the OS-level TCP timeout.
        return psycopg.connect(self._dsn, connect_timeout=self._CONNECT_TIMEOUT_S)  # type: ignore[return-value]

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_DDL)
            conn.commit()

    # --- pipeline lifecycle -------------------------------------------------

    def record_pipeline_spec(self, spec: PipelineSpec) -> None:
        tools_json = _j_required([to_dict(t) for t in spec.tools])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_prod_pipeline_specs
                  (run_id, usecase_id, promoted_at, forked_from,
                   change_reason, system_prompt, tools_snapshot)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    spec.run_id,
                    spec.usecase_id,
                    spec.promoted_at,
                    spec.forked_from,
                    spec.change_reason,
                    spec.system_prompt,
                    tools_json,
                ),
            )
            conn.commit()

    # --- task lifecycle -----------------------------------------------------

    def open_task(
        self,
        task_id: str,
        run_id: str,
        usecase_id: str,
        user_session_id: str,
        started_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_prod_tasks
                  (task_id, run_id, usecase_id, user_session_id, status, started_at)
                VALUES (%s, %s, %s, %s, 'running', %s)
                ON CONFLICT (task_id) DO NOTHING
                """,
                (task_id, run_id, usecase_id, user_session_id, started_at),
            )
            conn.commit()

    def append_input_entry(self, task_id: str, entry: InputTimelineEntry) -> None:
        entry_json = _j_required(entry)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tf_prod_tasks
                SET input_timeline = input_timeline || %s::jsonb
                WHERE task_id = %s
                """,
                (f"[{entry_json}]", task_id),
            )
            conn.commit()

    def close_task(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        final_output: str | None,
        cost: TaskCost,
        dag: DAG,
        ended_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tf_prod_tasks
                SET status       = %s,
                    ended_at     = %s,
                    final_output = %s,
                    cost         = %s,
                    dag          = %s
                WHERE task_id = %s
                """,
                (
                    status,
                    ended_at,
                    final_output,
                    _j_required(cost),
                    _j_required(dag),
                    task_id,
                ),
            )
            conn.commit()

    def record_user_feedback(self, task_id: str, feedback: UserFeedback) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tf_prod_tasks
                SET user_feedback = %s
                WHERE task_id = %s
                """,
                (_j_required(feedback), task_id),
            )
            conn.commit()

    # --- span recording -----------------------------------------------------

    def record_span(self, span: Span) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_prod_spans (
                    span_id, task_id, run_id, usecase_id,
                    type, llm_turn_index, call_index_in_turn, parent_spans,
                    started_at, ended_at, duration_ms, status,
                    tool_id, tool_version, input, output, retries, nested_llm_calls,
                    llm_model, prompt_hash, system_prompt, messages, response,
                    tokens_in, tokens_out,
                    user_turn, user_message,
                    contribution
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s
                )
                ON CONFLICT (span_id) DO NOTHING
                """,
                (
                    span.span_id, span.task_id, span.run_id, span.usecase_id,
                    span.type, span.llm_turn_index, span.call_index_in_turn,
                    _j_required(span.parent_spans),
                    span.started_at, span.ended_at, span.duration_ms, span.status,
                    # tool_call
                    span.tool_id, span.tool_version,
                    _j(span.input), _j(span.output),
                    _j_required(span.retries),
                    _j_required(span.nested_llm_calls),
                    # llm_call
                    span.llm_meta.model if span.llm_meta else None,
                    span.llm_meta.prompt_hash if span.llm_meta else None,
                    span.system_prompt,
                    _j(span.messages),
                    _j(span.response),
                    span.llm_meta.tokens_in if span.llm_meta else None,
                    span.llm_meta.tokens_out if span.llm_meta else None,
                    # user_wait
                    span.user_turn, span.user_message,
                    # judge
                    span.contribution,
                ),
            )
            conn.commit()

    # --- task maintenance (TUI Runs tab) -----------------------------------

    def delete_tasks(self, task_ids: list[str]) -> int:
        """Hard-delete tasks and their spans in one transaction (FK order)."""
        if not task_ids:
            return 0
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM tf_prod_spans WHERE task_id = ANY(%s)", (task_ids,)
            )
            cur = conn.execute(
                "DELETE FROM tf_prod_tasks WHERE task_id = ANY(%s)", (task_ids,)
            )
            conn.commit()
            return cur.rowcount

    def set_task_status(self, task_ids: list[str], status: TaskStatus) -> int:
        """Overwrite the status of the given tasks."""
        if not task_ids:
            return 0
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tf_prod_tasks SET status = %s WHERE task_id = ANY(%s)",
                (status, task_ids),
            )
            conn.commit()
            return cur.rowcount

    # --- maintenance --------------------------------------------------------

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        """Repoint every production row from ``old_id`` to ``new_id``.

        Specs, tasks, and spans are updated together in one transaction. The
        operation is idempotent, so retrying after a partial failure is safe.
        """
        with self._connect() as conn:
            for table in ("tf_prod_pipeline_specs", "tf_prod_tasks", "tf_prod_spans"):
                conn.execute(
                    f"UPDATE {table} SET usecase_id = %s WHERE usecase_id = %s",
                    (new_id, old_id),
                )
            conn.commit()
