"""Persistence for static-judge output.

Three tables, plus a write-back of each verdict's ``contribution`` into the
reserved ``tf_prod_spans.contribution`` column so the metric layer picks it up:

- ``tf_judge_static_results`` — one row per judged task (latest judgment wins).
- ``tf_judge_span_verdicts``  — one row per (task, span): backward-pass verdict.
- ``tf_judge_tool_notes``     — one row per (task, tool): local note. Each tool
  is a separate row, so a tool's comments are independent of every other tool.

Re-running the judge over a task overwrites its previous rows (idempotent on the
primary key), which is what lets the judge be replayed over a window to chain
iterations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .models import RecommendationTarget, StaticJudgeResult


@dataclass
class ToolNoteRecord:
    """A stored per-(task, tool) local note, as read back for aggregation.

    This is the unit the dynamic judge averages into global notes; it is the
    read-side mirror of :class:`toolforge_judge.static.models.ToolNote`.
    """

    task_id: str
    tool_id: str
    scores: dict[str, float] = field(default_factory=dict)
    recommendation: str | None = None
    recommendation_target: RecommendationTarget = "none"


class JudgeStore:
    """Interface for static-judge persistence."""

    def save_result(self, result: StaticJudgeResult) -> None:
        raise NotImplementedError

    def judged_task_ids(self, usecase_id: str, *, run_id: str | None = None) -> set[str]:
        """Task ids already judged — used to skip work on incremental re-runs."""
        raise NotImplementedError

    def load_tool_notes(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ToolNoteRecord]:
        """All per-tool local notes for a use case (optionally one run)."""
        raise NotImplementedError


class NullJudgeStore(JudgeStore):
    """No-op store for tests / dry runs; remembers nothing."""

    def save_result(self, result: StaticJudgeResult) -> None:
        pass

    def judged_task_ids(self, usecase_id: str, *, run_id: str | None = None) -> set[str]:
        return set()

    def load_tool_notes(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ToolNoteRecord]:
        return []


_DDL = """
CREATE TABLE IF NOT EXISTS tf_judge_static_results (
    task_id        TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    usecase_id     TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    raw_response   TEXT
);
CREATE INDEX IF NOT EXISTS tf_judge_results_uc
    ON tf_judge_static_results (usecase_id, run_id);

CREATE TABLE IF NOT EXISTS tf_judge_span_verdicts (
    task_id                TEXT NOT NULL,
    span_id                TEXT NOT NULL,
    run_id                 TEXT NOT NULL,
    usecase_id             TEXT NOT NULL,
    tool_id                TEXT,
    contribution           TEXT NOT NULL,
    selection_appropriate  BOOLEAN,
    param_fidelity         DOUBLE PRECISION,
    rationale              TEXT,
    PRIMARY KEY (task_id, span_id)
);
CREATE INDEX IF NOT EXISTS tf_judge_verdicts_tool
    ON tf_judge_span_verdicts (usecase_id, tool_id);

CREATE TABLE IF NOT EXISTS tf_judge_tool_notes (
    task_id                TEXT NOT NULL,
    tool_id                TEXT NOT NULL,
    run_id                 TEXT NOT NULL,
    usecase_id             TEXT NOT NULL,
    scores                 JSONB NOT NULL DEFAULT '{}',
    recommendation         TEXT,
    recommendation_target  TEXT NOT NULL DEFAULT 'none',
    created_at             TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (task_id, tool_id)
);
CREATE INDEX IF NOT EXISTS tf_judge_notes_tool
    ON tf_judge_tool_notes (usecase_id, tool_id);
CREATE INDEX IF NOT EXISTS tf_judge_notes_recs
    ON tf_judge_tool_notes (usecase_id, tool_id)
    WHERE recommendation IS NOT NULL;
"""


class PostgresJudgeStore(JudgeStore):
    """Postgres backend; shares the DSN with the production telemetry store."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "psycopg is not installed. "
                "Add the postgres extra: uv add 'toolforge-judge[postgres]'"
            ) from exc
        self._dsn = dsn
        with self._connect() as conn:
            conn.execute(_DDL)
            conn.commit()

    def _connect(self) -> Any:
        import psycopg  # type: ignore[import-untyped]

        return psycopg.connect(self._dsn)

    def save_result(self, result: StaticJudgeResult) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_judge_static_results
                    (task_id, run_id, usecase_id, judge_model, created_at, raw_response)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    judge_model = EXCLUDED.judge_model,
                    created_at  = EXCLUDED.created_at,
                    raw_response = EXCLUDED.raw_response
                """,
                (result.task_id, result.run_id, result.usecase_id,
                 result.judge_model, now, result.raw_response),
            )

            for v in result.span_verdicts:
                conn.execute(
                    """
                    INSERT INTO tf_judge_span_verdicts
                        (task_id, span_id, run_id, usecase_id, tool_id, contribution,
                         selection_appropriate, param_fidelity, rationale)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id, span_id) DO UPDATE SET
                        tool_id = EXCLUDED.tool_id,
                        contribution = EXCLUDED.contribution,
                        selection_appropriate = EXCLUDED.selection_appropriate,
                        param_fidelity = EXCLUDED.param_fidelity,
                        rationale = EXCLUDED.rationale
                    """,
                    (result.task_id, v.span_id, result.run_id, result.usecase_id,
                     v.tool_id, v.contribution, v.selection_appropriate,
                     v.param_fidelity, v.rationale),
                )
                # Write-back into the reserved production column for the metrics.
                conn.execute(
                    "UPDATE tf_prod_spans SET contribution = %s WHERE span_id = %s",
                    (v.contribution, v.span_id),
                )

            for n in result.tool_notes:
                conn.execute(
                    """
                    INSERT INTO tf_judge_tool_notes
                        (task_id, tool_id, run_id, usecase_id, scores,
                         recommendation, recommendation_target, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (task_id, tool_id) DO UPDATE SET
                        scores = EXCLUDED.scores,
                        recommendation = EXCLUDED.recommendation,
                        recommendation_target = EXCLUDED.recommendation_target,
                        created_at = EXCLUDED.created_at
                    """,
                    (result.task_id, n.tool_id, result.run_id, result.usecase_id,
                     json.dumps(n.scores, ensure_ascii=False),
                     n.recommendation, n.recommendation_target, now),
                )
            conn.commit()

    def judged_task_ids(self, usecase_id: str, *, run_id: str | None = None) -> set[str]:
        sql = "SELECT task_id FROM tf_judge_static_results WHERE usecase_id = %s"
        params: list[Any] = [usecase_id]
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {r[0] for r in rows}

    def load_tool_notes(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ToolNoteRecord]:
        sql = (
            "SELECT task_id, tool_id, scores, recommendation, recommendation_target "
            "FROM tf_judge_tool_notes WHERE usecase_id = %s"
        )
        params: list[Any] = [usecase_id]
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ToolNoteRecord(
                task_id=r[0],
                tool_id=r[1],
                scores=dict(r[2] or {}),
                recommendation=r[3],
                recommendation_target=r[4] or "none",
            )
            for r in rows
        ]


def get_judge_store(dsn: str) -> JudgeStore:
    """Return a Postgres-backed judge store for the given DSN."""
    return PostgresJudgeStore(dsn)
