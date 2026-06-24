"""Persistence for dynamic-judge global notes.

Two tables, keyed so a re-run over the same use case / pipeline version
overwrites the previous global snapshot (idempotent):

- ``tf_judge_dynamic_runs``   — one row per (usecase, run) assessment: windows,
  mean structural stability, dag diversity, diagnosis, full report JSON.
- ``tf_judge_global_notes``   — one row per (usecase, run, tool): the global
  per-tool note (means, %, breaches). One row per tool → independent per tool.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .models import DynamicJudgeReport


class DynamicJudgeStore:
    """Interface for dynamic-judge persistence."""

    def save_report(self, report: DynamicJudgeReport) -> None:
        raise NotImplementedError

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        """Repoint every persisted row from ``old_id`` to ``new_id`` (no-op by default)."""


class NullDynamicJudgeStore(DynamicJudgeStore):
    def save_report(self, report: DynamicJudgeReport) -> None:
        pass


_DDL = """
CREATE TABLE IF NOT EXISTS tf_judge_dynamic_runs (
    usecase_id                 TEXT NOT NULL,
    run_id                     TEXT NOT NULL DEFAULT '',
    computed_at                TIMESTAMPTZ NOT NULL,
    short_window               INTEGER NOT NULL,
    long_window                INTEGER NOT NULL,
    n_tasks                    INTEGER NOT NULL,
    mean_structural_stability  DOUBLE PRECISION,
    dag_diversity              DOUBLE PRECISION,
    stability_breached         BOOLEAN NOT NULL DEFAULT FALSE,
    diagnosis                  TEXT,
    report                     JSONB NOT NULL,
    PRIMARY KEY (usecase_id, run_id)
);

CREATE TABLE IF NOT EXISTS tf_judge_global_notes (
    usecase_id            TEXT NOT NULL,
    run_id                TEXT NOT NULL DEFAULT '',
    tool_id               TEXT NOT NULL,
    n_tasks               INTEGER NOT NULL,
    mean_scores           JSONB NOT NULL DEFAULT '{}',
    recommendation_rate   DOUBLE PRECISION NOT NULL DEFAULT 0,
    dominant_target       TEXT NOT NULL DEFAULT 'none',
    breaches              JSONB NOT NULL DEFAULT '[]',
    computed_at           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (usecase_id, run_id, tool_id)
);
CREATE INDEX IF NOT EXISTS tf_judge_global_notes_tool
    ON tf_judge_global_notes (usecase_id, tool_id);
"""


class PostgresDynamicJudgeStore(DynamicJudgeStore):
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

    def save_report(self, report: DynamicJudgeReport) -> None:
        run_id = report.run_id or ""
        now = datetime.now(UTC).isoformat()
        ss = report.structural_stability
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_judge_dynamic_runs
                    (usecase_id, run_id, computed_at, short_window, long_window,
                     n_tasks, mean_structural_stability, dag_diversity,
                     stability_breached, diagnosis, report)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (usecase_id, run_id) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    short_window = EXCLUDED.short_window,
                    long_window = EXCLUDED.long_window,
                    n_tasks = EXCLUDED.n_tasks,
                    mean_structural_stability = EXCLUDED.mean_structural_stability,
                    dag_diversity = EXCLUDED.dag_diversity,
                    stability_breached = EXCLUDED.stability_breached,
                    diagnosis = EXCLUDED.diagnosis,
                    report = EXCLUDED.report
                """,
                (
                    report.usecase_id, run_id, now,
                    report.short_window, report.long_window, report.n_tasks,
                    ss.mean_structural_stability, ss.dag_diversity, ss.breached,
                    report.diagnosis,
                    json.dumps(report.to_dict(), ensure_ascii=False, default=str),
                ),
            )
            for note in report.tool_global_notes:
                conn.execute(
                    """
                    INSERT INTO tf_judge_global_notes
                        (usecase_id, run_id, tool_id, n_tasks, mean_scores,
                         recommendation_rate, dominant_target, breaches, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (usecase_id, run_id, tool_id) DO UPDATE SET
                        n_tasks = EXCLUDED.n_tasks,
                        mean_scores = EXCLUDED.mean_scores,
                        recommendation_rate = EXCLUDED.recommendation_rate,
                        dominant_target = EXCLUDED.dominant_target,
                        breaches = EXCLUDED.breaches,
                        computed_at = EXCLUDED.computed_at
                    """,
                    (
                        report.usecase_id, run_id, note.tool_id, note.n_tasks,
                        json.dumps(note.mean_scores, ensure_ascii=False),
                        note.recommendation_rate, note.dominant_target,
                        json.dumps(note.breaches, ensure_ascii=False), now,
                    ),
                )
            conn.commit()

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        with self._connect() as conn:
            for table in ("tf_judge_dynamic_runs", "tf_judge_global_notes"):
                conn.execute(
                    f"UPDATE {table} SET usecase_id = %s WHERE usecase_id = %s",
                    (new_id, old_id),
                )
            conn.commit()


def get_dynamic_judge_store(dsn: str) -> DynamicJudgeStore:
    return PostgresDynamicJudgeStore(dsn)
