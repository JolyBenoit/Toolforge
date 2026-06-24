"""Persistence for the creator-facing judge's output.

Two tables, keyed so a re-run over the same use case / pipeline version
overwrites the previous snapshot (idempotent):

- ``tf_judge_creator_runs``         — one row per (usecase, run) assessment:
  computed_at and the full report JSON (axes + instructions).
- ``tf_judge_creator_instructions`` — one row per (usecase, run, instruction_id):
  the individual corrective instruction, so the TUI can list/select them and a
  later step can stamp ``change_reason=judge_instruction_id:<id>`` onto the
  pipeline version that acted on it.

Re-running clears the prior instruction set for that (usecase, run) before
inserting the new one, so an instruction that disappears does not linger.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .models import CreatorJudgeReport


class CreatorJudgeStore:
    """Interface for creator-judge persistence."""

    def save_report(self, report: CreatorJudgeReport) -> None:
        raise NotImplementedError

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> CreatorJudgeReport | None:
        raise NotImplementedError

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        """Repoint every persisted row from ``old_id`` to ``new_id`` (no-op by default)."""


class NullCreatorJudgeStore(CreatorJudgeStore):
    """No-op store for tests / dry runs; remembers nothing."""

    def save_report(self, report: CreatorJudgeReport) -> None:
        pass

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> CreatorJudgeReport | None:
        return None


_DDL = """
CREATE TABLE IF NOT EXISTS tf_judge_creator_runs (
    usecase_id    TEXT NOT NULL,
    run_id        TEXT NOT NULL DEFAULT '',
    computed_at   TIMESTAMPTZ NOT NULL,
    n_instructions INTEGER NOT NULL DEFAULT 0,
    report        JSONB NOT NULL,
    PRIMARY KEY (usecase_id, run_id)
);

CREATE TABLE IF NOT EXISTS tf_judge_creator_instructions (
    usecase_id       TEXT NOT NULL,
    run_id           TEXT NOT NULL DEFAULT '',
    instruction_id   TEXT NOT NULL,
    action           TEXT NOT NULL,
    target_tools     JSONB NOT NULL DEFAULT '[]',
    body             TEXT NOT NULL,
    rationale        TEXT NOT NULL DEFAULT '',
    priority         TEXT NOT NULL DEFAULT 'medium',
    expected_effect  TEXT NOT NULL DEFAULT '',
    computed_at      TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (usecase_id, run_id, instruction_id)
);
CREATE INDEX IF NOT EXISTS tf_judge_creator_instructions_uc
    ON tf_judge_creator_instructions (usecase_id, run_id);
"""


class PostgresCreatorJudgeStore(CreatorJudgeStore):
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

    def save_report(self, report: CreatorJudgeReport) -> None:
        run_id = report.run_id or ""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_judge_creator_runs
                    (usecase_id, run_id, computed_at, n_instructions, report)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (usecase_id, run_id) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    n_instructions = EXCLUDED.n_instructions,
                    report = EXCLUDED.report
                """,
                (
                    report.usecase_id, run_id, now, len(report.instructions),
                    json.dumps(report.to_dict(), ensure_ascii=False, default=str),
                ),
            )
            # Replace the instruction set wholesale, so an instruction that no
            # longer fires does not linger from a prior run.
            conn.execute(
                "DELETE FROM tf_judge_creator_instructions "
                "WHERE usecase_id = %s AND run_id = %s",
                (report.usecase_id, run_id),
            )
            for instr in report.instructions:
                conn.execute(
                    """
                    INSERT INTO tf_judge_creator_instructions
                        (usecase_id, run_id, instruction_id, action, target_tools,
                         body, rationale, priority, expected_effect, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (usecase_id, run_id, instruction_id) DO UPDATE SET
                        action = EXCLUDED.action,
                        target_tools = EXCLUDED.target_tools,
                        body = EXCLUDED.body,
                        rationale = EXCLUDED.rationale,
                        priority = EXCLUDED.priority,
                        expected_effect = EXCLUDED.expected_effect,
                        computed_at = EXCLUDED.computed_at
                    """,
                    (
                        report.usecase_id, run_id, instr.instruction_id,
                        instr.action,
                        json.dumps(instr.target_tools, ensure_ascii=False),
                        instr.body, instr.rationale, instr.priority,
                        instr.expected_effect, now,
                    ),
                )
            conn.commit()

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> CreatorJudgeReport | None:
        sql = (
            "SELECT report FROM tf_judge_creator_runs WHERE usecase_id = %s"
        )
        params: list[Any] = [usecase_id]
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        sql += " ORDER BY computed_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return CreatorJudgeReport.from_dict(row[0])

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        with self._connect() as conn:
            for table in (
                "tf_judge_creator_runs",
                "tf_judge_creator_instructions",
            ):
                conn.execute(
                    f"UPDATE {table} SET usecase_id = %s WHERE usecase_id = %s",
                    (new_id, old_id),
                )
            conn.commit()


def get_creator_judge_store(dsn: str) -> CreatorJudgeStore:
    """Return a Postgres-backed creator judge store for the given DSN."""
    return PostgresCreatorJudgeStore(dsn)
