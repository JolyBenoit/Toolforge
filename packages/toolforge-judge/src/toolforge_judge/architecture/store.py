"""Persistence for architecture-judge output.

Two tables, keyed so a re-run over the same use case / pipeline version
overwrites the previous snapshot (idempotent):

- ``tf_judge_architecture_runs``     — one row per (usecase, run) assessment:
  mode, judge model, and the full report JSON (contracts + findings).
- ``tf_judge_architecture_findings`` — one row per (usecase, run, finding_id):
  the individual pipeline finding, so the TUI and the creator judge can read
  them back without rehydrating the whole report.

Re-running clears the prior findings for that (usecase, run) before inserting the
new set, so a finding that disappears between runs does not linger.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .models import ArchitectureFinding, ArchitectureJudgeReport


class ArchitectureJudgeStore:
    """Interface for architecture-judge persistence."""

    def save_report(self, report: ArchitectureJudgeReport) -> None:
        raise NotImplementedError

    def load_findings(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ArchitectureFinding]:
        raise NotImplementedError

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> ArchitectureJudgeReport | None:
        raise NotImplementedError

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        """Repoint every persisted row from ``old_id`` to ``new_id`` (no-op by default)."""


class NullArchitectureJudgeStore(ArchitectureJudgeStore):
    """No-op store for tests / dry runs; remembers nothing."""

    def save_report(self, report: ArchitectureJudgeReport) -> None:
        pass

    def load_findings(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ArchitectureFinding]:
        return []

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> ArchitectureJudgeReport | None:
        return None


_DDL = """
CREATE TABLE IF NOT EXISTS tf_judge_architecture_runs (
    usecase_id    TEXT NOT NULL,
    run_id        TEXT NOT NULL DEFAULT '',
    computed_at   TIMESTAMPTZ NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'design_time',
    judge_model   TEXT NOT NULL DEFAULT '',
    n_findings    INTEGER NOT NULL DEFAULT 0,
    report        JSONB NOT NULL,
    PRIMARY KEY (usecase_id, run_id)
);

CREATE TABLE IF NOT EXISTS tf_judge_architecture_findings (
    usecase_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL DEFAULT '',
    finding_id              TEXT NOT NULL,
    category                TEXT NOT NULL,
    severity                TEXT NOT NULL DEFAULT 'warning',
    tools_involved          JSONB NOT NULL DEFAULT '[]',
    requirement_threatened  TEXT NOT NULL DEFAULT '',
    body                    TEXT NOT NULL,
    evidence                TEXT NOT NULL DEFAULT '',
    proposed_action         TEXT NOT NULL DEFAULT 'none',
    computed_at             TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (usecase_id, run_id, finding_id)
);
CREATE INDEX IF NOT EXISTS tf_judge_arch_findings_uc
    ON tf_judge_architecture_findings (usecase_id, run_id);
"""


class PostgresArchitectureJudgeStore(ArchitectureJudgeStore):
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

    def save_report(self, report: ArchitectureJudgeReport) -> None:
        run_id = report.run_id or ""
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tf_judge_architecture_runs
                    (usecase_id, run_id, computed_at, mode, judge_model,
                     n_findings, report)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (usecase_id, run_id) DO UPDATE SET
                    computed_at = EXCLUDED.computed_at,
                    mode = EXCLUDED.mode,
                    judge_model = EXCLUDED.judge_model,
                    n_findings = EXCLUDED.n_findings,
                    report = EXCLUDED.report
                """,
                (
                    report.usecase_id, run_id, now, report.mode,
                    report.judge_model, len(report.findings),
                    json.dumps(report.to_dict(), ensure_ascii=False, default=str),
                ),
            )
            # Replace the finding set for this (usecase, run) wholesale, so a
            # finding that no longer fires does not linger from a prior run.
            conn.execute(
                "DELETE FROM tf_judge_architecture_findings "
                "WHERE usecase_id = %s AND run_id = %s",
                (report.usecase_id, run_id),
            )
            for f in report.findings:
                conn.execute(
                    """
                    INSERT INTO tf_judge_architecture_findings
                        (usecase_id, run_id, finding_id, category, severity,
                         tools_involved, requirement_threatened, body, evidence,
                         proposed_action, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (usecase_id, run_id, finding_id) DO UPDATE SET
                        category = EXCLUDED.category,
                        severity = EXCLUDED.severity,
                        tools_involved = EXCLUDED.tools_involved,
                        requirement_threatened = EXCLUDED.requirement_threatened,
                        body = EXCLUDED.body,
                        evidence = EXCLUDED.evidence,
                        proposed_action = EXCLUDED.proposed_action,
                        computed_at = EXCLUDED.computed_at
                    """,
                    (
                        report.usecase_id, run_id, f.finding_id, f.category,
                        f.severity, json.dumps(f.tools_involved, ensure_ascii=False),
                        f.requirement_threatened, f.body, f.evidence,
                        f.proposed_action, now,
                    ),
                )
            conn.commit()

    def load_findings(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> list[ArchitectureFinding]:
        sql = (
            "SELECT finding_id, category, severity, tools_involved, "
            "requirement_threatened, body, evidence, proposed_action "
            "FROM tf_judge_architecture_findings WHERE usecase_id = %s"
        )
        params: list[Any] = [usecase_id]
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            ArchitectureFinding(
                finding_id=r[0],
                category=r[1],
                severity=r[2],
                tools_involved=list(r[3] or []),
                requirement_threatened=r[4] or "",
                body=r[5],
                evidence=r[6] or "",
                proposed_action=r[7] or "none",
            )
            for r in rows
        ]

    def load_report(
        self, usecase_id: str, *, run_id: str | None = None
    ) -> ArchitectureJudgeReport | None:
        """Rehydrate the report metadata + findings (contracts omitted).

        The creator judge consumes only ``findings`` / ``problematic_tools``, so
        the heavy pass-1 contracts are not reconstructed; the latest assessment
        is returned when ``run_id`` is not pinned.
        """
        sql = (
            "SELECT run_id, computed_at, mode, judge_model "
            "FROM tf_judge_architecture_runs WHERE usecase_id = %s"
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
        found_run = row[0] or None
        return ArchitectureJudgeReport(
            usecase_id=usecase_id,
            run_id=found_run,
            computed_at=str(row[1]),
            judge_model=row[3] or "",
            mode=row[2] or "design_time",
            contracts=[],
            findings=self.load_findings(usecase_id, run_id=row[0]),
        )

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        with self._connect() as conn:
            for table in (
                "tf_judge_architecture_runs",
                "tf_judge_architecture_findings",
            ):
                conn.execute(
                    f"UPDATE {table} SET usecase_id = %s WHERE usecase_id = %s",
                    (new_id, old_id),
                )
            conn.commit()


def get_architecture_judge_store(dsn: str) -> ArchitectureJudgeStore:
    """Return a Postgres-backed architecture judge store for the given DSN."""
    return PostgresArchitectureJudgeStore(dsn)
