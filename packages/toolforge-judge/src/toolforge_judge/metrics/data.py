"""Data-access layer for the metric engine.

The metrics never touch Postgres directly. They operate on plain in-memory
records (:class:`TaskRecord` / :class:`SpanRecord`) so every metric is unit
testable without a live database. :class:`TelemetryReader` is the only piece
that knows the ``tf_prod_*`` schema; it maps rows into records and slices them
into a short/long :class:`MetricWindow`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .env import MetricEnv

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _as_dt(value: Any) -> datetime:
    """Coerce an ISO string or datetime into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _opt_dt(value: Any) -> datetime | None:
    return None if value is None else _as_dt(value)


@dataclass
class SpanRecord:
    """One span (llm_call | tool_call | user_wait) as needed by metrics."""

    span_id: str
    task_id: str
    run_id: str
    usecase_id: str
    type: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: str | None = None
    parent_spans: list[str] = field(default_factory=list)
    llm_turn_index: int | None = None
    call_index_in_turn: int | None = None

    # tool_call
    tool_id: str | None = None
    tool_version: int | None = None
    input: dict[str, Any] | None = None
    output: Any = None
    retries: list[dict[str, Any]] = field(default_factory=list)
    nested_llm_calls: list[dict[str, Any]] = field(default_factory=list)

    # llm_call
    response: Any = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    # user_wait
    user_turn: int | None = None
    user_message: str | None = None

    # judge-filled (None until the backward pass runs)
    contribution: str | None = None

    @property
    def is_tool_call(self) -> bool:
        return self.type == "tool_call"

    @property
    def retried(self) -> bool:
        return len(self.retries) > 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SpanRecord:
        return cls(
            span_id=row["span_id"],
            task_id=row["task_id"],
            run_id=row["run_id"],
            usecase_id=row["usecase_id"],
            type=row["type"],
            started_at=_as_dt(row["started_at"]),
            ended_at=_opt_dt(row.get("ended_at")),
            duration_ms=row.get("duration_ms"),
            status=row.get("status"),
            parent_spans=list(row.get("parent_spans") or []),
            llm_turn_index=row.get("llm_turn_index"),
            call_index_in_turn=row.get("call_index_in_turn"),
            tool_id=row.get("tool_id"),
            tool_version=row.get("tool_version"),
            input=row.get("input"),
            output=row.get("output"),
            retries=list(row.get("retries") or []),
            nested_llm_calls=list(row.get("nested_llm_calls") or []),
            response=row.get("response"),
            tokens_in=row.get("tokens_in"),
            tokens_out=row.get("tokens_out"),
            user_turn=row.get("user_turn"),
            user_message=row.get("user_message"),
            contribution=row.get("contribution"),
        )


@dataclass
class TaskRecord:
    """One complete conversation session plus its spans and DAG."""

    task_id: str
    run_id: str
    usecase_id: str
    status: str
    started_at: datetime
    ended_at: datetime | None = None
    input_timeline: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    user_feedback: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    dag: dict[str, Any] | None = None
    spans: list[SpanRecord] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def tool_calls(self) -> list[SpanRecord]:
        return [s for s in self.spans if s.is_tool_call]

    def tool_calls_for(self, tool_id: str) -> list[SpanRecord]:
        return [s for s in self.spans if s.is_tool_call and s.tool_id == tool_id]

    @classmethod
    def from_row(cls, row: dict[str, Any], spans: list[SpanRecord]) -> TaskRecord:
        return cls(
            task_id=row["task_id"],
            run_id=row["run_id"],
            usecase_id=row["usecase_id"],
            status=row["status"],
            started_at=_as_dt(row["started_at"]),
            ended_at=_opt_dt(row.get("ended_at")),
            input_timeline=list(row.get("input_timeline") or []),
            final_output=row.get("final_output"),
            user_feedback=row.get("user_feedback"),
            cost=row.get("cost"),
            dag=row.get("dag"),
            spans=sorted(spans, key=lambda s: s.started_at),
        )


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def _selected_runs(
    run_id: str | None, run_ids: list[str] | None
) -> set[str] | None:
    """Normalise the ``run_id`` / ``run_ids`` selection into a set (or None)."""
    if run_ids:
        return set(run_ids)
    if run_id is not None:
        return {run_id}
    return None


def _run_label(run_id: str | None, run_ids: list[str] | None) -> str | None:
    """A stable identifier for a run selection (used as the report's key).

    One version → its id; several → the sorted ids joined with ``+``; none →
    None (the "all runs" slot).
    """
    selected = _selected_runs(run_id, run_ids)
    if selected is None:
        return None
    ordered = sorted(selected)
    return ordered[0] if len(ordered) == 1 else "+".join(ordered)


@dataclass
class MetricWindow:
    """A short and long slice of tasks for SPC-style comparison.

    ``short`` and ``long`` are both ordered newest-first. ``long`` is a superset
    baseline (the current pipeline version, or a 100+ task slide); ``short`` is
    the most recent ``env.short_window`` tasks. The two overlap by design — the
    short window is the leading edge of the long one.
    """

    usecase_id: str
    short: list[TaskRecord]
    long: list[TaskRecord]
    env: MetricEnv
    run_id: str | None = None
    run_ids: list[str] | None = None
    # Judge-supplied per-tool scores, attached after construction by the caller
    # that has the judge notes (TUI / dynamic judge). ``judge_scores`` maps
    # ``(tool_id, score_key)`` → ``(mean_value, n)``; ``judged_tools`` is every
    # tool the judge has assessed (so a scored-but-valueless metric can still be
    # reported as evaluated rather than pending). Both ``None`` = no judge pass.
    judge_scores: dict[tuple[str, str], tuple[float, int]] | None = None
    judged_tools: set[str] | None = None

    @property
    def tool_ids(self) -> list[str]:
        """All distinct tool_ids seen anywhere in the long window."""
        seen: dict[str, None] = {}
        for task in self.long:
            for sp in task.tool_calls:
                if sp.tool_id is not None:
                    seen.setdefault(sp.tool_id, None)
        return list(seen)

    @classmethod
    def from_tasks(
        cls,
        usecase_id: str,
        tasks: Iterable[TaskRecord],
        env: MetricEnv,
        *,
        run_id: str | None = None,
        run_ids: list[str] | None = None,
    ) -> MetricWindow:
        """Slice an ordered (or unordered) task list into short/long windows.

        The long window is restricted to a chosen set of pipeline versions:
        ``run_ids`` selects several versions at once, ``run_id`` a single one
        (a convenience for the common case). With neither, it slides over the
        latest ``env.long_window`` tasks across every version. ``run_id`` then
        carries a stable label for the selection (the lone id, or the joined
        set) so a re-run over the same versions overwrites its stored report.
        """
        selected = _selected_runs(run_id, run_ids)
        ordered = sorted(tasks, key=lambda t: t.started_at, reverse=True)
        if selected is not None:
            long = [t for t in ordered if t.run_id in selected][: env.long_window]
        else:
            long = ordered[: env.long_window]
        short = long[: env.short_window]
        return cls(
            usecase_id=usecase_id,
            short=short,
            long=long,
            env=env,
            run_id=_run_label(run_id, run_ids),
            run_ids=sorted(selected) if selected is not None else None,
        )


# ---------------------------------------------------------------------------
# Postgres reader
# ---------------------------------------------------------------------------


class TelemetryReader:
    """Reads production telemetry from the ``tf_prod_*`` tables into records.

    Lazy-imports ``psycopg`` (the ``postgres`` extra) so importing the metric
    package never requires a database driver.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "psycopg is not installed. "
                "Add the postgres extra: uv add 'toolforge-judge[postgres]'"
            ) from exc
        self._dsn = dsn

    def _connect(self) -> Any:
        import psycopg  # type: ignore[import-untyped]

        return psycopg.connect(self._dsn, row_factory=_dict_row())

    def load_tasks(
        self,
        usecase_id: str,
        *,
        run_id: str | None = None,
        run_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        """Load tasks (newest first) with their spans for a use case.

        ``run_ids`` restricts to a set of pipeline versions; ``run_id`` to a
        single one (a convenience). With neither, every version is included.
        """
        params: list[Any] = [usecase_id]
        sql = "SELECT * FROM tf_prod_tasks WHERE usecase_id = %s"
        if run_ids:
            sql += " AND run_id = ANY(%s)"
            params.append(list(run_ids))
        elif run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        sql += " ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self._connect() as conn:
            task_rows = conn.execute(sql, params).fetchall()
            if not task_rows:
                return []
            task_ids = [r["task_id"] for r in task_rows]
            span_rows = conn.execute(
                "SELECT * FROM tf_prod_spans WHERE task_id = ANY(%s)",
                (task_ids,),
            ).fetchall()

        spans_by_task: dict[str, list[SpanRecord]] = {}
        for sr in span_rows:
            spans_by_task.setdefault(sr["task_id"], []).append(SpanRecord.from_row(sr))
        return [TaskRecord.from_row(r, spans_by_task.get(r["task_id"], [])) for r in task_rows]

    def load_task(self, task_id: str) -> TaskRecord | None:
        """Load a single task with its spans (for a detail / timeline view)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tf_prod_tasks WHERE task_id = %s", (task_id,)
            ).fetchone()
            if row is None:
                return None
            span_rows = conn.execute(
                "SELECT * FROM tf_prod_spans WHERE task_id = %s", (task_id,)
            ).fetchall()
        spans = [SpanRecord.from_row(sr) for sr in span_rows]
        return TaskRecord.from_row(row, spans)

    def load_window(
        self,
        usecase_id: str,
        env: MetricEnv,
        *,
        run_id: str | None = None,
        run_ids: list[str] | None = None,
    ) -> MetricWindow:
        """Load enough tasks to populate the short and long windows."""
        tasks = self.load_tasks(
            usecase_id, run_id=run_id, run_ids=run_ids, limit=env.long_window
        )
        return MetricWindow.from_tasks(
            usecase_id, tasks, env, run_id=run_id, run_ids=run_ids
        )

    def count_by_status(
        self,
        usecase_id: str,
        *,
        run_id: str | None = None,
    ) -> dict[str, int]:
        """Count *all* recorded tasks grouped by status (not window-capped).

        Unlike :meth:`load_window`, this scans the whole history so the TUI can
        report the true total number of runs / successes / failures, not just
        the last ``long_window`` tasks.
        """
        params: list[Any] = [usecase_id]
        sql = "SELECT status, COUNT(*) AS n FROM tf_prod_tasks WHERE usecase_id = %s"
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        sql += " GROUP BY status"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def list_task_summaries(
        self,
        usecase_id: str,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """One lightweight summary row per task, newest first (no span scan).

        Tokens and tool counts come from the task-level ``cost`` aggregate
        (:class:`TaskCost`), so the Runs list stays a single ``tf_prod_tasks``
        query. ``user_feedback`` is included for the detail summary.
        """
        params: list[Any] = [usecase_id]
        sql = (
            "SELECT task_id, run_id, status, started_at, ended_at, "
            "cost, user_feedback "
            "FROM tf_prod_tasks WHERE usecase_id = %s"
        )
        if run_id is not None:
            sql += " AND run_id = %s"
            params.append(run_id)
        sql += " ORDER BY started_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        summaries: list[dict[str, Any]] = []
        for r in rows:
            cost = r.get("cost") or {}
            tokens = (
                int(cost.get("agent_tokens_in", 0))
                + int(cost.get("agent_tokens_out", 0))
                + int(cost.get("tool_tokens_in", 0))
                + int(cost.get("tool_tokens_out", 0))
            )
            summaries.append(
                {
                    "task_id": r["task_id"],
                    "run_id": r["run_id"],
                    "status": r["status"],
                    "started_at": _as_dt(r["started_at"]),
                    "ended_at": _opt_dt(r.get("ended_at")),
                    "tool_calls": int(cost.get("tool_calls", 0)),
                    "tokens": tokens,
                    "latency_ms": float(cost.get("latency_ms", 0.0)),
                    "user_feedback": r.get("user_feedback"),
                }
            )
        return summaries

    def count_tasks_by_run(self, usecase_id: str) -> list[tuple[str, int]]:
        """List the pipeline versions of a use case with their task counts.

        Ordered most-recent-first (by latest task), so the TUI can offer the
        set of runs to feed the dynamic judge.
        """
        sql = (
            "SELECT run_id, COUNT(*) AS n, MAX(started_at) AS last "
            "FROM tf_prod_tasks WHERE usecase_id = %s "
            "GROUP BY run_id ORDER BY last DESC"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, [usecase_id]).fetchall()
        return [(r["run_id"], int(r["n"])) for r in rows]


def _dict_row() -> Any:
    """Return psycopg's dict_row factory (imported lazily)."""
    from psycopg.rows import dict_row  # type: ignore[import-untyped]

    return dict_row
