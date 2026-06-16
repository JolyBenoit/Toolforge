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
    ) -> MetricWindow:
        """Slice an ordered (or unordered) task list into short/long windows.

        If ``run_id`` is given, the long window is restricted to that pipeline
        version; otherwise it slides over the latest ``env.long_window`` tasks.
        """
        ordered = sorted(tasks, key=lambda t: t.started_at, reverse=True)
        if run_id is not None:
            long = [t for t in ordered if t.run_id == run_id][: env.long_window]
        else:
            long = ordered[: env.long_window]
        short = long[: env.short_window]
        return cls(usecase_id=usecase_id, short=short, long=long, env=env, run_id=run_id)


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
        limit: int | None = None,
    ) -> list[TaskRecord]:
        """Load tasks (newest first) with their spans for a use case."""
        params: list[Any] = [usecase_id]
        sql = "SELECT * FROM tf_prod_tasks WHERE usecase_id = %s"
        if run_id is not None:
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

    def load_window(
        self,
        usecase_id: str,
        env: MetricEnv,
        *,
        run_id: str | None = None,
    ) -> MetricWindow:
        """Load enough tasks to populate the short and long windows."""
        tasks = self.load_tasks(usecase_id, run_id=run_id, limit=env.long_window)
        return MetricWindow.from_tasks(usecase_id, tasks, env, run_id=run_id)


def _dict_row() -> Any:
    """Return psycopg's dict_row factory (imported lazily)."""
    from psycopg.rows import dict_row  # type: ignore[import-untyped]

    return dict_row
