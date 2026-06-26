"""ProductionTelemetryStore — protocol and built-in implementations.

Only used for ``in_production`` runs.  Draft and validated runs continue to
use the existing ``TelemetryStore`` / ``JSONLTelemetryStore`` from the parent
package.

Implementations
---------------
NullProductionTelemetryStore  — no-op, for tests.
PostgresProductionTelemetryStore  — full Postgres backend (lazy import via
    ``_pg_store``; requires the ``postgres`` extra).
"""
from __future__ import annotations

from .models import (
    DAG,
    InputTimelineEntry,
    PipelineSpec,
    Span,
    TaskCost,
    TaskStatus,
    UserFeedback,
)


# ---------------------------------------------------------------------------
# Base class (protocol)
# ---------------------------------------------------------------------------


class ProductionTelemetryStore:
    """Interface for production telemetry.

    All write operations are idempotent on the primary key so callers can
    retry safely on transient failures.
    """

    # --- pipeline lifecycle -------------------------------------------------

    def record_pipeline_spec(self, spec: PipelineSpec) -> None:
        """Persist an immutable pipeline snapshot at promote-to-production time."""
        raise NotImplementedError

    # --- task lifecycle -----------------------------------------------------

    def open_task(
        self,
        task_id: str,
        run_id: str,
        usecase_id: str,
        user_session_id: str,
        started_at: str,
    ) -> None:
        """Create a task record with status='running'."""
        raise NotImplementedError

    def append_input_entry(self, task_id: str, entry: InputTimelineEntry) -> None:
        """Append one user message to the task's input timeline."""
        raise NotImplementedError

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
        """Finalise a task with its outcome, aggregated cost, and DAG."""
        raise NotImplementedError

    def record_user_feedback(self, task_id: str, feedback: UserFeedback) -> None:
        """Attach explicit user feedback to a completed task."""
        raise NotImplementedError

    # --- span recording -----------------------------------------------------

    def record_span(self, span: Span) -> None:
        """Persist a single span (llm_call, tool_call, or user_wait)."""
        raise NotImplementedError

    # --- task maintenance (TUI Runs tab) -----------------------------------

    def delete_tasks(self, task_ids: list[str]) -> int:
        """Hard-delete the given tasks and their spans. Returns rows removed.

        Spans are removed before tasks to respect the foreign key. Default
        no-op (only the Postgres backend persists rows).
        """
        return 0

    def set_task_status(self, task_ids: list[str], status: TaskStatus) -> int:
        """Overwrite the status of the given tasks. Returns rows updated.

        Default no-op; only the Postgres backend persists rows.
        """
        return 0

    # --- maintenance --------------------------------------------------------

    def rename_usecase(self, old_id: str, new_id: str) -> None:
        """Repoint every persisted row from ``old_id`` to ``new_id``.

        Default no-op; only the Postgres backend persists ``usecase_id``.
        """


# ---------------------------------------------------------------------------
# NullProductionTelemetryStore — no-op for tests
# ---------------------------------------------------------------------------


class NullProductionTelemetryStore(ProductionTelemetryStore):
    def record_pipeline_spec(self, spec: PipelineSpec) -> None:
        pass

    def open_task(
        self,
        task_id: str,
        run_id: str,
        usecase_id: str,
        user_session_id: str,
        started_at: str,
    ) -> None:
        pass

    def append_input_entry(self, task_id: str, entry: InputTimelineEntry) -> None:
        pass

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
        pass

    def record_user_feedback(self, task_id: str, feedback: UserFeedback) -> None:
        pass

    def record_span(self, span: Span) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_production_store(pg_dsn: str) -> ProductionTelemetryStore:
    """Return a ``PostgresProductionTelemetryStore`` for the given DSN.

    Raises ``ImportError`` if the ``postgres`` extra is not installed.
    """
    from ._pg_store import PostgresProductionTelemetryStore

    return PostgresProductionTelemetryStore(pg_dsn)
