"""StaticJudgeRunner — drive the static judge over a use case's tasks.

Ties together the read side (``TelemetryReader``), the judge, and the store.
Supports running over a single task, an explicit list, or a window for a use
case / pipeline version, skipping already-judged tasks so iterations chain
cleanly.
"""
from __future__ import annotations

from typing import Any, Protocol

from ..metrics.data import TaskRecord, TelemetryReader
from .judge import StaticJudge
from .models import StaticJudgeResult, ToolSpec, UseCaseSpec
from .store import JudgeStore, NullJudgeStore


class StaticJudgeRunner:
    def __init__(
        self,
        judge: StaticJudge,
        usecase: UseCaseSpec,
        *,
        store: JudgeStore | None = None,
    ) -> None:
        self._judge = judge
        self._usecase = usecase
        self._store = store or NullJudgeStore()

    async def run_tasks(
        self, tasks: list[TaskRecord], *, persist: bool = True
    ) -> list[StaticJudgeResult]:
        results: list[StaticJudgeResult] = []
        for task in tasks:
            result = await self._judge.judge_task(self._usecase, task)
            if persist:
                self._store.save_result(result)
            results.append(result)
        return results

    async def run_window(
        self,
        reader: TelemetryReader,
        *,
        run_id: str | None = None,
        limit: int | None = None,
        skip_judged: bool = True,
        persist: bool = True,
    ) -> list[StaticJudgeResult]:
        """Judge the most recent tasks of the use case (a window of runs)."""
        tasks = reader.load_tasks(
            self._usecase.usecase_id, run_id=run_id, limit=limit
        )
        if skip_judged:
            done = self._store.judged_task_ids(
                self._usecase.usecase_id, run_id=run_id
            )
            tasks = [t for t in tasks if t.task_id not in done]
        return await self.run_tasks(tasks, persist=persist)


# ---------------------------------------------------------------------------
# UseCaseSpec adapter — duck-typed registry, like telemetry's builder
# ---------------------------------------------------------------------------


class _RegistryLike(Protocol):
    def get_usecase(self, usecase_id: str) -> Any: ...
    def get_active_tools(self, usecase_id: str, run_id: str) -> list[Any]: ...
    def get_tool_schema(self, usecase_id: str, run_id: str, name: str) -> dict[str, Any]: ...
    def get_consumer_prompt(self, usecase_id: str) -> str | None: ...


def build_usecase_spec(
    registry: _RegistryLike, usecase_id: str, run_id: str
) -> UseCaseSpec:
    """Assemble a :class:`UseCaseSpec` from the registry's active tools.

    ``utility`` = the use-case prompt (what the pipeline is for); ``rules`` = the
    consumer system prompt (how it must behave). Kept duck-typed so the judge
    package never hard-depends on ``toolforge-registry``.
    """
    usecase = registry.get_usecase(usecase_id)
    tools = [
        ToolSpec(
            tool_id=t.name,
            description=t.description,
            schema=registry.get_tool_schema(usecase_id, run_id, t.name),
        )
        for t in registry.get_active_tools(usecase_id, run_id)
    ]
    return UseCaseSpec(
        usecase_id=usecase_id,
        utility=getattr(usecase, "prompt", ""),
        rules=registry.get_consumer_prompt(usecase_id) or "",
        tools=tools,
    )
