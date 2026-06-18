"""StaticJudgeRunner — drive the static judge over a use case's tasks.

Ties together the read side (``TelemetryReader``), the judge, and the store.
Supports running over a single task, an explicit list, or a window for a use
case / pipeline version, skipping already-judged tasks so iterations chain
cleanly.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from ..metrics.data import TaskRecord, TelemetryReader
from .judge import StaticJudge
from .models import StaticJudgeResult, ToolSpec, UseCaseSpec
from .store import JudgeStore, NullJudgeStore

# Called after each task is judged, with (judged_so_far, total) — lets the TUI
# render incremental progress over a long window.
ProgressFn = Callable[[int, int], None]


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
        self,
        tasks: list[TaskRecord],
        *,
        persist: bool = True,
        on_result: Callable[[StaticJudgeResult], None] | None = None,
    ) -> list[StaticJudgeResult]:
        results: list[StaticJudgeResult] = []
        for task in tasks:
            result = await self._judge.judge_task(self._usecase, task)
            if persist:
                self._store.save_result(result)
            results.append(result)
            if on_result is not None:
                on_result(result)
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


def unjudged_tasks(
    reader: TelemetryReader,
    store: JudgeStore,
    usecase_id: str,
) -> list[TaskRecord]:
    """Tasks of a use case not yet covered by the static judge (newest first)."""
    done = store.judged_task_ids(usecase_id)
    return [t for t in reader.load_tasks(usecase_id) if t.task_id not in done]


async def run_usecase(
    judge: StaticJudge,
    registry: _RegistryLike,
    reader: TelemetryReader,
    usecase_id: str,
    *,
    store: JudgeStore | None = None,
    skip_judged: bool = True,
    persist: bool = True,
    progress: ProgressFn | None = None,
    on_error: Callable[[TaskRecord, Exception], None] | None = None,
    max_concurrency: int = 6,
) -> list[StaticJudgeResult]:
    """Judge every task of a use case, across all its pipeline versions.

    Each task is judged against the right version of the tool specs (a fresh
    :class:`UseCaseSpec` per ``run_id``, built once and reused). With
    ``skip_judged`` (default), already-judged tasks are left untouched, so the
    pass is incremental and chains cleanly across iterations.

    Tasks are judged **concurrently** — the static judge is stateless and each
    task is an independent, network-bound LLM call, so up to ``max_concurrency``
    are kept in flight at once (the bound protects against API rate limits).
    Persistence is offloaded to a thread so DB writes never stall the loop. A
    task that raises is isolated: it is reported via ``on_error`` and skipped,
    the rest of the batch still completes. Returns the successful results.
    """
    store = store or NullJudgeStore()
    tasks = reader.load_tasks(usecase_id)
    if skip_judged:
        done = store.judged_task_ids(usecase_id)
        tasks = [t for t in tasks if t.task_id not in done]
    if not tasks:
        return []

    # One spec per pipeline version, built once and shared by its tasks.
    specs = {
        run_id: build_usecase_spec(registry, usecase_id, run_id)
        for run_id in {t.run_id for t in tasks}
    }

    sem = asyncio.Semaphore(max_concurrency)
    total = len(tasks)
    seen = 0

    async def _judge_one(task: TaskRecord) -> StaticJudgeResult | None:
        nonlocal seen
        async with sem:
            try:
                result = await judge.judge_task(specs[task.run_id], task)
                if persist:
                    await asyncio.to_thread(store.save_result, result)
            except Exception as exc:  # noqa: BLE001 - isolate one task's failure
                if on_error is not None:
                    on_error(task, exc)
                return None
        # No await between read and write: atomic in a single-threaded loop.
        seen += 1
        if progress is not None:
            progress(seen, total)
        return result

    gathered = await asyncio.gather(*(_judge_one(t) for t in tasks))
    return [r for r in gathered if r is not None]
