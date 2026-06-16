"""StaticJudge — turn one task's telemetry into a StaticJudgeResult."""
from __future__ import annotations

from ..metrics.data import TaskRecord
from .llm import JudgeLLM
from .models import StaticJudgeResult, UseCaseSpec
from .prompt import build_user_message, parse_result


class StaticJudge:
    """Judges a single execution trace in isolation, using one LLM completion.

    Stateless and side-effect free: it produces results, it does not persist
    them (the runner owns persistence). This keeps each task judged purely from
    its own telemetry — the property the global statistics rely on.
    """

    def __init__(self, llm: JudgeLLM) -> None:
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def judge_task(
        self, usecase: UseCaseSpec, task: TaskRecord
    ) -> StaticJudgeResult:
        user_message = build_user_message(usecase, task)
        raw = await self._llm.complete(user_message)
        return parse_result(task, self._llm.model, raw)

    async def judge_window(
        self, usecase: UseCaseSpec, tasks: list[TaskRecord]
    ) -> list[StaticJudgeResult]:
        results: list[StaticJudgeResult] = []
        for task in tasks:
            results.append(await self.judge_task(usecase, task))
        return results
