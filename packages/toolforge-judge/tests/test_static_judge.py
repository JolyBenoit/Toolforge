"""Static-judge tests with a fake LLM and an in-memory store (no DB, no API)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from toolforge_core.llm.base import LLMRateLimitError
from toolforge_core.types import TextDelta
from toolforge_judge.metrics.data import SpanRecord, TaskRecord
from toolforge_judge.static import (
    AgentLLMJudge,
    StaticJudge,
    StaticJudgeRunner,
    ToolSpec,
    UseCaseSpec,
    run_usecase,
    unjudged_tasks,
)
from toolforge_judge.static.models import StaticJudgeResult
from toolforge_judge.static.prompt import build_user_message, parse_result
from toolforge_judge.static.store import JudgeStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


# --- fakes -----------------------------------------------------------------


class FakeJudgeLLM:
    model = "fake-judge-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, user_message: str) -> str:
        self.prompts.append(user_message)
        return self.reply


class RecordingStore(JudgeStore):
    def __init__(self) -> None:
        self.saved: list[StaticJudgeResult] = []
        self.contribution_writes: list[tuple[str, str]] = []

    def save_result(self, result: StaticJudgeResult) -> None:
        self.saved.append(result)
        for v in result.span_verdicts:
            self.contribution_writes.append((v.span_id, v.contribution))

    def judged_task_ids(self, usecase_id, *, run_id=None):
        return {r.task_id for r in self.saved}


# --- fixtures --------------------------------------------------------------


def _tool_span(task_id, span_id, tool_id, *, inp=None, offset=0) -> SpanRecord:
    return SpanRecord(
        span_id=span_id, task_id=task_id, run_id="run1", usecase_id="uc1",
        type="tool_call", started_at=_T0 + timedelta(seconds=offset),
        status="ok", tool_id=tool_id, input=inp or {},
    )


def _task(task_id="t1") -> TaskRecord:
    return TaskRecord(
        task_id=task_id, run_id="run1", usecase_id="uc1", status="success",
        started_at=_T0,
        input_timeline=[{"user_turn": 1, "content": "budget of 500"}],
        final_output="done",
        spans=[
            _tool_span(task_id, "s1", "search", inp={"q": "x"}, offset=0),
            _tool_span(task_id, "s2", "book", inp={"budget": 500}, offset=1),
        ],
    )


def _usecase() -> UseCaseSpec:
    return UseCaseSpec(
        usecase_id="uc1", utility="Book trips.", rules="Always confirm budget.",
        tools=[ToolSpec("search", "search the web"), ToolSpec("book", "book a trip")],
    )


_GOOD_REPLY = json.dumps({
    "span_verdicts": [
        {"span_id": "s1", "tool_id": "search", "contribution": "dead",
         "selection_appropriate": False, "param_fidelity": 0.4, "rationale": "unused"},
        {"span_id": "s2", "tool_id": "book", "contribution": "necessary",
         "selection_appropriate": True, "param_fidelity": 1.0, "rationale": "ok"},
    ],
    "tool_notes": [
        {"tool_id": "search", "scores": {"selection_precision": 0.2, "output_quality": 0.5},
         "recommendation": "Restrict search usage; it was dead here.",
         "recommendation_target": "usage"},
        {"tool_id": "book", "scores": {"selection_precision": 1.0, "output_quality": 0.9},
         "recommendation": None, "recommendation_target": "none"},
    ],
})


# --- prompt assembly -------------------------------------------------------


def test_user_message_includes_rules_utility_and_telemetry():
    msg = build_user_message(_usecase(), _task())
    assert "Book trips." in msg
    assert "Always confirm budget." in msg
    assert "s1" in msg and "s2" in msg
    assert "budget" in msg


def test_user_message_handles_user_wait_and_llm_call_spans():
    # Regression: SpanRecord must expose user_turn/user_message/response so the
    # prompt builder doesn't raise AttributeError on non-tool spans.
    task = TaskRecord(
        task_id="t9", run_id="run1", usecase_id="uc1", status="success",
        started_at=_T0,
        spans=[
            SpanRecord(
                span_id="u1", task_id="t9", run_id="run1", usecase_id="uc1",
                type="user_wait", started_at=_T0,
                user_turn=1, user_message="please book a trip",
            ),
            SpanRecord(
                span_id="l1", task_id="t9", run_id="run1", usecase_id="uc1",
                type="llm_call", started_at=_T0, response={"text": "on it"},
            ),
        ],
    )
    msg = build_user_message(_usecase(), task)
    assert "please book a trip" in msg
    assert "on it" in msg


# --- parsing ---------------------------------------------------------------


def test_parse_good_reply():
    result = parse_result(_task(), "fake-judge-1", _GOOD_REPLY)
    assert {v.span_id for v in result.span_verdicts} == {"s1", "s2"}
    assert result.verdict_for("s1").contribution == "dead"
    assert result.verdict_for("s2").param_fidelity == pytest.approx(1.0)


def test_parse_tolerates_prose_and_fences():
    wrapped = "Sure! Here you go:\n```json\n" + _GOOD_REPLY + "\n```\nHope that helps."
    result = parse_result(_task(), "m", wrapped)
    assert len(result.span_verdicts) == 2


def test_parse_clamps_and_defaults_invalid_fields():
    reply = json.dumps({
        "span_verdicts": [
            {"span_id": "s1", "tool_id": "search", "contribution": "bogus",
             "selection_appropriate": "yes", "param_fidelity": 5.0},
        ],
        "tool_notes": [],
    })
    v = parse_result(_task(), "m", reply).verdict_for("s1")
    assert v.contribution == "necessary"       # invalid -> safe default
    assert v.param_fidelity == 1.0             # clamped to [0,1]
    assert v.selection_appropriate is True


# --- recommendations only when necessary -----------------------------------


def test_recommendations_only_present_when_needed():
    result = parse_result(_task(), "m", _GOOD_REPLY)
    recs = {n.tool_id for n in result.recommendations}
    assert recs == {"search"}                  # 'book' had no recommendation
    assert result.note_for("book").recommendation is None
    # tool notes are independent rows, one per tool
    assert {n.tool_id for n in result.tool_notes} == {"search", "book"}


# --- AgentLLMJudge rate-limit backoff --------------------------------------


class _FlakyClient:
    """An LLMClient that rejects the first ``fail_times`` calls with a 429."""

    def __init__(self, fail_times: int, reply: str = '{"tool_notes": []}') -> None:
        self.fail_times = fail_times
        self.reply = reply
        self.attempts = 0

    async def stream(self, messages, *, system, tools, model, max_tokens, temperature):  # noqa: ANN001, ANN201
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise LLMRateLimitError("429: rate limit reached")
        yield TextDelta(text=self.reply)


def _agent_judge(client: _FlakyClient) -> AgentLLMJudge:
    return AgentLLMJudge(
        client=client, model="m", system_prompt="s", max_tokens=10, temperature=0.0,
    )


async def test_agent_llm_judge_retries_then_succeeds(monkeypatch):
    # Neutralise the real backoff sleeps so the test is instant.
    monkeypatch.setattr(
        "toolforge_judge.static.llm.asyncio.sleep",
        _instant_sleep,
    )
    client = _FlakyClient(fail_times=2)
    out = await _agent_judge(client).complete("hi")
    assert out == '{"tool_notes": []}'
    assert client.attempts == 3  # two 429s, third succeeds


async def test_agent_llm_judge_reraises_after_backoff_exhausted(monkeypatch):
    monkeypatch.setattr(
        "toolforge_judge.static.llm.asyncio.sleep",
        _instant_sleep,
    )
    client = _FlakyClient(fail_times=99)
    with pytest.raises(LLMRateLimitError):
        await _agent_judge(client).complete("hi")
    assert client.attempts == 6  # initial try + 5 backoff retries


async def _instant_sleep(*_args, **_kwargs) -> None:
    return None


# --- judge + runner end to end (async) -------------------------------------


async def test_judge_task_calls_llm_and_parses():
    judge = StaticJudge(FakeJudgeLLM(_GOOD_REPLY))
    result = await judge.judge_task(_usecase(), _task())
    assert result.judge_model == "fake-judge-1"
    assert result.verdict_for("s1").contribution == "dead"


async def test_runner_persists_and_writes_back_contribution():
    store = RecordingStore()
    runner = StaticJudgeRunner(StaticJudge(FakeJudgeLLM(_GOOD_REPLY)), _usecase(),
                               store=store)
    results = await runner.run_tasks([_task("t1"), _task("t2")])
    assert len(results) == 2
    assert len(store.saved) == 2
    # contribution write-back reaches the metric layer's column
    assert ("s1", "dead") in store.contribution_writes
    assert ("s2", "necessary") in store.contribution_writes


# --- whole-use-case pass across pipeline versions --------------------------


def _task_in(task_id: str, run_id: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id, run_id=run_id, usecase_id="uc1", status="success",
        started_at=_T0, spans=[_tool_span(task_id, f"{task_id}s", "search")],
    )


class FakeReader:
    """Minimal TelemetryReader stand-in: serves a fixed task list."""

    def __init__(self, tasks: list[TaskRecord]) -> None:
        self._tasks = tasks

    def load_tasks(self, usecase_id, *, run_id=None, run_ids=None, limit=None):
        return list(self._tasks)


class FakeRegistry:
    """Duck-typed registry: records which (usecase, run) specs were built."""

    def __init__(self) -> None:
        self.spec_runs: list[str] = []

    def get_usecase(self, usecase_id):
        return type("UC", (), {"prompt": "Book trips."})()

    def get_active_tools(self, usecase_id, run_id):
        self.spec_runs.append(run_id)
        return [type("T", (), {"name": "search", "description": "search"})()]

    def get_tool_schema(self, usecase_id, run_id, name):
        return {"type": "object"}

    def get_consumer_prompt(self, usecase_id):
        return "Always confirm budget."


async def test_run_usecase_spans_all_runs_and_reports_progress():
    store = RecordingStore()
    reader = FakeReader([
        _task_in("t1", "runA"), _task_in("t2", "runA"), _task_in("t3", "runB"),
    ])
    registry = FakeRegistry()
    seen: list[tuple[int, int]] = []

    results = await run_usecase(
        StaticJudge(FakeJudgeLLM(_GOOD_REPLY)), registry, reader, "uc1",
        store=store, progress=lambda done, total: seen.append((done, total)),
    )

    assert len(results) == 3
    assert len(store.saved) == 3
    # a fresh spec was built per pipeline version, not per task
    assert set(registry.spec_runs) == {"runA", "runB"}
    # progress is cumulative across runs and ends at total
    assert seen[-1] == (3, 3)
    assert [d for d, _ in seen] == [1, 2, 3]


async def test_run_usecase_isolates_a_failing_task():
    class FlakyStore(RecordingStore):
        def save_result(self, result):
            if result.task_id == "t2":
                raise RuntimeError("boom")
            super().save_result(result)

    store = FlakyStore()
    reader = FakeReader([
        _task_in("t1", "runA"), _task_in("t2", "runA"), _task_in("t3", "runB"),
    ])
    errors: list[tuple[str, str]] = []

    results = await run_usecase(
        StaticJudge(FakeJudgeLLM(_GOOD_REPLY)), FakeRegistry(), reader, "uc1",
        store=store, on_error=lambda task, exc: errors.append((task.task_id, str(exc))),
    )

    # the failing task is skipped; the other two still succeed
    assert {r.task_id for r in results} == {"t1", "t3"}
    assert errors == [("t2", "boom")]


async def test_run_usecase_respects_concurrency_bound():
    import asyncio

    inflight = 0
    peak = 0

    class CountingLLM(FakeJudgeLLM):
        async def complete(self, user_message: str) -> str:
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0)  # let other coroutines pile up
            inflight -= 1
            return await super().complete(user_message)

    reader = FakeReader([_task_in(f"t{i}", "runA") for i in range(10)])
    await run_usecase(
        StaticJudge(CountingLLM(_GOOD_REPLY)), FakeRegistry(), reader, "uc1",
        store=RecordingStore(), max_concurrency=3,
    )
    assert 1 < peak <= 3  # bounded, but genuinely parallel


async def test_run_usecase_skips_already_judged_tasks():
    store = RecordingStore()
    # pre-judge t1 so the incremental pass leaves it untouched
    store.saved.append(parse_result(_task_in("t1", "runA"), "m", _GOOD_REPLY))
    reader = FakeReader([_task_in("t1", "runA"), _task_in("t2", "runA")])

    results = await run_usecase(
        StaticJudge(FakeJudgeLLM(_GOOD_REPLY)), FakeRegistry(), reader, "uc1",
        store=store,
    )

    assert {r.task_id for r in results} == {"t2"}
    assert unjudged_tasks(reader, store, "uc1") == []  # t2 now recorded too
