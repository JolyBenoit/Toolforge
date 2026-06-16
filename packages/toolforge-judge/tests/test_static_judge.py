"""Static-judge tests with a fake LLM and an in-memory store (no DB, no API)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from toolforge_judge.metrics.data import SpanRecord, TaskRecord
from toolforge_judge.static import (
    StaticJudge,
    StaticJudgeRunner,
    ToolSpec,
    UseCaseSpec,
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
