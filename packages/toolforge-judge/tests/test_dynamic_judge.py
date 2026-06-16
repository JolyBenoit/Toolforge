"""Dynamic-judge tests over in-memory notes and tasks (no DB, no API)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from toolforge_judge.dynamic import (
    DynamicJudge,
    aggregate_tool_notes,
    compute_structural_stability,
)
from toolforge_judge.metrics import MetricEngine, MetricEnv, MetricWindow
from toolforge_judge.metrics.data import SpanRecord, TaskRecord
from toolforge_judge.static.store import ToolNoteRecord

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _note(task_id, tool_id, scores, rec=None, target="none") -> ToolNoteRecord:
    return ToolNoteRecord(task_id, tool_id, scores, rec, target)


def _span(task_id, span_id, tool_id, *, inp=None, output=None, offset=0) -> SpanRecord:
    return SpanRecord(
        span_id=span_id, task_id=task_id, run_id="run1", usecase_id="uc1",
        type="tool_call", started_at=_T0 + timedelta(seconds=offset),
        status="ok", tool_id=tool_id, input=inp or {}, output=output,
    )


def _task(task_id, spans, *, offset=0) -> TaskRecord:
    return TaskRecord(
        task_id=task_id, run_id="run1", usecase_id="uc1", status="success",
        started_at=_T0 + timedelta(minutes=offset), spans=spans,
    )


def _window(tasks, env=None) -> MetricWindow:
    return MetricWindow.from_tasks("uc1", tasks, env or MetricEnv(min_samples=1))


# --- aggregation -----------------------------------------------------------


def test_aggregate_means_and_recommendation_rate():
    env = MetricEnv(selection_precision_min=0.8)
    records = [
        _note("t1", "search", {"selection_precision": 0.4}, rec="restrict it", target="usage"),
        _note("t2", "search", {"selection_precision": 0.6}),
        _note("t1", "book", {"selection_precision": 1.0}),
    ]
    notes = {n.tool_id: n for n in aggregate_tool_notes(records, env)}
    search = notes["search"]
    assert search.n_tasks == 2
    assert search.mean_scores["selection_precision"] == pytest.approx(0.5)
    assert search.recommendation_rate == pytest.approx(0.5)
    assert search.dominant_target == "usage"
    assert "selection_precision" in search.breaches      # 0.5 < 0.8
    # 'book' is healthy and independent of 'search'
    assert notes["book"].breaches == []
    assert notes["book"].recommendation_rate == 0.0


# --- structural stability --------------------------------------------------


def test_structural_stability_penalises_premature_calls():
    env = MetricEnv(min_samples=1, structural_stability_min=0.9)
    # 'book' is called prematurely (later enriched) in every task -> unstable.
    tasks = []
    for i in range(3):
        tasks.append(_task(f"t{i}", [
            _span(f"t{i}", f"a{i}", "book", inp={"city": "P"}, offset=0),
            _span(f"t{i}", f"b{i}", "book", inp={"city": "P", "budget": 5}, offset=1),
        ], offset=i))
    report = MetricEngine().compute(_window(tasks, env))
    stab = compute_structural_stability(report, env)
    book = next(t for t in stab.per_tool if t.tool_id == "book")
    assert book.premature_call_rate is not None and book.premature_call_rate > 0
    assert book.stability_score < 1.0
    assert stab.mean_structural_stability is not None
    assert stab.breached is True                          # below 0.9


def test_structural_stability_healthy_pipeline_not_breached():
    env = MetricEnv(min_samples=1, structural_stability_min=0.7)
    tasks = [_task("t1", [_span("t1", "s1", "search", inp={"q": "x"})])]
    report = MetricEngine().compute(_window(tasks, env))
    stab = compute_structural_stability(report, env)
    assert stab.mean_structural_stability == pytest.approx(1.0)
    assert stab.breached is False


# --- full report + optional LLM diagnosis ----------------------------------


class FakeLLM:
    model = "fake-dyn-1"

    def __init__(self):
        self.calls = 0

    async def complete(self, user_message: str) -> str:
        self.calls += 1
        assert "structural_stability" in user_message
        return "search is the weak point; param fidelity low."


async def test_assess_builds_full_report_without_llm():
    env = MetricEnv(min_samples=1)
    tasks = [_task("t1", [_span("t1", "s1", "search", inp={"q": "x"})])]
    records = [_note("t1", "search", {"selection_precision": 0.9})]
    report = await DynamicJudge().assess(_window(tasks, env), records)
    assert report.n_tasks == 1
    assert report.note_for("search").mean_scores["selection_precision"] == pytest.approx(0.9)
    assert report.structural_stability.mean_structural_stability is not None
    assert report.diagnosis is None                       # no LLM provided
    assert isinstance(report.to_dict()["structural_stability"], dict)


async def test_assess_with_llm_writes_diagnosis():
    env = MetricEnv(min_samples=1)
    tasks = [_task("t1", [_span("t1", "s1", "search", inp={"q": "x"})])]
    llm = FakeLLM()
    report = await DynamicJudge(llm=llm).assess(_window(tasks, env), [])
    assert llm.calls == 1
    assert report.diagnosis is not None
