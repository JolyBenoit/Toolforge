"""Metric-engine tests over in-memory records (no database required)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from toolforge_judge.metrics import (
    MetricEngine,
    MetricEnv,
    MetricWindow,
    SpanRecord,
    TaskRecord,
    version_delta,
)
from toolforge_judge.metrics.contribution import RedundancyRate
from toolforge_judge.metrics.meta import DagDiversity
from toolforge_judge.metrics.reliability import (
    ErrorRate,
    ErrorTaxonomyDistribution,
    RetryRate,
)
from toolforge_judge.metrics.structural import PrematureCallRate

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _span(
    task_id: str,
    span_id: str,
    tool_id: str,
    *,
    status: str = "ok",
    retries: list[dict] | None = None,
    inp: dict | None = None,
    output=None,
    duration_ms: float = 10.0,
    contribution: str | None = None,
    offset: int = 0,
) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        task_id=task_id,
        run_id="run1",
        usecase_id="uc1",
        type="tool_call",
        started_at=_T0 + timedelta(seconds=offset),
        duration_ms=duration_ms,
        status=status,
        tool_id=tool_id,
        input=inp or {},
        output=output,
        retries=retries or [],
        contribution=contribution,
    )


def _task(task_id: str, spans: list[SpanRecord], *, status: str = "success",
          offset: int = 0, dag: dict | None = None) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        run_id="run1",
        usecase_id="uc1",
        status=status,
        started_at=_T0 + timedelta(minutes=offset),
        spans=spans,
        dag=dag,
    )


def _window(tasks: list[TaskRecord], env: MetricEnv | None = None) -> MetricWindow:
    env = env or MetricEnv(min_samples=1)
    return MetricWindow.from_tasks("uc1", tasks, env)


# ---------------------------------------------------------------------------
# Family 1 — reliability
# ---------------------------------------------------------------------------


def test_error_rate_counts_failed_calls():
    tasks = [
        _task("t1", [
            _span("t1", "s1", "search", status="ok"),
            _span("t1", "s2", "search", status="error"),
        ]),
        _task("t2", [_span("t2", "s3", "search", status="ok")]),
    ]
    [val] = ErrorRate().compute(_window(tasks))
    assert val.tool_id == "search"
    assert val.value == pytest.approx(1 / 3)
    assert val.n == 3


def test_error_rate_breaches_above_threshold():
    env = MetricEnv(min_samples=2, error_rate_abs=0.10)
    tasks = [_task("t1", [
        _span("t1", "s1", "x", status="error"),
        _span("t1", "s2", "x", status="ok"),
        _span("t1", "s3", "x", status="ok"),
    ])]
    [val] = ErrorRate().compute(_window(tasks, env))
    assert val.value == pytest.approx(1 / 3)
    assert val.breached is True


def test_error_rate_does_not_breach_below_min_samples():
    env = MetricEnv(min_samples=10, error_rate_abs=0.10)
    tasks = [_task("t1", [_span("t1", "s1", "x", status="error")])]
    [val] = ErrorRate().compute(_window(tasks, env))
    assert val.breached is False
    assert val.status == "insufficient_data"


def test_retry_rate_and_sample_errors():
    tasks = [_task("t1", [
        _span("t1", "s1", "x", retries=[{"error": "boom"}]),
        _span("t1", "s2", "x"),
    ])]
    [val] = RetryRate().compute(_window(tasks))
    assert val.value == pytest.approx(0.5)
    assert "boom" in val.detail["sample_errors"]


def test_error_taxonomy_concentration():
    env = MetricEnv(min_samples=1, taxonomy_concentration=0.8)
    retries = [{"error": "validation error: 'budget' is not of type integer"}]
    tasks = [_task("t1", [
        _span("t1", "s1", "x", status="error", retries=retries),
        _span("t1", "s2", "x", status="error", retries=retries),
    ])]
    [val] = ErrorTaxonomyDistribution().compute(_window(tasks, env))
    assert val.detail["top_class"] == "schema_invalid"
    assert val.value == pytest.approx(1.0)
    assert val.breached is True


# ---------------------------------------------------------------------------
# Family 3 — contribution
# ---------------------------------------------------------------------------


def test_redundancy_detects_duplicate_input_in_task():
    env = MetricEnv(min_samples=1, redundancy_rate_abs=0.1)
    same = {"q": "weather"}
    tasks = [_task("t1", [
        _span("t1", "s1", "x", inp=same),
        _span("t1", "s2", "x", inp=same),     # duplicate -> redundant
        _span("t1", "s3", "x", inp={"q": "news"}),
    ])]
    [val] = RedundancyRate().compute(_window(tasks, env))
    assert val.detail["duplicate_inputs"] == 1
    assert val.value == pytest.approx(1 / 3)
    assert val.breached is True


# ---------------------------------------------------------------------------
# Family 4 — structural
# ---------------------------------------------------------------------------


def test_premature_call_detected_on_enriched_reinvocation():
    env = MetricEnv(min_samples=1, premature_call_rate_abs=0.1)
    tasks = [_task("t1", [
        _span("t1", "s1", "book", inp={"city": "Paris"}),
        _span("t1", "s2", "book", inp={"city": "Paris", "budget": 500}),
    ])]
    [val] = PrematureCallRate().compute(_window(tasks, env))
    assert val.detail["premature"] == 1
    assert val.value == pytest.approx(0.5)
    assert val.breached is True


# ---------------------------------------------------------------------------
# Family 6 — dag diversity
# ---------------------------------------------------------------------------


def test_dag_diversity_zero_when_all_dags_identical():
    dag = {
        "nodes": ["s1", "s2"],
        "edges": [{"from_span": "s1", "to_span": "s2", "via": "data"}],
    }
    tasks = [
        _task("t1", [_span("t1", "s1", "a"), _span("t1", "s2", "b")], dag=dag, offset=1),
        _task("t2", [_span("t2", "s1", "a"), _span("t2", "s2", "b")], dag=dag, offset=2),
    ]
    [val] = DagDiversity().compute(_window(tasks))
    assert val.value == pytest.approx(0.0)
    assert val.detail["distinct_short"] == 1


# ---------------------------------------------------------------------------
# Engine + version delta
# ---------------------------------------------------------------------------


def test_engine_runs_full_catalogue_and_serialises():
    tasks = [_task("t1", [_span("t1", "s1", "x", status="error")])]
    report = MetricEngine().compute_from_tasks("uc1", tasks, MetricEnv(min_samples=1))
    names = {v.metric for v in report.values}
    # one metric from each family present
    assert {"error_rate", "selection_precision", "dead_call_rate",
            "premature_call_rate", "node_assertion_pass_rate",
            "dag_diversity"} <= names
    # judge-scored metrics are catalogued but not computed yet
    sel = report.get("selection_precision", "x")
    assert sel is not None and sel.status == "requires_judge"
    # round-trips to plain dict
    assert isinstance(report.to_dict()["values"], list)


def test_judge_scored_metrics_reflect_judge_scores():
    from toolforge_judge.metrics.base import JUDGED, OK
    from toolforge_judge.metrics.judge_scored import SelectionPrecision

    env = MetricEnv(min_samples=1, selection_precision_min=0.8)
    tasks = [_task("t1", [
        _span("t1", "s1", "search"),
        _span("t1", "s2", "book"),
        _span("t1", "s3", "pay"),
    ])]
    win = MetricWindow.from_tasks("uc1", tasks, env)
    # search scored; book judged but without this score; pay never judged.
    win.judge_scores = {("search", "selection_precision"): (0.5, 3)}
    win.judged_tools = {"search", "book"}

    vals = {v.tool_id: v for v in SelectionPrecision().compute(win)}
    assert vals["search"].status == OK
    assert vals["search"].value == pytest.approx(0.5)
    assert vals["search"].breached is True          # 0.5 < 0.8 floor
    assert vals["book"].status == JUDGED and vals["book"].value is None
    assert vals["pay"].status == "requires_judge"


def test_version_delta_flags_regression():
    env = MetricEnv(min_samples=1, version_delta_regression=0.05)
    before = MetricEngine().compute_from_tasks(
        "uc1", [_task("t1", [_span("t1", "s1", "x", status="ok")])], env)
    after = MetricEngine().compute_from_tasks(
        "uc1", [_task("t2", [
            _span("t2", "s1", "x", status="error"),
            _span("t2", "s2", "x", status="error"),
        ])], env)
    deltas = version_delta(after, before, env)
    err = next(d for d in deltas if d.metric == "error_rate")
    assert err.before == pytest.approx(0.0)
    assert err.after == pytest.approx(1.0)
    assert err.regressed is True


def test_env_overrides_and_unknown_key_rejected():
    env = MetricEnv.with_overrides({"error_rate_abs": 0.42, "short_window": 5})
    assert env.error_rate_abs == 0.42
    assert env.short_window == 5
    with pytest.raises(KeyError):
        MetricEnv.with_overrides({"nope": 1})


# ---------------------------------------------------------------------------
# Window run selection (single / multi / all)
# ---------------------------------------------------------------------------


def _run_task(task_id: str, run_id: str, *, offset: int = 0) -> TaskRecord:
    return TaskRecord(
        task_id=task_id, run_id=run_id, usecase_id="uc1", status="success",
        started_at=_T0 + timedelta(minutes=offset),
        spans=[_span(task_id, f"{task_id}s", "search")],
    )


def _mixed_tasks() -> list[TaskRecord]:
    return [
        _run_task("t1", "runA", offset=0),
        _run_task("t2", "runB", offset=1),
        _run_task("t3", "runC", offset=2),
    ]


def test_window_all_runs_keeps_everything():
    win = MetricWindow.from_tasks("uc1", _mixed_tasks(), MetricEnv(min_samples=1))
    assert {t.run_id for t in win.long} == {"runA", "runB", "runC"}
    assert win.run_id is None
    assert win.run_ids is None


def test_window_single_run_filters_and_labels():
    win = MetricWindow.from_tasks(
        "uc1", _mixed_tasks(), MetricEnv(min_samples=1), run_id="runB"
    )
    assert {t.run_id for t in win.long} == {"runB"}
    assert win.run_id == "runB"
    assert win.run_ids == ["runB"]


def test_window_multi_run_filters_and_joins_label():
    win = MetricWindow.from_tasks(
        "uc1", _mixed_tasks(), MetricEnv(min_samples=1), run_ids=["runC", "runA"]
    )
    assert {t.run_id for t in win.long} == {"runA", "runC"}
    # label is deterministic (sorted, joined), independent of input order
    assert win.run_id == "runA+runC"
    assert win.run_ids == ["runA", "runC"]


def test_window_single_element_run_ids_labels_like_run_id():
    win = MetricWindow.from_tasks(
        "uc1", _mixed_tasks(), MetricEnv(min_samples=1), run_ids=["runB"]
    )
    assert win.run_id == "runB"
