"""Creator-judge tests over in-memory reports (no DB, no API)."""
from __future__ import annotations

from datetime import UTC, datetime

from toolforge_judge.creator import CreatorInstruction, CreatorJudge
from toolforge_judge.creator.judge import problematic_tools
from toolforge_judge.creator.prompt import parse_instructions
from toolforge_judge.dynamic.models import (
    DynamicJudgeReport,
    StructuralStability,
    ToolGlobalNote,
)
from toolforge_judge.metrics.base import MetricValue
from toolforge_judge.metrics.engine import MetricReport
from toolforge_judge.static.models import ToolSpec, UseCaseSpec

_NOW = datetime(2026, 1, 1, tzinfo=UTC).isoformat()


def _breach(metric, tool_id, value) -> MetricValue:
    return MetricValue(
        metric=metric, family="reliability", scope="tool", window="short",
        value=value, n=5, breached=True, tool_id=tool_id,
    )


def _report(*, metric_values, global_notes) -> DynamicJudgeReport:
    metric_report = MetricReport(
        usecase_id="uc1", run_id="run1", computed_at=_NOW,
        short_window=5, long_window=20, values=list(metric_values),
    )
    return DynamicJudgeReport(
        usecase_id="uc1", run_id="run1", computed_at=_NOW,
        short_window=5, long_window=20, n_tasks=10,
        metric_report=metric_report,
        tool_global_notes=list(global_notes),
        structural_stability=StructuralStability(mean_structural_stability=0.8),
        diagnosis="search is the weak point",
    )


def _usecase() -> UseCaseSpec:
    return UseCaseSpec(
        usecase_id="uc1", utility="plan a trip", rules="be concise",
        tools=[
            ToolSpec("search", "search for places"),
            ToolSpec("book", "book a place"),
        ],
    )


class FakeAxesLLM:
    model = "fake-axes"

    def __init__(self):
        self.calls: list[str] = []

    async def complete(self, user_message: str) -> str:
        self.calls.append(user_message)
        return '{"summary": "weak selection", "axes": ["tighten selection precision"]}'


class FakeInstructionLLM:
    model = "fake-instr"

    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[str] = []

    async def complete(self, user_message: str) -> str:
        self.calls.append(user_message)
        return self.payload


# --- problematic-tool selection -------------------------------------------


def test_problematic_tools_union_of_metric_and_note_breaches():
    report = _report(
        metric_values=[
            _breach("latency_p95", "search", 0.9),
            MetricValue("error_rate", "reliability", "tool", "short", 0.0, 5,
                        breached=False, tool_id="book"),
        ],
        global_notes=[
            ToolGlobalNote(tool_id="book", n_tasks=10, breaches=[]),
            ToolGlobalNote(tool_id="pay", n_tasks=10, breaches=["selection_precision"]),
        ],
    )
    # search: metric breach; pay: note breach; book: healthy on both -> excluded.
    assert problematic_tools(report) == ["search", "pay"]


# --- stage 1: axes ---------------------------------------------------------


async def test_only_problematic_tools_get_axes():
    report = _report(
        metric_values=[_breach("latency_p95", "search", 0.9)],
        global_notes=[
            ToolGlobalNote(tool_id="search", n_tasks=10, breaches=["selection_precision"]),
            ToolGlobalNote(tool_id="book", n_tasks=10, breaches=[]),
        ],
    )
    axes_llm = FakeAxesLLM()
    judge = CreatorJudge(axes_llm=axes_llm)
    axes = await judge.synthesize_axes(report)
    assert [a.tool_id for a in axes] == ["search"]
    assert len(axes_llm.calls) == 1
    assert axes[0].axes == ["tighten selection precision"]
    # evidence carries the breach it was grounded on
    assert axes[0].evidence["breaching_metrics"][0]["metric"] == "latency_p95"


async def test_no_problem_means_no_axes_and_no_llm_needed():
    report = _report(
        metric_values=[MetricValue("error_rate", "reliability", "tool", "short",
                                   0.0, 5, breached=False, tool_id="search")],
        global_notes=[ToolGlobalNote(tool_id="search", n_tasks=10, breaches=[])],
    )
    judge = CreatorJudge()  # no LLMs at all
    result = await judge.assess(report, _usecase())
    assert result.tool_axes == []
    assert result.instructions == []


# --- stage 2 + full assess -------------------------------------------------


async def test_assess_produces_instructions():
    report = _report(
        metric_values=[_breach("latency_p95", "search", 0.9)],
        global_notes=[ToolGlobalNote(tool_id="search", n_tasks=10,
                                     breaches=["selection_precision"])],
    )
    instr_payload = (
        '{"instructions": [{"action": "modify_usage", "target_tools": ["search"],'
        ' "body": "only call search when no cached result exists",'
        ' "rationale": "redundant calls", "priority": "high",'
        ' "expected_effect": "lower redundancy"}]}'
    )
    instr_llm = FakeInstructionLLM(instr_payload)
    judge = CreatorJudge(axes_llm=FakeAxesLLM(), instruction_llm=instr_llm)
    result = await judge.assess(report, _usecase())

    assert len(result.instructions) == 1
    instr = result.instructions[0]
    assert instr.action == "modify_usage"
    assert instr.priority == "high"
    assert instr.instruction_id  # stable id auto-computed
    # stage-2 prompt saw the full pipeline (book included, though healthy)
    assert '"book"' in instr_llm.calls[0]


# --- parsing robustness ----------------------------------------------------


def test_parse_instructions_drops_invalid_entries():
    raw = (
        '{"instructions": ['
        '{"action": "frobnicate", "target_tools": ["x"], "body": "nope"},'
        '{"action": "remove_tool", "target_tools": [], "body": "missing target"},'
        '{"action": "modify_implementation", "target_tools": ["search"], "body": ""},'
        '{"action": "create_tool", "target_tools": [], "body": "add a cache tool"},'
        '{"action": "merge_tools", "target_tools": ["a", "b"], "body": "fold b into a"}'
        ']}'
    )
    instrs = parse_instructions(raw)
    actions = [(i.action, tuple(i.target_tools)) for i in instrs]
    # invalid action, missing target, empty body all dropped; create w/o target kept.
    assert actions == [
        ("create_tool", ()),
        ("merge_tools", ("a", "b")),
    ]


def test_instruction_id_is_stable_and_content_addressed():
    a = CreatorInstruction(action="modify_usage", target_tools=["b", "a"], body=" do x ")
    b = CreatorInstruction(action="modify_usage", target_tools=["a", "b"], body="do x")
    # order-insensitive targets + trimmed body -> same id (idempotent on replay)
    assert a.instruction_id == b.instruction_id
    c = CreatorInstruction(action="remove_tool", target_tools=["a"], body="do x")
    assert c.instruction_id != a.instruction_id
