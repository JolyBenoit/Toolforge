"""CreatorJudge — synthesise corrective instructions for the Creator agent.

Two stages, each its own stateless LLM pass on the shared backend:

1. :meth:`synthesize_axes` — one pass per *problematic* tool (a tool with a
   breaching metric or a breaching run/run note), producing its improvement
   axes. Tools with no problem are left alone — the judge only speaks when there
   is something to fix.
2. :meth:`propose_instructions` — one pass over all the axes + the dynamic
   report + the *full* pipeline description, producing the corrective
   instructions (per-tool and structural).

:meth:`assess` chains the two and is side-effect free; persistence and loading
live in the runner/store.
"""
from __future__ import annotations

from datetime import UTC, datetime

from ..architecture.models import ArchitectureFinding, ArchitectureJudgeReport
from ..dynamic.models import DynamicJudgeReport, ToolGlobalNote
from ..static.llm import JudgeLLM
from ..static.models import UseCaseSpec
from .models import CreatorInstruction, CreatorJudgeReport, ToolImprovementAxes
from .prompt import (
    axes_evidence,
    build_axes_message,
    build_instructions_message,
    parse_axes,
    parse_instructions,
    tool_breach_views,
)


def problematic_tools(
    report: DynamicJudgeReport,
    architecture_report: ArchitectureJudgeReport | None = None,
) -> list[str]:
    """Tools the judge should act on (order-stable).

    A tool is problematic if it has a breaching metric, a breaching run/run
    note, *or* an architecture finding naming it — the last lets a design-level
    problem with no metric breach (e.g. a silent truncation) still be addressed.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(tool_id: str | None) -> None:
        if tool_id and tool_id not in seen:
            seen.add(tool_id)
            ordered.append(tool_id)

    for v in report.metric_report.breaches:
        add(v.tool_id)
    for note in report.tool_global_notes:
        if note.breaches:
            add(note.tool_id)
    if architecture_report is not None:
        for tool_id in architecture_report.problematic_tools:
            add(tool_id)
    return ordered


class CreatorJudge:
    def __init__(
        self,
        *,
        axes_llm: JudgeLLM | None = None,
        instruction_llm: JudgeLLM | None = None,
    ) -> None:
        self._axes_llm = axes_llm
        self._instruction_llm = instruction_llm

    async def synthesize_axes(
        self,
        report: DynamicJudgeReport,
        architecture_report: ArchitectureJudgeReport | None = None,
    ) -> list[ToolImprovementAxes]:
        """Stage 1: per-tool improvement axes for every problematic tool."""
        tools = problematic_tools(report, architecture_report)
        if not tools:
            return []
        if self._axes_llm is None:
            raise ValueError("synthesize_axes requires an axes_llm")
        notes: dict[str, ToolGlobalNote] = {
            n.tool_id: n for n in report.tool_global_notes
        }
        arch_by_tool: dict[str, list[ArchitectureFinding]] = {}
        if architecture_report is not None:
            for f in architecture_report.findings:
                for tool_id in f.tools_involved:
                    arch_by_tool.setdefault(tool_id, []).append(f)
        out: list[ToolImprovementAxes] = []
        for tool_id in tools:
            breaches = tool_breach_views(report.metric_report, tool_id)
            note = notes.get(tool_id)
            arch = arch_by_tool.get(tool_id, [])
            evidence = axes_evidence(breaches, note, arch)
            raw = await self._axes_llm.complete(
                build_axes_message(tool_id, breaches, note, arch)
            )
            out.append(parse_axes(tool_id, evidence, raw))
        return out

    async def propose_instructions(
        self,
        tool_axes: list[ToolImprovementAxes],
        report: DynamicJudgeReport,
        usecase: UseCaseSpec,
        architecture_report: ArchitectureJudgeReport | None = None,
    ) -> list[CreatorInstruction]:
        """Stage 2: pipeline-aware corrective instructions from the axes.

        Runs when there are per-tool axes *or* architecture findings — the
        latter includes structural findings (coverage gaps, wiring) that name no
        tool and so have no axis, yet still warrant an instruction.
        """
        arch_findings = list(architecture_report.findings) if architecture_report else []
        if not tool_axes and not arch_findings:
            return []
        if self._instruction_llm is None:
            raise ValueError("propose_instructions requires an instruction_llm")
        raw = await self._instruction_llm.complete(
            build_instructions_message(tool_axes, report, usecase, arch_findings)
        )
        return parse_instructions(raw)

    async def assess(
        self,
        report: DynamicJudgeReport,
        usecase: UseCaseSpec,
        architecture_report: ArchitectureJudgeReport | None = None,
    ) -> CreatorJudgeReport:
        """Chain both stages into a full report (side-effect free).

        Returns an empty report (no axes, no instructions) when nothing in the
        pipeline is breaching *and* the architecture judge raised no finding —
        the judge only acts on problems.
        """
        tool_axes = await self.synthesize_axes(report, architecture_report)
        instructions = await self.propose_instructions(
            tool_axes, report, usecase, architecture_report
        )
        return CreatorJudgeReport(
            usecase_id=report.usecase_id,
            run_id=report.run_id,
            computed_at=datetime.now(UTC).isoformat(),
            tool_axes=tool_axes,
            instructions=instructions,
        )
