"""Orchestration + briefing for the creator-facing judge.

Two responsibilities, both side-effect-light:

- :func:`run_creator_judge` chains :meth:`CreatorJudge.assess` and persists the
  resulting report (the only place that writes the creator-judge tables).
- :func:`render_briefing` turns the operator-approved subset of instructions
  into the *frozen* seed message handed to the Creator agent. The briefing is
  generated, not editable: the operator's control is the checkbox selection, not
  the wording.
"""
from __future__ import annotations

from ..architecture.models import ArchitectureJudgeReport
from ..dynamic.models import DynamicJudgeReport
from ..static.models import UseCaseSpec
from .judge import CreatorJudge
from .models import CreatorInstruction, CreatorJudgeReport
from .store import CreatorJudgeStore, NullCreatorJudgeStore

# High-priority instructions lead the briefing so the Creator does them first.
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


async def run_creator_judge(
    judge: CreatorJudge,
    dynamic_report: DynamicJudgeReport,
    usecase: UseCaseSpec,
    *,
    architecture_report: ArchitectureJudgeReport | None = None,
    store: CreatorJudgeStore | None = None,
) -> CreatorJudgeReport:
    """Assess and persist the creator-judge report for one (usecase, run)."""
    report = await judge.assess(dynamic_report, usecase, architecture_report)
    (store or NullCreatorJudgeStore()).save_report(report)
    return report


def select_instructions(
    report: CreatorJudgeReport, selected_ids: set[str] | None
) -> list[CreatorInstruction]:
    """The approved instructions, priority-ordered.

    ``selected_ids`` of ``None`` means "all" (the default — every box checked);
    an empty set means "none". Order is high→low priority, then the report's
    own order, so the briefing is stable across re-renders.
    """
    chosen = [
        i for i in report.instructions
        if selected_ids is None or i.instruction_id in selected_ids
    ]
    return sorted(chosen, key=lambda i: _PRIORITY_RANK.get(i.priority, 1))


def render_briefing(
    report: CreatorJudgeReport, selected_ids: set[str] | None = None
) -> str:
    """Build the frozen seed message for the Creator agent.

    Lists the approved corrective instructions and tells the Creator how to
    apply each (update/propose then sandbox-validate), leaving promotion to the
    operator. Each line carries its ``instruction_id`` so a follow-up can stamp
    ``change_reason=judge_instruction_id:<id>`` onto the resulting version.
    """
    chosen = select_instructions(report, selected_ids)
    if not chosen:
        return ""

    lines = [
        f"You are improving the tool pipeline for use case "
        f"\"{report.usecase_id}\". An operator has reviewed the Judge's report "
        f"and approved the {len(chosen)} corrective change(s) below.",
        "",
        "For each change: call update_tool (or propose_tool for a brand-new "
        "tool) to apply it, then validate_in_sandbox to test it. Do NOT promote "
        "— the operator decides promotion. Work through them in the order given "
        "(highest priority first). Account for the whole pipeline: do not break "
        "another tool and do not introduce redundancy.",
        "",
    ]
    for n, instr in enumerate(chosen, start=1):
        targets = ", ".join(instr.target_tools) or "(new tool)"
        lines.append(
            f"{n}. [{instr.priority.upper()}] {instr.action} — tool(s): {targets}"
        )
        lines.append(f"   Change: {instr.body}")
        if instr.rationale:
            lines.append(f"   Why: {instr.rationale}")
        if instr.expected_effect:
            lines.append(f"   Expected effect: {instr.expected_effect}")
        lines.append(f"   (judge_instruction_id: {instr.instruction_id})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
