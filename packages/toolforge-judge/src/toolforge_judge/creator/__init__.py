"""Creator-facing judge — turns diagnosis into corrective instructions.

The third and final judge. Unlike the static (per-task) and dynamic (cross-run)
judges, which are purely *descriptive*, the creator judge is the **action**
layer: it reads the breaching metrics, the per-tool run/run comments and the
dynamic report, and writes concrete corrective instructions addressed to the
Creator agent. Each instruction carries a stable id; when the Creator forks a
new pipeline version it records ``change_reason=judge_instruction_id:<id>``,
closing the feedback loop the ``version_delta`` metric measures.

It works in two stages:

1. **synthesize_axes** — for every tool that actually has a problem, summarise
   its improvement axes from its breaching metrics + run/run comments.
2. **propose_instructions** — take those summaries + the dynamic report + the
   *full* pipeline description, so a proposed change accounts for the whole use
   case (no breakage, no redundancy), and emit per-tool *and* structural
   instructions.
"""
from __future__ import annotations

from .judge import CreatorJudge
from .models import (
    CreatorAction,
    CreatorInstruction,
    CreatorJudgeReport,
    ToolImprovementAxes,
)
from .runner import render_briefing, run_creator_judge, select_instructions
from .store import (
    CreatorJudgeStore,
    NullCreatorJudgeStore,
    PostgresCreatorJudgeStore,
    get_creator_judge_store,
)

__all__ = [
    "CreatorJudge",
    "CreatorAction",
    "CreatorInstruction",
    "CreatorJudgeReport",
    "ToolImprovementAxes",
    "render_briefing",
    "run_creator_judge",
    "select_instructions",
    "CreatorJudgeStore",
    "NullCreatorJudgeStore",
    "PostgresCreatorJudgeStore",
    "get_creator_judge_store",
]
