"""Dynamic Judge — cross-run, global assessment of a pipeline.

Where the static judge scores ONE trace, the dynamic judge takes the whole
window of runs and produces the *global* picture:

- global per-tool notes (the static judge's local notes averaged into means/%);
- the **mean structural stability** of the runs (its headline dimension);
- the deterministic SPC metric report over the window;
- an optional concise LLM diagnosis (same model backend as the Creator).

It is re-runnable over all runs or a window, and persists global notes/% to
Postgres.
"""
from __future__ import annotations

from .aggregate import (
    aggregate_tool_notes,
    judge_scores_from_notes,
    scores_from_global_notes,
)
from .judge import DynamicJudge
from .models import (
    DynamicJudgeReport,
    StructuralStability,
    ToolGlobalNote,
    ToolStability,
)
from .stability import compute_structural_stability
from .store import (
    DynamicJudgeStore,
    NullDynamicJudgeStore,
    PostgresDynamicJudgeStore,
    get_dynamic_judge_store,
)

__all__ = [
    "DynamicJudge",
    "DynamicJudgeReport",
    "ToolGlobalNote",
    "ToolStability",
    "StructuralStability",
    "aggregate_tool_notes",
    "judge_scores_from_notes",
    "scores_from_global_notes",
    "compute_structural_stability",
    "DynamicJudgeStore",
    "NullDynamicJudgeStore",
    "PostgresDynamicJudgeStore",
    "get_dynamic_judge_store",
]
