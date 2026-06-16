"""Data models for the creator-facing judge.

Two output shapes, one per stage:

- :class:`ToolImprovementAxes` — stage 1: the synthesised improvement axes for a
  single problematic tool, grounded in its breaches + run/run comments.
- :class:`CreatorInstruction` — stage 2: a concrete, pipeline-aware corrective
  instruction addressed to the Creator agent. Per-tool *or* structural.

:class:`CreatorJudgeReport` wraps both for one (usecase, run) assessment.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Per-tool edits keep the static judge's vocabulary; the structural actions are
# new — they let the Creator restructure the pipeline (and so must be reasoned
# about against the whole use case, which is exactly what stage 2 does).
CreatorAction = Literal[
    "modify_implementation",
    "modify_description",
    "modify_usage",
    "create_tool",
    "remove_tool",
    "merge_tools",
    "split_tool",
]

Priority = Literal["low", "medium", "high"]

_STRUCTURAL_ACTIONS: set[str] = {
    "create_tool",
    "remove_tool",
    "merge_tools",
    "split_tool",
}


# ---------------------------------------------------------------------------
# Stage 1 — per-tool improvement axes
# ---------------------------------------------------------------------------


@dataclass
class ToolImprovementAxes:
    """Stage-1 synthesis for one problematic tool.

    ``summary`` is a short prose digest; ``axes`` are the discrete improvement
    directions; ``evidence`` is the raw breach/comment material the synthesis
    was grounded on (kept so stage 2 and the store stay auditable).
    """

    tool_id: str
    summary: str
    axes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 2 — corrective instructions to the Creator
# ---------------------------------------------------------------------------


@dataclass
class CreatorInstruction:
    """One corrective instruction the Creator agent can act on.

    ``instruction_id`` is a stable hash of the canonical fields, so re-running
    the judge over the same window yields the same id (idempotent persistence,
    and a stable target for ``change_reason=judge_instruction_id:<id>``).
    """

    action: CreatorAction
    target_tools: list[str]
    body: str
    rationale: str = ""
    priority: Priority = "medium"
    expected_effect: str = ""
    instruction_id: str = ""

    def __post_init__(self) -> None:
        if not self.instruction_id:
            self.instruction_id = self.compute_id()

    @property
    def is_structural(self) -> bool:
        return self.action in _STRUCTURAL_ACTIONS

    def compute_id(self) -> str:
        canonical = "|".join(
            [self.action, ",".join(sorted(self.target_tools)), self.body.strip()]
        )
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class CreatorJudgeReport:
    """Everything the creator judge produced for one (usecase, run) assessment."""

    usecase_id: str
    run_id: str | None
    computed_at: str
    tool_axes: list[ToolImprovementAxes] = field(default_factory=list)
    instructions: list[CreatorInstruction] = field(default_factory=list)

    @property
    def problematic_tools(self) -> list[str]:
        return [a.tool_id for a in self.tool_axes]

    @property
    def structural_instructions(self) -> list[CreatorInstruction]:
        return [i for i in self.instructions if i.is_structural]

    def axes_for(self, tool_id: str) -> ToolImprovementAxes | None:
        for a in self.tool_axes:
            if a.tool_id == tool_id:
                return a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "run_id": self.run_id,
            "computed_at": self.computed_at,
            "tool_axes": [a.to_dict() for a in self.tool_axes],
            "instructions": [i.to_dict() for i in self.instructions],
        }
