"""Data models for the static judge: inputs (UseCaseSpec) and outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Contribution = Literal["necessary", "redundant", "dead"]
RecommendationTarget = Literal["none", "implementation", "description", "usage"]


# ---------------------------------------------------------------------------
# Inputs — the use case context the judge reasons against
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """One tool as the judge needs to know it (decoupled from the registry)."""

    tool_id: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class UseCaseSpec:
    """The use case rules + utility + tool catalogue handed to the judge.

    ``utility`` is *what the pipeline is for*; ``rules`` is *how it must behave*.
    Both are free text (typically drawn from the use-case prompt and consumer
    prompt). Kept duck-typed so the judge has no hard registry dependency.
    """

    usecase_id: str
    utility: str
    rules: str
    tools: list[ToolSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Outputs — the judge's verdicts for one task
# ---------------------------------------------------------------------------


@dataclass
class SpanVerdict:
    """Per tool_call verdict (backward pass + per-call scores)."""

    span_id: str
    tool_id: str
    contribution: Contribution
    selection_appropriate: bool
    param_fidelity: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolNote:
    """Per-tool local note for one task — independent of every other tool.

    ``recommendation`` is None unless this trace revealed a concrete, actionable
    problem with the tool (the judge writes recommendations only when needed).
    """

    tool_id: str
    scores: dict[str, float] = field(default_factory=dict)
    recommendation: str | None = None
    recommendation_target: RecommendationTarget = "none"

    @property
    def has_recommendation(self) -> bool:
        return bool(self.recommendation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StaticJudgeResult:
    """Everything the static judge produced for a single task."""

    task_id: str
    run_id: str
    usecase_id: str
    judge_model: str
    span_verdicts: list[SpanVerdict] = field(default_factory=list)
    tool_notes: list[ToolNote] = field(default_factory=list)
    raw_response: str | None = None

    def verdict_for(self, span_id: str) -> SpanVerdict | None:
        for v in self.span_verdicts:
            if v.span_id == span_id:
                return v
        return None

    def note_for(self, tool_id: str) -> ToolNote | None:
        for n in self.tool_notes:
            if n.tool_id == tool_id:
                return n
        return None

    @property
    def recommendations(self) -> list[ToolNote]:
        return [n for n in self.tool_notes if n.has_recommendation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "usecase_id": self.usecase_id,
            "judge_model": self.judge_model,
            "span_verdicts": [v.to_dict() for v in self.span_verdicts],
            "tool_notes": [n.to_dict() for n in self.tool_notes],
        }
