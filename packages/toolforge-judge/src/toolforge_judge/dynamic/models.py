"""Data models for the dynamic judge — global, cross-run aggregates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..metrics.engine import MetricReport


@dataclass
class ToolGlobalNote:
    """A tool's global note: the static judge's local notes averaged over runs.

    ``mean_scores`` are means of the per-task scores (selection_precision,
    param_extraction, output_quality, …). ``recommendation_rate`` is the share
    of tasks in which the static judge wrote a recommendation for this tool.
    Independent per tool, by construction.
    """

    tool_id: str
    n_tasks: int
    mean_scores: dict[str, float] = field(default_factory=dict)
    n_recommendations: int = 0
    recommendation_rate: float = 0.0
    dominant_target: str = "none"
    sample_recommendations: list[str] = field(default_factory=list)
    breaches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolStability:
    """Per-tool structural-stability components and rolled-up score (0..1)."""

    tool_id: str
    position_variance: float | None
    output_divergence: float | None
    premature_call_rate: float | None
    order_sensitivity: float | None
    stability_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralStability:
    """The dynamic judge's headline dimension: mean structural stability.

    ``mean_structural_stability`` is the pipeline-level average of the per-tool
    ``stability_score`` over the window; ``breached`` fires below
    ``env.structural_stability_min``.
    """

    per_tool: list[ToolStability] = field(default_factory=list)
    mean_structural_stability: float | None = None
    dag_diversity: float | None = None
    breached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_tool": [t.to_dict() for t in self.per_tool],
            "mean_structural_stability": self.mean_structural_stability,
            "dag_diversity": self.dag_diversity,
            "breached": self.breached,
        }


@dataclass
class DynamicJudgeReport:
    """Everything the dynamic judge produced over a window of runs."""

    usecase_id: str
    run_id: str | None
    computed_at: str
    short_window: int
    long_window: int
    n_tasks: int
    metric_report: MetricReport
    tool_global_notes: list[ToolGlobalNote] = field(default_factory=list)
    structural_stability: StructuralStability = field(default_factory=StructuralStability)
    diagnosis: str | None = None

    @property
    def breaching_metrics(self) -> list[dict[str, Any]]:
        return [v.to_dict() for v in self.metric_report.breaches]

    def note_for(self, tool_id: str) -> ToolGlobalNote | None:
        for n in self.tool_global_notes:
            if n.tool_id == tool_id:
                return n
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "run_id": self.run_id,
            "computed_at": self.computed_at,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "n_tasks": self.n_tasks,
            "tool_global_notes": [n.to_dict() for n in self.tool_global_notes],
            "structural_stability": self.structural_stability.to_dict(),
            "metric_breaches": self.breaching_metrics,
            "diagnosis": self.diagnosis,
        }
