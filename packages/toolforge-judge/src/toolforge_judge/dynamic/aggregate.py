"""Aggregate the static judge's local notes into global per-tool notes.

This is the bridge that turns the static judge's per-task scores (the "notes
locales") into the global notes — means and percentages — the dynamic judge
reports and stores. It also evaluates the judge-scored thresholds
(selection_precision, param_extraction) from ``MetricEnv`` against the means.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ..metrics.env import MetricEnv
from ..static.store import ToolNoteRecord
from .models import ToolGlobalNote

# Map a score key to the env attribute holding its lower-bound threshold.
_SCORE_THRESHOLDS: dict[str, str] = {
    "selection_precision": "selection_precision_min",
    "param_extraction": "param_extraction_min",
    "output_quality": "node_assertion_pass_min",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_tool_notes(
    records: Iterable[ToolNoteRecord], env: MetricEnv
) -> list[ToolGlobalNote]:
    """Group local notes by tool and roll them up into global notes."""
    by_tool: dict[str, list[ToolNoteRecord]] = defaultdict(list)
    for r in records:
        by_tool[r.tool_id].append(r)

    notes: list[ToolGlobalNote] = []
    for tool_id, recs in by_tool.items():
        n = len(recs)
        # Mean each score key across the tasks that reported it.
        score_values: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            for key, val in r.scores.items():
                score_values[key].append(float(val))
        mean_scores = {k: _mean(v) for k, v in score_values.items()}

        with_rec = [r for r in recs if r.recommendation]
        targets = Counter(
            r.recommendation_target for r in with_rec if r.recommendation_target != "none"
        )
        dominant_target = targets.most_common(1)[0][0] if targets else "none"

        # Breaches: any thresholded score whose mean falls below its floor.
        breaches: list[str] = []
        for key, attr in _SCORE_THRESHOLDS.items():
            if key in mean_scores and mean_scores[key] < getattr(env, attr):
                breaches.append(key)

        notes.append(
            ToolGlobalNote(
                tool_id=tool_id,
                n_tasks=n,
                mean_scores={k: round(v, 4) for k, v in mean_scores.items()},
                n_recommendations=len(with_rec),
                recommendation_rate=round(len(with_rec) / n, 4) if n else 0.0,
                dominant_target=dominant_target,
                sample_recommendations=[
                    r.recommendation for r in with_rec[:3] if r.recommendation
                ],
                breaches=breaches,
            )
        )
    notes.sort(key=lambda x: x.tool_id)
    return notes
