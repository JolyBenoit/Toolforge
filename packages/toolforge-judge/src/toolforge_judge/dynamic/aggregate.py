"""Aggregate the static judge's local notes into global per-tool notes.

This is the bridge that turns the static judge's per-task scores (the "notes
locales") into the global notes — means and percentages — the dynamic judge
reports and stores. It also evaluates the judge-scored thresholds (the five
keys in ``_SCORE_THRESHOLDS``) from ``MetricEnv`` against the means, honouring
each score's direction (precision-like vs trouble-rate).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ..metrics.env import MetricEnv
from ..static.store import ToolNoteRecord
from .models import ToolGlobalNote

# Map a score key to (env attribute holding its threshold, higher_is_better).
# ``higher_is_better`` keys breach when the mean falls *below* the floor; the
# others (rates of trouble) breach when the mean rises *above* the ceiling.
_SCORE_THRESHOLDS: dict[str, tuple[str, bool]] = {
    "selection_precision": ("selection_precision_min", True),
    "param_extraction": ("param_extraction_min", True),
    "output_quality": ("node_assertion_pass_min", True),
    "downstream_correction": ("downstream_correction_rate_abs", False),
    "output_schema_drift": ("output_schema_drift_abs", False),
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

        # Breaches: any thresholded score whose mean crosses its bound, in the
        # direction that counts as worse for that score.
        breaches: list[str] = []
        for key, (attr, higher_is_better) in _SCORE_THRESHOLDS.items():
            if key not in mean_scores:
                continue
            mean, bound = mean_scores[key], getattr(env, attr)
            if (mean < bound) if higher_is_better else (mean > bound):
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


def scores_from_global_notes(
    notes: Iterable[ToolGlobalNote],
) -> tuple[dict[tuple[str, str], tuple[float, int]], set[str]]:
    """Turn global notes into the ``MetricWindow.judge_scores`` lookup.

    Returns ``((tool_id, score_key) → (mean, n_tasks), {judged tool_ids})`` —
    exactly what the judge-scored metrics read to flip from ``requires_judge``
    to ``ok``/``judged``.
    """
    scores: dict[tuple[str, str], tuple[float, int]] = {}
    judged: set[str] = set()
    for g in notes:
        judged.add(g.tool_id)
        for key, val in g.mean_scores.items():
            scores[(g.tool_id, key)] = (float(val), g.n_tasks)
    return scores, judged


def judge_scores_from_notes(
    records: Iterable[ToolNoteRecord], env: MetricEnv
) -> tuple[dict[tuple[str, str], tuple[float, int]], set[str]]:
    """Aggregate raw per-task notes straight into the judge-scores lookup."""
    return scores_from_global_notes(aggregate_tool_notes(records, env))
