"""Assemble the dynamic judge's diagnosis message (data only).

The judging instructions live in ``prompts/judge_dynamic_system.md``. Here we
only serialise the already-computed global picture (breaching metrics, global
notes, mean structural stability) for the optional LLM diagnosis pass.
"""
from __future__ import annotations

import json

from .models import DynamicJudgeReport


def build_diagnosis_message(report: DynamicJudgeReport) -> str:
    payload = {
        "usecase_id": report.usecase_id,
        "run_id": report.run_id,
        "n_tasks": report.n_tasks,
        "windows": {"short": report.short_window, "long": report.long_window},
        "structural_stability": report.structural_stability.to_dict(),
        "tool_global_notes": [n.to_dict() for n in report.tool_global_notes],
        "metric_breaches": report.breaching_metrics,
    }
    return (
        "Here is the aggregated, cross-run picture for this pipeline. Write the "
        "concise global diagnosis described in your instructions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
