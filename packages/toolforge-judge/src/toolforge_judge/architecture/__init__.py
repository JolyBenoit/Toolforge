"""Architecture Judge — evaluates the toolset as a *designed system*.

The 4th judge. Where the static/dynamic/creator judges all reason from execution
telemetry and per-tool behaviour, this one reads the **handler source** and the
tool contracts and asks whether the pipeline, as designed and wired, actually
serves the use case end to end: coverage, information-flow sufficiency,
over-simplification (e.g. a PDF extractor that returns only the first 500 tokens
when the use case needs the full text), overkill/redundant steps, wiring/order,
and technical constraints (context window / max tokens → propose chunking or a
``split_tool``).

Two modes share one entry point (``ArchitectureJudge.assess``): *design-time*
(the spec alone, no runs) and *post-run* (enriched with a dynamic-judge report).
It runs on the same LLM backend as the other judges and *feeds the creator judge*
— it describes and proposes, the creator judge prescribes.
"""
from __future__ import annotations

from .judge import ArchitectureJudge
from .models import (
    ArchitectureFinding,
    ArchitectureJudgeReport,
    ArchitectureSpec,
    RichToolSpec,
    ToolContract,
)
from .prompt import digest_from_dynamic_report
from .runner import build_architecture_spec
from .store import (
    ArchitectureJudgeStore,
    NullArchitectureJudgeStore,
    get_architecture_judge_store,
)

__all__ = [
    "ArchitectureJudge",
    "ArchitectureSpec",
    "RichToolSpec",
    "ToolContract",
    "ArchitectureFinding",
    "ArchitectureJudgeReport",
    "build_architecture_spec",
    "digest_from_dynamic_report",
    "ArchitectureJudgeStore",
    "NullArchitectureJudgeStore",
    "get_architecture_judge_store",
]
