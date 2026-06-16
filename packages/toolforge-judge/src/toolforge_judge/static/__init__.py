"""Static Judge — evaluates one execution trace (task) in isolation.

The static judge produces the *local* notes that feed the global metrics:
- a per-span ``contribution`` verdict (the backward pass that fills
  ``tf_prod_spans.contribution``), plus per-call selection/param scores;
- per-tool local notes (scores + an optional recommendation written *only when
  necessary*), each tool judged independently of the others.

It runs on the same LLM as the Creator (``config.llm.judge``, which defaults to
``config.llm.creator``). It can be re-run over a single task, a window of tasks,
or every task of a use case to chain iterations.
"""
from __future__ import annotations

from .judge import StaticJudge
from .llm import AgentLLMJudge, JudgeLLM
from .models import (
    SpanVerdict,
    StaticJudgeResult,
    ToolNote,
    ToolSpec,
    UseCaseSpec,
)
from .runner import StaticJudgeRunner, build_usecase_spec
from .store import JudgeStore, NullJudgeStore, ToolNoteRecord, get_judge_store

__all__ = [
    "StaticJudge",
    "StaticJudgeRunner",
    "build_usecase_spec",
    "JudgeLLM",
    "AgentLLMJudge",
    "SpanVerdict",
    "ToolNote",
    "StaticJudgeResult",
    "UseCaseSpec",
    "ToolSpec",
    "JudgeStore",
    "NullJudgeStore",
    "ToolNoteRecord",
    "get_judge_store",
]
