"""Production telemetry data models.

Three-level hierarchy: Task → Span → NestedLLMCall.
One reference object: PipelineSpec (written once at promote-to-production time).

All timestamps are ISO-8601 strings in UTC.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------

SpanType = Literal["llm_call", "tool_call", "user_wait"]
SpanStatus = Literal["ok", "error", "retried"]
TaskStatus = Literal["running", "success", "partial", "failed", "user_aborted"]
EdgeVia = Literal["control", "data"]
ExplicitFeedback = Literal["thumbs_up", "correction", "none"]
Contribution = Literal["necessary", "redundant", "dead"]

# ---------------------------------------------------------------------------
# Nested sub-objects
# ---------------------------------------------------------------------------


@dataclass
class NestedLLMCall:
    """One LLM call made internally by a tool during its execution.

    ``sequence`` is 1-based and reflects the order the calls were issued
    within the parent tool call.  Full prompts and responses are stored so
    the Judge can audit the tool's internal reasoning chain.
    """

    call_id: str
    sequence: int
    model: str
    messages: list[dict[str, Any]]
    response: dict[str, Any]
    tokens_in: int
    tokens_out: int
    duration_ms: float
    started_at: str
    system: str | None = None


@dataclass
class LLMMeta:
    """Metadata attached to an agent-level ``llm_call`` span."""

    model: str
    tokens_in: int
    tokens_out: int
    prompt_hash: str | None = None


@dataclass
class Retry:
    """One failed attempt recorded inside a ``tool_call`` span."""

    error: str
    fix: str | None = None


@dataclass
class DAGEdge:
    from_span: str
    to_span: str
    via: EdgeVia


@dataclass
class DAG:
    """Directed acyclic graph of spans for a task.

    ``nodes`` is an ordered list of span_ids.
    ``edges`` capture control flow (LLM → tool it triggered) and
    data flow (tool result → LLM that consumed it).
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)


@dataclass
class InputTimelineEntry:
    """One user message recorded in chronological order within a task."""

    t: str
    user_turn: int
    content: str
    type: Literal["user_message"] = "user_message"


@dataclass
class TaskCost:
    """Aggregated cost for a complete task.

    Agent tokens = LLM calls made by the orchestrator loop.
    Tool tokens  = LLM calls made internally by tools (nested_llm_calls).
    """

    agent_tokens_in: int = 0
    agent_tokens_out: int = 0
    tool_tokens_in: int = 0
    tool_tokens_out: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0


@dataclass
class UserFeedback:
    explicit: ExplicitFeedback = "none"
    correction_text: str | None = None


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """One node in the task DAG.

    Fields are type-specific:
    - ``llm_call``:  llm_meta, system_prompt, messages, response
    - ``tool_call``: tool_id, tool_version, input, output, retries, nested_llm_calls
    - ``user_wait``: user_turn, user_message  (PauseForUserEvent → next user input)

    ``contribution`` is always None at capture time; the Judge fills it in
    post-hoc after analysing the full task.
    """

    span_id: str
    task_id: str
    run_id: str
    usecase_id: str
    type: SpanType
    started_at: str

    # Ordering — derivable into parent_spans by the store layer
    llm_turn_index: int | None = None
    call_index_in_turn: int | None = None

    # DAG linkage
    parent_spans: list[str] = field(default_factory=list)

    # Timing
    ended_at: str | None = None
    duration_ms: float | None = None
    status: SpanStatus | None = None

    # tool_call fields
    tool_id: str | None = None
    tool_version: int | None = None
    input: dict[str, Any] | None = None
    output: Any = None
    retries: list[Retry] = field(default_factory=list)
    nested_llm_calls: list[NestedLLMCall] = field(default_factory=list)

    # llm_call fields
    llm_meta: LLMMeta | None = None
    system_prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    response: dict[str, Any] | None = None

    # user_wait fields
    user_turn: int | None = None
    user_message: str | None = None

    # Filled post-hoc by the Judge
    contribution: Contribution | None = None


# ---------------------------------------------------------------------------
# PipelineSpec  (written once at promote-to-production)
# ---------------------------------------------------------------------------


@dataclass
class PipelineToolSnapshot:
    """Immutable snapshot of one tool at the moment of pipeline promotion."""

    tool_id: str
    tool_version: int
    implementation_hash: str
    schema: dict[str, Any]


@dataclass
class PipelineSpec:
    """Immutable record of a pipeline version promoted to production.

    ``change_reason`` links this version to the Judge instruction that caused
    the fork (format: ``judge_instruction_id:<id>``), enabling the Judge to
    measure whether its hypothesis held.
    """

    run_id: str
    usecase_id: str
    promoted_at: str
    tools: list[PipelineToolSnapshot]
    system_prompt: str | None = None
    forked_from: str | None = None
    change_reason: str | None = None


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts (JSON-serialisable)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
