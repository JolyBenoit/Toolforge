"""Consumer agent — wraps LLMAgent + MCPToolProvider targeting the Usecase MCP server."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .agent import LLMAgent
from .mcp_client import MCPToolProvider
from .types import (
    Message,
    MessageComplete,
    PauseForUserEvent,
    StreamEvent,
    TextDelta,
    ToolCallComplete,
    ToolCallStart,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from toolforge_telemetry.production import ProductionTelemetryStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _msg_content_to_dict(content: Any) -> Any:
    """Serialize a Message.content (str or list of MessageContent dataclasses)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                result.append(dataclasses.asdict(item))
            else:
                result.append(str(item))
        return result
    return str(content)


def _prompt_hash(system: str, history: list[Message]) -> str:
    payload = system + json.dumps(
        [{"role": m.role, "content": _msg_content_to_dict(m.content)} for m in history],
        ensure_ascii=False,
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class ConsumerAgent:
    """LLM agent that executes a use case via tools served by the Usecase MCP server.

    When ``prod_store`` is provided (in_production runs), the agent records the
    full three-level telemetry (Task → Span → NestedLLMCall) to Postgres.

    Call run_task() / run_turn() to stream execution events. Call
    close_session() when the conversation is over.
    """

    def __init__(
        self,
        agent: LLMAgent,
        provider: MCPToolProvider,
        *,
        prod_store: ProductionTelemetryStore | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        usecase_id: str | None = None,
    ) -> None:
        self._agent = agent
        self._provider = provider
        self._prod_store = prod_store
        self._task_id = task_id
        self._run_id = run_id
        self._usecase_id = usecase_id

        # Wire telemetry callback on the provider when production mode is active.
        if prod_store is not None:
            self._provider._on_tool_telemetry = self._on_tool_meta

        # --- telemetry state (only used when prod_store is set) ---
        self._task_opened: bool = False
        self._task_closed: bool = False
        self._task_started_at: str | None = None
        self._user_turn: int = 0
        self._llm_turn_index: int = 0
        self._call_index_in_turn: int = 0

        # DAG tracking
        self._all_span_ids: list[str] = []
        self._dag_edges: list[Any] = []           # list[DAGEdge] at runtime
        self._prev_turn_tool_span_ids: list[str] = []
        self._current_turn_tool_span_ids: list[str] = []
        self._last_llm_span_id: str | None = None

        # Per-call tracking
        self._pending_tool_starts: dict[str, str] = {}   # tc_id -> started_at
        self._pending_tool_data: dict[str, tuple[str, dict, str]] = {}  # tc_id -> (name, input, started_at)
        self._tool_meta_queue: deque[dict[str, Any]] = deque()

        # LLM phase timing
        self._llm_phase_started_at: str | None = None
        self._in_llm_phase: bool = False

        # Cost accumulation
        self._agent_tokens_in: int = 0
        self._agent_tokens_out: int = 0
        self._tool_tokens_in: int = 0
        self._tool_tokens_out: int = 0
        self._tool_call_count: int = 0
        self._task_ended_at: str | None = None
        self._last_text_output: str | None = None

        # user_wait span preserved across run_turn() calls
        self._pending_user_wait_span: Any | None = None  # Span at runtime

        # Explicit user feedback buffered until the task is open / closed.
        self._pending_feedback: tuple[str, str | None] | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._agent.model

    @property
    def history(self) -> list[Message]:
        return self._agent.history

    @property
    def task_id(self) -> str | None:
        return self._task_id

    def reset_history(self) -> None:
        self._agent.reset_history()

    def new_session(self, *, task_id: str | None = None) -> str | None:
        """Start a fresh task within the same run — a new telemetry line.

        Resets the conversation history and all telemetry state, then assigns a
        new ``task_id`` so the next ``run_task()`` opens a brand-new task record.
        The run_id, usecase_id and MCP connection are kept, so this is a cheap
        in-place refresh rather than a full relaunch.

        Call ``close_session()`` first to finalise the previous line. Returns
        the new task_id (or ``None`` when production telemetry is inactive).
        """
        self.reset_history()

        # Reset telemetry state to constructor defaults.
        self._task_opened = False
        self._task_closed = False
        self._task_started_at = None
        self._user_turn = 0
        self._llm_turn_index = 0
        self._call_index_in_turn = 0
        self._all_span_ids = []
        self._dag_edges = []
        self._prev_turn_tool_span_ids = []
        self._current_turn_tool_span_ids = []
        self._last_llm_span_id = None
        self._pending_tool_starts = {}
        self._pending_tool_data = {}
        self._tool_meta_queue.clear()
        self._llm_phase_started_at = None
        self._in_llm_phase = False
        self._agent_tokens_in = 0
        self._agent_tokens_out = 0
        self._tool_tokens_in = 0
        self._tool_tokens_out = 0
        self._tool_call_count = 0
        self._task_ended_at = None
        self._last_text_output = None
        self._pending_user_wait_span = None
        self._pending_feedback = None

        # A new telemetry line only exists when production telemetry is wired.
        if self._prod_store is not None:
            self._task_id = task_id or secrets.token_hex(12)
        return self._task_id

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _telemetry_active(self) -> bool:
        return self._prod_store is not None and self._task_id is not None

    def _on_tool_meta(self, name: str, meta: dict[str, Any]) -> None:
        """Callback from MCPToolProvider when a __tf__ envelope is detected."""
        self._tool_meta_queue.append(meta)

    def _ensure_task_open(self, first_message: str) -> None:
        if not self._telemetry_active() or self._task_opened:
            return
        self._task_started_at = _now_iso()
        self._prod_store.open_task(  # type: ignore[union-attr]
            self._task_id,      # type: ignore[arg-type]
            self._run_id,       # type: ignore[arg-type]
            self._usecase_id,   # type: ignore[arg-type]
            self._task_id,      # type: ignore[arg-type]
            self._task_started_at,
        )
        self._task_opened = True

    def _record_input_entry(self, content: str) -> None:
        if not self._telemetry_active():
            return
        from toolforge_telemetry.production import InputTimelineEntry
        self._user_turn += 1
        self._prod_store.append_input_entry(  # type: ignore[union-attr]
            self._task_id,  # type: ignore[arg-type]
            InputTimelineEntry(t=_now_iso(), user_turn=self._user_turn, content=content),
        )

    def _close_pending_user_wait(self) -> None:
        if self._pending_user_wait_span is None:
            return
        from toolforge_telemetry.production import DAGEdge
        span = self._pending_user_wait_span
        span.ended_at = _now_iso()
        self._prod_store.record_span(span)  # type: ignore[union-attr]
        self._all_span_ids.append(span.span_id)
        # user_wait feeds the next LLM call
        self._prev_turn_tool_span_ids = [span.span_id]
        self._current_turn_tool_span_ids = []
        if self._last_llm_span_id:
            self._dag_edges.append(
                DAGEdge(from_span=self._last_llm_span_id, to_span=span.span_id, via="control")
            )
        self._pending_user_wait_span = None

    def _on_tool_call_start(self, ev: ToolCallStart) -> None:
        if not self._telemetry_active():
            return
        if not self._in_llm_phase:
            self._llm_phase_started_at = _now_iso()
            self._in_llm_phase = True
        self._pending_tool_starts[ev.id] = _now_iso()

    def _on_tool_call_complete(self, ev: ToolCallComplete) -> None:
        if not self._telemetry_active():
            return
        started_at = self._pending_tool_starts.pop(ev.id, _now_iso())
        self._pending_tool_data[ev.id] = (ev.name, ev.input, started_at)

    def _on_tool_result(self, ev: ToolResultEvent) -> None:
        if not self._telemetry_active():
            return
        from toolforge_telemetry.production import (
            DAGEdge, LLMMeta, NestedLLMCall, Span,
        )

        data = self._pending_tool_data.pop(ev.id, None)
        if data is None:
            return

        name, input_data, started_at = data
        meta: dict[str, Any] = self._tool_meta_queue.popleft() if self._tool_meta_queue else {}

        ended_at = _now_iso()
        duration_ms: float | None = meta.get("duration_ms")

        span_id = f"sp_{ev.id}"

        status: str = "error" if (ev.is_error or ev.result.startswith("ERROR:")) else "ok"

        try:
            output_val: Any = json.loads(ev.result) if ev.result and not ev.is_error else None
        except (json.JSONDecodeError, ValueError):
            output_val = ev.result

        raw_nlc: list[dict] = meta.get("nested_llm_calls") or []
        nested = []
        for nlc in raw_nlc:
            try:
                nested.append(NestedLLMCall(**nlc))
                self._tool_tokens_in += nlc.get("tokens_in", 0)
                self._tool_tokens_out += nlc.get("tokens_out", 0)
            except TypeError:
                pass

        parent_spans = [self._last_llm_span_id] if self._last_llm_span_id else []

        span = Span(
            span_id=span_id,
            task_id=self._task_id,            # type: ignore[arg-type]
            run_id=self._run_id,              # type: ignore[arg-type]
            usecase_id=self._usecase_id,      # type: ignore[arg-type]
            type="tool_call",
            llm_turn_index=self._llm_turn_index,
            call_index_in_turn=self._call_index_in_turn,
            parent_spans=parent_spans,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=status,  # type: ignore[arg-type]
            tool_id=name,
            tool_version=meta.get("tool_version"),
            input=input_data,
            output=output_val,
            retries=[],
            nested_llm_calls=nested,
        )

        self._prod_store.record_span(span)  # type: ignore[union-attr]
        self._all_span_ids.append(span_id)
        self._current_turn_tool_span_ids.append(span_id)
        self._call_index_in_turn += 1
        self._tool_call_count += 1
        self._in_llm_phase = False

        if self._last_llm_span_id:
            self._dag_edges.append(
                DAGEdge(from_span=self._last_llm_span_id, to_span=span_id, via="control")
            )

    def _on_message_complete(self, ev: MessageComplete) -> None:
        if not self._telemetry_active():
            return
        from toolforge_telemetry.production import DAGEdge, LLMMeta, Span

        self._llm_turn_index += 1
        span_id = f"sp_llm_{secrets.token_hex(6)}"
        ended_at = _now_iso()
        started_at = self._llm_phase_started_at or ended_at

        # History at this moment (before agent appends the new message) is the
        # exact prompt sent to the LLM for this call.
        ph = _prompt_hash(self._agent.system_prompt, self._agent.history)

        usage = ev.usage or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        self._agent_tokens_in += tokens_in
        self._agent_tokens_out += tokens_out

        # parent_spans = tools that fed this LLM call (from previous tool batch)
        parent_spans = list(self._prev_turn_tool_span_ids)

        # Serialize the response content
        content = ev.message.content
        if isinstance(content, str):
            response_dict: dict[str, Any] = {"role": "assistant", "content": content}
        else:
            response_dict = {"role": "assistant", "content": _msg_content_to_dict(content)}

        # Serialize history for the messages field
        messages_snapshot = [
            {"role": m.role, "content": _msg_content_to_dict(m.content)}
            for m in self._agent.history
        ]

        span = Span(
            span_id=span_id,
            task_id=self._task_id,          # type: ignore[arg-type]
            run_id=self._run_id,            # type: ignore[arg-type]
            usecase_id=self._usecase_id,    # type: ignore[arg-type]
            type="llm_call",
            llm_turn_index=self._llm_turn_index,
            call_index_in_turn=0,
            parent_spans=parent_spans,
            started_at=started_at,
            ended_at=ended_at,
            status="ok",
            system_prompt=self._agent.system_prompt,
            messages=messages_snapshot,
            response=response_dict,
            llm_meta=LLMMeta(
                model=self._agent.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                prompt_hash=ph,
            ),
        )

        self._prod_store.record_span(span)  # type: ignore[union-attr]
        self._all_span_ids.append(span_id)
        self._last_llm_span_id = span_id
        self._llm_phase_started_at = None
        self._in_llm_phase = False

        # DAG: prev tools → this LLM
        for tool_span_id in parent_spans:
            self._dag_edges.append(
                DAGEdge(from_span=tool_span_id, to_span=span_id, via="data")
            )

        # Rotate turn tracking: the tools that ran BETWEEN the previous
        # MessageComplete and this one become the input for the NEXT LLM call.
        self._prev_turn_tool_span_ids = list(self._current_turn_tool_span_ids)
        self._current_turn_tool_span_ids = []
        self._call_index_in_turn = 0

    def _on_pause_for_user(self, ev: PauseForUserEvent) -> None:
        if not self._telemetry_active():
            return
        from toolforge_telemetry.production import Span

        span_id = f"sp_wait_{secrets.token_hex(6)}"
        parent_spans = [self._last_llm_span_id] if self._last_llm_span_id else []

        self._pending_user_wait_span = Span(
            span_id=span_id,
            task_id=self._task_id,          # type: ignore[arg-type]
            run_id=self._run_id,            # type: ignore[arg-type]
            usecase_id=self._usecase_id,    # type: ignore[arg-type]
            type="user_wait",
            parent_spans=parent_spans,
            started_at=_now_iso(),
            user_turn=self._user_turn + 1,
        )

    # ------------------------------------------------------------------
    # Telemetry-wrapped generator
    # ------------------------------------------------------------------

    async def _telemetry_wrap(
        self,
        inner: AsyncGenerator[StreamEvent, None],
        user_message: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        text_buf: list[str] = []

        # Open / resume phase
        self._ensure_task_open(user_message)
        self._record_input_entry(user_message)
        self._close_pending_user_wait()

        # Reset LLM phase timing for this new turn
        self._llm_phase_started_at = _now_iso()
        self._in_llm_phase = True

        async for ev in inner:
            yield ev

            if isinstance(ev, TextDelta):
                text_buf.append(ev.text)
                if not self._in_llm_phase:
                    self._llm_phase_started_at = _now_iso()
                    self._in_llm_phase = True

            elif isinstance(ev, ToolCallStart):
                self._on_tool_call_start(ev)
                if text_buf:
                    self._last_text_output = "".join(text_buf)
                    text_buf = []

            elif isinstance(ev, ToolCallComplete):
                self._on_tool_call_complete(ev)

            elif isinstance(ev, ToolResultEvent):
                self._on_tool_result(ev)
                # Next LLM call starts right after all tool results are processed.
                self._llm_phase_started_at = _now_iso()
                self._in_llm_phase = True

            elif isinstance(ev, MessageComplete):
                if text_buf:
                    self._last_text_output = "".join(text_buf)
                    text_buf = []
                self._on_message_complete(ev)

            elif isinstance(ev, PauseForUserEvent):
                if text_buf:
                    self._last_text_output = "".join(text_buf)
                    text_buf = []
                self._on_pause_for_user(ev)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_task(self, task: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream all events for a complete task execution (tool calls included)."""
        tools = await self._provider.refresh_tools()
        inner = await self._agent.run_turn(
            task,
            tools=tools,
            tool_handler=self._provider.call_tool,
        )
        if self._telemetry_active():
            return self._telemetry_wrap(inner, task)
        return inner

    async def run_turn(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Alias for interactive / multi-turn use."""
        return await self.run_task(user_message)

    def record_feedback(
        self, *, explicit: str, correction_text: str | None = None
    ) -> None:
        """Attach explicit user feedback to the current task (production only).

        ``explicit`` is one of ``thumbs_up`` | ``correction`` | ``none``.
        No-op when telemetry is inactive. If the task is not open yet, the
        feedback is buffered and flushed when the session closes.
        """
        if self._prod_store is None or self._task_id is None:
            return
        if not self._task_opened or self._task_closed:
            self._pending_feedback = (explicit, correction_text)
            return
        from toolforge_telemetry.production import UserFeedback

        self._prod_store.record_user_feedback(
            self._task_id,  # type: ignore[arg-type]
            UserFeedback(explicit=explicit, correction_text=correction_text),  # type: ignore[arg-type]
        )
        self._pending_feedback = None

    def close_session(self, *, status: str = "success") -> None:
        """Finalise the task record. Call once when the conversation ends.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._telemetry_active() or not self._task_opened or self._task_closed:
            return
        from toolforge_telemetry.production import DAG, DAGEdge, TaskCost

        self._task_closed = True
        self._task_ended_at = _now_iso()

        # Close any pending user_wait span (e.g. session closed while waiting).
        if self._pending_user_wait_span is not None:
            span = self._pending_user_wait_span
            span.ended_at = self._task_ended_at
            self._prod_store.record_span(span)  # type: ignore[union-attr]
            self._pending_user_wait_span = None

        cost = TaskCost(
            agent_tokens_in=self._agent_tokens_in,
            agent_tokens_out=self._agent_tokens_out,
            tool_tokens_in=self._tool_tokens_in,
            tool_tokens_out=self._tool_tokens_out,
            tool_calls=self._tool_call_count,
            latency_ms=(
                (
                    datetime.fromisoformat(self._task_ended_at)
                    - datetime.fromisoformat(self._task_started_at)  # type: ignore[arg-type]
                ).total_seconds() * 1000
                if self._task_started_at
                else 0.0
            ),
        )

        dag = DAG(
            nodes=list(self._all_span_ids),
            edges=list(self._dag_edges),
        )

        self._prod_store.close_task(  # type: ignore[union-attr]
            self._task_id,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            final_output=self._last_text_output,
            cost=cost,
            dag=dag,
            ended_at=self._task_ended_at,
        )

        # Flush feedback recorded before the task was open (now that it exists).
        if self._pending_feedback is not None:
            from toolforge_telemetry.production import UserFeedback

            explicit, correction_text = self._pending_feedback
            self._prod_store.record_user_feedback(  # type: ignore[union-attr]
                self._task_id,  # type: ignore[arg-type]
                UserFeedback(explicit=explicit, correction_text=correction_text),  # type: ignore[arg-type]
            )
            self._pending_feedback = None


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def consumer_agent_stdio(
    *,
    stdio_params: StdioServerParameters,
    agent: LLMAgent,
    prod_store: ProductionTelemetryStore | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    usecase_id: str | None = None,
) -> AsyncIterator[ConsumerAgent]:
    """Open a stdio-transport MCP connection to the Usecase server and yield a ConsumerAgent."""
    async with stdio_client(stdio_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            provider = MCPToolProvider(session)
            yield ConsumerAgent(
                agent=agent,
                provider=provider,
                prod_store=prod_store,
                task_id=task_id,
                run_id=run_id,
                usecase_id=usecase_id,
            )


@asynccontextmanager
async def consumer_agent_sse(
    *,
    url: str,
    agent: LLMAgent,
    prod_store: ProductionTelemetryStore | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    usecase_id: str | None = None,
) -> AsyncIterator[ConsumerAgent]:
    """Open an SSE-transport MCP connection to the Usecase server and yield a ConsumerAgent."""
    from mcp.client.sse import sse_client  # lazy — avoids httpx dep if not used

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            provider = MCPToolProvider(session)
            yield ConsumerAgent(
                agent=agent,
                provider=provider,
                prod_store=prod_store,
                task_id=task_id,
                run_id=run_id,
                usecase_id=usecase_id,
            )
