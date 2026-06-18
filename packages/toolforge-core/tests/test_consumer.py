"""Tests for ConsumerAgent — no real MCP server required."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from toolforge_core.consumer import ConsumerAgent
from toolforge_core.mcp_client import MCPToolProvider
from toolforge_core.types import MessageComplete, StreamEvent, TextDelta, ToolCallComplete


# --- helpers ---


def _make_mcp_tool(name: str, schema: dict | None = None) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = f"{name} description"
    tool.inputSchema = schema or {"type": "object"}
    return tool


def _make_content(text: str) -> Any:
    item = MagicMock()
    item.text = text
    return item


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    list_result = MagicMock()
    list_result.tools = [_make_mcp_tool("extract"), _make_mcp_tool("transform")]
    session.list_tools.return_value = list_result
    call_result = MagicMock()
    call_result.content = [_make_content("done")]
    session.call_tool.return_value = call_result
    return session


class _MockLLMAgent:
    def __init__(self) -> None:
        self.history: list = []
        self._last_tools: list = []
        self._last_handler: Any = None

    def reset_history(self) -> None:
        self.history = []

    async def run_turn(
        self,
        user_message: str,
        *,
        tools: list | None = None,
        tool_handler: Any = None,
    ):
        self._last_tools = tools or []
        self._last_handler = tool_handler

        async def _gen():
            yield TextDelta(text="Task complete.")
            yield MessageComplete(stop_reason="end_turn", message=MagicMock())

        return _gen()


class _FakeProdStore:
    """Captures production telemetry calls for assertions."""

    def __init__(self) -> None:
        self.feedback: list[tuple[str, str | None]] = []
        self.closed: list[str] = []

    def open_task(self, *a: Any, **k: Any) -> None: ...
    def append_input_entry(self, *a: Any, **k: Any) -> None: ...
    def record_span(self, *a: Any, **k: Any) -> None: ...

    def close_task(self, task_id: str, *, status: str, **k: Any) -> None:
        self.closed.append(status)

    def record_user_feedback(self, task_id: str, feedback: Any) -> None:
        self.feedback.append((feedback.explicit, feedback.correction_text))


def _prod_agent(mock_session: AsyncMock, store: _FakeProdStore) -> ConsumerAgent:
    return ConsumerAgent(
        agent=_MockLLMAgent(),
        provider=MCPToolProvider(mock_session),
        prod_store=store,  # type: ignore[arg-type]
        task_id="task_1",
        run_id="run_1",
        usecase_id="uc_1",
    )


# --- ConsumerAgent.record_feedback ---


def test_record_feedback_noop_without_prod_store(mock_session: AsyncMock) -> None:
    agent = ConsumerAgent(agent=_MockLLMAgent(), provider=MCPToolProvider(mock_session))
    agent.record_feedback(explicit="thumbs_up")  # must not raise
    assert agent._pending_feedback is None


def test_record_feedback_immediate_when_task_open(
    mock_session: AsyncMock,
) -> None:
    store = _FakeProdStore()
    agent = _prod_agent(mock_session, store)
    agent._task_opened = True
    agent._task_started_at = "2026-01-01T00:00:00+00:00"

    agent.record_feedback(explicit="thumbs_up")

    assert store.feedback == [("thumbs_up", None)]
    assert agent._pending_feedback is None


def test_record_feedback_buffered_then_flushed_on_close(
    mock_session: AsyncMock,
) -> None:
    store = _FakeProdStore()
    agent = _prod_agent(mock_session, store)

    # Task not open yet → feedback is buffered, not persisted.
    agent.record_feedback(explicit="correction", correction_text="wrong total")
    assert store.feedback == []
    assert agent._pending_feedback == ("correction", "wrong total")

    # Closing the session flushes the buffered feedback.
    agent._task_opened = True
    agent._task_started_at = "2026-01-01T00:00:00+00:00"
    agent.close_session(status="failed")

    assert store.closed == ["failed"]
    assert store.feedback == [("correction", "wrong total")]
    assert agent._pending_feedback is None


# --- ConsumerAgent.run_task ---


async def test_run_task_fetches_tools(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    agent = ConsumerAgent(agent=inner, provider=provider)
    gen = await agent.run_task("extract all invoices")
    events = [e async for e in gen]
    assert len(events) == 2
    assert isinstance(events[0], TextDelta)
    assert len(inner._last_tools) == 2
    assert inner._last_tools[0]["name"] == "extract"


async def test_run_task_passes_tool_handler(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    agent = ConsumerAgent(agent=inner, provider=provider)
    await agent.run_task("do something")
    assert inner._last_handler == provider.call_tool


async def test_run_turn_is_alias_for_run_task(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    agent = ConsumerAgent(agent=inner, provider=provider)
    gen = await agent.run_turn("hello")
    events = [e async for e in gen]
    assert isinstance(events[0], TextDelta)
    assert len(inner._last_tools) == 2


# --- history proxy ---


def test_history_proxy(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    sentinel = [MagicMock()]
    inner.history = sentinel
    agent = ConsumerAgent(agent=inner, provider=provider)
    assert agent.history is sentinel


def test_reset_history(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    inner.history = [MagicMock()]
    agent = ConsumerAgent(agent=inner, provider=provider)
    agent.reset_history()
    assert inner.history == []


# --- ConsumerAgent.new_session ---


def test_new_session_assigns_new_task_id_and_resets_state(
    mock_session: AsyncMock,
) -> None:
    store = _FakeProdStore()
    agent = _prod_agent(mock_session, store)
    assert agent.task_id == "task_1"

    # Simulate an active task with accumulated state.
    agent._task_opened = True
    agent._task_closed = True
    agent._user_turn = 3
    agent._all_span_ids = ["sp_a", "sp_b"]
    agent._agent_tokens_in = 100
    agent.history.append(MagicMock())

    new_id = agent.new_session()

    assert new_id is not None
    assert new_id != "task_1"
    assert agent.task_id == new_id
    # State fully reset for the new line.
    assert agent._task_opened is False
    assert agent._task_closed is False
    assert agent._user_turn == 0
    assert agent._all_span_ids == []
    assert agent._agent_tokens_in == 0
    assert agent.history == []


def test_new_session_without_prod_store_keeps_task_id_none(
    mock_session: AsyncMock,
) -> None:
    agent = ConsumerAgent(agent=_MockLLMAgent(), provider=MCPToolProvider(mock_session))
    agent.history.append(MagicMock())
    assert agent.new_session() is None
    assert agent.task_id is None
    assert agent.history == []


def test_close_then_new_session_opens_a_distinct_line(
    mock_session: AsyncMock,
) -> None:
    store = _FakeProdStore()
    agent = _prod_agent(mock_session, store)
    agent._task_opened = True
    agent._task_started_at = "2026-01-01T00:00:00+00:00"

    agent.close_session(status="success")
    first_id = agent.task_id
    agent.new_session()

    assert store.closed == ["success"]
    assert agent.task_id != first_id
    assert agent._task_closed is False  # rearmed for the next run_task()


# --- multi-turn accumulates history ---


async def test_multiple_run_task_calls_accumulate(mock_session: AsyncMock) -> None:
    """Verify that each run_task call is independent (history managed by inner agent)."""
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    agent = ConsumerAgent(agent=inner, provider=provider)

    for _ in range(3):
        gen = await agent.run_task("step")
        async for _ in gen:
            pass

    # inner agent's run_turn was called 3 times with 2 tools each time
    assert len(inner._last_tools) == 2
