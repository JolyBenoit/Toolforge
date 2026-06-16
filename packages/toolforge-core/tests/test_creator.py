"""Tests for MCPToolProvider and CreatorAgent — no real MCP server required."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from toolforge_core.mcp_client import MCPToolProvider, _mcp_tool_to_dict
from toolforge_core.creator import CreatorAgent
from toolforge_core.types import MessageComplete, StreamEvent, TextDelta


# --- helpers ---


def _make_mcp_tool(name: str, description: str = "desc", schema: dict | None = None) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {}}
    return tool


def _make_content_item(text: str) -> Any:
    item = MagicMock()
    item.text = text
    return item


# --- _mcp_tool_to_dict ---


def test_mcp_tool_to_dict_full() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tool = _make_mcp_tool("add", "Add numbers", schema)
    result = _mcp_tool_to_dict(tool)
    assert result == {"name": "add", "description": "Add numbers", "input_schema": schema}


def test_mcp_tool_to_dict_none_description() -> None:
    tool = _make_mcp_tool("add")
    tool.description = None
    result = _mcp_tool_to_dict(tool)
    assert result["description"] == ""


def test_mcp_tool_to_dict_none_schema() -> None:
    tool = _make_mcp_tool("add")
    tool.inputSchema = None
    result = _mcp_tool_to_dict(tool)
    assert result["input_schema"] == {}


# --- MCPToolProvider ---


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    # list_tools returns an object with .tools list
    list_result = MagicMock()
    list_result.tools = [_make_mcp_tool("propose_tool"), _make_mcp_tool("validate_in_sandbox")]
    session.list_tools.return_value = list_result
    # call_tool returns an object with .content list
    call_result = MagicMock()
    call_result.content = [_make_content_item("Tool proposed: v1")]
    session.call_tool.return_value = call_result
    return session


async def test_refresh_tools_returns_dicts(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    tools = await provider.refresh_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "propose_tool"
    assert tools[1]["name"] == "validate_in_sandbox"
    assert isinstance(tools[0]["input_schema"], dict)


async def test_call_tool_joins_content(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    result = await provider.call_tool("propose_tool", {"name": "t"})
    assert result == "Tool proposed: v1"
    mock_session.call_tool.assert_called_once_with("propose_tool", {"name": "t"})


async def test_call_tool_empty_content(mock_session: AsyncMock) -> None:
    call_result = MagicMock()
    call_result.content = []
    mock_session.call_tool.return_value = call_result
    provider = MCPToolProvider(mock_session)
    result = await provider.call_tool("x", {})
    assert result == ""


async def test_call_tool_multi_content(mock_session: AsyncMock) -> None:
    call_result = MagicMock()
    call_result.content = [_make_content_item("line1"), _make_content_item("line2")]
    mock_session.call_tool.return_value = call_result
    provider = MCPToolProvider(mock_session)
    result = await provider.call_tool("x", {})
    assert result == "line1\nline2"


# --- CreatorAgent ---


class _MockLLMAgent:
    """Minimal stand-in that records what run_turn received."""

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
            yield TextDelta(text="hello")
            yield MessageComplete(
                stop_reason="end_turn",
                message=MagicMock(),
            )

        return _gen()


async def test_creator_agent_passes_tools(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    agent = CreatorAgent(agent=inner, provider=provider)
    gen = await agent.run_turn("propose a tool")
    events = [e async for e in gen]
    assert len(events) == 2
    assert isinstance(events[0], TextDelta)
    # tools came from mock_session.list_tools (2 tools)
    assert len(inner._last_tools) == 2
    # handler is provider.call_tool (use == since bound methods aren't identical)
    assert inner._last_handler == provider.call_tool


async def test_creator_agent_reset_history(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    inner.history = [MagicMock()]
    agent = CreatorAgent(agent=inner, provider=provider)
    agent.reset_history()
    assert inner.history == []


def test_creator_agent_history_proxy(mock_session: AsyncMock) -> None:
    provider = MCPToolProvider(mock_session)
    inner = _MockLLMAgent()
    sentinel = [MagicMock()]
    inner.history = sentinel
    agent = CreatorAgent(agent=inner, provider=provider)
    assert agent.history is sentinel
