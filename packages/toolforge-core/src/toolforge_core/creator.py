"""Creator agent — wraps LLMAgent + MCPToolProvider with connection context managers."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .agent import LLMAgent
from .mcp_client import MCPToolProvider
from .types import Message, StreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class CreatorAgent:
    """LLM agent whose tools come from a live MCP server."""

    def __init__(self, agent: LLMAgent, provider: MCPToolProvider) -> None:
        self._agent = agent
        self._provider = provider

    @property
    def history(self) -> list[Message]:
        return self._agent.history

    def reset_history(self) -> None:
        self._agent.reset_history()

    async def run_turn(
        self,
        user_message: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        tools = await self._provider.refresh_tools()
        return await self._agent.run_turn(
            user_message,
            tools=tools,
            tool_handler=self._provider.call_tool,
        )


@asynccontextmanager
async def creator_agent_stdio(
    *,
    stdio_params: StdioServerParameters,
    agent: LLMAgent,
) -> AsyncIterator[CreatorAgent]:
    """Open a stdio-transport MCP connection and yield a ready CreatorAgent."""
    async with stdio_client(stdio_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            provider = MCPToolProvider(session)
            yield CreatorAgent(agent=agent, provider=provider)


@asynccontextmanager
async def creator_agent_sse(
    *,
    url: str,
    agent: LLMAgent,
) -> AsyncIterator[CreatorAgent]:
    """Open an SSE-transport MCP connection and yield a ready CreatorAgent."""
    from mcp.client.sse import sse_client  # lazy — avoids httpx dep if not used

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            provider = MCPToolProvider(session)
            yield CreatorAgent(agent=agent, provider=provider)
