"""Low-level MCP Server — uses custom inputSchema per tool loaded from the registry."""
from __future__ import annotations

from typing import Any

import mcp.types as types
from mcp.server import Server

from ._handlers import RunContext, h_call_tool, h_list_tools


def create_server(ctx: RunContext) -> Server:
    server = Server(f"toolforge-usecase/{ctx.usecase_id}/{ctx.run_id}")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tool_dicts = await h_list_tools(ctx)
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in tool_dicts
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any] | None,
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        try:
            output = await h_call_tool(ctx, name, arguments or {})
            return [types.TextContent(type="text", text=output)]
        except Exception as exc:  # noqa: BLE001
            return [types.TextContent(type="text", text=f"ERROR: {exc}")]

    return server
