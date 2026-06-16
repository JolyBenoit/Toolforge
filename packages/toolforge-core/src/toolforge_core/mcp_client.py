"""MCP tool provider — wraps a ClientSession to supply tools and dispatch calls."""
from __future__ import annotations

import json
from typing import Any, Callable

from mcp.client.session import ClientSession


class ToolError(RuntimeError):
    """Raised when a tool execution returns an error result via MCP."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


def _mcp_tool_to_dict(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema if tool.inputSchema is not None else {},
    }


class MCPToolProvider:
    """Fetches tool definitions and dispatches calls through an MCP ClientSession.

    ``on_tool_telemetry`` is an optional callback invoked when a production
    telemetry envelope (``__tf__`` key) is detected in the tool response.
    Signature: ``on_tool_telemetry(tool_name: str, meta: dict) -> None``

    The envelope is stripped before returning the result to the caller, so the
    LLM never sees it.
    """

    def __init__(
        self,
        session: ClientSession,
        on_tool_telemetry: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._session = session
        self._on_tool_telemetry = on_tool_telemetry

    async def refresh_tools(self) -> list[dict[str, Any]]:
        result = await self._session.list_tools()
        return [_mcp_tool_to_dict(t) for t in result.tools]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, args)
        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        text = "\n".join(parts) if parts else ""

        # Detect and strip the production telemetry envelope.
        if self._on_tool_telemetry is not None and text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "__tf__" in parsed:
                    self._on_tool_telemetry(name, parsed["__tf__"])
                    output_val = parsed.get("output")
                    text = json.dumps(output_val, ensure_ascii=False) if output_val is not None else ""
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        if getattr(result, "isError", None) is True:
            raise ToolError(name, text or "tool returned an error with no message")
        return text
