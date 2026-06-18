from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import anthropic

from toolforge_core.types import (
    Message,
    MessageContent,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    TextDelta,
    ToolCallStart,
    ToolCallInputDelta,
    ToolCallComplete,
    MessageComplete,
    StreamEvent,
)
from .base import LLMClient, LLMRateLimitError


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def stream(
        self,
        messages: list[Message],
        *,
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[StreamEvent, None]:
        return self._stream(
            messages,
            system=system,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _stream(
        self,
        messages: list[Message],
        *,
        system: str,
        tools: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[StreamEvent, None]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [_to_anthropic_message(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        # block_content[index] = assembled MessageContent after block_stop
        text_by_block: dict[int, str] = {}
        tool_by_block: dict[int, dict[str, Any]] = {}
        block_content: dict[int, MessageContent] = {}
        stop_reason = "end_turn"
        input_tokens: int = 0
        output_tokens: int = 0

        # Enter the stream manually so a 429 on the initial request surfaces as a
        # typed LLMRateLimitError (the callers retry on it); the try/finally then
        # guarantees the stream is closed once iteration is done or interrupted.
        stream_ctx = self._client.messages.stream(**kwargs)
        try:
            stream = await stream_ctx.__aenter__()
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(f"Rate limit reached: {exc}") from exc
        try:
            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens or 0

                elif event.type == "content_block_start":
                    cb = event.content_block
                    if cb.type == "text":
                        text_by_block[event.index] = ""
                    elif cb.type == "tool_use":
                        tool_by_block[event.index] = {"id": cb.id, "name": cb.name, "args": ""}
                        yield ToolCallStart(id=cb.id, name=cb.name)

                elif event.type == "content_block_delta":
                    d = event.delta
                    if d.type == "text_delta":
                        text_by_block[event.index] += d.text
                        yield TextDelta(text=d.text)
                    elif d.type == "input_json_delta":
                        buf = tool_by_block[event.index]
                        buf["args"] += d.partial_json
                        yield ToolCallInputDelta(id=buf["id"], json_delta=d.partial_json)

                elif event.type == "content_block_stop":
                    idx = event.index
                    if idx in text_by_block:
                        text = text_by_block.pop(idx)
                        if text:
                            block_content[idx] = TextContent(text=text)
                    elif idx in tool_by_block:
                        buf = tool_by_block.pop(idx)
                        tool_input: dict[str, Any] = json.loads(buf["args"]) if buf["args"] else {}
                        yield ToolCallComplete(id=buf["id"], name=buf["name"], input=tool_input)
                        block_content[idx] = ToolUseContent(
                            id=buf["id"], name=buf["name"], input=tool_input
                        )

                elif event.type == "message_delta":
                    stop_reason = event.delta.stop_reason or "end_turn"
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens or 0

                elif event.type == "message_stop":
                    sorted_content = [block_content[i] for i in sorted(block_content)]
                    usage = (
                        {"input_tokens": input_tokens, "output_tokens": output_tokens}
                        if input_tokens or output_tokens
                        else None
                    )
                    yield MessageComplete(
                        stop_reason=stop_reason,
                        message=Message(role="assistant", content=sorted_content),
                        usage=usage,
                    )
        finally:
            await stream_ctx.__aexit__(None, None, None)


# --- format converters ---


def _to_anthropic_message(msg: Message) -> dict[str, Any]:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}
    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if isinstance(item, TextContent):
            parts.append({"type": "text", "text": item.text})
        elif isinstance(item, ToolUseContent):
            parts.append({"type": "tool_use", "id": item.id, "name": item.name, "input": item.input})
        elif isinstance(item, ToolResultContent):
            parts.append({
                "type": "tool_result",
                "tool_use_id": item.tool_use_id,
                "content": item.content,
                "is_error": item.is_error,
            })
    return {"role": msg.role, "content": parts}


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
    }
