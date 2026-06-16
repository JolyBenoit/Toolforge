from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx
import openai

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
from .base import LLMClient


class OpenAICompatClient(LLMClient):
    """Handles OpenAI, Z.ai, OpenRouter, local vLLM — anything OpenAI-compatible."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._timeout = timeout
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=httpx.Timeout(connect=15.0, read=timeout, write=30.0, pool=10.0),
        )

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
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        oai_messages.extend(_to_openai_messages(messages))

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": oai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        accumulated_text = ""
        # index -> {id, name, args}
        tool_buffers: dict[int, dict[str, Any]] = {}
        stop_reason = "stop"
        usage: dict[str, int] | None = None

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.APITimeoutError as exc:
            raise TimeoutError(
                f"Model '{model}' did not respond within {self._timeout}s"
            ) from exc
        except openai.APIConnectionError as exc:
            raise ConnectionError(f"Could not reach LLM API: {exc}") from exc
        except openai.RateLimitError as exc:
            raise RuntimeError(f"Rate limit reached: {exc}") from exc

        try:
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    accumulated_text += delta.content
                    yield TextDelta(text=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_buffers:
                            tool_id = tc.id or ""
                            tool_name = tc.function.name if tc.function else ""
                            tool_buffers[idx] = {"id": tool_id, "name": tool_name, "args": ""}
                            yield ToolCallStart(id=tool_id, name=tool_name)
                        if tc.function and tc.function.arguments:
                            tool_buffers[idx]["args"] += tc.function.arguments
                            yield ToolCallInputDelta(
                                id=tool_buffers[idx]["id"],
                                json_delta=tc.function.arguments,
                            )

                if choice.finish_reason:
                    stop_reason = choice.finish_reason

                # usage arrives in the final chunk (stream_options include_usage)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
        except openai.APITimeoutError as exc:
            raise TimeoutError(
                f"Model '{model}' stopped responding (read timeout)"
            ) from exc
        except openai.APIConnectionError as exc:
            raise ConnectionError(f"LLM API connection lost: {exc}") from exc

        content: list[MessageContent] = []
        if accumulated_text:
            content.append(TextContent(text=accumulated_text))

        for idx in sorted(tool_buffers):
            buf = tool_buffers[idx]
            tool_input: dict[str, Any] = json.loads(buf["args"]) if buf["args"] else {}
            yield ToolCallComplete(id=buf["id"], name=buf["name"], input=tool_input)
            content.append(ToolUseContent(id=buf["id"], name=buf["name"], input=tool_input))

        yield MessageComplete(
            stop_reason=stop_reason,
            message=Message(role="assistant", content=content),
            usage=usage,
        )


# --- format converters ---


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        if msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for item in msg.content:
                if isinstance(item, TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, ToolUseContent):
                    tool_calls.append({
                        "id": item.id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": json.dumps(item.input)},
                    })
            oai_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            result.append(oai_msg)

        elif msg.role == "user":
            tool_results = [i for i in msg.content if isinstance(i, ToolResultContent)]
            text_items = [i for i in msg.content if isinstance(i, TextContent)]
            if tool_results and not text_items:
                for tr in tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": tr.content,
                    })
            else:
                combined = " ".join(t.text for t in text_items)
                result.append({"role": "user", "content": combined})

    return result


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }
