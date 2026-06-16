from typing import Any, AsyncGenerator

import pytest

from toolforge_core.agent import LLMAgent
from toolforge_core.llm.base import LLMClient
from toolforge_core.types import (
    Message,
    MessageComplete,
    TextContent,
    TextDelta,
    ToolCallComplete,
    ToolResultContent,
    StreamEvent,
)


class _EchoClient(LLMClient):
    """Returns a single text response with no tool calls."""

    def __init__(self, reply: str = "pong") -> None:
        self._reply = reply

    def stream(self, messages, *, system, tools, model, max_tokens, temperature) -> AsyncGenerator[StreamEvent, None]:
        reply = self._reply

        async def _gen() -> AsyncGenerator[StreamEvent, None]:
            yield TextDelta(text=reply)
            yield MessageComplete(
                stop_reason="end_turn",
                message=Message(role="assistant", content=[TextContent(text=reply)]),
            )

        return _gen()


class _ToolCallingClient(LLMClient):
    """Calls a tool on first turn, then replies normally."""

    def __init__(self) -> None:
        self._call_count = 0

    def stream(self, messages, *, system, tools, model, max_tokens, temperature) -> AsyncGenerator[StreamEvent, None]:
        call_count = self._call_count
        self._call_count += 1

        async def _gen() -> AsyncGenerator[StreamEvent, None]:
            if call_count == 0:
                yield ToolCallComplete(id="tc_1", name="add", input={"a": 1, "b": 2})
                yield MessageComplete(
                    stop_reason="tool_use",
                    message=Message(
                        role="assistant",
                        content=[],
                    ),
                )
            else:
                yield TextDelta(text="result is 3")
                yield MessageComplete(
                    stop_reason="end_turn",
                    message=Message(role="assistant", content=[TextContent(text="result is 3")]),
                )

        return _gen()


def _make_agent(client: LLMClient) -> LLMAgent:
    return LLMAgent(
        client=client,
        model="test-model",
        system_prompt="You are helpful.",
        max_tokens=512,
        temperature=0.0,
    )


async def _collect(agent: LLMAgent, msg: str, **kwargs: Any) -> list[StreamEvent]:
    events = []
    async for event in await agent.run_turn(msg, **kwargs):
        events.append(event)
    return events


async def test_simple_turn_emits_text_and_complete() -> None:
    agent = _make_agent(_EchoClient())
    events = await _collect(agent, "ping")
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, MessageComplete) for e in events)


async def test_history_grows_after_turn() -> None:
    agent = _make_agent(_EchoClient())
    await _collect(agent, "ping")
    assert len(agent.history) == 2  # user + assistant
    assert agent.history[0].role == "user"
    assert agent.history[1].role == "assistant"


async def test_tool_call_dispatched_and_result_in_history() -> None:
    agent = _make_agent(_ToolCallingClient())
    results: list[str] = []

    async def handler(name: str, args: dict[str, Any]) -> str:
        result = str(args["a"] + args["b"])
        results.append(result)
        return result

    events = await _collect(agent, "add 1 and 2", tool_handler=handler)

    assert results == ["3"]
    # history: user, assistant (tool call), user (tool result), assistant (final)
    assert len(agent.history) == 4
    tool_result_msg = agent.history[2]
    assert isinstance(tool_result_msg.content, list)
    assert isinstance(tool_result_msg.content[0], ToolResultContent)
    assert tool_result_msg.content[0].content == "3"

    final_events = [e for e in events if isinstance(e, TextDelta)]
    assert final_events[0].text == "result is 3"


async def test_no_tool_handler_stops_after_first_response() -> None:
    agent = _make_agent(_ToolCallingClient())
    events = await _collect(agent, "add 1 and 2")
    # Without handler, loop exits after first LLM response even if it issued tool calls
    complete_events = [e for e in events if isinstance(e, MessageComplete)]
    assert len(complete_events) == 1


async def test_reset_clears_history() -> None:
    agent = _make_agent(_EchoClient())
    await _collect(agent, "ping")
    agent.reset_history()
    assert agent.history == []
