from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TextContent:
    text: str
    type: str = field(default="text", init=False, repr=False)


@dataclass
class ToolUseContent:
    id: str
    name: str
    input: dict[str, Any]
    type: str = field(default="tool_use", init=False, repr=False)


@dataclass
class ToolResultContent:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = field(default="tool_result", init=False, repr=False)


MessageContent = TextContent | ToolUseContent | ToolResultContent


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str | list[MessageContent]


# --- Stream events emitted by LLMClient.stream() ---


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStart:
    id: str
    name: str


@dataclass
class ToolCallInputDelta:
    id: str
    json_delta: str


@dataclass
class ToolCallComplete:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultEvent:
    id: str
    name: str
    result: str  # raw string returned by the tool handler
    is_error: bool = False


@dataclass
class MessageComplete:
    stop_reason: str
    message: Message
    usage: dict[str, int] | None = None  # {"input_tokens": N, "output_tokens": M}


@dataclass
class PauseForUserEvent:
    """Emitted when a tool result contains __pause_for_user__: true.

    Signals the agent loop to stop before the next LLM call and hand control
    back to the user. History is preserved so the next run_turn() continues
    from the same context.
    """
    tool_name: str
    message: str = ""  # optional display message extracted from the tool result


StreamEvent = (
    TextDelta
    | ToolCallStart
    | ToolCallInputDelta
    | ToolCallComplete
    | ToolResultEvent
    | MessageComplete
    | PauseForUserEvent
)
