from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable

from toolforge_core.config import AgentLLMConfig, load_system_prompt
from toolforge_core.llm.base import LLMClient
from toolforge_core.types import (
    Message,
    MessageComplete,
    PauseForUserEvent,
    ToolCallComplete,
    ToolResultContent,
    ToolResultEvent,
    StreamEvent,
)

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


@dataclass
class LLMAgent:
    client: LLMClient
    model: str
    system_prompt: str
    max_tokens: int
    temperature: float
    tools: list[dict[str, Any]] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: AgentLLMConfig, client: LLMClient) -> LLMAgent:
        system_prompt = load_system_prompt(config.system_prompt_file)
        return cls(
            client=client,
            model=config.model,
            system_prompt=system_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    def reset_history(self) -> None:
        self.history = []

    async def run_turn(
        self,
        user_message: str,
        tools: list[dict[str, Any]] | None = None,
        tool_handler: ToolHandler | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        return self._run_turn(user_message, tools=tools, tool_handler=tool_handler)

    async def _run_turn(
        self,
        user_message: str,
        tools: list[dict[str, Any]] | None,
        tool_handler: ToolHandler | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        effective_tools = tools if tools is not None else self.tools
        self.history.append(Message(role="user", content=user_message))

        while True:
            pending_tool_calls: list[ToolCallComplete] = []

            async for event in self.client.stream(
                self.history,
                system=self.system_prompt,
                tools=effective_tools,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ):
                yield event
                if isinstance(event, ToolCallComplete):
                    pending_tool_calls.append(event)
                elif isinstance(event, MessageComplete):
                    self.history.append(event.message)

            if not pending_tool_calls or tool_handler is None:
                break

            tool_results = []
            pause_event: PauseForUserEvent | None = None
            for tc in pending_tool_calls:
                try:
                    result_str = await tool_handler(tc.name, tc.input)
                    yield ToolResultEvent(id=tc.id, name=tc.name, result=result_str)
                    tool_results.append(
                        ToolResultContent(tool_use_id=tc.id, content=result_str)
                    )
                    # Detect __pause_for_user__ — finish the full batch first
                    if pause_event is None:
                        try:
                            parsed = json.loads(result_str)
                            if isinstance(parsed, dict) and parsed.get("__pause_for_user__"):
                                pause_event = PauseForUserEvent(
                                    tool_name=tc.name,
                                    message=parsed.get("message", ""),
                                )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                except Exception as exc:
                    error_msg = str(exc)
                    yield ToolResultEvent(
                        id=tc.id, name=tc.name, result=error_msg, is_error=True
                    )
                    tool_results.append(
                        ToolResultContent(
                            tool_use_id=tc.id, content=error_msg, is_error=True
                        )
                    )
            self.history.append(Message(role="user", content=tool_results))
            if pause_event is not None:
                yield pause_event
                return  # Stop before next LLM call — next run_turn() resumes from here
