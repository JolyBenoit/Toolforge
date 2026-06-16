"""JudgeLLM — the static judge's LLM access.

Built from the very same pieces as the Creator (an ``AgentLLMConfig`` and an
``LLMClient`` from ``toolforge_core``), so the judge runs on the same model
backend. Unlike the Creator it is **stateless**: every call is one fresh,
single-message completion with no tools, so each task is judged in isolation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from toolforge_core.config import AgentLLMConfig, load_system_prompt
from toolforge_core.llm.base import LLMClient
from toolforge_core.types import Message, TextDelta


@runtime_checkable
class JudgeLLM(Protocol):
    """Minimal interface the judge needs: a model name and a text completion."""

    model: str

    async def complete(self, user_message: str) -> str: ...


class AgentLLMJudge:
    """Stateless structured-completion wrapper over an ``LLMClient``."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_config(cls, config: AgentLLMConfig, client: LLMClient) -> AgentLLMJudge:
        return cls(
            client=client,
            model=config.model,
            system_prompt=load_system_prompt(config.system_prompt_file),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    async def complete(self, user_message: str) -> str:
        """Run one stateless completion and return the concatenated text."""
        history = [Message(role="user", content=user_message)]
        parts: list[str] = []
        async for event in self.client.stream(
            history,
            system=self.system_prompt,
            tools=[],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ):
            if isinstance(event, TextDelta):
                parts.append(event.text)
        return "".join(parts)
