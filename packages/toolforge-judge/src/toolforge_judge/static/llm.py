"""JudgeLLM — the static judge's LLM access.

Built from the very same pieces as the Creator (an ``AgentLLMConfig`` and an
``LLMClient`` from ``toolforge_core``), so the judge runs on the same model
backend. Unlike the Creator it is **stateless**: every call is one fresh,
single-message completion with no tools, so each task is judged in isolation.
"""
from __future__ import annotations

import asyncio
import random
from typing import Protocol, runtime_checkable

from toolforge_core.config import AgentLLMConfig, load_system_prompt
from toolforge_core.llm.base import LLMClient, LLMRateLimitError
from toolforge_core.types import Message, TextDelta

# Exponential backoff for provider rate limits (HTTP 429). The judges fan out
# many completions at once (e.g. the architecture judge reads every tool's source
# concurrently), so a 429 is expected under load and should be retried, not
# surfaced as a hard failure. Delays are in seconds; jitter is added per attempt.
_RATE_LIMIT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 16.0)


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
        """Run one stateless completion, retrying on provider rate limits.

        The provider raises :class:`LLMRateLimitError` at request start (before
        any text is streamed), so retrying the whole call is safe — no partial
        output can have leaked. After the backoff schedule is exhausted the last
        rate-limit error is re-raised.
        """
        for delay in (*_RATE_LIMIT_BACKOFF, None):
            try:
                return await self._complete_once(user_message)
            except LLMRateLimitError:
                if delay is None:
                    raise
                await asyncio.sleep(delay + random.uniform(0.0, delay / 2))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _complete_once(self, user_message: str) -> str:
        """One stateless completion: concatenate the streamed text."""
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
