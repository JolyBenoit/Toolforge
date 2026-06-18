from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from toolforge_core.types import Message, StreamEvent


class LLMRateLimitError(RuntimeError):
    """Raised when the provider rejects a request for rate limiting (HTTP 429).

    A typed error (rather than a bare ``RuntimeError``) so callers — e.g. the
    judges, which fan out many completions at once — can catch it specifically
    and retry with backoff instead of treating it as a hard failure.
    """


class LLMClient(ABC):
    @abstractmethod
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
        raise NotImplementedError
