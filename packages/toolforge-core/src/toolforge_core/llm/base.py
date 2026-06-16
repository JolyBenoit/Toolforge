from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from toolforge_core.types import Message, StreamEvent


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
