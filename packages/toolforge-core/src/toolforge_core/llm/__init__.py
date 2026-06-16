from __future__ import annotations

from toolforge_core.config import ProviderConfig
from .base import LLMClient
from .anthropic import AnthropicClient
from .openai_compat import OpenAICompatClient

__all__ = ["LLMClient", "AnthropicClient", "OpenAICompatClient", "create_client"]


def create_client(provider: str, config: ProviderConfig) -> LLMClient:
    api_key = config.resolve_api_key()
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, base_url=config.base_url, timeout=config.timeout)
    return OpenAICompatClient(api_key=api_key, base_url=config.base_url, timeout=config.timeout)
