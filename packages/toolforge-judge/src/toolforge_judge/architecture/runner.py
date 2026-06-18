"""Assemble an :class:`ArchitectureSpec` from the registry.

Kept duck-typed (a ``Protocol``) so the architecture judge never hard-depends on
``toolforge-registry`` — the same approach the static judge's ``build_usecase_spec``
takes. The one thing this judge needs beyond the static spec is the handler
**source** (``get_handler_source``), where the truncation/cap behaviour lives.
"""
from __future__ import annotations

from typing import Any, Protocol

from .models import ArchitectureSpec, RichToolSpec


class _RegistryLike(Protocol):
    def get_usecase(self, usecase_id: str) -> Any: ...
    def get_active_tools(self, usecase_id: str, run_id: str) -> list[Any]: ...
    def get_tool_schema(
        self, usecase_id: str, run_id: str, name: str
    ) -> dict[str, Any]: ...
    def get_handler_source(self, usecase_id: str, run_id: str, name: str) -> str: ...
    def get_tool_requirements(
        self, usecase_id: str, run_id: str, name: str
    ) -> list[str]: ...
    def get_consumer_prompt(self, usecase_id: str) -> str | None: ...


def build_architecture_spec(
    registry: _RegistryLike, usecase_id: str, run_id: str
) -> ArchitectureSpec:
    """Assemble the full pipeline (use case + every active tool's source)."""
    usecase = registry.get_usecase(usecase_id)
    tools = [
        RichToolSpec(
            tool_id=t.name,
            description=t.description,
            input_schema=registry.get_tool_schema(usecase_id, run_id, t.name),
            source=registry.get_handler_source(usecase_id, run_id, t.name),
            requirements=registry.get_tool_requirements(usecase_id, run_id, t.name),
        )
        for t in registry.get_active_tools(usecase_id, run_id)
    ]
    return ArchitectureSpec(
        usecase_id=usecase_id,
        run_id=run_id,
        utility=getattr(usecase, "prompt", ""),
        rules=registry.get_consumer_prompt(usecase_id) or "",
        tools=tools,
    )
