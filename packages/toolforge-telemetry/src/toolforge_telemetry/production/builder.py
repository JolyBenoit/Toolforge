"""Build a :class:`PipelineSpec` snapshot from a registry at promote time.

The builder is deliberately decoupled from ``toolforge-registry``: it only
calls a small set of duck-typed methods on the ``registry`` object passed in,
so this package keeps no hard dependency on the registry.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Protocol

from .models import PipelineSpec, PipelineToolSnapshot


class _RegistryLike(Protocol):
    """Subset of the registry API needed to snapshot a run's active tools."""

    def get_run(self, usecase_id: str, run_id: str) -> Any: ...
    def get_active_tools(self, usecase_id: str, run_id: str) -> list[Any]: ...
    def get_handler_source(self, usecase_id: str, run_id: str, name: str) -> str: ...
    def get_tool_schema(self, usecase_id: str, run_id: str, name: str) -> dict[str, Any]: ...
    def get_consumer_prompt(self, usecase_id: str) -> str | None: ...


def build_pipeline_spec(
    registry: _RegistryLike,
    usecase_id: str,
    run_id: str,
    *,
    change_reason: str | None = None,
) -> PipelineSpec:
    """Snapshot every active tool of ``run_id`` into an immutable PipelineSpec.

    Captures, for each active tool, its version, a sha256 of its handler
    source (``implementation_hash``) and its input schema. The use-case
    consumer prompt becomes the pipeline ``system_prompt``.
    """
    run_info = registry.get_run(usecase_id, run_id)

    tools: list[PipelineToolSnapshot] = []
    for tool in registry.get_active_tools(usecase_id, run_id):
        source = registry.get_handler_source(usecase_id, run_id, tool.name)
        schema = registry.get_tool_schema(usecase_id, run_id, tool.name)
        impl_hash = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
        tools.append(
            PipelineToolSnapshot(
                tool_id=tool.name,
                tool_version=tool.active_version,
                implementation_hash=impl_hash,
                schema=schema,
            )
        )

    promoted = getattr(run_info, "promoted_to_production_at", None)
    promoted_at = (
        promoted.isoformat()
        if isinstance(promoted, datetime)
        else datetime.now(timezone.utc).isoformat()
    )

    return PipelineSpec(
        run_id=run_id,
        usecase_id=usecase_id,
        promoted_at=promoted_at,
        tools=tools,
        system_prompt=registry.get_consumer_prompt(usecase_id),
        forked_from=getattr(run_info, "forked_from", None),
        change_reason=change_reason,
    )
