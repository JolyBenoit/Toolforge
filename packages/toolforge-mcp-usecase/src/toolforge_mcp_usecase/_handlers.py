"""Pure handler logic — no MCP wiring, fully unit-testable."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from toolforge_registry import Registry
from toolforge_sandbox import Sandbox
from toolforge_telemetry import TelemetryEvent, TelemetryWriter

_PREVIEW_LEN = 300


def _preview(value: Any) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(value)
    return s[:_PREVIEW_LEN] + ("…" if len(s) > _PREVIEW_LEN else "")


def _classify_error(message: str) -> str:
    msg = message.lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if any(k in msg for k in ("permission", "access denied", "not found", "no such")):
        return "validation"
    if any(k in msg for k in ("import", "module", "syntax", "name error", "attribute")):
        return "runtime"
    return "runtime"


@dataclass
class RunContext:
    usecase_id: str
    run_id: str
    registry: Registry
    sandbox: Any  # Sandbox | test mock
    telemetry: TelemetryWriter | None = field(default=None, compare=False)
    llm_tool_configs: dict[str, Any] | None = field(default=None, compare=False)
    is_production: bool = False


async def h_list_tools(ctx: RunContext) -> list[dict[str, Any]]:
    """Return active tools as dicts with name, description, and inputSchema."""
    active = ctx.registry.get_active_tools(ctx.usecase_id, ctx.run_id)
    result: list[dict[str, Any]] = []
    for tool in active:
        schema = ctx.registry.get_tool_schema(ctx.usecase_id, ctx.run_id, tool.name)
        result.append({
            "name": tool.name,
            "description": tool.description,
            "inputSchema": schema,
        })
    return result


async def h_call_tool(
    ctx: RunContext,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute a named active tool in the sandbox; return JSON-encoded output.

    For in_production runs the response is wrapped in a telemetry envelope so
    the ConsumerAgent can extract metadata (tool_version, duration_ms,
    nested_llm_calls) without polluting the output seen by the LLM.
    """
    active_names = {t.name for t in ctx.registry.get_active_tools(ctx.usecase_id, ctx.run_id)}
    if name not in active_names:
        raise ValueError(f"Tool {name!r} is not active in this run")

    handler_source = ctx.registry.get_handler_source(ctx.usecase_id, ctx.run_id, name)
    requirements = ctx.registry.get_run_requirements(ctx.usecase_id, ctx.run_id)
    result = await ctx.sandbox.run(
        handler_source,
        arguments,
        requirements=requirements,
        llm_configs=ctx.llm_tool_configs,
        inputs_dir=ctx.registry.inputs_dir(ctx.usecase_id),
        outputs_dir=ctx.registry.outputs_dir(ctx.usecase_id),
    )
    if result.success:
        try:
            output_val = json.loads(json.dumps(result.output))
        except (TypeError, ValueError):
            output_val = str(result.output)

        output_str = json.dumps(output_val, ensure_ascii=False)

        if ctx.telemetry is not None:
            ctx.telemetry.append_event(
                TelemetryEvent(
                    kind="execution",
                    event="call_tool",
                    tool=name,
                    duration_ms=result.duration_ms,
                    input_preview=_preview(arguments),
                    output_preview=output_str[:_PREVIEW_LEN] + ("…" if len(output_str) > _PREVIEW_LEN else ""),
                )
            )

        if ctx.is_production:
            tool_info = ctx.registry.get_tool(ctx.usecase_id, ctx.run_id, name)
            return json.dumps({
                "__tf__": {
                    "tool_version": tool_info.active_version,
                    "duration_ms": result.duration_ms,
                    "nested_llm_calls": result.nested_llm_calls,
                },
                "output": output_val,
            }, ensure_ascii=False)

        return output_str

    error_str = result.stderr or "Tool execution failed with no output"
    if ctx.telemetry is not None:
        ctx.telemetry.append_event(
            TelemetryEvent(
                kind="execution",
                event="call_tool_error",
                tool=name,
                duration_ms=result.duration_ms,
                error=error_str,
                error_kind=_classify_error(error_str),
                input_preview=_preview(arguments),
            )
        )
    raise RuntimeError(error_str)
