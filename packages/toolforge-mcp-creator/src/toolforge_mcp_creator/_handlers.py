"""Business logic for all Creator meta-tools.

Kept separate from the MCP wiring so tests can call these functions directly
without starting a server.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from toolforge_registry import Registry, RegistryError
from toolforge_sandbox import Sandbox
from toolforge_telemetry import TelemetryEvent, TelemetryWriter


@dataclass
class RunContext:
    usecase_id: str
    run_id: str
    registry: Registry
    sandbox: Sandbox
    telemetry: TelemetryWriter | None = field(default=None, compare=False)
    llm_tool_configs: dict[str, Any] | None = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Telemetry helper
# ---------------------------------------------------------------------------


def _emit(ctx: RunContext, event: str, tool: str, **extra: Any) -> None:
    if ctx.telemetry is None:
        return
    ctx.telemetry.append_event(
        TelemetryEvent(kind="creation", event=event, tool=tool, **extra)
    )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def h_list_tools(ctx: RunContext) -> str:
    tools = ctx.registry.list_tools(ctx.usecase_id, ctx.run_id)
    if not tools:
        return "No tools defined in this run yet."
    return json.dumps([_tool_dict(t) for t in tools], indent=2)


def h_propose_tool(
    ctx: RunContext,
    name: str,
    description: str,
    handler_source: str,
    input_schema: str,
    requirements: str = "[]",
) -> str:
    try:
        schema: dict[str, Any] = json.loads(input_schema)
    except json.JSONDecodeError as e:
        return f"ERROR: input_schema is not valid JSON — {e}"
    try:
        reqs: list[str] = json.loads(requirements)
    except json.JSONDecodeError as e:
        return f"ERROR: requirements is not valid JSON — {e}"
    conflicts = ctx.registry.check_requirements_conflicts(
        ctx.usecase_id, ctx.run_id, reqs, tool_name=name
    )
    try:
        v = ctx.registry.propose_tool(
            ctx.usecase_id, ctx.run_id, name, description, handler_source, schema, reqs
        )
    except RegistryError as e:
        return f"ERROR: {e}"
    _emit(ctx, "propose_tool", name, version=v.version)
    return _propose_response(name, v.version, conflicts, action="created")


def h_update_tool(
    ctx: RunContext,
    name: str,
    description: str,
    handler_source: str,
    input_schema: str,
    requirements: str = "[]",
) -> str:
    try:
        schema: dict[str, Any] = json.loads(input_schema)
    except json.JSONDecodeError as e:
        return f"ERROR: input_schema is not valid JSON — {e}"
    try:
        reqs: list[str] = json.loads(requirements)
    except json.JSONDecodeError as e:
        return f"ERROR: requirements is not valid JSON — {e}"
    conflicts = ctx.registry.check_requirements_conflicts(
        ctx.usecase_id, ctx.run_id, reqs, tool_name=name
    )
    try:
        v = ctx.registry.update_tool(
            ctx.usecase_id, ctx.run_id, name, description, handler_source, schema, reqs
        )
    except RegistryError as e:
        return f"ERROR: {e}"
    _emit(ctx, "update_tool", name, version=v.version)
    return _propose_response(name, v.version, conflicts, action="updated to")


async def h_validate_in_sandbox(
    ctx: RunContext,
    name: str,
    version: int,
    test_args: str = "{}",
) -> str:
    try:
        args: dict[str, Any] = json.loads(test_args)
    except json.JSONDecodeError as e:
        return f"ERROR: test_args is not valid JSON — {e}"

    try:
        source = ctx.registry.get_handler_source(ctx.usecase_id, ctx.run_id, name, version)
        reqs = ctx.registry.get_tool_requirements(ctx.usecase_id, ctx.run_id, name, version)
    except RegistryError as e:
        return f"ERROR: {e}"

    result = await ctx.sandbox.run(
        source,
        args,
        requirements=reqs,
        llm_configs=ctx.llm_tool_configs,
        inputs_dir=ctx.registry.inputs_dir(ctx.usecase_id),
        outputs_dir=ctx.registry.outputs_dir(ctx.usecase_id),
        mode="validation",
    )

    lines = [
        f"=== Sandbox: {name} v{version} ===",
        f"exit_code : {result.exit_code}",
        f"duration  : {result.duration_ms:.0f}ms",
        f"output    : {json.dumps(result.output)}",
    ]
    if result.stderr.strip():
        lines.append(f"stderr:\n{result.stderr.strip()}")

    if result.success:
        try:
            ctx.registry.mark_sandbox_validated(ctx.usecase_id, ctx.run_id, name, version)
            _emit(ctx, "sandbox_validated", name, version=version)
            lines.append(
                f"\n✓ Validated. Call promote_tool('{name}', {version}) to make it active."
            )
        except RegistryError as e:
            lines.append(f"\nWARNING: could not record validation — {e}")
    else:
        lines.append("\n✗ Validation failed. Fix the handler and retry.")

    return "\n".join(lines)


def h_promote_tool(ctx: RunContext, name: str, version: int) -> str:
    try:
        info = ctx.registry.promote_tool(ctx.usecase_id, ctx.run_id, name, version)
    except RegistryError as e:
        return f"ERROR: {e}"
    _emit(ctx, "promote", name, version=version)
    merged = ctx.registry.get_run_requirements(ctx.usecase_id, ctx.run_id)
    req_summary = (
        f"Run requirements ({len(merged)}): {', '.join(merged)}"
        if merged
        else "Run has no external requirements."
    )
    return f"Tool '{name}' active version is now {info.active_version}. {req_summary}"


def h_deprecate_tool(ctx: RunContext, name: str) -> str:
    try:
        ctx.registry.deprecate_tool(ctx.usecase_id, ctx.run_id, name)
    except RegistryError as e:
        return f"ERROR: {e}"
    _emit(ctx, "deprecate", name)
    merged = ctx.registry.get_run_requirements(ctx.usecase_id, ctx.run_id)
    req_summary = (
        f"Run requirements ({len(merged)}): {', '.join(merged)}"
        if merged
        else "Run has no external requirements."
    )
    return f"Tool '{name}' has been deprecated and removed from the active set. {req_summary}"


def h_read_telemetry(ctx: RunContext, limit: int = 20) -> str:
    """Read recent telemetry events.  Full integration arrives in step 9."""
    telemetry_path = (
        ctx.registry._run_dir(ctx.usecase_id, ctx.run_id) / "telemetry.jsonl"
    )
    if not telemetry_path.exists():
        return json.dumps(
            {
                "run_id": ctx.run_id,
                "usecase_id": ctx.usecase_id,
                "events": [],
                "note": "No telemetry recorded yet.",
            },
            indent=2,
        )
    raw_lines = telemetry_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(ln) for ln in raw_lines[-limit:] if ln.strip()]
    return json.dumps(
        {"run_id": ctx.run_id, "events": events, "count": len(events)},
        indent=2,
    )


def h_consumer_instructions(ctx: RunContext, action: str, instructions: str = "") -> str:
    """Get or set the use-case-specific Consumer agent instructions."""
    if action == "get":
        prompt = ctx.registry.get_consumer_prompt(ctx.usecase_id)
        if prompt is None:
            return "No use-case-specific consumer instructions set yet."
        return f"Current consumer instructions:\n\n{prompt}"
    if action == "set":
        if not instructions.strip():
            return "ERROR: instructions cannot be empty when action='set'."
        ctx.registry.set_consumer_prompt(ctx.usecase_id, instructions.strip())
        return (
            f"Consumer instructions saved ({len(instructions.strip())} chars). "
            "They will be injected into the Consumer agent's system prompt at next launch."
        )
    return f"ERROR: unknown action {action!r}. Use 'get' or 'set'."


def h_list_inputs(ctx: RunContext) -> str:
    """Return files available to tool handlers, with their container-side paths."""
    files = ctx.registry.list_inputs(ctx.usecase_id)
    if not files:
        return json.dumps({
            "available": False,
            "note": (
                "No input files found. "
                f"Drop files into data/usecases/{ctx.usecase_id}/inputs/ "
                "to make them available to tool handlers."
            ),
        }, indent=2)
    return json.dumps({
        "available": True,
        "files": [f.name for f in files],
        "note": (
            "Inside the sandbox, tool handlers read input files via the INPUTS_DIR "
            "global that is automatically injected by the runner. "
            "Example: open(os.path.join(INPUTS_DIR, 'invoice.pdf'), 'rb'). "
            "Do NOT hardcode /inputs/ — use INPUTS_DIR."
        ),
    }, indent=2)


def h_list_outputs(ctx: RunContext) -> str:
    """Return files present in the outputs folder, with the container write path."""
    files = ctx.registry.list_outputs(ctx.usecase_id)
    return json.dumps({
        "files": [f.name for f in files],
        "note": (
            "Tool handlers must write all generated files (reports, CSV, markdown, etc.) "
            "using the OUTPUTS_DIR global that is automatically injected by the runner. "
            "Example: open(os.path.join(OUTPUTS_DIR, 'report.md'), 'w') or "
            "os.makedirs(OUTPUTS_DIR, exist_ok=True) then write there. "
            "Do NOT hardcode /outputs/ — use OUTPUTS_DIR. "
            "Files written to OUTPUTS_DIR are persisted to "
            f"data/usecases/{ctx.usecase_id}/outputs/ on the host."
        ),
    }, indent=2)


def h_request_human_validation(ctx: RunContext) -> str:
    try:
        active = ctx.registry.get_active_tools(ctx.usecase_id, ctx.run_id)
    except RegistryError as e:
        return f"ERROR: {e}"
    tool_list = ", ".join(t.name for t in active) if active else "(none)"
    return (
        f"Human validation requested for run {ctx.run_id!r}.\n\n"
        f"Use case    : {ctx.usecase_id}\n"
        f"Active tools: {len(active)} — {tool_list}\n\n"
        f"To lock this run as immutable the operator should run:\n"
        f"  toolforge run validate --usecase {ctx.usecase_id} --run {ctx.run_id}\n\n"
        f"After validation no tools can be modified; fork to iterate:\n"
        f"  toolforge run fork --usecase {ctx.usecase_id} --from {ctx.run_id}"
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _propose_response(
    name: str, version: int, conflicts: list[str], *, action: str
) -> str:
    msg = f"Tool '{name}' {action} version {version}."
    if conflicts:
        lines = "\n".join(f"  - {c}" for c in conflicts)
        msg += (
            f"\n⚠ Dependency conflicts with currently active tools:\n{lines}"
            f"\nResolve version conflicts before promoting."
        )
    else:
        msg += f" Call validate_in_sandbox('{name}', {version}) to test it before promoting."
    return msg


def _tool_dict(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "status": tool.status,
        "active_version": tool.active_version,
        "versions": [
            {
                "version": v.version,
                "sandbox_validated": v.sandbox_validated,
                "created_at": v.created_at.isoformat(),
            }
            for v in tool.versions
        ],
    }
