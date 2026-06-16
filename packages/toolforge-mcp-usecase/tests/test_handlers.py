"""Handler tests — no MCP server or Docker required."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolforge_registry import Registry
from toolforge_sandbox import SandboxResult

from toolforge_mcp_usecase._handlers import RunContext, h_call_tool, h_list_tools

_HANDLER = "def run(args):\n    return args.get('x', 0) * 2"
_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}


# --- mock sandbox ---


class _MockSandbox:
    def __init__(self, *, success: bool = True, output: Any = 42) -> None:
        self._success = success
        self._output = output

    async def run(
        self,
        handler_source: str,
        args: dict[str, Any],
        *,
        requirements: list[str] | None = None,
        llm_configs: dict[str, Any] | None = None,
        inputs_dir: Any = None,
        outputs_dir: Any = None,
        mode: str = "runtime",
    ) -> SandboxResult:
        return SandboxResult(
            output=self._output if self._success else None,
            stdout=json.dumps({"output": self._output, "error": None}),
            stderr="" if self._success else "exec error",
            duration_ms=1.0,
            exit_code=0 if self._success else 1,
        )


# --- fixtures ---


def _make_active_tool(reg: Registry, uc: str, run_id: str, name: str) -> None:
    """Propose, sandbox-validate, and promote a tool so it appears in get_active_tools."""
    reg.propose_tool(uc, run_id, name, f"{name} description", _HANDLER, _SCHEMA)
    reg.mark_sandbox_validated(uc, run_id, name, 1)
    reg.promote_tool(uc, run_id, name, 1)


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    reg = Registry(tmp_path / "data")
    reg.create_usecase("uc", "Test use case.")
    run = reg.create_run("uc")
    return RunContext(
        usecase_id="uc",
        run_id=run.run_id,
        registry=reg,
        sandbox=_MockSandbox(success=True, output=42),
    )


@pytest.fixture
def ctx_failing(tmp_path: Path) -> RunContext:
    reg = Registry(tmp_path / "data")
    reg.create_usecase("uc", "Test")
    run = reg.create_run("uc")
    return RunContext(
        usecase_id="uc",
        run_id=run.run_id,
        registry=reg,
        sandbox=_MockSandbox(success=False),
    )


# --- h_list_tools ---


async def test_list_tools_empty(ctx: RunContext) -> None:
    result = await h_list_tools(ctx)
    assert result == []


async def test_list_tools_returns_active_only(ctx: RunContext) -> None:
    reg, uc, run_id = ctx.registry, ctx.usecase_id, ctx.run_id
    _make_active_tool(reg, uc, run_id, "alpha")
    # propose but don't promote beta — should NOT appear
    reg.propose_tool(uc, run_id, "beta", "desc", _HANDLER, _SCHEMA)

    tools = await h_list_tools(ctx)
    assert len(tools) == 1
    assert tools[0]["name"] == "alpha"
    assert tools[0]["description"] == "alpha description"
    assert tools[0]["inputSchema"] == _SCHEMA


async def test_list_tools_schema_shape(ctx: RunContext) -> None:
    _make_active_tool(ctx.registry, ctx.usecase_id, ctx.run_id, "t")
    tools = await h_list_tools(ctx)
    assert "name" in tools[0]
    assert "description" in tools[0]
    assert "inputSchema" in tools[0]


async def test_list_tools_multiple(ctx: RunContext) -> None:
    reg, uc, rid = ctx.registry, ctx.usecase_id, ctx.run_id
    _make_active_tool(reg, uc, rid, "t1")
    _make_active_tool(reg, uc, rid, "t2")
    tools = await h_list_tools(ctx)
    names = {t["name"] for t in tools}
    assert names == {"t1", "t2"}


# --- h_call_tool ---


async def test_call_tool_success(ctx: RunContext) -> None:
    _make_active_tool(ctx.registry, ctx.usecase_id, ctx.run_id, "double")
    result = await h_call_tool(ctx, "double", {"x": 5})
    # MockSandbox always returns output=42; h_call_tool JSON-encodes it
    assert result == "42"


async def test_call_tool_inactive_raises(ctx: RunContext) -> None:
    # Tool proposed but not promoted
    ctx.registry.propose_tool(ctx.usecase_id, ctx.run_id, "ghost", "d", _HANDLER, _SCHEMA)
    with pytest.raises(ValueError, match="not active"):
        await h_call_tool(ctx, "ghost", {})


async def test_call_tool_unknown_raises(ctx: RunContext) -> None:
    with pytest.raises(ValueError, match="not active"):
        await h_call_tool(ctx, "no_such_tool", {})


async def test_call_tool_sandbox_failure_raises(ctx_failing: RunContext) -> None:
    _make_active_tool(ctx_failing.registry, ctx_failing.usecase_id, ctx_failing.run_id, "bad")
    with pytest.raises(RuntimeError, match="exec error"):
        await h_call_tool(ctx_failing, "bad", {})


async def test_call_tool_empty_args(ctx: RunContext) -> None:
    _make_active_tool(ctx.registry, ctx.usecase_id, ctx.run_id, "t")
    result = await h_call_tool(ctx, "t", {})
    assert result == "42"


async def test_call_tool_output_is_json_encoded(ctx: RunContext) -> None:
    # Verify dict output round-trips through JSON
    ctx.sandbox._output = {"key": "value"}
    _make_active_tool(ctx.registry, ctx.usecase_id, ctx.run_id, "t")
    result = await h_call_tool(ctx, "t", {})
    assert json.loads(result) == {"key": "value"}
