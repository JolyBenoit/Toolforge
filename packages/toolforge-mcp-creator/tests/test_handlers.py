"""Handler tests — no MCP server or Docker required."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolforge_registry import Registry
from toolforge_sandbox import SandboxResult

from toolforge_mcp_creator._handlers import (
    RunContext,
    h_deprecate_tool,
    h_list_tools,
    h_promote_tool,
    h_propose_tool,
    h_read_telemetry,
    h_request_human_validation,
    h_update_tool,
    h_validate_in_sandbox,
)

_HANDLER = "def run(args):\n    return args.get('x', 0) + 1"
_SCHEMA = json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})


# --- mock sandbox ---


class _MockSandbox:
    def __init__(self, *, success: bool = True, output: Any = "ok") -> None:
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
            stderr="",
            duration_ms=1.0,
            exit_code=0 if self._success else 1,
        )


# --- fixtures ---


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    reg = Registry(tmp_path / "data")
    reg.create_usecase("uc", "Test use case prompt.")
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


# --- list_tools ---


def test_list_tools_empty(ctx: RunContext) -> None:
    result = h_list_tools(ctx)
    assert "no tools" in result.lower()


def test_list_tools_after_propose(ctx: RunContext) -> None:
    h_propose_tool(ctx, "extract", "Extract data", _HANDLER, _SCHEMA)
    data = json.loads(h_list_tools(ctx))
    assert len(data) == 1
    assert data[0]["name"] == "extract"
    assert data[0]["active_version"] is None
    assert data[0]["versions"][0]["sandbox_validated"] is False


# --- propose_tool ---


def test_propose_tool_success(ctx: RunContext) -> None:
    result = h_propose_tool(ctx, "extract", "Extract data", _HANDLER, _SCHEMA)
    assert "version 1" in result
    assert "extract" in result


def test_propose_tool_bad_schema(ctx: RunContext) -> None:
    result = h_propose_tool(ctx, "x", "desc", _HANDLER, "not json")
    assert result.startswith("ERROR:")


def test_propose_tool_duplicate(ctx: RunContext) -> None:
    h_propose_tool(ctx, "extract", "v1", _HANDLER, _SCHEMA)
    result = h_propose_tool(ctx, "extract", "v1 again", _HANDLER, _SCHEMA)
    assert result.startswith("ERROR:")


# --- update_tool ---


def test_update_tool_adds_version(ctx: RunContext) -> None:
    h_propose_tool(ctx, "extract", "v1", _HANDLER, _SCHEMA)
    result = h_update_tool(ctx, "extract", "v2 desc", _HANDLER, _SCHEMA)
    assert "version 2" in result


def test_update_nonexistent_tool(ctx: RunContext) -> None:
    result = h_update_tool(ctx, "ghost", "desc", _HANDLER, _SCHEMA)
    assert result.startswith("ERROR:")


def test_update_tool_bad_schema(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    result = h_update_tool(ctx, "t", "desc", _HANDLER, "{bad}")
    assert result.startswith("ERROR:")


# --- validate_in_sandbox ---


async def test_validate_success_marks_validated(ctx: RunContext) -> None:
    h_propose_tool(ctx, "extract", "desc", _HANDLER, _SCHEMA)
    result = await h_validate_in_sandbox(ctx, "extract", 1, '{"x": 5}')
    assert "✓" in result
    assert "promote_tool" in result
    tool = ctx.registry.get_tool(ctx.usecase_id, ctx.run_id, "extract")
    assert tool.versions[0].sandbox_validated is True


async def test_validate_failure_does_not_mark(ctx_failing: RunContext) -> None:
    h_propose_tool(ctx_failing, "bad", "desc", _HANDLER, _SCHEMA)
    result = await h_validate_in_sandbox(ctx_failing, "bad", 1)
    assert "✗" in result
    tool = ctx_failing.registry.get_tool(ctx_failing.usecase_id, ctx_failing.run_id, "bad")
    assert tool.versions[0].sandbox_validated is False


async def test_validate_bad_test_args(ctx: RunContext) -> None:
    result = await h_validate_in_sandbox(ctx, "x", 1, "not-json")
    assert result.startswith("ERROR:")


async def test_validate_nonexistent_tool(ctx: RunContext) -> None:
    result = await h_validate_in_sandbox(ctx, "ghost", 1)
    assert result.startswith("ERROR:")


async def test_validate_reports_output(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    result = await h_validate_in_sandbox(ctx, "t", 1, '{"x": 10}')
    assert "42" in result  # MockSandbox always returns 42


# --- promote_tool ---


def test_promote_validated_tool(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    ctx.registry.mark_sandbox_validated(ctx.usecase_id, ctx.run_id, "t", 1)
    result = h_promote_tool(ctx, "t", 1)
    assert "active version is now 1" in result


def test_promote_unvalidated_returns_error(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    result = h_promote_tool(ctx, "t", 1)
    assert result.startswith("ERROR:")


def test_promote_nonexistent_returns_error(ctx: RunContext) -> None:
    result = h_promote_tool(ctx, "ghost", 1)
    assert result.startswith("ERROR:")


# --- deprecate_tool ---


def test_deprecate_tool(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    result = h_deprecate_tool(ctx, "t")
    assert "deprecated" in result
    assert ctx.registry.get_tool(ctx.usecase_id, ctx.run_id, "t").status == "deprecated"


def test_deprecate_nonexistent_returns_error(ctx: RunContext) -> None:
    result = h_deprecate_tool(ctx, "ghost")
    assert result.startswith("ERROR:")


# --- read_telemetry ---


def test_read_telemetry_no_file(ctx: RunContext) -> None:
    data = json.loads(h_read_telemetry(ctx))
    assert data["events"] == []
    assert data["run_id"] == ctx.run_id


def test_read_telemetry_with_file(ctx: RunContext) -> None:
    run_dir = ctx.registry._run_dir(ctx.usecase_id, ctx.run_id)
    events = [
        {"kind": "creation", "event": "propose_tool", "tool": "t"},
        {"kind": "creation", "event": "validate", "tool": "t"},
    ]
    (run_dir / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )
    data = json.loads(h_read_telemetry(ctx, limit=10))
    assert data["count"] == 2
    assert data["events"][0]["event"] == "propose_tool"


def test_read_telemetry_limit(ctx: RunContext) -> None:
    run_dir = ctx.registry._run_dir(ctx.usecase_id, ctx.run_id)
    lines = [json.dumps({"n": i}) for i in range(10)]
    (run_dir / "telemetry.jsonl").write_text("\n".join(lines), encoding="utf-8")
    data = json.loads(h_read_telemetry(ctx, limit=3))
    assert data["count"] == 3
    assert data["events"][0]["n"] == 7  # last 3: 7, 8, 9


# --- request_human_validation ---


def test_request_human_validation_contains_run_id(ctx: RunContext) -> None:
    result = h_request_human_validation(ctx)
    assert ctx.run_id in result
    assert "validate" in result.lower()


def test_request_human_validation_lists_active_tools(ctx: RunContext) -> None:
    h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    ctx.registry.mark_sandbox_validated(ctx.usecase_id, ctx.run_id, "t", 1)
    ctx.registry.promote_tool(ctx.usecase_id, ctx.run_id, "t", 1)
    result = h_request_human_validation(ctx)
    assert "1 —" in result
    assert "t" in result


# --- run-locked guard (immutable after validation) ---


def test_propose_on_validated_run_returns_error(ctx: RunContext) -> None:
    ctx.registry.validate_run(ctx.usecase_id, ctx.run_id)
    result = h_propose_tool(ctx, "t", "desc", _HANDLER, _SCHEMA)
    assert result.startswith("ERROR:")
