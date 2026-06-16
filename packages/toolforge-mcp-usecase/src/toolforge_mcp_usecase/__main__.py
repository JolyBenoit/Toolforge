"""Entry point: python -m toolforge_mcp_usecase

Usage:
    python -m toolforge_mcp_usecase \\
        --usecase uc_invoice \\
        --run r_20260522_abc123 \\
        [--data-root ./data] \\
        [--config toolforge.toml] \\
        [--transport stdio]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from toolforge_registry import Registry
from toolforge_sandbox import Sandbox
from toolforge_telemetry import TelemetryWriter

from ._handlers import RunContext
from ._server import create_server


async def _run() -> None:
    parser = argparse.ArgumentParser(description="ToolForge Usecase MCP Server")
    parser.add_argument("--usecase", required=True, help="Use case ID")
    parser.add_argument("--run", required=True, dest="run_id", help="Run ID")
    parser.add_argument("--data-root", default="data", type=Path, help="Registry data root")
    parser.add_argument("--config", default="toolforge.toml", type=Path, help="Config file path")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio"],
        help="MCP transport (default: stdio; SSE requires an external ASGI host)",
    )
    args = parser.parse_args()

    cfg = None
    llm_tool_configs = None
    if args.config.exists():
        from toolforge_core.config import load_config, resolve_llm_tool_configs
        try:
            cfg = load_config(args.config)
            llm_tool_configs = resolve_llm_tool_configs(cfg) if cfg.llm.tools else None
        except Exception:
            pass

    registry = Registry(args.data_root)
    sandbox = Sandbox.from_config(cfg.sandbox) if cfg is not None else Sandbox()
    telemetry_path = registry._run_dir(args.usecase, args.run_id) / "telemetry.jsonl"
    telemetry = TelemetryWriter(telemetry_path)

    try:
        run_info = registry.get_run(args.usecase, args.run_id)
        is_production = run_info.status == "in_production"
    except Exception:
        is_production = False

    ctx = RunContext(
        usecase_id=args.usecase,
        run_id=args.run_id,
        registry=registry,
        sandbox=sandbox,
        telemetry=telemetry,
        llm_tool_configs=llm_tool_configs,
        is_production=is_production,
    )
    server = create_server(ctx)

    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
