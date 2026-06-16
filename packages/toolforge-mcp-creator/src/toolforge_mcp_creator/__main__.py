"""Entry point: python -m toolforge_mcp_creator

Usage:
    python -m toolforge_mcp_creator \\
        --usecase uc_invoice \\
        --run r_20260522_abc123 \\
        [--data-root ./data] \\
        [--config toolforge.toml] \\
        [--transport stdio|sse] \\
        [--host localhost] \\
        [--port 8765]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from toolforge_registry import Registry
from toolforge_sandbox import Sandbox

from ._handlers import RunContext
from .server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="ToolForge Creator MCP Server")
    parser.add_argument("--usecase", required=True, help="Use case ID")
    parser.add_argument("--run", required=True, dest="run_id", help="Run ID")
    parser.add_argument("--data-root", default="data", type=Path, help="Registry data root")
    parser.add_argument("--config", default="toolforge.toml", type=Path, help="Config file path")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8765, type=int)
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

    ctx = RunContext(
        usecase_id=args.usecase,
        run_id=args.run_id,
        registry=registry,
        sandbox=sandbox,
        llm_tool_configs=llm_tool_configs,
    )
    mcp = create_server(ctx)

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
