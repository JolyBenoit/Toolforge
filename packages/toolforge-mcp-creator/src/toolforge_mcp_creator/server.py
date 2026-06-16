"""FastMCP server wiring for the Creator meta-tools.

Each MCP tool is a thin wrapper that delegates to the corresponding h_*
function in _handlers.py.  FastMCP infers the JSON Schema from the wrapper's
type-annotated signature and docstring.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ._handlers import (
    RunContext,
    h_consumer_instructions,
    h_deprecate_tool,
    h_list_inputs,
    h_list_outputs,
    h_list_tools,
    h_promote_tool,
    h_propose_tool,
    h_read_telemetry,
    h_request_human_validation,
    h_update_tool,
    h_validate_in_sandbox,
)


def create_server(ctx: RunContext) -> FastMCP:
    """Return a configured FastMCP server scoped to one (usecase_id, run_id)."""
    mcp = FastMCP(f"toolforge-creator/{ctx.usecase_id}/{ctx.run_id}")

    @mcp.tool()
    def consumer_instructions(action: str = "get", instructions: str = "") -> str:
        """Get or set the Consumer agent's use-case-specific instructions.

        These instructions are appended to the Consumer's system prompt at every
        launch, giving it context on how to use this use case's tools.

        Args:
            action: "get" to read the current instructions, "set" to write/replace them.
            instructions: The instructions to save (required when action="set").
                          Describe tool call order, naming conventions, expected
                          inputs/outputs, error handling, etc.
        """
        return h_consumer_instructions(ctx, action, instructions)

    @mcp.tool()
    def list_outputs() -> str:
        """List files already written to the outputs folder by tool handlers.

        IMPORTANT: tool handlers must write all generated files to /outputs/{filename}
        inside the sandbox — never use hardcoded host paths.
        Call this before proposing any tool that produces output files.
        """
        return h_list_outputs(ctx)

    @mcp.tool()
    def list_inputs() -> str:
        """List input files available to tool handlers inside the sandbox.

        IMPORTANT: tool handler code must always read files using the path
        /inputs/{filename} — never use host filesystem paths or relative paths.
        Call this before proposing any tool that needs to read a file.
        """
        return h_list_inputs(ctx)

    @mcp.tool()
    def list_tools() -> str:
        """List all tools in the current run with status, versions, and validation state."""
        return h_list_tools(ctx)

    @mcp.tool()
    def list_llm_tools() -> str:
        """List the LLM tools available to handler code via the injected ``llm`` object.

        Returns a JSON object mapping tool name to provider/model info.
        Use these names to access tools in handlers:
          - ``llm.<name>.complete(prompt)``  — single user message
          - ``llm.<name>.chat(messages)``    — full messages list (system + user + …)
          - ``llm["<name>"]``                — dict-style access, same result
        Returns an empty object if no LLM tools are configured in toolforge.toml.
        """
        import json as _json
        if not ctx.llm_tool_configs:
            return _json.dumps({
                "available": False,
                "note": "No [llm.tools.*] sections found in toolforge.toml. "
                        "Add one to enable LLM calls inside handler code.",
            })
        tools = {
            name: {"provider": cfg["provider"], "model": cfg["model"]}
            for name, cfg in ctx.llm_tool_configs.items()
        }
        return _json.dumps({"available": True, "tools": tools})

    @mcp.tool()
    def propose_tool(
        name: str,
        description: str,
        handler_source: str,
        input_schema: str,
        requirements: str = "[]",
    ) -> str:
        """Create a new tool in the current run (version 1).

        Args:
            name: Snake_case identifier (e.g. extract_invoice).
            description: One-sentence explanation shown to the Consumer agent.
            handler_source: Python source that defines ``run(args: dict) -> Any``.
                The global ``llm`` object is always available — no import needed.
                Usage:
                  llm.<name>.complete(prompt)        # single user message
                  llm.<name>.chat(messages)          # full messages list
                  llm["<name>"].complete(prompt)     # dict-style access
                Call ``list_llm_tools()`` first to know which names are available.
                No SDK required — llm uses stdlib urllib only.
            input_schema: JSON Schema object describing the args dict.
            requirements: JSON array of pip package specifiers needed by the handler,
                e.g. ``["pandas>=2.0", "httpx"]``. Omit or pass ``[]`` for stdlib-only
                or LLM-only handlers.
        """
        return h_propose_tool(ctx, name, description, handler_source, input_schema, requirements)

    @mcp.tool()
    def update_tool(
        name: str,
        description: str,
        handler_source: str,
        input_schema: str,
        requirements: str = "[]",
    ) -> str:
        """Add a new version to an existing tool.

        Args:
            name: Existing tool name.
            description: Updated description (replaces the previous one).
            handler_source: Updated Python source. The ``llm`` registry is available
                exactly as in ``propose_tool`` — use ``llm.<name>.complete()`` or
                ``llm.<name>.chat(messages)``.
            input_schema: Updated JSON Schema for args.
            requirements: JSON array of pip package specifiers (replaces previous list).
        """
        return h_update_tool(ctx, name, description, handler_source, input_schema, requirements)

    @mcp.tool()
    async def validate_in_sandbox(
        name: str,
        version: int,
        test_args: str = "{}",
    ) -> str:
        """Run a tool version in the isolated sandbox and report the result.

        If execution succeeds the version is automatically marked as
        sandbox-validated and ready for promotion.

        Args:
            name: Tool name.
            version: Version number to validate.
            test_args: JSON object passed as args to run(). Defaults to {}.
        """
        return await h_validate_in_sandbox(ctx, name, version, test_args)

    @mcp.tool()
    def promote_tool(name: str, version: int) -> str:
        """Make a sandbox-validated tool version the active one exposed to consumers.

        Args:
            name: Tool name.
            version: Version number (must have passed sandbox validation).
        """
        return h_promote_tool(ctx, name, version)

    @mcp.tool()
    def deprecate_tool(name: str) -> str:
        """Remove a tool from the active set of this run.

        The tool and its source are preserved for audit purposes but will
        no longer appear to consumer agents.

        Args:
            name: Tool name to deprecate.
        """
        return h_deprecate_tool(ctx, name)

    @mcp.tool()
    def read_telemetry(limit: int = 20) -> str:
        """Read recent telemetry events recorded for this run.

        Args:
            limit: Maximum number of events to return (newest first).
        """
        return h_read_telemetry(ctx, limit)

    @mcp.tool()
    def request_human_validation() -> str:
        """Signal that the tool library is complete and ready for human review.

        Returns the exact CLI command the operator must run to lock the run
        as immutable.  After validation, forking is required to iterate.
        """
        return h_request_human_validation(ctx)

    return mcp
