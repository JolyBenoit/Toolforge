"""ToolForge CLI — single entry point for all subcommands."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from toolforge_registry import Registry, RegistryError


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--data-root",
    default="data",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the registry data directory.",
)
@click.pass_context
def cli(ctx: click.Context, data_root: Path) -> None:
    """ToolForge — tool-creating tools for agent orchestration."""
    ctx.ensure_object(dict)
    ctx.obj["data_root"] = data_root


# ---------------------------------------------------------------------------
# usecase subcommands
# ---------------------------------------------------------------------------


@cli.group()
def usecase() -> None:
    """Manage use cases."""


@usecase.command("create")
@click.option("--id", "usecase_id", required=True, help="Unique identifier for the use case.")
@click.option("--prompt", "prompt_text", default=None, help="Use case prompt text.")
@click.option(
    "--prompt-file",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Read prompt from a file.",
)
@click.pass_obj
def usecase_create(
    obj: dict[str, Any],
    usecase_id: str,
    prompt_text: str | None,
    prompt_file: Path | None,
) -> None:
    """Create a new use case."""
    if prompt_file is not None:
        prompt = prompt_file.read_text(encoding="utf-8")
    elif prompt_text is not None:
        prompt = prompt_text
    else:
        raise click.UsageError("Provide --prompt or --prompt-file.")
    try:
        Registry(obj["data_root"]).create_usecase(usecase_id, prompt)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Created use case: {usecase_id}")


@usecase.command("list")
@click.pass_obj
def usecase_list(obj: dict[str, Any]) -> None:
    """List all use cases."""
    items = Registry(obj["data_root"]).list_usecases()
    if not items:
        click.echo("No use cases found.")
        return
    for uc in items:
        click.echo(f"{uc.usecase_id}  ({uc.created_at.date()})")


# ---------------------------------------------------------------------------
# run subcommands
# ---------------------------------------------------------------------------


@cli.group()
def run() -> None:
    """Manage runs within a use case."""


@run.command("create")
@click.option("--usecase", "usecase_id", required=True)
@click.pass_obj
def run_create(obj: dict[str, Any], usecase_id: str) -> None:
    """Create a new draft run and print its ID."""
    try:
        info = Registry(obj["data_root"]).create_run(usecase_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(info.run_id)


@run.command("list")
@click.option("--usecase", "usecase_id", required=True)
@click.pass_obj
def run_list(obj: dict[str, Any], usecase_id: str) -> None:
    """List all runs for a use case."""
    try:
        items = Registry(obj["data_root"]).list_runs(usecase_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    if not items:
        click.echo("No runs found.")
        return
    for r in items:
        tools = f"{r.tool_count} tool(s)" if r.tool_count else "no tools"
        click.echo(f"{r.run_id}  {r.status:<12}  {tools}")


@run.command("validate")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--run", "run_id", required=True)
@click.pass_obj
def run_validate(obj: dict[str, Any], usecase_id: str, run_id: str) -> None:
    """Lock a run as immutable (human validation gate)."""
    try:
        Registry(obj["data_root"]).validate_run(usecase_id, run_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Run {run_id!r} is now validated and locked.")


@run.command("promote")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--run", "run_id", required=True)
@click.option(
    "--config",
    "config_path",
    default="toolforge.toml",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Config file (used to read the telemetry DSN for the pipeline snapshot).",
)
@click.pass_obj
def run_promote(
    obj: dict[str, Any], usecase_id: str, run_id: str, config_path: Path
) -> None:
    """Promote a validated run to production and record its pipeline snapshot."""
    registry = Registry(obj["data_root"])
    try:
        registry.promote_run_to_production(usecase_id, run_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Run {run_id!r} is now in production.")

    # Record the immutable pipeline snapshot if a DSN is configured.
    if not config_path.exists():
        return
    from toolforge_core import load_config

    try:
        config = load_config(config_path)
    except Exception:  # noqa: BLE001
        return
    dsn = config.telemetry.dsn
    if not dsn:
        click.echo(
            "Warning: no DSN configured — pipeline snapshot not recorded. "
            "Set [telemetry] dsn = … in toolforge.toml.",
            err=True,
        )
        return
    try:
        from toolforge_telemetry.production import build_pipeline_spec, get_production_store

        store = get_production_store(dsn)
        store.record_pipeline_spec(build_pipeline_spec(registry, usecase_id, run_id))
        click.echo("Pipeline snapshot recorded.")
    except Exception as e:  # noqa: BLE001
        click.echo(f"Warning: pipeline snapshot not recorded: {e}", err=True)


@run.command("fork")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--from", "from_run_id", required=True)
@click.pass_obj
def run_fork(obj: dict[str, Any], usecase_id: str, from_run_id: str) -> None:
    """Fork a validated run into a new draft run and print the new ID."""
    try:
        info = Registry(obj["data_root"]).fork_run(usecase_id, from_run_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    click.echo(info.run_id)


@run.command("tools")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--run", "run_id", required=True)
@click.pass_obj
def run_tools(obj: dict[str, Any], usecase_id: str, run_id: str) -> None:
    """List tools defined in a run."""
    try:
        tools = Registry(obj["data_root"]).list_tools(usecase_id, run_id)
    except RegistryError as e:
        raise click.ClickException(str(e)) from e
    if not tools:
        click.echo("No tools defined in this run.")
        return
    for t in tools:
        active = f"v{t.active_version}" if t.active_version else "—"
        click.echo(f"{t.name:<30}  {t.status:<12}  active={active}  versions={len(t.versions)}")


# ---------------------------------------------------------------------------
# sandbox subcommands
# ---------------------------------------------------------------------------


@cli.group()
def sandbox() -> None:
    """Sandbox management (Docker image)."""


@sandbox.command("build")
@click.option(
    "--tag",
    default="toolforge-sandbox:latest",
    show_default=True,
    help="Docker image tag.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable Docker layer cache.",
)
def sandbox_build(tag: str, no_cache: bool) -> None:
    """Build the Docker sandbox image from Dockerfile.sandbox."""
    import subprocess as _sp

    cmd = ["docker", "build", "-f", "Dockerfile.sandbox", "-t", tag, "."]
    if no_cache:
        cmd.insert(2, "--no-cache")
    click.echo(f"Building {tag!r} …")
    result = _sp.run(cmd, check=False)
    if result.returncode != 0:
        raise click.ClickException("Docker build failed — see output above.")
    click.echo(f"Image {tag!r} ready. Set mode = \"docker\" in toolforge.toml to use it.")


# ---------------------------------------------------------------------------
# tui subcommand  (full interactive launcher)
# ---------------------------------------------------------------------------


@cli.command("tui")
@click.option(
    "--config",
    "config_path",
    default="toolforge.toml",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.pass_obj
def tui_command(obj: dict[str, Any], config_path: Path) -> None:
    """Launch the full ToolForge TUI (selector -> creator / consumer)."""
    asyncio.run(_tui(data_root=obj["data_root"], config_path=config_path))


async def _tui(data_root: Path, config_path: Path) -> None:
    from toolforge_tui import ToolForgeApp

    app = ToolForgeApp(data_root=data_root, config_path=config_path)
    await app.run_async()


# ---------------------------------------------------------------------------
# creator subcommands
# ---------------------------------------------------------------------------


@cli.group()
def creator() -> None:
    """Creator agent — interactively build tools for a use case."""


@creator.command("run")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--run", "run_id", required=True)
@click.option(
    "--config",
    "config_path",
    default="toolforge.toml",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.pass_obj
def creator_run(
    obj: dict[str, Any],
    usecase_id: str,
    run_id: str,
    config_path: Path,
) -> None:
    """Launch the interactive Creator TUI for a run."""
    asyncio.run(
        _creator_run(
            data_root=obj["data_root"],
            usecase_id=usecase_id,
            run_id=run_id,
            config_path=config_path,
        )
    )


def _resolve_provider(config: Any, name: str, config_path: Path) -> Any:
    if name not in config.llm.providers:
        defined = ", ".join(f"[llm.providers.{k}]" for k in config.llm.providers) or "(none)"
        raise click.ClickException(
            f"Provider {name!r} is referenced in the config but not defined.\n"
            f"Defined providers: {defined}\n"
            f"Add a [llm.providers.{name}] section to {config_path}."
        )
    return config.llm.providers[name]


async def _creator_run(
    data_root: Path,
    usecase_id: str,
    run_id: str,
    config_path: Path,
) -> None:
    from mcp.client.stdio import StdioServerParameters

    from toolforge_core import LLMAgent, create_client, creator_agent_stdio, load_config
    from toolforge_tui import CreatorApp

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "Create toolforge.toml or pass --config <path>."
        )
    config = load_config(config_path)
    provider_conf = _resolve_provider(config, config.llm.creator.provider, config_path)
    client = create_client(config.llm.creator.provider, provider_conf)
    agent = LLMAgent.from_config(config.llm.creator, client)

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "toolforge_mcp_creator",
            "--usecase", usecase_id,
            "--run", run_id,
            "--data-root", str(data_root),
            "--config", str(config_path),
        ],
    )
    async with creator_agent_stdio(stdio_params=params, agent=agent) as creator_agent:
        app = CreatorApp(
            agent=creator_agent,
            usecase_id=usecase_id,
            run_id=run_id,
            data_root=data_root,
        )
        await app.run_async()


# ---------------------------------------------------------------------------
# consumer subcommands
# ---------------------------------------------------------------------------


@cli.group()
def consumer() -> None:
    """Consumer agent — execute a use case using its active tools."""


@consumer.command("run")
@click.option("--usecase", "usecase_id", required=True)
@click.option("--run", "run_id", required=True)
@click.option("--task", required=True, help="Task description for the agent.")
@click.option(
    "--input-file", "input_files",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="File(s) to stage in inputs/ and reference in the task (repeatable).",
)
@click.option(
    "--config",
    "config_path",
    default="toolforge.toml",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.pass_obj
def consumer_run(
    obj: dict[str, Any],
    usecase_id: str,
    run_id: str,
    task: str,
    input_files: tuple[Path, ...],
    config_path: Path,
) -> None:
    """Run a task using the active tools of a validated run."""
    effective_task = task
    if input_files:
        try:
            registry = Registry(obj["data_root"])
            staged = [registry.add_input(usecase_id, f) for f in input_files]
        except RegistryError as e:
            raise click.ClickException(str(e)) from e
        paths_str = "\n".join(f"  - {p}" for p in staged)
        effective_task = f"Available inputs:\n{paths_str}\n\n{task}"
        for p in staged:
            click.echo(f"Staged: {p}", err=True)
    asyncio.run(
        _consumer_run(
            data_root=obj["data_root"],
            usecase_id=usecase_id,
            run_id=run_id,
            task=effective_task,
            config_path=config_path,
        )
    )


async def _consumer_run(
    data_root: Path,
    usecase_id: str,
    run_id: str,
    task: str,
    config_path: Path,
) -> None:
    import secrets

    from mcp.client.stdio import StdioServerParameters

    from toolforge_core import (
        LLMAgent, MessageComplete, TextDelta,
        ToolCallComplete, ToolCallStart, ToolResultEvent,
    )
    from toolforge_core import consumer_agent_stdio, create_client, load_config

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "Create toolforge.toml or pass --config <path>."
        )
    config = load_config(config_path)
    provider_conf = _resolve_provider(config, config.llm.consumer.provider, config_path)
    client = create_client(config.llm.consumer.provider, provider_conf)
    agent = LLMAgent.from_config(config.llm.consumer, client)
    registry = Registry(data_root)
    uc_instructions = registry.get_consumer_prompt(usecase_id)
    if uc_instructions:
        agent.system_prompt = uc_instructions

    # Wire production telemetry for in_production runs.
    prod_store = None
    task_id: str | None = None
    try:
        run_info = registry.get_run(usecase_id, run_id)
        if run_info.status == "in_production":
            task_id = secrets.token_hex(12)
            dsn = config.telemetry.dsn
            if dsn:
                from toolforge_telemetry.production import get_production_store
                prod_store = get_production_store(dsn)
            else:
                click.echo(
                    "Warning: in_production run but no DSN configured — "
                    "telemetry will not be recorded. Set [telemetry] dsn = … in toolforge.toml.",
                    err=True,
                )
    except Exception:
        pass

    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "toolforge_mcp_usecase",
            "--usecase", usecase_id,
            "--run", run_id,
            "--data-root", str(data_root),
            "--config", str(config_path),
        ],
    )
    async with consumer_agent_stdio(
        stdio_params=params,
        agent=agent,
        prod_store=prod_store,
        task_id=task_id,
        run_id=run_id,
        usecase_id=usecase_id,
    ) as agent_inst:
        final_status = "success"
        try:
            gen = await agent_inst.run_task(task)
            async for ev in gen:
                if isinstance(ev, TextDelta):
                    click.echo(ev.text, nl=False)
                elif isinstance(ev, ToolCallStart):
                    click.echo(f"\n⚙  {ev.name}", err=True)
                elif isinstance(ev, ToolCallComplete):
                    args_preview = ", ".join(
                        f"{k}={v!r}" for k, v in list(ev.input.items())[:2]
                    )
                    click.echo(f"   ({args_preview})", err=True)
                elif isinstance(ev, ToolResultEvent):
                    click.echo(f"   → {ev.result[:120]}", err=True)
                elif isinstance(ev, MessageComplete):
                    click.echo("")
        except Exception:
            final_status = "failed"
            raise
        finally:
            agent_inst.close_session(status=final_status)
