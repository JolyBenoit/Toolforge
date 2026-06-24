"""Selector screen — browse use cases and runs, launch creator or consumer."""
from __future__ import annotations

import asyncio
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Header, Input, Label, Tree

from toolforge_registry import Registry
from toolforge_registry.models import RunInfo


class _ConfirmScreen(ModalScreen[bool]):
    """Minimal yes/no confirmation modal."""

    CSS = """\
    _ConfirmScreen {
        align: center middle;
    }
    _ConfirmScreen #dialog {
        width: 62;
        height: auto;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }
    _ConfirmScreen #buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    _ConfirmScreen #buttons Button {
        margin-left: 1;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(self._message, markup=True)
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Confirm", variant="warning", id="confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class _RenameUsecaseScreen(ModalScreen[str | None]):
    """Modal that collects a new id for an existing use case."""

    CSS = """\
    _RenameUsecaseScreen {
        align: center middle;
    }
    _RenameUsecaseScreen #dialog {
        width: 64;
        height: auto;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    _RenameUsecaseScreen #buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    _RenameUsecaseScreen #buttons Button {
        margin-left: 1;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, old_id: str) -> None:
        super().__init__()
        self._old_id = old_id

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(f"[bold]Rename use case[/] [cyan]{self._old_id}[/]", markup=True)
            yield Label("New ID (letters, digits, underscores):")
            yield Input(id="id-input", value=self._old_id)
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Rename", variant="primary", id="rename")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        new_id = self.query_one("#id-input", Input).value.strip()
        if not new_id:
            self.notify("A new ID is required.", severity="warning")
            return
        self.dismiss(new_id)


@dataclass
class _UsecaseData:
    usecase_id: str


@dataclass
class _RunData:
    usecase_id: str
    run: RunInfo


CSS = """\
SelectorScreen {
    layout: vertical;
}
SelectorScreen #tree {
    height: 1fr;
    border: solid $accent;
    padding: 0 1;
}
SelectorScreen #detail {
    height: 4;
    border: solid $panel;
    padding: 0 1;
    color: $text-muted;
}
"""


# ---------------------------------------------------------------------------
# Command palette provider
# ---------------------------------------------------------------------------


class _SelectorCommands(Provider):
    """Commands available in the selector screen, shown in the command palette."""

    def _commands(self) -> list[tuple[str, str, object]]:
        screen: SelectorScreen = self.screen  # type: ignore[assignment]
        sel = screen._selected_run()
        result: list[tuple[str, str, object]] = []

        if sel is not None:
            run_id = sel.run.run_id
            if sel.run.status == "draft":
                result += [
                    (
                        f"Open Creator  [{run_id}]",
                        "Design and iterate on tools for this run",
                        screen.action_open_creator,
                    ),
                    (
                        f"Validate run  [{run_id}]",
                        "Freeze this run's tools (human validation gate) so it can be tested in Consumer",
                        screen.action_validate_run,
                    ),
                ]
            if sel.run.status == "validated":
                result += [
                    (
                        f"Open Consumer  [{run_id}]",
                        "Test this run by running a task against its active tools",
                        screen.action_open_consumer,
                    ),
                    (
                        f"Promote to production  [{run_id}]",
                        "Start recording Consumer runs for the Judge (pins the current tool versions)",
                        screen.action_promote_to_production,
                    ),
                    (
                        f"Fork run  [{run_id}]",
                        "Create a new draft run that inherits the current tools",
                        screen.action_fork_run,
                    ),
                    (
                        f"Edit run  [{run_id}]",
                        "Unlock this run back to draft and open it in Creator to modify its tools",
                        screen.action_edit_run,
                    ),
                ]
            if sel.run.status == "in_production":
                result += [
                    (
                        f"Open Consumer  [{run_id}]  [production]",
                        "Run a task against this production pipeline (telemetry recorded)",
                        screen.action_open_consumer,
                    ),
                    (
                        f"Open Judge  [{run_id}]",
                        "Inspect telemetry-backed metrics (run counts, reliability, …) for this production run",
                        screen.action_open_judge,
                    ),
                    (
                        f"Fork run  [{run_id}]",
                        "Create a new draft run that inherits the current tools",
                        screen.action_fork_run,
                    ),
                ]

        uc_id = screen._selected_usecase_id()
        if uc_id is not None:
            result.append(
                (
                    f"Rename use case  [{uc_id}]",
                    "Rename this use case everywhere (folder + judge/telemetry rows)",
                    screen.action_rename_usecase,
                )
            )

        result += [
            ("New use case", "Create a new use case with an ID and a prompt", screen.action_new_usecase),
            ("Refresh", "Reload the use case list from disk", screen.action_refresh),
        ]
        return result

    async def discover(self) -> Hits:
        """Show all commands immediately when the palette opens (before typing)."""
        for name, help_text, action in self._commands():
            yield DiscoveryHit(
                display=name,
                command=action,  # type: ignore[arg-type]
                text=name,
                help=help_text,
            )

    async def search(self, query: str) -> Hits:
        q = query.lower()
        for name, help_text, action in self._commands():
            if q in name.lower():
                yield Hit(
                    score=1.0,
                    match_display=name,
                    command=action,  # type: ignore[arg-type]
                    text=name,
                    help=help_text,
                )


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------


class SelectorScreen(Screen[None]):
    """Browse use cases and runs; open creator or consumer via the command palette."""

    CSS = CSS
    COMMANDS = {_SelectorCommands}
    BINDINGS = [
        ("n", "new_usecase", "New use case"),
        ("f2", "rename_usecase", "Rename"),
        ("v", "validate_run", "Validate"),
        ("p", "promote_to_production", "Promote"),
        ("f", "fork_run", "Fork"),
        ("e", "edit_run", "Edit"),
        ("r", "run", "Run"),
        ("j", "open_judge", "Judge"),
        ("ctrl+r", "refresh", "Refresh"),
        ("ctrl+c", "app.quit", "Quit"),
    ]

    def __init__(self, data_root: Path, config_path: Path) -> None:
        super().__init__()
        self._data_root = data_root
        self._config_path = config_path
        self._registry = Registry(data_root)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Tree("Use Cases", id="tree")
        yield Label("", id="detail", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ToolForge"
        self.sub_title = "Use Cases"
        self._populate_tree()

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _populate_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.expand()

        usecases = self._registry.list_usecases()
        if not usecases:
            tree.root.add_leaf("[dim]No use cases yet — press n to create one[/]")
            return

        for uc in usecases:
            uc_node = tree.root.add(
                f"[bold]{uc.usecase_id}[/]",
                data=_UsecaseData(usecase_id=uc.usecase_id),
                expand=True,
            )
            runs = self._registry.list_runs(uc.usecase_id)
            if not runs:
                uc_node.add_leaf("[dim](no runs)[/]")
            else:
                for run in runs:
                    if run.status == "validated":
                        color = "green"
                    elif run.status == "in_production":
                        color = "blue"
                    else:
                        color = "yellow"
                    tools_str = f"{run.tool_count} tool(s)" if run.tool_count else "no tools"
                    uc_node.add_leaf(
                        f"[{color}]{run.status:<13}[/]  "
                        f"[cyan]{run.run_id}[/]  "
                        f"[dim]{tools_str}[/]",
                        data=_RunData(usecase_id=uc.usecase_id, run=run),
                    )

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self.refresh_bindings()
        detail = self.query_one("#detail", Label)
        data = event.node.data
        if isinstance(data, _RunData):
            run = data.run
            forked = f"  forked from [cyan]{run.forked_from}[/]" if run.forked_from else ""
            if run.status == "validated":
                status_color = "green"
            elif run.status == "in_production":
                status_color = "blue"
            else:
                status_color = "yellow"
            detail.update(
                f"[bold]{run.run_id}[/]  "
                f"status=[{status_color}]{run.status}[/]  "
                f"tools={run.tool_count}  "
                f"created={run.created_at.date()}{forked}\n"
                f"[dim]Press [/][bold]Ctrl+P[/][dim] to open Creator, Consumer, or other actions[/]"
            )
        elif isinstance(data, _UsecaseData):
            detail.update(
                f"[bold]{data.usecase_id}[/]  "
                f"[dim]Select a run, then press [/][bold]Ctrl+P[/][dim] to act on it[/]"
            )
        else:
            detail.update("[dim]Press [/][bold]Ctrl+P[/][dim] to open the command palette[/]")

    # ------------------------------------------------------------------
    # Binding visibility
    # ------------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        try:
            sel = self._selected_run()
        except Exception:
            return True
        status = sel.run.status if sel is not None else None
        if action == "rename_usecase":
            return True if self._selected_usecase_id() is not None else None
        if action == "run":
            return True if status in ("validated", "in_production") else None
        if action == "validate_run":
            return True if status == "draft" else None
        if action == "promote_to_production":
            return True if status == "validated" else None
        if action == "fork_run":
            return True if status in ("validated", "in_production") else None
        if action == "open_judge":
            return True if status == "in_production" else None
        if action == "edit_run":
            return True if status in ("draft", "validated") else None
        return True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh(self) -> None:
        self._registry = Registry(self._data_root)
        self._populate_tree()
        self.notify("Refreshed.", severity="information", timeout=1)

    def action_validate_run(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status != "draft":
            self.notify("Run is already validated.", severity="warning")
            return
        try:
            self._registry.validate_run(sel.usecase_id, sel.run.run_id)
            self.notify(f"{sel.run.run_id!r} validated and locked.", severity="information")
            self._populate_tree()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")

    def action_promote_to_production(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status != "validated":
            self.notify("Only validated runs can be promoted to production.", severity="warning")
            return
        self.app.push_screen(
            _ConfirmScreen(
                f"Promote [bold cyan]{sel.run.run_id}[/] to [blue]production[/]?\n\n"
                "From now on, Consumer runs are [blue]recorded for the Judge[/]. "
                "The current tool versions are pinned so scores stay attributable.\n\n"
                "[dim]Validated runs are pre-production: they can be tested in Consumer "
                "but their telemetry is not persisted.[/]"
            ),
            callback=lambda confirmed: self._on_promote_confirmed(sel, confirmed),
        )

    def _on_promote_confirmed(self, sel: _RunData, confirmed: bool) -> None:
        if not confirmed:
            return
        self._promote_run(sel)

    @work
    async def _promote_run(self, sel: _RunData) -> None:
        """Promote the run and pin its pipeline spec off the UI thread.

        The DB work (``promote_run_to_production`` writes to disk, the pipeline
        spec opens a Postgres connection) is offloaded with ``to_thread`` so a
        slow or unreachable database can never freeze the TUI event loop.
        """
        try:
            await asyncio.to_thread(
                self._registry.promote_run_to_production, sel.usecase_id, sel.run.run_id
            )
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")
            return
        await self._record_pipeline_spec(sel.usecase_id, sel.run.run_id)
        self._populate_tree()
        self.notify(f"{sel.run.run_id!r} promoted to production.", severity="information")
        self._launch_consumer(sel.usecase_id, sel.run.run_id, "in_production")

    async def _record_pipeline_spec(self, usecase_id: str, run_id: str) -> None:
        """Persist the immutable pipeline snapshot if a Postgres DSN is set."""
        from toolforge_core import load_config

        try:
            config = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Pipeline version not pinned: {exc}", severity="warning", timeout=6)
            return
        dsn = config.telemetry.dsn
        if not dsn:
            self.notify(
                "No DSN configured — pipeline version not pinned and runs won't be recorded. "
                "Set [telemetry] dsn = … in toolforge.toml to enable production telemetry.",
                severity="warning",
                timeout=6,
            )
            return

        def _persist() -> None:
            from toolforge_telemetry.production import build_pipeline_spec, get_production_store

            store = get_production_store(dsn)
            spec = build_pipeline_spec(self._registry, usecase_id, run_id)
            store.record_pipeline_spec(spec)

        try:
            await asyncio.to_thread(_persist)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Pipeline version not pinned: {exc}", severity="warning", timeout=6)

    def action_fork_run(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status != "validated":
            self.notify("Only validated runs can be forked.", severity="warning")
            return
        try:
            new_run = self._registry.fork_run(sel.usecase_id, sel.run.run_id)
            self.notify(f"Forked to {new_run.run_id!r}", severity="information")
            self._populate_tree()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")

    def action_new_usecase(self) -> None:
        from .new_usecase import NewUsecaseScreen
        self.app.push_screen(NewUsecaseScreen(), callback=self._on_new_usecase_result)

    def _on_new_usecase_result(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        usecase_id, prompt = result
        try:
            self._registry.create_usecase(usecase_id, prompt)
            self.notify(f"Use case {usecase_id!r} created.", severity="information")
            self._populate_tree()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")

    def action_rename_usecase(self) -> None:
        uc_id = self._selected_usecase_id()
        if uc_id is None:
            self.notify("Select a use case (or one of its runs) first.", severity="warning")
            return
        self.app.push_screen(
            _RenameUsecaseScreen(uc_id),
            callback=lambda new_id: self._on_rename_result(uc_id, new_id),
        )

    def _on_rename_result(self, old_id: str, new_id: str | None) -> None:
        if new_id is None:
            return
        new_id = new_id.strip()
        if not new_id or new_id == old_id:
            return
        self._rename_usecase(old_id, new_id)

    @work
    async def _rename_usecase(self, old_id: str, new_id: str) -> None:
        """Rename a use case across the filesystem and every Postgres store.

        Order matters for crash safety: the Postgres updates run first (each is
        idempotent — ``UPDATE … WHERE usecase_id = old`` is a no-op once moved),
        then the folder is moved last. So a failure after a partial DB update
        heals on a simple retry: the folder is still under the old id, the DB
        rows already moved are skipped, and the move completes.
        """
        if self._registry.usecase_exists(new_id):
            self.notify(f"A use case named {new_id!r} already exists.", severity="error")
            return

        from toolforge_core import load_config

        dsn = ""
        try:
            config = load_config(self._config_path)
            dsn = config.telemetry.dsn or ""
        except Exception as exc:  # noqa: BLE001
            self.notify(
                f"Could not read config ({exc}); renaming folder only, "
                "judge/telemetry rows left untouched.",
                severity="warning",
                timeout=8,
            )

        if dsn:
            try:
                await asyncio.to_thread(self._rename_db_rows, dsn, old_id, new_id)
            except Exception as exc:  # noqa: BLE001
                self.notify(
                    f"Database rename failed ({exc}); folder left untouched. "
                    "Fix the database and retry.",
                    severity="error",
                    timeout=10,
                )
                return

        try:
            await asyncio.to_thread(self._registry.rename_usecase, old_id, new_id)
        except Exception as exc:  # noqa: BLE001
            self.notify(
                f"Folder rename failed after DB update ({exc}). "
                f"Database rows now reference {new_id!r} but the folder is still "
                f"{old_id!r} — retry the rename to finish.",
                severity="error",
                timeout=12,
            )
            return

        self._registry = Registry(self._data_root)
        self._populate_tree()
        self.notify(f"Renamed {old_id!r} → {new_id!r}.", severity="information")

    def _rename_db_rows(self, dsn: str, old_id: str, new_id: str) -> None:
        """Repoint usecase_id in every Postgres-backed store. Runs off the UI thread."""
        from toolforge_judge.architecture import get_architecture_judge_store
        from toolforge_judge.creator import get_creator_judge_store
        from toolforge_judge.dynamic import get_dynamic_judge_store
        from toolforge_judge.static import get_judge_store
        from toolforge_telemetry import get_store
        from toolforge_telemetry.production import get_production_store

        stores = [
            get_store("in_production", telemetry_dir=self._data_root, pg_dsn=dsn),
            get_production_store(dsn),
            get_judge_store(dsn),
            get_dynamic_judge_store(dsn),
            get_creator_judge_store(dsn),
            get_architecture_judge_store(dsn),
        ]
        for store in stores:
            store.rename_usecase(old_id, new_id)

    def action_edit_run(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status == "draft":
            self._launch_creator(sel.usecase_id, sel.run.run_id)
            return
        if sel.run.status != "validated":
            self.notify("Select a draft or validated run.", severity="warning")
            return
        self.app.push_screen(
            _ConfirmScreen(
                f"Unlock [bold cyan]{sel.run.run_id}[/] for editing?\n\n"
                "The run will return to [yellow]draft[/] status. "
                "Re-validate when you are done."
            ),
            callback=lambda confirmed: self._on_edit_run_confirmed(sel, confirmed),
        )

    def _on_edit_run_confirmed(self, sel: _RunData, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            self._registry.unlock_run(sel.usecase_id, sel.run.run_id)
            self._populate_tree()
            self.notify(f"{sel.run.run_id!r} unlocked — opening Creator…", severity="information")
            self._launch_creator(sel.usecase_id, sel.run.run_id)
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")

    def action_open_creator(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status != "draft":
            self.notify("Creator requires a draft run. Fork it to iterate.", severity="warning")
            return
        self._launch_creator(sel.usecase_id, sel.run.run_id)

    def action_open_consumer(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status not in ("validated", "in_production"):
            self.notify("Consumer requires a validated or in_production run.", severity="warning")
            return
        self._launch_consumer(sel.usecase_id, sel.run.run_id, sel.run.status)

    def action_open_judge(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status != "in_production":
            self.notify("The Judge is only available for production runs.", severity="warning")
            return
        from toolforge_core import load_config

        from .judge import JudgeScreen

        try:
            config = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not open Judge: {exc}", severity="error")
            return
        self.app.push_screen(
            JudgeScreen(
                usecase_id=sel.usecase_id,
                run_id=sel.run.run_id,
                dsn=config.telemetry.dsn,
                config_path=self._config_path,
                data_root=self._data_root,
            )
        )

    def action_run(self) -> None:
        sel = self._selected_run()
        if sel is None:
            self.notify("Select a run first.", severity="warning")
            return
        if sel.run.status not in ("validated", "in_production"):
            self.notify("Only validated or in_production runs can be launched.", severity="warning")
            return
        self._launch_consumer(sel.usecase_id, sel.run.run_id, sel.run.status)

    # ------------------------------------------------------------------
    # MCP screen launchers
    # ------------------------------------------------------------------

    @work
    async def _launch_creator(self, usecase_id: str, run_id: str) -> None:
        from mcp.client.stdio import StdioServerParameters

        from toolforge_core import LLMAgent, create_client, creator_agent_stdio, load_config

        from .creator import CreatorScreen

        try:
            config = load_config(self._config_path)
            provider_conf = config.llm.providers[config.llm.creator.provider]
            client = create_client(config.llm.creator.provider, provider_conf)
            agent = LLMAgent.from_config(config.llm.creator, client)
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m", "toolforge_mcp_creator",
                    "--usecase", usecase_id,
                    "--run", run_id,
                    "--data-root", str(self._data_root),
                    "--config", str(self._config_path),
                ],
            )
            async with creator_agent_stdio(stdio_params=params, agent=agent) as creator_agent:
                await self.app.push_screen_wait(
                    CreatorScreen(
                        agent=creator_agent,
                        usecase_id=usecase_id,
                        run_id=run_id,
                        data_root=self._data_root,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not open Creator: {exc}", severity="error")
        finally:
            self._registry = Registry(self._data_root)
            self._populate_tree()

    @work
    async def _launch_consumer(
        self, usecase_id: str, run_id: str, run_status: str = "validated"
    ) -> None:
        from mcp.client.stdio import StdioServerParameters

        from toolforge_core import LLMAgent, create_client, consumer_agent_stdio, load_config

        from .consumer import ConsumerScreen

        try:
            config = load_config(self._config_path)
            provider_conf = config.llm.providers[config.llm.consumer.provider]
            client = create_client(config.llm.consumer.provider, provider_conf)
            agent = LLMAgent.from_config(config.llm.consumer, client)
            uc_instructions = self._registry.get_consumer_prompt(usecase_id)
            if uc_instructions:
                agent.system_prompt = uc_instructions

            # Wire production telemetry for in_production runs.
            prod_store = None
            task_id: str | None = None
            dsn = config.telemetry.dsn if config.telemetry.dsn else ""
            if run_status == "in_production":
                if dsn:
                    try:
                        from toolforge_telemetry.production import get_production_store
                        prod_store = await asyncio.to_thread(get_production_store, dsn)
                    except Exception as exc:  # noqa: BLE001
                        self.notify(
                            f"Production telemetry unavailable: {exc}",
                            severity="warning",
                        )
                else:
                    self.notify(
                        "No DSN configured — production run will proceed without telemetry. "
                        "Set [telemetry] dsn = … in toolforge.toml to enable it.",
                        severity="warning",
                        timeout=6,
                    )
                task_id = secrets.token_hex(12)

            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m", "toolforge_mcp_usecase",
                    "--usecase", usecase_id,
                    "--run", run_id,
                    "--data-root", str(self._data_root),
                    "--config", str(self._config_path),
                ],
            )
            async with consumer_agent_stdio(
                stdio_params=params,
                agent=agent,
                prod_store=prod_store,
                task_id=task_id,
                run_id=run_id,
                usecase_id=usecase_id,
            ) as consumer_agent:
                await self.app.push_screen_wait(
                    ConsumerScreen(
                        agent=consumer_agent,
                        usecase_id=usecase_id,
                        run_id=run_id,
                        inputs_dir=self._registry.inputs_dir(usecase_id),
                        outputs_dir=self._registry.outputs_dir(usecase_id),
                        uc_instructions=uc_instructions,
                        is_production=(run_status == "in_production"),
                        dsn=dsn,
                        config_path=self._config_path,
                        data_root=self._data_root,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not open Consumer: {exc}", severity="error")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_run(self) -> _RunData | None:
        node = self.query_one(Tree).cursor_node
        if node is None or not isinstance(node.data, _RunData):
            return None
        return node.data

    def _selected_usecase_id(self) -> str | None:
        """The use case under the cursor, whether a use case row or one of its runs."""
        node = self.query_one(Tree).cursor_node
        if node is None:
            return None
        data = node.data
        if isinstance(data, _UsecaseData):
            return data.usecase_id
        if isinstance(data, _RunData):
            return data.usecase_id
        return None
