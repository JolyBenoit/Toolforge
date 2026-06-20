"""Creator screen — interactive tool-design loop."""
from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, LoadingIndicator, RichLog

from toolforge_core.creator import CreatorAgent
from toolforge_core.types import MessageComplete, TextDelta, ToolCallComplete, ToolCallStart
from toolforge_registry import Registry

from .._utils import format_args

CSS = """\
CreatorScreen {
    layout: vertical;
}
CreatorScreen #main-area {
    height: 1fr;
}
CreatorScreen #chat-panel {
    width: 1fr;
}
CreatorScreen #log {
    height: 1fr;
    border: solid $accent;
    padding: 0 1;
}
CreatorScreen #streaming {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}
CreatorScreen #spinner {
    height: 1;
    display: none;
}
CreatorScreen #spinner.-busy {
    display: block;
}
CreatorScreen #input {
    height: 3;
    border: solid $accent;
    margin-top: 1;
}
CreatorScreen #tools-panel {
    width: 38;
    border: solid $accent;
    padding: 0 1;
    margin-left: 1;
}
CreatorScreen #tools-header {
    height: 1;
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}
CreatorScreen #tools-log {
    height: 1fr;
}
"""


class CreatorScreen(Screen[None]):
    """Interactive Creator agent session."""

    CSS = CSS
    BINDINGS = [("escape", "dismiss", "Back to selector")]

    stream_text: reactive[str] = reactive("")
    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        agent: CreatorAgent,
        usecase_id: str,
        run_id: str,
        data_root: Path,
        seed_message: str | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._usecase_id = usecase_id
        self._run_id = run_id
        self._registry = Registry(data_root)
        # An optional opening turn (e.g. the Judge's approved-instruction
        # briefing) auto-dispatched on mount, so the operator lands on a Creator
        # already working through the corrective changes.
        self._seed_message = seed_message

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            with Vertical(id="chat-panel"):
                yield RichLog(id="log", highlight=True, markup=True, wrap=True)
                yield Label("", id="streaming", markup=True)
                yield LoadingIndicator(id="spinner")
                yield Input(id="input", placeholder="Message the Creator agent… (Escape to go back)")
            with Vertical(id="tools-panel"):
                yield Label("Active Tools", id="tools-header")
                yield RichLog(id="tools-log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ToolForge Creator"
        self.sub_title = f"{self._usecase_id}  ·  {self._run_id}"
        log = self.query_one("#log", RichLog)
        log.write(
            f"[bold]Creator[/]  use-case [cyan]{self._usecase_id}[/]  run [cyan]{self._run_id}[/]"
        )
        log.write("Type a message to the Creator agent. Press [bold]Escape[/] to return.\n")
        self.query_one(Input).focus()
        self._refresh_tools()
        if self._seed_message:
            self._dispatch_seed(self._seed_message)

    def _dispatch_seed(self, message: str) -> None:
        """Auto-send the opening briefing as the first turn."""
        self.busy = True
        log = self.query_one("#log", RichLog)
        log.write("[bold green]You[/] [dim](Judge briefing):[/]")
        log.write(message)
        self._stream_turn(message)

    def watch_stream_text(self, text: str) -> None:
        label = self.query_one("#streaming", Label)
        label.update(f"[bold blue]Assistant:[/] {text}▋" if text else "")

    def watch_busy(self, is_busy: bool) -> None:
        self.query_one("#spinner", LoadingIndicator).set_class(is_busy, "-busy")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.busy:
            return
        event.input.clear()
        self.busy = True
        self.query_one("#log", RichLog).write(f"\n[bold green]You:[/] {text}")
        self._stream_turn(text)

    def _refresh_tools(self) -> None:
        log = self.query_one("#tools-log", RichLog)
        log.clear()
        try:
            tools = self._registry.get_active_tools(self._usecase_id, self._run_id)
        except Exception:  # noqa: BLE001
            log.write("[dim]Could not load tools[/]")
            return
        if not tools:
            log.write("[dim]No active tools yet[/]")
            return
        for tool in tools:
            active_v = tool.active_version
            validated_mark = ""
            for v in tool.versions:
                if v.version == active_v and v.sandbox_validated:
                    validated_mark = " [green]✓[/]"
                    break
            log.write(f"[bold cyan]{tool.name}[/] v{active_v}{validated_mark}")
            desc = tool.description if len(tool.description) <= 34 else tool.description[:33] + "…"
            log.write(f"[dim]{desc}[/]")

    @work
    async def _stream_turn(self, message: str) -> None:
        log = self.query_one("#log", RichLog)
        text_buf: list[str] = []
        try:
            gen = await self._agent.run_turn(message)
            async for ev in gen:
                if isinstance(ev, TextDelta):
                    text_buf.append(ev.text)
                    self.stream_text = "".join(text_buf)
                elif isinstance(ev, ToolCallStart):
                    if text_buf:
                        log.write("[bold blue]Assistant:[/] " + "".join(text_buf))
                        text_buf = []
                        self.stream_text = ""
                    log.write(f"[bold yellow]  ⚙ {ev.name}[/]")
                elif isinstance(ev, ToolCallComplete):
                    args_str = format_args(ev.input)
                    if args_str:
                        log.write(f"    [dim]{args_str}[/]")
                elif isinstance(ev, MessageComplete):
                    if text_buf:
                        log.write("[bold blue]Assistant:[/] " + "".join(text_buf))
                        text_buf = []
                    self.stream_text = ""
        except Exception as exc:  # noqa: BLE001
            log.write(f"[bold red]ERROR:[/] {exc}")
        finally:
            self.stream_text = ""
            self.busy = False
            self._refresh_tools()
            self.query_one(Input).focus()
