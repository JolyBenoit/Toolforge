"""Consumer screen — run tasks against active tools."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, ListView,
    LoadingIndicator, RichLog, TabbedContent, TabPane,
)

from toolforge_core.consumer import ConsumerAgent
from toolforge_core.types import (
    MessageComplete, PauseForUserEvent, TextDelta,
    ToolCallComplete, ToolCallStart, ToolResultEvent,
)

from .._utils import format_args

CSS = """\
ConsumerScreen {
    layout: vertical;
}
ConsumerScreen #main {
    height: 1fr;
}
ConsumerScreen #log {
    width: 1fr;
    border: solid $accent;
    padding: 0 1;
}
ConsumerScreen #side-panel {
    width: 44;
    margin-left: 1;
}
ConsumerScreen #calls-log {
    padding: 0 1;
}
ConsumerScreen #perf-log {
    padding: 0 1;
}
ConsumerScreen #inputs-list {
    height: 1fr;
}
ConsumerScreen #streaming {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}
ConsumerScreen #spinner {
    height: 1;
    display: none;
}
ConsumerScreen #spinner.-busy {
    display: block;
}
ConsumerScreen #input {
    height: 3;
    border: solid $accent;
    margin-top: 1;
}
"""


def _perf_now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _result_preview(result_str: str, max_len: int = 120) -> tuple[str, bool]:
    """Return (markup_preview, is_error) from a raw tool result string."""
    try:
        parsed = json.loads(result_str)
        if isinstance(parsed, dict) and "success" in parsed:
            if not parsed["success"]:
                msg = str(parsed.get("error", "error"))[:max_len]
                return f"[bold red]✗[/] {msg}", True
            skip = {"success", "content", "text", "content_type", "metadata", "pages"}
            parts = [
                f"{k}={v!r}"
                for k, v in parsed.items()
                if k not in skip and not isinstance(v, (dict, list))
            ]
            summary = ", ".join(parts[:4]) or "ok"
            return f"[bold green]✓[/] {summary}", False
    except Exception:  # noqa: BLE001
        pass
    preview = result_str[:max_len] + ("…" if len(result_str) > max_len else "")
    return f"[dim]{preview}[/]", False


class _FeedbackNoteScreen(ModalScreen[str | None]):
    """Optional free-text note captured when marking a session as failed."""

    CSS = """\
    _FeedbackNoteScreen {
        align: center middle;
    }
    _FeedbackNoteScreen #dialog {
        width: 70;
        height: auto;
        background: $surface;
        border: thick $error;
        padding: 1 2;
    }
    _FeedbackNoteScreen #note {
        margin-top: 1;
        border: solid $accent;
    }
    _FeedbackNoteScreen #buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    _FeedbackNoteScreen #buttons Button {
        margin-left: 1;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label(
                "[bold red]✗ Marquer cette session comme ÉCHEC[/]\n"
                "[dim]Note de correction (optionnelle) — ce qui aurait dû se passer :[/]",
                markup=True,
            )
            yield Input(id="note", placeholder="Ex. l'agent a mal extrait le total…")
            with Horizontal(id="buttons"):
                yield Button("Annuler", variant="default", id="cancel")
                yield Button("Enregistrer l'échec", variant="error", id="save")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.dismiss(self.query_one(Input).value.strip())
        else:
            self.dismiss(None)


class ConsumerScreen(Screen[None]):
    """Run tasks against a validated run's active tools."""

    CSS = CSS
    BINDINGS = [
        ("escape", "dismiss", "Back to selector"),
        ("f2", "mark_success", "✓ Succès"),
        ("f3", "mark_failure", "✗ Échec"),
    ]

    stream_text: reactive[str] = reactive("")
    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        agent: ConsumerAgent,
        usecase_id: str,
        run_id: str,
        inputs_dir: Path,
        outputs_dir: Path,
        uc_instructions: str | None = None,
        is_production: bool = False,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._usecase_id = usecase_id
        self._run_id = run_id
        self._inputs_dir = inputs_dir
        self._outputs_dir = outputs_dir
        self._uc_instructions = uc_instructions
        self._is_production = is_production
        self._call_count = 0
        # Session verdict for telemetry; None until the user marks it.
        self._verdict_status: str | None = None

    def _input_files(self) -> list[Path]:
        if not self._inputs_dir.exists():
            return []
        return sorted(f for f in self._inputs_dir.iterdir() if f.is_file())

    def _output_files(self) -> list[Path]:
        if not self._outputs_dir.exists():
            return []
        return sorted(f for f in self._outputs_dir.iterdir() if f.is_file())

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield RichLog(id="log", highlight=True, markup=True, wrap=True)
            with TabbedContent(id="side-panel", initial="tab-inputs"):
                with TabPane("Inputs", id="tab-inputs"):
                    files = self._input_files()
                    if files:
                        items: list[ListItem] = [
                            ListItem(Label(f.name), name=f.name) for f in files
                        ]
                    else:
                        items = [ListItem(Label("[dim]No inputs yet[/]"))]
                    yield ListView(*items, id="inputs-list")
                with TabPane("Calls", id="tab-calls"):
                    yield RichLog(
                        id="calls-log", highlight=True, markup=True, wrap=True
                    )
                with TabPane("Outputs", id="tab-outputs"):
                    out_files = self._output_files()
                    if out_files:
                        out_items: list[ListItem] = [
                            ListItem(Label(f.name)) for f in out_files
                        ]
                    else:
                        out_items = [ListItem(Label("[dim]No outputs yet[/]"))]
                    yield ListView(*out_items, id="outputs-list")
                with TabPane("Perf", id="tab-perf"):
                    yield RichLog(
                        id="perf-log", highlight=True, markup=True, wrap=True
                    )
        yield Label("", id="streaming", markup=True)
        yield LoadingIndicator(id="spinner")
        yield Input(
            id="input",
            placeholder="Describe a task… (click an input to attach it, Escape to go back)",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ToolForge Consumer"
        self.sub_title = f"{self._usecase_id}  ·  {self._run_id}"
        log = self.query_one("#log", RichLog)
        log.write(
            f"[bold]Consumer[/]  use-case [cyan]{self._usecase_id}[/]"
            f"  run [cyan]{self._run_id}[/]"
        )
        if self._uc_instructions:
            log.write(
                "[bold]System prompt[/] [dim](use-case specific — set by Creator):[/]"
            )
            for line in self._uc_instructions.splitlines():
                log.write(f"  [dim]{line}[/]")
            log.write("")
        files = self._input_files()
        if files:
            log.write(
                f"[dim]{len(files)} input(s) available"
                " — click a file in the Inputs panel to attach its name.[/]"
            )
        log.write("Describe a task for the agent to perform using its active tools.")
        if self._is_production:
            log.write(
                "[blue]● Mode production[/] — la télémétrie de cette session est enregistrée.\n"
                "When done, press [bold]F2[/] for ✓ success or [bold]F3[/] for ✗ failure "
                "to record your verdict."
            )
        log.write("Press [bold]Escape[/] to return.\n")
        self.query_one(Input).focus()

    def watch_stream_text(self, text: str) -> None:
        self.query_one("#streaming", Label).update(
            f"[bold green]Agent:[/] {text}▋" if text else ""
        )

    def watch_busy(self, is_busy: bool) -> None:
        self.query("#spinner").set_class(is_busy, "-busy")

    def on_unmount(self) -> None:
        self.workers.cancel_all()
        # Use the user's verdict if they marked one; otherwise the session was
        # left without an explicit judgement.
        self._agent.close_session(status=self._verdict_status or "user_aborted")

    # ------------------------------------------------------------------
    # Production telemetry verdict (F2 / F3 — production runs only)
    # ------------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in ("mark_success", "mark_failure"):
            return True if self._is_production else None
        return True

    def action_mark_success(self) -> None:
        if not self._is_production:
            return
        self._verdict_status = "success"
        self._agent.record_feedback(explicit="thumbs_up")
        self.query_one("#log", RichLog).write(
            "[bold green]✓ Session marquée comme SUCCÈS[/] "
            "[dim](enregistré dans la télémétrie)[/]"
        )
        self.notify("Verdict enregistré : succès.", severity="information", timeout=3)

    def action_mark_failure(self) -> None:
        if not self._is_production:
            return
        self.app.push_screen(_FeedbackNoteScreen(), callback=self._on_failure_note)

    def _on_failure_note(self, note: str | None) -> None:
        if note is None:  # cancelled — leave any existing verdict untouched
            return
        self._verdict_status = "failed"
        self._agent.record_feedback(
            explicit="correction", correction_text=note or None
        )
        log = self.query_one("#log", RichLog)
        log.write(
            "[bold red]✗ Session marquée comme ÉCHEC[/] "
            "[dim](enregistré dans la télémétrie)[/]"
        )
        if note:
            log.write(f"  [dim]note: {note}[/]")
        self.notify("Verdict enregistré : échec.", severity="warning", timeout=3)

    def _refresh_outputs(self) -> None:
        lv = self.query_one("#outputs-list", ListView)
        lv.clear()
        files = self._output_files()
        if files:
            for f in files:
                lv.append(ListItem(Label(f.name)))
        else:
            lv.append(ListItem(Label("[dim]No outputs yet[/]")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.name is None:
            return
        inp = self.query_one(Input)
        current = inp.value.strip()
        inp.value = f"{current} {event.item.name}".strip() if current else event.item.name
        inp.cursor_position = len(inp.value)
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.busy:
            return
        event.input.clear()
        self.busy = True
        log = self.query_one("#log", RichLog)
        log.write(f"\n[bold cyan]Task:[/] {text}")
        self.query_one("#side-panel", TabbedContent).active = "tab-perf"
        self._stream_task(text)

    @work
    async def _stream_task(self, task: str) -> None:
        log = self.query_one("#log", RichLog)
        calls_log = self.query_one("#calls-log", RichLog)
        perf_log = self.query_one("#perf-log", RichLog)
        text_buf: list[str] = []

        # Timing state
        _perf_last: float = 0.0
        _llm_turn: int = 1
        _pending_tool_names: list[str] = []  # collected from ToolCallComplete, in call order
        _model: str = self._agent.model

        try:
            gen = await self._agent.run_task(task)

            # LLM turn 1 starts now (generator is lazy — first iteration triggers the API call)
            _perf_last = time.monotonic()
            perf_log.write(
                f"[dim]{_perf_now()}[/]  [bold cyan]→[/]  LLM   [dim]{_model}[/]"
            )

            async for ev in gen:
                if isinstance(ev, TextDelta):
                    text_buf.append(ev.text)
                    self.stream_text = "".join(text_buf)

                elif isinstance(ev, ToolCallStart):
                    if text_buf:
                        log.write("[bold green]Agent:[/] " + "".join(text_buf))
                        text_buf = []
                        self.stream_text = ""
                    self._call_count += 1
                    log.write(f"[bold yellow]  ⚙ {ev.name}[/]")
                    calls_log.write(
                        f"\n[bold yellow]⚙ [{self._call_count}] {ev.name}[/]"
                    )

                elif isinstance(ev, ToolCallComplete):
                    args_str = format_args(ev.input)
                    if args_str:
                        log.write(f"    [dim]{args_str}[/]")
                    for k, v in ev.input.items():
                        v_str = repr(v)
                        if len(v_str) > 80:
                            v_str = v_str[:77] + "…"
                        calls_log.write(f"  [dim]{k}=[/]{v_str}")
                    # Collect tool name — execution order matches ToolCallComplete order
                    _pending_tool_names.append(ev.name)
                    # Show input snapshot in perf log
                    if ev.input:
                        try:
                            inp_str = json.dumps(ev.input, ensure_ascii=False)
                        except Exception:  # noqa: BLE001
                            inp_str = str(ev.input)
                        if len(inp_str) > 120:
                            inp_str = inp_str[:117] + "…"
                        perf_log.write(f"       [dim]↳ in  {inp_str}[/]")

                elif isinstance(ev, ToolResultEvent):
                    now = time.monotonic()
                    elapsed = now - _perf_last
                    _perf_last = now

                    preview, is_error = _result_preview(ev.result)
                    is_error = is_error or ev.is_error
                    log.write(f"    {preview}")
                    calls_log.write(f"  {preview}")
                    self._refresh_outputs()

                    # Perf: tool response received
                    result_color = "red" if is_error else "green"
                    arrow = "✗" if is_error else "←"
                    perf_log.write(
                        f"[dim]{_perf_now()}[/]  [{result_color}]{arrow}[/]  tool  "
                        f"[bold]{ev.name}[/]"
                        f"  [yellow]{elapsed:.3f}s[/]"
                    )
                    if is_error:
                        detail = ev.result[:200].replace("[", "\\[")
                        perf_log.write(f"       [dim red]↳ err {detail}[/]")
                    elif ev.result:
                        out_preview = ev.result[:120].replace("[", "\\[")
                        if len(ev.result) > 120:
                            out_preview += "…"
                        perf_log.write(f"       [dim]↳ out {out_preview}[/]")
                    if _pending_tool_names:
                        _pending_tool_names.pop(0)

                    if _pending_tool_names:
                        # Next tool starts immediately after this result
                        perf_log.write(
                            f"[dim]{_perf_now()}[/]  [bold cyan]→[/]  tool  "
                            f"{_pending_tool_names[0]}"
                        )
                    else:
                        # All tool results received → LLM will be called again
                        _llm_turn += 1
                        _perf_last = time.monotonic()
                        perf_log.write(
                            f"[dim]{_perf_now()}[/]  [bold cyan]→[/]  LLM   "
                            f"[dim]{_model}[/]  [dim](turn {_llm_turn})[/]"
                        )

                elif isinstance(ev, MessageComplete):
                    now = time.monotonic()
                    elapsed = now - _perf_last
                    _perf_last = now

                    if text_buf:
                        log.write("[bold green]Agent:[/] " + "".join(text_buf))
                        text_buf = []
                    self.stream_text = ""

                    # Perf: LLM turn completed
                    perf_log.write(
                        f"[dim]{_perf_now()}[/]  [bold green]←[/]  LLM   "
                        f"[dim]{_model}[/]"
                        f"  [yellow]{elapsed:.3f}s[/]"
                    )
                    if ev.usage:
                        in_tok = ev.usage.get("input_tokens", 0)
                        out_tok = ev.usage.get("output_tokens", 0)
                        perf_log.write(
                            f"       [dim]↳ {in_tok:,} prompt / {out_tok:,} completion tokens[/]"
                        )

                    # If there are pending tool calls, the first one starts right now
                    if _pending_tool_names:
                        perf_log.write(
                            f"[dim]{_perf_now()}[/]  [bold cyan]→[/]  tool  "
                            f"{_pending_tool_names[0]}"
                        )

                elif isinstance(ev, PauseForUserEvent):
                    self.stream_text = ""
                    log.write(
                        f"\n[bold yellow]⏸  En attente de votre décision[/]"
                        + (f"\n[dim]{ev.message}[/]" if ev.message else "")
                    )
                    perf_log.write(
                        f"[dim]{_perf_now()}[/]  [yellow]⏸[/]  pause  "
                        f"[dim]{ev.tool_name}[/]"
                    )
                    # busy = False is handled by the finally block when the generator ends

        except TimeoutError as exc:
            # httpx.ReadTimeout → LLM API didn't respond in time (likely large context)
            exc_type = type(exc).__name__
            origin = "LLM API" if "read" in exc_type.lower() or not str(exc) else "LLM API"
            detail = str(exc) or "(no detail)"
            msg = f"[bold red]TIMEOUT ({origin}):[/] {detail}"
            log.write(msg)
            calls_log.write(msg)
            perf_log.write(
                f"[dim]{_perf_now()}[/]  [bold red]✗[/]  "
                f"[red]{exc_type}[/] — LLM API call timed out"
            )
            perf_log.write(
                f"       [dim red]↳ probable cause: context window too large "
                f"(check token counts above)[/]"
            )
        except ConnectionError as exc:
            msg = f"[bold red]CONNECTION ERROR:[/] {exc}"
            log.write(msg)
            calls_log.write(msg)
            perf_log.write(
                f"[dim]{_perf_now()}[/]  [bold red]✗[/]  CONNECTION: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            log.write(f"[bold red]ERROR:[/] {exc}")
            calls_log.write(f"[bold red]ERROR:[/] {exc}")
            perf_log.write(
                f"[dim]{_perf_now()}[/]  [bold red]✗[/]  {type(exc).__name__}: {exc}"
            )
        finally:
            self.stream_text = ""
            self.busy = False
            try:
                self.query_one(Input).focus()
            except Exception:  # noqa: BLE001
                pass
