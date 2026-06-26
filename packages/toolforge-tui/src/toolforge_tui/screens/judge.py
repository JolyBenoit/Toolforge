"""Judge screen — telemetry-backed metrics dashboard for a production run.

Entry points (see ``selector`` and ``consumer`` screens):

* from the selector, on any run already promoted to ``in_production``;
* from the consumer, while running a task against a production pipeline.

Three tabs:

* **Metrics** — the run counts and every metric the
  :class:`~toolforge_judge.metrics.engine.MetricEngine` can compute without an
  LLM verdict (metrics still needing a Judge pass are listed dimmed);
* **Static** — runs the *static* judge over the use case's not-yet-judged
  tasks, across every pipeline version (incremental, skip-judged);
* **Dynamic** — runs the *dynamic* judge over a chosen set of pipeline versions
  and window sizes, producing a cross-run report.

Launching the judges needs a config + data root (to build the judge LLM and the
per-run tool specs); without them the two tabs stay read-only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)
from toolforge_judge.metrics.base import JUDGED, OK, REQUIRES_JUDGE
from toolforge_judge.metrics.data import TelemetryReader
from toolforge_judge.metrics.engine import MetricEngine, MetricReport
from toolforge_judge.metrics.env import MetricEnv

CSS = """\
JudgeScreen {
    layout: vertical;
}
JudgeScreen #summary {
    height: auto;
    border: solid $accent;
    padding: 0 1;
}
JudgeScreen TabbedContent {
    height: 1fr;
}
JudgeScreen #metrics {
    height: 1fr;
    border: solid $panel;
}
JudgeScreen #status {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}
JudgeScreen .pane-info {
    height: auto;
    padding: 1 1 0 1;
}
JudgeScreen .pane-controls {
    height: auto;
    padding: 1;
}
JudgeScreen .pane-controls Button {
    margin-right: 2;
}
JudgeScreen .pane-log {
    height: 1fr;
    border: solid $panel;
    margin: 0 1 1 1;
}
JudgeScreen #creator-list {
    height: 1fr;
    border: solid $panel;
    margin: 0 1 1 1;
    padding: 0 1;
}
JudgeScreen .rec-section {
    height: auto;
    margin-top: 1;
    text-style: bold;
}
JudgeScreen #creator-list Checkbox {
    height: auto;
    margin: 0 0 1 0;
}
JudgeScreen #runs-table {
    height: 2fr;
    border: solid $panel;
    margin: 0 1;
}
JudgeScreen #runs-detail {
    height: 3fr;
    border: solid $panel;
    margin: 0 1;
}
JudgeScreen #runs-status {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}
"""

# Statuses telemetry assigns to a finished task, in display order.
_SUCCESS = "success"
_FAILED = "failed"

# Every status a task may carry (mirrors telemetry's ``TaskStatus``); the Runs
# tab can set any of these on the selected tasks.
_TASK_STATUSES = ("success", "partial", "failed", "user_aborted", "running")


@dataclass
class _JudgeData:
    """Everything the metrics tab needs, computed off the UI thread."""

    counts: dict[str, int]
    report: MetricReport
    judge_info: str = ""  # diagnostic: how many judge notes were folded in
    window_tasks: int = 0  # tasks loaded into the metric window
    window_tools: int = 0  # distinct tools seen — 0 ⇒ no per-tool metric rows


@dataclass
class _RunSelection:
    """The dynamic judge's target: a set of pipeline versions + window sizes.

    ``run_ids`` of ``None`` means "all versions" (the leading window across the
    whole history).
    """

    run_ids: list[str] | None
    short_window: int
    long_window: int


def _fmt_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


# Colour per task status, for the Runs table and detail header.
_STATUS_COLORS = {
    "success": "green",
    "failed": "red",
    "partial": "yellow",
    "user_aborted": "yellow",
    "running": "cyan",
}


def _status_text(status: str) -> Text:
    return Text(status, style=_STATUS_COLORS.get(status, "dim"))


def _status_markup(status: str) -> str:
    return f"[{_STATUS_COLORS.get(status, 'dim')}]{status}[/]"


# ---------------------------------------------------------------------------
# Run / window selection modal (dynamic judge)
# ---------------------------------------------------------------------------


class _RunSelectScreen(ModalScreen[_RunSelection | None]):
    """Pick which pipeline versions and window sizes feed the dynamic judge."""

    CSS = """\
    _RunSelectScreen {
        align: center middle;
    }
    _RunSelectScreen #dialog {
        width: 72;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    _RunSelectScreen SelectionList {
        height: auto;
        max-height: 14;
        margin: 1 0;
        border: solid $panel;
    }
    _RunSelectScreen .field {
        height: auto;
        margin-bottom: 1;
    }
    _RunSelectScreen Input {
        width: 12;
    }
    _RunSelectScreen #buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    _RunSelectScreen #buttons Button {
        margin-left: 1;
    }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        runs: list[tuple[str, int]],
        *,
        current_run_id: str | None,
        short_window: int,
        long_window: int,
    ) -> None:
        super().__init__()
        self._runs = runs
        self._current = current_run_id
        self._short = short_window
        self._long = long_window

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(
                "[bold]Dynamic judge[/] — choose pipeline versions to aggregate.\n"
                "[dim]Leave all unselected to assess every version.[/]",
                markup=True,
            )
            options = [
                (f"{rid}   [{n} task(s)]", rid, rid == self._current)
                for rid, n in self._runs
            ]
            yield SelectionList[str](*options, id="runs")
            with Horizontal(classes="field"):
                yield Label("Short window: ")
                yield Input(value=str(self._short), id="short", type="integer")
            with Horizontal(classes="field"):
                yield Label("Long window:  ")
                yield Input(value=str(self._long), id="long", type="integer")
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Run", variant="primary", id="ok")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        selected = list(self.query_one("#runs", SelectionList).selected)
        try:
            short = int(self.query_one("#short", Input).value or self._short)
            long = int(self.query_one("#long", Input).value or self._long)
        except ValueError:
            self.notify("Window sizes must be integers.", severity="error")
            return
        if short < 1 or long < 1 or short > long:
            self.notify(
                "Need 1 ≤ short ≤ long for the windows.", severity="error"
            )
            return
        self.dismiss(
            _RunSelection(
                run_ids=selected or None, short_window=short, long_window=long
            )
        )


class _ConfirmScreen(ModalScreen[bool]):
    """A small yes/no modal — used to gate the irreversible run deletion."""

    CSS = """\
    _ConfirmScreen {
        align: center middle;
    }
    _ConfirmScreen #dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $error;
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
        with Vertical(id="dialog"):
            yield Label(self._message, markup=True)
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Delete", variant="error", id="ok")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "ok")


class JudgeScreen(Screen[None]):
    """Metrics dashboard + judge launcher for one production use case."""

    CSS = CSS
    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("ctrl+r", "reload", "Reload"),
        ("s", "run_static", "Run static"),
        ("d", "run_dynamic", "Run dynamic"),
        ("a", "run_architecture", "Run architecture"),
        ("c", "run_creator", "Recommendations"),
    ]

    def __init__(
        self,
        usecase_id: str,
        run_id: str | None,
        dsn: str,
        env: MetricEnv | None = None,
        *,
        config_path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._usecase_id = usecase_id
        self._run_id = run_id
        self._dsn = dsn
        self._env = env or MetricEnv()
        self._config_path = config_path
        self._data_root = data_root
        self._busy = False
        # The most recent dynamic report computed in this session — the creator
        # judge consumes it (avoids a heavy MetricReport rehydration from the DB).
        self._last_dynamic_report: object | None = None
        # The creator report currently rendered (freshly built or loaded), so
        # "Send to Creator" can map checked boxes back to instructions.
        self._current_creator_report: object | None = None
        # Runs tab: task summaries by task_id, and the multi-selected task_ids
        # (toggled with Enter on a row). The Runs tab needs only the DSN, so it
        # works even when the screen is read-only (judges disabled).
        self._run_summaries: dict[str, dict] = {}
        self._runs_selected: set[str] = set()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def _can_run(self) -> bool:
        """Whether we have everything needed to actually launch the judges."""
        return bool(self._dsn and self._config_path and self._data_root)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary", markup=True)
        with TabbedContent(initial="tab-metrics"):
            with TabPane("Metrics", id="tab-metrics"):
                yield DataTable(id="metrics", zebra_stripes=True)
                yield Static("[dim]Loading telemetry…[/]", id="status", markup=True)
            with TabPane("Static", id="tab-static"):
                yield Static("", id="static-info", classes="pane-info", markup=True)
                with Horizontal(classes="pane-controls"):
                    yield Button(
                        "Run static judge", id="run-static", variant="primary"
                    )
                    yield Checkbox("Re-judge all", id="static-all")
                yield RichLog(id="static-log", classes="pane-log", markup=True)
            with TabPane("Dynamic", id="tab-dynamic"):
                yield Static("", id="dynamic-info", classes="pane-info", markup=True)
                with Horizontal(classes="pane-controls"):
                    yield Button(
                        "Select runs & window…", id="run-dynamic", variant="primary"
                    )
                yield RichLog(id="dynamic-log", classes="pane-log", markup=True)
            with TabPane("Architecture", id="tab-architecture"):
                yield Static(
                    "", id="architecture-info", classes="pane-info", markup=True
                )
                with Horizontal(classes="pane-controls"):
                    yield Button(
                        "Run architecture judge", id="run-architecture",
                        variant="primary",
                    )
                yield RichLog(
                    id="architecture-log", classes="pane-log", markup=True
                )
            with TabPane("Recommendations", id="tab-creator"):
                yield Static("", id="creator-info", classes="pane-info", markup=True)
                with Horizontal(classes="pane-controls"):
                    yield Button(
                        "Build recommendations", id="run-creator", variant="primary"
                    )
                    yield Button(
                        "Send to Creator (0)", id="send-creator",
                        variant="success", disabled=True,
                    )
                yield VerticalScroll(id="creator-list")
            with TabPane("Runs", id="tab-runs"):
                yield DataTable(id="runs-table", zebra_stripes=True)
                yield RichLog(id="runs-detail", markup=True)
                with Horizontal(classes="pane-controls"):
                    yield Button("✓ success", id="runs-mark-success")
                    yield Button("~ partial", id="runs-mark-partial")
                    yield Button("✗ failed", id="runs-mark-failed")
                    yield Button("⏹ aborted", id="runs-mark-user_aborted")
                    yield Button(
                        "Delete selected", id="runs-delete", variant="error"
                    )
                    yield Button("Delete all", id="runs-delete-all", variant="error")
                yield Static("", id="runs-status", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ToolForge Judge"
        scope = self._run_id or "all runs"
        self.sub_title = f"{self._usecase_id}  ·  {scope}"
        table = self.query_one("#metrics", DataTable)
        table.add_columns("Family", "Metric", "Tool", "Value", "n", "Status")
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.cursor_type = "row"
        cols = runs_table.add_columns(
            " ", "Date/heure", "Run", "Status", "#tools", "Tokens", "Latency"
        )
        self._runs_mark_col = cols[0]
        if not self._can_run:
            hint = (
                "[yellow]Read-only — open the Judge from the run selector to "
                "launch the static/dynamic judges.[/]"
            )
            self.query_one("#static-info", Static).update(hint)
            self.query_one("#dynamic-info", Static).update(hint)
            self.query_one("#architecture-info", Static).update(hint)
            self.query_one("#creator-info", Static).update(hint)
            self.query_one("#run-static", Button).disabled = True
            self.query_one("#run-dynamic", Button).disabled = True
            self.query_one("#run-architecture", Button).disabled = True
            self.query_one("#run-creator", Button).disabled = True
        self._load()
        self._load_runs()
        if self._can_run:
            self._refresh_pane_info()
            self.query_one("#architecture-info", Static).update(
                "[bold]Design-time review[/] of the pipeline's tools — reads each "
                "handler's [i]source[/] to find over-simplifications (e.g. silent "
                "truncation), coverage gaps, redundancy and wiring issues.  "
                "[dim]Advisory; feeds the creator judge. Changes nothing.[/]"
            )
            self._load_architecture_report()
            self.query_one("#creator-info", Static).update(
                "[bold]Corrective recommendations[/] for the Creator, synthesised "
                "from the dynamic report + architecture findings.  [dim]Run the "
                "Dynamic judge first; uncheck what you don't want, then send the "
                "selection to a Creator on a fresh draft fork.[/]"
            )
            self._load_creator_report()

    def action_reload(self) -> None:
        self.query_one("#status", Static).update("[dim]Reloading telemetry…[/]")
        self._load()
        if self._can_run:
            self._refresh_pane_info()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-static":
            self.action_run_static()
        elif event.button.id == "run-dynamic":
            self.action_run_dynamic()
        elif event.button.id == "run-architecture":
            self.action_run_architecture()
        elif event.button.id == "run-creator":
            self.action_run_creator()
        elif event.button.id == "send-creator":
            self._send_to_creator()
        elif event.button.id and event.button.id.startswith("runs-mark-"):
            self._set_runs_status(event.button.id[len("runs-mark-"):])
        elif event.button.id == "runs-delete":
            self._delete_runs(all_listed=False)
        elif event.button.id == "runs-delete-all":
            self._delete_runs(all_listed=True)

    # ------------------------------------------------------------------
    # Metrics tab — read-only telemetry (off the UI thread)
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _load(self) -> None:
        if not self._dsn:
            self.query_one("#status", Static).update(
                "[yellow]No DSN configured — set [telemetry] dsn = … in "
                "toolforge.toml to enable production telemetry.[/]"
            )
            return
        try:
            data = await asyncio.to_thread(self._compute)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#status", Static).update(
                f"[red]Could not read telemetry: {exc}[/]"
            )
            return
        self._show(data)

    def _compute(self) -> _JudgeData:
        """Blocking telemetry read + metric computation (runs in a thread)."""
        reader = TelemetryReader(self._dsn)
        counts = reader.count_by_status(self._usecase_id, run_id=self._run_id)
        window = reader.load_window(self._usecase_id, self._env, run_id=self._run_id)
        info = self._attach_judge_scores(window)
        report = MetricEngine().compute(window)
        return _JudgeData(
            counts=counts, report=report, judge_info=info,
            window_tasks=len(window.long), window_tools=len(window.tool_ids),
        )

    def _attach_judge_scores(self, window) -> str:  # noqa: ANN001
        """Fold any persisted static-judge notes into the metric window so the
        judge-scored families show as evaluated rather than pending.

        Returns a short diagnostic for the status line (note count or error).
        """
        try:
            from toolforge_judge.dynamic import judge_scores_from_notes
            from toolforge_judge.static import get_judge_store

            notes = get_judge_store(self._dsn).load_tool_notes(
                self._usecase_id, run_id=self._run_id
            )
        except Exception as exc:  # noqa: BLE001 - judge tables may not exist yet
            return f"[red]judge notes unreadable: {exc}[/]"
        if not notes:
            return "no judge notes stored yet"
        window.judge_scores, window.judged_tools = judge_scores_from_notes(
            notes, self._env
        )
        tools = ", ".join(sorted(window.judged_tools or set())) or "—"
        return f"{len(notes)} judge note(s) on tool(s): {tools}"

    def _show(self, data: _JudgeData) -> None:
        self._render_summary(data.counts)
        self._render_metrics(data.report)
        ok = sum(1 for v in data.report.values if v.status == OK)
        judged = sum(1 for v in data.report.values if v.status == JUDGED)
        pending = sum(1 for v in data.report.values if v.status == REQUIRES_JUDGE)
        judged_txt = f"{judged} judged (no value), " if judged else ""
        info = f"  ·  {data.judge_info}" if data.judge_info else ""
        tools_warn = "red" if data.window_tools == 0 else "dim"
        window_txt = (
            f"[{tools_warn}]window: {data.window_tasks} task(s), "
            f"{data.window_tools} tool(s)[/]"
        )
        self.query_one("#status", Static).update(
            f"[dim]{ok} computed metric(s), {judged_txt}{pending} awaiting a "
            f"Judge pass.[/]  {window_txt}{info}  "
            "[dim]Ctrl+R to reload · Escape to go back.[/]"
        )

    def _render_summary(self, counts: dict[str, int]) -> None:
        total = sum(counts.values())
        success = counts.get(_SUCCESS, 0)
        failed = counts.get(_FAILED, 0)
        other = total - success - failed
        decided = success + failed
        rate = f"{success / decided * 100:.1f}%" if decided else "—"
        summary = self.query_one("#summary", Static)
        if total == 0:
            summary.update(
                "[bold]No recorded runs yet.[/]  "
                "[dim]Run a task against this production pipeline to start "
                "collecting telemetry.[/]"
            )
            return
        parts = [
            f"[bold]{total}[/] run(s)",
            f"[green]✓ {success} success[/]",
            f"[red]✗ {failed} failed[/]",
        ]
        if other:
            parts.append(f"[yellow]· {other} other[/]")
        parts.append(f"success rate [bold]{rate}[/]")
        summary.update("    ".join(parts))

    def _render_metrics(self, report: MetricReport) -> None:
        table = self.query_one("#metrics", DataTable)
        table.clear()
        # Computed first, then judged-without-value, then pending Judge passes.
        rank = {OK: 0, JUDGED: 1}
        ordered = sorted(
            report.values,
            key=lambda v: (rank.get(v.status, 2), v.family, v.metric, v.tool_id or ""),
        )
        for v in ordered:
            cells = [v.family, v.metric, v.tool_id or "—", _fmt_value(v.value), str(v.n)]
            if v.breached:
                status = Text("breach", style="red bold")
                row_style = "red"
            elif v.status == OK:
                status = Text("ok", style="green")
                row_style = ""
            elif v.status == JUDGED:
                status = Text("judged", style="cyan")
                row_style = "cyan"
            else:
                status = Text(v.status, style="dim")
                row_style = "dim"
            table.add_row(*(Text(c, style=row_style) for c in cells), status)

    # ------------------------------------------------------------------
    # Static / Dynamic tab info (judged-progress + available runs)
    # ------------------------------------------------------------------

    @work(exclusive=False)
    async def _refresh_pane_info(self) -> None:
        try:
            info = await asyncio.to_thread(self._read_pane_info)
        except Exception as exc:  # noqa: BLE001
            self.query_one("#static-info", Static).update(
                f"[red]Could not read judge state: {exc}[/]"
            )
            return
        total, judged, runs = info
        remaining = total - judged
        self.query_one("#static-info", Static).update(
            f"[bold]{judged}/{total}[/] task(s) judged across all versions — "
            f"[{'green' if remaining == 0 else 'yellow'}]{remaining} remaining[/].  "
            "[dim]‘Re-judge all’ re-runs even already-judged tasks.[/]"
        )
        self.query_one("#run-static", Button).disabled = remaining == 0
        run_list = ", ".join(f"{rid} ({n})" for rid, n in runs) or "—"
        self.query_one("#dynamic-info", Static).update(
            f"[bold]{len(runs)}[/] pipeline version(s) available: [dim]{run_list}[/]"
        )

    def _read_pane_info(self) -> tuple[int, int, list[tuple[str, int]]]:
        """Blocking: total tasks, judged count, and per-version task counts."""
        from toolforge_judge.static import get_judge_store

        reader = TelemetryReader(self._dsn)
        runs = reader.count_tasks_by_run(self._usecase_id)
        total = sum(n for _, n in runs)
        store = get_judge_store(self._dsn)
        judged = len(store.judged_task_ids(self._usecase_id))
        return total, judged, runs

    # ------------------------------------------------------------------
    # Static judge — judge every not-yet-judged task of the use case
    # ------------------------------------------------------------------

    def action_run_static(self) -> None:
        if not self._can_run or self._busy:
            return
        skip_judged = not self.query_one("#static-all", Checkbox).value
        self.query_one(TabbedContent).active = "tab-static"
        self._run_static(skip_judged)

    @work(exclusive=True)
    async def _run_static(self, skip_judged: bool) -> None:
        log = self.query_one("#static-log", RichLog)
        self._set_busy(True)
        log.write("[bold]Static judge[/] — starting…")
        try:
            # Built on the UI loop: the judge LLM is awaited here, so its async
            # client must be created on the same loop it runs on.
            judge, registry, reader, store, concurrency = (
                self._build_static_resources()
            )
            from toolforge_judge.static import run_usecase

            failures = 0

            def _on_error(task, exc) -> None:  # noqa: ANN001
                nonlocal failures
                failures += 1
                log.write(f"  [red]✗ {task.task_id}: {exc}[/]")

            results = await run_usecase(
                judge, registry, reader, self._usecase_id,
                store=store, skip_judged=skip_judged,
                progress=lambda d, t: log.write(f"  judged {d}/{t}"),
                on_error=_on_error, max_concurrency=concurrency,
            )
        except Exception as exc:  # noqa: BLE001
            log.write(f"[red]Static judge failed: {exc}[/]")
            self.notify(f"Static judge failed: {exc}", severity="error")
            self._set_busy(False)
            return
        tail = f", {failures} failed" if failures else ""
        log.write(f"[green]Done — {len(results)} task(s) judged{tail}.[/]")
        self.notify(
            f"Static judge: {len(results)} task(s) judged{tail}.", timeout=4
        )
        self._set_busy(False)
        # Metrics now have fresh contribution write-backs; refresh everything.
        self._load()
        self._refresh_pane_info()

    def _build_static_resources(self):  # noqa: ANN202 - internal tuple
        """Build the judge LLM, registry, reader, store and concurrency bound."""
        from toolforge_judge.static import StaticJudge, get_judge_store
        from toolforge_registry import Registry

        config = self._load_config()
        judge = StaticJudge(self._build_judge_llm(config))
        registry = Registry(self._data_root)
        reader = TelemetryReader(self._dsn)
        store = get_judge_store(self._dsn)
        return judge, registry, reader, store, config.judge.max_concurrency

    # ------------------------------------------------------------------
    # Dynamic judge — cross-run report over a chosen set of versions
    # ------------------------------------------------------------------

    def action_run_dynamic(self) -> None:
        if not self._can_run or self._busy:
            return
        self.query_one(TabbedContent).active = "tab-dynamic"
        self._open_run_selector()

    @work(exclusive=True)
    async def _open_run_selector(self) -> None:
        try:
            runs = await asyncio.to_thread(
                lambda: TelemetryReader(self._dsn).count_tasks_by_run(self._usecase_id)
            )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not list runs: {exc}", severity="error")
            return
        if not runs:
            self.notify("No recorded runs to assess yet.", severity="warning")
            return
        selection = await self.app.push_screen_wait(
            _RunSelectScreen(
                runs,
                current_run_id=self._run_id,
                short_window=self._env.short_window,
                long_window=self._env.long_window,
            )
        )
        if selection is None:
            return
        self._run_dynamic(selection)

    @work(exclusive=True)
    async def _run_dynamic(self, selection: _RunSelection) -> None:
        log = self.query_one("#dynamic-log", RichLog)
        self._set_busy(True)
        scope = ", ".join(selection.run_ids) if selection.run_ids else "all versions"
        log.write(f"[bold]Dynamic judge[/] — assessing {scope}…")
        try:
            report = await asyncio.to_thread(self._dynamic_blocking, selection)
        except Exception as exc:  # noqa: BLE001
            log.write(f"[red]Dynamic judge failed: {exc}[/]")
            self.notify(f"Dynamic judge failed: {exc}", severity="error")
            self._set_busy(False)
            return
        # Cache it so the Recommendations tab can build on it without a re-run.
        self._last_dynamic_report = report
        self._render_dynamic_report(log, report)
        self.query_one("#creator-info", Static).update(
            "[bold]Corrective recommendations[/] — dynamic report ready.  "
            "[dim]Press [b]Build recommendations[/] to synthesise the changes.[/]"
        )
        self.notify("Dynamic judge done.", timeout=4)
        self._set_busy(False)

    def _dynamic_blocking(self, selection: _RunSelection):  # noqa: ANN202
        """Blocking: build resources and run the dynamic judge in a thread.

        The judge mixes blocking DB reads with an optional async LLM diagnosis,
        so it runs on its own event loop off the UI thread.
        """
        from toolforge_judge.dynamic import DynamicJudge
        from toolforge_judge.dynamic.store import get_dynamic_judge_store
        from toolforge_judge.static import get_judge_store

        config = self._load_config()
        llm = self._build_judge_llm(config)
        env = replace(
            self._env,
            short_window=selection.short_window,
            long_window=selection.long_window,
        )
        reader = TelemetryReader(self._dsn)
        notes_store = get_judge_store(self._dsn)
        dyn_store = get_dynamic_judge_store(self._dsn)
        return asyncio.run(
            DynamicJudge(llm=llm).run(
                reader, notes_store, self._usecase_id, env,
                run_ids=selection.run_ids, store=dyn_store, diagnose=True,
            )
        )

    def _render_dynamic_report(self, log: RichLog, report) -> None:  # noqa: ANN001
        ss = report.structural_stability
        mean = ss.mean_structural_stability
        mean_txt = f"{mean:.3f}" if mean is not None else "—"
        breach = "[red]breached[/]" if ss.breached else "[green]ok[/]"
        log.write(
            f"[green]Report[/] [dim]({report.run_id or 'all'}, "
            f"{report.n_tasks} task(s))[/]"
        )
        log.write(f"  structural stability: [bold]{mean_txt}[/] {breach}")
        breaches = report.metric_report.breaches
        if breaches:
            names = ", ".join(sorted({b.metric for b in breaches}))
            log.write(f"  [red]metric breaches:[/] {names}")
        for note in report.tool_global_notes:
            flags = f" [red]{','.join(note.breaches)}[/]" if note.breaches else ""
            log.write(
                f"  tool [cyan]{note.tool_id}[/]: "
                f"rec rate {note.recommendation_rate:.0%}{flags}"
            )
        if report.diagnosis:
            log.write(f"  [italic]{report.diagnosis}[/]")

    # ------------------------------------------------------------------
    # Architecture judge — design-time review of the pipeline's tools
    # ------------------------------------------------------------------

    def action_run_architecture(self) -> None:
        if not self._can_run or self._busy:
            return
        self.query_one(TabbedContent).active = "tab-architecture"
        self._run_architecture()

    @work(exclusive=True)
    async def _run_architecture(self) -> None:
        log = self.query_one("#architecture-log", RichLog)
        self._set_busy(True)
        log.write("[bold]Architecture judge[/] — building pipeline spec…")
        try:
            run_id = await asyncio.to_thread(self._resolve_arch_run_id)
            if run_id is None:
                log.write("[yellow]No pipeline version found to assess.[/]")
                self._set_busy(False)
                return
            # Spec assembly is blocking (registry file reads); the judge's LLM is
            # awaited here so its async client is built on the UI loop it runs on.
            config = self._load_config()
            spec = await asyncio.to_thread(
                self._build_architecture_spec, run_id
            )
            log.write(
                f"[dim]Assessing {run_id} — {len(spec.tools)} tool(s), "
                "design-time (reading handler source)…[/]"
            )
            judge = self._build_architecture_judge(config)
            report = await judge.assess(
                spec,
                max_concurrency=config.judge.max_concurrency,
                on_phase=lambda name, total: self._log_arch_phase(log, name, total),
                on_tool=lambda tid, done, total: self._log_arch_tool(
                    log, tid, done, total
                ),
            )
            await asyncio.to_thread(self._save_arch_report, report)
        except Exception as exc:  # noqa: BLE001
            from rich.markup import escape

            log.write(f"[red]Architecture judge failed:[/] {escape(str(exc))}")
            self.notify(f"Architecture judge failed: {exc}", severity="error")
            self._set_busy(False)
            return
        self._render_architecture_report(log, report)
        self.notify(
            f"Architecture judge: {len(report.findings)} finding(s).", timeout=4
        )
        self._set_busy(False)

    def _log_arch_phase(self, log: RichLog, name: str, total: int) -> None:
        """Announce a stage of the architecture pass (fires on the UI loop)."""
        if name == "contracts":
            log.write(f"[dim]Reading {total} tool contract(s)…[/]")
        elif name == "findings":
            log.write("[dim]Synthesising pipeline findings…[/]")

    def _log_arch_tool(self, log: RichLog, tool_id: str, done: int, total: int) -> None:
        """Tick off one tool as its contract is read (fires on the UI loop)."""
        from rich.markup import escape

        log.write(f"  [green]✓[/] {escape(tool_id)} [dim]({done}/{total})[/]")

    @work(exclusive=False)
    async def _load_architecture_report(self) -> None:
        """Render the last persisted architecture report so it is consultable on
        reopen, without re-running the (LLM-backed) assessment."""
        try:
            report = await asyncio.to_thread(self._read_architecture_report)
        except Exception:  # noqa: BLE001 - tables may not exist yet
            return
        if report is not None:
            self._render_architecture_report(
                self.query_one("#architecture-log", RichLog), report, persisted=True
            )

    def _read_architecture_report(self):  # noqa: ANN202
        from toolforge_judge.architecture import get_architecture_judge_store

        return get_architecture_judge_store(self._dsn).load_report(
            self._usecase_id, run_id=self._resolve_arch_run_id()
        )

    def _resolve_arch_run_id(self) -> str | None:
        """The pipeline version to assess: the screen's run, else the latest."""
        if self._run_id:
            return self._run_id
        runs = TelemetryReader(self._dsn).count_tasks_by_run(self._usecase_id)
        return runs[0][0] if runs else None

    def _build_architecture_spec(self, run_id: str):  # noqa: ANN202
        from toolforge_judge.architecture import build_architecture_spec
        from toolforge_registry import Registry

        registry = Registry(self._data_root)
        return build_architecture_spec(registry, self._usecase_id, run_id)

    def _build_architecture_judge(self, config):  # noqa: ANN001, ANN202
        """Build the two-pass architecture judge on the judge LLM backend.

        Both passes share the judge's model/client but use their own system
        prompts, resolved next to the judge prompt configured in toolforge.toml.
        """
        from toolforge_core import create_client
        from toolforge_core.config import load_system_prompt
        from toolforge_judge.architecture import ArchitectureJudge
        from toolforge_judge.static import AgentLLMJudge

        jc = config.llm.judge
        provider_conf = config.llm.providers[jc.provider]
        client = create_client(jc.provider, provider_conf)
        prompts_dir = jc.system_prompt_file.parent

        def _llm(filename: str) -> AgentLLMJudge:
            return AgentLLMJudge(
                client=client,
                model=jc.model,
                system_prompt=load_system_prompt(prompts_dir / filename),
                max_tokens=jc.max_tokens,
                temperature=jc.temperature,
            )

        return ArchitectureJudge(
            contract_llm=_llm("judge_architecture_tool_system.md"),
            findings_llm=_llm("judge_architecture_system.md"),
        )

    def _save_arch_report(self, report) -> None:  # noqa: ANN001
        from toolforge_judge.architecture import get_architecture_judge_store

        get_architecture_judge_store(self._dsn).save_report(report)

    def _render_architecture_report(  # noqa: ANN001
        self, log: RichLog, report, *, persisted: bool = False
    ) -> None:
        if persisted:
            # A reloaded report omits the pass-1 contracts (the store keeps only
            # findings), so report on when it was saved rather than "0 read".
            meta = f"saved {report.computed_at}"
            log.write("[dim]— last saved architecture assessment —[/]")
        else:
            meta = f"{len(report.contracts)} tool contract(s) read"
        if not report.findings:
            log.write(f"[green]No design issues found.[/] [dim]({meta}.)[/]")
            return
        sev_color = {"error": "red", "warning": "yellow", "info": "cyan"}
        log.write(
            f"[green]{len(report.findings)} finding(s)[/] "
            f"[dim]({report.mode}, {meta})[/]"
        )
        for f in report.findings:
            color = sev_color.get(f.severity, "white")
            tools = ", ".join(f.tools_involved) or "—"
            log.write(
                f"  [{color}]{f.severity.upper()}[/] [bold]{f.category}[/] "
                f"[dim]({tools})[/] → [italic]{f.proposed_action}[/]"
            )
            log.write(f"    {f.body}")
            if f.requirement_threatened:
                log.write(f"    [dim]at risk: {f.requirement_threatened}[/]")

    # ------------------------------------------------------------------
    # Recommendations tab — corrective instructions for the Creator
    # ------------------------------------------------------------------

    def action_run_creator(self) -> None:
        if not self._can_run or self._busy:
            return
        self.query_one(TabbedContent).active = "tab-creator"
        if self._last_dynamic_report is None:
            self.notify(
                "Run the Dynamic judge first (Dynamic tab) — its report feeds "
                "the recommendations.",
                severity="warning", timeout=6,
            )
            return
        self._run_creator()

    @work(exclusive=True)
    async def _run_creator(self) -> None:
        self._set_busy(True)
        info = self.query_one("#creator-info", Static)
        info.update("[bold]Synthesising recommendations…[/]")
        try:
            # The judge LLMs are awaited here, so build them on the UI loop.
            config = self._load_config()
            judge = self._build_creator_judge(config)
            usecase, arch_report, store = await asyncio.to_thread(
                self._build_creator_inputs
            )
            from toolforge_judge.creator import run_creator_judge

            report = await run_creator_judge(
                judge, self._last_dynamic_report, usecase,
                architecture_report=arch_report, store=store,
            )
        except Exception as exc:  # noqa: BLE001
            info.update(f"[red]Recommendations failed: {exc}[/]")
            self.notify(f"Recommendations failed: {exc}", severity="error")
            self._set_busy(False)
            return
        self._render_creator_report(report)
        n = len(report.instructions)
        info.update(
            f"[bold]{n}[/] corrective change(s) proposed.  "
            "[dim]Uncheck what you don't want, then ‘Send to Creator’.[/]"
            if n else
            "[green]No corrective changes — nothing breaching.[/]"
        )
        self.notify(f"Recommendations: {n} change(s).", timeout=4)
        self._set_busy(False)

    def _build_creator_inputs(self):  # noqa: ANN202
        """Blocking: the usecase spec, persisted architecture report, store."""
        from toolforge_judge.architecture import get_architecture_judge_store
        from toolforge_judge.creator import get_creator_judge_store
        from toolforge_judge.static.runner import build_usecase_spec
        from toolforge_registry import Registry

        run_id = self._resolve_arch_run_id()
        registry = Registry(self._data_root)
        usecase = build_usecase_spec(registry, self._usecase_id, run_id)
        arch_report = get_architecture_judge_store(self._dsn).load_report(
            self._usecase_id, run_id=run_id
        )
        store = get_creator_judge_store(self._dsn)
        return usecase, arch_report, store

    def _build_creator_judge(self, config):  # noqa: ANN001, ANN202
        """Build the two-stage creator judge on the judge LLM backend.

        Both stages share the judge's model/client but use their own system
        prompts, resolved next to the judge prompt configured in toolforge.toml.
        """
        from toolforge_core import create_client
        from toolforge_core.config import load_system_prompt
        from toolforge_judge.creator import CreatorJudge
        from toolforge_judge.static import AgentLLMJudge

        jc = config.llm.judge
        provider_conf = config.llm.providers[jc.provider]
        client = create_client(jc.provider, provider_conf)
        prompts_dir = jc.system_prompt_file.parent

        def _llm(filename: str) -> AgentLLMJudge:
            return AgentLLMJudge(
                client=client,
                model=jc.model,
                system_prompt=load_system_prompt(prompts_dir / filename),
                max_tokens=jc.max_tokens,
                temperature=jc.temperature,
            )

        return CreatorJudge(
            axes_llm=_llm("judge_creator_axes_system.md"),
            instruction_llm=_llm("judge_creator_system.md"),
        )

    @work(exclusive=False)
    async def _load_creator_report(self) -> None:
        """Render the last persisted creator report so it is consultable on
        reload, without rebuilding (which needs a fresh dynamic run)."""
        try:
            report = await asyncio.to_thread(self._read_creator_report)
        except Exception:  # noqa: BLE001 - tables may not exist yet
            return
        if report is not None:
            self._render_creator_report(report)

    def _read_creator_report(self):  # noqa: ANN202
        from toolforge_judge.creator import get_creator_judge_store

        return get_creator_judge_store(self._dsn).load_report(
            self._usecase_id, run_id=self._resolve_arch_run_id()
        )

    def _render_creator_report(self, report) -> None:  # noqa: ANN001
        """Deconstruct the report into per-tool + structural sub-reports, each
        instruction a checkbox (all checked by default)."""
        self._current_creator_report = report
        container = self.query_one("#creator-list", VerticalScroll)
        container.remove_children()
        if report is None or not report.instructions:
            container.mount(
                Static("[dim]No recommendations yet.[/]", markup=True)
            )
            self._update_send_count()
            return

        per_tool: dict[str, list] = {}
        structural: list = []
        for instr in report.instructions:
            if instr.is_structural or not instr.target_tools:
                structural.append(instr)
            else:
                per_tool.setdefault(instr.target_tools[0], []).append(instr)

        for tool_id, instrs in per_tool.items():
            axes = report.axes_for(tool_id)
            head = f"▸ {tool_id}"
            if axes and axes.summary:
                head += f"  [dim]— {axes.summary}[/]"
            container.mount(Static(head, classes="rec-section", markup=True))
            for instr in instrs:
                self._mount_instruction(container, instr)

        if structural:
            container.mount(
                Static("▸ Structural", classes="rec-section", markup=True)
            )
            for instr in structural:
                self._mount_instruction(container, instr)
        self._update_send_count()

    def _mount_instruction(self, container, instr) -> None:  # noqa: ANN001
        targets = ", ".join(instr.target_tools) or "(new tool)"
        container.mount(
            Checkbox(
                f"[{instr.priority.upper()}] {instr.action} · {targets}",
                value=True, id=f"instr-{instr.instruction_id}",
            )
        )
        detail = f"    Change: {instr.body}"
        if instr.rationale:
            detail += f"\n    Why: {instr.rationale}"
        if instr.expected_effect:
            detail += f"\n    Effect: {instr.expected_effect}"
        container.mount(Static(f"[dim]{detail}[/]", markup=True))

    def _checked_instruction_ids(self) -> set[str]:
        ids: set[str] = set()
        for cb in self.query("#creator-list Checkbox").results(Checkbox):
            if cb.value and cb.id and cb.id.startswith("instr-"):
                ids.add(cb.id[len("instr-"):])
        return ids

    def _update_send_count(self) -> None:
        n = len(self._checked_instruction_ids())
        btn = self.query_one("#send-creator", Button)
        btn.label = f"Send to Creator ({n})"
        btn.disabled = n == 0 or self._busy

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id and event.checkbox.id.startswith("instr-"):
            self._update_send_count()

    def _send_to_creator(self) -> None:
        if self._busy or self._current_creator_report is None:
            return
        checked = self._checked_instruction_ids()
        if not checked:
            self.notify("Nothing selected to send.", severity="warning")
            return
        from toolforge_judge.creator import render_briefing

        briefing = render_briefing(self._current_creator_report, checked)
        if not briefing.strip():
            self.notify("Nothing selected to send.", severity="warning")
            return
        self._launch_creator_with_briefing(briefing, len(checked))

    @work
    async def _launch_creator_with_briefing(self, briefing: str, n: int) -> None:
        """Fork the assessed production version into a fresh draft run, then open
        a Creator on that fork seeded with the approved-instruction briefing.

        The Creator can only edit tools on a ``draft`` run, so we never mutate the
        immutable production version — we fork it (copying its tools) and let the
        operator promote the result afterwards.
        """
        import sys

        from mcp.client.stdio import StdioServerParameters
        from toolforge_core import (
            LLMAgent,
            create_client,
            creator_agent_stdio,
            load_config,
        )
        from toolforge_registry import Registry

        from .creator import CreatorScreen

        self._set_busy(True)
        try:
            source_run = self._resolve_arch_run_id()
            if source_run is None:
                self.notify(
                    "No production version to fork.", severity="error"
                )
                self._set_busy(False)
                return
            registry = Registry(self._data_root)
            fork = await asyncio.to_thread(
                registry.fork_run, self._usecase_id, source_run
            )
            config = load_config(self._config_path)
            provider_conf = config.llm.providers[config.llm.creator.provider]
            client = create_client(config.llm.creator.provider, provider_conf)
            agent = LLMAgent.from_config(config.llm.creator, client)
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m", "toolforge_mcp_creator",
                    "--usecase", self._usecase_id,
                    "--run", fork.run_id,
                    "--data-root", str(self._data_root),
                    "--config", str(self._config_path),
                ],
            )
            self.notify(
                f"Forked {source_run} → draft {fork.run_id}; launching Creator "
                f"with {n} change(s).",
                timeout=6,
            )
            async with creator_agent_stdio(
                stdio_params=params, agent=agent
            ) as creator_agent:
                await self.app.push_screen_wait(
                    CreatorScreen(
                        agent=creator_agent,
                        usecase_id=self._usecase_id,
                        run_id=fork.run_id,
                        data_root=self._data_root,
                        seed_message=briefing,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not open Creator: {exc}", severity="error")
        finally:
            self._set_busy(False)

    # ------------------------------------------------------------------
    # Runs tab — browse / edit / delete recorded sessions
    # ------------------------------------------------------------------

    @work(exclusive=False)
    async def _load_runs(self) -> None:
        if not self._dsn:
            self.query_one("#runs-status", Static).update(
                "[yellow]No DSN configured — production telemetry disabled.[/]"
            )
            return
        try:
            summaries = await asyncio.to_thread(
                lambda: TelemetryReader(self._dsn).list_task_summaries(
                    self._usecase_id, run_id=self._run_id
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#runs-status", Static).update(
                f"[red]Could not list runs: {exc}[/]"
            )
            return
        self._render_runs_table(summaries)

    def _render_runs_table(self, summaries: list[dict]) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        self._run_summaries = {s["task_id"]: s for s in summaries}
        # Drop selections for tasks that no longer exist (after a delete/reload).
        self._runs_selected &= set(self._run_summaries)
        for s in summaries:
            mark = "✓" if s["task_id"] in self._runs_selected else ""
            run = s["run_id"]
            run_short = run if len(run) <= 16 else "…" + run[-15:]
            table.add_row(
                mark,
                s["started_at"].strftime("%Y-%m-%d %H:%M"),
                run_short,
                _status_text(s["status"]),
                str(s["tool_calls"]),
                f"{s['tokens']:,}",
                f"{s['latency_ms'] / 1000:.1f}s",
                key=s["task_id"],
            )
        self._update_runs_status_line()
        if summaries:
            self._show_run_detail(summaries[0]["task_id"])
        else:
            self.query_one("#runs-detail", RichLog).clear()

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if event.data_table.id == "runs-table" and event.row_key.value:
            self._show_run_detail(event.row_key.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "runs-table" or not event.row_key.value:
            return
        task_id = event.row_key.value
        if task_id in self._runs_selected:
            self._runs_selected.discard(task_id)
            mark = ""
        else:
            self._runs_selected.add(task_id)
            mark = "✓"
        self.query_one("#runs-table", DataTable).update_cell(
            event.row_key, self._runs_mark_col, mark
        )
        self._update_runs_status_line()

    @work(exclusive=True)
    async def _show_run_detail(self, task_id: str) -> None:
        detail = self.query_one("#runs-detail", RichLog)
        summary = self._run_summaries.get(task_id)
        if summary is None:
            return
        try:
            task = await asyncio.to_thread(self._read_task, task_id)
        except Exception as exc:  # noqa: BLE001
            detail.clear()
            detail.write(f"[red]Could not read run: {exc}[/]")
            return
        detail.clear()
        detail.write(f"[bold]{task_id}[/]  [dim]· run {summary['run_id']}[/]")
        fb = summary.get("user_feedback") or {}
        fb_txt = fb.get("explicit", "none") if isinstance(fb, dict) else "none"
        detail.write(
            f"  {_status_markup(summary['status'])}  ·  "
            f"[bold]{summary['tokens']:,}[/] tokens  ·  "
            f"{summary['tool_calls']} tool call(s)  ·  "
            f"{summary['latency_ms'] / 1000:.1f}s  ·  feedback: {fb_txt}"
        )
        if isinstance(fb, dict) and fb.get("correction_text"):
            detail.write(f"  [dim]↳ {fb['correction_text']}[/]")
        detail.write("[dim]— timeline —[/]")
        if task is None or not task.spans:
            detail.write("[dim]No spans recorded for this run.[/]")
            return
        from ._perf import reconstruct_timeline

        for line in reconstruct_timeline(task.spans):
            detail.write(line)

    def _read_task(self, task_id: str):  # noqa: ANN202
        """Blocking: load one task with its spans for the detail timeline."""
        return TelemetryReader(self._dsn).load_task(task_id)

    def _act_targets(self, *, all_listed: bool) -> list[str]:
        """Task ids an action applies to: the whole list, the multi-selection,
        else the highlighted row."""
        if all_listed:
            return list(self._run_summaries)
        if self._runs_selected:
            return list(self._runs_selected)
        table = self.query_one("#runs-table", DataTable)
        if table.row_count:
            try:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
            except Exception:  # noqa: BLE001 - no valid cursor cell
                return []
            if row_key.value:
                return [row_key.value]
        return []

    @work(exclusive=True)
    async def _set_runs_status(self, status: str) -> None:
        targets = self._act_targets(all_listed=False)
        if not targets:
            self.notify("Select run(s) first.", severity="warning")
            return
        try:
            from toolforge_telemetry.production import get_production_store

            n = await asyncio.to_thread(
                lambda: get_production_store(self._dsn).set_task_status(
                    targets, status
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not update status: {exc}", severity="error")
            return
        self.notify(f"Marked {n} run(s) as {status}.", timeout=3)
        self._runs_selected.clear()
        self._load_runs()
        self._load()  # success/failed counts feed the Metrics summary

    @work(exclusive=True)
    async def _delete_runs(self, *, all_listed: bool) -> None:
        targets = self._act_targets(all_listed=all_listed)
        if not targets:
            self.notify("Select run(s) first.", severity="warning")
            return
        scope = "ALL listed run(s)" if all_listed else f"{len(targets)} run(s)"
        ok = await self.app.push_screen_wait(
            _ConfirmScreen(
                f"[bold]Delete {scope}?[/]\n"
                f"[dim]Hard-deletes {len(targets)} task(s) and their spans. "
                "Irreversible.[/]"
            )
        )
        if not ok:
            return
        try:
            from toolforge_telemetry.production import get_production_store

            n = await asyncio.to_thread(
                lambda: get_production_store(self._dsn).delete_tasks(targets)
            )
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Could not delete: {exc}", severity="error")
            return
        self.notify(f"Deleted {n} run(s).", timeout=3)
        self._runs_selected.clear()
        self._load_runs()
        self._load()

    def _update_runs_status_line(self) -> None:
        total = len(self._run_summaries)
        sel = len(self._runs_selected)
        self.query_one("#runs-status", Static).update(
            f"[dim]{total} run(s) · {sel} selected.  Enter to (de)select a row, "
            "buttons act on the selection (else the highlighted row).[/]"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_config(self):  # noqa: ANN202
        from toolforge_core import load_config

        return load_config(self._config_path)

    def _build_judge_llm(self, config):  # noqa: ANN001, ANN202
        from toolforge_core import create_client
        from toolforge_judge.static import AgentLLMJudge

        provider_conf = config.llm.providers[config.llm.judge.provider]
        client = create_client(config.llm.judge.provider, provider_conf)
        return AgentLLMJudge.from_config(config.llm.judge, client)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.query_one("#run-static", Button).disabled = busy
        self.query_one("#run-dynamic", Button).disabled = busy
        self.query_one("#run-architecture", Button).disabled = busy
        self.query_one("#run-creator", Button).disabled = busy
