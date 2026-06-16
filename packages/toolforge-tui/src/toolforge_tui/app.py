"""ToolForge TUI — app entry points."""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult

from toolforge_core.creator import CreatorAgent

from .screens.creator import CreatorScreen
from .screens.selector import SelectorScreen


class ToolForgeApp(App[None]):
    """Full ToolForge TUI: selector → creator or consumer."""

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, data_root: Path, config_path: Path) -> None:
        super().__init__()
        self._data_root = data_root
        self._config_path = config_path

    def on_mount(self) -> None:
        self.push_screen(
            SelectorScreen(data_root=self._data_root, config_path=self._config_path)
        )


class CreatorApp(App[None]):
    """Standalone creator app used by `toolforge creator run` (deep-link shortcut)."""

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(
        self, agent: CreatorAgent, usecase_id: str, run_id: str, data_root: Path
    ) -> None:
        super().__init__()
        self._agent = agent
        self._usecase_id = usecase_id
        self._run_id = run_id
        self._data_root = data_root

    def on_mount(self) -> None:
        self.push_screen(
            CreatorScreen(
                agent=self._agent,
                usecase_id=self._usecase_id,
                run_id=self._run_id,
                data_root=self._data_root,
            )
        )
