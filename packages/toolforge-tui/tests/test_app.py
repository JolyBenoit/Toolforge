"""TUI tests — helper logic + Textual smoke test (no real LLM or MCP)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from toolforge_tui._utils import format_args
from toolforge_core.types import MessageComplete, TextDelta


# --- format_args ---


def test_format_args_empty() -> None:
    assert format_args({}) == ""


def test_format_args_single() -> None:
    result = format_args({"name": "extract"})
    assert result == "name='extract'"


def test_format_args_multiple() -> None:
    result = format_args({"a": 1, "b": 2})
    assert "a=1" in result
    assert "b=2" in result


def test_format_args_truncates_long_values() -> None:
    result = format_args({"x": "a" * 50})
    assert "..." in result
    assert len(result) < 60


def test_format_args_max_items() -> None:
    args = {str(i): i for i in range(6)}
    result = format_args(args, max_items=3)
    assert "+3 more" in result


def test_format_args_exactly_max_items() -> None:
    args = {"a": 1, "b": 2, "c": 3}
    result = format_args(args, max_items=3)
    assert "more" not in result


# --- CreatorApp smoke test ---


class _MockAgent:
    """Minimal stand-in for CreatorAgent used in TUI tests."""

    history: list = []

    def reset_history(self) -> None:
        self.history = []

    async def run_turn(self, message: str):
        async def _gen():
            yield TextDelta(text="Hello from mock agent!")
            yield MessageComplete(stop_reason="end_turn", message=MagicMock())

        return _gen()


async def test_app_mounts_and_shows_header(tmp_path: Path) -> None:
    from toolforge_tui.app import CreatorApp

    app = CreatorApp(
        agent=_MockAgent(), usecase_id="uc_test", run_id="r_abc123", data_root=tmp_path
    )
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
        screen = app.screen
        assert screen.title == "ToolForge Creator"
        assert "uc_test" in screen.sub_title
        assert "r_abc123" in screen.sub_title
        assert screen.query_one("#log") is not None
        assert screen.query_one("#input") is not None


async def test_app_input_submit_streams_response(tmp_path: Path) -> None:
    from toolforge_tui.app import CreatorApp

    app = CreatorApp(
        agent=_MockAgent(), usecase_id="uc_test", run_id="r_abc123", data_root=tmp_path
    )
    async with app.run_test(headless=True) as pilot:
        await pilot.pause()
        screen = app.screen
        inp = screen.query_one("#input")
        inp.value = "hello agent"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await pilot.pause(delay=0.5)
        assert screen.query_one("#log") is not None
        assert not screen.busy
