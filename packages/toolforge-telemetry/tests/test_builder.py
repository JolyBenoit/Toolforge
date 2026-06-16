"""Tests for build_pipeline_spec — snapshots a run's active tools."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from toolforge_telemetry.production import build_pipeline_spec


@dataclass
class _Tool:
    name: str
    active_version: int


@dataclass
class _Run:
    forked_from: str | None = None
    promoted_to_production_at: datetime | None = None


class _FakeRegistry:
    """Minimal duck-typed registry for the builder."""

    def __init__(self, run: _Run) -> None:
        self._run = run
        self._sources = {"extract": "def handler(args):\n    return {}\n"}
        self._schemas = {"extract": {"type": "object", "properties": {}}}

    def get_run(self, usecase_id: str, run_id: str) -> _Run:
        return self._run

    def get_active_tools(self, usecase_id: str, run_id: str) -> list[_Tool]:
        return [_Tool(name="extract", active_version=3)]

    def get_handler_source(self, usecase_id: str, run_id: str, name: str) -> str:
        return self._sources[name]

    def get_tool_schema(self, usecase_id: str, run_id: str, name: str) -> dict[str, Any]:
        return self._schemas[name]

    def get_consumer_prompt(self, usecase_id: str) -> str | None:
        return "You are an invoice agent."


def test_build_pipeline_spec_snapshots_active_tools() -> None:
    promoted = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    reg = _FakeRegistry(_Run(forked_from="r_old", promoted_to_production_at=promoted))

    spec = build_pipeline_spec(reg, "uc_1", "r_new", change_reason="judge_instruction_id:42")

    assert spec.run_id == "r_new"
    assert spec.usecase_id == "uc_1"
    assert spec.forked_from == "r_old"
    assert spec.change_reason == "judge_instruction_id:42"
    assert spec.promoted_at == promoted.isoformat()
    assert spec.system_prompt == "You are an invoice agent."

    assert len(spec.tools) == 1
    tool = spec.tools[0]
    assert tool.tool_id == "extract"
    assert tool.tool_version == 3
    assert tool.implementation_hash.startswith("sha256:")
    assert tool.schema == {"type": "object", "properties": {}}


def test_build_pipeline_spec_defaults_promoted_at_when_missing() -> None:
    reg = _FakeRegistry(_Run(forked_from=None, promoted_to_production_at=None))
    spec = build_pipeline_spec(reg, "uc_1", "r_new")
    assert spec.forked_from is None
    assert spec.change_reason is None
    # Falls back to "now" as a valid ISO-8601 timestamp.
    datetime.fromisoformat(spec.promoted_at)
