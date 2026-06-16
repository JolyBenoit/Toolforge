"""CLI tests — registry commands via CliRunner; no LLM or Docker required."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from toolforge_cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def invoke(runner: CliRunner, data_root: Path, *args: str):
    return runner.invoke(cli, ["--data-root", str(data_root), *args])


# --- usecase create ---


def test_usecase_create(runner: CliRunner, data_root: Path) -> None:
    result = invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "Test prompt")
    assert result.exit_code == 0
    assert "uc1" in result.output


def test_usecase_create_prompt_file(runner: CliRunner, data_root: Path, tmp_path: Path) -> None:
    pf = tmp_path / "prompt.md"
    pf.write_text("File prompt.", encoding="utf-8")
    result = invoke(runner, data_root, "usecase", "create", "--id", "uc2", "--prompt-file", str(pf))
    assert result.exit_code == 0


def test_usecase_create_no_prompt_errors(runner: CliRunner, data_root: Path) -> None:
    result = invoke(runner, data_root, "usecase", "create", "--id", "uc3")
    assert result.exit_code != 0


def test_usecase_create_duplicate_errors(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    result = invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    assert result.exit_code != 0


# --- usecase list ---


def test_usecase_list_empty(runner: CliRunner, data_root: Path) -> None:
    result = invoke(runner, data_root, "usecase", "list")
    assert result.exit_code == 0
    assert "No use cases" in result.output


def test_usecase_list_shows_created(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    result = invoke(runner, data_root, "usecase", "list")
    assert result.exit_code == 0
    assert "uc1" in result.output


# --- run create ---


def test_run_create_prints_run_id(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    result = invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    assert result.exit_code == 0
    assert result.output.strip().startswith("r_")


def test_run_create_unknown_usecase_errors(runner: CliRunner, data_root: Path) -> None:
    result = invoke(runner, data_root, "run", "create", "--usecase", "ghost")
    assert result.exit_code != 0


# --- run list ---


def test_run_list_empty(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    result = invoke(runner, data_root, "run", "list", "--usecase", "uc1")
    assert result.exit_code == 0
    assert "No runs" in result.output


def test_run_list_shows_run(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    result = invoke(runner, data_root, "run", "list", "--usecase", "uc1")
    assert result.exit_code == 0
    assert "draft" in result.output


# --- run validate ---


def test_run_validate(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    run_result = invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    run_id = run_result.output.strip()
    result = invoke(runner, data_root, "run", "validate", "--usecase", "uc1", "--run", run_id)
    assert result.exit_code == 0
    assert "validated" in result.output


# --- run fork ---


def test_run_fork(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    run_result = invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    run_id = run_result.output.strip()
    invoke(runner, data_root, "run", "validate", "--usecase", "uc1", "--run", run_id)
    result = invoke(runner, data_root, "run", "fork", "--usecase", "uc1", "--from", run_id)
    assert result.exit_code == 0
    new_id = result.output.strip()
    assert new_id.startswith("r_")
    assert new_id != run_id


def test_run_fork_unvalidated_errors(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    run_result = invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    run_id = run_result.output.strip()
    result = invoke(runner, data_root, "run", "fork", "--usecase", "uc1", "--from", run_id)
    assert result.exit_code != 0


# --- run tools ---


def test_run_tools_empty(runner: CliRunner, data_root: Path) -> None:
    invoke(runner, data_root, "usecase", "create", "--id", "uc1", "--prompt", "p")
    run_result = invoke(runner, data_root, "run", "create", "--usecase", "uc1")
    run_id = run_result.output.strip()
    result = invoke(runner, data_root, "run", "tools", "--usecase", "uc1", "--run", run_id)
    assert result.exit_code == 0
    assert "No tools" in result.output


# --- creator run (no config) ---


def test_creator_run_missing_config(runner: CliRunner, data_root: Path, tmp_path: Path) -> None:
    result = invoke(
        runner, data_root,
        "creator", "run",
        "--usecase", "uc1",
        "--run", "r_20260524_abc",
        "--config", str(tmp_path / "nonexistent.toml"),
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "not found" in (result.exception or "")


# --- consumer run (no config) ---


def test_consumer_run_missing_config(runner: CliRunner, data_root: Path, tmp_path: Path) -> None:
    result = invoke(
        runner, data_root,
        "consumer", "run",
        "--usecase", "uc1",
        "--run", "r_20260524_abc",
        "--task", "do something",
        "--config", str(tmp_path / "nonexistent.toml"),
    )
    assert result.exit_code != 0
