"""Sandbox model + command-builder unit tests (no Docker required)."""
import json
import sys
from pathlib import Path

import pytest

import asyncio

from toolforge_sandbox import Sandbox, SandboxResult
from toolforge_sandbox.sandbox import (
    _build_docker_cmd,
    _build_uv_cmd,
    _parse_runner_output,
    _venv_python_path,
)


# --- SandboxResult ---


def test_success_true_on_zero_exit() -> None:
    r = SandboxResult(output=42, stdout="", stderr="", duration_ms=1.0, exit_code=0)
    assert r.success is True


def test_success_false_on_nonzero_exit() -> None:
    r = SandboxResult(output=None, stdout="", stderr="err", duration_ms=1.0, exit_code=1)
    assert r.success is False


def test_timeout_result_not_success() -> None:
    r = SandboxResult(output=None, stdout="", stderr="Timed out", duration_ms=30000.0, exit_code=-1)
    assert not r.success


# --- _parse_runner_output ---


def test_parse_success_line() -> None:
    line = json.dumps({"output": 7, "error": None}) + "\n"
    output, stderr, _ = _parse_runner_output(line, "")
    assert output == 7
    assert stderr == ""


def test_parse_error_injects_traceback_into_stderr() -> None:
    line = json.dumps({
        "output": None,
        "error": {"type": "ValueError", "message": "oops", "traceback": "Traceback..."},
    }) + "\n"
    output, stderr, _ = _parse_runner_output(line, "original")
    assert output is None
    assert "Traceback" in stderr
    assert "original" in stderr


def test_parse_ignores_handler_print_lines() -> None:
    stdout = "debug output from handler\n" + json.dumps({"output": 5, "error": None}) + "\n"
    output, stderr, _ = _parse_runner_output(stdout, "")
    assert output == 5


def test_parse_invalid_json_returns_none() -> None:
    output, stderr, _ = _parse_runner_output("not json at all\n", "err")
    assert output is None
    assert stderr == "err"


def test_parse_empty_stdout() -> None:
    output, stderr, _ = _parse_runner_output("", "existing")
    assert output is None
    assert stderr == "existing"


# --- Sandbox construction ---


def test_sandbox_defaults() -> None:
    s = Sandbox()
    assert s.timeout_seconds == 30
    assert s.mode == "uv"
    assert s.image == "toolforge-sandbox:latest"


def test_sandbox_from_config() -> None:
    class _Cfg:
        timeout_seconds = 60
        mode = "docker"
        image = "custom-sandbox:v2"

    s = Sandbox.from_config(_Cfg())
    assert s.timeout_seconds == 60
    assert s.mode == "docker"
    assert s.image == "custom-sandbox:v2"


def test_sandbox_from_config_defaults_mode_and_image() -> None:
    class _Cfg:
        timeout_seconds = 10

    s = Sandbox.from_config(_Cfg())
    assert s.mode == "uv"
    assert s.image == "toolforge-sandbox:latest"


# --- _build_uv_cmd ---


def test_build_uv_cmd_no_requirements_uses_uv_isolated() -> None:
    cmd = _build_uv_cmd([])
    assert cmd[0] == "uv"
    assert "--isolated" in cmd
    assert "-c" not in cmd


def test_build_uv_cmd_uses_temp_file_not_c_flag() -> None:
    cmd = _build_uv_cmd([])
    assert "-c" not in cmd
    assert cmd[-1].endswith(".py")


def test_build_uv_cmd_with_requirements_uses_uv() -> None:
    cmd = _build_uv_cmd(["pandas==2.2.0", "httpx"])
    assert cmd[0] == "uv"
    assert "run" in cmd
    assert "--isolated" in cmd
    assert "--with" in cmd
    assert "pandas==2.2.0" in cmd
    assert "httpx" in cmd
    assert "-c" not in cmd


def test_build_uv_cmd_ends_with_runner_py() -> None:
    cmd = _build_uv_cmd([])
    assert cmd[-1].endswith(".py")
    assert "tf_runner_" in cmd[-1]


def test_build_uv_cmd_with_reqs_ends_with_runner_py() -> None:
    cmd = _build_uv_cmd(["numpy"])
    assert cmd[-1].endswith(".py")


# --- persistent venv ---


def test_build_uv_cmd_with_venv_python_invokes_interpreter_directly() -> None:
    cmd = _build_uv_cmd(["pandas"], venv_python="/run/.venv/bin/python")
    assert cmd[0] == "/run/.venv/bin/python"
    assert cmd[-1].endswith(".py")
    assert "uv" not in cmd
    assert "--with" not in cmd  # deps live in the venv, not passed per call


def test_venv_python_path_is_os_aware() -> None:
    p = _venv_python_path(Path("/run/.venv"))
    assert p.name in ("python", "python.exe")
    assert ".venv" in str(p)


def test_prepare_is_noop_in_docker_mode() -> None:
    s = Sandbox(mode="docker")

    async def go() -> None:
        await s.prepare(["pandas"], Path("/nonexistent/.venv"))

    asyncio.run(go())
    assert s._venv_python is None


def test_prepare_builds_venv_and_run_uses_it(tmp_path) -> None:
    s = Sandbox(timeout_seconds=60, mode="uv")
    venv_dir = tmp_path / ".venv"

    async def go() -> SandboxResult:
        await s.prepare([], venv_dir)
        return await s.run("def run(args):\n    return args['x'] * 2", {"x": 21})

    result = asyncio.run(go())
    assert s._venv_python is not None
    assert _venv_python_path(venv_dir).exists()
    assert result.success
    assert result.output == 42


def test_prepare_reuses_existing_venv(tmp_path) -> None:
    s = Sandbox(mode="uv")
    venv_dir = tmp_path / ".venv"

    async def go() -> None:
        await s.prepare([], venv_dir)
        first_marker = (venv_dir / ".tf_reqs").read_text(encoding="utf-8")
        # Second prepare with identical reqs must short-circuit (no rebuild).
        await s.prepare([], venv_dir)
        assert (venv_dir / ".tf_reqs").read_text(encoding="utf-8") == first_marker

    asyncio.run(go())
    assert s._venv_python is not None


# --- _build_docker_cmd ---


def test_build_docker_cmd_basic(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", [], inputs, outputs)
    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--rm" in cmd
    assert "--network=bridge" in cmd
    assert "-i" in cmd
    assert "toolforge-sandbox:latest" in cmd
    assert "python" in cmd
    assert "/app/runner.py" in cmd


def test_build_docker_cmd_mounts_inputs_readonly(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", [], inputs, outputs)
    vol_str = " ".join(cmd)
    assert "/inputs:ro" in vol_str


def test_build_docker_cmd_mounts_outputs_readwrite(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", [], inputs, outputs)
    vol_str = " ".join(cmd)
    assert "/outputs:rw" in vol_str


def test_build_docker_cmd_skips_missing_inputs(tmp_path) -> None:
    inputs = tmp_path / "inputs"   # does NOT exist
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", [], inputs, outputs)
    assert "/inputs:ro" not in " ".join(cmd)


def test_build_docker_cmd_with_requirements_uses_uv(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", ["pandas", "httpx"], inputs, outputs)
    assert "uv" in cmd
    assert "--with" in cmd
    assert "pandas" in cmd
    assert "httpx" in cmd
    assert "/app/runner.py" in cmd


def test_build_docker_cmd_no_c_flag(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outputs = tmp_path / "outputs"
    cmd = _build_docker_cmd("toolforge-sandbox:latest", [], inputs, outputs)
    assert "-c" not in cmd
