"""Sandbox: isolated tool-handler execution.

Two execution modes, selected by ``Sandbox.mode``:

uv mode (default)
  The runner script is written once to a temp file (avoids the ``python -c``
  pattern that EDR solutions like CrowdStrike flag as fileless execution).
  Dependencies are installed by uv into an ephemeral cached env.
  No filesystem or network isolation — suitable for development.

docker mode
  The runner is baked into the image at ``/app/runner.py``.
  Only /inputs (read-only) and /outputs (read-write) are mounted.
  Network access is allowed (LLM API calls work); host filesystem is isolated.
  Build the image first: ``toolforge sandbox build``
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from ._runner import RUNNER_SCRIPT
from .models import SandboxResult

# ---------------------------------------------------------------------------
# Temp-file runner (uv mode) — written once, reused across calls
# ---------------------------------------------------------------------------

_runner_tmp: str | None = None


def _get_runner_tmp() -> str:
    global _runner_tmp
    if _runner_tmp is None:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="tf_runner_", encoding="utf-8"
        )
        f.write(RUNNER_SCRIPT)
        f.close()
        _runner_tmp = f.name
        atexit.register(_delete_runner_tmp)
    return _runner_tmp


def _delete_runner_tmp() -> None:
    global _runner_tmp
    if _runner_tmp and os.path.exists(_runner_tmp):
        try:
            os.unlink(_runner_tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def _build_uv_cmd(requirements: list[str]) -> list[str]:
    """Build a command that runs the temp-file runner via uv in an isolated env.

    --isolated ensures the ToolForge workspace packages are never leaked into
    the handler env, regardless of whether the tool declares requirements.
    """
    path = _get_runner_tmp()
    cmd = ["uv", "run", "--isolated"]
    for req in requirements:
        cmd += ["--with", req]
    return cmd + ["python", path]


def _build_docker_cmd(
    image: str,
    requirements: list[str],
    inputs_dir: Path | None,
    outputs_dir: Path | None,
) -> list[str]:
    """Build a ``docker run`` command with volume mounts and optional uv deps."""
    cmd = ["docker", "run", "--rm", "--network=bridge", "-i"]
    if inputs_dir and inputs_dir.exists():
        cmd += ["-v", f"{inputs_dir.resolve()}:/inputs:ro"]
    if outputs_dir:
        cmd += ["-v", f"{outputs_dir.resolve()}:/outputs:rw"]
    cmd.append(image)
    if requirements:
        cmd += ["uv", "run"]
        for req in requirements:
            cmd += ["--with", req]
        cmd += ["python", "/app/runner.py"]
    else:
        cmd += ["python", "/app/runner.py"]
    return cmd


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    def __init__(
        self,
        *,
        timeout_seconds: int = 30,
        mode: Literal["uv", "docker"] = "uv",
        image: str = "toolforge-sandbox:latest",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.mode = mode
        self.image = image

    @classmethod
    def from_config(cls, config: Any) -> "Sandbox":
        return cls(
            timeout_seconds=config.timeout_seconds,
            mode=getattr(config, "mode", "uv"),
            image=getattr(config, "image", "toolforge-sandbox:latest"),
        )

    async def run(
        self,
        handler_source: str,
        args: dict[str, Any],
        *,
        requirements: list[str] | None = None,
        llm_configs: dict[str, Any] | None = None,
        inputs_dir: Path | None = None,
        outputs_dir: Path | None = None,
        mode: Literal["validation", "runtime"] = "runtime",
    ) -> SandboxResult:
        """Execute ``handler_source`` in an isolated subprocess.

        validation mode — 2× timeout (Creator testing).
        runtime mode    — configured timeout (Consumer execution).
        inputs_dir      — host path mounted as /inputs:ro  (Docker mode only).
        outputs_dir     — host path mounted as /outputs:rw (Docker mode only).
        """
        timeout = self.timeout_seconds * (2.0 if mode == "validation" else 1.0)
        reqs = requirements or []

        if outputs_dir:
            outputs_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "docker":
            cmd = _build_docker_cmd(self.image, reqs, inputs_dir, outputs_dir)
        else:
            cmd = _build_uv_cmd(reqs)

        # In Docker mode the handler runs inside the container, so inject the
        # container-side mount paths (/inputs, /outputs) rather than the host paths
        # that are only reachable from outside the container.
        if self.mode == "docker":
            payload_inputs_dir = "/inputs" if inputs_dir and inputs_dir.exists() else None
            payload_outputs_dir = "/outputs" if outputs_dir else None
        else:
            payload_inputs_dir = str(inputs_dir.resolve()) if inputs_dir else None
            payload_outputs_dir = str(outputs_dir.resolve()) if outputs_dir else None

        payload = json.dumps({
            "handler_source": handler_source,
            "args": args,
            "llm_configs": llm_configs,
            "inputs_dir": payload_inputs_dir,
            "outputs_dir": payload_outputs_dir,
        })

        t0 = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(payload.encode()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
            return SandboxResult(
                output=None,
                stdout="",
                stderr=f"Sandbox timed out after {timeout:.0f}s",
                duration_ms=(time.monotonic() - t0) * 1000,
                exit_code=-1,
            )

        duration_ms = (time.monotonic() - t0) * 1000
        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else 0
        output, stderr, nested_llm_calls = _parse_runner_output(stdout, stderr)
        return SandboxResult(
            output=output,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            exit_code=exit_code,
            nested_llm_calls=nested_llm_calls,
        )


def _parse_runner_output(stdout: str, stderr: str) -> tuple[Any, str, list]:
    """Extract the JSON result from the last stdout line; fold error traceback into stderr."""
    last_line = stdout.strip().rsplit("\n", 1)[-1] if stdout.strip() else ""
    if not last_line:
        return None, stderr, []
    try:
        parsed = json.loads(last_line)
    except json.JSONDecodeError:
        return None, stderr, []

    output = parsed.get("output")
    nested_llm_calls: list = parsed.get("nested_llm_calls") or []
    err_info = parsed.get("error")
    if err_info:
        tb = err_info.get("traceback", "")
        if tb:
            stderr = tb + ("\n" + stderr if stderr else "")
    return output, stderr, nested_llm_calls
