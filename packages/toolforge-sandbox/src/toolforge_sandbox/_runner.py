"""Exposes RUNNER_SCRIPT — the content of runner.py as a string.

runner.py is the canonical source. This module reads it so that sandbox.py
and tests can reference RUNNER_SCRIPT without duplication, and so that
Dockerfile.sandbox can COPY the same file directly into the image.
"""
from pathlib import Path

RUNNER_SCRIPT: str = (Path(__file__).parent / "runner.py").read_text(encoding="utf-8")
