"""toolforge-judge — metrics and LLM Judges over production telemetry.

The package is split in two layers, built in order:

1. ``toolforge_judge.metrics`` — pure, deterministic computation of per-tool
   and per-pipeline indicators from the Postgres production telemetry. These
   feed the Judges; they have no LLM dependency and are fully testable on
   in-memory records.
2. ``toolforge_judge`` (judges) — the static, dynamic and creator-facing
   Judges, built on top of the metrics. *Added in a second step.*
"""
from __future__ import annotations

__all__: list[str] = []
