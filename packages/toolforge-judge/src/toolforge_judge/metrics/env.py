"""metric_env — every tunable input for the metric layer in one place.

The thresholds shipped here are *examples* (per the spec). Tune them by:

1. editing the defaults below, or
2. passing overrides at runtime::

       env = MetricEnv.with_overrides({"error_rate_abs": 0.08, "short_window": 20})

3. loading from a ``[judge.metrics]`` table in ``toolforge.toml``::

       env = MetricEnv.from_toml_table(cfg_dict)

Every metric reads its knobs from this object — no magic numbers live in the
metric implementations themselves.

Threshold semantics
-------------------
SPC principle: a breach fires either on an **absolute** level on the short
window, or on a **divergence** between the short and long windows. Each metric
documents which it uses. A breach is a *trigger to diagnose traces*, never a
standalone alert — the Judges consume it as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MetricEnv:
    # -- windows ------------------------------------------------------------
    # Short window: the last N tasks (fast SPC signal).
    short_window: int = 15
    # Long window: baseline since the current pipeline version, capped to this
    # many tasks when sliding. ``None`` short window entries are never counted.
    long_window: int = 100
    # Minimum sample size before a metric is allowed to breach a threshold.
    # Below this, the metric is reported but ``breached`` is forced False to
    # avoid chasing noise on tiny samples.
    min_samples: int = 8
    # Divergence factor: short-window value must exceed long-window value by
    # this relative amount (e.g. 1.5 = +50%) to count as a divergence breach.
    divergence_factor: float = 1.5

    # -- family 1: execution reliability -----------------------------------
    error_rate_abs: float = 0.10            # > 10% errors after retries
    retry_rate_abs: float = 0.30            # > 30% calls needing a retry
    # error_taxonomy concentration: if one error class is >= this share of all
    # errors for a tool, the diagnosis is treated as near-automatic.
    taxonomy_concentration: float = 0.80
    timeout_rate_abs: float = 0.05          # > 5% timeouts
    # latency_p95 drift: short p95 / long p95 above this factor = breach.
    latency_p95_drift_factor: float = 1.75

    # -- family 2: invocation relevance (judge-scored) ---------------------
    selection_precision_min: float = 0.85
    selection_recall_min: float = 0.80
    param_extraction_min: float = 0.90

    # -- family 3: contribution (backward pass) ----------------------------
    dead_call_rate_abs: float = 0.20        # > 20% dead calls
    redundancy_rate_abs: float = 0.20
    # A tool whose criticality stays below this over the long window is a
    # candidate for removal (pipeline simplification).
    criticality_floor: float = 0.05

    # -- family 4: structural stability ------------------------------------
    # position_variance only matters paired with output divergence; this is the
    # variance above which a tool's DAG position is considered "high".
    position_variance_high: float = 1.0
    order_sensitivity_abs: float = 0.0      # > 0 on a deterministic tool = breach
    premature_call_rate_abs: float = 0.15

    # -- family 5: output quality (contract / per-node judge) --------------
    node_assertion_pass_min: float = 0.95
    downstream_correction_rate_abs: float = 0.25
    output_schema_drift_abs: float = 0.05

    # -- family 6: loop / meta indicators ----------------------------------
    # version_delta: a degradation worse than this (absolute) flags a regression.
    version_delta_regression: float = 0.05
    # dag_diversity collapse: short entropy / long entropy below this = collapse.
    dag_diversity_collapse_ratio: float = 0.6
    judge_agreement_min: float = 0.80       # recalibrate the judge below this
    # oscillation: instruction direction flips across this many consecutive
    # versions on the same tool -> freeze + human escalation.
    oscillation_versions: int = 3

    # -- dynamic judge: structural stability -------------------------------
    # Pipeline mean structural-stability score (0..1) below which the dynamic
    # judge flags the pipeline as structurally unstable.
    structural_stability_min: float = 0.70

    # ------------------------------------------------------------------
    @classmethod
    def with_overrides(cls, overrides: dict[str, Any]) -> MetricEnv:
        """Return a copy with the given fields overridden.

        Unknown keys raise ``KeyError`` so typos in a config don't silently
        leave a threshold at its default.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"Unknown MetricEnv override(s): {sorted(unknown)}")
        base = {f.name: getattr(cls(), f.name) for f in fields(cls)}
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_toml_table(cls, table: dict[str, Any] | None) -> MetricEnv:
        """Build from a ``[judge.metrics]`` TOML table (or defaults if None)."""
        if not table:
            return cls()
        return cls.with_overrides(dict(table))
