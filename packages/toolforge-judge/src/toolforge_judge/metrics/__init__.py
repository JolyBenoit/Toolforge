"""toolforge_judge.metrics — deterministic per-tool / per-pipeline indicators.

Six families (spec order):

1. reliability    — error_rate, retry_rate, error_taxonomy_distribution,
                    timeout_rate, latency_p95                     (telemetry)
2. invocation     — selection_precision, selection_recall,
                    param_extraction_accuracy                    (judge-scored)
3. contribution   — dead_call_rate, redundancy_rate, criticality_score
4. structural     — position_variance, premature_call_rate, order_sensitivity
5. output_quality — node_assertion_pass_rate, downstream_correction_rate,
                    output_schema_drift                          (judge-scored)
6. meta           — dag_diversity, version_delta, judge_agreement,
                    oscillation_detector

Every tunable input lives in :class:`MetricEnv` (``env.py``). The engine runs
the full catalogue over a :class:`MetricWindow` and returns a serialisable
:class:`MetricReport`.
"""
from __future__ import annotations

from .base import (
    INSUFFICIENT_DATA,
    NOT_COMPUTABLE,
    OK,
    REQUIRES_JUDGE,
    Metric,
    MetricValue,
)
from .data import (
    MetricWindow,
    SpanRecord,
    TaskRecord,
    TelemetryReader,
)
from .engine import (
    MetricDelta,
    MetricEngine,
    MetricReport,
    default_catalogue,
    version_delta,
)
from .env import MetricEnv

__all__ = [
    "MetricEnv",
    "MetricWindow",
    "TaskRecord",
    "SpanRecord",
    "TelemetryReader",
    "Metric",
    "MetricValue",
    "MetricEngine",
    "MetricReport",
    "MetricDelta",
    "default_catalogue",
    "version_delta",
    "OK",
    "REQUIRES_JUDGE",
    "INSUFFICIENT_DATA",
    "NOT_COMPUTABLE",
]
