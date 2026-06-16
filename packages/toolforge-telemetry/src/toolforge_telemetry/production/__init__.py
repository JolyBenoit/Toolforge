"""Production telemetry — in_production runs only.

Exposes the full three-level schema (PipelineSpec → Task → Span) backed by
PostgreSQL.  Draft and validated runs continue to use the existing
``TelemetryStore`` from the parent package.
"""
from .models import (
    DAG,
    DAGEdge,
    EdgeVia,
    InputTimelineEntry,
    LLMMeta,
    NestedLLMCall,
    PipelineSpec,
    PipelineToolSnapshot,
    Retry,
    Span,
    SpanStatus,
    SpanType,
    TaskCost,
    TaskStatus,
    UserFeedback,
    to_dict,
)
from .builder import build_pipeline_spec
from .store import (
    NullProductionTelemetryStore,
    ProductionTelemetryStore,
    get_production_store,
)

__all__ = [
    # models
    "DAG",
    "DAGEdge",
    "EdgeVia",
    "InputTimelineEntry",
    "LLMMeta",
    "NestedLLMCall",
    "PipelineSpec",
    "PipelineToolSnapshot",
    "Retry",
    "Span",
    "SpanStatus",
    "SpanType",
    "TaskCost",
    "TaskStatus",
    "UserFeedback",
    "to_dict",
    # store
    "ProductionTelemetryStore",
    "NullProductionTelemetryStore",
    "get_production_store",
    # builder
    "build_pipeline_spec",
]
