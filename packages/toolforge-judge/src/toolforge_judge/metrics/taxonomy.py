"""Rule-based error classifier (MAST-inspired).

The Judge later refines this with the LLM, but a deterministic first pass on the
``retries[].error`` strings is enough to drive ``error_taxonomy_distribution``
and ``timeout_rate`` without any model call. Keeping it rule-based here means
the taxonomy is reproducible and cheap.
"""
from __future__ import annotations

import re
from typing import Final

# Error classes — the diagnosis a high concentration points the Judge toward.
SCHEMA_INVALID: Final = "schema_invalid"           # arg failed schema validation
HALLUCINATED_PARAM: Final = "hallucinated_param"   # param/key that doesn't exist
MISSING_PRECONDITION: Final = "missing_precondition"  # needed state/file absent
TIMEOUT: Final = "timeout"                          # exceeded the time budget
RUNTIME_ERROR: Final = "runtime_error"             # handler raised mid-execution
UNKNOWN: Final = "unknown"

ALL_CLASSES: Final = (
    SCHEMA_INVALID,
    HALLUCINATED_PARAM,
    MISSING_PRECONDITION,
    TIMEOUT,
    RUNTIME_ERROR,
    UNKNOWN,
)

# Ordered: first match wins, most specific first.
_RULES: Final[list[tuple[str, re.Pattern[str]]]] = [
    (TIMEOUT, re.compile(r"\b(timed?\s?out|timeout|deadline exceeded)\b", re.I)),
    (
        SCHEMA_INVALID,
        re.compile(
            r"(validation error|schema|invalid (type|argument|value)|"
            r"required (property|field)|is not of type|additionalproperties|"
            r"failed to parse|json ?decode)",
            re.I,
        ),
    ),
    (
        HALLUCINATED_PARAM,
        re.compile(
            r"(unexpected keyword argument|got an unexpected|no such (parameter|argument)|"
            r"unknown (field|parameter|argument)|has no attribute|keyerror)",
            re.I,
        ),
    ),
    (
        MISSING_PRECONDITION,
        re.compile(
            r"(no such file|file ?not ?found|does not exist|not found|"
            r"connection refused|missing (input|dependency|precondition)|"
            r"permission denied|not initialized)",
            re.I,
        ),
    ),
]


def classify_error(message: str | None) -> str:
    """Map a raw error string to one of :data:`ALL_CLASSES`."""
    if not message:
        return UNKNOWN
    for label, pattern in _RULES:
        if pattern.search(message):
            return label
    # A bare exception name with a traceback is a runtime error.
    if re.search(r"\b(error|exception|traceback)\b", message, re.I):
        return RUNTIME_ERROR
    return UNKNOWN
