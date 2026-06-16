from __future__ import annotations

from typing import Any


def format_args(args: dict[str, Any], max_items: int = 3) -> str:
    items = list(args.items())[:max_items]
    parts: list[str] = []
    for k, v in items:
        r = repr(v)
        parts.append(f"{k}={r if len(r) <= 40 else r[:37] + '...'}")
    suffix = f", +{len(args) - max_items} more" if len(args) > max_items else ""
    return ", ".join(parts) + suffix
