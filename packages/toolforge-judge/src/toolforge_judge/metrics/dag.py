"""DAG helpers shared by the contribution and structural-stability families.

Operates on the ``dag`` JSON stored on each task (``{"nodes": [...],
"edges": [{"from_span", "to_span", "via"}]}``) plus the task's spans to map
span_ids onto tool_ids.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .data import TaskRecord


def span_to_tool(task: TaskRecord) -> dict[str, str]:
    """Map each tool_call span_id to its tool_id."""
    return {s.span_id: s.tool_id for s in task.tool_calls if s.tool_id is not None}


def _adjacency(task: TaskRecord) -> tuple[dict[str, list[str]], list[str]]:
    """Return (successors, nodes) from the task DAG; empty if no DAG present."""
    dag = task.dag or {}
    nodes = list(dag.get("nodes") or [])
    succ: dict[str, list[str]] = defaultdict(list)
    for edge in dag.get("edges") or []:
        succ[edge["from_span"]].append(edge["to_span"])
    return succ, nodes


def depths(task: TaskRecord) -> dict[str, int]:
    """Longest-path depth (0-based) of every node, via topological relaxation.

    Falls back to ordering by (llm_turn_index, call_index_in_turn) when the task
    has no DAG recorded, so position metrics still degrade gracefully.
    """
    succ, nodes = _adjacency(task)
    if not nodes:
        return _ordinal_depths(task)

    indeg: dict[str, int] = {n: 0 for n in nodes}
    for _src, outs in succ.items():
        for dst in outs:
            if dst in indeg:
                indeg[dst] += 1
    # Kahn topological order.
    queue = [n for n in nodes if indeg[n] == 0]
    depth: dict[str, int] = {n: 0 for n in nodes}
    order: list[str] = []
    indeg_work = dict(indeg)
    while queue:
        n = queue.pop()
        order.append(n)
        for dst in succ.get(n, []):
            if dst not in depth:
                continue
            if depth[n] + 1 > depth[dst]:
                depth[dst] = depth[n] + 1
            indeg_work[dst] -= 1
            if indeg_work[dst] == 0:
                queue.append(dst)
    if len(order) != len(nodes):  # cycle / malformed DAG -> ordinal fallback
        return _ordinal_depths(task)
    return depth


def _ordinal_depths(task: TaskRecord) -> dict[str, int]:
    ordered = sorted(
        task.spans,
        key=lambda s: (
            s.llm_turn_index if s.llm_turn_index is not None else 0,
            s.call_index_in_turn if s.call_index_in_turn is not None else 0,
            s.started_at,
        ),
    )
    return {s.span_id: i for i, s in enumerate(ordered)}


def critical_path_nodes(task: TaskRecord) -> set[str]:
    """Node ids lying on a longest path (by node count) through the DAG.

    Used by ``criticality_score``: a tool on the critical path carries weight;
    one almost never on it is a candidate for removal.
    """
    succ, nodes = _adjacency(task)
    if not nodes:
        return set()
    depth = depths(task)
    if not depth:
        return set()
    max_depth = max(depth.values())
    # Walk back from the deepest leaves along edges that increase depth by 1.
    on_path: set[str] = set()
    pred: dict[str, list[str]] = defaultdict(list)
    for src, outs in succ.items():
        for dst in outs:
            pred[dst].append(src)
    frontier = [n for n, d in depth.items() if d == max_depth]
    seen: set[str] = set()
    while frontier:
        n = frontier.pop()
        if n in seen:
            continue
        seen.add(n)
        on_path.add(n)
        for p in pred.get(n, []):
            if depth.get(p, -1) == depth[n] - 1:
                frontier.append(p)
    return on_path


def canonical_signature(task: TaskRecord) -> tuple[Any, ...]:
    """Tool-level canonical signature of a task's DAG for diversity counting.

    Maps span edges to (from_tool, to_tool, via) and returns a sorted tuple, so
    two tasks with the same tool topology hash identically regardless of span
    ids. Falls back to the sorted multiset of tool_ids when no DAG is present.
    """
    s2t = span_to_tool(task)
    dag = task.dag or {}
    edges = dag.get("edges") or []
    sig_edges = sorted(
        (s2t.get(e["from_span"], e["from_span"]),
         s2t.get(e["to_span"], e["to_span"]),
         e.get("via", "control"))
        for e in edges
        if s2t.get(e["from_span"]) or s2t.get(e["to_span"])
    )
    if sig_edges:
        return tuple(sig_edges)
    return tuple(sorted(s2t.values()))
