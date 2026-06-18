"""Data models for the Architecture judge: inputs, pass-1 contracts, findings.

Two input shapes (``RichToolSpec`` / ``ArchitectureSpec``) and two output shapes:

- :class:`ToolContract` — pass 1: what one tool actually returns + its limits +
  local risks, *derived from the handler source* (the registry stores neither an
  output schema nor declared limits, so the judge infers them).
- :class:`ArchitectureFinding` — pass 2: one evidenced pipeline-level problem and
  its proposed remediation, addressed at the toolset as a whole.

``ArchitectureJudgeReport`` wraps both for one (usecase, run) assessment.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# The taxonomy of pipeline-level problems this judge reports.
FindingCategory = Literal[
    "over_simplification",
    "technical_constraint",
    "overkill",
    "redundant_step",
    "coverage_gap",
    "wiring",
    "ordering",
]

Severity = Literal["info", "warning", "error"]

# Proposed remediation. The values intentionally mirror
# ``creator.models.CreatorAction`` so a finding flows straight into the creator
# judge — but they are declared here, not imported, to keep the dependency
# one-way (architecture → creator, never the reverse).
ProposedAction = Literal[
    "modify_implementation",
    "modify_description",
    "modify_usage",
    "create_tool",
    "remove_tool",
    "merge_tools",
    "split_tool",
    "none",
]

_STRUCTURAL_ACTIONS: set[str] = {
    "create_tool",
    "remove_tool",
    "merge_tools",
    "split_tool",
}


# ---------------------------------------------------------------------------
# Inputs — the pipeline as a designed system (decoupled from the registry)
# ---------------------------------------------------------------------------


@dataclass
class RichToolSpec:
    """One tool with everything the architecture judge needs to read it.

    Unlike the static judge's ``ToolSpec`` this carries the handler ``source``:
    the only place the truncation/cap behaviour the judge hunts for is visible.
    """

    tool_id: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    requirements: list[str] = field(default_factory=list)


@dataclass
class ArchitectureSpec:
    """The whole pipeline handed to the judge: use case + every tool's source."""

    usecase_id: str
    run_id: str
    utility: str
    rules: str
    tools: list[RichToolSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass 1 — per-tool derived contract
# ---------------------------------------------------------------------------


@dataclass
class ToolContract:
    """Pass-1 read of one tool, inferred from its handler source.

    ``output_contract`` is the prose shape/semantics of what it returns;
    ``limits`` are the information-reducing or bounding behaviours found in the
    code (e.g. "truncates to 500 tokens"); ``local_risks`` are behaviours that
    could hurt the pipeline regardless of the use case. Whether a limit/risk
    actually matters is decided in pass 2.
    """

    tool_id: str
    output_contract: str = ""
    limits: list[str] = field(default_factory=list)
    local_risks: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "raw_response"}


# ---------------------------------------------------------------------------
# Pass 2 — pipeline findings
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureFinding:
    """One evidenced pipeline-level problem + its proposed remediation.

    ``finding_id`` is a stable hash of the canonical fields so re-running the
    judge over the same spec yields the same id (idempotent persistence).
    """

    category: FindingCategory
    severity: Severity
    tools_involved: list[str]
    requirement_threatened: str
    body: str
    evidence: str = ""
    proposed_action: ProposedAction = "none"
    finding_id: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = self.compute_id()

    @property
    def is_structural(self) -> bool:
        return self.proposed_action in _STRUCTURAL_ACTIONS or not self.tools_involved

    def compute_id(self) -> str:
        canonical = "|".join(
            [self.category, ",".join(sorted(self.tools_involved)), self.body.strip()]
        )
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ArchitectureJudgeReport:
    """Everything the architecture judge produced for one (usecase, run)."""

    usecase_id: str
    run_id: str | None
    computed_at: str
    judge_model: str = ""
    mode: Literal["design_time", "post_run"] = "design_time"
    contracts: list[ToolContract] = field(default_factory=list)
    findings: list[ArchitectureFinding] = field(default_factory=list)

    @property
    def problematic_tools(self) -> list[str]:
        """Tools named by at least one finding (order-stable) — the extra
        'problematic' signal the creator judge folds in alongside breaches."""
        ordered: list[str] = []
        seen: set[str] = set()
        for f in self.findings:
            for tool_id in f.tools_involved:
                if tool_id and tool_id not in seen:
                    seen.add(tool_id)
                    ordered.append(tool_id)
        return ordered

    @property
    def structural_findings(self) -> list[ArchitectureFinding]:
        return [f for f in self.findings if f.is_structural]

    def contract_for(self, tool_id: str) -> ToolContract | None:
        for c in self.contracts:
            if c.tool_id == tool_id:
                return c
        return None

    def findings_for(self, tool_id: str) -> list[ArchitectureFinding]:
        return [f for f in self.findings if tool_id in f.tools_involved]

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "run_id": self.run_id,
            "computed_at": self.computed_at,
            "judge_model": self.judge_model,
            "mode": self.mode,
            "contracts": [c.to_dict() for c in self.contracts],
            "findings": [f.to_dict() for f in self.findings],
        }
