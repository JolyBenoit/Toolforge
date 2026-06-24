"""ArchitectureJudge — evaluate the toolset as a designed system.

Two stages, each a stateless LLM pass on the shared backend (the same building
blocks as the static/creator judges):

1. :meth:`read_contracts` — one pass per tool, reading its handler source into a
   derived output contract + limits + local risks (pass 1). Tools are read
   concurrently: each is an independent, network-bound completion.
2. :meth:`synthesize_findings` — one holistic pass over the compact contracts
   (and, in post-run mode, a telemetry digest), producing the pipeline findings
   (pass 2). The handler source enters pass 1 only, keeping pass 2 lean.

:meth:`assess` chains both and is side-effect free; persistence lives in the
store, the spec assembly in the runner. The judge feeds the creator judge — it
*describes and proposes*, the creator judge *prescribes*.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

# Progress callbacks (all optional, fired on the caller's event loop):
#   on_phase(name, total) — a stage begins ("contracts" then "findings").
#   on_tool(tool_id, done, total) — one tool's contract finished reading.
PhaseFn = Callable[[str, int], None]
ToolFn = Callable[[str, int, int], None]

from ..static.llm import JudgeLLM
from .models import (
    ArchitectureFinding,
    ArchitectureJudgeReport,
    ArchitectureSpec,
    ToolContract,
)
from .prompt import (
    build_contract_message,
    build_findings_message,
    digest_from_dynamic_report,
    parse_contract,
    parse_findings,
)


class ArchitectureJudge:
    def __init__(
        self,
        *,
        contract_llm: JudgeLLM | None = None,
        findings_llm: JudgeLLM | None = None,
    ) -> None:
        self._contract_llm = contract_llm
        self._findings_llm = findings_llm

    async def read_contracts(
        self,
        spec: ArchitectureSpec,
        *,
        max_concurrency: int = 6,
        on_tool: ToolFn | None = None,
    ) -> list[ToolContract]:
        """Pass 1: derive each tool's contract from its handler source.

        ``on_tool`` (if given) fires as each tool finishes — tools complete in
        whatever order their network call returns, so it is a liveness signal,
        not a strict 1, 2, 3 sequence.
        """
        if not spec.tools:
            return []
        if self._contract_llm is None:
            raise ValueError("read_contracts requires a contract_llm")
        sem = asyncio.Semaphore(max_concurrency)
        total = len(spec.tools)
        done = 0  # safe to mutate without a lock: asyncio is single-threaded

        async def _one(tool: Any) -> ToolContract:
            nonlocal done
            async with sem:
                raw = await self._contract_llm.complete(
                    build_contract_message(spec, tool)
                )
            contract = parse_contract(tool.tool_id, raw)
            done += 1
            if on_tool is not None:
                on_tool(tool.tool_id, done, total)
            return contract

        # Preserve spec order so the report is stable across runs.
        return list(await asyncio.gather(*(_one(t) for t in spec.tools)))

    async def synthesize_findings(
        self,
        spec: ArchitectureSpec,
        contracts: list[ToolContract],
        *,
        dynamic_report: Any = None,
    ) -> list[ArchitectureFinding]:
        """Pass 2: holistic coherence findings over the derived contracts."""
        if self._findings_llm is None:
            raise ValueError("synthesize_findings requires a findings_llm")
        digest = digest_from_dynamic_report(dynamic_report)
        raw = await self._findings_llm.complete(
            build_findings_message(spec, contracts, digest or None)
        )
        return parse_findings(raw)

    async def assess(
        self,
        spec: ArchitectureSpec,
        *,
        dynamic_report: Any = None,
        max_concurrency: int = 6,
        on_phase: PhaseFn | None = None,
        on_tool: ToolFn | None = None,
    ) -> ArchitectureJudgeReport:
        """Chain both passes into a full report (side-effect free).

        ``dynamic_report=None`` is *design-time* mode (the spec alone, no runs);
        passing a dynamic report switches to *post-run* mode and enriches pass 2
        with a telemetry digest.

        ``on_phase`` / ``on_tool`` (optional) report progress so a UI can show
        which tools have been processed without waiting for the full report.
        """
        if on_phase is not None:
            on_phase("contracts", len(spec.tools))
        contracts = await self.read_contracts(
            spec, max_concurrency=max_concurrency, on_tool=on_tool
        )
        if on_phase is not None:
            on_phase("findings", 1)
        findings = await self.synthesize_findings(
            spec, contracts, dynamic_report=dynamic_report
        )
        mode: Literal["design_time", "post_run"] = (
            "post_run" if dynamic_report is not None else "design_time"
        )
        model = getattr(self._findings_llm, "model", "") or getattr(
            self._contract_llm, "model", ""
        )
        return ArchitectureJudgeReport(
            usecase_id=spec.usecase_id,
            run_id=spec.run_id,
            computed_at=datetime.now(UTC).isoformat(),
            judge_model=model,
            mode=mode,
            contracts=contracts,
            findings=findings,
        )
