"""DynamicJudge — aggregate all runs into a global, cross-run assessment.

It takes the whole window (not a single trace): it runs the metric engine over
it, averages the static judge's local notes into global notes, computes the mean
structural stability, and (optionally) asks the shared LLM for a concise global
diagnosis. The deterministic parts need no LLM; the LLM pass is opt-in.
"""
from __future__ import annotations

from datetime import UTC, datetime

from ..metrics.data import MetricWindow, TelemetryReader
from ..metrics.engine import MetricEngine
from ..metrics.env import MetricEnv
from ..static.llm import JudgeLLM
from ..static.store import JudgeStore, ToolNoteRecord
from .aggregate import aggregate_tool_notes, scores_from_global_notes
from .models import DynamicJudgeReport
from .prompt import build_diagnosis_message
from .stability import compute_structural_stability
from .store import DynamicJudgeStore


class DynamicJudge:
    def __init__(
        self,
        *,
        engine: MetricEngine | None = None,
        llm: JudgeLLM | None = None,
    ) -> None:
        self._engine = engine or MetricEngine()
        self._llm = llm

    async def assess(
        self,
        window: MetricWindow,
        note_records: list[ToolNoteRecord],
        *,
        diagnose: bool = True,
    ) -> DynamicJudgeReport:
        """Build the cross-run report from a window + the static notes."""
        env = window.env
        global_notes = aggregate_tool_notes(note_records, env)
        # Feed the judge scores back into the metric window so the judge-scored
        # families (2 & 5) report as evaluated, not pending.
        window.judge_scores, window.judged_tools = scores_from_global_notes(global_notes)
        metric_report = self._engine.compute(window)
        stability = compute_structural_stability(metric_report, env)

        report = DynamicJudgeReport(
            usecase_id=window.usecase_id,
            run_id=window.run_id,
            computed_at=datetime.now(UTC).isoformat(),
            short_window=env.short_window,
            long_window=env.long_window,
            n_tasks=len(window.long),
            metric_report=metric_report,
            tool_global_notes=global_notes,
            structural_stability=stability,
        )

        if diagnose and self._llm is not None:
            report.diagnosis = await self._llm.complete(build_diagnosis_message(report))
        return report

    async def run(
        self,
        reader: TelemetryReader,
        notes_store: JudgeStore,
        usecase_id: str,
        env: MetricEnv | None = None,
        *,
        run_id: str | None = None,
        run_ids: list[str] | None = None,
        store: DynamicJudgeStore | None = None,
        diagnose: bool = True,
    ) -> DynamicJudgeReport:
        """Load a window from Postgres, assess it, and optionally persist.

        ``run_ids`` aggregates a chosen set of pipeline versions into one
        cross-run report; ``run_id`` restricts to a single version.
        """
        env = env or MetricEnv()
        window = reader.load_window(usecase_id, env, run_id=run_id, run_ids=run_ids)
        notes = self._load_notes(notes_store, usecase_id, run_id, run_ids)
        report = await self.assess(window, notes, diagnose=diagnose)
        if store is not None:
            store.save_report(report)
        return report

    @staticmethod
    def _load_notes(
        notes_store: JudgeStore,
        usecase_id: str,
        run_id: str | None,
        run_ids: list[str] | None,
    ) -> list[ToolNoteRecord]:
        """Static notes for the selection (one run, a set of runs, or all)."""
        if run_ids:
            notes: list[ToolNoteRecord] = []
            for rid in run_ids:
                notes.extend(notes_store.load_tool_notes(usecase_id, run_id=rid))
            return notes
        return notes_store.load_tool_notes(usecase_id, run_id=run_id)
