"""Architecture-judge tests over in-memory specs and a fake LLM (no DB, no API)."""
from __future__ import annotations

import json

import pytest
from toolforge_judge.architecture import (
    ArchitectureJudge,
    ArchitectureSpec,
    RichToolSpec,
    build_architecture_spec,
)
from toolforge_judge.architecture.prompt import (
    digest_from_dynamic_report,
    parse_findings,
)

_PDF_SOURCE = '''
def handle(path: str) -> dict:
    text = extract_text(path)
    # only keep the first 500 tokens to stay under the model limit
    tokens = text.split()[:500]
    return {"text": " ".join(tokens)}
'''


class FakeLLM:
    """A scripted JudgeLLM: a responder callable maps a message to a reply."""

    def __init__(self, responder, model="fake-judge"):
        self.model = model
        self._responder = responder
        self.calls: list[str] = []

    async def complete(self, user_message: str) -> str:
        self.calls.append(user_message)
        if callable(self._responder):
            return self._responder(user_message)
        return self._responder


def _spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        usecase_id="uc1",
        run_id="run1",
        utility="Answer questions over the full content of long PDFs.",
        rules="Always ground answers in the document.",
        tools=[
            RichToolSpec(
                tool_id="pdf_extract",
                description="Extract the text of a PDF.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                source=_PDF_SOURCE,
            ),
            RichToolSpec(
                tool_id="summarize",
                description="Summarise text.",
                source="def handle(text: str) -> dict:\n    return {'summary': llm(text)}\n",
            ),
        ],
    )


def _contract_responder(msg: str) -> str:
    """Pass-1: flag the 500-token cap on pdf_extract, clean for summarize."""
    if '"tool_id": "pdf_extract"' in msg:
        return json.dumps({
            "output_contract": "{'text': str} — the extracted text.",
            "limits": ["truncates extracted text to the first 500 tokens"],
            "local_risks": ["lossy: drops the tail of long documents"],
        })
    return json.dumps({
        "output_contract": "{'summary': str}",
        "limits": [],
        "local_risks": [],
    })


def _findings_responder(_msg: str) -> str:
    return json.dumps({
        "findings": [
            {
                "category": "over_simplification",
                "severity": "error",
                "tools_involved": ["pdf_extract"],
                "requirement_threatened": "answering over the FULL document",
                "body": "pdf_extract caps text at 500 tokens; chunk instead.",
                "evidence": "limits: truncates extracted text to the first 500 tokens",
                "proposed_action": "split_tool",
            }
        ]
    })


# --- pass 1 + pass 2 chained ----------------------------------------------


@pytest.mark.asyncio
async def test_assess_design_time_flags_truncation():
    judge = ArchitectureJudge(
        contract_llm=FakeLLM(_contract_responder),
        findings_llm=FakeLLM(_findings_responder),
    )
    report = await judge.assess(_spec())

    assert report.mode == "design_time"
    assert report.judge_model == "fake-judge"
    # pass 1: one contract per tool, the cap captured as a limit
    assert {c.tool_id for c in report.contracts} == {"pdf_extract", "summarize"}
    pdf = report.contract_for("pdf_extract")
    assert pdf is not None and "500 tokens" in pdf.limits[0]
    assert report.contract_for("summarize").limits == []
    # pass 2: the truncation becomes a finding on the right tool
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.category == "over_simplification"
    assert f.proposed_action == "split_tool"
    assert report.problematic_tools == ["pdf_extract"]
    assert f.is_structural  # split_tool


@pytest.mark.asyncio
async def test_assess_reports_progress():
    judge = ArchitectureJudge(
        contract_llm=FakeLLM(_contract_responder),
        findings_llm=FakeLLM(_findings_responder),
    )
    phases: list[tuple[str, int]] = []
    tools: list[tuple[str, int, int]] = []

    await judge.assess(
        _spec(),
        on_phase=lambda name, total: phases.append((name, total)),
        on_tool=lambda tid, done, total: tools.append((tid, done, total)),
    )

    # Both phases fire, in order, with the right totals.
    assert phases == [("contracts", 2), ("findings", 1)]
    # One tick per tool, with a monotonic done count over a stable total.
    assert {t[0] for t in tools} == {"pdf_extract", "summarize"}
    assert [t[1] for t in tools] == [1, 2]
    assert all(t[2] == 2 for t in tools)


@pytest.mark.asyncio
async def test_post_run_mode_injects_telemetry_digest():
    captured: dict[str, str] = {}

    def findings_responder(msg: str) -> str:
        captured["msg"] = msg
        return json.dumps({"findings": []})

    class _SS:
        mean_structural_stability = 0.42

    class _Report:
        structural_stability = _SS()
        metric_report = None
        tool_global_notes = []
        diagnosis = "unstable ordering"

    judge = ArchitectureJudge(
        contract_llm=FakeLLM(_contract_responder),
        findings_llm=FakeLLM(findings_responder),
    )
    report = await judge.assess(_spec(), dynamic_report=_Report())

    assert report.mode == "post_run"
    assert report.findings == []
    # the digest reached pass 2
    assert "telemetry_digest" in captured["msg"]
    assert "unstable ordering" in captured["msg"]


# --- parsing robustness ----------------------------------------------------


def test_parse_findings_drops_invalid_and_defaults_enums():
    raw = "```json\n" + json.dumps({
        "findings": [
            {"category": "bogus", "body": "ignored"},        # bad category
            {"severity": "warning", "body": "no category"},  # missing category
            {
                "category": "coverage_gap",
                "body": "no tool can fetch the source PDF",
                # severity + proposed_action omitted -> defaults
                "tools_involved": [],
            },
        ]
    }) + "\n```"
    findings = parse_findings(raw)
    assert len(findings) == 1
    f = findings[0]
    assert f.category == "coverage_gap"
    assert f.severity == "warning"           # default
    assert f.proposed_action == "none"       # default
    assert f.is_structural                   # empty tools_involved


def test_finding_id_is_stable():
    raw = json.dumps({"findings": [{
        "category": "overkill", "severity": "info",
        "tools_involved": ["b", "a"], "body": "  trim me  ",
    }]})
    id1 = parse_findings(raw)[0].finding_id
    id2 = parse_findings(raw)[0].finding_id
    assert id1 == id2 and len(id1) == 12


def test_digest_from_none_is_empty():
    assert digest_from_dynamic_report(None) == {}


# --- builder (duck-typed registry) ----------------------------------------


def test_build_architecture_spec_pulls_source():
    class _Tool:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    class _UC:
        prompt = "do the thing"

    class _Reg:
        def get_usecase(self, uc):
            return _UC()

        def get_active_tools(self, uc, run):
            return [_Tool("pdf_extract", "Extract a PDF.")]

        def get_tool_schema(self, uc, run, name):
            return {"type": "object"}

        def get_handler_source(self, uc, run, name):
            return _PDF_SOURCE

        def get_tool_requirements(self, uc, run, name):
            return ["pypdf"]

        def get_consumer_prompt(self, uc):
            return "be grounded"

    spec = build_architecture_spec(_Reg(), "uc1", "run1")
    assert spec.utility == "do the thing"
    assert spec.rules == "be grounded"
    assert len(spec.tools) == 1
    assert "500 tokens" in spec.tools[0].source
    assert spec.tools[0].requirements == ["pypdf"]
