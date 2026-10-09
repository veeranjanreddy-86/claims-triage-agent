from __future__ import annotations

import json
from pathlib import Path

from claims_agent import tools as tools_mod
from claims_agent.agent import TriageAgent
from claims_agent.config import Settings
from claims_agent.schemas import LLMResponse, ToolCall
from helpers import BadArgsOnceProvider, LoopingProvider, ScriptedResponses


def _events(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [e.type for e in result.guardrail_events]


def test_success_path_appeal_with_cited_evidence(agent: TriageAgent) -> None:
    r = agent.run("CLM-1004", "Payer says no auth. Next step?")
    assert r.status == "completed"
    assert r.action == "file_appeal"
    assert r.tool_calls == [
        "lookup_claim",
        "get_denial_reason_policy",
        "check_prior_auth_status",
        "draft_appeal_summary",
    ]
    sources = {e.source for e in r.decision.evidence}
    assert {"lookup_claim", "check_prior_auth_status", "draft_appeal_summary"} <= sources
    assert r.steps == 5


def test_every_step_is_traced(agent: TriageAgent) -> None:
    r = agent.run("CLM-1007", "Worth appealing?")
    lines = [json.loads(x) for x in Path(r.trace_file).read_text().splitlines()]
    kinds = [x["event"] for x in lines]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("llm_call") == r.steps
    tool_events = [x for x in lines if x["event"] == "tool_call"]
    assert [x["tool"] for x in tool_events] == r.tool_calls
    assert all("latency_ms" in x and "args" in x for x in tool_events)


def test_max_steps_stops_the_loop(settings: Settings) -> None:
    s = Settings(
        db_path=settings.db_path,
        trace_dir=None,
        max_steps=3,
        max_tool_calls=50,
        max_calls_per_tool=50,
    )
    provider = LoopingProvider()
    r = TriageAgent(s, provider=provider).run("CLM-1001", "loop forever")
    assert r.status == "max_steps_exceeded"
    assert r.steps == 3 and provider.n == 3 and r.decision is None


def test_tool_budget_guardrail(settings: Settings) -> None:
    s = Settings(db_path=settings.db_path, trace_dir=None, max_steps=6, max_calls_per_tool=2)
    r = TriageAgent(s, provider=LoopingProvider()).run("CLM-1001", "loop")
    assert r.status == "max_steps_exceeded"
    assert _events(r).count("tool_budget") == 4


def test_invalid_tool_args_are_reported_and_recovered(settings: Settings) -> None:
    r = TriageAgent(settings, provider=BadArgsOnceProvider()).run("CLM-1001", "Triage this.")
    assert "invalid_arguments" in _events(r)
    assert r.status == "completed" and r.action == "resubmit_corrected_claim"
    assert r.tool_calls[:2] == ["lookup_claim", "lookup_claim"]


def test_unparseable_json_args_are_handled(settings: Settings) -> None:
    provider = BadArgsOnceProvider(arguments="{not json")
    r = TriageAgent(settings, provider=provider).run("CLM-1002", "Triage this.")
    assert "invalid_arguments" in _events(r) and r.status == "completed"


def test_transient_tool_errors_are_retried(settings: Settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    original = tools_mod.ToolContext.query
    calls = {"n": 0}

    def flaky(self, sql, params):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise tools_mod.TransientToolError("database is locked")
        return original(self, sql, params)

    monkeypatch.setattr(tools_mod.ToolContext, "query", flaky)
    r = TriageAgent(settings).run("CLM-1011", "Why denied?")
    assert r.status == "completed" and r.action == "write_off_review"
    first = next(
        json.loads(x) for x in Path(r.trace_file).read_text().splitlines() if '"tool_call"' in x
    )
    assert first["attempts"] == 2


def test_allowlist_blocks_tools(settings: Settings) -> None:
    rogue = ScriptedResponses(
        [
            LLMResponse(tool_calls=[ToolCall(id="x", name="draft_appeal_summary", arguments="{}")]),
            LLMResponse(content="not json"),
            LLMResponse(content="still not json"),
        ]
    )
    agent = TriageAgent(
        settings, provider=rogue, allowed_tools=["lookup_claim", "get_denial_reason_policy"]
    )
    r = agent.run("CLM-1004", "Next step?")
    assert "tool_not_allowed" in _events(r)
    assert r.status == "failed"  # invalid final answer even after one repair prompt


def test_planner_escalates_when_required_tool_not_allowed(settings: Settings) -> None:
    agent = TriageAgent(settings, allowed_tools=["lookup_claim", "get_denial_reason_policy"])
    r = agent.run("CLM-1004", "Next step?")
    assert r.status == "completed" and r.action == "escalate_to_human"


def test_unknown_claim_escalates(agent: TriageAgent) -> None:
    r = agent.run("CLM-9999", "What happened?")
    assert r.action == "escalate_to_human" and r.tool_calls == ["lookup_claim"]


def test_final_answer_repair(settings: Settings) -> None:
    good = (
        '{"action": "write_off_review", "rationale": "Duplicate of a paid claim.", '
        '"evidence": [{"source": "lookup_claim", "detail": "CO-18"}]}'
    )
    provider = ScriptedResponses(
        [LLMResponse(content="I think write it off"), LLMResponse(content=f"```json\n{good}\n```")]
    )
    r = TriageAgent(settings, provider=provider).run("CLM-1011", "Why?")
    assert r.status == "completed" and r.action == "write_off_review" and r.steps == 2
