"""Shared data contracts: planner messages, final answer, run result."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claims_agent.guardrails import GuardrailEvent

Action = Literal[
    "resubmit_corrected_claim",
    "file_appeal",
    "request_prior_auth",
    "write_off_review",
    "escalate_to_human",
]
RunStatus = Literal["completed", "refused", "max_steps_exceeded", "failed"]


class ToolCall(BaseModel):
    """A function call exactly as an OpenAI-style model returns it (arguments = JSON text)."""

    id: str
    name: str
    arguments: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(description="Tool name or policy doc id, e.g. 'policy:POL-AUTH-001'.")
    detail: str = Field(max_length=400)


class FinalDecision(BaseModel):
    """What the planner must return as its final message (JSON)."""

    model_config = ConfigDict(extra="forbid")
    action: Action
    rationale: str = Field(min_length=10, max_length=1200)
    evidence: list[Evidence] = Field(min_length=1, max_length=10)
    confidence: Literal["low", "medium", "high"] = "medium"


class TriageRequest(BaseModel):
    claim_id: str = Field(pattern=r"^CLM-\d{4,6}$", examples=["CLM-1004"])
    question: str = Field(
        min_length=3, max_length=1000, examples=["Why was this denied and what should we do next?"]
    )


class TriageResult(BaseModel):
    run_id: str
    claim_id: str
    status: RunStatus
    decision: FinalDecision | None = None
    refusal_reason: str | None = None
    steps: int
    tool_calls: list[str]
    guardrail_events: list[GuardrailEvent]
    latency_ms: float
    trace_file: str | None = None

    @property
    def action(self) -> str | None:
        return self.decision.action if self.decision else None
