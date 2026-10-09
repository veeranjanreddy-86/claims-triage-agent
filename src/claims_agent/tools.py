"""Typed tools exposed to the planner.

Each tool is a plain function ``fn(ctx, args) -> output`` with pydantic input and
output models. The input model's JSON schema is what a function-calling LLM sees.
Tool outputs are *raw* (they may contain PHI); the agent passes them through the
guardrails before anything reaches the model or the trace.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claims_agent.db import connect
from claims_agent.retrieval import PolicyCorpus

CLAIM_ID = Field(pattern=r"^CLM-\d{4,6}$", description="Claim identifier, e.g. CLM-1004.")


# --------------------------------------------------------------------------- errors
class ToolError(Exception):
    """A tool failed in a way the planner should be told about."""

    code = "tool_error"
    retryable = False


class NotFoundError(ToolError):
    code = "not_found"


class TransientToolError(ToolError):
    """A failure worth retrying (lock contention, timeouts, ...)."""

    code = "transient"
    retryable = True


# --------------------------------------------------------------------------- context
@dataclass
class ToolContext:
    db_path: Path
    as_of: date
    corpus: PolicyCorpus = field(default_factory=PolicyCorpus.from_directory)

    def query(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        try:
            conn = connect(self.db_path)
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:  # e.g. "database is locked"
            raise TransientToolError(f"database error: {exc}") from exc


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- lookup_claim
class LookupClaimInput(_Strict):
    claim_id: str = CLAIM_ID


class ClaimRecord(BaseModel):
    claim_id: str
    patient_name: str
    dob: str
    member_id: str
    ssn: str
    phone: str
    payer: str
    rendering_npi: str
    cpt_code: str
    modifier: str
    icd10_code: str
    billed_amount: float
    service_date: str
    submitted_date: str
    denial_code: str
    denial_date: str
    days_since_denial: int
    has_clinical_notes: bool
    timely_filing_proof: bool
    notes: str


def lookup_claim(ctx: ToolContext, args: LookupClaimInput) -> ClaimRecord:
    rows = ctx.query("SELECT * FROM claims WHERE claim_id = ?", (args.claim_id,))
    if not rows:
        raise NotFoundError(f"claim {args.claim_id} not found")
    row = dict(rows[0])
    row["days_since_denial"] = (ctx.as_of - date.fromisoformat(row["denial_date"])).days
    return ClaimRecord(**row)


# --------------------------------------------------------------------------- denial policy
class DenialPolicyInput(_Strict):
    denial_code: str = Field(pattern=r"^[A-Z]{2}-\d{1,4}$", description="CARC group+code.")
    payer: str = Field(min_length=2, max_length=80, description="Payer name from the claim.")


class DenialPolicy(BaseModel):
    denial_code: str
    payer_specific: bool
    category: Literal["coding", "authorization", "medical_necessity", "timely_filing", "duplicate"]
    description: str
    guidance: str
    appeal_window_days: int
    retro_auth_window_days: int
    policy_doc_ids: list[str]


def get_denial_reason_policy(ctx: ToolContext, args: DenialPolicyInput) -> DenialPolicy:
    rows = ctx.query(
        "SELECT * FROM denial_policies WHERE denial_code = ? AND payer IN (?, '*') "
        "ORDER BY payer = '*'",
        (args.denial_code, args.payer),
    )
    if not rows:
        raise NotFoundError(f"no policy for denial code {args.denial_code}")
    row = dict(rows[0])
    return DenialPolicy(
        denial_code=row["denial_code"],
        payer_specific=row["payer"] != "*",
        category=row["category"],
        description=row["description"],
        guidance=row["guidance"],
        appeal_window_days=row["appeal_window_days"],
        retro_auth_window_days=row["retro_auth_window_days"],
        policy_doc_ids=row["policy_doc_ids"].split(","),
    )


# --------------------------------------------------------------------------- prior auth
class PriorAuthInput(_Strict):
    claim_id: str = CLAIM_ID


class PriorAuthStatus(BaseModel):
    claim_id: str
    status: Literal["approved", "pending", "denied", "not_found"]
    auth_number: str | None
    decision_date: str | None
    days_since_service: int
    retro_auth_window_days: int
    retro_auth_eligible: bool


def check_prior_auth_status(ctx: ToolContext, args: PriorAuthInput) -> PriorAuthStatus:
    claims = ctx.query(
        "SELECT payer, service_date, denial_code FROM claims WHERE claim_id = ?", (args.claim_id,)
    )
    if not claims:
        raise NotFoundError(f"claim {args.claim_id} not found")
    claim = claims[0]
    window_rows = ctx.query(
        "SELECT retro_auth_window_days FROM denial_policies WHERE denial_code = 'CO-197' "
        "AND payer IN (?, '*') ORDER BY payer = '*'",
        (claim["payer"],),
    )
    window = window_rows[0][0] if window_rows else 0
    days = (ctx.as_of - date.fromisoformat(claim["service_date"])).days
    auths = ctx.query(
        "SELECT status, auth_number, decision_date FROM prior_auths WHERE claim_id = ?",
        (args.claim_id,),
    )
    status, number, decided = (
        (auths[0]["status"], auths[0]["auth_number"], auths[0]["decision_date"])
        if auths
        else ("not_found", None, None)
    )
    return PriorAuthStatus(
        claim_id=args.claim_id,
        status=status,
        auth_number=number,
        decision_date=decided,
        days_since_service=days,
        retro_auth_window_days=window,
        retro_auth_eligible=status == "not_found" and days <= window,
    )


# --------------------------------------------------------------------------- policy search
class SearchPolicyInput(_Strict):
    query: str = Field(min_length=3, max_length=300, description="Keywords to search for.")
    payer: str | None = Field(default=None, description="Restrict to general + payer docs.")
    top_k: int = Field(default=3, ge=1, le=5)


class PolicyHit(BaseModel):
    doc_id: str
    title: str
    payer: str
    section: str
    text: str
    score: float


class SearchPolicyOutput(BaseModel):
    hits: list[PolicyHit]


def search_policy_docs(ctx: ToolContext, args: SearchPolicyInput) -> SearchPolicyOutput:
    results = ctx.corpus.search(args.query, top_k=args.top_k, payer=args.payer)
    return SearchPolicyOutput(
        hits=[
            PolicyHit(
                doc_id=c.doc_id,
                title=c.title,
                payer=c.payer,
                section=c.section,
                text=c.text,
                score=s,
            )
            for c, s in results
        ]
    )


# --------------------------------------------------------------------------- appeal draft
AppealBasis = Literal[
    "authorization_on_file",
    "authorization_denied_clinical",
    "medical_necessity",
    "timely_filing_proof",
]

_BASIS_TEXT: dict[str, str] = {
    "authorization_on_file": "a valid prior authorization was approved before the date of "
    "service but was not transmitted on the claim",
    "authorization_denied_clinical": "clinical documentation supports medical necessity for "
    "the service despite the authorization denial",
    "medical_necessity": "the clinical record documents failed conservative treatment and "
    "meets the payer's coverage criteria",
    "timely_filing_proof": "the original claim was received within the filing limit, as shown "
    "by the attached acceptance report",
}


class DraftAppealInput(_Strict):
    claim_id: str = CLAIM_ID
    denial_code: str = Field(pattern=r"^[A-Z]{2}-\d{1,4}$")
    appeal_basis: AppealBasis
    supporting_doc_ids: list[str] = Field(default_factory=list, max_length=5)


class AppealDraft(BaseModel):
    claim_id: str
    summary: str
    attachments_checklist: list[str]
    cited_doc_ids: list[str]


def draft_appeal_summary(ctx: ToolContext, args: DraftAppealInput) -> AppealDraft:
    rows = ctx.query(
        "SELECT cpt_code, service_date, payer FROM claims WHERE claim_id = ?", (args.claim_id,)
    )
    if not rows:
        raise NotFoundError(f"claim {args.claim_id} not found")
    r = rows[0]
    summary = (
        f"Request reconsideration of {args.claim_id} (CPT {r['cpt_code']}, DOS "
        f"{r['service_date']}, {r['payer']}) denied with {args.denial_code}: "
        f"{_BASIS_TEXT[args.appeal_basis]}."
    )
    checklist = {
        "authorization_on_file": ["Authorization approval letter", "Original claim copy"],
        "authorization_denied_clinical": [
            "Clinical notes",
            "Imaging/test results",
            "Letter of medical necessity",
        ],
        "medical_necessity": [
            "Clinical notes",
            "Conservative-care history",
            "Letter of medical necessity",
        ],
        "timely_filing_proof": ["Clearinghouse acceptance or 277CA report", "Original claim copy"],
    }[args.appeal_basis]
    return AppealDraft(
        claim_id=args.claim_id,
        summary=summary,
        attachments_checklist=checklist,
        cited_doc_ids=args.supporting_doc_ids,
    )


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], BaseModel]

    def json_schema(self) -> dict[str, Any]:
        """OpenAI-style function-calling schema."""
        params = self.input_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "lookup_claim",
            "Fetch a denied claim's billing details by claim id.",
            LookupClaimInput,
            ClaimRecord,
            lookup_claim,
        ),
        ToolSpec(
            "get_denial_reason_policy",
            "Get the policy for a denial code (category, guidance, appeal window).",
            DenialPolicyInput,
            DenialPolicy,
            get_denial_reason_policy,
        ),
        ToolSpec(
            "check_prior_auth_status",
            "Check prior-authorization status and retro-auth eligibility for a claim.",
            PriorAuthInput,
            PriorAuthStatus,
            check_prior_auth_status,
        ),
        ToolSpec(
            "search_policy_docs",
            "Keyword search over payer policy documents.",
            SearchPolicyInput,
            SearchPolicyOutput,
            search_policy_docs,
        ),
        ToolSpec(
            "draft_appeal_summary",
            "Draft a PHI-free appeal summary and attachment checklist for a claim.",
            DraftAppealInput,
            AppealDraft,
            draft_appeal_summary,
        ),
    ]
}


def tool_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    selected = TOOLS if names is None else {n: TOOLS[n] for n in names}
    return [spec.json_schema() for spec in selected.values()]
