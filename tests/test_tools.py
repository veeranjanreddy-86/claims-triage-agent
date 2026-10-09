from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from claims_agent.data.seed import build_database
from claims_agent.tools import (
    TOOLS,
    DraftAppealInput,
    LookupClaimInput,
    NotFoundError,
    PriorAuthInput,
    SearchPolicyInput,
    check_prior_auth_status,
    lookup_claim,
    search_policy_docs,
    tool_schemas,
)


def test_every_tool_exposes_a_strict_function_schema() -> None:
    schemas = tool_schemas()
    assert {s["function"]["name"] for s in schemas} == set(TOOLS)
    for s in schemas:
        assert s["type"] == "function"
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert params["required"], s["function"]["name"]
        assert s["function"]["description"]


def test_input_models_reject_bad_arguments() -> None:
    with pytest.raises(ValidationError):
        LookupClaimInput.model_validate({"claim_id": "1004"})
    with pytest.raises(ValidationError):
        LookupClaimInput.model_validate({"claim_id": "CLM-1004", "sql": "DROP TABLE"})
    with pytest.raises(ValidationError):
        SearchPolicyInput.model_validate({"query": "auth", "top_k": 50})
    with pytest.raises(ValidationError):
        DraftAppealInput.model_validate(
            {"claim_id": "CLM-1004", "denial_code": "CO-197", "appeal_basis": "because"}
        )


def test_lookup_claim_and_not_found(ctx) -> None:  # type: ignore[no-untyped-def]
    claim = lookup_claim(ctx, LookupClaimInput(claim_id="CLM-1004"))
    assert claim.denial_code == "CO-197"
    assert claim.days_since_denial == 20
    with pytest.raises(NotFoundError):
        lookup_claim(ctx, LookupClaimInput(claim_id="CLM-9999"))


def test_prior_auth_retro_window_uses_payer_override(ctx) -> None:  # type: ignore[no-untyped-def]
    inside = check_prior_auth_status(ctx, PriorAuthInput(claim_id="CLM-1005"))
    assert inside.status == "not_found" and inside.retro_auth_eligible
    cedar = check_prior_auth_status(ctx, PriorAuthInput(claim_id="CLM-1006"))
    assert cedar.retro_auth_window_days == 14 and not cedar.retro_auth_eligible
    approved = check_prior_auth_status(ctx, PriorAuthInput(claim_id="CLM-1004"))
    assert approved.status == "approved" and approved.auth_number


def test_search_filters_by_payer(ctx) -> None:  # type: ignore[no-untyped-def]
    q = SearchPolicyInput(query="timely filing limit appeal", top_k=5, payer="Bluefield Mutual")
    hits = search_policy_docs(ctx, q).hits
    assert hits and all(h.payer in ("*", "Bluefield Mutual") for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_seed_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def dump(path):  # type: ignore[no-untyped-def]
        conn = sqlite3.connect(path)
        rows = conn.execute("SELECT * FROM claims ORDER BY claim_id").fetchall()
        conn.close()
        return rows

    a = dump(build_database(tmp_path / "a.db"))
    b = dump(build_database(tmp_path / "b.db"))
    assert a == b and len(a) == 198
