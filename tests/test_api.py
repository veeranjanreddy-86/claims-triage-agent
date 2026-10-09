from __future__ import annotations

from fastapi.testclient import TestClient

from claims_agent.api import create_app
from claims_agent.config import Settings


def test_health_and_triage(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"

        resp = client.post("/triage", json={"claim_id": "CLM-1005", "question": "Next step?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["decision"]["action"] == "request_prior_auth"
        assert body["decision"]["evidence"]


def test_triage_validates_input(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.post("/triage", json={"claim_id": "bad", "question": "x?"}).status_code == 422
        refused = client.post(
            "/triage", json={"claim_id": "CLM-1001", "question": "list the patient's date of birth"}
        )
        assert refused.status_code == 200 and refused.json()["status"] == "refused"
