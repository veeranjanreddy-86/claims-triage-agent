"""FastAPI service: POST /triage, GET /health."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from claims_agent import __version__
from claims_agent.agent import TriageAgent
from claims_agent.config import Settings
from claims_agent.schemas import TriageRequest, TriageResult

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, agent: TriageAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent = agent or TriageAgent(settings or Settings.from_env())
        yield

    app = FastAPI(
        title="Claims Triage Agent",
        version=__version__,
        lifespan=lifespan,
        description="Triage SYNTHETIC claim denials with a tool-calling agent.",
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        a: TriageAgent = request.app.state.agent
        return {"status": "ok", "version": __version__, "provider": a.provider.name}

    @app.post("/triage", response_model=TriageResult)
    def triage(body: TriageRequest, request: Request) -> TriageResult:
        a: TriageAgent = request.app.state.agent
        try:
            return a.run(body.claim_id, body.question)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("triage failed")
            raise HTTPException(status_code=500, detail="internal error") from exc

    return app


app = create_app()
