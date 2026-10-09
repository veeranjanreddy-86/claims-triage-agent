from __future__ import annotations

from pathlib import Path

import pytest

from claims_agent.agent import TriageAgent
from claims_agent.config import Settings
from claims_agent.data.seed import build_database
from claims_agent.tools import ToolContext


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_database(tmp_path_factory.mktemp("db") / "claims.db")


@pytest.fixture
def settings(db_path: Path, tmp_path: Path) -> Settings:
    return Settings(db_path=db_path, trace_dir=tmp_path / "traces")


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return ToolContext(db_path=settings.db_path, as_of=settings.as_of)


@pytest.fixture
def agent(settings: Settings) -> TriageAgent:
    return TriageAgent(settings)
