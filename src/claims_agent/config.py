"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Fixed "today" for the synthetic dataset so window calculations are reproducible.
DEFAULT_AS_OF = "2026-06-30"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path("var/claims.db"))
    trace_dir: Path | None = field(default_factory=lambda: Path("traces"))
    llm_provider: str = "scripted"
    max_steps: int = 8
    max_tool_calls: int = 8
    max_calls_per_tool: int = 3
    tool_retries: int = 2
    as_of: date = field(default_factory=lambda: date.fromisoformat(DEFAULT_AS_OF))

    @classmethod
    def from_env(cls) -> Settings:
        trace_dir = os.getenv("CLAIMS_TRACE_DIR", "traces")
        return cls(
            db_path=Path(os.getenv("CLAIMS_DB_PATH", "var/claims.db")),
            trace_dir=Path(trace_dir) if trace_dir.strip() else None,
            llm_provider=os.getenv("LLM_PROVIDER", "scripted").strip().lower(),
            max_steps=_int_env("AGENT_MAX_STEPS", 8),
            max_tool_calls=_int_env("AGENT_MAX_TOOL_CALLS", 8),
            max_calls_per_tool=_int_env("AGENT_MAX_CALLS_PER_TOOL", 3),
            tool_retries=_int_env("AGENT_TOOL_RETRIES", 2),
            as_of=date.fromisoformat(os.getenv("CLAIMS_AS_OF_DATE", DEFAULT_AS_OF)),
        )
