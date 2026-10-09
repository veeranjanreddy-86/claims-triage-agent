"""Per-run JSONL tracing (in the spirit of MLflow Tracing spans, without the dependency).

Every event is one JSON line: ``{"ts", "run_id", "seq", "event", ...fields}``.
Callers are responsible for redacting payloads *before* recording them.
Optional: if ``mlflow`` is installed and ``MLFLOW_TRACKING_URI`` is set, a run
summary and the trace file are logged to MLflow as well.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, run_id: str, trace_dir: Path | None) -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.path: Path | None = None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = trace_dir / f"{run_id}.jsonl"

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "seq": len(self.events),
            "event": event,
            **fields,
        }
        self.events.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    @contextmanager
    def timed(self) -> Iterator[dict[str, float]]:
        """Yield a dict whose ``latency_ms`` is filled in on exit."""
        box: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield box
        finally:
            box["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)


def log_to_mlflow(summary: dict[str, Any], trace_path: Path | None) -> bool:
    """Best-effort MLflow logging; returns True if something was logged."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return False
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("mlflow not installed; skipping")
        return False
    try:
        with mlflow.start_run(run_name=f"triage-{summary.get('run_id')}"):
            mlflow.log_params(
                {k: summary[k] for k in ("claim_id", "status", "action") if k in summary}
            )
            mlflow.log_metrics(
                {
                    k: float(summary[k])
                    for k in ("steps", "tool_calls", "latency_ms")
                    if k in summary
                }
            )
            if trace_path is not None and trace_path.exists():
                mlflow.log_artifact(str(trace_path), artifact_path="traces")
        return True
    except Exception:  # never let telemetry break the agent
        logger.warning("MLflow logging failed", exc_info=True)
        return False
