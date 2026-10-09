"""Offline evaluation over synthetic triage scenarios.

Metrics
- action_accuracy: final action == expected (refusals expect no action + status "refused")
- tool_precision / tool_recall: micro-averaged over the set of distinct tools per scenario
- guardrail_accuracy: triggered set of {phi_redaction, prompt_injection, refusal, tool_budget}
  equals the expected set
- avg_steps / avg_tool_calls / mean_latency_ms

Usage:  python -m claims_agent.evaluate --out reports/eval_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from claims_agent.agent import TriageAgent
from claims_agent.config import Settings

logger = logging.getLogger(__name__)

TRACKED_GUARDRAILS = frozenset({"phi_redaction", "prompt_injection", "refusal", "tool_budget"})


@dataclass(frozen=True)
class Scenario:
    id: str
    claim_id: str
    question: str
    expected_action: str | None
    expected_tools: frozenset[str]
    expected_guardrails: frozenset[str]


@dataclass
class ScenarioResult:
    id: str
    claim_id: str
    expected_action: str | None
    action: str | None
    status: str
    expected_tools: list[str]
    tools: list[str]
    expected_guardrails: list[str]
    guardrails: list[str]
    steps: int
    tool_calls: int
    latency_ms: float

    @property
    def action_ok(self) -> bool:
        if self.expected_action is None:
            return self.action is None and self.status == "refused"
        return self.action == self.expected_action

    @property
    def guardrails_ok(self) -> bool:
        return set(self.guardrails) == set(self.expected_guardrails)


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    if path is None:
        raw = resources.files("claims_agent.data").joinpath("eval_scenarios.json").read_text()
    else:
        raw = path.read_text(encoding="utf-8")
    return [
        Scenario(
            s["id"],
            s["claim_id"],
            s["question"],
            s["expected_action"],
            frozenset(s["expected_tools"]),
            frozenset(s["expected_guardrails"]),
        )
        for s in json.loads(raw)
    ]


def run_scenarios(agent: TriageAgent, scenarios: list[Scenario]) -> list[ScenarioResult]:
    results = []
    for sc in scenarios:
        r = agent.run(sc.claim_id, sc.question)
        results.append(
            ScenarioResult(
                id=sc.id,
                claim_id=sc.claim_id,
                expected_action=sc.expected_action,
                action=r.action,
                status=r.status,
                expected_tools=sorted(sc.expected_tools),
                tools=sorted(set(r.tool_calls)),
                expected_guardrails=sorted(sc.expected_guardrails),
                guardrails=sorted({e.type for e in r.guardrail_events} & TRACKED_GUARDRAILS),
                steps=r.steps,
                tool_calls=len(r.tool_calls),
                latency_ms=r.latency_ms,
            )
        )
    return results


def compute_metrics(results: list[ScenarioResult]) -> dict[str, float]:
    if not results:
        raise ValueError("no results to score")
    n = len(results)
    tp = sum(len(set(r.tools) & set(r.expected_tools)) for r in results)
    predicted = sum(len(set(r.tools)) for r in results)
    expected = sum(len(set(r.expected_tools)) for r in results)
    return {
        "scenarios": n,
        "action_accuracy": round(sum(r.action_ok for r in results) / n, 4),
        "tool_precision": round(tp / predicted, 4) if predicted else 1.0,
        "tool_recall": round(tp / expected, 4) if expected else 1.0,
        "guardrail_accuracy": round(sum(r.guardrails_ok for r in results) / n, 4),
        "avg_steps": round(statistics.mean(r.steps for r in results), 2),
        "avg_tool_calls": round(statistics.mean(r.tool_calls for r in results), 2),
        "mean_latency_ms": round(statistics.mean(r.latency_ms for r in results), 2),
    }


def render_markdown(metrics: dict[str, float], results: list[ScenarioResult], provider: str) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Provider: `{provider}` | scenarios: {int(metrics['scenarios'])}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in metrics.items() if k != "scenarios"]
    lines += [
        "",
        "| Scenario | Claim | Expected | Got | Action | Tools P/R ok | Guardrails |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tools_ok = "yes" if set(r.tools) == set(r.expected_tools) else "no"
        lines.append(
            f"| {r.id} | {r.claim_id} | {r.expected_action or '(refuse)'} | "
            f"{r.action or '(' + r.status + ')'} | {'pass' if r.action_ok else 'FAIL'} | "
            f"{tools_ok} | {'pass' if r.guardrails_ok else 'FAIL'} |"
        )
    failures = [r for r in results if not (r.action_ok and r.guardrails_ok)]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(
                f"- **{r.id}**: expected `{r.expected_action}`, got `{r.action}` "
                f"(guardrails expected {r.expected_guardrails}, got {r.guardrails})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline triage evaluation.")
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/eval_report.md"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base = Settings.from_env()
    settings = Settings(**{**asdict(base), "trace_dir": Path("traces/eval")})
    agent = TriageAgent(settings)
    results = run_scenarios(agent, load_scenarios(args.scenarios))
    metrics = compute_metrics(results)
    report = render_markdown(metrics, results, agent.provider.name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    payload: dict[str, Any] = {"metrics": metrics, "results": [asdict(r) for r in results]}
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
