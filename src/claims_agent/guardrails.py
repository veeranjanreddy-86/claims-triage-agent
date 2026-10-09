"""Guardrails: PHI redaction, prompt-injection screening, tool budgets, refusals.

These are deliberately simple, auditable heuristics. They reduce risk; they are
not a substitute for a reviewed de-identification pipeline or a policy engine.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

GuardrailType = Literal[
    "phi_redaction",
    "prompt_injection",
    "tool_budget",
    "tool_not_allowed",
    "invalid_arguments",
    "refusal",
]


class GuardrailEvent(BaseModel):
    type: GuardrailType
    detail: str
    step: int | None = None
    tool: str | None = None


# --------------------------------------------------------------------------- PHI
# Structured fields that are always PHI, regardless of their value.
PHI_KEYS = frozenset({"patient_name", "dob", "member_id", "ssn", "phone", "address", "email"})

_PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.\s])?\b\d{3}[-.]\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("member_id", re.compile(r"\bMBR\d{6,}\b")),
    ("dob", re.compile(r"\b(?:DOB|date of birth)\s*[:#]?\s*\d{4}-\d{2}-\d{2}\b", re.I)),
]


def redact_text(text: str) -> tuple[str, Counter[str]]:
    hits: Counter[str] = Counter()
    for label, pattern in _PHI_PATTERNS:
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        if n:
            hits[label] += n
    return text, hits


def redact_phi(obj: Any) -> tuple[Any, Counter[str]]:
    """Recursively redact PHI keys and PHI-looking substrings. Returns (copy, counts)."""
    hits: Counter[str] = Counter()

    def walk(value: Any, key: str | None = None) -> Any:
        if key in PHI_KEYS and value not in (None, ""):
            hits[key] += 1
            return f"[REDACTED:{key}]"
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            new, found = redact_text(value)
            hits.update(found)
            return new
        return value

    return walk(obj), hits


# --------------------------------------------------------------------------- injection
_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your)\b"
    r".{0,20}\b(instructions?|rules|guidelines|prompts?)\b",
    r"\byou are now\b",
    r"\b(admin|developer|god|jailbreak)\s+mode\b",
    r"\bsystem prompt\b",
    r"\breveal\b.{0,40}\b(ssn|social security|password|secret|prompt|date of birth)\b",
    r"\bdo not (mention|tell|reveal)\b.{0,30}\b(this|user|anyone)\b",
    r"\bapprove (every|all) claims?\b",
    r"<\|?(im_start|system)\|?>|\[/?INST\]",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I | re.S)
INJECTION_PLACEHOLDER = "[WITHHELD: retrieved text matched prompt-injection heuristics]"


def detect_injection(text: str) -> list[str]:
    return [m.group(0) for m in _INJECTION_RE.finditer(text)]


def neutralize_injection(obj: Any) -> tuple[Any, list[str]]:
    """Replace any string containing injection markers with a placeholder.

    The whole string is withheld (not just the match) because surrounding text in
    the same chunk is attacker-controlled too.
    """
    found: list[str] = []

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out = {k: walk(v) for k, v in value.items()}
            if any(v == INJECTION_PLACEHOLDER for v in out.values()):
                out["injection_detected"] = True
            return out
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            matches = detect_injection(value)
            if matches:
                found.extend(matches)
                return INJECTION_PLACEHOLDER
        return value

    return walk(obj), found


# --------------------------------------------------------------------------- user request
_REFUSAL_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "requests patient identifiers",
        re.compile(
            r"\b(ssn|social security|date of birth|dob|home address|phone number|"
            r"member id|patient'?s? name)\b",
            re.I,
        ),
    ),
    (
        "requests bulk data export",
        re.compile(
            r"\b(export|dump|list|download)\b.{0,30}\b(all|every)\b.{0,20}"
            r"\b(patients?|claims|records|members)\b",
            re.I,
        ),
    ),
]


def check_user_request(question: str) -> str | None:
    """Return a refusal reason if the request is out of policy, else None."""
    if detect_injection(question):
        return "request contains instruction-override text"
    for reason, pattern in _REFUSAL_RULES:
        if pattern.search(question):
            return reason
    return None


# --------------------------------------------------------------------------- budget
@dataclass
class ToolBudget:
    max_total: int
    max_per_tool: int
    used: Counter[str] = field(default_factory=Counter)

    def try_consume(self, tool: str) -> str | None:
        """Consume one call; return a reason string if the budget is exhausted."""
        if sum(self.used.values()) >= self.max_total:
            return f"total tool-call budget of {self.max_total} exhausted"
        if self.used[tool] >= self.max_per_tool:
            return f"per-tool budget of {self.max_per_tool} exhausted for {tool}"
        self.used[tool] += 1
        return None
