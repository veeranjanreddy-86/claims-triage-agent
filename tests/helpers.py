"""Test-only providers that speak the same protocol as a function-calling LLM."""

from __future__ import annotations

import json
from typing import Any

from claims_agent.llm import ScriptedPlanner
from claims_agent.schemas import LLMResponse, ToolCall


class RecordingPlanner(ScriptedPlanner):
    """Scripted planner that keeps a copy of every transcript it was shown."""

    def __init__(self) -> None:
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools):  # type: ignore[no-untyped-def]
        self.seen.append(json.loads(json.dumps(messages)))
        return super().complete(messages, tools)


class LoopingProvider:
    """Never finishes: always asks for the same tool again."""

    name = "looping"

    def __init__(self) -> None:
        self.n = 0

    def complete(self, messages, tools):  # type: ignore[no-untyped-def]
        self.n += 1
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id=f"c{self.n}",
                    name="search_policy_docs",
                    arguments='{"query": "timely filing"}',
                )
            ]
        )


class BadArgsOnceProvider(ScriptedPlanner):
    """First emits malformed arguments, then behaves like the scripted planner."""

    name = "bad-args-once"

    def __init__(self, arguments: str = '{"claim_id": 1004, "extra": true}') -> None:
        self.arguments = arguments
        self.sent = False

    def complete(self, messages, tools):  # type: ignore[no-untyped-def]
        if not self.sent:
            self.sent = True
            return LLMResponse(
                tool_calls=[ToolCall(id="bad_1", name="lookup_claim", arguments=self.arguments)]
            )
        return super().complete(messages, tools)


class ScriptedResponses:
    """Replays a fixed list of responses."""

    name = "replay"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    def complete(self, messages, tools):  # type: ignore[no-untyped-def]
        return self.responses.pop(0)
