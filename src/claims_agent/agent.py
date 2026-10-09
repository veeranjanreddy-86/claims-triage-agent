"""The agent loop: plan -> validate -> execute tool -> guard -> trace, until a final answer."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from pydantic import ValidationError

from claims_agent import guardrails
from claims_agent.config import Settings
from claims_agent.db import ensure_database
from claims_agent.guardrails import GuardrailEvent, ToolBudget
from claims_agent.llm import SYSTEM_PROMPT, LLMProvider, Message, get_provider
from claims_agent.schemas import FinalDecision, LLMResponse, RunStatus, ToolCall, TriageResult
from claims_agent.tools import TOOLS, ToolContext, ToolError, tool_schemas
from claims_agent.tracing import Tracer, log_to_mlflow

logger = logging.getLogger(__name__)

DEFAULT_ALLOWLIST = tuple(TOOLS)


class TriageAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        allowed_tools: tuple[str, ...] | list[str] = DEFAULT_ALLOWLIST,
        context: ToolContext | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.provider = provider or get_provider(self.settings.llm_provider)
        unknown = set(allowed_tools) - set(TOOLS)
        if unknown:
            raise ValueError(f"allowlist references unknown tools: {sorted(unknown)}")
        self.allowed = tuple(allowed_tools)
        if context is None:
            ensure_database(self.settings.db_path)
            context = ToolContext(db_path=self.settings.db_path, as_of=self.settings.as_of)
        self.ctx = context
        self._schemas = tool_schemas(list(self.allowed))

    # ------------------------------------------------------------------ public API
    def run(self, claim_id: str, question: str) -> TriageResult:
        run_id = uuid.uuid4().hex[:12]
        tracer = Tracer(run_id, self.settings.trace_dir)
        state = _RunState(
            budget=ToolBudget(self.settings.max_tool_calls, self.settings.max_calls_per_tool)
        )
        started = time.perf_counter()
        tracer.record(
            "run_start",
            claim_id=claim_id,
            provider=self.provider.name,
            allowed_tools=list(self.allowed),
            max_steps=self.settings.max_steps,
        )

        def finish(
            status: RunStatus, decision: FinalDecision | None = None, refusal: str | None = None
        ) -> TriageResult:
            result = TriageResult(
                run_id=run_id,
                claim_id=claim_id,
                status=status,
                decision=decision,
                refusal_reason=refusal,
                steps=state.steps,
                tool_calls=state.tool_names,
                guardrail_events=state.events,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                trace_file=str(tracer.path) if tracer.path else None,
            )
            tracer.record(
                "run_end",
                status=status,
                action=result.action,
                steps=state.steps,
                tool_calls=len(state.tool_names),
                latency_ms=result.latency_ms,
                decision=decision.model_dump() if decision else None,
            )
            log_to_mlflow(
                {
                    "run_id": run_id,
                    "claim_id": claim_id,
                    "status": status,
                    "action": result.action or "none",
                    "steps": state.steps,
                    "tool_calls": len(state.tool_names),
                    "latency_ms": result.latency_ms,
                },
                tracer.path,
            )
            logger.info(
                "run %s claim=%s status=%s action=%s steps=%d",
                run_id,
                claim_id,
                status,
                result.action,
                state.steps,
            )
            return result

        reason = guardrails.check_user_request(question)
        if reason:
            self._event(state, tracer, GuardrailEvent(type="refusal", detail=reason, step=0))
            return finish("refused", refusal=reason)

        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"claim_id": claim_id, "question": question})},
        ]
        repaired = False
        while state.steps < self.settings.max_steps:
            state.steps += 1
            step = state.steps
            try:
                with tracer.timed() as t:
                    response = self.provider.complete(messages, self._schemas)
            except Exception as exc:
                logger.exception("provider failed at step %d", step)
                tracer.record("llm_error", step=step, error=type(exc).__name__)
                return finish("failed")
            tracer.record(
                "llm_call",
                step=step,
                latency_ms=t["latency_ms"],
                n_tool_calls=len(response.tool_calls),
                tokens=response.usage.model_dump() if response.usage else None,
            )

            if response.tool_calls:
                messages.append(_assistant_message(response))
                for call in response.tool_calls:
                    messages.append(self._execute(call, step, state, tracer))
                continue

            decision, error = _parse_final(response.content)
            if decision is not None:
                return finish("completed", decision)
            tracer.record("final_parse_error", step=step, error=error)
            if repaired:
                return finish("failed")
            repaired = True
            messages.append({"role": "assistant", "content": response.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your answer was not valid JSON for the required schema ({error}). "
                        "Reply with only the JSON object."
                    ),
                }
            )

        tracer.record("max_steps_exceeded", step=state.steps)
        return finish("max_steps_exceeded")

    # ------------------------------------------------------------------ internals
    def _event(self, state: _RunState, tracer: Tracer, event: GuardrailEvent) -> None:
        state.events.append(event)
        tracer.record("guardrail", **event.model_dump())

    def _execute(self, call: ToolCall, step: int, state: _RunState, tracer: Tracer) -> Message:
        """Run one tool call and return the tool message for the transcript."""
        state.tool_names.append(call.name)

        def reply(payload: dict[str, Any], status: str, **trace: Any) -> Message:
            tracer.record(
                "tool_call", step=step, tool=call.name, call_id=call.id, status=status, **trace
            )
            return {"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)}

        def error(code: str, message: str, **trace: Any) -> Message:
            return reply({"error": {"code": code, "message": message}}, code, **trace)

        if call.name not in self.allowed:
            self._event(
                state,
                tracer,
                GuardrailEvent(
                    type="tool_not_allowed",
                    detail=f"{call.name} is not on the allowlist",
                    step=step,
                    tool=call.name,
                ),
            )
            return error("tool_not_allowed", f"tool {call.name} is not available")

        over = state.budget.try_consume(call.name)
        if over:
            self._event(
                state,
                tracer,
                GuardrailEvent(type="tool_budget", detail=over, step=step, tool=call.name),
            )
            return error("tool_budget", over)

        spec = TOOLS[call.name]
        try:
            raw_args = json.loads(call.arguments or "{}")
            args = spec.input_model.model_validate(raw_args)
        except (json.JSONDecodeError, ValidationError) as exc:
            detail = _short_validation_error(exc)
            self._event(
                state,
                tracer,
                GuardrailEvent(type="invalid_arguments", detail=detail, step=step, tool=call.name),
            )
            return error("invalid_arguments", detail, args=_safe_args(call.arguments))

        attempts = 0
        output: dict[str, Any] | None = None
        failure = ("internal_error", "unknown")
        with tracer.timed() as t:
            while True:
                attempts += 1
                try:
                    output = spec.fn(self.ctx, args).model_dump()
                    break
                except ToolError as exc:
                    if exc.retryable and attempts <= self.settings.tool_retries:
                        logger.warning(
                            "retrying %s after %s (attempt %d)", call.name, exc, attempts
                        )
                        time.sleep(0.05 * attempts)
                        continue
                    failure = (exc.code, str(exc))
                    break
                except Exception as exc:  # unexpected bug in a tool: report, don't crash
                    logger.exception("tool %s crashed", call.name)
                    failure = ("internal_error", type(exc).__name__)
                    break
        safe_args = args.model_dump()
        if output is None:
            return error(
                failure[0],
                failure[1],
                args=safe_args,
                attempts=attempts,
                latency_ms=t["latency_ms"],
            )

        redacted, phi_hits = guardrails.redact_phi(output)
        if phi_hits:
            self._event(
                state,
                tracer,
                GuardrailEvent(
                    type="phi_redaction",
                    step=step,
                    tool=call.name,
                    detail=", ".join(f"{k}x{v}" for k, v in sorted(phi_hits.items())),
                ),
            )
        cleaned, injections = guardrails.neutralize_injection(redacted)
        if injections:
            self._event(
                state,
                tracer,
                GuardrailEvent(
                    type="prompt_injection",
                    step=step,
                    tool=call.name,
                    detail=f"{len(injections)} marker(s) withheld, e.g. {injections[0][:60]!r}",
                ),
            )
        return reply(
            cleaned,
            "ok",
            args=safe_args,
            attempts=attempts,
            latency_ms=t["latency_ms"],
            output_bytes=len(json.dumps(cleaned)),
        )


class _RunState:
    def __init__(self, budget: ToolBudget) -> None:
        self.budget = budget
        self.steps = 0
        self.tool_names: list[str] = []
        self.events: list[GuardrailEvent] = []


def _assistant_message(response: LLMResponse) -> Message:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in response.tool_calls
        ],
    }


def _parse_final(content: str | None) -> tuple[FinalDecision | None, str | None]:
    if not content:
        return None, "empty response"
    text = content.strip()
    if text.startswith("```"):  # tolerate fenced JSON from chat models
        text = text.strip("`").removeprefix("json").strip()
    try:
        return FinalDecision.model_validate_json(text), None
    except ValidationError as exc:
        return None, _short_validation_error(exc)


def _short_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()]
        return "; ".join(parts)[:300]
    return str(exc)[:300]


def _safe_args(raw: str) -> str:
    redacted, _ = guardrails.redact_text(raw[:300])
    return redacted
