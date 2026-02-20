"""Base protocol interfaces and runtime orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.env_utils import env_float, env_int
from core.token_tracker import TRACKER


@dataclass
class ProtocolResult:
    """Structured protocol run output."""

    answer: str
    confidence: float = 0.0
    reasoning_trace: list[str] = field(default_factory=list)
    verification_passed: bool = False
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProtocol(ABC):
    """Abstract protocol with perception, cognition, and verification hooks."""

    max_llm_calls_per_task: int = 10

    def __init__(self, llm_client, model_name: str = "gpt-4o-mini") -> None:
        self.llm = llm_client
        self.model = model_name
        self._call_count = 0
        self._task_tokens_used = 0
        self.api_timeout_seconds = env_float(
            ["WORKER_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
        self.max_completion_tokens = env_int(
            ["WORKER_MAX_TOKENS", "EVO_WORKER_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=65536,
        )

    @abstractmethod
    def perception(self, context: str) -> dict[str, Any]:
        """Transform raw context into structured signals."""

    @abstractmethod
    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        """Produce a candidate answer from query and perceived context."""

    @abstractmethod
    def verification(self, answer: str, context: str) -> bool:
        """Return True when answer passes protocol-defined checks."""

    def run(self, context: str, query: str, max_retries: int = 2) -> ProtocolResult:
        """Execute perception -> cognition -> verification with retry feedback."""

        self._call_count = 0
        self._task_tokens_used = 0
        trace: list[str] = []

        perceived = self.perception(context)
        trace.append(f"[Perception] keys={sorted(perceived.keys())}")

        feedback = ""
        last_answer = ""
        for attempt in range(max_retries + 1):
            if feedback:
                perceived["retry_feedback"] = feedback

            answer = self.cognition(query, perceived)
            last_answer = answer
            trace.append(f"[Cognition-{attempt}] {answer[:120]!r}")

            passed = bool(self.verification(answer, context))
            trace.append(f"[Verification-{attempt}] passed={passed}")
            if passed:
                return ProtocolResult(
                    answer=answer,
                    confidence=max(0.2, 1.0 - 0.2 * attempt),
                    reasoning_trace=trace,
                    verification_passed=True,
                    tokens_used=self._task_tokens_used,
                    metadata={"attempts": attempt + 1, "llm_calls": self._call_count},
                )

            feedback = "Previous answer failed verification. Re-check context contradictions and missing steps."

        trace.append("[Guardrail] verification_failed -> output_blocked")
        return ProtocolResult(
            answer="",
            confidence=0.0,
            reasoning_trace=trace,
            verification_passed=False,
            tokens_used=self._task_tokens_used,
            metadata={
                "attempts": max_retries + 1,
                "llm_calls": self._call_count,
                "output_blocked": True,
                "blocked_answer_preview": last_answer[:500],
            },
        )

    def _call_llm(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Single LLM call wrapper with token accounting and call budget guard."""

        self._call_count += 1
        if self._call_count > self.max_llm_calls_per_task:
            raise RuntimeError(
                f"LLM call budget exceeded ({self.max_llm_calls_per_task} per task)."
            )

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_completion_tokens,
            timeout=self.api_timeout_seconds,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self._task_tokens_used += prompt_tokens + completion_tokens
            TRACKER.record(self.model, prompt_tokens, completion_tokens)

        return (response.choices[0].message.content or "").strip()
