"""Meta-architect generation with validation-guided self-repair."""

from __future__ import annotations

import re
import time
import uuid
from typing import Optional, Union

from core.base_protocol import BaseProtocol
from core.env_utils import env_float, env_int
from core.protocol_loader import ProtocolLoader, SandboxProtocolLoader


def extract_python_code(text: str) -> str:
    """Extract code from a markdown fenced block if present."""

    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _extract_supported_max_tokens(error_text: str) -> Optional[int]:
    match = re.search(r"supports at most\s+(\d+)\s+completion tokens", error_text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _call_architect(
    architect_client,
    architect_model: str,
    prompt: str,
    request_tag: str = "",
) -> str:
    request_timeout = env_float(
        ["ARCHITECT_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
        default=120.0,
    )
    max_tokens = env_int(
        ["ARCHITECT_MAX_TOKENS", "EVO_ARCHITECT_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
        default=8192,
    )
    max_retries = env_int(["ARCHITECT_MAX_RETRIES", "EVO_ARCHITECT_MAX_RETRIES"], default=3)
    max_retries = max(1, max_retries)
    temperature = env_float(["ARCHITECT_TEMPERATURE", "EVO_ARCHITECT_TEMPERATURE"], default=0.0)
    token_limit = max(256, int(max_tokens))
    base_prompt = str(prompt or "")

    for attempt in range(max_retries):
        nonce = f"{request_tag or 'architect'}-{attempt}-{uuid.uuid4().hex[:12]}"
        prompt_with_nonce = (
            f"{base_prompt}\n\n"
            f"[request_nonce={nonce}]"
        )
        try:
            response = architect_client.chat.completions.create(
                model=architect_model,
                messages=[{"role": "user", "content": prompt_with_nonce}],
                temperature=temperature,
                max_tokens=token_limit,
                timeout=request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            error_text = str(exc)
            lowered = error_text.lower()
            supported = _extract_supported_max_tokens(error_text)
            if supported is not None and supported < token_limit:
                token_limit = max(256, supported)
                time.sleep(0.5)
                continue

            if "same request has failed before" in lowered:
                # Retry with a new nonce so the provider sees a distinct request.
                time.sleep(1.0 + 0.5 * attempt)
                continue

            if attempt >= max_retries - 1:
                raise RuntimeError(f"Architect API failed after {max_retries} attempts: {exc}") from exc
            time.sleep(1.0 + attempt)
    raise RuntimeError("Architect API failed with unknown error")


def generate_with_repair(
    architect_client,
    architect_model: str,
    architect_prompt: str,
    loader: Union[ProtocolLoader, SandboxProtocolLoader],
    max_repair_attempts: int = 2,
    request_tag: str = "",
) -> tuple[Optional[BaseProtocol], str]:
    """Generate protocol code and retry repairs on validation failures.

    Works with both ProtocolLoader (legacy) and SandboxProtocolLoader (CaS).
    """

    request_prefix = request_tag or f"repair-{uuid.uuid4().hex[:8]}"
    try:
        raw = _call_architect(
            architect_client,
            architect_model,
            architect_prompt,
            request_tag=f"{request_prefix}-init",
        )
    except Exception as exc:
        return None, str(exc)
    code = extract_python_code(raw)

    for attempt in range(max_repair_attempts + 1):
        protocol, error = loader.load_from_code(code)
        if protocol is not None:
            return protocol, code

        if error is None:
            return None, "Unknown loader error"

        if attempt >= max_repair_attempts:
            return None, f"Validation failed after {max_repair_attempts + 1} attempts: {error.message}"

        repair_prompt = (
            "The generated protocol failed validation. Please fix it.\n\n"
            "Hard constraints:\n"
            "- Output only executable Python code (no markdown, no prose).\n"
            "- Include exactly one class inheriting BaseCaSCompiler.\n"
            "- Keep exact method signatures:\n"
            "  def compile_sandbox(self, context: str) -> str\n"
            "  def generate_solver(self, query: str, sandbox_schema: str) -> str\n"
            "- Do NOT override immutable runtime methods: __init__, run, _call_llm,\n"
            "  _make_oracle_fn, _sanitize_generated_code, _attempt_syntax_repair,\n"
            "  _answer_with_oracle_fallback, _derive_dynamic_call_budget.\n\n"
            f"Feedback:\n{error.to_feedback()}\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            "Output only corrected Python code."
        )
        try:
            raw = _call_architect(
                architect_client,
                architect_model,
                repair_prompt,
                request_tag=f"{request_prefix}-repair-{attempt}",
            )
        except Exception as exc:
            return None, str(exc)
        code = extract_python_code(raw)

    return None, "Unreachable state"
