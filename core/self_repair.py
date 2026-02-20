"""Meta-architect generation with validation-guided self-repair."""

from __future__ import annotations

import time
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


def _call_architect(architect_client, architect_model: str, prompt: str) -> str:
    request_timeout = env_float(
        ["ARCHITECT_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
        default=120.0,
    )
    max_tokens = env_int(
        ["ARCHITECT_MAX_TOKENS", "EVO_ARCHITECT_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
        default=65536,
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = architect_client.chat.completions.create(
                model=architect_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt >= max_retries - 1:
                raise RuntimeError(f"Architect API failed after {max_retries} attempts: {exc}") from exc
            time.sleep(2)
    raise RuntimeError("Architect API failed with unknown error")


def generate_with_repair(
    architect_client,
    architect_model: str,
    architect_prompt: str,
    loader: Union[ProtocolLoader, SandboxProtocolLoader],
    max_repair_attempts: int = 2,
) -> tuple[Optional[BaseProtocol], str]:
    """Generate protocol code and retry repairs on validation failures.

    Works with both ProtocolLoader (legacy) and SandboxProtocolLoader (CaS).
    """

    try:
        raw = _call_architect(architect_client, architect_model, architect_prompt)
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
            f"Feedback:\n{error.to_feedback()}\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            "Output only corrected Python code."
        )
        try:
            raw = _call_architect(architect_client, architect_model, repair_prompt)
        except Exception as exc:
            return None, str(exc)
        code = extract_python_code(raw)

    return None, "Unreachable state"
