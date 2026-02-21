"""Meta-architect generation with validation-guided self-repair."""

from __future__ import annotations

import ast
import json
import re
import time
import uuid
from typing import Optional, Union

from core.base_protocol import BaseProtocol
from core.env_utils import env_float, env_int
from core.protocol_loader import ProtocolLoader, SandboxProtocolLoader, TDGProtocolLoader


def extract_python_code(text: str) -> str:
    """Extract code from a markdown fenced block if present."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    # Some providers return JSON wrappers like {"code":"..."}.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("code", "python", "python_code", "source", "content"):
                    code_field = parsed.get(key)
                    if isinstance(code_field, str) and code_field.strip():
                        raw = code_field.strip()
                        break
        except Exception:
            pass

    # Some providers return the Python source as an escaped string literal.
    if raw and raw[0] in {"'", '"'}:
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, str) and literal.strip():
                raw = literal.strip()
        except Exception:
            pass

    if raw and raw[0] in {"'", '"'} and not raw.startswith(('"""', "'''")):
        quote = raw[0]
        if raw.count(quote) == 1:
            raw = raw[1:].strip()
        elif raw.endswith(quote):
            inner = raw[1:-1].strip()
            if inner:
                raw = inner

    if "\\n" in raw and "\n" not in raw and ("class " in raw or "def " in raw):
        try:
            decoded = raw.encode("utf-8").decode("unicode_escape").strip()
            if "\n" in decoded:
                raw = decoded
        except Exception:
            pass

    if raw.startswith("python\n"):
        raw = raw.split("\n", 1)[1].strip()

    if "```python" in raw:
        return raw.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw


def _local_syntax_salvage_candidates(code: str) -> list[str]:
    """Generate local normalization candidates for quoted/escaped code payloads."""

    base = str(code or "").strip()
    if not base:
        return []

    candidates: list[str] = []

    def _add(value: str) -> None:
        normalized = str(value or "").strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(base)
    _add(extract_python_code(base))

    # Try stripping single unmatched wrapper quotes.
    if base and base[0] in {"'", '"'} and not base.startswith(('"""', "'''")):
        quote = base[0]
        if base.count(quote) == 1:
            _add(base[1:])
        elif base.endswith(quote):
            _add(base[1:-1])

    # Try decoding escaped newlines/backslashes.
    for candidate in list(candidates):
        if "\\n" in candidate or "\\t" in candidate or "\\\\" in candidate:
            try:
                decoded = candidate.encode("utf-8").decode("unicode_escape")
                _add(decoded)
            except Exception:
                pass

    # Re-run extractor on all transformed variants.
    for candidate in list(candidates):
        _add(extract_python_code(candidate))

    return candidates


def _error_stage_rank(stage: str) -> int:
    """Higher is better (later validation stage means more code-like output)."""

    order = {
        "syntax": 0,
        "lint": 1,
        "security": 2,
        "contract": 3,
        "load": 4,
    }
    return order.get(str(stage or "").lower(), -1)


def _expr_mentions_base_tdg(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "BaseTDGCompiler":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "BaseTDGCompiler":
        return True
    for child in ast.iter_child_nodes(node):
        if _expr_mentions_base_tdg(child):
            return True
    return False


def _class_inherits_base_tdg(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _expr_mentions_base_tdg(base):
            return True
    return False


def _calls_self_method(func_node: ast.AST, method_name: str) -> bool:
    for child in ast.walk(func_node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr == method_name
        ):
            return True
    return False


def _attempt_local_tdg_contract_repair(code: str, error_message: str) -> tuple[Optional[str], str]:
    """Local downgrade repair for common TDG contract misses."""

    message = str(error_message or "")
    if "must call self._call_llm" not in message:
        return None, "skip:non_call_llm_contract"

    try:
        tree = ast.parse(code)
    except Exception as exc:
        return None, f"parse_failed:{type(exc).__name__}"

    found_tdg_subclass = False
    changed = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _class_inherits_base_tdg(node):
            continue
        found_tdg_subclass = True

        method_nodes: dict[str, ast.FunctionDef] = {
            item.name: item
            for item in node.body
            if isinstance(item, ast.FunctionDef)
        }

        compile_fn = method_nodes.get("compile_tests")
        if compile_fn is not None and not _calls_self_method(compile_fn, "_call_llm"):
            compile_fn.body = ast.parse(
                """
prompt = (
    "Write Python test functions named test_* that accept answer:str and use assert checks. "
    "Generate executable Python only.\\n\\n"
    f"Context:\\n{context}\\n\\nQuery:\\n{query}"
)
raw_tests = self._call_llm([{"role": "user", "content": prompt}], temperature=0.0)
if "```python" in raw_tests:
    return raw_tests.split("```python", 1)[1].split("```", 1)[0].strip()
if "```" in raw_tests:
    return raw_tests.split("```", 1)[1].split("```", 1)[0].strip()
return str(raw_tests).strip()
"""
            ).body
            changed = True

        answer_fn = method_nodes.get("generate_answer")
        if answer_fn is not None and not _calls_self_method(answer_fn, "_call_llm"):
            answer_fn.body = ast.parse(
                """
if messages_raw:
    structured = []
    for msg in messages_raw:
        role = str(msg.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        structured.append({"role": role, "content": str(msg.get("content", ""))})
    if structured:
        structured[-1]["content"] = (
            str(structured[-1].get("content", ""))
            + "\\n\\nAnswer using only provided context and follow explicit constraints."
        )
    return str(self._call_llm(structured, temperature=0.0)).strip()
messages = [
    {"role": "system", "content": "Answer using only provided context.\\n\\nContext:\\n" + str(context)},
    {"role": "user", "content": str(query)},
]
return str(self._call_llm(messages, temperature=0.0)).strip()
"""
            ).body
            changed = True

    if not found_tdg_subclass:
        return None, "no_base_tdg_subclass"
    if not changed:
        return None, "no_missing_call_llm_path"

    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree), "ok"
    except Exception as exc:
        return None, f"unparse_failed:{type(exc).__name__}"


def _response_code_candidates(raw_response: str) -> list[str]:
    """Extract multiple plausible code candidates from a model response."""

    raw = str(raw_response or "")
    seeds: list[str] = []

    def _add_seed(value: str) -> None:
        text = str(value or "").strip()
        if text and text not in seeds:
            seeds.append(text)

    _add_seed(raw)
    _add_seed(extract_python_code(raw))

    # Gather all fenced blocks rather than trusting the first one.
    for match in re.finditer(r"```(?:python|py)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE):
        _add_seed(match.group(1))
    for match in re.finditer(r"```[^\n]*\n([\s\S]*?)```", raw):
        _add_seed(match.group(1))

    # Common anchors for protocol code in plain-text responses.
    for anchor in (
        "from core.base_tdg_protocol import BaseTDGCompiler",
        "from core.base_sandbox_protocol import BaseCaSCompiler",
        "from core.base_protocol import BaseProtocol",
        "class ",
    ):
        idx = raw.find(anchor)
        if idx != -1:
            _add_seed(raw[idx:])

    candidates: list[str] = []

    def _add_candidate(value: str) -> None:
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    for seed in seeds:
        for candidate in _local_syntax_salvage_candidates(seed):
            _add_candidate(candidate)

    return candidates


def _pick_best_candidate(
    raw_response: str,
    loader: Union[ProtocolLoader, SandboxProtocolLoader, TDGProtocolLoader],
) -> tuple[Optional[BaseProtocol], str, Optional[object]]:
    """Try multiple extraction candidates and return best protocol/code/error triple."""

    candidates = _response_code_candidates(raw_response)
    if not candidates:
        return None, extract_python_code(raw_response), None

    best_code = candidates[0]
    best_error = None
    best_rank = (-10, -1)  # (stage_rank, code_len)

    for candidate in candidates:
        protocol, error = loader.load_from_code(candidate)
        if protocol is not None:
            return protocol, candidate, None
        if error is None:
            continue
        rank = (_error_stage_rank(getattr(error, "stage", "")), len(candidate))
        if best_error is None or rank > best_rank:
            best_error = error
            best_rank = rank
            best_code = candidate

    return None, best_code, best_error


def _extract_supported_max_tokens(error_text: str) -> Optional[int]:
    patterns = (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"supports at most\s+(\d+)\s+output tokens",
        r"max(?:imum)?\s+(?:output\s+)?tokens?\s*(?:is|=)\s*(\d+)",
    )
    match = None
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            break
    if not match:
        return None
    try:
        value = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _extract_responses_text(response_obj) -> str:
    """Best-effort extraction from Responses API payloads."""

    output_text = getattr(response_obj, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    output_items = getattr(response_obj, "output", None)
    if isinstance(output_items, list):
        for item in output_items:
            content_items = getattr(item, "content", None)
            if content_items is None and isinstance(item, dict):
                content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                text_value = getattr(content, "text", None)
                if text_value is None and isinstance(content, dict):
                    text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value)

    if parts:
        return "\n".join(parts)
    return ""


def _is_chat_model_incompat_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    markers = (
        "chatcompletion operation does not work",
        "chat completion operation does not work",
        "chat.completions operation does not work",
        "operation does not work with the specified model",
        "use responses api",
    )
    return any(marker in lowered for marker in markers)


def _is_unsupported_temperature_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return "temperature" in lowered and "unsupported parameter" in lowered


def _call_architect(
    architect_client,
    architect_model: str,
    prompt: str,
    request_tag: str = "",
    temperature_override: float | None = None,
) -> str:
    request_timeout = env_float(
        ["ARCHITECT_API_TIMEOUT_SECONDS", "EVO_ARCHITECT_API_TIMEOUT_SECONDS"],
        default=25.0,
    )
    max_tokens = env_int(
        ["ARCHITECT_MAX_TOKENS", "EVO_ARCHITECT_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
        default=8192,
    )
    max_retries = env_int(["ARCHITECT_MAX_RETRIES", "EVO_ARCHITECT_MAX_RETRIES"], default=1)
    max_retries = max(1, max_retries)
    total_timeout = env_float(
        ["ARCHITECT_TOTAL_TIMEOUT_SECONDS", "EVO_ARCHITECT_TOTAL_TIMEOUT_SECONDS"],
        default=max(30.0, float(request_timeout) * max_retries),
    )
    if temperature_override is None:
        temperature = env_float(["ARCHITECT_TEMPERATURE", "EVO_ARCHITECT_TEMPERATURE"], default=0.0)
    else:
        temperature = float(max(0.0, min(2.0, temperature_override)))
    token_limit = max(256, int(max_tokens))
    base_prompt = str(prompt or "")
    started = time.monotonic()
    prefers_responses = "codex" in architect_model.lower()
    # Many codex deployments are responses-only and may reject temperature.
    suppress_temperature = prefers_responses

    for attempt in range(max_retries):
        elapsed = time.monotonic() - started
        remaining = total_timeout - elapsed
        if remaining <= 0:
            raise RuntimeError(
                f"Architect API timed out after {elapsed:.1f}s "
                f"(budget={total_timeout:.1f}s, retries={max_retries})"
            )

        nonce = f"{request_tag or 'architect'}-{attempt}-{uuid.uuid4().hex[:12]}"
        prompt_with_nonce = (
            f"{base_prompt}\n\n"
            f"[request_nonce={nonce}]"
        )
        effective_timeout = max(1.0, min(float(request_timeout), remaining))
        if prefers_responses:
            endpoint_order = ["responses"]
        else:
            endpoint_order = ["chat", "responses"]

        last_exc: Exception | None = None
        endpoint_failures: list[tuple[str, Exception]] = []

        for endpoint in endpoint_order:
            endpoint_no_temp = suppress_temperature
            while True:
                try:
                    if endpoint == "chat":
                        request_kwargs = {
                            "model": architect_model,
                            "messages": [{"role": "user", "content": prompt_with_nonce}],
                            "max_tokens": token_limit,
                            "timeout": effective_timeout,
                        }
                        if not endpoint_no_temp:
                            request_kwargs["temperature"] = temperature
                        response = architect_client.chat.completions.create(**request_kwargs)
                        return response.choices[0].message.content or ""

                    if not hasattr(architect_client, "responses"):
                        raise RuntimeError("Responses API not available on architect client")
                    request_kwargs = {
                        "model": architect_model,
                        "input": prompt_with_nonce,
                        "max_output_tokens": token_limit,
                        "timeout": effective_timeout,
                    }
                    if not endpoint_no_temp:
                        request_kwargs["temperature"] = temperature
                    response = architect_client.responses.create(**request_kwargs)
                    extracted = _extract_responses_text(response)
                    return extracted if extracted else str(response)
                except Exception as exc:
                    error_text = str(exc)
                    if not endpoint_no_temp and _is_unsupported_temperature_error(error_text):
                        endpoint_no_temp = True
                        suppress_temperature = True
                        continue

                    last_exc = exc
                    endpoint_failures.append((endpoint, exc))
                    if endpoint == "chat" and _is_chat_model_incompat_error(error_text):
                        break
                    # Try next endpoint in this same attempt if available.
                    break

        if last_exc is None:
            raise RuntimeError("Architect API failed with unknown error")

        # If chat fails with model incompatibility after trying responses,
        # surface the responses error for clearer diagnosis.
        if _is_chat_model_incompat_error(str(last_exc)) and endpoint_failures:
            for endpoint, exc in endpoint_failures:
                if endpoint == "responses":
                    last_exc = exc
                    break

        error_text = str(last_exc)
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
            raise RuntimeError(f"Architect API failed after {max_retries} attempts: {last_exc}") from last_exc
        time.sleep(1.0 + attempt)
    raise RuntimeError("Architect API failed with unknown error")


def generate_with_repair(
    architect_client,
    architect_model: str,
    architect_prompt: str,
    loader: Union[ProtocolLoader, SandboxProtocolLoader, TDGProtocolLoader],
    max_repair_attempts: int = 2,
    request_tag: str = "",
    architect_temperature: float | None = None,
    repair_temperature: float | None = None,
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
            temperature_override=architect_temperature,
        )
    except Exception as exc:
        return None, str(exc)
    initial_protocol, code, initial_error = _pick_best_candidate(raw, loader)
    if initial_protocol is not None:
        return initial_protocol, code
    error = initial_error

    for attempt in range(max_repair_attempts + 1):
        if error is None:
            protocol, error = loader.load_from_code(code)
            if protocol is not None:
                return protocol, code

        if error is None:
            return None, "Unknown loader error"

        # Local salvage path for common provider formatting artifacts
        # (quoted/escaped code payloads leading to line-1 syntax failures).
        if str(error.stage) == "syntax":
            error_msg = str(error.message or "").lower()
            if "line 1" in error_msg:
                for candidate in _local_syntax_salvage_candidates(code):
                    if candidate == code:
                        continue
                    repaired_protocol, repaired_error = loader.load_from_code(candidate)
                    if repaired_protocol is not None:
                        return repaired_protocol, candidate
                    if repaired_error is not None and str(repaired_error.stage) != "syntax":
                        # Keep the best transformed candidate for the next repair prompt.
                        code = candidate
                        error = repaired_error
                        break

        # Local downgrade repair for repeated TDG contract misses.
        local_contract_note = ""
        if isinstance(loader, TDGProtocolLoader) and str(error.stage) == "contract":
            maybe_fixed, local_contract_note = _attempt_local_tdg_contract_repair(code, str(error.message))
            if maybe_fixed and maybe_fixed != code:
                repaired_protocol, repaired_error = loader.load_from_code(maybe_fixed)
                if repaired_protocol is not None:
                    return repaired_protocol, maybe_fixed
                code = maybe_fixed
                if repaired_error is not None:
                    error = repaired_error

        if attempt >= max_repair_attempts:
            if str(getattr(error, "stage", "")) == "syntax":
                code_head = str(code or "").strip().replace("\n", "\\n")[:180]
                return (
                    None,
                    f"Validation failed after {max_repair_attempts + 1} attempts: {error.message} | code_head={code_head}",
                )
            return None, f"Validation failed after {max_repair_attempts + 1} attempts: {error.message}"

        if isinstance(loader, TDGProtocolLoader):
            contract_hint = ""
            if str(error.stage) == "contract" and "must call self._call_llm" in str(error.message):
                contract_hint = (
                    "Targeted contract fix:\n"
                    "- compile_tests MUST call self._call_llm in executable code path and return test code.\n"
                    "- generate_answer MUST call self._call_llm in executable code path and return its text.\n"
                    "- Do not return constant strings or placeholder text.\n\n"
                )
            local_hint = ""
            if local_contract_note and not local_contract_note.startswith("skip:"):
                local_hint = f"Local contract auto-fix note: {local_contract_note}\n\n"
            repair_prompt = (
                "The generated protocol failed validation. Please fix it.\n\n"
                "Hard constraints:\n"
                "- Output only executable Python code (no markdown, no prose).\n"
                "- Include exactly one class inheriting BaseTDGCompiler.\n"
                "- Keep exact method signatures:\n"
                "  def compile_tests(self, context: str, query: str) -> str\n"
                "  def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str\n"
                "- Do NOT override immutable runtime methods: __init__, run, _call_llm,\n"
                "  _make_oracle_fn, _sanitize_generated_code, _attempt_syntax_repair,\n"
                "  _answer_with_oracle_fallback, _derive_dynamic_call_budget.\n\n"
                f"{contract_hint}"
                f"{local_hint}"
                f"Feedback:\n{error.to_feedback()}\n\n"
                f"Current code:\n```python\n{code}\n```\n\n"
                "Output only corrected Python code."
            )
        else:
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
                temperature_override=repair_temperature,
            )
        except Exception as exc:
            return None, str(exc)
        repaired_protocol, repaired_code, repaired_error = _pick_best_candidate(raw, loader)
        if repaired_protocol is not None:
            return repaired_protocol, repaired_code
        code = repaired_code
        error = repaired_error

    return None, "Unreachable state"
