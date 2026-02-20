"""Failure mode classification and structured feedback for evolution."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from core.env_utils import env_float, env_int

@dataclass
class FailureFeedback:
    """Structured failure packet consumed by the mutation prompt."""

    mode: str
    confidence: float
    root_cause: str
    repair_actions: list[str]
    unsatisfied_rubrics: list[str]
    judge_rationale: str
    source: str = "heuristic"
    raw_response: Optional[str] = None
    stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert feedback object to plain dictionary."""

        payload = asdict(self)
        payload["confidence"] = float(max(0.0, min(1.0, self.confidence)))
        return payload


def _normalize_mode(mode: str | None, fallback: str = "F3") -> str:
    candidate = str(mode or fallback).strip().upper()
    if candidate not in {"F1", "F2", "F3", "F4"}:
        return fallback
    return candidate


def _short_text(text: Any, limit: int = 300) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _extract_unsatisfied_rubrics(rubrics: list[Any], status_list: Any) -> list[str]:
    """Map judge requirement status to rubric text when possible."""

    if not isinstance(status_list, list):
        return []

    unsatisfied: list[str] = []
    for idx, status in enumerate(status_list):
        normalized = str(status).strip().lower()
        if normalized in {"no", "false", "0", "fail", "failed"}:
            if idx < len(rubrics):
                unsatisfied.append(_short_text(rubrics[idx], limit=220))
            else:
                unsatisfied.append(f"Requirement #{idx + 1} not satisfied")
    return unsatisfied


def _extract_judge_rationale(eval_detail: Any) -> str:
    """Extract concise rationale from judge result payload."""

    if not isinstance(eval_detail, dict):
        return ""

    rationale = eval_detail.get("Grading Rationale") or eval_detail.get("reason") or ""
    return _short_text(rationale, limit=400)


def _extract_json_from_text(raw: str) -> dict[str, Any] | None:
    """Robustly parse a JSON object from plain text or fenced markdown."""

    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _heuristic_feedback(
    context: str,
    query: str,
    model_answer: str,
    default_mode: str,
    unsatisfied_rubrics: list[str],
    judge_rationale: str,
    verification_passed: Optional[bool],
) -> FailureFeedback:
    """Fallback classifier when LLM classifier is unavailable."""

    mode = _normalize_mode(default_mode)
    answer = str(model_answer or "").strip()

    if not answer:
        root_cause = "Protocol produced empty output or timed out; reasoning did not complete."
        actions = [
            "Add runtime guards and fallback answer path.",
            "Reduce per-task complexity and LLM call count.",
        ]
        return FailureFeedback(
            mode=mode,
            confidence=0.85,
            root_cause=root_cause,
            repair_actions=actions,
            unsatisfied_rubrics=unsatisfied_rubrics,
            judge_rationale=judge_rationale,
            source="heuristic",
        )

    if verification_passed is False:
        root_cause = "Candidate answer failed protocol verification; likely context-faithfulness issue."
        actions = [
            "Strengthen claim-to-context verification before final output.",
            "Force citation of supporting context spans in reasoning.",
        ]
        return FailureFeedback(
            mode=mode,
            confidence=0.7,
            root_cause=root_cause,
            repair_actions=actions,
            unsatisfied_rubrics=unsatisfied_rubrics,
            judge_rationale=judge_rationale,
            source="heuristic",
        )

    root_cause = "Answer does not satisfy all rubric constraints based on judge feedback."
    actions = [
        "Improve perception granularity for long context sections.",
        "Add intermediate reasoning checks before final synthesis.",
    ]
    if not judge_rationale and not unsatisfied_rubrics:
        actions.append("Collect richer failure traces for targeted mutation.")

    return FailureFeedback(
        mode=mode,
        confidence=0.55,
        root_cause=root_cause,
        repair_actions=actions,
        unsatisfied_rubrics=unsatisfied_rubrics,
        judge_rationale=judge_rationale,
        source="heuristic",
    )


def _infer_default_mode(
    context: str,
    query: str,
    model_answer: str,
    judge_rationale: str,
    unsatisfied_rubrics: list[str],
    verification_passed: Optional[bool],
    hint_mode: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Infer initial failure mode from observed behavior instead of task category.

    When CaS metadata is present (compilation_failed, execution_traceback),
    also infers the failure stage.
    """

    hint = _normalize_mode(hint_mode, fallback="F3") if hint_mode else "F3"
    answer = str(model_answer or "").strip()

    # CaS-specific: if compilation failed, the sandbox never existed
    if metadata and metadata.get("compilation_failed"):
        return "F2"

    # CaS-specific: execution traceback present -> code generation issue
    if metadata and metadata.get("execution_traceback"):
        return "F3"

    if not answer:
        # Empty output usually means execution/retrieval collapse rather than category-specific failure.
        return "F2" if (len(context) + len(query)) > 5000 else "F3"

    if verification_passed is False:
        return "F1"

    signals = " ".join(
        [
            str(judge_rationale or ""),
            " ".join(str(item) for item in unsatisfied_rubrics),
        ]
    ).lower()

    retrieval_markers = (
        "retrieve",
        "retrieval",
        "not found",
        "did not mention",
        "missing evidence",
        "failed to extract",
        "not in the response",
        "context was available",
        "appendix",
    )
    if any(marker in signals for marker in retrieval_markers):
        return "F2"

    instruction_markers = (
        "did not follow",
        "failed to follow",
        "instruction",
        "format",
        "template",
        "constraint",
        "policy",
        "required",
        "must",
        "prior knowledge",
        "general knowledge",
        "ignored",
    )
    if any(marker in signals for marker in instruction_markers):
        return "F1"

    induction_markers = (
        "infer",
        "inference",
        "pattern",
        "hypothesis",
        "generalize",
        "extrapolat",
        "simulation",
        "rule from examples",
    )
    if any(marker in signals for marker in induction_markers):
        return "F4"

    return hint if hint in {"F1", "F2", "F3", "F4"} else "F3"


def _infer_cas_stage(metadata: dict | None) -> str:
    """Infer CaS failure stage from metadata fields."""

    if not metadata:
        return ""
    if metadata.get("compilation_failed"):
        return "compile"
    if metadata.get("execution_traceback"):
        exec_output = str(metadata.get("execution_output", ""))
        if "NameError" in exec_output or "AttributeError" in exec_output:
            return "translate"
        return "execute"
    return "judge"


def classify_failure_mode(
    context: str,
    query: str,
    model_answer: str,
    correct_answer: str,
    llm_client,
    model: str = "gpt-4o",
) -> dict:
    """Backward-compatible wrapper for mode-only classification."""

    feedback = build_failure_feedback(
        record={
            "context": context,
            "query": query,
            "model_output": model_answer,
            "eval_detail": {"expected_answer": correct_answer},
            "metadata": {},
            "verification_passed": None,
            "rubrics": [],
        },
        llm_client=llm_client,
        model=model,
    )
    return {
        "mode": feedback["mode"],
        "confidence": feedback["confidence"],
        "explanation": feedback["root_cause"],
    }


def build_failure_feedback(
    record: Any,
    llm_client=None,
    model: str = "gpt-4o",
    max_context_chars: int = 1200,
    max_query_chars: int = 400,
) -> dict[str, Any]:
    """Build a structured failure packet for mutation feedback."""

    if hasattr(record, "context"):
        context = str(getattr(record, "context", ""))
        query = str(getattr(record, "query", ""))
        answer = str(getattr(record, "model_output", ""))
        eval_detail = getattr(record, "eval_detail", {})
        metadata = getattr(record, "metadata", {})
        verification_passed = getattr(record, "verification_passed", None)
        rubrics = getattr(record, "rubrics", [])
    else:
        payload = dict(record)
        context = str(payload.get("context", ""))
        query = str(payload.get("query", ""))
        answer = str(payload.get("model_output", ""))
        eval_detail = payload.get("eval_detail", {})
        metadata = payload.get("metadata", {})
        verification_passed = payload.get("verification_passed", None)
        rubrics = payload.get("rubrics", [])

    judge_rationale = _extract_judge_rationale(eval_detail)
    status_list = eval_detail.get("List of Requirement Satisfaction Status") if isinstance(eval_detail, dict) else None
    unsatisfied_rubrics = _extract_unsatisfied_rubrics(rubrics, status_list)
    hint_mode = _normalize_mode(str(metadata.get("gravity_type", "F3")), fallback="F3") if metadata else None
    default_mode = _infer_default_mode(
        context=context,
        query=query,
        model_answer=answer,
        judge_rationale=judge_rationale,
        unsatisfied_rubrics=unsatisfied_rubrics,
        verification_passed=verification_passed,
        hint_mode=hint_mode,
        metadata=metadata,
    )

    cas_stage = _infer_cas_stage(metadata)

    heuristic = _heuristic_feedback(
        context=context,
        query=query,
        model_answer=answer,
        default_mode=default_mode,
        unsatisfied_rubrics=unsatisfied_rubrics,
        judge_rationale=judge_rationale,
        verification_passed=verification_passed,
    )
    heuristic.stage = cas_stage

    if llm_client is None or not answer.strip():
        return heuristic.to_dict()

    cas_context = ""
    if metadata and (metadata.get("compilation_failed") or metadata.get("execution_traceback") or metadata.get("sandbox_code")):
        cas_context = (
            "\n\nCaS (Context-as-Sandbox) failure context:\n"
            f"- Compilation failed: {metadata.get('compilation_failed', False)}\n"
            f"- Execution traceback: {_short_text(metadata.get('execution_traceback', ''), 300)}\n"
            f"- Sandbox code snippet: {_short_text(metadata.get('sandbox_code', ''), 300)}\n"
            "When sandbox compilation or execution failed, consider stage-specific causes:\n"
            "- compile: LLM produced invalid Python for context encoding\n"
            "- translate: LLM produced code referencing wrong namespace keys\n"
            "- execute: runtime error in query code (type errors, missing data)\n"
        )

    tdg_context = ""
    if metadata and "test_pass_rate" in metadata:
        tdg_context = (
            "\n\nTDG (Test-Driven Generation) context:\n"
            f"- Test pass rate: {metadata.get('test_pass_rate', 0.0)}\n"
            f"- Tests compiled: {metadata.get('tests_compiled', False)}\n"
            f"- Repair attempts: {metadata.get('repair_attempts', 0)}\n"
            "When test_pass_rate < 1.0, consider whether:\n"
            "- Tests are too strict (rejecting valid answers)\n"
            "- Answer generation missed key context facts\n"
            "- Repair loop failed to fix the identified issues\n"
        )

    prompt = (
        "You are a failure-analysis model for context-learning protocols.\n"
        "Classify the failure into one mode: F1/F2/F3/F4.\n"
        "Return JSON only with keys: mode, confidence, root_cause, repair_actions.\n"
        "repair_actions must be an array of 2-4 short, concrete actions.\n\n"
        "Failure mode definitions:\n"
        "- F1 Parametric Override: answer follows prior knowledge and ignores context.\n"
        "- F2 Context Navigation Failure: key evidence exists but was not retrieved.\n"
        "- F3 Reasoning Breakdown: evidence exists but multi-step reasoning collapses.\n"
        "- F4 Induction Failure: failed to infer rules/patterns from examples.\n\n"
        f"Context excerpt:\n{_short_text(context, max_context_chars)}\n\n"
        f"Query excerpt:\n{_short_text(query, max_query_chars)}\n\n"
        f"Model answer excerpt:\n{_short_text(answer, 600)}\n\n"
        f"Unsatisfied rubrics:\n{json.dumps(unsatisfied_rubrics, ensure_ascii=False)}\n\n"
        f"Judge rationale:\n{judge_rationale}\n\n"
        f"Default mode hint: {default_mode}\n"
        f"{cas_context}"
        f"{tdg_context}"
    )

    try:
        request_timeout = env_float(
            [
                "FAILURE_CLASSIFIER_API_TIMEOUT_SECONDS",
                "JUDGE_API_TIMEOUT_SECONDS",
                "OPENAI_API_TIMEOUT_SECONDS",
                "OPENAI_API_TIMEOUT",
                "API_TIMEOUT_SECONDS",
            ],
            default=90.0,
        )
        max_tokens = env_int(
            ["FAILURE_CLASSIFIER_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=65536,
        )
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _extract_json_from_text(raw)
        if not isinstance(parsed, dict):
            feedback = heuristic
            feedback.raw_response = _short_text(raw, 500)
            return feedback.to_dict()

        mode = _normalize_mode(parsed.get("mode"), fallback=default_mode)
        confidence = parsed.get("confidence", heuristic.confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = heuristic.confidence

        actions = parsed.get("repair_actions", [])
        if not isinstance(actions, list):
            actions = []
        actions = [_short_text(action, 160) for action in actions if str(action).strip()]
        if not actions:
            actions = heuristic.repair_actions

        feedback = FailureFeedback(
            mode=mode,
            confidence=confidence,
            root_cause=_short_text(parsed.get("root_cause", ""), 400) or heuristic.root_cause,
            repair_actions=actions[:4],
            unsatisfied_rubrics=unsatisfied_rubrics,
            judge_rationale=judge_rationale,
            source="llm",
            raw_response=_short_text(raw, 500),
            stage=cas_stage,
        )
        return feedback.to_dict()
    except Exception as exc:
        feedback = heuristic
        feedback.raw_response = _short_text(str(exc), 500)
        return feedback.to_dict()
