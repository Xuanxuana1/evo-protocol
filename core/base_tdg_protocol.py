"""Base TDG (Test-Driven Generation) compiler: compile tests -> generate answer -> verify -> repair.

Architecture:
  - compile_tests(context, query) -> str:  LLM generates Python test functions
  - generate_answer(context, query, messages_raw) -> str:  LLM generates NL answer
  - run() is a FIXED deterministic runtime, NOT evolvable by Meta-Agent

The Meta-Agent evolves only compile_tests() and generate_answer().
The execution loop, verification (Python interpreter), and oracle plumbing
are immutable infrastructure.

Key property: worst case (tests fail to compile) degrades to direct inference
(generate_answer only), so TDG is never worse than baseline.

Neural Oracles:
  Test code may call _oracle(prompt, return_type) for semantic/tone checks.
  Logic and control flow remain in deterministic Python.
"""

from __future__ import annotations

import ast
import re
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.base_protocol import ProtocolResult
from core.base_sandbox_protocol import SandboxResult
from core.env_utils import env_float, env_int
from core.token_tracker import TRACKER


class _TDGCodeSanitizer(ast.NodeTransformer):
    """Strip fragile/forbidden oracle placeholders from generated test code."""

    # Builtins that must not appear in generated test functions.
    # locals()/globals() cause fragile introspection; exec/eval/compile are
    # outright dangerous.  Any test_ function that contains one of these is
    # removed wholesale rather than silently rewritten.
    _FORBIDDEN_IN_TESTS: frozenset[str] = frozenset(
        {"locals", "globals", "exec", "eval", "compile"}
    )

    def __init__(self) -> None:
        self.changed = False

    # ------------------------------------------------------------------
    # Function-level: remove _oracle definitions and unsafe test functions
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name == "_oracle":
            self.changed = True
            return None
        # Drop test_ functions that use forbidden builtins.  Rewriting them
        # in-place (e.g. locals() → {}) changes semantics in hard-to-predict
        # ways; deleting the function is safer — remaining tests still run.
        if node.name.startswith("test_") and self._contains_forbidden_builtins(node):
            self.changed = True
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if node.name == "_oracle":
            self.changed = True
            return None
        return self.generic_visit(node)

    def _contains_forbidden_builtins(self, func_node: ast.AST) -> bool:
        for child in ast.walk(func_node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in self._FORBIDDEN_IN_TESTS
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Import-level: remove any attempt to import _oracle as a module.
    # _oracle is injected by the runtime; importing it as a module is always
    # wrong.  We replace with Pass (not None) so that empty try-bodies remain
    # syntactically valid.
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> ast.AST:
        remaining = [a for a in node.names if a.name.split(".", 1)[0] != "_oracle"]
        if len(remaining) == len(node.names):
            return self.generic_visit(node)
        self.changed = True
        if not remaining:
            return ast.Pass()
        node.names = remaining
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        module = node.module or ""
        if module == "_oracle" or module.startswith("_oracle."):
            self.changed = True
            return ast.Pass()
        return self.generic_visit(node)

    # ------------------------------------------------------------------
    # Assignment-level: remove assignments to _oracle
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if any(self._is_oracle_target(target) for target in node.targets):
            self.changed = True
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        if self._is_oracle_target(node.target):
            self.changed = True
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | None:
        if self._is_oracle_target(node.target):
            self.changed = True
            return None
        return self.generic_visit(node)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        if self._is_not_implemented_error(node.exc):
            self.changed = True
            return ast.Pass()
        return self.generic_visit(node)

    def _is_oracle_target(self, target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return target.id == "_oracle"
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._is_oracle_target(item) for item in target.elts)
        return False

    @staticmethod
    def _is_not_implemented_error(exc: ast.AST | None) -> bool:
        if isinstance(exc, ast.Name):
            return exc.id == "NotImplementedError"
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            return exc.func.id == "NotImplementedError"
        return False


class BaseTDGCompiler(ABC):
    """Base interface for all evolved TDG (Test-Driven Generation) compilers.

    The Meta-Agent modifies the internal prompts and generation logic
    of compile_tests() and generate_answer(). Everything else is fixed
    infrastructure that the Meta-Agent cannot touch.

    Two evolved methods:
      compile_tests   -- Generates Python test functions from context+query
      generate_answer -- Generates NL answer from context+query

    The deterministic runtime (run method) handles:
      - Executing test code to validate syntax
      - Running tests against generated answers
      - Repair loop: re-generate answer if tests fail
      - Graceful degradation: if tests fail to compile, return answer as-is
    """

    max_llm_calls_per_task: int = 20

    def __init__(self, llm_client, model_name: str = "gpt-4o-mini") -> None:
        self.llm = llm_client
        self.model = model_name
        self._call_count = 0
        self._task_tokens_used = 0
        self._task_prompt_tokens = 0
        self._task_completion_tokens = 0
        self._oracle_call_count = 0
        self._max_llm_calls_current = int(getattr(self, "max_llm_calls_per_task", 20))
        self.api_timeout_seconds = env_float(
            ["WORKER_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
        self.max_completion_tokens = env_int(
            ["WORKER_MAX_TOKENS", "EVO_WORKER_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=8192,
        )
        self.oracle_max_completion_tokens = env_int(
            ["ORACLE_MAX_TOKENS", "EVO_ORACLE_MAX_TOKENS", "OPENAI_ORACLE_MAX_TOKENS"],
            default=512,
        )
        self.sandbox_timeout_seconds = env_int(
            ["SANDBOX_TIMEOUT_SECONDS", "EVO_SANDBOX_TIMEOUT_SECONDS"],
            default=30,
        )
        self.context_char_limit = env_int(
            ["TDG_CONTEXT_CHAR_LIMIT", "CAS_CONTEXT_CHAR_LIMIT", "EVO_CAS_CONTEXT_CHAR_LIMIT"],
            default=90000,
        )
        self.query_char_limit = env_int(
            ["TDG_QUERY_CHAR_LIMIT", "CAS_QUERY_CHAR_LIMIT", "EVO_CAS_QUERY_CHAR_LIMIT"],
            default=24000,
        )

    # ------------------------------------------------------------------
    # Evolved methods (Meta-Agent mutates these)
    # ------------------------------------------------------------------

    @abstractmethod
    def compile_tests(self, context: str, query: str) -> str:
        """Phase 1: Test compilation.

        Takes natural language context and query and generates Python test
        functions that validate a correct answer.

        Each test function should:
          - Accept a single ``answer`` string parameter
          - Assert properties the correct answer must satisfy
          - Raise AssertionError (or any exception) on failure
          - Be named with ``test_`` prefix

        Test code may call ``_oracle(prompt, return_type)`` for semantic
        checks (e.g., tone, style, factual consistency).

        Returns:
            String of executable Python test code with def test_*() functions.
        """

    @abstractmethod
    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        """Phase 2: Answer generation.

        Takes context + query and generates a natural language answer.
        If messages_raw is provided, reconstruct structured messages for
        multi-turn conversation support.

        Returns:
            Natural language answer string.
        """

    # ------------------------------------------------------------------
    # Fixed deterministic runtime (NOT evolvable)
    # ------------------------------------------------------------------

    def run(self, context: str, query: str, max_retries: int = 2, messages_raw: list = None) -> SandboxResult:
        """Fixed deterministic runtime. The Meta-Agent CANNOT modify this method.

        Pipeline:
          1. compile_tests(context, query) -> test_code
          2. generate_answer(context, query, messages_raw) -> draft answer
          3. Verify: run tests against draft answer
          4. Repair loop: if tests fail, re-generate with feedback
        """

        self._call_count = 0
        self._task_tokens_used = 0
        self._task_prompt_tokens = 0
        self._task_completion_tokens = 0
        self._oracle_call_count = 0
        self._max_llm_calls_current = self._derive_dynamic_call_budget(context=context, query=query)
        trace: list[str] = []
        sandbox_timeout = max(1, int(getattr(self, "sandbox_timeout_seconds", 30)))
        trace.append(f"[Budget] llm_call_limit={self._max_llm_calls_current}")

        # --- Stage 1: Compile tests ---
        raw_test_count = 0
        sanitized_test_count = 0
        sanitized_test_drop_count = 0
        adversarial_passed_tests = 0
        adversarial_total_tests = 0
        adversarial_test_pass_rate = 0.0
        test_code = ""
        test_names: list[str] = []
        compilation_success = False
        try:
            raw_test_code = self.compile_tests(context, query)
            raw_test_count = len(self._extract_test_names(raw_test_code))
            test_code = self._sanitize_generated_code(raw_test_code)
            sanitized_test_count = len(self._extract_test_names(test_code))
            sanitized_test_drop_count = max(0, raw_test_count - sanitized_test_count)
            trace.append(f"[CompileTests] code_len={len(test_code)}")
            if sanitized_test_drop_count > 0:
                trace.append(
                    "[Sanitizer] dropped_tests="
                    f"{sanitized_test_drop_count} raw={raw_test_count} kept={sanitized_test_count}"
                )
        except Exception as exc:
            trace.append(f"[CompileTests] error={str(exc)[:200]}")
            test_code = ""

        if test_code:
            # Validate test code syntax by executing it
            from core.sandbox_executor import execute_sandbox_code

            compile_result = execute_sandbox_code(
                test_code,
                oracle_fn=self._make_oracle_fn(),
                timeout=sandbox_timeout,
            )
            compile_error = str(compile_result.error or "")

            if (not compile_result.success) and self._is_syntax_error(compile_error):
                repaired_tests = self._attempt_syntax_repair(
                    stage="compile_tests",
                    broken_code=test_code,
                    error_text=compile_error,
                )
                if repaired_tests and repaired_tests != test_code:
                    test_code = repaired_tests
                    trace.append(f"[TestRepair] applied=True code_len={len(test_code)}")
                    compile_result = execute_sandbox_code(
                        test_code,
                        oracle_fn=self._make_oracle_fn(),
                        timeout=sandbox_timeout,
                    )
                    compile_error = str(compile_result.error or "")

            compilation_success = compile_result.success
            trace.append(f"[TestCompileExec] success={compilation_success} error={compile_error[:200]}")

            if compilation_success:
                test_names = self._extract_test_names(test_code)
                trace.append(f"[TestNames] found={len(test_names)} names={test_names[:10]}")

        # --- Stage 2: Generate answer ---
        draft = ""
        try:
            draft = self.generate_answer(context, query, messages_raw=messages_raw)
            trace.append(f"[GenerateAnswer] len={len(draft)}")
        except Exception as exc:
            trace.append(f"[GenerateAnswer] error={str(exc)[:200]}")

        if not draft:
            # Fallback: try oracle direct answer
            draft = self._answer_with_oracle_fallback(context=context, query=query)
            if draft:
                trace.append("[Fallback] oracle_direct_answer=True")

        # If no tests compiled or no test functions found, return answer as-is
        if not compilation_success or not test_names:
            trace.append("[NoTests] returning answer without verification")
            return SandboxResult(
                answer=draft,
                confidence=0.3 if draft else 0.0,
                reasoning_trace=trace,
                verification_passed=bool(draft),
                tokens_used=self._task_tokens_used,
                prompt_tokens=self._task_prompt_tokens,
                completion_tokens=self._task_completion_tokens,
                metadata={
                    "stage": "no_tests",
                    "llm_calls": self._call_count,
                    "oracle_calls": self._oracle_call_count,
                    "test_pass_rate": 0.0,
                    "test_results": {},
                    "repair_attempts": 0,
                    "raw_test_count": int(raw_test_count),
                    "sanitized_test_count": int(sanitized_test_count),
                    "sanitized_test_drop_count": int(sanitized_test_drop_count),
                    "adversarial_passed_tests": int(adversarial_passed_tests),
                    "adversarial_total_tests": int(adversarial_total_tests),
                    "adversarial_test_pass_rate": float(adversarial_test_pass_rate),
                },
                sandbox_code=test_code,
                solver_code="",
                execution_output="",
                execution_success=bool(draft),
                compilation_success=False,
            )

        # --- Stage 3: Verify ---
        adversarial_answer = self._build_adversarial_answer(context=context, query=query)
        adversarial_results = self._run_tests(
            test_code,
            adversarial_answer,
            test_names,
            sandbox_timeout,
        )
        adversarial_total_tests = len(adversarial_results)
        adversarial_passed_tests = sum(1 for passed in adversarial_results.values() if passed)
        adversarial_test_pass_rate = (
            adversarial_passed_tests / adversarial_total_tests if adversarial_total_tests > 0 else 0.0
        )
        trace.append(
            "[AdversarialCheck] pass_rate="
            f"{adversarial_test_pass_rate:.2f} "
            f"passed={adversarial_passed_tests}/{adversarial_total_tests}"
        )

        test_results = self._run_tests(test_code, draft, test_names, sandbox_timeout)
        trace.append(f"[Verify] results={test_results}")

        # --- Stage 4: Repair loop ---
        repair_attempts = 0
        for attempt in range(max_retries):
            failed_tests = [name for name, passed in test_results.items() if not passed]
            if not failed_tests:
                break

            repair_attempts += 1
            trace.append(f"[Repair-{attempt}] failed={failed_tests}")

            repaired = self._repair_answer(
                context=context,
                query=query,
                answer=draft,
                test_code=test_code,
                failed_tests=failed_tests,
                messages_raw=messages_raw,
            )
            if repaired and repaired != draft:
                draft = repaired
                trace.append(f"[Repair-{attempt}] new_answer_len={len(draft)}")
                test_results = self._run_tests(test_code, draft, test_names, sandbox_timeout)
                trace.append(f"[ReVerify-{attempt}] results={test_results}")
            else:
                trace.append(f"[Repair-{attempt}] no_change")
                break

        # Compute final metrics
        total_tests = len(test_results)
        passed_tests = sum(1 for v in test_results.values() if v)
        test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        all_passed = total_tests > 0 and passed_tests == total_tests

        trace.append(
            f"[Final] pass_rate={test_pass_rate:.2f} "
            f"passed={passed_tests}/{total_tests} "
            f"repair_attempts={repair_attempts}"
        )

        return SandboxResult(
            answer=draft,
            confidence=max(0.2, test_pass_rate),
            reasoning_trace=trace,
            verification_passed=all_passed,
            tokens_used=self._task_tokens_used,
            prompt_tokens=self._task_prompt_tokens,
            completion_tokens=self._task_completion_tokens,
            metadata={
                "stage": "verified" if all_passed else "partial_pass",
                "llm_calls": self._call_count,
                "oracle_calls": self._oracle_call_count,
                "test_pass_rate": test_pass_rate,
                "test_results": test_results,
                "repair_attempts": repair_attempts,
                "tests_compiled": True,
                "num_tests": total_tests,
                "raw_test_count": int(raw_test_count),
                "sanitized_test_count": int(sanitized_test_count),
                "sanitized_test_drop_count": int(sanitized_test_drop_count),
                "adversarial_passed_tests": int(adversarial_passed_tests),
                "adversarial_total_tests": int(adversarial_total_tests),
                "adversarial_test_pass_rate": float(adversarial_test_pass_rate),
            },
            sandbox_code=test_code,
            solver_code="",
            execution_output=str(test_results),
            execution_success=bool(draft),
            compilation_success=True,
        )

    # ------------------------------------------------------------------
    # Infrastructure helpers (NOT evolvable)
    # ------------------------------------------------------------------

    def _extract_test_names(self, test_code: str) -> list[str]:
        """AST-parse test code for def test_* function names."""

        try:
            tree = ast.parse(test_code)
        except SyntaxError:
            return []

        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.append(node.name)
        return names

    def _run_tests(
        self,
        test_code: str,
        answer: str,
        test_names: list[str],
        timeout: int = 30,
    ) -> dict[str, bool]:
        """Build runner code, execute in sandbox, return pass/fail per test."""

        if not test_names:
            return {}

        runner = self._build_test_runner(test_code, answer, test_names)

        from core.sandbox_executor import execute_sandbox_code

        result = execute_sandbox_code(
            runner,
            oracle_fn=self._make_oracle_fn(),
            timeout=timeout,
        )

        if not result.success or not result.namespace:
            # All tests fail if runner itself crashes
            return {name: False for name in test_names}

        test_results = result.namespace.get("_test_results", {})
        if not isinstance(test_results, dict):
            return {name: False for name in test_names}

        # Ensure all test names present
        return {name: bool(test_results.get(name, False)) for name in test_names}

    def _build_test_runner(self, test_code: str, answer: str, test_names: list[str]) -> str:
        """Generate explicit runner code that avoids banned locals()."""

        runner = test_code + "\n\n"
        runner += f"_answer = {repr(answer)}\n"
        runner += "_test_results = {}\n"
        for name in test_names:
            runner += f"try:\n"
            runner += f"    {name}(_answer)\n"
            runner += f"    _test_results['{name}'] = True\n"
            runner += f"except Exception:\n"
            runner += f"    _test_results['{name}'] = False\n"
        return runner

    def _repair_answer(
        self,
        context: str,
        query: str,
        answer: str,
        test_code: str,
        failed_tests: list[str],
        messages_raw: list = None,
    ) -> str:
        """One LLM call to fix answer based on failed test feedback."""

        if self._call_count >= self._call_budget():
            return answer

        failed_list = "\n".join(f"- {name}" for name in failed_tests)
        # Include context so the repair LLM can fix factual errors (F1/F2)
        context_excerpt = self._prepare_prompt_text(context, self.context_char_limit)
        prompt = (
            "Your previous answer failed some verification tests. "
            "Please revise your answer to pass all tests.\n\n"
            f"Context:\n{context_excerpt}\n\n"
            f"Query:\n{self._prepare_prompt_text(query, self.query_char_limit)}\n\n"
            f"Your previous answer:\n{answer[:8000]}\n\n"
            f"Failed tests:\n{failed_list}\n\n"
            f"Test code (for reference):\n{test_code[:6000]}\n\n"
            "Provide ONLY the revised answer text, no explanations or code blocks."
        )

        try:
            revised = self._call_llm(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return revised.strip() if revised.strip() else answer
        except Exception:
            return answer

    def _derive_dynamic_call_budget(self, context: str, query: str) -> int:
        """Increase call budget for long-context tasks to reduce premature failure."""

        base_budget = max(1, int(getattr(self, "max_llm_calls_per_task", 20)))
        allow_boost = env_int(
            ["TDG_DYNAMIC_BUDGET_BOOST", "CAS_DYNAMIC_BUDGET_BOOST", "EVO_CAS_DYNAMIC_BUDGET_BOOST"],
            default=0,
        )
        if allow_boost <= 0:
            return base_budget
        total_chars = len(str(context or "")) + len(str(query or ""))
        if total_chars >= 160000:
            return min(max(base_budget, 24), 40)
        if total_chars >= 90000:
            return min(max(base_budget, 22), 36)
        if total_chars >= 45000:
            return min(max(base_budget, 20), 32)
        return base_budget

    @staticmethod
    def _build_adversarial_answer(context: str, query: str) -> str:
        """Create an intentionally context-agnostic answer to sanity-check tests."""

        query_head = str(query or "").strip().splitlines()[0][:120]
        context_hint = str(context or "").strip().splitlines()[0][:80]
        return (
            "This answer intentionally ignores the provided context and relies on generic prior knowledge. "
            "It should fail strict context-grounded verification tests. "
            f"Query hint: {query_head}. Context hint ignored: {context_hint}. "
            "Defaulting to common world assumptions even if they conflict with the prompt."
        )

    def _call_budget(self) -> int:
        return max(1, int(getattr(self, "_max_llm_calls_current", self.max_llm_calls_per_task)))

    def _prepare_prompt_text(self, text: str, limit: int) -> str:
        value = str(text or "")
        if limit <= 0:
            return value
        return value[: max(1, int(limit))]

    def _call_llm(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Single LLM call wrapper with token accounting and call budget guard."""

        self._call_count += 1
        budget = self._call_budget()
        if self._call_count > budget:
            raise RuntimeError(
                f"LLM call budget exceeded ({budget} per task)."
            )

        token_limit = max(256, int(self.max_completion_tokens))
        max_retries = env_int(["WORKER_LLM_MAX_RETRIES", "EVO_WORKER_LLM_MAX_RETRIES"], default=2)
        max_retries = max(0, max_retries)
        last_error: Exception | None = None
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=token_limit,
                    timeout=self.api_timeout_seconds,
                )
                break
            except Exception as exc:
                last_error = exc
                adjusted = self._extract_max_tokens_cap(str(exc))
                if adjusted is not None and adjusted < token_limit:
                    token_limit = max(256, adjusted)
                    time.sleep(0.4)
                    continue
                if attempt >= max_retries or not self._is_transient_llm_error(exc):
                    raise
                time.sleep(0.8 + 0.6 * attempt)
        if response is None:
            raise RuntimeError(f"LLM call failed after retries: {last_error}")
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self._task_prompt_tokens += prompt_tokens
            self._task_completion_tokens += completion_tokens
            self._task_tokens_used += prompt_tokens + completion_tokens
            TRACKER.record(self.model, prompt_tokens, completion_tokens)

        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _extract_max_tokens_cap(error_text: str) -> Optional[int]:
        match = re.search(r"supports at most\s+(\d+)\s+completion tokens", str(error_text), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _is_transient_llm_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "temporarily unavailable",
            "gateway timeout",
            "503",
            "504",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _sanitize_generated_code(text: str) -> str:
        """Normalize LLM code output into executable Python."""

        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            return ""

        fenced = re.search(r"```(?:python)?\s*([\s\S]*?)```", value, flags=re.IGNORECASE)
        if fenced:
            value = fenced.group(1).strip()
        elif value.startswith("```"):
            value = value.strip("`").strip()

        lines = value.split("\n")
        if lines and lines[0].strip().lower() == "python":
            lines = lines[1:]

        cleaned: list[str] = []
        for line in lines:
            if re.match(r"^\s*from\s+__future__\s+import\b", line):
                continue
            cleaned.append(line)

        normalized = "\n".join(cleaned).strip()
        if not normalized:
            return ""

        try:
            tree = ast.parse(normalized)
        except SyntaxError:
            return normalized

        sanitizer = _TDGCodeSanitizer()
        tree = sanitizer.visit(tree)
        ast.fix_missing_locations(tree)
        if not sanitizer.changed:
            return normalized

        try:
            return ast.unparse(tree).strip()
        except Exception:
            return normalized

    @staticmethod
    def _is_syntax_error(error_text: str) -> bool:
        message = str(error_text or "").lower()
        markers = (
            "syntax error",
            "invalid syntax",
            "unexpected indent",
            "unexpected eof",
            "unterminated string",
            "eol while scanning",
        )
        return any(marker in message for marker in markers)

    def _attempt_syntax_repair(self, stage: str, broken_code: str, error_text: str) -> str:
        """Use one extra LLM pass to repair syntax-only failures."""

        if self._call_count >= self._call_budget():
            return broken_code

        prompt = (
            "Fix the Python code so it is syntactically valid.\n\n"
            "Hard constraints:\n"
            "- Output executable Python code only (no markdown).\n"
            "- Do not use `from __future__` imports.\n"
            "- Do not define, assign, or override `_oracle`.\n"
            "- Do not use `raise NotImplementedError(...)` placeholders.\n"
            "- Preserve the original intent and variable names where possible.\n\n"
            f"Stage: {stage}\n"
            f"Observed error: {error_text[:400]}\n\n"
            "Broken code:\n"
            f"{broken_code[:12000]}"
        )

        try:
            repaired_raw = self._call_llm(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception:
            return broken_code

        repaired = self._sanitize_generated_code(repaired_raw)
        return repaired if repaired else broken_code

    def _answer_with_oracle_fallback(self, context: str, query: str) -> str:
        """Last-resort direct answer path when everything else fails."""

        if self._call_count >= self._call_budget():
            return ""
        oracle = self._make_oracle_fn()
        context_excerpt = self._prepare_prompt_text(
            str(context or ""),
            min(int(self.context_char_limit), 24000),
        )
        query_excerpt = self._prepare_prompt_text(
            str(query or ""),
            min(int(self.query_char_limit), 8000),
        )
        prompt = (
            "You must answer using ONLY the provided context. "
            "Follow persona/tone/format constraints from all roles if present.\n\n"
            f"Context:\n{context_excerpt}\n\n"
            f"User query:\n{query_excerpt}"
        )
        try:
            return str(oracle(prompt, str)).strip()
        except Exception:
            return ""

    def _make_oracle_fn(self):
        """Create the _oracle() function injected into sandbox namespace.

        Neural Oracles are strictly-typed LLM perception calls. The test
        code can call _oracle(prompt, return_type) where return_type is one
        of: bool, int, float, str.
        """

        def _oracle(prompt: str, return_type: type = str) -> Any:
            self._oracle_call_count += 1
            self._call_count += 1
            budget = self._call_budget()
            if self._call_count > budget:
                raise RuntimeError(f"Oracle call budget exceeded ({budget} per task).")

            type_instruction = {
                bool: "Answer ONLY 'True' or 'False'.",
                int: "Answer ONLY with a single integer number.",
                float: "Answer ONLY with a single decimal number.",
                str: "Provide a complete final response that follows the requested format and constraints.",
            }.get(return_type, "Answer briefly.")

            messages = [
                {"role": "user", "content": f"{prompt}\n\n{type_instruction}"},
            ]
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=min(self.max_completion_tokens, max(64, int(self.oracle_max_completion_tokens))),
                timeout=self.api_timeout_seconds,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                self._task_prompt_tokens += pt
                self._task_completion_tokens += ct
                self._task_tokens_used += pt + ct
                TRACKER.record(self.model, pt, ct)

            raw = (response.choices[0].message.content or "").strip()

            if return_type is bool:
                return raw.lower().startswith("true")
            if return_type is int:
                try:
                    match = re.search(r"-?\d+", raw)
                    return int(match.group()) if match else 0
                except (ValueError, AttributeError):
                    return 0
            if return_type is float:
                try:
                    match = re.search(r"-?\d+\.?\d*", raw)
                    return float(match.group()) if match else 0.0
                except (ValueError, AttributeError):
                    return 0.0
            return raw

        return _oracle
