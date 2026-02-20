"""Base CaS compiler: 2-method interface for evolved neuro-symbolic compilers.

Architecture:
  - compile_sandbox(context) -> str:  LLM generates Python environment code
  - generate_solver(query, schema) -> str:  LLM generates Python solver code
  - run() is a FIXED deterministic runtime, NOT evolvable by Meta-Agent

The Meta-Agent evolves only compile_sandbox() and generate_solver().
The execution loop, verification (Python interpreter), and oracle plumbing
are immutable infrastructure.

Neural Oracles:
  Sandbox code may define methods that call _oracle(prompt, return_type)
  for strictly-typed perception tasks (e.g., sentiment -> bool).
  Logic and control flow remain in deterministic Python.
"""

from __future__ import annotations

import ast
import traceback
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from core.base_protocol import ProtocolResult
from core.env_utils import env_float, env_int
from core.token_tracker import TRACKER


class _SandboxCodeSanitizer(ast.NodeTransformer):
    """Strip fragile/forbidden oracle placeholders from generated code."""

    def __init__(self) -> None:
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name == "_oracle":
            self.changed = True
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if node.name == "_oracle":
            self.changed = True
            return None
        return self.generic_visit(node)

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


@dataclass
class SandboxEnvironment:
    """Artifacts from compiling natural-language context into Python code."""

    source_code: str = ""
    namespace: dict[str, Any] = field(default_factory=dict)
    schema_description: str = ""
    data_structures: list[str] = field(default_factory=list)
    compile_error: str = ""


@dataclass
class SandboxResult(ProtocolResult):
    """Extended result carrying sandbox execution artifacts."""

    sandbox_code: str = ""
    solver_code: str = ""
    execution_output: str = ""
    execution_success: bool = False
    compilation_success: bool = False


class BaseCaSCompiler(ABC):
    """Base interface for all evolved Neuro-Symbolic Compilers.

    The Meta-Agent modifies the internal prompts and code-generation logic
    of compile_sandbox() and generate_solver(). Everything else is fixed
    infrastructure that the Meta-Agent cannot touch.

    Two evolved methods:
      compile_sandbox  -- Representation: defeats Context Navigation Failure
      generate_solver  -- Action: defeats Parametric Override and Reasoning Breakdown

    The deterministic runtime (run method) handles:
      - Executing env_code + solver_code together
      - Injecting the _oracle() function for Neural Oracles
      - Retries on solver failures (not compile failures)
      - Result extraction from FINAL_ANSWER variable
    """

    max_llm_calls_per_task: int = 10

    def __init__(self, llm_client, model_name: str = "gpt-4o-mini") -> None:
        self.llm = llm_client
        self.model = model_name
        self._call_count = 0
        self._task_tokens_used = 0
        self._oracle_call_count = 0
        self.api_timeout_seconds = env_float(
            ["WORKER_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
        self.max_completion_tokens = env_int(
            ["WORKER_MAX_TOKENS", "EVO_WORKER_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=65536,
        )
        self.sandbox_timeout_seconds = env_int(
            ["SANDBOX_TIMEOUT_SECONDS", "EVO_SANDBOX_TIMEOUT_SECONDS"],
            default=30,
        )

    # ------------------------------------------------------------------
    # Evolved methods (Meta-Agent mutates these)
    # ------------------------------------------------------------------

    @abstractmethod
    def compile_sandbox(self, context: str) -> str:
        """Phase 1: Representation.

        Takes natural language context and translates it into strictly typed
        Python code (e.g., Pydantic classes, Graph initializations, FSMs).
        This isolates the rules from the LLM's parametric priors.

        The compiled code may define Neural Oracle methods that call
        ``_oracle(prompt, return_type)`` for bounded perception tasks.

        Returns:
            String of executable Python environment code.
        """

    @abstractmethod
    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        """Phase 2: Action.

        Takes the user query and the compiled sandbox schema, and generates
        a Python script that imports/uses the sandbox objects to find the answer.

        The generated code must store its final answer in ``FINAL_ANSWER``.

        Returns:
            String of executable Python solver code.
        """

    # ------------------------------------------------------------------
    # Fixed deterministic runtime (NOT evolvable)
    # ------------------------------------------------------------------

    def run(self, context: str, query: str, max_retries: int = 2) -> SandboxResult:
        """Fixed deterministic runtime. The Meta-Agent CANNOT modify this method.

        Pipeline:
          1. compile_sandbox(context) -> env_code
          2. Execute env_code to validate compilation
          3. generate_solver(query, env_code) -> solver_code
          4. Execute env_code + solver_code together
          5. Extract FINAL_ANSWER from namespace
        """

        self._call_count = 0
        self._task_tokens_used = 0
        self._oracle_call_count = 0
        trace: list[str] = []
        sandbox_timeout = max(1, int(getattr(self, "sandbox_timeout_seconds", 30)))

        # --- Stage 1: Compile context into sandbox code ---
        env_code = self._sanitize_generated_code(self.compile_sandbox(context))
        trace.append(f"[Compile] code_len={len(env_code)}")

        # Validate compilation by executing env_code alone
        from core.sandbox_executor import execute_sandbox_code

        compile_result = execute_sandbox_code(
            env_code,
            oracle_fn=self._make_oracle_fn(),
            timeout=sandbox_timeout,
        )

        compile_error = str(compile_result.error or "")
        if (not compile_result.success) and self._is_syntax_error(compile_error):
            repaired_env = self._attempt_syntax_repair(
                stage="compile",
                broken_code=env_code,
                error_text=compile_error,
            )
            if repaired_env and repaired_env != env_code:
                env_code = repaired_env
                trace.append(f"[CompileRepair] applied=True code_len={len(env_code)}")
                compile_result = execute_sandbox_code(
                    env_code,
                    oracle_fn=self._make_oracle_fn(),
                    timeout=sandbox_timeout,
                )
                compile_error = str(compile_result.error or "")

        compilation_success = compile_result.success
        trace.append(
            f"[CompileExec] success={compilation_success} "
            f"keys={sorted((compile_result.namespace or {}).keys())[:20]} "
            f"error={str(compile_result.error)[:200]}"
        )

        if not compilation_success:
            return SandboxResult(
                answer="",
                confidence=0.0,
                reasoning_trace=trace,
                verification_passed=False,
                tokens_used=self._task_tokens_used,
                metadata={"stage": "compile", "llm_calls": self._call_count},
                sandbox_code=env_code,
                solver_code="",
                execution_output=str(compile_result.error),
                execution_success=False,
                compilation_success=False,
            )

        # --- Stage 2 & 3: Generate solver + Execute (with retries) ---
        feedback = ""
        last_solver_code = ""
        last_output = ""
        for attempt in range(max_retries + 1):
            query_with_feedback = query
            if feedback:
                query_with_feedback = f"{query}\n\n[Retry feedback] {feedback}"

            solver_code = self._sanitize_generated_code(
                self.generate_solver(query_with_feedback, env_code)
            )
            last_solver_code = solver_code
            trace.append(f"[Solver-{attempt}] code_len={len(solver_code)}")

            # Deterministic execution: env_code + solver_code together
            exec_result = execute_sandbox_code(
                env_code + "\n" + solver_code,
                oracle_fn=self._make_oracle_fn(),
                timeout=sandbox_timeout,
            )

            last_output = str(exec_result.output)
            success = exec_result.success
            answer = ""
            if success and exec_result.namespace:
                answer = str(exec_result.namespace.get("FINAL_ANSWER", ""))

            trace.append(
                f"[Execute-{attempt}] success={success} "
                f"answer={answer[:120]!r} "
                f"error={str(exec_result.error)[:200]}"
            )

            if success and answer:
                return SandboxResult(
                    answer=answer,
                    confidence=max(0.2, 1.0 - 0.2 * attempt),
                    reasoning_trace=trace,
                    verification_passed=True,
                    tokens_used=self._task_tokens_used,
                    metadata={
                        "stage": "execute",
                        "attempts": attempt + 1,
                        "llm_calls": self._call_count,
                        "oracle_calls": self._oracle_call_count,
                    },
                    sandbox_code=env_code,
                    solver_code=solver_code,
                    execution_output=last_output,
                    execution_success=True,
                    compilation_success=True,
                )

            # Build feedback for retry
            error_msg = str(exec_result.error or "")
            if error_msg:
                feedback = f"Previous solver code failed: {error_msg[:400]}"
            elif not answer:
                feedback = "Previous solver code did not set FINAL_ANSWER."
            else:
                feedback = "Previous attempt produced empty answer."

        trace.append("[Runtime] all_retries_exhausted -> output_blocked")
        return SandboxResult(
            answer="",
            confidence=0.0,
            reasoning_trace=trace,
            verification_passed=False,
            tokens_used=self._task_tokens_used,
            metadata={
                "stage": "execute",
                "attempts": max_retries + 1,
                "llm_calls": self._call_count,
                "oracle_calls": self._oracle_call_count,
                "output_blocked": True,
            },
            sandbox_code=env_code,
            solver_code=last_solver_code,
            execution_output=last_output,
            execution_success=False,
            compilation_success=True,
        )

    # ------------------------------------------------------------------
    # Infrastructure helpers (NOT evolvable)
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        """Single LLM call wrapper with token accounting and call budget guard.

        Used by evolved methods (compile_sandbox, generate_solver) to ask the
        LLM to generate code.
        """

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

    @staticmethod
    def _sanitize_generated_code(text: str) -> str:
        """Normalize LLM code output into executable sandbox Python."""

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

        sanitizer = _SandboxCodeSanitizer()
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

        if self._call_count >= self.max_llm_calls_per_task:
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

    def _make_oracle_fn(self):
        """Create the _oracle() function injected into sandbox namespace.

        Neural Oracles are strictly-typed LLM perception calls. The sandbox
        code can call _oracle(prompt, return_type) where return_type is one
        of: bool, int, float, str. This keeps perception neural but logic
        deterministic.
        """

        def _oracle(prompt: str, return_type: type = str) -> Any:
            self._oracle_call_count += 1
            self._call_count += 1
            if self._call_count > self.max_llm_calls_per_task:
                raise RuntimeError("Oracle call budget exceeded.")

            type_instruction = {
                bool: "Answer ONLY 'True' or 'False'.",
                int: "Answer ONLY with a single integer number.",
                float: "Answer ONLY with a single decimal number.",
                str: "Answer with a brief, direct response (one sentence max).",
            }.get(return_type, "Answer briefly.")

            messages = [
                {"role": "user", "content": f"{prompt}\n\n{type_instruction}"},
            ]
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=64,
                timeout=self.api_timeout_seconds,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                self._task_tokens_used += pt + ct
                TRACKER.record(self.model, pt, ct)

            raw = (response.choices[0].message.content or "").strip()

            # Strictly-typed parsing
            if return_type is bool:
                return raw.lower().startswith("true")
            if return_type is int:
                try:
                    import re
                    match = re.search(r"-?\d+", raw)
                    return int(match.group()) if match else 0
                except (ValueError, AttributeError):
                    return 0
            if return_type is float:
                try:
                    import re
                    match = re.search(r"-?\d+\.?\d*", raw)
                    return float(match.group()) if match else 0.0
                except (ValueError, AttributeError):
                    return 0.0
            return raw

        return _oracle


# Backward compatibility alias
BaseSandboxProtocol = BaseCaSCompiler
