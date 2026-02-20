"""Prompt construction and protocol mutation helpers for the meta-architect."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Union

from core.base_protocol import BaseProtocol
from core.protocol_loader import ProtocolLoader, SandboxProtocolLoader
from core.self_repair import generate_with_repair


@dataclass
class ParentPerformance:
    """Compact parent protocol scorecard."""

    overall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    f4: float = 0.0
    compilation_success_rate: float = 1.0
    execution_success_rate: float = 1.0


class MetaArchitect:
    """Generate mutated protocol code from parent code and failure traces."""

    def __init__(self, client, architect_model: str = "gpt-4o", mode: str = "cas") -> None:
        self.client = client
        self.model = architect_model
        self.mode = mode

    def build_prompt(
        self,
        generation: int,
        parent_code: str,
        parent_performance: ParentPerformance,
        failure_examples: list[dict[str, Any]],
    ) -> str:
        """Build architect prompt grounded on observed failures."""

        if self.mode == "cas":
            return self._build_cas_prompt(generation, parent_code, parent_performance, failure_examples)
        return self._build_legacy_prompt(generation, parent_code, parent_performance, failure_examples)

    def mutate(
        self,
        loader: Union[ProtocolLoader, SandboxProtocolLoader],
        generation: int,
        parent_code: str,
        parent_performance: ParentPerformance,
        failure_examples: list[dict[str, Any]],
        max_repair_attempts: int = 2,
    ) -> tuple[BaseProtocol | None, str]:
        """Generate and validate a child protocol using self-repair loop."""

        prompt = self.build_prompt(
            generation=generation,
            parent_code=parent_code,
            parent_performance=parent_performance,
            failure_examples=failure_examples,
        )
        return generate_with_repair(
            architect_client=self.client,
            architect_model=self.model,
            architect_prompt=prompt,
            loader=loader,
            max_repair_attempts=max_repair_attempts,
        )

    # ------------------------------------------------------------------
    # CaS prompt (Context-as-Sandbox)
    # ------------------------------------------------------------------

    def _build_cas_prompt(
        self,
        generation: int,
        parent_code: str,
        parent_performance: ParentPerformance,
        failure_examples: list[dict[str, Any]],
    ) -> str:
        failures_text = self._format_failures(failure_examples)

        return textwrap.dedent(
            f"""
            You are an expert AI systems architect designing Context-as-Sandbox (CaS) compilers.

            PARADIGM: Context-as-Sandbox (CaS) with 2-Method Co-Evolution
            CaS compiles context into executable Python objects (the "sandbox"),
            then answers queries by running Python code against those objects.
            The execution is DETERMINISTIC -- the Python interpreter is the verifier.

            Pipeline (FIXED runtime -- you do NOT write this):
              env_code    = compiler.compile_sandbox(context)       # Phase 1: Representation
              solver_code = compiler.generate_solver(query, env_code)  # Phase 2: Action
              exec(env_code + solver_code)                          # DETERMINISTIC
              answer      = FINAL_ANSWER                            # Extracted from namespace

            YOUR TASK: Create one class inheriting from BaseCaSCompiler.

            Required methods (ONLY these two):

            1) compile_sandbox(self, context: str) -> str
               - Use self._call_llm() to ask the LLM to generate Python code
               - The generated code creates strictly typed objects from context:
                 Pydantic models, dataclasses, networkx graphs, dicts, enums, FSMs
               - Add assertions/validators that enforce context-specific values
                 (e.g., assert banana.color == "purple")
               - May define Neural Oracle methods using _oracle(prompt, return_type)
                 for semantic perception that resists symbolic extraction
               - Return the generated code as a string

            2) generate_solver(self, query: str, sandbox_schema: str) -> str
               - Use self._call_llm() to generate Python code that queries the sandbox
               - sandbox_schema is the env_code string from compile_sandbox
               - The code must store its final answer in FINAL_ANSWER variable
               - All logic (loops, conditionals, math) stays in Python
               - For semantic questions, use _oracle(prompt, return_type) as a sensor
               - Return the generated code as a string

            IMPORTANT: Do NOT implement run() or any execution logic.
            The runtime is fixed infrastructure that you cannot modify.

            Neural Oracles -- Bridging the Semantic-Symbolic Gap:
            When context contains nuance that resists symbolic extraction (tone,
            implication, vague language), sandbox code can define perception methods:
              _oracle(prompt, bool)   -> strictly True/False
              _oracle(prompt, int)    -> integer
              _oracle(prompt, float)  -> decimal
              _oracle(prompt, str)    -> brief text
            The LLM is used ONLY as a perception sensor, never as a reasoner.
            All control flow remains in deterministic Python.

            Example Neural Oracle in sandbox code:
              class Contract:
                  def __init__(self, text):
                      self.text = text
                      self.standard_days = 30
                  def is_voided_by_behavior(self):
                      return _oracle(f"Does this imply refusal? {{self.text}}", bool)

            Compilation strategy guidance per context type:
            - Factual/override-prone: Pydantic models with field_validator + assert
            - Spatial/relational: networkx.Graph with typed edges
            - Temporal/sequential: ordered lists, enums, state machines
            - Rule-based: dicts with callable rule functions
            - Tabular: list[dict] or list[dataclass]
            - Nuanced/semantic: Neural Oracle methods for perception

            Constraints:
            1) Import from core.base_sandbox_protocol: BaseCaSCompiler
            2) Use only allowed libraries (re, json, math, collections, itertools,
               functools, typing, dataclasses, copy, random, statistics, string,
               operator, abc, enum, datetime, pydantic, networkx).
            3) No file I/O, no external APIs, no network calls.
            4) Keep total LLM calls within 10 per task.
            5) Do NOT use while-loops. Use bounded for-loops.
            6) Never define, assign, or override `_oracle`; runtime injects it.
            7) Never use `raise NotImplementedError(...)` placeholders.
            8) Return only executable Python code.

            Current generation: {generation}

            Parent performance:
            - Overall accuracy: {parent_performance.overall:.1%}
            - Compilation success rate: {parent_performance.compilation_success_rate:.1%}
            - Execution success rate: {parent_performance.execution_success_rate:.1%}
            - F1 (Parametric Override): {parent_performance.f1:.1%}
            - F2 (Context Navigation): {parent_performance.f2:.1%}
            - F3 (Reasoning Breakdown): {parent_performance.f3:.1%}
            - F4 (Induction Failure): {parent_performance.f4:.1%}

            Parent compiler:
            ```python
            {parent_code}
            ```

            Failure examples:
            {failures_text}

            Improvement strategy by failure mode:
            - F1 (Parametric Override): Add assertions/Pydantic validators enforcing
              context values. _oracle() for semantic checks blocks parametric priors.
            - F2 (Context Navigation): Compile ALL facts into namespace objects.
              Missing data = NameError at runtime. Use _oracle for nuanced facts.
            - F3 (Reasoning Breakdown): Move multi-step logic to Python loops/functions
              in generate_solver. Let Python handle the reasoning chain.
            - F4 (Induction Failure): Create lookup tables and callable rule functions
              in compile_sandbox. Pattern application becomes deterministic code.

            Improvement by failure stage:
            - Compile errors: Simpler data structures, try/except in prompts, better
              code generation instructions to the LLM.
            - Execute errors: Fix namespace references, check variable existence in
              solver code, ensure types match between sandbox and solver.
            - Wrong answers: Richer sandbox with more complete fact encoding, add
              _oracle for nuanced facts that were missed.

            Output only complete Python code.
            """
        ).strip()

    # ------------------------------------------------------------------
    # Legacy prompt (original Evo-Protocol)
    # ------------------------------------------------------------------

    def _build_legacy_prompt(
        self,
        generation: int,
        parent_code: str,
        parent_performance: ParentPerformance,
        failure_examples: list[dict[str, Any]],
    ) -> str:
        failures_text = self._format_failures(failure_examples)

        return textwrap.dedent(
            f"""
            You are an expert AI systems architect.
            Create Python code for one class inheriting from BaseProtocol.

            Constraints:
            1) Implement perception(self, context) -> dict
            2) Implement cognition(self, query, perceived_info) -> str
            3) Implement verification(self, answer, context) -> bool
            4) Use only standard Python libraries.
            5) No file I/O, no external APIs, no network calls.
            6) Keep total LLM calls within 10 per task.
            7) Do NOT use while-loops. Use bounded for-loops with explicit max iterations.
            8) If chunking text, ensure cursor/index updates are strictly monotonic and cannot stall.
            9) Return only executable Python code.

            Current generation: {generation}

            Parent performance:
            - Overall: {parent_performance.overall:.1%}
            - F1: {parent_performance.f1:.1%}
            - F2: {parent_performance.f2:.1%}
            - F3: {parent_performance.f3:.1%}
            - F4: {parent_performance.f4:.1%}

            Parent protocol:
            ```python
            {parent_code}
            ```

            Failure examples:
            {failures_text}

            Improvement strategy:
            - F1: enforce context-faithful verification, reject unsupported claims.
            - F2: chunk and index context before reasoning.
            - F3: use explicit intermediate steps and consistency checks.
            - F4: propose and test hypotheses against provided evidence.
            - Runtime safety: avoid loop patterns that can hang; prefer fixed-iteration loops.

            Output only complete Python code.
            """
        ).strip()

    # ------------------------------------------------------------------
    # Failure formatting
    # ------------------------------------------------------------------

    def _format_failures(self, failure_examples: list[dict[str, Any]]) -> str:
        if not failure_examples:
            return "- No failure logs available yet. Improve robustness without overfitting."

        lines: list[str] = []
        for idx, item in enumerate(failure_examples[:5], start=1):
            feedback = item.get("failure_feedback", {}) if isinstance(item, dict) else {}
            actions = feedback.get("repair_actions", []) if isinstance(feedback, dict) else []
            actions_text = "; ".join(str(action) for action in actions[:3]) if actions else "N/A"
            unsatisfied = feedback.get("unsatisfied_rubrics", []) if isinstance(feedback, dict) else []
            unsatisfied_text = " | ".join(str(r) for r in unsatisfied[:3]) if unsatisfied else "N/A"
            root_cause = str(feedback.get("root_cause", "N/A")) if isinstance(feedback, dict) else "N/A"
            confidence = feedback.get("confidence", "N/A") if isinstance(feedback, dict) else "N/A"
            source = feedback.get("source", "N/A") if isinstance(feedback, dict) else "N/A"

            # CaS-specific failure fields
            compilation_failed = item.get("compilation_failed", None)
            sandbox_error = item.get("sandbox_error", "")
            execution_traceback = item.get("execution_traceback", "")
            stage = feedback.get("stage", "") if isinstance(feedback, dict) else ""

            cas_info = ""
            if compilation_failed is not None or sandbox_error or execution_traceback or stage:
                cas_parts = []
                if stage:
                    cas_parts.append(f"stage={stage}")
                if compilation_failed is not None:
                    cas_parts.append(f"compilation_failed={compilation_failed}")
                if sandbox_error:
                    cas_parts.append(f"sandbox_error={str(sandbox_error)[:200]}")
                if execution_traceback:
                    cas_parts.append(f"exec_traceback={str(execution_traceback)[:200]}")
                cas_info = "\n                    cas_info: " + " | ".join(cas_parts)

            sandbox_code_snippet = str(item.get("sandbox_code_snippet", ""))[:200]
            snippet_line = ""
            if sandbox_code_snippet:
                snippet_line = f"\n                    sandbox_code: {sandbox_code_snippet}"

            lines.append(
                textwrap.dedent(
                    f"""
                    [{idx}] mode={item.get("failure_mode", "unknown")} score={item.get("score", 0)}
                    query: {item.get("query", "")[:200]}
                    answer: {item.get("answer", "")[:200]}
                    trace: {" | ".join(item.get("trace", [])[:4])}
                    root_cause: {root_cause[:260]}
                    classifier: source={source} confidence={confidence}
                    unsatisfied_rubrics: {unsatisfied_text}
                    repair_actions: {actions_text}{cas_info}{snippet_line}
                    """
                ).strip()
            )
        return "\n\n".join(lines)
