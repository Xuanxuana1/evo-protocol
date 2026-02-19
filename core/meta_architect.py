"""Prompt construction and protocol mutation helpers for the meta-architect."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

from core.base_protocol import BaseProtocol
from core.protocol_loader import ProtocolLoader
from core.self_repair import generate_with_repair


@dataclass
class ParentPerformance:
    """Compact parent protocol scorecard."""

    overall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    f3: float = 0.0
    f4: float = 0.0


class MetaArchitect:
    """Generate mutated protocol code from parent code and failure traces."""

    def __init__(self, client, architect_model: str = "gpt-4o") -> None:
        self.client = client
        self.model = architect_model

    def build_prompt(
        self,
        generation: int,
        parent_code: str,
        parent_performance: ParentPerformance,
        failure_examples: list[dict[str, Any]],
    ) -> str:
        """Build architect prompt grounded on observed failures."""

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

    def mutate(
        self,
        loader: ProtocolLoader,
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

    def _format_failures(self, failure_examples: list[dict[str, Any]]) -> str:
        if not failure_examples:
            return "- No failure logs available yet. Improve robustness without overfitting."

        lines: list[str] = []
        for idx, item in enumerate(failure_examples[:5], start=1):
            feedback = item.get("failure_feedback", {}) if isinstance(item, dict) else {}
            actions = feedback.get("repair_actions", []) if isinstance(feedback, dict) else []
            actions_text = "; ".join(str(action) for action in actions[:3]) if actions else "N/A"
            unsatisfied = feedback.get("unsatisfied_rubrics", []) if isinstance(feedback, dict) else []
            unsatisfied_text = " | ".join(str(item) for item in unsatisfied[:3]) if unsatisfied else "N/A"
            root_cause = str(feedback.get("root_cause", "N/A")) if isinstance(feedback, dict) else "N/A"
            confidence = feedback.get("confidence", "N/A") if isinstance(feedback, dict) else "N/A"
            source = feedback.get("source", "N/A") if isinstance(feedback, dict) else "N/A"
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
                    repair_actions: {actions_text}
                    """
                ).strip()
            )
        return "\n\n".join(lines)
