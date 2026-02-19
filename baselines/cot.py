"""Chain-of-thought style baseline protocol."""

from __future__ import annotations

from typing import Any

from core.base_protocol import BaseProtocol


class CoTProtocol(BaseProtocol):
    """Single reasoning call plus lightweight context-faithfulness check."""

    def perception(self, context: str) -> dict[str, Any]:
        return {"full_context": context}

    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Reason step by step internally and provide the final answer "
                    "strictly grounded in context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{perceived_info['full_context']}\n\n"
                    f"Question:\n{query}\n\n"
                    "Give a concise final answer."
                ),
            },
        ]
        return self._call_llm(messages, temperature=0)

    def verification(self, answer: str, context: str) -> bool:
        verdict = self._call_llm(
            [
                {
                    "role": "system",
                    "content": "Check if answer is fully supported by context. Reply PASS or FAIL.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nAnswer:\n{answer}",
                },
            ],
            temperature=0,
        )
        return verdict.strip().upper().startswith("PASS")
