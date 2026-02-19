"""ReAct-inspired baseline protocol with small iterative loop."""

from __future__ import annotations

from typing import Any

from core.base_protocol import BaseProtocol


class ReActProtocol(BaseProtocol):
    """Two-step reasoning: plan evidence -> synthesize answer -> verify."""

    def perception(self, context: str) -> dict[str, Any]:
        chunks = [context[i : i + 2000] for i in range(0, len(context), 2000)]
        return {"chunks": chunks, "num_chunks": len(chunks)}

    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        chunk_text = "\n\n".join(
            f"[Chunk {idx}]\n{chunk}" for idx, chunk in enumerate(perceived_info["chunks"])
        )

        evidence = self._call_llm(
            [
                {
                    "role": "system",
                    "content": "Extract the minimum evidence needed to answer the question.",
                },
                {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{chunk_text}"},
            ],
            temperature=0,
        )

        answer = self._call_llm(
            [
                {
                    "role": "system",
                    "content": "Answer using only the listed evidence and avoid outside knowledge.",
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{evidence}\n\nQuestion:\n{query}",
                },
            ],
            temperature=0,
        )
        return answer

    def verification(self, answer: str, context: str) -> bool:
        verdict = self._call_llm(
            [
                {
                    "role": "system",
                    "content": "Return PASS if every answer claim is supported by context, else FAIL.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nAnswer:\n{answer}",
                },
            ],
            temperature=0,
        )
        return verdict.strip().upper().startswith("PASS")
