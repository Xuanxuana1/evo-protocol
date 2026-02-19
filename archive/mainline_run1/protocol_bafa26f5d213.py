class NaiveProtocol(BaseProtocol):
    """Directly pass full context to one LLM call with no hard verification."""

    def perception(self, context: str) -> dict[str, Any]:
        return {"full_context": context}

    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "Answer based only on the given context."},
            {
                "role": "user",
                "content": (
                    f"Context:\n{perceived_info['full_context']}\n\n"
                    f"Question:\n{query}"
                ),
            },
        ]
        return self._call_llm(messages)

    def verification(self, answer: str, context: str) -> bool:
        return True
