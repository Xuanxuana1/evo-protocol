# Assuming BaseTDGCompiler is available in the current environment without direct import

class EvoTDGCompiler(BaseTDGCompiler):
    def compile_tests(self, context: str, query: str) -> str:
        # Use full context and query for test generation
        context_text = context
        query_text = query
        prompt = (
            "You are a test engineer. Generate Python test functions that verify "
            "a correct answer to the given query based on the provided context.\n\n"
            "Rules:\n"
            "- Each test function must be named with a test_ prefix.\n"
            "- Each test function accepts a single parameter: answer (str).\n"
            "- Use assert statements to check properties of the answer.\n"
            "- Write tests for:\n"
            "  * Factual correctness: key facts from context must appear in answer\n"
            "  * Format compliance: required format/structure from the query\n"
            "  * Keyword presence: critical terms that must be mentioned\n"
            "  * Anti-parametric-override: facts that contradict common knowledge\n"
            "    (e.g., if context says sky is green, assert 'green' in answer)\n"
            "  * Constraint satisfaction: length limits, persona requirements, etc.\n"
            "- For semantic/tone checks that cannot be expressed as string asserts,\n"
            "  use _oracle(prompt, bool) which returns True/False.\n"
            "  Example: assert _oracle(f'Is this answer polite? {answer}', bool)\n"
            "- Do NOT define or override _oracle; it is injected by the runtime.\n"
            "- Do NOT use while-loops.\n"
            "- Keep tests focused and independent.\n"
            "- Output ONLY executable Python code, no explanations.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Query:\n{query_text}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        if messages_raw:
            # Reconstruct structured messages for multi-turn conversations
            structured: list[dict[str, str]] = []
            for msg in messages_raw:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                if role == "system":
                    structured.append({"role": "system", "content": content})
                elif role == "assistant":
                    structured.append({"role": "assistant", "content": content})
                else:
                    structured.append({"role": "user", "content": content})
            # Append instruction to use full context
            if structured:
                last_content = structured[-1].get("content", "")
                structured[-1]["content"] = (
                    f"{last_content}\n\n"
                    "Answer using ONLY the information provided in this conversation. "
                    "Follow all persona, tone, format, and style constraints from the system message."
                )
            answer = self._call_llm(structured, temperature=0.0)
        else:
            # Simple context + query
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You must answer using ONLY the provided context. "
                        "Follow persona/tone/format constraints from all roles if present.\n\n"
                        f"Context:\n{context}"
                    ),
                },
                {"role": "user", "content": query},
            ]
            answer = self._call_llm(messages, temperature=0.0)
        return answer.strip()

    @staticmethod
    def _extract_code(text: str) -> str:
        return text.strip()