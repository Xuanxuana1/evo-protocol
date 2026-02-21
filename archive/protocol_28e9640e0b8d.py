from core.base_tdg_protocol import BaseTDGCompiler

class EvoTDGCompiler(BaseTDGCompiler):
    def compile_tests(self, context: str, query: str) -> str:
        prompt = (
            "You are a strict test engineer tasked with writing Python test functions "
            "to verify the correctness of answers generated from the given context and query. "
            "Each test must fail if the answer does not perfectly satisfy the constraints. "
            "Design tests to be guilty-until-proven-innocent, ensuring that vague or partially "
            "correct answers fail most tests.\n\n"
            "=== TEST FUNCTION CONTRACT ===\n"
            "- Name: must start with test_\n"
            "- Signature: def test_*(answer: str) -> None\n"
            "- Failure: raise AssertionError (or any exception) when answer is wrong\n"
            "- No return value needed\n\n"
            "=== REQUIRED TEST CATEGORIES ===\n"
            "1. Factual asserts: Verify key facts from context appear in the answer.\n"
            "2. Format checks: Ensure required structure, headers, or formatting.\n"
            "3. Keyword checks: Assert critical terms from context/query are present.\n"
            "4. Semantic/tone checks: Use _oracle to verify persona, tone, or style.\n"
            "5. Anti-parametric-override: Enforce context-derived constraints, ensuring "
            "context facts override common knowledge.\n"
            "6. Constraint checks: Verify length, persona, style, or other requirements.\n"
            "7. Completeness: Ensure all required points are addressed.\n"
            "8. Ordering/sequencing: Verify correct order of steps or events.\n\n"
            "=== ORACLE USAGE RULES ===\n"
            "- Use _oracle(prompt: str, bool) for binary semantic checks.\n"
            "- Make oracle questions specific and binary.\n"
            "- Example: _oracle(f'Does this answer use formal tone? {answer}', bool)\n\n"
            "=== ADDITIONAL RULES ===\n"
            "- Write 5-10 independent test functions.\n"
            "- Avoid while-loops; use bounded for-loops.\n"
            "- Include helpful assert messages.\n"
            "- Output ONLY executable Python code.\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return code.strip()

    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        if messages_raw:
            structured_messages = []
            for msg in messages_raw:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                structured_messages.append({"role": role, "content": content})
            if structured_messages:
                last_content = structured_messages[-1].get("content", "")
                structured_messages[-1]["content"] = (
                    f"{last_content}\n\n"
                    "Answer using ONLY the information provided in this conversation. "
                    "Follow all persona, tone, format, and style constraints from the system message."
                )
            answer = self._call_llm(structured_messages, temperature=0.0)
        else:
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