from core.base_tdg_protocol import BaseTDGCompiler

class EvoTDGCompiler(BaseTDGCompiler):
    def compile_tests(self, context: str, query: str) -> str:
        prompt = (
            "You are a strict test engineer. Your job is to write Python test functions "
            "that FAIL on any answer that does not perfectly satisfy every constraint in "
            "the query and context. A test that passes 90% of wrong answers is useless.\n\n"
            "=== CORE PHILOSOPHY ===\n"
            "Design each test to be GUILTY-UNTIL-PROVEN-INNOCENT:\n"
            "- Assume the answer is wrong; the test should catch it unless it is exactly right.\n"
            "- A vague or partially-correct answer must fail most tests.\n"
            "- Prefer over-strict tests over under-strict tests.\n\n"
            "=== TEST FUNCTION CONTRACT ===\n"
            "- Name: must start with test_\n"
            "- Signature: def test_*(answer: str) -> None\n"
            "- Failure: raise AssertionError (or any exception) when answer is wrong\n"
            "- No return value needed\n\n"
            "=== REQUIRED TEST CATEGORIES (write at least one per applicable category) ===\n\n"
            "1. EXACT REQUIRED PHRASES\n"
            "   If the query or context specifies a required exact string (a phrase, a signal "
            "word, a code word, a quoted sentence), assert it appears verbatim.\n"
            "2. FORMAT / STRUCTURE\n"
            "   If a format is required (quest log, numbered steps, bullet list, JSON, markdown "
            "headers, table), assert structural markers are present.\n"
            "3. ANTI-PARAMETRIC-OVERRIDE (most important)\n"
            "   Identify facts in the context that DIFFER from common world knowledge. "
            "Assert the context fact is used and the common-knowledge fact is NOT used.\n"
            "4. PROHIBITED CONTENT (negative tests)\n"
            "   If the query restricts sources ('use ONLY the provided rules', 'do not use "
            "external knowledge'), assert that content from outside the context is absent.\n"
            "5. SPECIFIC NAMED ENTITIES / VALUES\n"
            "   Extract dates, names, numbers, technical terms that are critical to a correct "
            "answer. Assert they appear.\n"
            "6. PERSONA / ROLE / TONE (use _oracle for these)\n"
            "   If the system prompt assigns a role, persona, or tone constraint, write an "
            "oracle test with a SPECIFIC binary question.\n"
            "7. ORDERING / SEQUENCING\n"
            "   If the context or query specifies a required sequence of steps or events, "
            "verify they appear in order.\n"
            "8. COMPLETENESS (scope check)\n"
            "   If the query asks for N items or a complete list, verify the answer is not "
            "truncated or missing obvious required components.\n"
            "=== ORACLE USAGE RULES ===\n"
            "- _oracle(prompt: str, bool) returns True or False\n"
            "- Use ONLY when deterministic string checks are impossible\n"
            "- Make the oracle question BINARY and SPECIFIC: include the exact criterion\n"
            "=== ADDITIONAL RULES ===\n"
            "- Write 5-10 test functions total\n"
            "- Each test must be independent (no shared state)\n"
            "- Do NOT use while-loops\n"
            "- Do NOT use try/except to silence failures — let assertions propagate\n"
            "- Use .lower() for case-insensitive keyword checks\n"
            "- Include helpful assert messages: assert cond, 'Explanation of what failed'\n"
            "- Output ONLY executable Python code, no explanations, no markdown\n\n"
            f"Context:\n{context}\n\n"
            f"Query:\n{query}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        if messages_raw:
            structured = []
            for msg in messages_raw:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                if role == "system":
                    structured.append({"role": "system", "content": content})
                elif role == "assistant":
                    structured.append({"role": "assistant", "content": content})
                else:
                    structured.append({"role": "user", "content": content})
            if structured:
                last_content = structured[-1].get("content", "")
                structured[-1]["content"] = (
                    f"{last_content}\n\n"
                    "Answer using ONLY the information provided in this conversation. "
                    "Follow all persona, tone, format, and style constraints from the system message."
                )
            answer = self._call_llm(structured, temperature=0.0)
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