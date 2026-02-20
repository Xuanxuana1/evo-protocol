"""Generation-0 TDG seed compiler: test-driven generation from context."""

from __future__ import annotations

import json

from core.base_tdg_protocol import BaseTDGCompiler


class SeedTDGCompiler(BaseTDGCompiler):
    """Seed TDG compiler that generates test functions and NL answers via LLM.

    This is the generation-0 starting point for TDG evolution.
    Key advantage over CaS: no context truncation, full NL answer generation,
    and graceful degradation to direct inference when tests fail.
    """

    def compile_tests(self, context: str, query: str) -> str:
        # TDG key advantage: full context, no aggressive truncation
        context_text = context
        query_text = query
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
            "   Example: if context says the system must output 'READY', then:\n"
            "       assert 'READY' in answer\n\n"
            "2. FORMAT / STRUCTURE\n"
            "   If a format is required (quest log, numbered steps, bullet list, JSON, markdown "
            "headers, table), assert structural markers are present.\n"
            "   Example: quest log format → assert 'Quest:' in answer or '[]' in answer\n"
            "   Example: numbered steps  → assert any(f'{i}.' in answer for i in range(1,5))\n"
            "   Example: five-year-old language:\n"
            "       words = answer.split()\n"
            "       long_words = [w for w in words if len(w) > 8]\n"
            "       assert len(long_words) / max(len(words), 1) < 0.1, "
            "f'Too many long words: {long_words[:5]}'\n\n"
            "3. ANTI-PARAMETRIC-OVERRIDE (most important)\n"
            "   Identify facts in the context that DIFFER from common world knowledge. "
            "Assert the context fact is used and the common-knowledge fact is NOT used.\n"
            "   Example: context says capital of France is Lyon:\n"
            "       assert 'lyon' in answer.lower(), 'Must use context fact (Lyon), not Paris'\n"
            "       assert 'paris' not in answer.lower(), 'Must not override context with Paris'\n"
            "   Example: context uses a custom Monopoly rule that differs from standard:\n"
            "       assert '<custom rule keyword>' in answer.lower()\n"
            "       assert '<standard rule that was overridden>' not in answer.lower()\n\n"
            "4. PROHIBITED CONTENT (negative tests)\n"
            "   If the query restricts sources ('use ONLY the provided rules', 'do not use "
            "external knowledge'), assert that content from outside the context is absent.\n"
            "   List 2-3 specific things a model with parametric knowledge would incorrectly "
            "add, and assert they are NOT in the answer.\n"
            "   Example: 'use only Monopoly rules from this document' →\n"
            "       assert 'auction' not in answer.lower(), 'Standard auction rule not in doc'\n"
            "       assert 'mortgage' not in answer.lower(), 'Mortgage not in provided rules'\n\n"
            "5. SPECIFIC NAMED ENTITIES / VALUES\n"
            "   Extract dates, names, numbers, technical terms that are critical to a correct "
            "answer. Assert they appear.\n"
            "   Example: context says event was in 1791 involving Stadler:\n"
            "       assert '1791' in answer\n"
            "       assert 'stadler' in answer.lower()\n\n"
            "6. PERSONA / ROLE / TONE (use _oracle for these)\n"
            "   If the system prompt assigns a role, persona, or tone constraint, write an "
            "oracle test with a SPECIFIC binary question — not 'Is this good?' but:\n"
            "   'Does this response fully adopt the [specific role] persona throughout, "
            "without breaking character? Answer True or False only.'\n"
            "   Example for 'respond as a medieval wizard':\n"
            "       assert _oracle(\n"
            "           f'Does every sentence in this text use archaic/medieval language '\n"
            "           f'consistent with a wizard character? No modern slang or casual '\n"
            "           f'phrases. Answer True or False.\\n\\n{answer[:1000]}', bool), \\\n"
            "           'Must maintain medieval wizard persona throughout'\n\n"
            "7. ORDERING / SEQUENCING\n"
            "   If the context or query specifies a required sequence of steps or events, "
            "verify they appear in order.\n"
            "   Example: step A must precede step B:\n"
            "       idx_a = answer.lower().find('<step_a_keyword>')\n"
            "       idx_b = answer.lower().find('<step_b_keyword>')\n"
            "       assert idx_a != -1 and idx_b != -1 and idx_a < idx_b\n\n"
            "8. COMPLETENESS (scope check)\n"
            "   If the query asks for N items or a complete list, verify the answer is not "
            "truncated or missing obvious required components.\n"
            "   Use _oracle only when string matching is truly insufficient:\n"
            "       assert _oracle(\n"
            "           f'Does this answer address ALL the required points without omitting '\n"
            "           f'any? Required: [list key points]. Answer True or False.\\n\\n'\n"
            "           f'{answer[:1500]}', bool)\n\n"
            "=== ORACLE USAGE RULES ===\n"
            "- _oracle(prompt: str, bool) returns True or False\n"
            "- Use ONLY when deterministic string checks are impossible\n"
            "- Make the oracle question BINARY and SPECIFIC: include the exact criterion\n"
            "- Bad oracle: _oracle(f'Is this answer correct? {answer}', bool)\n"
            "- Good oracle: _oracle(f'Does this answer use ONLY vocabulary appropriate for "
            "a 5-year-old child (no words longer than 2 syllables, no jargon)? "
            "True or False.\\n\\n{answer[:800]}', bool)\n"
            "- Do NOT define or override _oracle; it is injected by the runtime\n\n"
            "=== ADDITIONAL RULES ===\n"
            "- Write 5-10 test functions total\n"
            "- Each test must be independent (no shared state)\n"
            "- Do NOT use while-loops\n"
            "- Do NOT use try/except to silence failures — let assertions propagate\n"
            "- Use .lower() for case-insensitive keyword checks\n"
            "- Include helpful assert messages: assert cond, 'Explanation of what failed'\n"
            "- Output ONLY executable Python code, no explanations, no markdown\n\n"
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
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text.strip()
