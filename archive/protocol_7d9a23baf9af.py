class SeedCaSCompiler(BaseCaSCompiler):
    """Seed CaS compiler that compiles context into dicts/dataclasses via LLM code generation.

    This is the generation-0 starting point for CaS evolution.
    """

    def compile_sandbox(self, context: str) -> str:
        prompt = (
            "You are a context compiler. Convert the following natural language context "
            "into executable Python code that creates structured data objects.\n\n"
            "Rules:\n"
            "- Create Python dicts, lists, dataclasses, or named variables for ALL facts.\n"
            "- Every rule, constraint, fact, or relationship must be encoded.\n"
            "- Use descriptive variable names.\n"
            "- Add assertions that enforce critical context-specific values.\n"
            "  Example: if context says 'the sky is green', add: assert sky_color == 'green'\n"
            "- For nuanced or ambiguous facts that resist symbolic extraction, define\n"
            "  helper functions that call _oracle(prompt, return_type) for perception.\n"
            "  _oracle returns strictly-typed values: bool, int, float, or str.\n"
            "  Example: def is_tone_hostile(): return _oracle('Is the tone hostile?', bool)\n"
            "- NEVER define or override a function/variable named _oracle.\n"
            "- NEVER raise NotImplementedError placeholders.\n"
            "- Only use: dataclasses, json, re, enum, collections, math.\n"
            "- Do NOT use while-loops.\n"
            "- Do NOT set FINAL_ANSWER in this code.\n\n"
            "Output ONLY executable Python code, no explanations.\n\n"
            f"Context:\n{context[:8000]}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        # Extract variable names from the sandbox code for the solver prompt
        import re
        assignments = re.findall(r"^([A-Za-z_]\w*)\s*=", sandbox_schema, re.MULTILINE)
        var_names = sorted(set(assignments))[:30]

        prompt = (
            "You are a query solver. Write Python code that answers the query "
            "by using variables already defined in the namespace.\n\n"
            "Rules:\n"
            f"- Available variables from sandbox: {var_names}\n"
            "- Store your final answer as a string in FINAL_ANSWER.\n"
            "- All logic (loops, conditionals, math) must be in Python.\n"
            "- For semantic questions, use _oracle(prompt, return_type) as sensor.\n"
            "  _oracle(prompt, bool) returns True/False\n"
            "  _oracle(prompt, str) returns brief text\n"
            "- NEVER define, assign, or override _oracle.\n"
            "- NEVER raise NotImplementedError placeholders.\n"
            "- Do NOT use while-loops.\n"
            "- Do NOT redefine variables from the sandbox.\n\n"
            "Output ONLY executable Python code, no explanations.\n\n"
            f"Query: {query[:2000]}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text.strip()
