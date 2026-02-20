class SeedCaSCompiler(BaseCaSCompiler):
    """Seed CaS compiler that compiles context into dicts/dataclasses via LLM code generation.

    This is the generation-0 starting point for CaS evolution.
    """

    def compile_sandbox(self, context: str) -> str:
        context_text = self._prepare_prompt_text(context, self.context_char_limit)
        prompt = (
            "You are a context compiler. Convert the following natural language context "
            "into executable Python code that creates structured data objects.\n\n"
            "Rules:\n"
            "- The input may include RAW_MESSAGES_JSON with role/content for every turn.\n"
            "- Preserve role information: build variables like conversation_messages,\n"
            "  system_constraints, and user_requests from RAW_MESSAGES_JSON.\n"
            "- Create Python dicts, lists, dataclasses, or named variables for ALL facts.\n"
            "- Every rule, constraint, fact, or relationship must be encoded.\n"
            "- Use descriptive variable names.\n"
            "- Add assertions that enforce critical context-specific values.\n"
            "  Example: if context says 'the sky is green', add: assert sky_color == 'green'\n"
            "- For nuanced or ambiguous facts that resist symbolic extraction, define\n"
            "  helper functions that call _oracle(prompt, return_type) for perception.\n"
            "  _oracle returns strictly-typed values: bool, int, float, or str.\n"
            "  Example: def is_tone_hostile(): return _oracle('Is the tone hostile?', bool)\n"
            "- Keep generated code compact (prefer concise data structures; avoid huge boilerplate).\n"
            "- NEVER define or override a function/variable named _oracle.\n"
            "- NEVER raise NotImplementedError placeholders.\n"
            "- Only use: dataclasses, json, re, enum, collections, math.\n"
            "- Do NOT use while-loops.\n"
            "- Do NOT set FINAL_ANSWER in this code.\n\n"
            "Output ONLY executable Python code, no explanations.\n\n"
            f"Context:\n{context_text}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        query_text = self._prepare_prompt_text(query, self.query_char_limit)
        schema_text = self._prepare_prompt_text(sandbox_schema, min(self.context_char_limit, 16000))
        # Extract variable names from the sandbox code for the solver prompt
        import re
        assignments = re.findall(r"^([A-Za-z_]\w*)\s*=", schema_text, re.MULTILINE)
        var_names = sorted(set(assignments))[:30]

        prompt = (
            "You are a query solver. Write Python code that produces FINAL_ANSWER.\n\n"
            "Rules:\n"
            f"- Available variables from sandbox: {var_names}\n"
            "- Build a compact constraints summary from sandbox variables.\n"
            "- Use ONE _oracle(..., str) call to draft the final free-form response.\n"
            "- The oracle prompt must include: user query + extracted constraints.\n"
            "- Preserve required persona/style/format directives from constraints.\n"
            "- Store your final answer as a string in FINAL_ANSWER.\n"
            "- NEVER define, assign, or override _oracle.\n"
            "- NEVER raise NotImplementedError placeholders.\n"
            "- Do NOT use while-loops.\n"
            "- Do NOT redefine variables from the sandbox.\n\n"
            "Output ONLY executable Python code, no explanations.\n\n"
            f"Query: {query_text}\n\n"
            f"Sandbox schema excerpt:\n{schema_text[:8000]}"
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
