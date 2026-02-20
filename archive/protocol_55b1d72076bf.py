from core.base_sandbox_protocol import BaseCaSCompiler

class EvoCaSCompiler(BaseCaSCompiler):
    def compile_sandbox(self, context: str) -> str:
        context_text = self._prepare_prompt_text(context, self.context_char_limit)
        prompt = (
            "You are a context compiler. Convert the following natural language context "
            "into executable Python code that creates structured data objects.\n\n"
            "Rules:\n"
            "- Use Pydantic models for factual data with validators.\n"
            "- Use networkx.Graph for spatial/relational data.\n"
            "- Use enums or state machines for temporal/sequential data.\n"
            "- Use dicts with callable rule functions for rule-based data.\n"
            "- Use list[dict] or list[dataclass] for tabular data.\n"
            "- Define Neural Oracle methods for nuanced/semantic data.\n"
            "- Add assertions to enforce context-specific values.\n"
            "- NEVER define or override a function/variable named _oracle.\n"
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
        
        import re
        assignments = re.findall(r"^([A-Za-z_]\w*)\s*=", schema_text, re.MULTILINE)
        var_names = sorted(set(assignments))[:30]

        prompt = (
            "You are a query solver. Write Python code that produces FINAL_ANSWER.\n\n"
            "Rules:\n"
            f"- Available variables from sandbox: {var_names}\n"
            "- Use Python logic to derive the answer from sandbox variables.\n"
            "- Use _oracle(prompt, return_type) for semantic perception if needed.\n"
            "- Store your final answer as a string in FINAL_ANSWER.\n"
            "- Guarantee FINAL_ANSWER is non-empty; if empty, call _oracle again with a stricter prompt.\n"
            "- NEVER define, assign, or override _oracle.\n"
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
        return text.strip()