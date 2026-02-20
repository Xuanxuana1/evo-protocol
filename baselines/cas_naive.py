"""Naive CaS baseline: simple key-value dict compilation."""

from __future__ import annotations

from core.base_sandbox_protocol import BaseCaSCompiler


class NaiveCaSCompiler(BaseCaSCompiler):
    """Minimal CaS baseline that compiles context into a flat key-value dict."""

    def compile_sandbox(self, context: str) -> str:
        context_text = self._prepare_prompt_text(context, self.context_char_limit)
        prompt = (
            "Extract ALL facts from the context as a flat Python dictionary.\n"
            "If RAW_MESSAGES_JSON is present, keep role-aware constraints in facts.\n"
            "Each key should be a descriptive snake_case string.\n"
            "Each value should be the corresponding fact (string, number, list, or bool).\n"
            "Assign the dictionary to a variable named 'facts'.\n"
            "Do NOT define or override _oracle.\n"
            "Do NOT set FINAL_ANSWER.\n"
            "Output ONLY executable Python code.\n\n"
            f"Context:\n{context_text}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        query_text = self._prepare_prompt_text(query, self.query_char_limit)
        import re
        assignments = re.findall(r"^([A-Za-z_]\w*)\s*=", sandbox_schema, re.MULTILINE)
        var_names = sorted(set(assignments))[:30]

        prompt = (
            "Write Python code to answer the query using variables in the namespace.\n"
            f"Available variables: {var_names}\n"
            "Build a constraints summary and use one _oracle(prompt, str) call for final text.\n"
            "Do NOT define or override _oracle.\n"
            "Store the answer in FINAL_ANSWER as a string.\n"
            "Output ONLY Python code.\n\n"
            f"Query: {query_text}"
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


# Backward compatibility alias
NaiveSandboxProtocol = NaiveCaSCompiler
