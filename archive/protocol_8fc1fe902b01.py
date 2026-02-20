from core.base_sandbox_protocol import BaseCaSCompiler

class EvoCaSCompiler(BaseCaSCompiler):
    def compile_sandbox(self, context: str) -> str:
        prompt = (
            "You are a context compiler. Convert the following natural language context "
            "into executable Python code that creates structured data objects.\n\n"
            "Rules:\n"
            "- Use Pydantic models for factual data with validators.\n"
            "- Use networkx.Graph for relational data.\n"
            "- Define Neural Oracle methods for nuanced perception.\n"
            "- Add assertions for context-specific values.\n"
            "- Output ONLY executable Python code, no explanations.\n\n"
            f"Context:\n{context}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        prompt = (
            "You are a query solver. Write Python code that produces FINAL_ANSWER.\n\n"
            "Rules:\n"
            "- Use sandbox variables to build constraints.\n"
            "- Use _oracle for nuanced queries.\n"
            "- Store the final answer in FINAL_ANSWER.\n"
            "- Output ONLY executable Python code, no explanations.\n\n"
            f"Query: {query}\n\n"
            f"Sandbox schema excerpt:\n{sandbox_schema[:8000]}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    @staticmethod
    def _extract_code(text: str) -> str:
        return text.strip()