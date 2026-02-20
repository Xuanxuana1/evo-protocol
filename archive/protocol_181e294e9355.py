from core.base_sandbox_protocol import BaseCaSCompiler

class EvoCaSCompiler(BaseCaSCompiler):
    def compile_sandbox(self, context: str) -> str:
        prompt = (
            "You are a context compiler. Convert the following natural language context "
            "into executable Python code that creates structured data objects.\n\n"
            "Rules:\n"
            "- Use Pydantic models for structured data with field validation.\n"
            "- Use networkx.Graph for spatial/relational data.\n"
            "- Encode temporal/sequential data as enums or ordered lists.\n"
            "- Add assertions to enforce critical context-specific values.\n"
            "- For nuanced/ambiguous facts, define helper methods using _oracle(prompt, return_type).\n"
            "- Output ONLY executable Python code, no explanations.\n\n"
            f"Context:\n{context}"
        )
        code = self._call_llm([{"role": "user", "content": prompt}], temperature=0.0)
        return code

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        prompt = (
            "You are a query solver. Write Python code that produces FINAL_ANSWER.\n\n"
            "Rules:\n"
            "- Use variables and objects defined in the sandbox schema.\n"
            "- Perform all reasoning and logic in Python.\n"
            "- For nuanced/semantic queries, use _oracle(prompt, return_type) as a sensor.\n"
            "- Store the final answer in the variable FINAL_ANSWER.\n"
            "- Ensure FINAL_ANSWER is non-empty; if necessary, refine the oracle prompt.\n"
            "- Output ONLY executable Python code, no explanations.\n\n"
            f"Query:\n{query}\n\n"
            f"Sandbox schema:\n{sandbox_schema[:8000]}"
        )
        code = self._call_llm([{"role": "user", "content": prompt}], temperature=0.0)
        return code