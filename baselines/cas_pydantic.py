"""Pydantic CaS baseline: always compiles context into Pydantic models."""

from __future__ import annotations

from core.base_sandbox_protocol import BaseCaSCompiler


class PydanticCaSCompiler(BaseCaSCompiler):
    """CaS baseline that always uses Pydantic models for context compilation.

    The compile_sandbox prompts the LLM to create Pydantic BaseModel classes
    with field_validator decorators that enforce context-specific values. Any
    attempt by the solver to use values contradicting the context will raise
    a ValidationError at runtime -- the "Cognitive Faraday Cage."
    """

    def compile_sandbox(self, context: str) -> str:
        prompt = (
            "You are a context compiler that uses Pydantic models.\n\n"
            "Convert the following context into Python code that:\n"
            "1. Imports from pydantic: BaseModel, field_validator\n"
            "2. Defines Pydantic model classes for the main entities/concepts\n"
            "3. Adds field validators that enforce context-specific constraints\n"
            "   (e.g., if context says color is purple, validator rejects other colors)\n"
            "4. Instantiates model objects with the actual data from context\n"
            "5. Creates a 'facts' dict summarizing key facts\n"
            "6. For nuanced semantic facts, define functions using _oracle():\n"
            "   def is_tone_negative(): return _oracle('Is the tone negative?', bool)\n\n"
            "Rules:\n"
            "- Do NOT use while-loops\n"
            "- Do NOT set FINAL_ANSWER\n"
            "- Output ONLY executable Python code\n\n"
            f"Context:\n{context[:8000]}"
        )
        code = self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._extract_code(code)

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        import re
        assignments = re.findall(r"^([A-Za-z_]\w*)\s*=", sandbox_schema, re.MULTILINE)
        classes = re.findall(r"^class\s+(\w+)", sandbox_schema, re.MULTILINE)
        var_names = sorted(set(assignments))[:30]
        class_names = sorted(set(classes))[:15]

        prompt = (
            "Write Python code to answer the query using the sandbox namespace.\n"
            "The namespace contains Pydantic model instances and dicts.\n"
            f"Available variables: {var_names}\n"
            f"Available classes: {class_names}\n\n"
            "Rules:\n"
            "- Access model attributes with dot notation (e.g., model.field)\n"
            "- Store the final answer in FINAL_ANSWER as a string\n"
            "- Use _oracle(prompt, type) for semantic questions\n"
            "- Do NOT use while-loops\n"
            "- Output ONLY executable Python code\n\n"
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


# Backward compatibility alias
PydanticSandboxProtocol = PydanticCaSCompiler
