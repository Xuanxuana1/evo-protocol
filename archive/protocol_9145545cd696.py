class TDGCompiler(BaseTDGCompiler):
    def compile_tests(self, context: str, query: str) -> str:
        return f"# Tests for: {query}\n# Context: {context}\n"
    
    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        return f"Answer based on context: {context[:50]}..." if context else "No context provided"