class MyTDGCompiler(BaseTDGCompiler):

    def compile_tests(self, context: str, query: str) -> str:
        prompt = f'Write Python test functions named test_* that accept answer:str and use assert checks. Generate executable Python only.\n\nContext:\n{context}\n\nQuery:\n{query}'
        raw_tests = self._call_llm([{'role': 'user', 'content': prompt}], temperature=0.0)
        if '```python' in raw_tests:
            return raw_tests.split('```python', 1)[1].split('```', 1)[0].strip()
        if '```' in raw_tests:
            return raw_tests.split('```', 1)[1].split('```', 1)[0].strip()
        return str(raw_tests).strip()

    def generate_answer(self, context: str, query: str, messages_raw: list=None) -> str:
        if messages_raw:
            structured = []
            for msg in messages_raw:
                role = str(msg.get('role', 'user'))
                if role not in {'system', 'user', 'assistant'}:
                    role = 'user'
                structured.append({'role': role, 'content': str(msg.get('content', ''))})
            if structured:
                structured[-1]['content'] = str(structured[-1].get('content', '')) + '\n\nAnswer using only provided context and follow explicit constraints.'
            return str(self._call_llm(structured, temperature=0.0)).strip()
        messages = [{'role': 'system', 'content': 'Answer using only provided context.\n\nContext:\n' + str(context)}, {'role': 'user', 'content': str(query)}]
        return str(self._call_llm(messages, temperature=0.0)).strip()