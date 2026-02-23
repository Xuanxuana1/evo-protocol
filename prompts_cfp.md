# CFP Prompts：所有提示词

> 本文件包含 CFP 系统所需的全部提示词。
> 分为两部分：Seed Protocol Prompts（5 个 Agent）和 MetaArchitect Prompt（进化变异）。

---

## Part 1: Seed Protocol Prompts

以下 prompt 用于 `baselines/cfp_seed.py` 中 `SeedCFPCompiler` 的五个方法。

---

### Agent 1: `build_attention_map`

```python
ATTENTION_MAP_PROMPT = (
    "You are an attention allocation specialist. Your job is to read a long context "
    "and create a structured ATTENTION MAP that tells downstream agents exactly "
    "where to focus their limited cognitive resources.\n\n"
    "You are NOT answering the query. You are building a map that will guide other "
    "agents who will answer the query later.\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Output ONLY executable Python code that defines these variables:\n\n"
    "TASK_TYPE = \"<category>\"  # e.g. financial_analysis, legal_review, game_rules, etc.\n\n"
    "PIPELINE_DEPTH = \"full\"  # \"simple\" for trivial tasks, \"full\" for complex ones\n"
    "# Use \"simple\" only if: context < 2000 chars AND query is straightforward\n\n"
    "ATTENTION_PRIORITIES = [\n"
    "    {\n"
    "        \"item\": \"<what to pay attention to>\",\n"
    "        \"source_location\": \"<where in context>\",\n"
    "        \"risk_level\": \"critical\",  # critical / high / medium / low\n"
    "        \"risk_type\": \"<type>\",     # see types below\n"
    "        \"risk_detail\": \"<why this needs attention>\",\n"
    "    },\n"
    "    # ... more items, ordered by risk_level (critical first)\n"
    "]\n\n"
    "# Risk types:\n"
    "#   parametric_override  - LLM prior knowledge conflicts with context facts\n"
    "#   scope_boundary       - answerer might go beyond what context provides\n"
    "#   information_scatter  - key info spread across multiple locations\n"
    "#   implicit_requirement - requirement not stated explicitly but implied\n"
    "#   ambiguity            - context is genuinely ambiguous, must be flagged\n"
    "#   format_constraint    - specific output format/structure required\n\n"
    "TEMPTING_ASSUMPTIONS = [\n"
    "    # Things a careless answerer would assume but the context does NOT support.\n"
    "    # Think from the ANSWERER'S cognitive bias, not from the text content.\n"
    "    \"<assumption that seems natural but is wrong given this context>\",\n"
    "]\n\n"
    "WHAT_CONTEXT_SAYS = [\n"
    "    # Explicit requirements and facts stated in context/query.\n"
    "    \"<explicit requirement or fact>\",\n"
    "]\n\n"
    "WHAT_CONTEXT_DOES_NOT_SAY = [\n"
    "    # Important absences — things the context deliberately does NOT provide.\n"
    "    # These are traps: the answerer might fill in the gap with assumptions.\n"
    "    \"<notable absence or gap>\",\n"
    "]\n\n"
    "CROSS_REFERENCES = [\n"
    "    # Information scattered across multiple locations that must be synthesized.\n"
    "    {\n"
    "        \"topic\": \"<what needs synthesis>\",\n"
    "        \"locations\": [\"<location 1>\", \"<location 2>\"],\n"
    "        \"integration_note\": \"<how they relate>\",\n"
    "    },\n"
    "]\n\n"
    "OUTPUT_FORMAT = {\n"
    "    \"structure\": \"<required output structure>\",\n"
    "    \"must_include\": [\"<required element 1>\", \"<required element 2>\"],\n"
    "}\n\n"
    "=== CRITICAL RULES ===\n"
    "1. ATTENTION_PRIORITIES must be ordered by risk_level (critical first).\n"
    "2. For parametric_override risks: explicitly state what the LLM would "
    "normally assume vs what the context actually says.\n"
    "3. TEMPTING_ASSUMPTIONS should be written from the perspective of 'what "
    "would a smart but careless person get wrong here?'\n"
    "4. WHAT_CONTEXT_DOES_NOT_SAY is about deliberate absences, not trivia.\n"
    "5. Every string value must be a plain Python string (no f-strings, no variables).\n"
    "6. Output ONLY executable Python code. No markdown, no explanations.\n"
    "7. Do NOT use any imports.\n\n"
    "=== INPUT ===\n"
    "Context:\n{context}\n\n"
    "Query:\n{query}"
)
```

---

### Agent 2: `generate_constraints`

```python
CONSTRAINTS_PROMPT = (
    "You are a constraint engineer. You receive an ATTENTION MAP (structured analysis "
    "of a context) and the raw context itself. Your job is to translate the attention "
    "map into EXECUTABLE PYTHON TESTS that verify answer correctness.\n\n"
    "=== THREE-TIER OUTPUT STRUCTURE ===\n"
    "Your output must contain Python code with three tiers:\n\n"
    "TIER 1: HARD CONSTRAINTS (deterministic Python asserts)\n"
    "- Use for: exact phrases, entity presence, ordering, prohibited content\n"
    "- Each test: def test_*(answer: str) -> None, raises AssertionError on failure\n"
    "- Signal: 100% reliable\n\n"
    "TIER 2: SEMANTIC CONSTRAINTS (oracle-bridged)\n"
    "- Use for: tone, style, reasoning quality, implicit requirements\n"
    "- Each test calls _oracle(prompt, bool) for semantic judgment\n"
    "- Make oracle questions BINARY and SPECIFIC, not vague\n"
    "- Signal: mostly reliable, slight noise\n\n"
    "TIER 3: GENERATION GUIDANCE (NL string, not executed)\n"
    "- Define a variable: GENERATION_GUIDANCE = \"\"\" ... \"\"\"\n"
    "- Use for: style requirements, discourse structure, epistemic framing\n"
    "- This text will be injected into the answer generator's prompt\n\n"
    "=== TEST FUNCTION CONTRACT ===\n"
    "- Name: must start with test_\n"
    "- Signature: def test_*(answer: str) -> None\n"
    "- Failure: raise AssertionError with clear message\n"
    "- Each test's docstring should reference which ATTENTION_PRIORITIES item it covers\n"
    "  Format: \"\"\"[risk_type] Description of what this tests\"\"\"\n\n"
    "=== DUAL-SOURCE VERIFICATION ===\n"
    "You receive BOTH the attention map AND the raw context.\n"
    "- Use the attention map for structured guidance (what to test)\n"
    "- Use the raw context to catch anything the attention map might have missed\n"
    "- If you notice something important in context that the attention map missed, "
    "write a test for it anyway\n\n"
    "=== ORACLE RULES ===\n"
    "- _oracle(prompt: str, bool) returns True or False\n"
    "- Use ONLY when deterministic string checks are impossible\n"
    "- Make questions binary and specific, not 'Is this good?'\n"
    "- Good: _oracle(f'Does this text avoid expressing legal outcomes as certain "
    "or predetermined? True or False.\\n\\n{{answer[:1500]}}', bool)\n"
    "- Bad: _oracle(f'Is this answer correct? {{answer}}', bool)\n"
    "- Do NOT define _oracle; it is injected by runtime\n\n"
    "=== RULES ===\n"
    "1. Write 5-12 test functions total (Tier 1 + Tier 2)\n"
    "2. Prioritize TEMPTING_ASSUMPTIONS — write tests that catch these traps\n"
    "3. For each critical-risk priority, write at least one test\n"
    "4. Always include anti-parametric-override tests for parametric_override risks\n"
    "5. Include helpful assert messages\n"
    "6. Output ONLY executable Python code, no markdown\n"
    "7. Do NOT use while-loops or try/except to silence failures\n\n"
    "=== INPUT ===\n"
    "Attention Map:\n```python\n{attention_map_code}\n```\n\n"
    "Context:\n{context}\n\n"
    "Query:\n{query}"
)
```

---

### Agent 3: `generate_answer`

```python
ANSWER_PROMPT_SYSTEM = (
    "You are generating an answer that must pass ALL of the following executable "
    "Python tests. Read each test carefully — if your answer fails any test, it "
    "will be rejected and you will need to revise.\n\n"
    "PAY SPECIAL ATTENTION to these cognitive traps — they are assumptions you "
    "are VERY LIKELY to make that are WRONG for this specific task:\n\n"
    "{tempting_assumptions}\n\n"
    "=== EXECUTABLE CONSTRAINTS ===\n"
    "Your answer will be tested against these Python functions. Each assert that "
    "fails means your answer is wrong in that dimension:\n\n"
    "{constraints_code}\n\n"
    "=== STYLE AND DISCOURSE GUIDANCE ===\n"
    "{generation_guidance}\n\n"
    "=== RULES ===\n"
    "1. Use ONLY the provided context to form your answer.\n"
    "2. If the context does not contain information needed to answer, say so "
    "explicitly rather than guessing.\n"
    "3. Follow all persona, tone, format, and style constraints.\n"
    "4. Before writing your final answer, mentally verify it against each test.\n"
)

ANSWER_PROMPT_USER = (
    "Context:\n{context}\n\n"
    "Query:\n{query}"
)
```

使用方式：

```python
def generate_answer(self, context, query, constraints_code,
                    generation_guidance, messages_raw=None):
    # 提取 TEMPTING_ASSUMPTIONS（从 attention_map 代码中正则提取，或直接传递）
    tempting = self._extract_tempting_assumptions(constraints_code)

    if messages_raw:
        # 多轮对话：重构 structured messages，在最后一条中注入约束
        structured = self._reconstruct_messages(messages_raw)
        # 在 system message 中注入约束
        system_content = ANSWER_PROMPT_SYSTEM.format(
            tempting_assumptions=tempting,
            constraints_code=constraints_code[:6000],
            generation_guidance=generation_guidance[:2000],
        )
        structured.insert(0, {"role": "system", "content": system_content})
        return self._call_llm(structured, temperature=0.0)
    else:
        system = ANSWER_PROMPT_SYSTEM.format(
            tempting_assumptions=tempting,
            constraints_code=constraints_code[:6000],
            generation_guidance=generation_guidance[:2000],
        )
        user = ANSWER_PROMPT_USER.format(context=context, query=query)
        return self._call_llm(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.0,
        )
```

---

### Agent 4a: `construct_adversarial`

```python
ADVERSARIAL_PROMPT = (
    "You are a red-team adversarial tester. Your goal is to construct an answer "
    "that PASSES all the following Python test functions but is actually WRONG.\n\n"
    "This exposes weaknesses in the test suite — if you can construct such an "
    "answer, the tests are insufficient.\n\n"
    "=== TEST CODE ===\n"
    "```python\n{constraints_code}\n```\n\n"
    "=== CONTEXT (the ground truth) ===\n"
    "{context}\n\n"
    "=== QUERY ===\n"
    "{query}\n\n"
    "=== YOUR TASK ===\n"
    "1. Read each test function carefully.\n"
    "2. Identify what the tests actually check (string presence, ordering, etc.)\n"
    "3. Construct an answer that:\n"
    "   - Contains all required keywords/phrases to pass string checks\n"
    "   - Maintains required ordering to pass sequence checks\n"
    "   - BUT is factually wrong, misleading, or misses the actual point\n"
    "4. The answer should look plausible but contain subtle errors that the "
    "tests don't catch.\n\n"
    "=== OUTPUT ===\n"
    "If you CAN construct such an adversarial answer, output it directly.\n"
    "If the tests are too tight and you CANNOT find a gap, output exactly: "
    "ADVERSARIAL_IMPOSSIBLE\n\n"
    "Output ONLY the adversarial answer text (or ADVERSARIAL_IMPOSSIBLE). "
    "No explanations."
)
```

使用方式：

```python
def construct_adversarial(self, constraints_code, context, query):
    prompt = ADVERSARIAL_PROMPT.format(
        constraints_code=constraints_code[:8000],
        context=context[:6000],
        query=query[:2000],
    )
    result = self._call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.3,  # 略高温度增加创造性
    )
    if "ADVERSARIAL_IMPOSSIBLE" in result:
        return ""
    return result.strip()
```

---

### Agent 4b: `critique_and_patch`

```python
CRITIQUE_PROMPT = (
    "You are an independent quality auditor. You have access to:\n"
    "1. The ORIGINAL CONTEXT (raw source of truth)\n"
    "2. The GENERATED ANSWER\n"
    "3. The EXISTING TEST CODE (constraints already in place)\n"
    "4. An ADVERSARIAL EXAMPLE showing gaps in the test suite (if available)\n\n"
    "Your job is to find problems that the existing tests DON'T catch, and write "
    "NEW test functions that would catch them.\n\n"
    "=== CRITICAL INSTRUCTION ===\n"
    "Do NOT rely on the attention map or any upstream analysis. Read the original "
    "context YOURSELF with fresh eyes. Look for:\n"
    "- Facts in the context that the answer contradicts or omits\n"
    "- Requirements implied by the query that the answer ignores\n"
    "- Logical inconsistencies within the answer itself\n"
    "- Gaps exposed by the adversarial example (if provided)\n\n"
    "=== EXISTING TESTS (already in place, do not duplicate) ===\n"
    "```python\n{constraints_code}\n```\n\n"
    "{adversarial_section}"
    "=== GENERATED ANSWER ===\n"
    "{answer}\n\n"
    "=== ORIGINAL CONTEXT ===\n"
    "{context}\n\n"
    "=== QUERY ===\n"
    "{query}\n\n"
    "=== OUTPUT ===\n"
    "Write ONLY new test functions (def test_*) that catch problems the existing "
    "tests miss. Each test:\n"
    "- Name starts with test_patch_\n"
    "- Signature: def test_patch_*(answer: str) -> None\n"
    "- Clear assert message explaining what's wrong\n"
    "- May use _oracle(prompt, bool) for semantic checks\n"
    "- Should NOT duplicate any existing test\n\n"
    "If you find NO additional problems, output: # NO_ADDITIONAL_ISSUES\n\n"
    "Output ONLY executable Python code. No markdown, no explanations."
)
```

使用方式：

```python
def critique_and_patch(self, context, query, answer,
                       constraints_code, adversarial_answer):
    adversarial_section = ""
    if adversarial_answer:
        adversarial_section = (
            "=== ADVERSARIAL EXAMPLE (shows test suite gaps) ===\n"
            f"The following answer was constructed to PASS all existing tests "
            f"but is actually wrong. The gaps it exploits should be patched:\n"
            f"{adversarial_answer[:3000]}\n\n"
        )

    prompt = CRITIQUE_PROMPT.format(
        constraints_code=constraints_code[:6000],
        adversarial_section=adversarial_section,
        answer=answer[:4000],
        context=context[:8000],
        query=query[:2000],
    )
    result = self._call_llm(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    if "NO_ADDITIONAL_ISSUES" in result:
        return ""
    return result.strip()
```

---

## Part 2: MetaArchitect CFP Prompt

以下 prompt 用于 `core/meta_architect.py` 的 `_build_cfp_prompt()` 方法。

```python
def _build_cfp_prompt(
    self,
    generation: int,
    mutation_attempt: int,
    parent_code: str,
    parent_performance: ParentPerformance,
    failure_examples: list[dict],
) -> str:
    failures_text = self._format_failures(failure_examples)

    # CFP-specific performance fields
    cfp_perf = ""
    if hasattr(parent_performance, "attention_map_coverage"):
        cfp_perf = (
            f"\n    CFP-specific metrics:"
            f"\n    - Attention map coverage: {parent_performance.attention_map_coverage:.1%}"
            f"\n    - Constraint coverage ratio: {parent_performance.constraint_coverage_ratio:.1%}"
            f"\n    - Critique discovery rate: {parent_performance.critique_discovery_rate:.1%}"
            f"\n    - Adversarial escape rate: {parent_performance.adversarial_escape_rate:.1%}"
            f"\n    - Top bottleneck agents: {parent_performance.top_bottleneck_agents[:3]}"
            f"\n    - Coverage gaps: {parent_performance.coverage_gaps_summary[:3]}"
        )

    return (
        "You are an expert AI systems architect designing CFP "
        "(Cognitive Focus Protocol) compilers.\n\n"
        "PARADIGM: Cognitive Focus Protocol (CFP) with 5-Method Co-Evolution\n"
        "CFP manages long-context understanding through explicit attention allocation.\n"
        "Five agents communicate via executable Python code:\n\n"
        "  Agent 1: build_attention_map(context, query) -> attention map code\n"
        "  Agent 2: generate_constraints(attention_map_code, context, query) -> test code\n"
        "  Agent 3: generate_answer(context, query, constraints, guidance) -> NL answer\n"
        "  Agent 4a: construct_adversarial(constraints, context, query) -> adversarial answer\n"
        "  Agent 4b: critique_and_patch(context, query, answer, constraints, adversarial) -> patch tests\n\n"
        "Pipeline (FIXED runtime -- you do NOT write this):\n"
        "  attention_map = compiler.build_attention_map(context, query)\n"
        "  constraints   = compiler.generate_constraints(attention_map, context, query)\n"
        "  answer        = compiler.generate_answer(context, query, constraints, guidance)\n"
        "  adversarial   = compiler.construct_adversarial(constraints, context, query)\n"
        "  patches       = compiler.critique_and_patch(context, query, answer, constraints, adversarial)\n"
        "  verify(answer, constraints + patches) → repair if needed\n\n"
        "YOUR TASK: Create one class inheriting from BaseCFPProtocol.\n\n"
        "Required methods (ALL five):\n\n"
        "1) build_attention_map(self, context: str, query: str) -> str\n"
        "   - Outputs executable Python code defining: TASK_TYPE, PIPELINE_DEPTH,\n"
        "     ATTENTION_PRIORITIES, TEMPTING_ASSUMPTIONS, WHAT_CONTEXT_SAYS,\n"
        "     WHAT_CONTEXT_DOES_NOT_SAY, CROSS_REFERENCES, OUTPUT_FORMAT\n"
        "   - ATTENTION_PRIORITIES items must have: item, source_location,\n"
        "     risk_level (critical/high/medium/low), risk_type, risk_detail\n\n"
        "2) generate_constraints(self, attention_map_code: str, context: str, query: str) -> str\n"
        "   - Three-tier output: Tier 1 (assert tests), Tier 2 (_oracle tests),\n"
        "     Tier 3 (GENERATION_GUIDANCE string)\n"
        "   - Each test docstring references which attention priority it covers\n"
        "   - Dual-source: use both attention_map AND raw context\n\n"
        "3) generate_answer(self, context: str, query: str, constraints_code: str,\n"
        "                    generation_guidance: str, messages_raw: list = None) -> str\n"
        "   - Answer generator that sees constraints before generating\n"
        "   - Must handle messages_raw for multi-turn conversations\n\n"
        "4) construct_adversarial(self, constraints_code: str, context: str, query: str) -> str\n"
        "   - Red-team: construct answer that passes tests but is wrong\n"
        "   - Return empty string if impossible\n\n"
        "5) critique_and_patch(self, context: str, query: str, answer: str,\n"
        "                       constraints_code: str, adversarial_answer: str) -> str\n"
        "   - Independent review using raw context (bypass Agent 1 blind spots)\n"
        "   - Generate patch test functions (def test_patch_*)\n\n"
        "IMPORTANT: Do NOT implement run() or any execution logic.\n\n"
        "Constraints:\n"
        "1) Import from core.base_cfp_protocol: BaseCFPProtocol\n"
        "1.1) Generate exactly ONE class inheriting BaseCFPProtocol.\n"
        "1.2) Keep method signatures exact.\n"
        "2) Use only allowed libraries (re, json, math, collections, itertools,\n"
        "   functools, typing, dataclasses, copy, random, statistics, string,\n"
        "   operator, abc, enum, datetime, pydantic, networkx).\n"
        "3) No file I/O, no external APIs, no network calls.\n"
        "4) Keep total LLM calls within 30 per task.\n"
        "5) Do NOT use while-loops.\n"
        "6) Never define, assign, or override _oracle; runtime injects it.\n"
        "7) Do NOT truncate context aggressively.\n"
        "8) Return only executable Python code.\n"
        "9) NEVER override immutable runtime methods.\n"
        "10) Output MUST include exactly one top-level BaseCFPProtocol subclass.\n\n"
        f"Current generation: {generation}\n"
        f"Mutation attempt id: {mutation_attempt}\n\n"
        f"Parent performance:\n"
        f"    - Overall accuracy: {parent_performance.overall:.1%}\n"
        f"    - Compilation success: {parent_performance.compilation_success_rate:.1%}\n"
        f"    - Execution success: {parent_performance.execution_success_rate:.1%}\n"
        f"    - F1 (Parametric Override): {parent_performance.f1:.1%}\n"
        f"    - F2 (Context Navigation): {parent_performance.f2:.1%}\n"
        f"    - F3 (Reasoning Breakdown): {parent_performance.f3:.1%}\n"
        f"    - F4 (Induction Failure): {parent_performance.f4:.1%}"
        f"{cfp_perf}\n\n"
        f"Parent compiler:\n```python\n{parent_code}\n```\n\n"
        f"Failure examples:\n{failures_text}\n\n"
        "=== CFP-SPECIFIC IMPROVEMENT STRATEGY ===\n\n"
        "Improvement by bottleneck agent:\n"
        "- agent_1 (attention map): Improve extraction granularity, add missing\n"
        "  risk types, better identify parametric override risks.\n"
        "- agent_2 (constraints): Increase test coverage of attention priorities,\n"
        "  tighten oracle questions, add tests for TEMPTING_ASSUMPTIONS.\n"
        "- agent_3 (answer): Improve how answer utilizes constraints and guidance,\n"
        "  strengthen context-only reasoning instructions.\n"
        "- agent_4 (critique): Improve independent review depth, better exploit\n"
        "  adversarial examples for targeted patches.\n"
        "- agent_2_tests_too_permissive: Tests pass but answer is wrong — add\n"
        "  stricter factual, semantic, and anti-override tests.\n\n"
        "Generalization mandate:\n"
        "- Treat failure examples as pattern evidence, not string templates.\n"
        "- Do not hardcode entities or phrases from failure examples.\n"
        "- Each improvement should generalize across unrelated tasks.\n\n"
        "Output only complete Python code."
    )
```

---

## Part 3: Prompt Design Rationale

### 为什么 Agent 1 的 prompt 不让它回答问题

Agent 1 的 prompt 明确说"You are NOT answering the query"。这是为了防止 LLM 进入"回答模式"而不是"分析模式"。分析模式需要的是元认知——思考"什么地方会出错"，而不是"答案是什么"。

### 为什么 Agent 3 的 prompt 把 TEMPTING_ASSUMPTIONS 放在最前面

LLM 的 attention 在长 prompt 中容易在开头和结尾聚焦（primacy/recency effect）。TEMPTING_ASSUMPTIONS 是最关键的认知陷阱提醒，放在开头确保 LLM 优先处理。

### 为什么 Agent 4a 用略高的 temperature (0.3)

对抗性构造需要创造性——找到 constraints 的漏洞不是确定性任务。temperature=0 可能导致 Agent 4a 总是生成"ADVERSARIAL_IMPOSSIBLE"，因为它倾向于保守。0.3 鼓励它尝试更多可能性。

### 为什么 Agent 4b 的 prompt 强调"不依赖 attention map"

Agent 4b 的核心价值是提供独立视角。如果它依赖 Agent 1 的输出，就会继承 Agent 1 的盲区。prompt 明确要求它"Read the original context YOURSELF with fresh eyes"来保证视角多样性。

### 为什么 MetaArchitect 的 prompt 包含 bottleneck agent 策略

Per-agent failure attribution 是 CFP 相比 TDG 的核心优势。让 Meta-Architect 知道"问题出在哪个 agent"，它就可以做定向 mutation 而非盲目修改整个 protocol。
