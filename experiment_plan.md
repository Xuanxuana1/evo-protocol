# Evolving Context Learning: From Context-as-Sandbox to Test-Driven Generation

## Experiment Manual

---

## 1. Research Overview

### Paper Title

**Evolving Context Learning via Test-Driven Generation: Deterministic Verification of Natural Language Answers through Evolved Python Test Suites**

### One-Line Pitch

We evolve LLM-generated **Python test compilers** that validate natural language answers against context-derived test functions, using deterministic Python execution as the verifier while preserving full NL generation capabilities.

### Motivation

CL-bench (Tencent Hunyuan, 2026) exposes a devastating finding: even GPT-5.1 achieves only **23.7% solving rate** on 1,899 expert-annotated context learning tasks. Models fail because their parametric memory actively overrides contextual information -- a phenomenon we call **Parametric Gravity**.

**CaS (Context-as-Sandbox)** attempted to address this by compiling context into executable Python objects, then answering queries via deterministic code execution. However, CaS faces critical limitations:
1. Context truncated to fit code generation windows (tasks up to 150k chars)
2. The compile-then-solve pipeline cannot produce persona/tone/style-compliant NL
3. Evolution stalls when accuracy = 0 (no fitness gradient)

**TDG (Test-Driven Generation)** keeps CaS's core insight (deterministic Python verification) but flips the pipeline:
- **CaS**: compile context -> code objects -> code solver -> extract answer
- **TDG**: compile context -> **test functions** -> LLM generates NL answer -> **Python verifies** -> repair if needed

Key property: **worst case degrades to direct inference** (never worse than baseline).

### Key Differentiation

| Dimension | GEPA | ALMA | CaS | **TDG (Ours)** |
|-----------|------|------|-----|----------------|
| Search object | System prompts (NL) | Memory modules (code) | Context compilers | **Test compilers + answer generators** |
| Answer space | Probabilistic text | Probabilistic text | Deterministic code | **NL text, verified by code** |
| Context repr. | Raw text | Structured memory | Live Python objects | **Python test functions** |
| Context limit | Full | Full | Truncated | **Full (no truncation)** |
| Failure signal | Binary | Reflection text | 2-stage pipeline | **Per-test pass/fail + repair** |
| Evolution target | Prompt tokens | Memory ops | compile/solve | **compile_tests / generate_answer** |
| Verification | LLM-based (soft) | LLM-based (soft) | Python interpreter (hard) | **Python test runner (hard)** |
| Worst case | Baseline | Baseline | Empty answer | **Direct inference (= baseline)** |

---

## 2. Architecture: TDG 4-Stage Pipeline

### Design Rationale

TDG addresses CaS's limitations while preserving its strengths:

1. **Full context access**: `generate_answer` receives the complete context (no truncation)
2. **NL answer quality**: Answers are generated as natural language, not extracted from code
3. **Deterministic verification**: Python tests catch factual errors, format violations, and constraint breaches
4. **Graceful degradation**: If test compilation fails, the answer is returned as-is (= direct inference)
5. **Dense fitness signal**: `test_pass_rate` provides a gradient even when rubric accuracy = 0

### Base Interface

```python
class BaseTDGCompiler(ABC):
    """The Meta-Agent evolves ONLY these two methods."""

    def compile_tests(self, context: str, query: str) -> str:
        """Phase 1: Test Generation.
        LLM generates Python test functions from context+query.
        Each test_* function accepts an answer string and asserts properties.
        Returns executable Python code string."""

    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        """Phase 2: Answer Generation.
        LLM generates a natural language answer from full context+query.
        Returns answer string."""
```

### Fixed Deterministic Runtime (Immutable)

```
Input: context + query + messages_raw
         |
    [Phase 1] compile_tests(context, query)
         -> test_code (Python with def test_* functions)
         -> sanitize, validate syntax, extract test names
         |
    [Phase 2] generate_answer(context, query, messages_raw)
         -> draft answer (natural language)
         |
    [Phase 3] Verify: run each test_*(answer) in sandbox
         -> {test_name: True/False} dict
         |
    [Phase 4] Repair loop (if any test failed):
         -> build feedback from failed test names + test code
         -> re-generate answer with feedback
         -> re-verify (up to max_retries)
         |
Output: SandboxResult with answer + test_pass_rate + metadata
```

**Graceful degradation**: If compile_tests fails or no test_* functions found -> return generate_answer result as-is.

### Test Runner Pattern

```python
# Avoids banned locals() call
runner = test_code + "\n\n"
runner += f"_answer = {repr(answer)}\n"
runner += "_test_results = {}\n"
for name in test_names:
    runner += f"try:\n    {name}(_answer)\n    _test_results['{name}'] = True\n"
    runner += f"except Exception:\n    _test_results['{name}'] = False\n"
```

---

## 3. Neural Oracles in Tests

### Semantic Testing

Tests can call `_oracle(prompt, return_type)` for checks that cannot be expressed as string operations:

```python
def test_formal_tone(answer):
    """Neural Oracle: semantic perception check."""
    assert _oracle(f"Is this text written in a formal tone? {answer}", bool)

def test_factual_accuracy(answer):
    """Hybrid: string + oracle check."""
    assert "purple" in answer.lower()  # Deterministic
    assert _oracle(f"Does this answer correctly describe the sky as purple? {answer}", bool)
```

### Anti-Parametric-Override Tests

```python
def test_context_specific_facts(answer):
    """Assert facts that contradict common knowledge."""
    # Context says "the capital of France is Lyon" (counterfactual)
    assert "lyon" in answer.lower(), "Answer must use context fact, not parametric knowledge"
    assert "paris" not in answer.lower(), "Answer must not use parametric override"
```

---

## 4. CaS vs TDG Comparison

### CaS Pipeline (Kept as Baseline)

```
Context -> compile_sandbox -> env_code -> generate_solver -> solver_code -> exec() -> FINAL_ANSWER
```

**Strengths**: Full deterministic execution, no LLM in final answer path
**Weaknesses**: Context truncation, cannot produce NL, no fitness gradient at accuracy=0

### TDG Pipeline (New)

```
Context -> compile_tests -> test_code -> generate_answer -> NL answer -> verify -> repair -> answer
```

**Strengths**: Full context, NL answers, graceful degradation, dense fitness signal
**Weaknesses**: Tests may be too permissive, answer repair limited by LLM calls

---

## 5. Five Failure Modes Through TDG Lens

### F1: Parametric Override
- **Mechanism**: LLM generates answer using parametric knowledge instead of context
- **TDG defense**: Anti-override tests assert context-specific facts that contradict common knowledge
- **Evolution response**: More anti-parametric-override tests, stronger factual assertions

### F2: Context Navigation Failure
- **Mechanism**: Answer misses key evidence from context
- **TDG defense**: Keyword/fact presence tests catch missing evidence
- **Evolution response**: More comprehensive factual extraction tests

### F3: Reasoning Breakdown
- **Mechanism**: Multi-step reasoning collapses in answer generation
- **TDG defense**: Tests check intermediate reasoning steps and logical consistency
- **Evolution response**: Tests for step-by-step reasoning artifacts

### F4: Induction Failure
- **Mechanism**: Pattern/rule application fails
- **TDG defense**: Tests check that patterns from context examples are applied correctly
- **Evolution response**: Pattern application tests derived from context examples

### F5: Test Quality (TDG-specific)
- **Mechanism**: Tests are too permissive (pass bad answers) or too strict (reject good answers)
- **TDG defense**: Evolution optimizes test quality via test_pass_rate + accuracy correlation
- **Evolution response**: Better test design, balanced strictness

---

## 6. Meta-Agent Evolution

### Evolution Loop

```
for generation in 1..N:
    for candidate in population:
        for task in sampled_tasks:
            test_code = candidate.compile_tests(task.context, task.query)
            answer    = candidate.generate_answer(task.context, task.query)
            results   = run_tests(test_code, answer)         # DETERMINISTIC
            if any_failed: answer = repair(answer, failed)   # Up to max_retries
            score     = judge(answer, task.rubrics)           # Rubric evaluation

    fitness = 0.55 * accuracy + 0.25 * test_pass_rate + 0.15 * exec_rate + 0.05 * compile_rate
    elites = select_top_k(population, fitness)
    children = meta_architect.mutate(elites, failure_traces)
    population = elites + children
```

### Fitness Function Comparison

| Mode | answer_correctness | test_pass_rate | execution_success | compilation_success |
|------|-------------------|----------------|-------------------|---------------------|
| CaS | 0.80 | N/A | 0.10 | 0.10 |
| **TDG** | **0.55** | **0.25** | **0.15** | **0.05** |

TDG's `test_pass_rate` weight provides a dense signal even when accuracy = 0, enabling evolution to find a gradient.

---

## 7. Experimental Design

### Baselines

| Baseline | Type | Description |
|----------|------|-------------|
| Naive | Legacy | Direct LLM call, no verification |
| CoT | Legacy | Chain-of-thought with LLM verification |
| ReAct | Legacy | Evidence extraction + synthesis |
| Evo-Protocol v1 | Legacy | Evolved perception/cognition/verification |
| CaS Seed | CaS | General-purpose compilation + solver |
| CaS Evolved | CaS | Meta-Agent evolved compiler |
| **TDG Seed** | TDG | General-purpose test generation + answer |
| **TDG Evolved** | TDG | Meta-Agent evolved test compiler (main result) |

### Metrics

1. **Solving Rate**: Binary (0/1) per task
2. **Test Pass Rate**: Fraction of tests passed per task (TDG only)
3. **Compilation Success Rate**: Fraction of tasks where tests compile (TDG) or sandbox compiles (CaS)
4. **Execution Success Rate**: Fraction where answer is generated successfully
5. **Per-Mode Accuracy**: By F1/F2/F3/F4 failure modes
6. **Attention Drift**: 0-1 parametric override measure
7. **Token Efficiency**: Solving rate per token

### Ablation Studies

| Variant | compile_tests | generate_answer | verify | repair |
|---------|:---:|:---:|:---:|:---:|
| TDG Full | Y | Y | Y | Y |
| No-tests | - | Y | - | - |
| No-repair | Y | Y | Y | - |
| No-evolution | Y (seed only) | Y (seed only) | Y | Y |

---

## 8. Hypotheses

### H1: TDG > Direct Inference
TDG's test-verify-repair loop should improve accuracy over direct inference by catching and correcting parametric override and factual errors.

### H2: TDG >= CaS on CL-bench
TDG should match or exceed CaS because:
- No context truncation (full 150k char access)
- NL answers preserve persona/tone/style
- Graceful degradation floor = direct inference (not empty answer)
- Dense fitness signal enables evolution even when accuracy = 0

### H3: Test Quality Evolves
After 10+ generations, the Meta-Agent discovers domain-specific test strategies:
- Factual contexts -> assertion-heavy tests
- Semantic contexts -> oracle-heavy tests
- Format-constrained -> regex/structure tests

### H4: Dense Signal Enables Evolution
TDG's test_pass_rate weight (0.25) provides a fitness gradient even when rubric accuracy = 0, enabling evolution to proceed where CaS stalls.

### H5: Repair Loop Has Diminishing Returns
Most improvements come from the first repair attempt. Additional attempts have diminishing marginal value, suggesting max_retries=2 is sufficient.

---

## 9. Repository Structure

```
context_learning/
+-- core/
|   +-- base_protocol.py              # Legacy protocol ABC
|   +-- base_sandbox_protocol.py      # BaseCaSCompiler: 2-method interface
|   +-- base_tdg_protocol.py          # BaseTDGCompiler: 4-stage pipeline (NEW)
|   +-- sandbox_executor.py           # Safe execution + Neural Oracle injection
|   +-- protocol_loader.py            # ProtocolLoader + SandboxProtocolLoader + TDGProtocolLoader
|   +-- meta_architect.py             # CaS + TDG mutation prompts
|   +-- evolution_loop.py             # Mode-switched evolution engine (cas/tdg/legacy)
|   +-- failure_classifier.py         # Stage-aware failure analysis (TDG-aware)
|   +-- compiler_library.py           # Strategy accumulation
|   +-- archive.py                    # SHA-indexed protocol storage
|   +-- self_repair.py                # Validation-guided code repair (TDG-aware)
|   +-- evaluator.py                  # Benchmark evaluation helpers (TDG-aware)
|   +-- token_tracker.py              # Token accounting
|   +-- env_utils.py                  # Environment variable helpers
+-- baselines/
|   +-- naive.py                      # Legacy: direct call
|   +-- cot.py                        # Legacy: chain-of-thought
|   +-- react.py                      # Legacy: evidence extraction
|   +-- cas_seed.py                   # CaS: general compilation
|   +-- cas_naive.py                  # CaS: flat dict
|   +-- cas_pydantic.py               # CaS: Pydantic models
|   +-- tdg_seed.py                   # TDG: general test generation (NEW)
+-- benchmarks/
|   +-- base.py                       # TaskRecord, BaseBenchmark
|   +-- cl_bench.py                   # CL-bench implementation
+-- configs/
|   +-- evolution.yaml                # mode: cas/tdg, fitness_weights, etc.
+-- run_evolution.py                  # CLI: --mode cas|tdg|legacy
+-- run_baselines.py                  # CLI: CaS + TDG baselines
+-- eval.py                           # Judge evaluation
+-- infer.py                          # Direct model inference
+-- requirements.txt                  # Dependencies
```

---

## 10. Running Experiments

### TDG Mode (Main Experiment)
```bash
python run_evolution.py --mode tdg --generations 30 --population-size 8 --tasks-per-eval 50
```

### TDG Smoke Test
```bash
python run_evolution.py \
  --no-config --mode tdg --env-file .env \
  --data-path data/CL-bench.jsonl --split train \
  --generations 2 --population-size 2 --elite-count 1 \
  --tasks-per-eval 3 --skip-final-eval \
  --output outputs/smoke/tdg_smoke.json
```

### CaS Mode (Baseline Comparison)
```bash
python run_evolution.py --mode cas --generations 30 --population-size 8 --tasks-per-eval 50
```

### Legacy Mode (Evo-Protocol v1)
```bash
python run_evolution.py --mode legacy --generations 5 --population-size 4 --tasks-per-eval 20
```

### Import Smoke Test
```bash
python -c "
from baselines.tdg_seed import SeedTDGCompiler
from core.protocol_loader import TDGProtocolLoader
print('Import OK')
"
```

---

## 11. Verification Plan

1. **Import test**: Verify all new classes import without errors
2. **Smoke test**: Run TDG evolution for 2 generations on 3 tasks -- no crashes
3. **Single-task test**: Run SeedTDGCompiler on 1 task, verify:
   - compile_tests produces valid Python with def test_* functions
   - generate_answer produces non-empty NL text
   - run_tests returns a dict with test results
   - SandboxResult has correct fields (test_pass_rate, test_results, repair_attempts)
4. **Graceful degradation**: Verify that when compile_tests fails, generate_answer result is returned as-is
5. **CaS regression**: Run same CaS smoke test, verify identical behavior
6. **Comparison**: Run both CaS and TDG on same 5 tasks, compare accuracy and test_pass_rate
