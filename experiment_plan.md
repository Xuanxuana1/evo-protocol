# Context-as-Sandbox (CaS): Evolving Neuro-Symbolic Context Compilers with Strictly Typed Neural Oracles

## Experiment Manual

---

## 1. Research Overview

### Paper Title

**Context-as-Sandbox: Migrating Context Learning from Natural Language Space to Symbolic Execution Space via Evolved Neuro-Symbolic Compilers**

### One-Line Pitch

We evolve LLM-generated **Python context compilers** that transform natural language context into executable sandbox environments with strictly typed Neural Oracles, then answer queries via deterministic code execution rather than probabilistic text generation.

### Motivation

CL-bench (Tencent Hunyuan, 2026) exposes a devastating finding: even GPT-5.1 achieves only **23.7% solving rate** on 1,899 expert-annotated context learning tasks. Models fail because their parametric memory actively overrides contextual information -- a phenomenon we call **Parametric Gravity**.

Existing approaches (ALMA, Reflection-Driven Control, Agentic Context Engineering) all operate in "natural language space": structured reflection produces natural language feedback that rewrites natural language interaction protocols. This is fundamentally limited because LLMs process both context and answers in the same probabilistic text space where parametric gravity operates.

**Core Insight**: LLM Text-to-Code has higher logical rigor than Text-to-Text. A Python interpreter won't tolerate parametric gravity hallucinations. If the sandbox encodes `banana.color = "purple"` with `assert banana.color == "purple"`, any code outputting `"yellow"` raises an `AssertionError` at runtime. By migrating context learning from natural language space to symbolic execution space, we exploit the deterministic guarantees of code execution to counteract probabilistic context-learning failures.

### Key Differentiation

| Dimension | GEPA | ALMA | Evo-Protocol v1 | **CaS (Ours)** |
|-----------|------|------|-----------------|----------------|
| Search object | System prompts (NL) | Memory modules (code) | Full protocols (prompt + code) | **Context compilers (NL -> executable sandbox)** |
| Answer space | Probabilistic text | Probabilistic text | Probabilistic text | **Deterministic code execution** |
| Context repr. | Raw text | Structured memory | Structured signals | **Live Python objects + Neural Oracles** |
| Failure signal | Binary | Reflection text | 4-mode classifier | **2-stage pipeline + 4-mode classifier** |
| Evolution target | Prompt tokens | Memory ops | perception/cognition/verify | **compile_sandbox / generate_solver** |
| Verification | LLM-based (soft) | LLM-based (soft) | LLM-based (soft) | **Python interpreter (hard)** |

---

## 2. Architecture: 2-Method Co-Evolution

### Design Rationale

We choose a **2-method interface** over alternatives:

- **3-method (compile/query/verify)**: Re-introducing a `verify()` method reintroduces the probabilistic "softness" we are trying to escape. The Python interpreter IS the verifier -- AssertionErrors, ValidationErrors, and NameErrors are deterministic verification.

- **Single compile() only**: Data structures and algorithms are inextricably coupled. If the Meta-Agent evolves a NetworkX graph for spatial reasoning, a fixed solver using dict lookups will catastrophically fail. The Meta-Agent must co-evolve both representation and action.

- **2-method (compile/solve)**: The sweet spot. `compile_sandbox` isolates Representation (defeating Context Navigation Failure). `generate_solver` isolates Action (defeating Parametric Override and Reasoning Breakdown). The execution runtime is FIXED infrastructure.

### Base Interface

```python
class BaseCaSCompiler(ABC):
    """The Meta-Agent evolves ONLY these two methods."""

    def compile_sandbox(self, context: str) -> str:
        """Phase 1: Representation.
        LLM generates Python code creating typed objects from context.
        Returns executable Python code string."""

    def generate_solver(self, query: str, sandbox_schema: str) -> str:
        """Phase 2: Action.
        LLM generates Python code querying sandbox objects.
        Must set FINAL_ANSWER variable. Returns code string."""
```

### Fixed Deterministic Runtime (Immutable)

```python
def deterministic_runtime(compiler, context, query):
    # Phase 1: Compile context
    env_code = compiler.compile_sandbox(context)

    # Phase 2: Generate solver
    solver_code = compiler.generate_solver(query, env_code)

    # Phase 3: DETERMINISTIC execution (the Ultimate Verifier)
    namespace = {}
    exec(env_code + "\n" + solver_code, namespace)
    return namespace.get("FINAL_ANSWER")
    # AssertionError -> constraint violation (gravity blocked!)
    # NameError -> missing fact (context navigation failure!)
    # TypeError -> type mismatch (reasoning breakdown!)
```

The Meta-Agent CANNOT modify the runtime. It can only evolve how it prompts the LLM inside `compile_sandbox` and `generate_solver`.

### Two-Stage Failure Decomposition

```
Context --> compile_sandbox --> env_code --> generate_solver --> solver_code --> exec() --> FINAL_ANSWER
              compile error?                   bad solver?        runtime error?    wrong answer?
              (Stage 1)                        (Stage 2)          (Stage 3)         (Stage 4)
```

---

## 3. Neural Oracles: Bridging the Semantic-Symbolic Gap

### The Problem

Pure symbolic compilation is impossible for all natural language. Nuanced prose, implicit sentiment, and ambiguous pragmatics cannot always be extracted into Pydantic schemas or NetworkX DAGs. But unconstrained LLM reasoning is vulnerable to Parametric Gravity.

### The Solution: Strictly Typed Neural Oracles

CaS introduces a middle ground: **Neural Oracles** -- LLM calls that are strictly bounded by type signatures and single-task perception.

The sandbox code can call `_oracle(prompt, return_type)` where:
- `_oracle(prompt, bool)` -> True/False only
- `_oracle(prompt, int)` -> single integer
- `_oracle(prompt, float)` -> single decimal
- `_oracle(prompt, str)` -> brief text (one sentence max)

**Key principle**: Python handles the syntax of reasoning (loops, conditionals, state tracking). The LLM acts as a semantic primitive (a perception function that takes text and returns typed values).

### Example: Nuanced Contract Analysis

Context: *"Despite the standard 30-day return policy, the manager's vaguely dismissive tone implies they have no intention of honoring it."*

```python
# Compiled Sandbox Environment
class ReturnPolicyState:
    def __init__(self, context_text):
        self.context = context_text
        self.standard_policy_days = 30

    def is_voided_by_behavior(self):
        """Neural Oracle: semantic perception, NOT reasoning."""
        return _oracle(
            f"Does this text imply refusal to honor agreement? {self.context}",
            bool
        )

policy = ReturnPolicyState(CONTEXT)

# Solver Code (deterministic logic)
if policy.is_voided_by_behavior():
    FINAL_ANSWER = "0 days (contract effectively void)"
else:
    FINAL_ANSWER = f"{policy.standard_policy_days} days"
```

**Why this works**:
- The LLM is never asked the final query (where parametric priors would dominate)
- Only asked a micro-perception question: "Is the tone dismissive?"
- All reasoning (if void -> 0, else -> 30) is deterministic Python

### Textual Backpropagation

Neural Oracles enable precise failure attribution:
- If `_oracle` returns wrong perception -> improve oracle prompt in `compile_sandbox`
- If logic is wrong despite correct perception -> fix solver code in `generate_solver`
- If sandbox missing facts -> add more objects in `compile_sandbox`

---

## 4. Compilation Strategy Taxonomy

| Context Type | Data Structure | Oracle Use | Example |
|-------------|---------------|-----------|---------|
| Factual/Override-prone | Pydantic + validators | Minimal | `assert planet.gravity == 3.7` |
| Spatial/Relational | networkx.Graph | Edge semantics | `_oracle("Is X reachable from Y?", bool)` |
| Temporal/Sequential | Enums, state machines | State transitions | `_oracle("Has state changed?", bool)` |
| Rule-based | Dict + callable functions | Rule interpretation | `_oracle("Does exception apply?", bool)` |
| Tabular | list[dataclass] | N/A | Direct attribute lookup |
| Nuanced/Semantic | Oracle-heavy classes | Heavy | `_oracle("What is the implied meaning?", str)` |

---

## 5. Four Failure Modes Through CaS Lens

### F1: Parametric Override
- **Mechanism**: Solver code uses parametric knowledge instead of sandbox values
- **CaS defense**: Assertions in sandbox catch violations at runtime (AssertionError)
- **Evolution response**: Add Pydantic validators, more assertions, Neural Oracles for semantic verification

### F2: Context Navigation Failure
- **Mechanism**: `compile_sandbox` misses encoding a fact -> NameError when solver accesses it
- **CaS defense**: Missing data = missing namespace key = immediate runtime error
- **Evolution response**: More comprehensive fact extraction, Neural Oracles for ambiguous facts

### F3: Reasoning Breakdown
- **Mechanism**: `generate_solver` produces logically flawed code
- **CaS defense**: Python executes exactly what's written -- no probabilistic collapse
- **Evolution response**: Move multi-step logic to Python loops/functions, use oracles only for perception

### F4: Induction Failure
- **Mechanism**: `compile_sandbox` doesn't create callable rule functions from examples
- **CaS defense**: Pattern application becomes deterministic lookup/function call
- **Evolution response**: Create lookup tables and callable rule functions

---

## 6. Meta-Agent Evolution

### Evolution Loop

```
for generation in 1..N:
    for candidate in population:
        for task in sampled_tasks:
            env_code    = candidate.compile_sandbox(task.context)
            solver_code = candidate.generate_solver(task.query, env_code)
            result      = exec(env_code + solver_code)  # FIXED runtime
            score       = judge(result.FINAL_ANSWER, task.rubrics)
            if score < 1: classify_failure(stage, mode)

    fitness = 0.6 * accuracy + 0.2 * exec_rate + 0.2 * compile_rate
    elites = select_top_k(population, fitness)
    children = meta_architect.mutate(elites, failure_traces)
    population = elites + children
```

### Compiler Library (Continuous Learning)

Successful compilation strategies accumulate in a persistent library indexed by domain and data structure type. Over generations, the system discovers:
- Physics contexts -> `networkx.Graph` for spatial relationships
- Legal contexts -> FSMs for procedural rules
- Factual contexts -> Pydantic models with validators
- Nuanced contexts -> Oracle-heavy classes

---

## 7. Experimental Design

### Baselines

| Baseline | Type | Description |
|----------|------|-------------|
| Naive | Legacy | Direct LLM call, no verification |
| CoT | Legacy | Chain-of-thought with LLM verification |
| ReAct | Legacy | Evidence extraction + synthesis |
| Evo-Protocol v1 | Legacy | Evolved perception/cognition/verification |
| **CaS Seed** | CaS | General-purpose compilation + solver |
| **CaS Naive** | CaS | Flat key-value dict compilation |
| **CaS Pydantic** | CaS | Always-Pydantic model compilation |
| **CaS Evolved** | CaS | Meta-Agent evolved compiler (main result) |

### Metrics

1. **Solving Rate**: Binary (0/1) per task
2. **Compilation Success Rate**: Fraction producing valid namespaces
3. **Execution Success Rate**: Fraction where solver executes without error
4. **Per-Mode Accuracy**: By F1/F2/F3/F4 failure modes
5. **Attention Drift**: 0-1 parametric override measure
6. **Oracle Utilization**: Oracles per task, oracle accuracy
7. **Token Efficiency**: Solving rate per token

### Fitness Function

```
fitness = 0.6 * answer_correctness + 0.2 * execution_success_rate + 0.2 * compilation_success_rate
```

---

## 8. Expected Results

### Hypothesis 1: CaS > Text-to-Text
CaS should outperform Evo-Protocol v1 because:
- F1 failures caught by runtime assertions, not probabilistic verification
- F3 failures reduced because reasoning is deterministic Python
- F4 failures reduced because induction becomes lookup construction

### Hypothesis 2: Emergent Specialization
After 10+ generations, the Meta-Agent discovers domain-specific strategies:
- Graphs for spatial, Pydantic for factual, FSMs for temporal
- Neural Oracles emerge for nuanced/semantic contexts

### Hypothesis 3: Oracle-Symbol Balance
The Meta-Agent learns to minimize oracle usage for contexts amenable to symbolic extraction and increase oracle usage for nuanced contexts. This represents learned "perception vs. logic" boundary detection.

### Hypothesis 4: Two-Stage Signal > One-Stage
The compile/solve failure decomposition enables faster convergence.

---

## 9. Repository Structure

```
context_learning/
├── core/
│   ├── base_protocol.py              # Legacy protocol ABC
│   ├── base_sandbox_protocol.py      # BaseCaSCompiler: 2-method interface (NEW)
│   ├── sandbox_executor.py           # Safe execution + Neural Oracle injection (NEW)
│   ├── protocol_loader.py            # ProtocolLoader + SandboxProtocolLoader
│   ├── meta_architect.py             # CaS-aware mutation prompts
│   ├── evolution_loop.py             # Mode-switched evolution engine
│   ├── failure_classifier.py         # Stage-aware failure analysis
│   ├── compiler_library.py           # Strategy accumulation (NEW)
│   ├── archive.py                    # SHA-indexed protocol storage
│   ├── self_repair.py                # Validation-guided code repair
│   ├── evaluator.py                  # Benchmark evaluation helpers
│   ├── token_tracker.py              # Token accounting
│   └── env_utils.py                  # Environment variable helpers
├── baselines/
│   ├── naive.py                      # Legacy: direct call
│   ├── cot.py                        # Legacy: chain-of-thought
│   ├── react.py                      # Legacy: evidence extraction
│   ├── cas_seed.py                   # CaS: general compilation (NEW)
│   ├── cas_naive.py                  # CaS: flat dict (NEW)
│   └── cas_pydantic.py              # CaS: Pydantic models (NEW)
├── benchmarks/
│   ├── base.py                       # TaskRecord, BaseBenchmark
│   └── cl_bench.py                   # CL-bench implementation
├── configs/
│   └── evolution.yaml                # mode: cas, fitness_weights, etc.
├── run_evolution.py                  # CLI: --mode cas|legacy
├── run_baselines.py                  # CLI: CaS baselines registered
├── eval.py                           # Judge evaluation
├── infer.py                          # Direct model inference
└── requirements.txt                  # pydantic, networkx added
```

---

## 10. Running Experiments

### Quick Start (CaS mode)
```bash
python run_evolution.py --mode cas --generations 5 --population-size 4 --tasks-per-eval 20
```

### Legacy Mode (Evo-Protocol v1)
```bash
python run_evolution.py --mode legacy --generations 5 --population-size 4 --tasks-per-eval 20
```

### CaS Baselines
```bash
python run_baselines.py --baseline cas_seed --split test
python run_baselines.py --baseline cas_naive --split test
python run_baselines.py --baseline cas_pydantic --split test
```

---

## 11. Verification Plan

1. **Smoke test**: `python run_evolution.py --mode cas --generations 1 --population-size 2 --tasks-per-eval 5` -- no crashes
2. **Compilation test**: Feed 10 diverse contexts to `SeedCaSCompiler.compile_sandbox()` -- verify >80% produce valid code
3. **Oracle test**: Verify `_oracle(prompt, bool)` returns strictly True/False in sandbox
4. **End-to-end**: Run 3 generations on 20 tasks -- verify fitness does not degrade
5. **Comparison**: Run all baselines on test split -- compare solving rates
6. **Emergence check**: After 10+ generations, inspect compiler templates for emergent structures
