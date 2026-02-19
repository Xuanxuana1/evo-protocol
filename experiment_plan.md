# Evo-Protocol: Taming Parametric Gravity via Evolving Executable Neuro-Symbolic Protocols in Code Space

## Experiment Manual

---

## 1. Research Overview

### Paper Title

**Evo-Protocol: Taming Parametric Gravity via Evolving Executable Neuro-Symbolic Protocols in Code Space for Robust Context Learning**

### One-Line Pitch

We use a Meta-Architect LLM to **evolve executable Python protocols** (not just prompts or memory modules) that impose deterministic hard constraints on LLM reasoning, counteracting the probabilistic fragility that causes context learning failures.

### Motivation

CL-bench (Tencent Hunyuan, 2026) exposes a devastating finding: even the strongest model (GPT-5.1) achieves only **23.7% solving rate** on 1,899 expert-annotated context learning tasks, with an average of 17.2% across all models. The benchmark requires models to learn new knowledge *from context*---domain-specific rules, procedures, and empirical laws that are **absent from pre-training**. Models fail because their parametric memory actively overrides contextual information.

We term this phenomenon **Parametric Gravity**: the tendency of LLM attention heads to activate pre-trained patterns even when the context explicitly contradicts them. This manifests in four failure modes (Parametric Override, Context Navigation Failure, Reasoning Breakdown, Induction Failure), each requiring different countermeasures.

### Key Differentiation

| Dimension | GEPA (2025/2026) | ALMA (Meta-Memory) | MCTS Arch Search | **Evo-Protocol (Ours)** |
|-----------|-------------------|---------------------|-------------------|--------------------------|
| Search object | System prompts (NL) | Memory modules (code) | Agent topology (graph) | **Full protocols (prompt + code logic)** |
| Constraint type | Soft (persuasion) | Soft (retrieval) | Structural only | **Hard (executable gates)** |
| Core problem | Sample efficiency | Continual memory | Architecture | **Parametric gravity** |
| Evolution target | Prompt text | `update()`/`retrieve()` | DAG structure | **`perception`/`cognition`/`verification`** |
| Output artifact | A better prompt | A memory class | A topology graph | **An executable protocol class** |

---

## 2. Problem Formulation

### Mathematical Definition

Let $x$ denote an input query and $c$ denote a novel context containing knowledge absent from pre-training. We define two competing distributions:

- **Parametric distribution**: $P_{\text{param}}(y \mid x) = \text{softmax}(W_{\text{head}} \cdot h_L(x))$, where $h_L$ is the final hidden state driven by pre-trained weights.
- **Contextual distribution**: $P_{\text{context}}(y \mid c, x)$, the ideal distribution that faithfully reflects the context $c$.

**Parametric Gravity** is the phenomenon where, even given $c$:

$$P_{\text{model}}(y \mid c, x) \approx \alpha \cdot P_{\text{param}}(y \mid x) + (1 - \alpha) \cdot P_{\text{context}}(y \mid c, x)$$

with $\alpha \gg 0.5$ in practice, especially when $P_{\text{param}}$ is concentrated (high-confidence pre-trained knowledge).

### Four Failure Modes

| Mode | Formal Description | Estimated Prevalence |
|------|--------------------|----------------------|
| **F1: Parametric Override** | $\arg\max P_{\text{param}}(y \mid x) \neq \arg\max P_{\text{context}}(y \mid c, x)$ and model outputs $\arg\max P_{\text{param}}$ | ~40% |
| **F2: Context Navigation Failure** | $\text{Attention}(Q, K_c) \to 0$ for relevant spans in $c$ due to length or distractor dilution; model acts as if $c = \emptyset$ | ~25% |
| **F3: Reasoning Breakdown** | Intermediate state $h_t$ entropy grows: $H(h_t) > H(h_{t-1})$ during multi-step reasoning, causing chain collapse | ~20% |
| **F4: Induction Failure** | Model fails to extract latent function $f$ from examples in $c$; generates plausible but fabricated $\hat{f} \neq f$ | ~15% |

### Optimization Objective

Find a protocol $\pi^* \in \Pi_{\text{code}}$ (the space of all valid Python protocol classes) such that:

$$\pi^* = \arg\max_{\pi \in \Pi_{\text{code}}} \mathbb{E}_{(c,x,y^*) \sim \mathcal{D}} \left[ \mathbb{1}[\pi(c, x) = y^*] \right]$$

subject to:
- $\pi$ is executable Python code inheriting from `BaseProtocol`
- $\pi$ uses LLM calls only through a sandboxed API
- Token budget constraint: $\text{Tokens}(\pi(c,x)) \leq B$

---

## 3. Methodology: Evo-Protocol Framework

### System Architecture

```
+=====================================================================+
|                        EVO-PROTOCOL SYSTEM                          |
+=====================================================================+
|                                                                     |
|  +---------------------+     +--------------------------+           |
|  | Benchmark Layer      |     |    Meta-Architect        |           |
|  | (Pluggable)          |     |    (GPT-4o / Claude)     |           |
|  |                      |     |                          |           |
|  | BaseBenchmark        |     | 1. Read failure logs     |           |
|  |  ├ CLBenchEnv (now)  |     | 2. Analyze root cause    |           |
|  |  ├ ARC_Env (future)  |     | 3. Generate/mutate code  |           |
|  |  └ GPQA_Env (future) |     | 4. Emit new Protocol     |           |
|  +--------+-------------+     +------------+-------------+           |
|           |                                |                         |
|           | TaskRecord                     | Python class file        |
|           v                                v                         |
|  +-----------------------------------------------------------+      |
|  |          Protocol Runtime (Sandboxed)                      |      |
|  |                                                            |      |
|  |  +------------------+  +--------------+  +-------------+   |      |
|  |  | perception(ctx)  |->| cognition()  |->| verify()    |   |      |
|  |  | (info filter)    |  | (reasoning)  |  | (judicial)  |   |      |
|  |  +------------------+  +--------------+  +-------------+   |      |
|  |                                                            |      |
|  |  Dynamic code loading: find_subclass(file, BaseProtocol)   |      |
|  |  Calls LLM via sandboxed OpenAI-compatible API             |      |
|  +----------------------------+-------------------------------+      |
|                               |                                      |
|                               | TaskRecord (with result)             |
|                               v                                      |
|  +-----------------------------------------------------------+      |
|  |          Feedback Loop                                     |      |
|  |                                                            |      |
|  |  - Per-task score via BaseBenchmark.evaluate()             |      |
|  |  - Failure mode classification (F1/F2/F3/F4)              |      |
|  |  - Token usage (TokenTracker)                              |      |
|  |  - Structured error trace for Meta-Architect               |      |
|  +-----------------------------------------------------------+      |
|                               |                                      |
|                               v                                      |
|  +-----------------------------------------------------------+      |
|  |          Protocol Archive (SHA-indexed)                    |      |
|  |                                                            |      |
|  |  protocol_{sha}.py  + metadata.json (score, parent, gen)   |      |
|  +-----------------------------------------------------------+      |
|                                                                      |
+======================================================================+
```

### Benchmark Abstraction Layer (Inspired by ALMA `Basic_Env` / `Basic_Recorder`)

To support future benchmarks beyond CL-bench, we introduce a pluggable benchmark layer. This borrows ALMA's key insight: **decouple the evolution engine from any specific benchmark by defining a uniform task interface**.

ALMA uses `Basic_Env` (with `set_task_env`, `run_step`, `cal_reward`, `get_prompt`) + `Basic_Recorder` (trajectory logging). Those are designed for **multi-step interactive environments** (ALFWorld, TextWorld). Our tasks are **single-turn or multi-turn QA** — a simpler but different paradigm. We adapt the pattern accordingly:

```python
"""
benchmarks/base.py — Abstract benchmark interface.

Inspired by ALMA's Basic_Env/Basic_Recorder, adapted for QA-style
context learning tasks instead of interactive environments.

Any benchmark that provides (context, query, ground_truth) triples
can plug into the evolution loop by implementing BaseBenchmark.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------
# TaskRecord: the universal data unit flowing through the system
# (analogous to ALMA's Basic_Recorder, but for QA tasks)
# ---------------------------------------------------------------
@dataclass
class TaskRecord:
    """
    A single task instance that flows through the entire pipeline.

    Lifecycle:
      1. BaseBenchmark.load_tasks() creates TaskRecords from raw data
      2. Protocol.run() populates model_output and reasoning_trace
      3. BaseBenchmark.evaluate() populates score and eval_detail
      4. The evolution loop reads score + trace for fitness/feedback

    This replaces ALMA's Basic_Recorder. Key difference: ALMA records
    multi-step trajectories (steps: list of action-observation pairs).
    We record single-pass protocol execution traces.
    """
    # --- Populated at load time ---
    task_id: str                                    # Unique identifier
    context: str                                    # The novel context (system message)
    query: str                                      # The question/task (user message)
    messages_raw: list[dict] = field(default_factory=list)  # Original messages (preserves multi-turn)
    rubrics: list[str] = field(default_factory=list)        # Evaluation criteria
    metadata: dict[str, Any] = field(default_factory=dict)  # Category, sub_category, etc.

    # --- Populated at inference time ---
    model_output: Optional[str] = None              # Protocol's final answer
    reasoning_trace: list[str] = field(default_factory=list)  # Step-by-step trace
    tokens_used: int = 0                            # Total tokens consumed
    verification_passed: Optional[bool] = None      # Protocol's self-check result

    # --- Populated at evaluation time ---
    score: Optional[float] = None                   # 0 or 1 (binary)
    eval_detail: dict[str, Any] = field(default_factory=dict)  # Grading rationale, etc.
    failure_mode: Optional[str] = None              # F1/F2/F3/F4 (if score == 0)


# ---------------------------------------------------------------
# BaseBenchmark: the pluggable benchmark interface
# (analogous to ALMA's Basic_Env, adapted for QA-style tasks)
# ---------------------------------------------------------------
class BaseBenchmark(ABC):
    """
    Abstract interface for any context-learning benchmark.

    ALMA's Basic_Env provides: set_task_env, run_step, cal_reward, get_prompt.
    Those are designed for multi-step interactive environments.

    For QA-style context learning, we need a simpler interface:
      - load_tasks()  : ingest raw data → list[TaskRecord]
      - evaluate()    : judge a single answer → score
      - get_metrics() : aggregate scores → summary dict

    To add a new benchmark:
      1. Subclass BaseBenchmark
      2. Implement load_tasks() to parse your data format
      3. Implement evaluate() with your scoring logic
      4. Register in BENCHMARK_REGISTRY
    """

    name: str = "base"

    @abstractmethod
    def load_tasks(self, data_path: str, split: str = "all") -> list[TaskRecord]:
        """
        Load raw data and convert to TaskRecord instances.

        Args:
            data_path: Path to the data file (JSONL, JSON, etc.)
            split: "train", "val", "test", or "all"

        Returns:
            List of TaskRecord with context, query, rubrics populated.
        """
        pass

    @abstractmethod
    def evaluate(self, record: TaskRecord, judge_client=None) -> TaskRecord:
        """
        Evaluate a single completed task.

        Args:
            record: TaskRecord with model_output populated.
            judge_client: Optional LLM client for LLM-as-judge evaluation.

        Returns:
            Same TaskRecord with score, eval_detail populated.
        """
        pass

    def get_metrics(self, records: list[TaskRecord]) -> dict[str, float]:
        """
        Compute aggregate metrics from a list of evaluated records.
        Default: overall accuracy + per-category accuracy.
        Subclasses can override for benchmark-specific metrics.
        """
        scored = [r for r in records if r.score is not None]
        if not scored:
            return {"accuracy": 0.0}

        overall = sum(r.score for r in scored) / len(scored)
        metrics = {"accuracy": overall, "total": len(scored)}

        # Per-category breakdown
        by_cat = {}
        for r in scored:
            cat = r.metadata.get("context_category", "unknown")
            by_cat.setdefault(cat, []).append(r.score)
        for cat, scores in by_cat.items():
            metrics[f"accuracy/{cat}"] = sum(scores) / len(scores)

        return metrics


# ---------------------------------------------------------------
# Benchmark Registry (inspired by ALMA's ENVS dict)
# ---------------------------------------------------------------
BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {}

def register_benchmark(name: str):
    """Decorator to register a benchmark implementation."""
    def decorator(cls):
        BENCHMARK_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator

def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """Instantiate a registered benchmark by name."""
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARK_REGISTRY[name](**kwargs)
```

### CL-bench Implementation (the first concrete benchmark)

```python
"""
benchmarks/cl_bench.py — CL-bench implementation of BaseBenchmark.

This is the primary benchmark for the Evo-Protocol paper.
Future benchmarks (ARC, GPQA, custom) follow the same pattern.
"""

import json
from benchmarks.base import BaseBenchmark, TaskRecord, register_benchmark


@register_benchmark("cl-bench")
class CLBenchmark(BaseBenchmark):
    """
    CL-bench: 1,899 context-learning tasks with rubric-based binary scoring.

    Data format: JSONL with {messages, rubrics, metadata}.
    Evaluation: LLM-as-judge, binary (1 iff ALL rubrics satisfied).
    """

    # Category → failure mode mapping
    CATEGORY_TO_MODE = {
        "Domain Knowledge Reasoning":       "F1",
        "Rule System Application":          "F2",
        "Procedural Task Execution":        "F3",
        "Empirical Discovery & Simulation": "F4",
    }

    def load_tasks(self, data_path: str, split: str = "all") -> list[TaskRecord]:
        records = []
        with open(data_path, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                raw = json.loads(line)
                messages = raw.get("messages", [])

                # Extract context and query from messages
                context = ""
                query = ""
                for msg in messages:
                    if msg["role"] == "system":
                        context = msg["content"]
                    elif msg["role"] == "user":
                        query = msg["content"]  # Last user message = the task

                meta = raw.get("metadata", {})
                cat = meta.get("context_category", "")
                record = TaskRecord(
                    task_id=meta.get("task_id", str(idx)),
                    context=context,
                    query=query,
                    messages_raw=messages,
                    rubrics=raw.get("rubrics", []),
                    metadata={
                        **meta,
                        "gravity_type": self.CATEGORY_TO_MODE.get(cat, "F1"),
                        "idx": idx,
                    },
                )
                records.append(record)

        if split != "all":
            records = self._apply_split(records, split)
        return records

    def evaluate(self, record: TaskRecord, judge_client=None) -> TaskRecord:
        """
        Reuse CL-bench's rubric-based LLM-as-judge evaluation.
        Delegates to the same grading logic as eval.py.
        """
        from eval import build_rubrics_text, call_judge_api

        if not record.model_output or not record.model_output.strip():
            record.score = 0
            record.eval_detail = {"reason": "Empty output"}
            return record

        rubrics_text = build_rubrics_text(record.rubrics)
        result_text = call_judge_api(
            judge_client, "gpt-5.1", rubrics_text, record.model_output,
        )

        if result_text:
            try:
                result_json = json.loads(result_text)
                record.score = result_json.get("Overall Score", 0)
                record.eval_detail = result_json
            except (json.JSONDecodeError, ValueError):
                record.score = 0
                record.eval_detail = {"reason": "Judge parse error",
                                      "raw": result_text[:500]}
        else:
            record.score = 0
            record.eval_detail = {"reason": "Judge API failed"}

        return record

    def _apply_split(self, records, split, seed=42):
        """Stratified split by context_category."""
        from sklearn.model_selection import train_test_split
        cats = [r.metadata.get("context_category", "") for r in records]
        train, temp, _, temp_cats = train_test_split(
            records, cats, test_size=0.30, stratify=cats, random_state=seed,
        )
        temp_cats2 = [r.metadata.get("context_category", "") for r in temp]
        val, test, _, _ = train_test_split(
            temp, temp_cats2, test_size=0.50, stratify=temp_cats2, random_state=seed,
        )
        return {"train": train, "val": val, "test": test}[split]
```

### How to add a future benchmark (e.g., ARC, GPQA)

```python
# benchmarks/arc_bench.py — Example: adding ARC benchmark

@register_benchmark("arc")
class ARCBenchmark(BaseBenchmark):
    """ARC (Abstraction and Reasoning Corpus) benchmark."""

    def load_tasks(self, data_path, split="all"):
        # Parse ARC JSON format → list[TaskRecord]
        # context = input-output examples; query = test input
        ...

    def evaluate(self, record, judge_client=None):
        # ARC uses exact-match evaluation, no LLM judge needed
        record.score = 1 if record.model_output == record.metadata["expected"] else 0
        return record
```

The evolution loop, Meta-Architect, and protocol archive work identically regardless of which benchmark is registered. Only `load_tasks()` and `evaluate()` differ.

### BaseProtocol Class Definition

```python
"""
base_protocol.py — The abstract base class that all evolved protocols must inherit.
Every protocol defines three core interfaces: perception, cognition, verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ProtocolResult:
    """Structured output from a protocol execution."""
    answer: str
    confidence: float = 0.0
    reasoning_trace: list[str] = field(default_factory=list)
    verification_passed: bool = False
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProtocol(ABC):
    """
    Abstract base class for all Evo-Protocols.

    A protocol is an executable reasoning strategy that orchestrates
    one or more LLM calls with deterministic logic (loops, filters,
    validators) to solve context-learning tasks.

    Three core interfaces:
      - perception(context)   : information filter — what to read
      - cognition(query, info): reasoning engine  — how to think
      - verification(answer, context): judicial gate — can we say this

    The Meta-Architect evolves subclasses of this base class.
    """

    def __init__(self, llm_client, model_name: str = "gpt-4o"):
        """
        Args:
            llm_client: An OpenAI-compatible client for making LLM calls.
            model_name: The base model to use for LLM calls within the protocol.
        """
        self.llm = llm_client
        self.model = model_name

    # ------------------------------------------------------------------
    # Core Interface 1: PERCEPTION — Information Filter
    # ------------------------------------------------------------------
    @abstractmethod
    def perception(self, context: str) -> dict[str, Any]:
        """
        Filter and structure raw context into actionable information.

        This method determines WHAT the model reads. It can:
        - Chunk long contexts and index them
        - Extract rules, constraints, or key facts
        - Build structured representations (dicts, tables)
        - Tag counter-factual or surprising information

        Args:
            context: The raw context string (may be very long).

        Returns:
            A structured dict of perceived information.
            Example: {"rules": [...], "facts": [...], "chunks": [...]}
        """
        pass

    # ------------------------------------------------------------------
    # Core Interface 2: COGNITION — Reasoning Engine
    # ------------------------------------------------------------------
    @abstractmethod
    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        """
        Generate an answer using the perceived information.

        This method determines HOW the model thinks. It can:
        - Implement chain-of-thought with explicit steps
        - Use iterative refinement loops
        - Call multiple LLM agents with different roles
        - Apply hypothesis-test cycles

        Args:
            query: The user's question or task.
            perceived_info: Output from self.perception().

        Returns:
            A candidate answer string.
        """
        pass

    # ------------------------------------------------------------------
    # Core Interface 3: VERIFICATION — Judicial Gate
    # ------------------------------------------------------------------
    @abstractmethod
    def verification(self, answer: str, context: str) -> bool:
        """
        Verify whether the answer is faithful to the context.

        This method determines WHETHER the answer can be emitted. It can:
        - Cross-check answer claims against context spans
        - Run a blind verifier agent that only sees rules
        - Apply deterministic checks (regex, n-gram overlap, logic)
        - Implement double-blind verification

        Args:
            answer: The candidate answer from cognition().
            context: The original raw context for ground-truth checking.

        Returns:
            True if the answer passes verification, False otherwise.
        """
        pass

    # ------------------------------------------------------------------
    # Orchestrator: run the full pipeline
    # ------------------------------------------------------------------
    def run(self, context: str, query: str, max_retries: int = 2) -> ProtocolResult:
        """
        Execute the full protocol pipeline with retry logic.

        Pipeline: perception -> cognition -> verification
        If verification fails, retry cognition with feedback.
        """
        trace = []
        total_tokens = 0

        # Step 1: Perception
        perceived = self.perception(context)
        trace.append(f"[Perception] Extracted {len(perceived)} keys")

        for attempt in range(max_retries + 1):
            # Step 2: Cognition
            answer = self.cognition(query, perceived)
            trace.append(f"[Cognition-{attempt}] Generated answer: {answer[:100]}...")

            # Step 3: Verification
            passed = self.verification(answer, context)
            trace.append(f"[Verification-{attempt}] Passed: {passed}")

            if passed:
                return ProtocolResult(
                    answer=answer,
                    confidence=1.0 - (attempt * 0.2),
                    reasoning_trace=trace,
                    verification_passed=True,
                    tokens_used=total_tokens,
                )

            # If verification failed, add feedback for next attempt
            trace.append(f"[Retry-{attempt}] Verification failed, retrying...")

        # All retries exhausted — return best-effort answer
        return ProtocolResult(
            answer=answer,
            confidence=0.1,
            reasoning_trace=trace,
            verification_passed=False,
            tokens_used=total_tokens,
        )

    # ------------------------------------------------------------------
    # Utility: Make an LLM call (used by subclasses)
    # ------------------------------------------------------------------
    def _call_llm(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Make a single LLM call through the sandboxed client."""
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
```

### Design Note: Relationship to ALMA (Xiong et al., 2025)

We borrow several architectural patterns from ALMA while redirecting them to a fundamentally different problem:

| What we borrow from ALMA | How we adapt it |
|--------------------------|-----------------|
| `Basic_Env` / `Basic_Recorder` abstraction | → `BaseBenchmark` / `TaskRecord` (QA-style, not interactive) |
| `ENVS` registry dict | → `BENCHMARK_REGISTRY` with `@register_benchmark` decorator |
| `find_subclass_in_file()` dynamic code loading | → Same pattern for loading evolved protocol `.py` files |
| `Sub_memo_layer` composable sub-modules | → Not adopted: our 3-method interface is flat by design (simpler search space for evolution) |
| Memo archive with SHA indexing + parent tracking | → Protocol archive with SHA, parent lineage, generation metadata |
| Softmax selection with visit penalty | → Same selection mechanism in evolution loop |
| Token tracking singleton | → `TokenTracker` for cost monitoring across evolution |
| Two-phase evaluation (collection + deployment) | → Not adopted: our tasks are stateless, no persistent memory |
| `async` execution throughout | → Not adopted: synchronous API calls suffice for our use case |

**The key difference remains**: ALMA evolves **memory** (how to store/retrieve across tasks). We evolve **reasoning protocols** (how to process context within a task, with executable verification gates).

### Example: A Naive Protocol (Generation 0)

```python
class NaiveProtocol(BaseProtocol):
    """
    Generation 0: A simple baseline protocol with no special handling.
    This is the starting point before evolution.
    """

    def perception(self, context: str) -> dict[str, Any]:
        # No filtering — pass everything through
        return {"full_context": context}

    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "Answer based on the given context."},
            {"role": "user", "content": (
                f"Context:\n{perceived_info['full_context']}\n\n"
                f"Question: {query}"
            )}
        ]
        return self._call_llm(messages)

    def verification(self, answer: str, context: str) -> bool:
        # No verification — always pass
        return True
```

### Example: An Evolved Protocol (Generation ~15)

```python
class DoubleBlindVerifiedProtocol(BaseProtocol):
    """
    An example of what evolution might produce after ~15 generations.
    Demonstrates: chunked perception, blind rule extraction, hard verification.
    """

    def perception(self, context: str) -> dict[str, Any]:
        # Chunk context into manageable pieces
        chunk_size = 1500
        chunks = [context[i:i+chunk_size]
                  for i in range(0, len(context), chunk_size)]

        # Extract rules from each chunk independently
        all_rules = []
        for i, chunk in enumerate(chunks):
            rules = self._call_llm([
                {"role": "system", "content": (
                    "Extract ALL explicit rules, constraints, and novel facts "
                    "from this text. Output as a numbered list. "
                    "Pay special attention to anything that contradicts "
                    "common knowledge."
                )},
                {"role": "user", "content": chunk}
            ])
            all_rules.append({"chunk_id": i, "rules": rules, "raw": chunk})

        return {"chunks": all_rules, "num_chunks": len(chunks)}

    def cognition(self, query: str, perceived_info: dict[str, Any]) -> str:
        # Consolidate all extracted rules
        rule_summary = "\n".join(
            f"[Chunk {c['chunk_id']}] {c['rules']}"
            for c in perceived_info["chunks"]
        )

        # Step 1: Generate answer using ONLY extracted rules (not raw context)
        answer = self._call_llm([
            {"role": "system", "content": (
                "You are a strict rule-follower. Answer ONLY based on "
                "the extracted rules below. If a rule contradicts your "
                "prior knowledge, the rule ALWAYS wins. "
                "Never use information not present in the rules."
            )},
            {"role": "user", "content": (
                f"Extracted Rules:\n{rule_summary}\n\n"
                f"Question: {query}\n\n"
                "Think step by step, citing which rule supports each claim."
            )}
        ])
        return answer

    def verification(self, answer: str, context: str) -> bool:
        # Blind verifier: only sees context + answer, not the question
        verdict = self._call_llm([
            {"role": "system", "content": (
                "You are a verification agent. You will receive a CONTEXT "
                "and an ANSWER. Your job is to check whether every claim "
                "in the ANSWER is supported by the CONTEXT. "
                "You do NOT know what question was asked. "
                "Output ONLY 'PASS' or 'FAIL: <reason>'."
            )},
            {"role": "user", "content": (
                f"CONTEXT:\n{context}\n\n"
                f"ANSWER:\n{answer}"
            )}
        ])
        return verdict.strip().startswith("PASS")
```

### Evolution Loop Algorithm

```
Algorithm: Evo-Protocol Evolution Loop
============================================================

Input:
  - D_train: training tasks from GravityBench
  - D_val: held-out validation tasks
  - G: number of generations
  - N: population size per generation
  - K: number of elite protocols to keep (selection)
  - Meta-Architect: LLM that generates Python code

Output:
  - pi*: best-performing protocol

Initialize:
  population = [NaiveProtocol] * N   // start with naive baselines
  archive = []                        // log of all protocols + scores

For generation g = 1 to G:

  // --- EVALUATE ---
  For each protocol pi_i in population:
    scores_i = []
    traces_i = []
    For each task (c, x, y*) in sample(D_train, size=50):
      result = pi_i.run(c, x)
      score = evaluate(result.answer, y*, rubrics)   // reuse CL-bench eval
      failure_mode = classify_failure(result, c, x)  // F1/F2/F3/F4
      scores_i.append(score)
      traces_i.append({
        "task": (c[:200], x),
        "answer": result.answer,
        "score": score,
        "failure_mode": failure_mode,
        "trace": result.reasoning_trace
      })
    fitness_i = mean(scores_i)
    archive.append((pi_i.source_code, fitness_i, traces_i))

  // --- SELECT ---
  ranked = sort(population, key=fitness, descending=True)
  elites = ranked[:K]

  // --- EVOLVE (Meta-Architect generates new protocols) ---
  new_population = list(elites)  // keep elites unchanged

  While len(new_population) < N:
    parent = sample(elites)
    // Find worst-performing traces for this parent
    failures = [t for t in parent.traces if t["score"] == 0]
    sampled_failures = sample(failures, min(5, len(failures)))

    // Ask Meta-Architect to mutate
    prompt = build_architect_prompt(
      parent_code=parent.source_code,
      failure_examples=sampled_failures,
      generation=g
    )
    new_code = Meta_Architect.generate(prompt)

    // Validate + auto-repair (ALMA Debugger pattern)
    protocol, new_code = generate_with_repair(
      Meta_Architect, architect_prompt, loader, max_repair=2
    )
    if protocol is not None:
      child = instantiate(new_code)
      new_population.append(child)

  population = new_population

  // --- LOG ---
  print(f"Gen {g}: best={max(fitness):.3f}, "
        f"mean={mean(fitness):.3f}, "
        f"diversity={code_diversity(population)}")

// --- FINAL EVALUATION on D_val ---
pi* = argmax(population, key=fitness_on(D_val))
Return pi*
```

### Key Difference Table: Evo-Protocol vs. GEPA vs. ALMA

| Dimension | GEPA | ALMA | **Evo-Protocol** |
|-----------|------|------|-------------------|
| What evolves | Prompt text | Memory `update()`/`retrieve()` | **Full protocol: perception + cognition + verification** |
| Constraint strength | Soft (NL instruction) | Soft (retrieval quality) | **Hard (Python `if`/`assert`/`return`)** |
| Can block bad outputs | No (model may ignore prompt) | No | **Yes (verification gate returns FAIL)** |
| Handles multi-step reasoning | Indirectly via prompt | Via memory replay | **Directly via code loops and state machines** |
| Architecture search | No (fixed pipeline) | No (fixed interface) | **Yes (code can define any topology)** |
| Cross-domain transfer | Limited (prompt is domain-specific) | Moderate (memory is generic) | **Strong (code logic generalizes)** |

---

## 4. GravityBench: CL-bench as the Evaluation Testbed

### Why CL-bench IS GravityBench

Rather than constructing a new synthetic benchmark, we use **CL-bench directly** as our evaluation testbed (renamed GravityBench in our framing). The rationale:

1. **CL-bench already tests parametric gravity.** Every task requires learning novel knowledge *from context* that is absent from pre-training. This is exactly the setting where parametric gravity manifests. GPT-5.1 at 23.7% confirms that current models fail systematically.

2. **Expert-annotated quality.** Each context requires ~20 hours of expert annotation. Self-constructed synthetic benchmarks cannot match this quality and would be a weakness for paper reviewers.

3. **Contamination-free by design.** Contexts are fictional, modified, or niche domain knowledge — models cannot rely on pre-trained knowledge. This is the exact property needed to measure parametric gravity resistance.

4. **Established baselines.** CL-bench's leaderboard provides direct comparison points for all major models, giving our results immediate context.

5. **Pipeline compatibility.** We directly reuse `eval.py` (LLM-as-judge, binary scoring) and `infer.py` (concurrent inference) with zero modification.

### Category-to-Failure-Mode Mapping

CL-bench's four categories correspond naturally to our four parametric gravity failure modes:

| CL-bench Category | Tasks | % | Primary Failure Mode | Mechanism |
|--------------------|-------|------|----------------------|-----------|
| **Domain Knowledge Reasoning** | 663 | 34.9% | **F1: Parametric Override** | Context defines novel domain facts (Healthcare, Finance, Science, Humanities, Lifestyle). Model must suppress real-world knowledge and adopt the context's definitions. |
| **Rule System Application** | 566 | 29.8% | **F2: Context Navigation Failure** | Context contains complex rule systems (Game Mechanics, Legal & Regulatory, Technical Standards, Programming Syntax, Mathematical Formalism). Rules are scattered across long contexts; model must locate and apply them precisely. |
| **Procedural Task Execution** | 471 | 24.8% | **F3: Reasoning Breakdown** | Context defines multi-step workflows (Workflow Orchestration, Operational Procedures, Instructional Procedures). Model must maintain a chain of reasoning without dropping steps or short-circuiting. |
| **Empirical Discovery & Simulation** | 199 | 10.5% | **F4: Induction Failure** | Context provides raw data (Observational Data, Experimental Data, Simulation Environment). Model must induce patterns from evidence rather than fabricating plausible rules. |

### Sub-category Distribution (18 sub-categories)

```
Domain Knowledge Reasoning (663):
  ├── Technical Standards:    201 (10.6%)
  ├── Humanities:             124 ( 6.5%)
  ├── Healthcare:             105 ( 5.5%)
  ├── Finance:                101 ( 5.3%)
  ├── Science:                 88 ( 4.6%)
  └── Lifestyle:               57 ( 3.0%)

Rule System Application (566):
  ├── Game Mechanics:          137 ( 7.2%)
  ├── Legal & Regulatory:       92 ( 4.8%)
  ├── Legal Advisory:           76 ( 4.0%)
  ├── Mathematical Formalism:   69 ( 3.6%)
  └── Programming Syntax:       67 ( 3.5%)

Procedural Task Execution (471):
  ├── Workflow Orchestration:  229 (12.1%)
  ├── Management:              112 ( 5.9%)
  ├── Operational Procedures:  185 ( 9.7%) *
  └── Instructional Procedures: 57 ( 3.0%) *

Empirical Discovery & Simulation (199):
  ├── Observational Data:       95 ( 5.0%)
  ├── Experimental Data:        45 ( 2.4%)
  └── Simulation Environment:   59 ( 3.1%)
```
*Note: Management (112) is counted under Procedural; Operational Procedures (185) contributes to this category's total.

### Data Format

Each task in CL-bench follows this JSONL format, directly compatible with our pipeline:

```json
{
  "messages": [
    {"role": "system", "content": "<novel context with knowledge absent from pre-training>"},
    {"role": "user", "content": "<task/question requiring context learning>"},
    // Optional: multi-turn interactions (assistant + user pairs)
  ],
  "rubrics": [
    "Rubric 1: description of requirement",
    "Rubric 2: description of requirement"
  ],
  "metadata": {
    "task_id": "uuid",
    "context_category": "Domain Knowledge Reasoning | Rule System Application | ...",
    "sub_category": "Healthcare | Game Mechanics | ..."
  }
}
```

**Key statistics:**
- **1,899 tasks** total
- **Single-turn tasks**: 1,278 (67.3%) — 2 messages (system + user)
- **Multi-turn tasks**: 621 (32.7%) — up to 12 messages, with task dependencies
- **Binary scoring**: Score 1 iff ALL rubrics are satisfied; 0 otherwise
- **Average 63.2 rubrics per context** (rigorous multi-dimensional verification)

### Multi-turn Task Handling

621 tasks (32.7%) involve multi-turn conversations where later questions depend on earlier answers. For protocol execution, we handle this by:

```python
def run_multi_turn(protocol, messages: list[dict]) -> str:
    """
    Execute protocol on multi-turn CL-bench tasks.

    For multi-turn tasks, messages alternate: system, user, assistant, user, ...
    The system message contains the context. Each user message is a task.
    Assistant messages represent expected prior answers that set up later tasks.

    Strategy: Feed all prior turns as part of the context, then apply the
    protocol to the FINAL user message (the actual task to solve).
    """
    # Separate context (system) from conversation history
    context = ""
    history = []
    final_query = ""

    for msg in messages:
        if msg["role"] == "system":
            context = msg["content"]
        elif msg["role"] == "user":
            final_query = msg["content"]
            history.append(msg)
        elif msg["role"] == "assistant":
            history.append(msg)

    # Build augmented context: original context + conversation history
    augmented_context = context
    if len(history) > 1:  # Multi-turn: prepend prior turns
        prior_turns = "\n\n---\nPrior conversation:\n"
        for msg in history[:-1]:  # All except final user query
            prior_turns += f"[{msg['role'].upper()}]: {msg['content']}\n"
        augmented_context = context + prior_turns

    # Run protocol on augmented context + final query
    return protocol.run(augmented_context, final_query)
```

### Train/Validation/Test Split

For evolution, we split CL-bench into three subsets:

| Split | Purpose | Size | Sampling |
|-------|---------|------|----------|
| **Train** | Evolution fitness evaluation (sampled each generation) | 70% (~1,329 tasks) | Stratified by category |
| **Validation** | Monitor overfitting during evolution | 15% (~285 tasks) | Stratified by category |
| **Test** | Final evaluation only (never seen during evolution) | 15% (~285 tasks) | Stratified by category |

```python
from sklearn.model_selection import train_test_split

def split_gravitybench(data: list[dict], seed: int = 42):
    """Split CL-bench into train/val/test stratified by category."""
    categories = [d["metadata"]["context_category"] for d in data]

    # First split: 70% train, 30% temp
    train, temp, _, temp_cats = train_test_split(
        data, categories, test_size=0.30,
        stratify=categories, random_state=seed,
    )
    temp_categories = [d["metadata"]["context_category"] for d in temp]

    # Second split: 50/50 of temp -> 15% val, 15% test
    val, test, _, _ = train_test_split(
        temp, temp_categories, test_size=0.50,
        stratify=temp_categories, random_state=seed,
    )

    return train, val, test
```

### Failure Mode Annotation

For detailed analysis, we annotate each task with its **primary failure mode** based on category mapping, and use the failure classifier (Appendix B) to determine the **actual failure mode** when a baseline model fails:

```python
# Category -> primary failure mode mapping
CATEGORY_TO_MODE = {
    "Domain Knowledge Reasoning":          "F1",  # Parametric Override
    "Rule System Application":             "F2",  # Context Navigation Failure
    "Procedural Task Execution":           "F3",  # Reasoning Breakdown
    "Empirical Discovery & Simulation":    "F4",  # Induction Failure
}

def annotate_failure_modes(data: list[dict]) -> list[dict]:
    """Add primary failure mode to each task based on category."""
    for task in data:
        cat = task["metadata"]["context_category"]
        task["metadata"]["gravity_type"] = CATEGORY_TO_MODE.get(cat, "F1")
    return data
```

Note: A single task may exhibit multiple failure modes (e.g., a Rule System task may require both navigation F2 and reasoning F3). The category mapping provides the **primary** mode; the failure classifier (Appendix B) provides **per-instance** diagnosis when models fail.

---

## 5. Experimental Design

### Baselines

| Baseline | Description | Category |
|----------|-------------|----------|
| **Vanilla** | Direct LLM call with context + query | Single-agent |
| **CoT** | Chain-of-Thought prompting ("think step by step") | Single-agent |
| **ReAct** | Reasoning + Acting interleaved | Single-agent |
| **RAG** | Retrieve from chunked context, then generate | Retrieval-augmented |
| **MetaGPT / ChatDev** | Fixed multi-agent roles (SOP-based) | Multi-agent (static) |
| **GEPA** | Prompt evolution via reflection (Pareto-front merging) | Prompt evolution |
| **Evo-Protocol (ours)** | Full protocol evolution in code space | Code evolution |
| **Ablation: Prompt-only** | Evo-Protocol but `verification` must be NL-only (no code logic) | Ablation |
| **Ablation: Code-only** | Evo-Protocol but no NL prompts (only code logic, no LLM calls) | Ablation |
| **Ablation: No-verification** | Evo-Protocol with `verification()` always returning True | Ablation |
| **Ablation: No-perception** | Evo-Protocol with `perception()` passing raw context through | Ablation |

### Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Accuracy (overall)** | % tasks where score = 1 (binary, all rubrics met) | Primary metric |
| **Accuracy (per-mode)** | Accuracy broken down by F1/F2/F3/F4 failure mode | Diagnose strengths |
| **Token Efficiency** | Accuracy / total tokens consumed per task | Cost-performance tradeoff |
| **Attention Drift** | % of tasks where answer contradicts context despite seeing it (measured by verifier) | Quantify parametric gravity |
| **Convergence Speed** | Number of generations to reach 80% of final accuracy | Evolution efficiency |
| **Cross-domain Transfer** | Accuracy on domain B using protocol evolved on domain A | Generalization |

#### Attention Drift Measurement

```python
def measure_attention_drift(answer: str, context: str, query: str) -> float:
    """
    Measure how much the answer drifts from context toward parametric priors.

    Method: Use an independent judge LLM to classify whether the answer:
      0.0 = fully faithful to context
      0.5 = partially faithful, partially parametric
      1.0 = fully parametric (ignores context)
    """
    judge_prompt = f"""
    Given the context and the answer below, rate how faithfully
    the answer follows the context on a scale of 0 to 1.

    0 = Answer is entirely based on the provided context.
    1 = Answer ignores the context and relies on general knowledge.

    Context (excerpt): {context[:2000]}
    Answer: {answer}

    Output ONLY a number between 0 and 1.
    """
    # Call judge model and parse float
    ...
```

### Hypothesis Testing

#### H1: Code > Prompt (Hard Constraints Beat Soft Constraints)

**Design**: Compare Evo-Protocol vs. GEPA on the same GravityBench tasks.

| Comparison | Independent Variable | Dependent Variable | Expected Result |
|------------|---------------------|-------------------|-----------------|
| Evo-Protocol vs. GEPA | Evolution space (code vs. prompt) | Accuracy on F1 (Parametric Override) | Evo-Protocol > GEPA by 15%+ |
| Evo-Protocol vs. GEPA | Evolution space | Accuracy on F2-F4 | Evo-Protocol > GEPA by 8-12% |
| Statistical test | — | — | Paired t-test, p < 0.01 |

**Rationale**: GEPA can produce excellent prompts like "ignore prior knowledge," but the model can still fail to comply. Evo-Protocol produces `if not verify(answer, context): reject`, which physically blocks bad outputs.

#### H2: Generalization (Protocols Transfer Across Domains)

**Design**: Evolve protocol on sub-task A (Counter-Factual QA), test zero-shot on sub-tasks B, C, D.

| Training Domain | Test Domain | Expected Accuracy | Rationale |
|-----------------|-------------|-------------------|-----------|
| A (Counter-Factual) | B (Needle) | 60%+ | Perception logic transfers |
| A (Counter-Factual) | C (Logic) | 50%+ | Verification logic transfers |
| C (Logic) | A (Counter-Factual) | 55%+ | Step-by-step reasoning transfers |
| Mixed (A+B+C+D) | Held-out from each | 75%+ | Full protocol is domain-general |

### Ablation Studies

| Ablation | What is removed/changed | Expected impact |
|----------|------------------------|-----------------|
| **Prompt-only** | `verification()` uses only NL prompts, no code logic | -15% on F1 tasks (soft constraint fails) |
| **Code-only** | No LLM calls inside protocol; pure Python logic | -25% overall (loses NL understanding) |
| **Joint (full)** | Both NL prompts and code logic (full Evo-Protocol) | Best performance |
| **No verification** | `verification()` always returns True | -20% on F1, -10% on F3 |
| **No perception** | `perception()` is identity function | -15% on F2 (no chunking/indexing) |
| **Generation count** | Compare performance at gen 5, 10, 20, 50 | Monotonic improvement, plateau ~30 |
| **Meta-Architect model** | Use GPT-4o-mini vs. GPT-4o vs. Claude Opus as architect | Stronger architect → better protocols |
| **Population size** | N=3 vs. N=5 vs. N=10 | Larger population → more diversity, better results |

---

## 6. Implementation Details

### Meta-Architect System Prompt

```
SYSTEM PROMPT FOR META-ARCHITECT
=================================

You are an expert AI systems architect. Your task is to design Python classes
that inherit from BaseProtocol to solve context-learning tasks.

## YOUR GOAL

Create a protocol that helps an LLM correctly answer questions based on
provided context, even when the context contradicts the model's pre-trained
knowledge. The protocol must overcome "Parametric Gravity" — the tendency
of LLMs to ignore context and rely on memorized knowledge.

## CONSTRAINTS

1. Your class MUST inherit from `BaseProtocol`.
2. You MUST implement all three abstract methods:
   - `perception(self, context) -> dict`
   - `cognition(self, query, perceived_info) -> str`
   - `verification(self, answer, context) -> bool`
3. You may call `self._call_llm(messages, temperature)` for LLM calls.
4. You may use standard Python libraries (re, json, collections, etc.).
5. You MUST NOT use external APIs, file I/O, or network calls.
6. Total LLM calls per task should not exceed 10.
7. Your code must be syntactically valid and executable.

## CURRENT GENERATION: {generation}

## PARENT PROTOCOL (Code):
```python
{parent_code}
```

## PARENT PERFORMANCE:
- Overall accuracy: {parent_accuracy:.1%}
- F1 (Parametric Override): {f1_accuracy:.1%}
- F2 (Context Navigation): {f2_accuracy:.1%}
- F3 (Reasoning Breakdown): {f3_accuracy:.1%}
- F4 (Induction Failure): {f4_accuracy:.1%}

## FAILURE ANALYSIS (sampled failed tasks):

{failure_examples}

## YOUR TASK

Analyze the failures above. Identify the ROOT CAUSE of each failure.
Then rewrite the protocol class to address these failures.

Key strategies to consider:
- For F1 failures: Add verification logic that cross-checks answers against
  context. Consider double-blind verification where the verifier does not
  see the original question.
- For F2 failures: Improve perception with chunking, indexing, or iterative
  scanning. Do not rely on the LLM reading the full context at once.
- For F3 failures: Break reasoning into explicit steps with intermediate
  verification. Use state-machine patterns with rollback on error.
- For F4 failures: Implement hypothesis-test cycles. Generate a candidate
  rule, test it against examples, refine if it fails.

Output ONLY the complete Python class definition. No explanation needed.
```

### Code Validation & Loading Pipeline

We rename `ProtocolSandbox` → **`ProtocolLoader`** to reflect what it actually does: a multi-stage validation pipeline + dynamic code loading + timeout enforcement. True OS-level sandboxing (Docker, WebAssembly) is unnecessary because our protocols only interact with the outside world through `self._call_llm()` API calls.

#### Tool Selection Rationale

| Tool | Verdict | Reasoning |
|------|---------|-----------|
| `ast.parse` (stdlib) | **YES** | Zero-cost syntax gate. Instant, no install needed. |
| `ruff` (Astral, Rust) | **YES** | Replaces pyflakes + flake8 in one tool. ~10ms per file. `--select F,E,B` catches undefined names, errors, and common bugs without style noise. `--fix` can auto-repair minor issues. |
| `pyflakes` | NO | Redundant: ruff's `F` rules are pyflakes. |
| `pylint` / `flake8` | NO | Slow and/or style-focused. Generated code is never read by humans. |
| `ty` / `mypy` / `pyright` | NO | Type errors produce clear runtime tracebacks that feed back to Meta-Architect. Adding type checking costs 1-3s/file and generates false positives on loosely-typed LLM code. Net negative ROI. |
| `bandit` | NO | Redundant with our AST import/builtin whitelist. |
| `langchain-sandbox` (Pyodide) | NO | **Fatal flaw**: Pyodide runs in WebAssembly — cannot call `openai` Python SDK. Our protocols must make real API calls via `self._call_llm()`. |
| `epicbox` / `piston` (Docker) | NO | ~12,000 protocol executions per experiment. Docker container startup overhead is prohibitive. |

#### Validation Pipeline (4 stages)

```
Meta-Architect generates code
         │
         ▼
  ┌──────────────┐
  │ Stage 1: AST │  ast.parse() — instant syntax check
  │   (stdlib)   │  Catches: SyntaxError, indentation
  └──────┬───────┘
         │ pass
         ▼
  ┌──────────────┐
  │ Stage 2: Ruff│  ruff check --select F,E,B
  │   (~10ms)    │  Catches: undefined names, unused imports,
  └──────┬───────┘  unreachable code, common bug patterns
         │ pass
         ▼
  ┌──────────────┐
  │ Stage 3: AST │  Custom walk: check ALLOWED_IMPORTS,
  │  Security    │  BANNED_BUILTINS, forbidden calls
  └──────┬───────┘
         │ pass
         ▼
  ┌──────────────┐
  │ Stage 4:     │  importlib dynamic load + trial run
  │  Load & Run  │  with signal.alarm timeout
  └──────┬───────┘
         │
    ┌────┴────┐
    │ FAIL?   │──yes──▶ Feed error to Meta-Architect
    └────┬────┘         for self-repair (max 2 retries,
         │ no           inspired by ALMA Debugger)
         ▼
    Return result
```

#### Full Implementation

```python
"""
core/protocol_loader.py — Multi-stage code validation + dynamic loading.

Pipeline: ast.parse → ruff lint → AST security scan → dynamic load → run with timeout.
If validation fails, returns structured error for Meta-Architect self-repair.

Dependencies: pip install ruff  (single additional dependency)
"""

import ast
import importlib.util
import inspect
import signal
import subprocess
import traceback
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

ALLOWED_IMPORTS = {
    "re", "json", "math", "collections", "itertools",
    "functools", "typing", "dataclasses", "copy", "random",
}

BANNED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "globals", "locals", "breakpoint", "exit", "quit",
}

TIMEOUT_SECONDS = 120


def compute_sha(code: str) -> str:
    """Compute a short SHA hash for protocol versioning."""
    return hashlib.sha256(code.encode()).hexdigest()[:12]


class ValidationError:
    """Structured validation error for Meta-Architect feedback."""
    def __init__(self, stage: str, message: str, fixable: bool = False):
        self.stage = stage
        self.message = message
        self.fixable = fixable

    def __str__(self):
        return f"[{self.stage}] {self.message}"

    def to_feedback(self) -> str:
        """Format as feedback string for the Meta-Architect."""
        return (
            f"Code validation failed at stage '{self.stage}'.\n"
            f"Error: {self.message}\n"
            f"{'This is likely auto-fixable.' if self.fixable else 'Please rewrite the problematic section.'}"
        )


class ProtocolLoader:
    """
    Multi-stage code validator + dynamic protocol loader.

    NOT a sandbox — our protocols make real API calls and don't need
    OS-level isolation. This class provides:
      1. Pre-execution validation (syntax, lint, security)
      2. Dynamic code loading (ALMA pattern)
      3. Timeout enforcement
      4. Structured error reporting for self-repair
    """

    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Stage 1: Syntax check (ast.parse, ~0ms)
    # ------------------------------------------------------------------
    def _check_syntax(self, code: str) -> Optional[ValidationError]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return ValidationError(
                stage="syntax",
                message=f"Line {e.lineno}: {e.msg}",
                fixable=True,  # Syntax errors are usually easy to fix
            )

    # ------------------------------------------------------------------
    # Stage 2: Ruff lint (subprocess, ~10ms)
    # Catches: undefined names (F821), unused imports (F401),
    #          unreachable code (B), common bugs
    # ------------------------------------------------------------------
    def _check_lint(self, code: str) -> Optional[ValidationError]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [
                    "ruff", "check", tmp_path,
                    "--select", "F,E,B",   # pyflakes + errors + bugbear
                    "--no-fix",            # Report only, don't auto-fix
                    "--output-format", "text",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 and result.stdout.strip():
                # Filter out style-only issues, keep real errors
                errors = [
                    line for line in result.stdout.strip().split("\n")
                    if any(code in line for code in ["F821", "F811", "E999", "B"])
                ]
                if errors:
                    return ValidationError(
                        stage="lint",
                        message="\n".join(errors[:5]),  # Top 5 issues
                        fixable=True,
                    )
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ruff not installed or timed out — skip this stage
            return None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Stage 3: AST security scan (custom, ~0ms)
    # ------------------------------------------------------------------
    def _check_security(self, code: str) -> Optional[ValidationError]:
        tree = ast.parse(code)  # Already validated in Stage 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden import: {alias.name}. "
                                    f"Allowed: {', '.join(sorted(ALLOWED_IMPORTS))}",
                            fixable=True,
                        )
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden import: {node.module}",
                        fixable=True,
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_BUILTINS:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden builtin call: {node.func.id}()",
                            fixable=True,
                        )
        return None

    # ------------------------------------------------------------------
    # Full validation pipeline (Stage 1 → 2 → 3)
    # ------------------------------------------------------------------
    def validate(self, code: str) -> Optional[ValidationError]:
        """Run all validation stages. Returns None if code is clean."""
        for check in [self._check_syntax, self._check_lint, self._check_security]:
            error = check(code)
            if error:
                return error
        return None

    # ------------------------------------------------------------------
    # Stage 4: Dynamic load (ALMA pattern)
    # ------------------------------------------------------------------
    def load_from_file(self, file_path: str) -> Optional[object]:
        """
        Dynamically load a BaseProtocol subclass from a .py file.
        Adapted from ALMA's find_subclass_in_file() in evals/launch.py.
        """
        try:
            spec = importlib.util.spec_from_file_location(
                "evolved_protocol", file_path
            )
            module = importlib.util.module_from_spec(spec)
            module.BaseProtocol = BaseProtocol
            module.ProtocolResult = ProtocolResult
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseProtocol) and obj is not BaseProtocol:
                    return obj(self.llm_client, self.model_name)

        except Exception as e:
            print(f"[Loader] Import error: {traceback.format_exc()}")
        return None

    def load_from_code(self, code: str) -> tuple[Optional[object], Optional[ValidationError]]:
        """
        Full pipeline: validate → save → load.
        Returns (protocol_instance, None) on success,
        or (None, ValidationError) on failure.
        """
        error = self.validate(code)
        if error:
            return None, error

        sha = compute_sha(code)
        tmp_path = Path(f"/tmp/evo_protocol_{sha}.py")
        tmp_path.write_text(code)

        protocol = self.load_from_file(str(tmp_path))
        if protocol is None:
            return None, ValidationError(
                stage="load",
                message="No valid BaseProtocol subclass found in generated code.",
                fixable=True,
            )
        return protocol, None

    # ------------------------------------------------------------------
    # Run with timeout
    # ------------------------------------------------------------------
    def run_with_timeout(
        self, protocol, context: str, query: str
    ) -> Optional[ProtocolResult]:
        """Execute protocol with timeout enforcement."""
        def handler(signum, frame):
            raise TimeoutError("Protocol execution timed out")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            result = protocol.run(context, query)
            signal.alarm(0)
            return result
        except TimeoutError:
            return None
        except Exception:
            signal.alarm(0)
            print(f"[Loader] Runtime error: {traceback.format_exc()}")
            return None
```

#### Self-Repair Loop (Inspired by ALMA's Debugger Agent)

ALMA's evolution pipeline has a Debugger agent: when generated code fails trial execution, the Debugger reads the error trace and attempts to fix the code (up to N retries). We adopt this pattern, integrated with our validation pipeline:

```python
"""
core/self_repair.py — Auto-fix loop for failed protocol validation.

When the Meta-Architect generates code that fails validation,
we feed the structured error back for self-repair before rejecting.

This is ALMA's Debugger pattern: examine_new_code → trial_run →
if error: Debugger(code, error) → retry up to N times.
"""

async def generate_with_repair(
    meta_architect_client,
    architect_model: str,
    architect_prompt: str,
    loader: ProtocolLoader,
    max_repair_attempts: int = 2,
) -> tuple[Optional[object], str]:
    """
    Generate a protocol and auto-repair validation failures.

    Returns: (protocol_instance, source_code) or (None, error_message)
    """
    # Initial generation
    response = meta_architect_client.chat.completions.create(
        model=architect_model,
        messages=[{"role": "user", "content": architect_prompt}],
    )
    code = extract_python_code(response.choices[0].message.content)

    for attempt in range(max_repair_attempts + 1):
        protocol, error = loader.load_from_code(code)

        if protocol is not None:
            return protocol, code  # Success

        if attempt == max_repair_attempts:
            return None, f"Failed after {max_repair_attempts} repair attempts: {error}"

        # Self-repair: feed error back to Meta-Architect
        repair_prompt = (
            f"The code you generated failed validation:\n\n"
            f"{error.to_feedback()}\n\n"
            f"Original code:\n```python\n{code}\n```\n\n"
            f"Fix the error and output the complete corrected Python class."
        )

        response = meta_architect_client.chat.completions.create(
            model=architect_model,
            messages=[{"role": "user", "content": repair_prompt}],
        )
        code = extract_python_code(response.choices[0].message.content)

    return None, "Unreachable"


def extract_python_code(text: str) -> str:
    """Extract Python code block from LLM response."""
    if "```python" in text:
        code = text.split("```python")[1].split("```")[0]
        return code.strip()
    if "```" in text:
        code = text.split("```")[1].split("```")[0]
        return code.strip()
    return text.strip()
```

### Protocol Archive (Inspired by ALMA's memo_archive with SHA indexing)

```python
"""
core/archive.py — Protocol versioning with lineage tracking.

Inspired by ALMA's memo_archive: each memory design is saved as
a .py file indexed by SHA hash, with metadata tracking parent,
generation, and performance.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional


class ProtocolArchive:
    """
    SHA-indexed archive of all evolved protocols.

    Directory layout (mirrors ALMA's memo_archive/):
      archive/
      ├── protocol_a1b2c3d4.py          # Source code
      ├── protocol_a1b2c3d4_meta.json   # Metadata
      ├── protocol_e5f6g7h8.py
      ├── protocol_e5f6g7h8_meta.json
      └── ...
    """

    def __init__(self, archive_dir: str = "archive"):
        self.dir = Path(archive_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db: dict[str, dict] = {}  # sha → metadata
        self._load_existing()

    def _load_existing(self):
        for meta_file in self.dir.glob("*_meta.json"):
            meta = json.loads(meta_file.read_text())
            self.db[meta["sha"]] = meta

    def save(self, code: str, generation: int, parent_sha: Optional[str] = None,
             score: Optional[float] = None, extra: dict = None) -> str:
        """Save a protocol to the archive. Returns its SHA."""
        sha = hashlib.sha256(code.encode()).hexdigest()[:12]
        code_path = self.dir / f"protocol_{sha}.py"
        meta_path = self.dir / f"protocol_{sha}_meta.json"

        code_path.write_text(code)
        meta = {
            "sha": sha,
            "generation": generation,
            "parent_sha": parent_sha,
            "score": score,
            "visit_count": 0,
            **(extra or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        self.db[sha] = meta
        return sha

    def update_score(self, sha: str, score: float):
        self.db[sha]["score"] = score
        self._write_meta(sha)

    def increment_visit(self, sha: str):
        self.db[sha]["visit_count"] = self.db[sha].get("visit_count", 0) + 1
        self._write_meta(sha)

    def get_code(self, sha: str) -> str:
        return (self.dir / f"protocol_{sha}.py").read_text()

    def select(self, k: int = 5, tau: float = 0.5, alpha: float = 0.5) -> list[str]:
        """
        Softmax selection with visit penalty (ALMA's selection mechanism).

        Score_i = sigmoid(reward_i - baseline) - alpha * log(1 + visit_count_i)
        Probability_i = softmax(Score_i / tau)
        """
        import math, random
        candidates = [(sha, m) for sha, m in self.db.items()
                      if m.get("score") is not None]
        if not candidates:
            return []

        baseline = min(m["score"] for _, m in candidates)
        scores = []
        for sha, m in candidates:
            normalized = 1 / (1 + math.exp(-(m["score"] - baseline)))
            penalty = alpha * math.log(1 + m.get("visit_count", 0))
            scores.append(normalized - penalty)

        # Softmax
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / tau) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        selected = random.choices(
            [sha for sha, _ in candidates],
            weights=probs,
            k=min(k, len(candidates)),
        )
        return selected

    def _write_meta(self, sha: str):
        meta_path = self.dir / f"protocol_{sha}_meta.json"
        meta_path.write_text(json.dumps(self.db[sha], indent=2))
```

### Token Tracker (Inspired by ALMA's global token accounting)

```python
"""
core/token_tracker.py — Global token usage tracking.

ALMA uses a singleton TokenTracker to monitor API costs across
all LLM calls during evolution. We adopt the same pattern.
"""

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TokenTracker:
    """Thread-safe global token usage tracker."""
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            if model not in self.usage:
                self.usage[model] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "calls": 0,
                }
            entry = self.usage[model]
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["total_tokens"] += prompt_tokens + completion_tokens
            entry["calls"] += 1

    def summary(self) -> dict:
        with self._lock:
            total = sum(e["total_tokens"] for e in self.usage.values())
            return {"per_model": dict(self.usage), "total_tokens": total}


# Global singleton (like ALMA's approach)
TRACKER = TokenTracker()
```

### Evolution Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Generations (G) | 30 | Sufficient for convergence based on pilot runs |
| Population size (N) | 8 | Balance diversity vs. evaluation cost |
| Elite count (K) | 3 | Top-3 survive unchanged |
| Tasks per evaluation | 50 (sampled) | Statistical significance within budget |
| Failure samples per mutation | 5 | Enough context for Meta-Architect |
| Max LLM calls per protocol per task | 10 | Token budget constraint |
| Selection strategy | Truncation (top-K) | Simple and effective |
| Mutation operator | LLM-based code rewrite | Semantic mutations via Meta-Architect |
| Crossover | Optional: merge methods from two parents | Tested in ablation |
| Timeout per task | 120 seconds | Prevent infinite loops |
| Validation set | 15% held-out from CL-bench (stratified) | Track overfitting |

### Base Model Choices

| Role | Primary Model | Fallback |
|------|---------------|----------|
| **Meta-Architect** (writes protocol code) | GPT-4o | Claude Opus |
| **Worker LLM** (called within protocols) | GPT-4o-mini | Claude Sonnet |
| **Judge LLM** (evaluation, reuse eval.py) | GPT-5.1 | GPT-4o |

**Rationale**: The Meta-Architect needs strong code generation; the Worker LLM is called frequently and needs to be cost-effective; the Judge needs to be the strongest available for reliable evaluation.

### Compute Resource Estimates

| Component | Cost Per Run | Runs Needed | Total |
|-----------|-------------|-------------|-------|
| Meta-Architect (code generation) | ~$0.50/generation | 30 generations | ~$15 |
| Worker LLM (protocol execution) | ~$0.02/task | 50 tasks x 8 protocols x 30 gen | ~$240 |
| Judge LLM (evaluation) | ~$0.05/task | 50 x 8 x 30 | ~$600 |
| Baseline evaluations | ~$0.05/task | 1,899 tasks x 6 baselines | ~$570 |
| Final full evaluation | ~$0.05/task | 1,899 tasks x 10 methods | ~$950 |
| **Total estimated API cost** | | | **~$2,375** |

**Hardware**: Standard machine with Python 3.10+. No GPU required (all LLM calls are API-based). Recommend concurrent execution with 10-20 workers (reuse `infer.py` concurrency pattern).

---

## 7. Expected Results & Analysis

### Predicted Performance Tables

**Table 1: Overall Accuracy on GravityBench (CL-bench) (%)**

| Method | DomainKnowl (F1) | RuleSystem (F2) | Procedural (F3) | Empirical (F4) | Overall |
|--------|-----------|-------------|------------|--------------|---------|
| Vanilla (ref: CL-bench leaderboard) | ~20 | ~15 | ~12 | ~18 | ~17 |
| CoT | 25 | 22 | 20 | 25 | 23 |
| ReAct | 28 | 25 | 22 | 22 | 25 |
| RAG | 30 | 35 | 15 | 20 | 27 |
| MetaGPT/ChatDev | 32 | 30 | 25 | 28 | 29 |
| GEPA | 40 | 35 | 32 | 35 | 36 |
| **Evo-Protocol** | **55** | **50** | **42** | **45** | **49** |
| Ablation: Prompt-only | 45 | 40 | 35 | 38 | 40 |
| Ablation: No-verification | 38 | 45 | 32 | 40 | 39 |
| Ablation: No-perception | 50 | 28 | 38 | 42 | 40 |

*Note: These are predictions based on the analysis of failure modes. Actual numbers will vary.*

**Table 2: Token Efficiency (Accuracy per 1K tokens)**

| Method | Tokens/Task (avg) | Accuracy | Efficiency |
|--------|-------------------|----------|------------|
| Vanilla | 500 | 16% | 0.32 |
| CoT | 1,500 | 24% | 0.16 |
| GEPA | 2,000 | 39% | 0.20 |
| **Evo-Protocol** | **3,500** | **56%** | **0.16** |

*Evo-Protocol uses more tokens but achieves much higher accuracy. Token-accuracy tradeoff analysis is important.*

### Emergent Case Studies

These are the "aha moments" that should appear in the paper:

#### Case Study 1: Emergent Double-Blind Verification

**Scenario**: Meta-Architect discovers that letting the verifier see the question causes it to be biased toward parametric answers.

**Generation 0**: Simple ask-and-answer. Accuracy on F1: 15%.

**Generation 12**: The Meta-Architect writes:
```python
def verification(self, answer, context):
    # Blind verifier: does NOT receive the question
    verdict = self._call_llm([{
        "role": "system",
        "content": "Check if every claim in the answer is explicitly "
                   "supported by the context. You do not know what "
                   "question was asked. Output PASS or FAIL."
    }, {
        "role": "user",
        "content": f"Context:\n{context}\n\nAnswer:\n{answer}"
    }])
    return "PASS" in verdict
```
**Result**: Accuracy on F1 jumps from 30% to 58%. The system independently discovered evidence isolation.

#### Case Study 2: Emergent Recursive Slicing

**Scenario**: For long documents, the Meta-Architect discovers that single-pass reading fails.

**Generation 8**: Introduces chunking:
```python
def perception(self, context):
    chunks = [context[i:i+2000] for i in range(0, len(context), 2000)]
    relevant = []
    for chunk in chunks:
        score = self._call_llm([{
            "role": "user",
            "content": f"Rate relevance 0-10:\n{chunk}"
        }])
        if int(score) > 5:
            relevant.append(chunk)
    return {"relevant_chunks": relevant}
```
**Result**: Accuracy on F2 (Needle) jumps from 20% to 50%.

#### Case Study 3: Emergent Hypothesis-Test Loop

**Scenario**: For pattern recognition, the Meta-Architect discovers that generating one hypothesis is unreliable.

**Generation 18**: Implements falsification:
```python
def cognition(self, query, perceived_info):
    examples = perceived_info["examples"]

    for attempt in range(3):
        # Generate hypothesis
        hypothesis = self._call_llm([{
            "role": "user",
            "content": f"Given these examples:\n{examples}\n"
                       f"Propose a transformation rule."
        }])

        # Test hypothesis against all examples
        all_match = True
        for inp, expected_out in examples:
            predicted = self._call_llm([{
                "role": "user",
                "content": f"Apply this rule: {hypothesis}\n"
                           f"To input: {inp}\nOutput only the result."
            }])
            if predicted.strip() != str(expected_out):
                all_match = False
                break

        if all_match:
            # Apply verified hypothesis to the test input
            return self._call_llm([{
                "role": "user",
                "content": f"Apply rule: {hypothesis}\nTo: {query}"
            }])

    return "Unable to determine pattern"
```
**Result**: Accuracy on F4 jumps from 18% to 45%.

### Evolution Trajectory Visualization

The paper should include the following figures:

**Figure 1: Accuracy vs. Generation (line plot)**
- X-axis: Generation (0 to 30)
- Y-axis: Accuracy (%)
- Lines: Overall, F1, F2, F3, F4
- Expected: Rapid improvement in first 10 generations, plateau around generation 25

**Figure 2: Code Complexity vs. Generation (line plot)**
- X-axis: Generation
- Y-axis: Lines of code in protocol / number of LLM calls per task
- Expected: Complexity increases then stabilizes (protocols become more sophisticated but not unbounded)

**Figure 3: Emergent Feature Timeline (annotated scatter)**
- X-axis: Generation
- Y-axis: Feature categories (chunking, verification, hypothesis-test, etc.)
- Plot when each feature first appears in the population

**Figure 4: Evo-Protocol vs. GEPA comparison (grouped bar chart)**
- Grouped by failure mode (F1-F4)
- Two bars per group: GEPA, Evo-Protocol

**Figure 5: Ablation heatmap**
- Rows: ablation variants
- Columns: failure modes (F1-F4)
- Cell color: accuracy

### Qualitative Analysis Framework

For each evolved protocol in the final population:

1. **Feature Annotation**: Manually tag which software patterns appear (chunking, verification, loops, hypothesis-testing, role separation, etc.)
2. **Novelty Assessment**: Does this pattern appear in prior work (CoT, ReAct, etc.)? Or is it genuinely emergent?
3. **Failure Mode Specialization**: Which failure modes does each feature primarily address?
4. **Cross-pollination**: Do features from one generation's protocol get adopted by others?

---

## 8. Continuous Learning Loop

### Protocol Zoo Architecture

```
+----------------------------------------------------------+
|                    PROTOCOL ZOO                           |
+----------------------------------------------------------+
|                                                           |
|  +-------------------+  +-------------------+             |
|  | Domain: Legal     |  | Domain: Medical   |             |
|  | Protocol: v12     |  | Protocol: v8      |             |
|  | Accuracy: 72%     |  | Accuracy: 65%     |             |
|  | Features:         |  | Features:         |             |
|  | - blind verify    |  | - dosage checker  |             |
|  | - clause extract  |  | - multi-source    |             |
|  +-------------------+  +-------------------+             |
|                                                           |
|  +-------------------+  +-------------------+             |
|  | Domain: Logic     |  | Domain: Science   |             |
|  | Protocol: v20     |  | Protocol: v15     |             |
|  | Accuracy: 68%     |  | Accuracy: 60%     |             |
|  | Features:         |  | Features:         |             |
|  | - state machine   |  | - hypothesis test |             |
|  | - step verifier   |  | - falsification   |             |
|  +-------------------+  +-------------------+             |
|                                                           |
|  Metadata per protocol:                                   |
|  - source_code: str                                       |
|  - domain_tags: list[str]                                 |
|  - feature_tags: list[str]                                |
|  - accuracy_per_mode: dict[str, float]                    |
|  - evolution_history: list[generation_snapshots]           |
|  - token_cost: float                                      |
+----------------------------------------------------------+
```

**Storage Format**: Each protocol is serialized as:
```json
{
  "protocol_id": "legal-v12",
  "source_code": "class LegalProtocol(BaseProtocol): ...",
  "domain_tags": ["legal", "long-document", "rule-heavy"],
  "feature_tags": ["blind_verification", "clause_extraction", "chunking"],
  "accuracy": {"overall": 0.72, "F1": 0.80, "F2": 0.75, "F3": 0.60, "F4": 0.55},
  "token_cost_avg": 3200,
  "generation": 12,
  "parent_id": "legal-v9"
}
```

### Meta-Router Task Feature Analysis

When a new task arrives, the Meta-Router analyzes its features to select the best protocol from the Zoo:

```python
"""
meta_router.py — Route incoming tasks to the best protocol from the Zoo.
"""

class MetaRouter:
    """Analyze task features and select the best protocol from Protocol Zoo."""

    def __init__(self, llm_client, protocol_zoo: list[dict]):
        self.llm = llm_client
        self.zoo = protocol_zoo

    def analyze_task(self, context: str, query: str) -> dict:
        """Extract task features for routing."""
        features = self._call_llm([{
            "role": "system",
            "content": (
                "Analyze this task and output a JSON with these features:\n"
                "- context_length: short/medium/long\n"
                "- domain: legal/medical/science/general/...\n"
                "- requires_counter_factual: true/false\n"
                "- requires_multi_step_reasoning: true/false\n"
                "- requires_induction: true/false\n"
                "- likely_failure_mode: F1/F2/F3/F4\n"
            )
        }, {
            "role": "user",
            "content": f"Context (first 500 chars): {context[:500]}\n\nQuery: {query}"
        }])
        return json.loads(features)

    def select_protocol(self, task_features: dict) -> dict:
        """Select the best matching protocol from the Zoo."""
        scores = []
        for protocol in self.zoo:
            score = 0
            # Match by failure mode strength
            mode = task_features.get("likely_failure_mode", "F1")
            score += protocol["accuracy"].get(mode, 0) * 2

            # Match by domain
            if task_features.get("domain") in protocol.get("domain_tags", []):
                score += 0.5

            # Match by feature requirements
            if (task_features.get("requires_counter_factual")
                    and "blind_verification" in protocol.get("feature_tags", [])):
                score += 0.3

            scores.append((score, protocol))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]  # Return best-matching protocol
```

### Zero-Shot Transfer Experiment Design

**Experiment**: Test whether protocols evolved on one domain generalize to unseen domains.

| Experiment | Train Domain | Test Domain | Protocol Evolved On | Metric |
|------------|-------------|-------------|---------------------|--------|
| Transfer-1 | Legal (F1+F2) | Medical (F1+F2) | legal-v12 | Accuracy on medical tasks |
| Transfer-2 | Logic (F3) | Math (F3) | logic-v20 | Accuracy on math puzzles |
| Transfer-3 | Science (F4) | Language (F4) | science-v15 | Accuracy on linguistic patterns |
| Transfer-4 | Mixed (all) | Held-out (all) | mixed-v25 | Overall accuracy |
| Meta-Route | All domains | All domains | Zoo + Router | Overall accuracy |

**Expected Result**: Protocols with code-level logic (chunking, verification gates) transfer better than prompt-only strategies because Python logic is domain-agnostic.

---

## 9. Execution Roadmap

### Phase 1: Foundation

**Deliverables**:
- Benchmark abstraction layer (`BaseBenchmark`, `TaskRecord`, registry)
- CL-bench implementation (`CLBenchmark`) with train/val/test split
- `BaseProtocol` class, `ProtocolLoader` (validation + dynamic loading)
- Protocol archive with SHA indexing and lineage tracking
- Baseline evaluations (Vanilla, CoT, ReAct, RAG) on CL-bench
- Confirm baseline accuracy aligns with CL-bench leaderboard (~17-24%)

**Key Files** (layered project structure inspired by ALMA's `core/` + `evals/` + `envs_archive/` separation):
```
context_learning/
├── benchmarks/                  # Benchmark abstraction layer
│   ├── __init__.py
│   ├── base.py                  # BaseBenchmark, TaskRecord, registry
│   └── cl_bench.py              # CL-bench implementation
│
├── core/                        # Evolution engine (like ALMA's core/)
│   ├── __init__.py
│   ├── base_protocol.py         # BaseProtocol abstract class
│   ├── protocol_loader.py       # Validation pipeline + dynamic loading
│   ├── self_repair.py           # ALMA-style debugger loop
│   ├── archive.py               # SHA-indexed protocol archive
│   └── token_tracker.py         # Global token usage monitoring
│
├── data/
│   └── CL-bench.jsonl           # CL-bench data (1,899 tasks)
│
├── eval.py                      # Reused from CL-bench (no changes)
├── infer.py                     # Original CL-bench inference
└── requirements.txt
```

### Phase 2: Evolution Engine

**Deliverables**:
- Meta-Architect prompt and code generation pipeline
- Evolution loop (evaluate → select → mutate → repeat)
- Pilot run: 10 generations on Domain Knowledge Reasoning (F1)
- Verify that evolved protocols outperform generation-0

**Key Files**:
```
context_learning/
├── core/
│   ├── meta_architect.py        # Meta-Architect prompt templates + code gen
│   ├── evolution_loop.py        # Main evolution algorithm
│   └── evaluator.py             # Batch protocol evaluation via BaseBenchmark
│
├── archive/                     # Generated protocol code (ALMA-style)
│   ├── protocol_a1b2c3d4.py     # Evolved protocol (gen 0)
│   ├── protocol_a1b2c3d4_meta.json
│   ├── protocol_e5f6g7h8.py     # Evolved protocol (gen 1)
│   ├── protocol_e5f6g7h8_meta.json
│   └── ...
│
├── baselines/                   # Baseline protocol implementations
│   ├── naive.py                 # NaiveProtocol (generation 0)
│   ├── cot.py                   # Chain-of-thought wrapper
│   └── react.py                 # ReAct wrapper
│
└── configs/
    └── evolution.yaml           # Hyperparameters, model choices, etc.
```

### Phase 3: Full Evolution & Baselines

**Deliverables**:
- Full 30-generation evolution run across all 4 sub-tasks
- GEPA baseline implementation and evaluation
- MetaGPT/ChatDev baseline evaluation
- Complete baseline comparison table

### Phase 4: Analysis & Ablation

**Deliverables**:
- Ablation experiments (prompt-only, code-only, no-verification, no-perception)
- Cross-domain transfer experiments
- Protocol Zoo construction
- Meta-Router implementation and evaluation
- Emergent feature analysis and case study selection
- Evolution trajectory visualizations

### Phase 5: Paper Writing

**Deliverables**:
- Full paper draft (8-10 pages + appendix)
- Key figures: architecture diagram, evolution trajectory, comparison tables, case studies
- Appendix: full protocol code examples, additional ablation results
- Reproducibility package: all code, data, and evolved protocols

---

## Appendix A: Adapting Existing Pipeline

### Adapting `infer.py` for Protocol Execution

The existing `infer.py` handles direct LLM calls. We wrap it with the new abstraction layer:

```python
"""
core/evaluator.py — Run evolved protocols on any registered benchmark.

Uses BaseBenchmark to load tasks as TaskRecords, execute protocols,
and evaluate results. Benchmark-agnostic by design.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from benchmarks.base import BaseBenchmark, TaskRecord, get_benchmark
from core.base_protocol import BaseProtocol
from core.protocol_loader import ProtocolLoader
from core.token_tracker import TRACKER


def run_protocol_on_benchmark(
    protocol: BaseProtocol,
    benchmark_name: str,
    data_path: str,
    split: str = "test",
    output_path: str = None,
    judge_client=None,
    workers: int = 1,
) -> list[TaskRecord]:
    """
    Execute a protocol on a benchmark and return evaluated TaskRecords.

    This is benchmark-agnostic: swapping benchmark_name from "cl-bench"
    to "arc" changes only the data loading and evaluation logic.
    """
    # Load benchmark via registry
    bench = get_benchmark(benchmark_name)
    tasks = bench.load_tasks(data_path, split=split)

    loader = ProtocolLoader(protocol.llm, protocol.model)
    results = []

    def process_one(record: TaskRecord) -> TaskRecord:
        # Build context (handle multi-turn)
        context = record.context
        if len(record.messages_raw) > 2:
            # Multi-turn: augment context with prior turns
            prior = "\n".join(
                f"[{m['role'].upper()}]: {m['content']}"
                for m in record.messages_raw[1:-1]  # skip system, skip last user
            )
            context = context + "\n\n---\nPrior conversation:\n" + prior

        result = loader.run_with_timeout(protocol, context, record.query)
        if result:
            record.model_output = result.answer
            record.reasoning_trace = result.reasoning_trace
            record.tokens_used = result.tokens_used
            record.verification_passed = result.verification_passed
        else:
            record.model_output = ""

        # Evaluate
        bench.evaluate(record, judge_client=judge_client)
        return record

    if workers == 1:
        for task in tqdm(tasks, desc="Evaluating"):
            results.append(process_one(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks)):
                results.append(future.result())

    # Save results (compatible with eval.py output format)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps({
                    "idx": r.metadata.get("idx"),
                    "messages": r.messages_raw,
                    "model_output": r.model_output,
                    "rubrics": r.rubrics,
                    "score": r.score,
                    "metadata": r.metadata,
                }, ensure_ascii=False) + "\n")

    # Print summary
    metrics = bench.get_metrics(results)
    print(f"\n=== Results on {benchmark_name} ({split}) ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


# Usage:
# python -c "
# from core.evaluator import run_protocol_on_benchmark
# from core.protocol_loader import ProtocolLoader
# loader = ProtocolLoader(client, 'gpt-4o-mini')
# protocol = loader.load_from_file('archive/protocol_a1b2c3d4.py')
# run_protocol_on_benchmark(protocol, 'cl-bench', 'data/CL-bench.jsonl', split='test')
# "
```

### Reusing `eval.py`

The existing `eval.py` requires **no modifications**. CL-bench tasks are the native format, so:

```bash
# Evaluate protocol outputs using the existing pipeline
python eval.py --input outputs/evo_protocol_gen30.jsonl --judge-model gpt-5.1

# Evaluate baseline outputs
python eval.py --input outputs/cot_baseline.jsonl --judge-model gpt-5.1
```

The binary scoring system (score 1 iff ALL rubrics are satisfied) is exactly what we need for fitness evaluation in the evolution loop.

---

## Appendix B: Failure Mode Classifier

```python
"""
failure_classifier.py — Classify which failure mode caused an incorrect answer.

Used in the evolution feedback loop to help the Meta-Architect
understand WHY a protocol failed, not just THAT it failed.
"""

def classify_failure_mode(
    context: str,
    query: str,
    model_answer: str,
    correct_answer: str,
    llm_client,
    model: str = "gpt-4o",
) -> dict:
    """
    Classify the failure mode of an incorrect answer.

    Returns:
        {
            "mode": "F1" | "F2" | "F3" | "F4",
            "confidence": float,
            "explanation": str
        }
    """
    prompt = f"""
    A model was given a context and a question. It produced an incorrect answer.
    Classify the failure into one of four modes:

    F1 (Parametric Override): The model's answer reflects pre-trained knowledge
       rather than the context. The context provides different information,
       but the model used its memorized knowledge instead.

    F2 (Context Navigation Failure): The answer shows the model missed or
       overlooked relevant information in the context. The key information
       was present but not attended to (e.g., buried in a long document).

    F3 (Reasoning Breakdown): The model found the relevant information but
       made errors in multi-step reasoning. Intermediate steps contain
       logical errors or the chain of reasoning collapsed.

    F4 (Induction Failure): The model needed to discover a pattern or rule
       from examples in the context. It proposed an incorrect rule that
       doesn't match the examples, or fabricated a plausible-sounding
       but wrong rule.

    Context (first 1000 chars): {context[:1000]}

    Question: {query}

    Model's (incorrect) answer: {model_answer}

    Expected answer: {correct_answer}

    Output JSON: {{"mode": "F1|F2|F3|F4", "confidence": 0.0-1.0, "explanation": "..."}}
    """

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content.strip())
```

---

## Appendix C: Complete Experiment Configuration

```yaml
# configs/evolution.yaml — Full experiment configuration

# === Benchmark ===
# Pluggable: change benchmark_name to use a different benchmark
benchmark:
  name: cl-bench                  # Key into BENCHMARK_REGISTRY
  data_path: data/CL-bench.jsonl
  split_ratio:
    train: 0.7
    val: 0.15
    test: 0.15
  split_stratify_by: context_category

# === Evolution ===
evolution:
  generations: 30
  population_size: 8
  elite_count: 3
  tasks_per_evaluation: 50       # Sampled from train split each generation
  failure_samples_per_mutation: 5
  max_llm_calls_per_task: 10
  timeout_seconds: 120
  selection:
    tau: 0.5                     # Softmax temperature (ALMA-style)
    alpha: 0.5                   # Visit penalty weight (ALMA-style)
  meta_architect_model: gpt-4o
  worker_model: gpt-4o-mini
  seed: 42
  archive_dir: archive/          # SHA-indexed protocol storage

# === Baselines ===
baselines:
  - name: vanilla
    type: direct_call
  - name: cot
    type: direct_call
    system_suffix: "Think step by step."
  - name: react
    type: react_loop
  - name: rag
    type: rag_pipeline
    chunk_size: 1000
  - name: metagpt
    type: multi_agent_static
  - name: gepa
    type: prompt_evolution
    generations: 30              # Match our evolution budget for fair comparison

# === Evaluation ===
evaluation:
  judge_model: gpt-5.1
  workers: 20
  max_retries: 3

# === Transfer Experiments ===
transfer:
  experiments:
    - train: [Domain Knowledge Reasoning]
      test: [Rule System Application]
    - train: [Procedural Task Execution]
      test: [Empirical Discovery & Simulation]
    - train: [Domain Knowledge Reasoning, Rule System Application]
      test: [Procedural Task Execution, Empirical Discovery & Simulation]
    - train: [all]
      test: [held_out]
```
