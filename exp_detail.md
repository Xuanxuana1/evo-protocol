# CFP 实验详细设计：给 Codex 的实现指南

> 基于 `new_plan.md` 中的 Cognitive Focus Protocol (CFP) 架构
> 目标：给 Codex 足够精确的实现规格，使其可以直接编码

---

## 一、代码变更总览

### 新增文件

| 文件 | 职责 |
|------|------|
| `core/base_cfp_protocol.py` | `BaseCFPProtocol` 抽象基类 + fixed runtime |
| `baselines/cfp_seed.py` | Generation-0 seed protocol（手写初始 prompt） |
| `core/cfp_metrics.py` | CFP 专属指标计算函数 |

### 修改文件

| 文件 | 变更内容 |
|------|---------|
| `core/protocol_loader.py` | 新增 `CFPProtocolLoader` 类 |
| `core/meta_architect.py` | 新增 `_build_cfp_prompt()` 方法，`ParentPerformance` 增加 CFP 字段 |
| `core/evolution_loop.py` | `mode="cfp"` 分支、CFP fitness weights、CFP failure attribution |
| `core/failure_classifier.py` | 新增 CFP 模式的 per-agent failure attribution |
| `configs/evolution.yaml` | 新增 `cfp_fitness_weights` 配置块 |
| `run_evolution.py` | 新增 `mode=cfp` 入口 |

---

## 二、`BaseCFPProtocol` 类设计

### 2.1 类定位

`BaseCFPProtocol` 是一个独立的 ABC，与 `BaseTDGCompiler` 平级（不继承它），但复用相同的基础设施模式。具体来说，直接从 `BaseTDGCompiler` 复制以下不可变方法到 `BaseCFPProtocol`，保持实现一致：

- `__init__` — LLM 客户端初始化、token 追踪、环境变量读取
- `_call_llm` — 带重试和 token 记账的 LLM 调用
- `_make_oracle_fn` — 创建 `_oracle()` 函数注入沙箱
- `_sanitize_generated_code` — 规范化 LLM 代码输出
- `_is_syntax_error` — 检测语法错误
- `_attempt_syntax_repair` — 一次性 LLM 语法修复
- `_answer_with_oracle_fallback` — 最后的兜底答案
- `_derive_dynamic_call_budget` — 动态调用预算
- `_prepare_prompt_text` — 文本截断
- `_extract_test_names` — AST 解析 test 函数名
- `_run_tests` — 在沙箱中执行测试
- `_build_test_runner` — 生成测试运行器代码
- `_repair_answer` — 基于失败测试修复答案
- `_build_adversarial_answer` — 生成静态对抗性答案（用于 adversarial validation）

**为什么不继承 `BaseTDGCompiler`**：BaseTDGCompiler 的 `compile_tests` 和 `generate_answer` 是抽象方法，CFP 有完全不同的 5 个方法签名。继承会引入不必要的耦合。

### 2.2 五个 Evolved Methods（抽象方法）

```python
from abc import ABC, abstractmethod

class BaseCFPProtocol(ABC):
    """Base interface for CFP (Cognitive Focus Protocol) compilers.

    Five evolved methods (Meta-Agent mutates these):
      1. build_attention_map    -- context -> 结构化注意力地图
      2. generate_constraints   -- 注意力地图 -> 三层可执行约束
      3. generate_answer        -- 约束引导下生成答案
      4. construct_adversarial  -- 构造对抗性答案暴露约束漏洞
      5. critique_and_patch     -- 独立审查 + 生成补丁测试

    The run() method is fixed infrastructure, NOT evolvable.
    """

    max_llm_calls_per_task: int = 30  # 5 个 agent，预算比 TDG(20) 高

    @abstractmethod
    def build_attention_map(self, context: str, query: str) -> str:
        """Agent 1: 将 context + query 编译为结构化注意力地图。

        输出必须是可被 exec() 执行的 Python code，定义以下变量：
          TASK_TYPE: str
          PIPELINE_DEPTH: "simple" | "full"
          ATTENTION_PRIORITIES: list[dict]
              每个 dict 含 keys: item, source_location, risk_level, risk_type, risk_detail
          TEMPTING_ASSUMPTIONS: list[str]
          WHAT_CONTEXT_SAYS: list[str]
          WHAT_CONTEXT_DOES_NOT_SAY: list[str]
          CROSS_REFERENCES: list[dict]
              每个 dict 含 keys: topic, locations, integration_note
          OUTPUT_FORMAT: dict
        """

    @abstractmethod
    def generate_constraints(self, attention_map_code: str,
                             context: str, query: str) -> str:
        """Agent 2: 将注意力地图转化为三层可执行约束。

        输入：
          attention_map_code: Agent 1 的输出
          context: 原始 context（双源校验，补充遗漏）
          query: 原始 query

        输出必须是 Python code，包含：
          def test_*() 函数 —— Tier 1 硬约束 + Tier 2 oracle 约束
          GENERATION_GUIDANCE = \"\"\" ... \"\"\" —— Tier 3 NL 指导
        每个 test 函数的 docstring 应标注对应的 attention_map 条目。
        """

    @abstractmethod
    def generate_answer(self, context: str, query: str,
                        constraints_code: str, generation_guidance: str,
                        messages_raw: list = None) -> str:
        """Agent 3: 在约束引导下生成答案。

        输入：
          context, query: 原始文本
          constraints_code: Agent 2 的 Tier 1 + Tier 2 代码
          generation_guidance: Agent 2 的 Tier 3 NL 指导
          messages_raw: 多轮对话消息（可选）
        输出：自然语言答案字符串。
        """

    @abstractmethod
    def construct_adversarial(self, constraints_code: str,
                              context: str, query: str) -> str:
        """Agent 4a: 构造对抗性答案。

        尝试构造一个表面通过 Tier 1 测试但实质错误的答案。
        如果无法构造，返回空字符串。
        """

    @abstractmethod
    def critique_and_patch(self, context: str, query: str,
                           answer: str, constraints_code: str,
                           adversarial_answer: str) -> str:
        """Agent 4b: 独立审查，生成补丁测试。

        输入：
          context: 原始 context（独立审查，绕过 Agent 1 盲区）
          query: 原始 query
          answer: Agent 3 的输出
          constraints_code: Agent 2 的约束代码
          adversarial_answer: Agent 4a 的输出（可能为空）
        输出：Python code，包含额外的 def test_*() 函数。
        """
```

### 2.3 Fixed Runtime（`run` 方法）

```python
def run(self, context: str, query: str, max_retries: int = 1,
        messages_raw: list = None) -> SandboxResult:
    """Fixed runtime. Meta-Agent CANNOT modify this method.

    Pipeline:
      Phase 1: build_attention_map → attention_map_code
      Phase 2: generate_constraints → constraints_code + generation_guidance
      Phase 3: generate_answer → draft answer
      Phase 4: construct_adversarial + critique_and_patch → patch_code
      Phase 5: verify all tests → repair if needed
      Phase 6: compute metrics → return SandboxResult
    """

    # --- 初始化 ---
    self._call_count = 0
    self._task_tokens_used = 0
    self._task_prompt_tokens = 0
    self._task_completion_tokens = 0
    self._oracle_call_count = 0
    self._max_llm_calls_current = self._derive_dynamic_call_budget(
        context=context, query=query
    )
    trace: list[str] = []
    sandbox_timeout = max(1, int(getattr(self, "sandbox_timeout_seconds", 30)))

    # ═══ Phase 1: Attention Allocation ═══
    attention_map_code = ""
    attention_map_ns = {}
    pipeline_depth = "full"
    try:
        raw_attention = self.build_attention_map(context, query)
        attention_map_code = self._sanitize_generated_code(raw_attention)
        attention_map_ns = self._exec_attention_map(attention_map_code)
        pipeline_depth = str(attention_map_ns.get("PIPELINE_DEPTH", "full"))
        trace.append(
            f"[AttentionMap] priorities="
            f"{len(attention_map_ns.get('ATTENTION_PRIORITIES', []))} "
            f"depth={pipeline_depth}"
        )
    except Exception as exc:
        trace.append(f"[AttentionMap] error={str(exc)[:200]}")

    # ═══ Phase 2: Constraint Specification ═══
    constraints_code = ""
    constraint_tests: list[str] = []
    generation_guidance = ""
    compilation_success = False
    try:
        raw_constraints = self.generate_constraints(
            attention_map_code, context, query
        )
        constraints_code = self._sanitize_generated_code(raw_constraints)
        constraint_tests = self._extract_test_names(constraints_code)
        generation_guidance = self._extract_generation_guidance(constraints_code)
        trace.append(
            f"[Constraints] tests={len(constraint_tests)} "
            f"guidance_len={len(generation_guidance)}"
        )
    except Exception as exc:
        trace.append(f"[Constraints] error={str(exc)[:200]}")

    # 回译校验
    coverage_gaps = self._compute_coverage_gaps(
        attention_map_ns, constraint_tests, constraints_code
    )

    # 验证 constraints 可执行
    if constraints_code and constraint_tests:
        from core.sandbox_executor import execute_sandbox_code
        compile_result = execute_sandbox_code(
            constraints_code,
            oracle_fn=self._make_oracle_fn(),
            timeout=sandbox_timeout,
        )
        compilation_success = compile_result.success
        if not compilation_success and self._is_syntax_error(
            str(compile_result.error)
        ):
            repaired = self._attempt_syntax_repair(
                "constraints", constraints_code, str(compile_result.error)
            )
            if repaired and repaired != constraints_code:
                constraints_code = repaired
                constraint_tests = self._extract_test_names(constraints_code)
                compile_result = execute_sandbox_code(
                    constraints_code,
                    oracle_fn=self._make_oracle_fn(),
                    timeout=sandbox_timeout,
                )
                compilation_success = compile_result.success
        trace.append(f"[ConstraintCompile] success={compilation_success}")

    # ═══ Phase 3: Focused Generation ═══
    answer = ""
    try:
        answer = self.generate_answer(
            context, query, constraints_code, generation_guidance,
            messages_raw=messages_raw,
        )
        trace.append(f"[GenerateAnswer] len={len(answer)}")
    except Exception as exc:
        trace.append(f"[GenerateAnswer] error={str(exc)[:200]}")
    if not answer:
        answer = self._answer_with_oracle_fallback(
            context=context, query=query
        )

    # ═══ Phase 4: Multi-Perspective Correction ═══
    adversarial_answer = ""
    patch_code = ""
    patch_tests: list[str] = []
    adversarial_constructed = False

    if pipeline_depth == "full" and compilation_success and constraint_tests:
        # 4a: 攻击步
        try:
            adversarial_answer = self.construct_adversarial(
                constraints_code, context, query
            )
            adversarial_constructed = bool(adversarial_answer.strip())
            trace.append(f"[Adversarial] constructed={adversarial_constructed}")
        except Exception as exc:
            trace.append(f"[Adversarial] error={str(exc)[:200]}")

        # 4b: 防御步
        try:
            raw_patch = self.critique_and_patch(
                context, query, answer, constraints_code, adversarial_answer
            )
            patch_code = self._sanitize_generated_code(raw_patch)
            patch_tests = self._extract_test_names(patch_code)
            trace.append(f"[Critique] patch_tests={len(patch_tests)}")
        except Exception as exc:
            trace.append(f"[Critique] error={str(exc)[:200]}")

    # ═══ Phase 5: Verify & Repair ═══
    all_code = constraints_code
    if patch_code:
        all_code = constraints_code + "\n\n" + patch_code
    all_tests = constraint_tests + patch_tests

    test_results: dict[str, bool] = {}
    if all_tests and compilation_success:
        test_results = self._run_tests(
            all_code, answer, all_tests, sandbox_timeout
        )
        trace.append(f"[Verify] pass={sum(test_results.values())}/{len(test_results)}")

    # Targeted repair
    repair_attempts = 0
    failed_original: list[str] = []
    failed_patch: list[str] = []
    for attempt in range(max_retries):
        failed = [n for n, p in test_results.items() if not p]
        if not failed:
            break
        repair_attempts += 1
        failed_original = [t for t in constraint_tests if not test_results.get(t, True)]
        failed_patch = [t for t in patch_tests if not test_results.get(t, True)]
        repaired = self._repair_answer(
            context=context, query=query, answer=answer,
            test_code=all_code, failed_tests=failed,
            messages_raw=messages_raw,
        )
        if repaired and repaired != answer:
            answer = repaired
            test_results = self._run_tests(
                all_code, answer, all_tests, sandbox_timeout
            )
        else:
            break

    # ═══ Phase 6: Compute Metrics ═══
    total_tests = len(test_results)
    passed_tests = sum(1 for v in test_results.values() if v)
    test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    n_priorities = len(attention_map_ns.get("ATTENTION_PRIORITIES", []))
    constraint_coverage_ratio = (
        len(constraint_tests) / max(n_priorities, 1) if n_priorities > 0 else 0.0
    )
    critique_additions = len(patch_tests)
    critique_caught = len(
        [t for t in patch_tests if not test_results.get(t, True)]
    )

    return SandboxResult(
        answer=answer,
        confidence=max(0.2, test_pass_rate),
        reasoning_trace=trace,
        verification_passed=(total_tests > 0 and passed_tests == total_tests),
        tokens_used=self._task_tokens_used,
        prompt_tokens=self._task_prompt_tokens,
        completion_tokens=self._task_completion_tokens,
        metadata={
            "stage": "verified" if (total_tests > 0 and passed_tests == total_tests) else "partial_pass",
            "mode": "cfp",
            "llm_calls": self._call_count,
            "oracle_calls": self._oracle_call_count,
            "test_pass_rate": test_pass_rate,
            "test_results": test_results,
            "repair_attempts": repair_attempts,
            "compilation_success": compilation_success,
            "num_constraint_tests": len(constraint_tests),
            "num_patch_tests": len(patch_tests),
            "pipeline_depth": pipeline_depth,
            "attention_map_code": attention_map_code[:2000],
            "attention_priority_count": n_priorities,
            "tempting_assumptions_count": len(
                attention_map_ns.get("TEMPTING_ASSUMPTIONS", [])
            ),
            "constraint_coverage_ratio": float(constraint_coverage_ratio),
            "coverage_gaps": coverage_gaps,
            "adversarial_constructed": adversarial_constructed,
            "critique_additions": critique_additions,
            "critique_caught_violations": critique_caught,
            "failed_original_tests": failed_original,
            "failed_patch_tests": failed_patch,
        },
        sandbox_code=all_code,
        solver_code="",
        execution_output=str(test_results),
        execution_success=bool(answer),
        compilation_success=compilation_success,
    )
```

### 2.4 新增基础设施方法

以下方法定义在 `BaseCFPProtocol` 内部，不可被进化修改：

```python
def _exec_attention_map(self, code: str) -> dict:
    """安全执行 attention_map code，提取变量到 dict。"""
    from core.sandbox_executor import execute_sandbox_code
    result = execute_sandbox_code(code, timeout=10)
    if not result.success or not result.namespace:
        return {}
    keys = [
        "TASK_TYPE", "PIPELINE_DEPTH", "ATTENTION_PRIORITIES",
        "TEMPTING_ASSUMPTIONS", "WHAT_CONTEXT_SAYS",
        "WHAT_CONTEXT_DOES_NOT_SAY", "CROSS_REFERENCES", "OUTPUT_FORMAT",
    ]
    return {k: result.namespace[k] for k in keys if k in result.namespace}

def _extract_generation_guidance(self, constraints_code: str) -> str:
    """从 constraints code 中提取 GENERATION_GUIDANCE 变量。"""
    from core.sandbox_executor import execute_sandbox_code
    result = execute_sandbox_code(constraints_code, timeout=10)
    if result.success and result.namespace:
        val = result.namespace.get("GENERATION_GUIDANCE", "")
        if val:
            return str(val)
    # Fallback: 正则匹配
    import re
    for quote in ('"""', "'''"):
        pattern = rf'GENERATION_GUIDANCE\s*=\s*{quote}([\s\S]*?){quote}'
        match = re.search(pattern, constraints_code)
        if match:
            return match.group(1).strip()
    return ""

def _compute_coverage_gaps(
    self,
    attention_map_ns: dict,
    constraint_tests: list[str],
    constraints_code: str,
) -> list[str]:
    """识别 attention_map 中未被 constraints 覆盖的优先项。

    方法：对每个 ATTENTION_PRIORITIES item，提取关键词（长度>3，取前5个），
    检查是否至少有一个出现在 constraints_code 中。
    """
    priorities = attention_map_ns.get("ATTENTION_PRIORITIES", [])
    if not priorities:
        return []
    gaps = []
    code_lower = constraints_code.lower()
    for p in priorities:
        item_text = str(p.get("item", "")).lower()
        keywords = [w for w in item_text.split() if len(w) > 3][:5]
        if not keywords:
            continue
        if not any(kw in code_lower for kw in keywords):
            gaps.append(str(p.get("item", "")))
    return gaps
```

---

## 三、指标计算详细规格

### 3.1 Runtime Metrics（运行时计算，不需要 rubric）

这些指标在 `BaseCFPProtocol.run()` 的 Phase 6 中直接计算，存入 `metadata`。

#### `constraint_coverage_ratio`

**含义**：注意力地图中的优先项有多少比例被转化为了可执行测试。

```python
constraint_coverage_ratio = len(constraint_tests) / max(len(ATTENTION_PRIORITIES), 1)
```

- 值域 `[0, +inf)`，可以 > 1.0（一个优先项对应多个测试）
- 作为 fitness 输入时 clip 到 `[0, 1]`
- `= 0`：Agent 2 没有为任何优先项生成测试 → 严重问题
- `< 0.5`：覆盖不足 → Meta-Architect 应改进 Agent 2
- `>= 1.0`：每个优先项至少有一个测试 → 理想状态

#### `critique_discovery_rate`

**含义**：critique 的补丁测试中，实际检测到 answer 问题的比例。

```python
critique_caught_violations = len([t for t in patch_tests if not test_results[t]])
critique_discovery_rate = critique_caught_violations / max(len(patch_tests), 1)
```

- `> 0` 且跨任务稳定：涌现能力的证据（critique 发现了 constraints 没覆盖的问题）
- `= 0`：两种可能 — constraints 已充分覆盖，或 critique 本身太弱
- `= 1`：所有 patch tests 都发现问题 → constraints 严重不足

#### `adversarial_constructed`

**含义**：Agent 4a 是否成功构造了对抗性答案。

```python
adversarial_constructed = bool(adversarial_answer.strip())
```

- 如果跨任务失败率 > 80%，应简化 Agent 4a 为"列举漏洞"模式而非完整构造答案

#### `coverage_gaps`

**含义**：attention_map 中哪些优先项没有被任何 constraint test 覆盖。

```python
# 对每个 priority:
#   提取 item 中长度 > 3 的词，取前 5 个作为 keywords
#   如果没有任何 keyword 出现在 constraints_code 中 → 标记为 gap
```

- 直接传入 failure feedback，让 Meta-Architect 知道 Agent 2 遗漏了什么
- 列表形式，每个元素是一个未覆盖的 priority item 文本

### 3.2 Post-hoc Metrics（评估后计算，需要 rubric/judge 结果）

这些指标在 `evolution_loop.py` 的 `_evaluate_protocol()` 中，拿到 judge 评分后计算。实现放在 `core/cfp_metrics.py`。

#### `attention_map_coverage`

**含义**：rubric 的评判标准中，有多少被注意力地图覆盖了。

**计算方法**：

```python
def compute_attention_map_coverage(
    attention_map_ns: dict,
    all_rubrics: list[str],
) -> float:
    """
    对每条 rubric，提取关键词（长度>3，取前8个）。
    检查 attention_map 所有文本字段中是否有 >= 1/3 关键词命中。
    coverage = 被覆盖的 rubric 数 / 总 rubric 数
    """
    if not all_rubrics:
        return 1.0
    attention_text = _flatten_attention_map_text(attention_map_ns).lower()
    covered = 0
    for rubric in all_rubrics:
        keywords = [w for w in rubric.lower().split() if len(w) > 3][:8]
        if not keywords:
            covered += 1  # 无法判断的 rubric 默认覆盖
            continue
        hits = sum(1 for kw in keywords if kw in attention_text)
        if hits >= max(1, len(keywords) // 3):
            covered += 1
    return covered / max(len(all_rubrics), 1)

def _flatten_attention_map_text(ns: dict) -> str:
    """展平 attention_map namespace 为单个文本字符串。"""
    parts = []
    for p in ns.get("ATTENTION_PRIORITIES", []):
        parts.append(str(p.get("item", "")))
        parts.append(str(p.get("risk_detail", "")))
    for item in ns.get("TEMPTING_ASSUMPTIONS", []):
        parts.append(str(item))
    for item in ns.get("WHAT_CONTEXT_SAYS", []):
        parts.append(str(item))
    for item in ns.get("WHAT_CONTEXT_DOES_NOT_SAY", []):
        parts.append(str(item))
    for ref in ns.get("CROSS_REFERENCES", []):
        parts.append(str(ref.get("topic", "")))
        parts.append(str(ref.get("integration_note", "")))
    return " ".join(parts)
```

**数据来源**：`all_rubrics` 从 CL-bench 的 task record 中获取（`task.rubrics` 字段）。

#### `attention_map_precision`

**含义**：注意力地图中的条目有多少与 rubric 真正相关（而非噪声）。

```python
def compute_attention_map_precision(
    attention_map_ns: dict,
    all_rubrics: list[str],
) -> float:
    """
    对每个 ATTENTION_PRIORITIES item，提取关键词。
    检查是否有 >= 1/3 关键词出现在 rubric 文本中。
    precision = 相关的 priority 数 / 总 priority 数
    """
    priorities = attention_map_ns.get("ATTENTION_PRIORITIES", [])
    if not priorities:
        return 0.0
    if not all_rubrics:
        return 1.0
    rubric_text = " ".join(r.lower() for r in all_rubrics)
    relevant = 0
    for p in priorities:
        keywords = [
            w for w in str(p.get("item", "")).lower().split() if len(w) > 3
        ][:8]
        if not keywords:
            continue
        hits = sum(1 for kw in keywords if kw in rubric_text)
        if hits >= max(1, len(keywords) // 3):
            relevant += 1
    return relevant / max(len(priorities), 1)
```

#### `answer_focus_adherence`

**含义**：answer 是否覆盖了 critical 级别的注意力优先项。

```python
def compute_answer_focus_adherence(
    attention_map_ns: dict,
    answer: str,
) -> float:
    """
    只检查 risk_level == "critical" 的优先项。
    对每个，提取关键词，检查 answer 中是否有 >= 1/3 命中。
    adherence = 被 answer 覆盖的 critical 项数 / 总 critical 项数
    """
    priorities = attention_map_ns.get("ATTENTION_PRIORITIES", [])
    critical = [p for p in priorities if p.get("risk_level") == "critical"]
    if not critical:
        return 1.0
    answer_lower = answer.lower()
    addressed = 0
    for p in critical:
        keywords = [
            w for w in str(p.get("item", "")).lower().split() if len(w) > 3
        ][:5]
        if not keywords:
            addressed += 1
            continue
        hits = sum(1 for kw in keywords if kw in answer_lower)
        if hits >= max(1, len(keywords) // 3):
            addressed += 1
    return addressed / max(len(critical), 1)
```

#### `bottleneck_agent`

**含义**：这次任务失败主要卡在哪个 agent。

```python
def identify_bottleneck(metadata: dict, task_passed: bool) -> str:
    """规则推理，基于 metadata 推断瓶颈 agent。"""
    if task_passed:
        return "none"

    # 优先级从高到低
    if metadata.get("attention_priority_count", 0) == 0:
        return "agent_1_attention_map_failed"

    if not metadata.get("compilation_success", False):
        return "agent_2_constraint_compilation_failed"

    n_pri = metadata.get("attention_priority_count", 0)
    n_gaps = len(metadata.get("coverage_gaps", []))
    if n_pri > 0 and n_gaps / n_pri > 0.5:
        return "agent_2_low_coverage"

    test_pass_rate = metadata.get("test_pass_rate", 0)
    if test_pass_rate > 0.8:
        # tests 通过但 judge 说答案错 → tests 太弱
        return "agent_2_tests_too_permissive"

    if metadata.get("failed_original_tests"):
        return "agent_3_ignored_constraints"

    if metadata.get("failed_patch_tests"):
        # critique 发现了 constraints 没覆盖的问题，这是涌现
        return "agent_4_found_gaps"

    return "agent_3_answer_quality"
```

#### `bottleneck_category`

```python
def describe_bottleneck(metadata: dict, bottleneck: str) -> str:
    """将 bottleneck_agent 转为人类可读的描述。"""
    if bottleneck == "agent_1_attention_map_failed":
        return "Attention map generation failed or produced no priorities"
    if bottleneck == "agent_2_constraint_compilation_failed":
        return "Constraint code failed to compile"
    if bottleneck == "agent_2_low_coverage":
        gaps = metadata.get("coverage_gaps", [])
        return f"Agent 2 missed {len(gaps)} priorities: {gaps[:3]}"
    if bottleneck == "agent_2_tests_too_permissive":
        return "All tests pass but answer is wrong - constraints too weak"
    if bottleneck == "agent_3_ignored_constraints":
        failed = metadata.get("failed_original_tests", [])
        return f"Answer failed {len(failed)} constraint tests: {failed[:3]}"
    if bottleneck == "agent_4_found_gaps":
        failed = metadata.get("failed_patch_tests", [])
        return f"Critique caught {len(failed)} gaps (emergence evidence)"
    return "General answer quality issue"
```

### 3.3 Cross-Generation Metrics（跨代追踪，在 evolution_loop 的代级日志中记录）

#### `focus_efficiency`

```python
focus_efficiency = accuracy / max(mean_tokens_per_task, 1)
```

单位 token 产出的准确率，CFP 的终极效率指标。

#### `schema_evolution_velocity`

```python
def compute_schema_velocity(current_keys: set[str], parent_keys: set[str]) -> float:
    """Jaccard distance: 当前 vs parent 的 attention_map 类别差异。"""
    union = current_keys | parent_keys
    if not union:
        return 0.0
    intersection = current_keys & parent_keys
    return 1.0 - len(intersection) / len(union)
```

- 快速收敛到 0 → 找到了稳定的认知语言
- 持续 > 0.3 → schema 还在剧烈变化，搜索未收敛

---

## 四、CFP Fitness Weights

在 `configs/evolution.yaml` 中新增：

```yaml
cfp_fitness_weights:
  answer_correctness: 0.50
  test_pass_rate: 0.15
  constraint_coverage_ratio: 0.10
  compilation_success: 0.10
  critique_effectiveness: 0.10
  execution_success: 0.05
  false_positive_penalty: 0.30
  adversarial_test_pass_penalty: 0.15
```

**计算公式**：

```python
def compute_cfp_fitness(eval_results: list[dict], weights: dict) -> float:
    n = len(eval_results)
    if n == 0:
        return 0.0
    accuracy = sum(r["score"] for r in eval_results) / n
    test_pass = sum(r["metadata"].get("test_pass_rate", 0) for r in eval_results) / n
    coverage = sum(
        min(1.0, r["metadata"].get("constraint_coverage_ratio", 0))
        for r in eval_results
    ) / n
    compilation = sum(
        1 if r["metadata"].get("compilation_success") else 0
        for r in eval_results
    ) / n
    critique_eff = sum(
        r["metadata"].get("critique_caught_violations", 0)
        / max(r["metadata"].get("critique_additions", 1), 1)
        for r in eval_results
    ) / n
    execution = sum(
        1 if r["metadata"].get("execution_success") else 0
        for r in eval_results
    ) / n
    # 惩罚：test 通过率高但 judge 说错 → tests 太弱
    fp_penalty = sum(
        1 if r["metadata"].get("test_pass_rate", 0) > 0.8 and r["score"] == 0
        else 0
        for r in eval_results
    ) / n

    fitness = (
        weights["answer_correctness"] * accuracy
        + weights["test_pass_rate"] * test_pass
        + weights["constraint_coverage_ratio"] * coverage
        + weights["compilation_success"] * compilation
        + weights["critique_effectiveness"] * critique_eff
        + weights["execution_success"] * execution
        - weights.get("false_positive_penalty", 0.3) * fp_penalty
    )
    return max(0.0, fitness)
```

---

## 五、CFPProtocolLoader

在 `core/protocol_loader.py` 中新增 `CFPProtocolLoader`，结构与 `TDGProtocolLoader` 对称。

关键差异：

### `_check_cfp_contract()`

```python
required_methods = {
    "build_attention_map",
    "generate_constraints",
    "generate_answer",
    "construct_adversarial",
    "critique_and_patch",
}
forbidden_overrides = {"run", "_call_llm", "_make_oracle_fn"}
```

- 验证 5 个 required methods 全部存在
- 验证 forbidden 方法未被覆盖
- 验证每个 evolved method 内部调用了 `self._call_llm`
- 验证不存在对未定义 `self.*` 方法的调用

### `load_from_code()`

动态加载时查找 `BaseCFPProtocol` 的子类（而非 `BaseTDGCompiler`）。

### Allowed imports

与 TDG 相同，加上 `"core.base_cfp_protocol"`。

---

## 六、Evolution Loop 集成

### 6.1 `_evaluate_protocol()` 中增加 post-hoc metrics

```python
# 在 task 评估结果拿到 judge score 后
if self.config.mode == "cfp" and result and result.metadata:
    from core.cfp_metrics import (
        compute_attention_map_coverage,
        compute_attention_map_precision,
        compute_answer_focus_adherence,
        identify_bottleneck,
        describe_bottleneck,
    )
    attn_ns = self._parse_attention_map(result.metadata.get("attention_map_code", ""))
    rubrics = getattr(task, "rubrics", []) or []

    result.metadata["attention_map_coverage"] = compute_attention_map_coverage(attn_ns, rubrics)
    result.metadata["attention_map_precision"] = compute_attention_map_precision(attn_ns, rubrics)
    result.metadata["answer_focus_adherence"] = compute_answer_focus_adherence(attn_ns, result.answer)
    result.metadata["bottleneck_agent"] = identify_bottleneck(result.metadata, score > 0)
    result.metadata["bottleneck_category"] = describe_bottleneck(
        result.metadata, result.metadata["bottleneck_agent"]
    )
```

### 6.2 Fitness 计算分支

```python
if self.config.mode == "cfp":
    fitness = compute_cfp_fitness(eval_results, self.config.cfp_fitness_weights)
```

### 6.3 `ParentPerformance` 扩展

新增字段：
```python
attention_map_coverage: float = 0.0
constraint_coverage_ratio: float = 0.0
critique_discovery_rate: float = 0.0
adversarial_escape_rate: float = 0.0
top_bottleneck_agents: list[str] = field(default_factory=list)
coverage_gaps_summary: list[str] = field(default_factory=list)
```

从评估结果中聚合填充这些字段，传入 Meta-Architect 的 mutation prompt。

---

## 七、Failure Classifier CFP 扩展

在 `build_failure_feedback()` 中增加 CFP 上下文：

```python
if metadata and metadata.get("mode") == "cfp":
    bottleneck = identify_bottleneck(metadata, False)
    cfp_context = (
        "\n\nCFP (Cognitive Focus Protocol) failure context:\n"
        f"- Attention priorities: {metadata.get('attention_priority_count', 0)}\n"
        f"- Constraint coverage: {metadata.get('constraint_coverage_ratio', 0):.2f}\n"
        f"- Coverage gaps: {metadata.get('coverage_gaps', [])[:3]}\n"
        f"- Pipeline depth: {metadata.get('pipeline_depth')}\n"
        f"- Critique additions: {metadata.get('critique_additions', 0)}\n"
        f"- Critique caught: {metadata.get('critique_caught_violations', 0)}\n"
        f"- Adversarial constructed: {metadata.get('adversarial_constructed')}\n"
        f"- Bottleneck: {bottleneck}\n"
        "\nPer-agent attribution guide:\n"
        "- agent_1 bottleneck → improve attention map prompt\n"
        "- agent_2 bottleneck → improve constraint generation\n"
        "- agent_3 bottleneck → improve answer generation\n"
        "- agent_4 found gaps → emergence working, strengthen constraints\n"
    )
```

---

## 八、Ablation 实验配置

每个 ablation 通过 CLI 参数 `--ablation` 控制，在 `BaseCFPProtocol.run()` 中通过环境变量或实例属性读取。

| 配置名 | 改动点 | 目的 |
|--------|--------|------|
| `full` | 无改动 | baseline |
| `no_attention_map` | Phase 1 跳过，`attention_map_code = ""` | 注意力地图的价值 |
| `no_critique` | `pipeline_depth` 强制 `"simple"` | 多视角校正的贡献 |
| `no_adversarial` | 跳过 Agent 4a，`adversarial_answer = ""` | 对抗性视角的价值 |
| `no_constraints_to_answer` | Agent 3 收到空 constraints 和空 guidance | 主动遵守 vs 被动修复 |
| `full_context_to_all` | 不做 relevant_spans 裁剪（阶段一不裁剪） | 注意力裁剪的价值 |
| `info_asymmetry_only` | Agent 4b 用 Agent 2 的 prompt 角色（看到 answer 但不用 critique 视角） | 分离信息不对称 |
| `perspective_only` | Agent 4b 不看 answer（用 critique 视角但无额外信息） | 分离视角多样性 |
| `direct_prompting` | 无 agent，直接 LLM 回答 | agent 系统 vs baseline |

---

## 九、运行命令

```bash
# 基本 CFP 进化
python run_evolution.py \
    --mode cfp \
    --initial-code baselines/cfp_seed.py \
    --generations 5 \
    --population-size 6 \
    --tasks-per-eval 20

# Ablation: no critique
python run_evolution.py \
    --mode cfp \
    --initial-code baselines/cfp_seed.py \
    --ablation no_critique \
    --generations 3 \
    --tasks-per-eval 20

# 涌现分离: 仅信息不对称
python run_evolution.py \
    --mode cfp \
    --initial-code baselines/cfp_seed.py \
    --ablation info_asymmetry_only \
    --generations 3 \
    --tasks-per-eval 20
```

---

## 十、预期输出格式

每次评估的 JSON 日志：

```json
{
    "generation": 1,
    "sha": "abc123def456",
    "task_id": "task_42",
    "score": 1,
    "fitness": 0.72,
    "metadata": {
        "mode": "cfp",
        "pipeline_depth": "full",
        "attention_priority_count": 8,
        "tempting_assumptions_count": 3,
        "constraint_coverage_ratio": 0.75,
        "coverage_gaps": ["No sector breakdown mentioned"],
        "num_constraint_tests": 6,
        "num_patch_tests": 2,
        "test_pass_rate": 0.875,
        "adversarial_constructed": true,
        "critique_additions": 2,
        "critique_caught_violations": 1,
        "attention_map_coverage": 0.85,
        "attention_map_precision": 0.72,
        "answer_focus_adherence": 0.80,
        "bottleneck_agent": "agent_2_low_coverage",
        "bottleneck_category": "Agent 2 missed 1 priorities: ['No sector breakdown']",
        "failed_original_tests": ["test_no_date_assumption"],
        "failed_patch_tests": ["test_sector_scope"]
    }
}
```
