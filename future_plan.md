# Future Plan: Constitutional Multi-Agent Protocol (CMAP)

> 基于两轮实验分析与头脑风暴整理，2026-02-23

---

## 一、背景与现状

### 研究出发点

CL-bench 是一个以**长上下文理解**为核心的评测集，任务难点在于：模型预训练获得的参数先验知识会与上下文中提供的专属规则发生冲突（parametric override）。目前业界所有前沿模型直接回答的 accuracy 均在 25% 以下。

当前系统采用 TDG（Test-Driven Generation）范式：evolved protocol 负责生成测试（`compile_tests`）和生成答案（`generate_answer`），通过遗传算法跨代优化。早期树形进化在 Gen 2 达到约 50% accuracy（6 任务评估），显著优于 baseline，说明"code 作为 agent 间通信语言"这一核心思路有效。

### 两轮实验暴露的工程问题

**实验一**（`tasks_per_eval=10`，树形进化）：
- Gen 2 最佳 accuracy 50%（sha=`01bc23989ca5`）
- Gen 3 大幅退化至 30%，同一 protocol 在相同 calibration 任务上得分从 1 变 0

**实验二**（`tasks_per_eval=16`，Gen 0 预分化）：
- 最佳 accuracy 仅 18.75%（3/16），均值 fitness Gen 3 转负
- 两个候选 fitness 为负值（false positive 惩罚项过强）

**根因分析（五条）**：

| # | 问题 | 根因 |
|---|------|------|
| 1 | 跨代评估不可比 | 每代独立采样不同任务，同一 protocol fitness 波动达 0%↔50% |
| 2 | Selection 排名被惩罚项反转 | `accuracy=18.75%` 的 protocol 排在 `6.25%` 后面（false_positive_rate 惩罚过重） |
| 3 | F3 失败占主导（50-60%） | `compile_tests` 与 `generate_answer` 使用同一 LLM，共享认知盲区；测试不覆盖 rubric 的真正要求 |
| 4 | Gen 0 预分化适得其反 | 盲 mutation（无 failure 反馈）+ 高 temperature 破坏有效部分；6 个不同 parent 的失败反馈分散无法叠加 |
| 5 | `textwrap.dedent` 对 architect prompt 失效 | 多行插值变量使公共缩进为零，LLM 收到缩进混乱的指令，mutation 质量下降 |

**结论**：树形进化（`init_population_size=1`）优于 Gen 0 预分化，因为 Gen 1 的 6 次同 protocol 评估提供了可叠加的失败画像，后续 mutation 是有信号的定向搜索而非盲猜。

---

## 二、核心研究假设

> **当 code 不仅作为验证工具（测试），而是作为 multi-agent 系统中所有 agent 之间的唯一通信介质时，agent 间的协作会产生涌现能力——系统整体表现显著超过任何单个 agent 的能力上界。**

现有 TDG 的涌现有限，原因：

1. **单向通信**：test_code 只在 verify 阶段接触 answer，answer generator 生成时完全不知道 tests 的存在
2. **无共享世界模型**：两个 method 各自独立解读 context，没有结构化的共同理解
3. **同源盲区**：同一个 LLM 的两次调用，认知盲区高度重叠

---

## 三、新架构：Constitutional Multi-Agent Protocol（CMAP）

### 3.1 信息流设计

```
context ──→ [Agent 1] decompose_context()
  query        │
               ▼
          requirements_code
               │
               ├──→ [Agent 2] generate_constraints(context, query, decomposition)
               │                      │
               │                      ▼
               │              constraints_code ──→ [Agent 3] generate_answer()
               │                      │                    │
               │                      │                    ▼
               │                      │                  answer
               │                      │                    │
               │                      ▼                    ▼
               │              constraints_code + answer ──→ [Agent 4] critique()
               │                                               │
               │                                               ▼
               └───────────────── patch_code ─────────────────┘

Fixed Runtime: verify(answer, constraints + patch) → repair if needed
```

**核心原则**：Agent 之间只通过 code 通信。自然语言只出现在最终 answer 中。

### 3.2 四个 Evolved Methods

#### Agent 1：`decompose_context(context, query) -> str`

**职责**：将非结构化 context 编译为结构化 Python 数据对象。这是整个系统的信息瓶颈，后续所有 agent 的表现上限取决于此步的提取质量。

**输出示例**：
```python
TASK_TYPE = "financial_analysis"

KEY_FACTS = [
    "Dataset contains 250 rows of S&P 500 data",
    "No date column exists in the data",
    "Columns: Open, High, Low, Close, Volume",
]

EXPLICIT_CONSTRAINTS = [
    "Do not assume a start date if not provided",
    "Provide R-squared for any regression",
    "Discuss correlation vs causation when mentioning external events",
]

IMPLICIT_REQUIREMENTS = [
    "User asks about 'April' but data has no dates — must flag this ambiguity",
]

PROHIBITED = [
    "Fabricating specific dates or date ranges",
    "Claiming causal relationships without evidence",
]

OUTPUT_FORMAT = {
    "structure": "analytical sections with headers",
    "must_include": ["quantified observations", "regression analysis", "anomaly discussion"],
}
```

**关键价值**：迫使 LLM 在生成答案前显式处理 context 中的每一个重要元素。F1 问题（参数覆盖）在此步被结构化暴露——decomposition 没出现的约束，后续测试也不会覆盖，failure trace 可以精确归因到"Agent 1 遗漏了这条"。

#### Agent 2：`generate_constraints(context, query, decomposition) -> str`

**职责**：将 decomposition 的每一条要求转化为可执行的三层约束文档（见第四节）。

**与当前 `compile_tests` 的关键差异**：
- 有结构化输入，从 decomposition 的分类列表中逐条生成对应类型的测试
- 双源校验：同时参考 raw context（补充遗漏）和 decomposition（提供结构）
- 每个测试函数 docstring 标注对应 decomposition 中的哪一条（便于 failure attribution）

#### Agent 3：`generate_answer(context, query, constraints_code, messages_raw) -> str`

**职责**：在已知约束条件下生成答案。

**与当前 `generate_answer` 的关键差异**：constraints_code 作为输入的一部分传入 answer generator，使其能**主动遵守**约束，而非事后被动修复。这是"code 作为宪法约束"的核心体现。

```
System prompt: "You are generating an answer that must pass ALL of the
following executable Python tests. Read each test carefully."

<constraints>
{constraints_code}  ← Agent 2 的输出，包含三层约束
</constraints>

User: [context + query]
```

#### Agent 4：`critique_and_patch(context, query, answer, constraints_code) -> str`

**职责**：这是产生涌现的关键 agent。

**工作方式**：
1. 重新阅读原始 context（不依赖 decomposition，避免继承 Agent 1 的盲区）
2. 审视当前 answer 的具体内容
3. 审视已有 constraints_code
4. 生成**补丁测试**——针对已有 constraints 未覆盖但 answer 实际违反的要求

**为什么能打破共享盲区**：
- Agent 1+2 在生成 constraints 时没有见过 answer，测试基于"对 context 的一般性理解"
- Agent 4 同时看到 answer 和 context，能发现 answer 的具体问题，生成针对性测试
- 信息不对称：Agent 4 拥有 Agent 1+2 编写时不具备的信息（actual answer），因此能产生 Agent 1+2 无法预见的测试

**涌现的可观测证据**：当 `critique_and_patch` 生成的 patch tests 检测到了 constraints 没有覆盖的问题并修复了 answer，这就是系统整体超越任何单个 agent 能力的量化证明。

### 3.3 Fixed Runtime（不被进化修改）

```python
def run(self, context, query, max_retries=1, messages_raw=None):
    # Phase 1: Understand
    decomposition = sanitize_as_python(self.decompose_context(context, query))

    # Phase 2: Specify
    constraints_code = sanitize(self.generate_constraints(context, query, decomposition))
    constraint_tests = extract_test_names(constraints_code)

    # Phase 3: Generate（answer generator 可见 constraints）
    answer = self.generate_answer(context, query, constraints_code, messages_raw)

    # Phase 4: Critique（生成补丁测试）
    patch_code = sanitize(self.critique_and_patch(context, query, answer, constraints_code))
    patch_tests = extract_test_names(patch_code)

    # Phase 5: Merge & Verify
    all_code  = constraints_code + "\n\n" + patch_code
    all_tests = constraint_tests + patch_tests

    adversarial_results = run_tests(all_code, all_tests, build_adversarial_answer())
    test_results = run_tests(all_code, all_tests, answer)

    # Phase 6: Targeted Repair（区分 original constraint 失败 vs patch 失败）
    if any_failed(test_results):
        failed_original = [t for t in constraint_tests if not test_results[t]]
        failed_patch    = [t for t in patch_tests    if not test_results[t]]
        answer = repair(context, query, answer, all_code, failed_original, failed_patch)

    return SandboxResult(answer=answer, metadata={
        "decomposition": decomposition,
        "constraint_count": len(constraint_tests),
        "patch_count": len(patch_tests),
        "phase_attribution": {
            "critique_additions": len(patch_tests),
            "critique_caught_violations": len(failed_patch),
        }
    })
```

---

## 四、三层约束文档（NL-Code Gap 的解决方案）

### 问题来源

rubric 中的约束横跨一个表达力谱系：

```
完全可 code 化             oracle 可桥接               不可机械化
      ↓                        ↓                           ↓
"包含实体 LB-334"         "法律语气不能确定化"          "论证质量是否充分"
"consent 出现在评分前"     "是否标注了 R² 的局限性"      "解释风格是否得体"
```

纯 code 只能处理最左边；最右边只能靠 judge；中间这一大块用 `_oracle` 桥接——把 NL 语义判断封装成 focused 的 binary LLM call，转化为可执行 code。

### 三层设计

Agent 2 生成的 constraints 文档包含三个显式分层：

```python
# ═══════════════════════════════════════════════════
# TIER 1: HARD CONSTRAINTS（确定性 code）
# 失败 → 立即触发 repair，信号可靠
# ═══════════════════════════════════════════════════

def test_consent_before_scorecard(answer):
    """[fd6f] Process gate: screening must precede risk score"""
    scorecard_pos = answer.lower().find("risk score")
    consent_pos   = answer.lower().find("consent")
    if scorecard_pos != -1:
        assert consent_pos != -1 and consent_pos < scorecard_pos, \
            "Risk scorecard generated before consent confirmation"

def test_all_entities_covered(answer):
    """[f672] Entity coverage: all key lots/devices/fabs must appear"""
    required = ["LB-334", "XYZ-100", "Fab B Texas"]
    missing  = [e for e in required if e not in answer]
    assert not missing, f"Missing entities: {missing}"

def test_no_date_assumption(answer):
    """[02cc] No assumed start date when dataset lacks dates"""
    forbidden = ["started in", "beginning of", "from january", "since q1"]
    hits = [p for p in forbidden if p in answer.lower()]
    assert not hits, f"Assumed date: {hits}"


# ═══════════════════════════════════════════════════
# TIER 2: SEMANTIC CONSTRAINTS（oracle 桥接）
# 失败 → repair，信号有轻微噪声
# ═══════════════════════════════════════════════════

def test_legal_hedging(answer):
    """[d5bc] Legal outcomes must not be expressed as predetermined"""
    return _oracle(
        "Does this legal analysis avoid expressing court outcomes as certain "
        "or highly likely? It should use conditional framing like 'courts might', "
        "'one argument would be', never 'courts will find' or 'this would succeed'. "
        f"Answer:\n{answer[:2000]}",
        bool
    )

def test_chess_pros_cons_present(answer):
    """[2abc] Pros/cons of winning move must be explicitly listed"""
    return _oracle(
        "Does this chess explanation explicitly list BOTH the advantages (pros) "
        "and disadvantages (cons) of Player 2's winning move, attributed specifically "
        f"to Player 2? Answer:\n{answer[:2000]}",
        bool
    )

def test_causal_boundary_discussed(answer):
    """[02cc] Must discuss correlation vs. causation limits"""
    return _oracle(
        "Does this financial analysis explicitly acknowledge the limits of "
        "inferring causation from correlation when discussing external factors "
        f"(e.g., tariffs, policy events)? Answer:\n{answer[:2000]}",
        bool
    )


# ═══════════════════════════════════════════════════
# TIER 3: GENERATION GUIDANCE（NL，不执行，注入 answer generator）
# 处理真正不可机械化的风格/语用约束
# ═══════════════════════════════════════════════════

GENERATION_GUIDANCE = """
STYLE REQUIREMENTS (apply throughout your response):

[d5bc - Legal tone]
- First time you use any legal term (HIPAA, estoppel, etc.), add a plain-language
  parenthetical immediately after: e.g. "HIPAA (a US law protecting health information)"
- Frame all legal predictions as arguments, not conclusions:
  WRONG: "This monitoring would succeed under the 4th Amendment"
  RIGHT: "One argument is that this monitoring could satisfy the narrow-tailoring test"

[02cc - Epistemic honesty]
- If the dataset lacks explicit dates, state this as your FIRST observation
  and qualify ALL temporal claims accordingly
- When mentioning external events (tariffs, elections), always note:
  "correlation does not establish causation; multiple factors could explain..."

[fd6f - Clinical communication]
- Before any risk assessment: verify screening completeness + restate consent scope
- Phrase any incomplete-screening situation as a question ("Has the Edinburgh
  scale been completed?"), not a deferral

[2abc - Discourse completeness]
- Structure: [winning move] → [pros for Player 2] → [cons for Player 2]
  → [blocking options for Player 1, each tied to the specific winning move]

[f672 - Entity coverage]
- When discussing yield issues, ensure EVERY lot, device type, and fab
  mentioned in the analysis report appears explicitly in your response
"""
```

### 三层的失败归因

| 失败类型 | 归因 | 修复方式 |
|---------|------|---------|
| Tier 1 test 失败 | Agent 2 正确编码了约束，answer generator 没遵守 | repair loop 有精确信号 |
| Tier 2 oracle test 失败 | Agent 2 的 oracle 问题措辞不够 focused | 改进 oracle 问题的表述 |
| Tier 3 guidance 被忽视 | Agent 3 的 prompt 没有足够强调 | 改进 answer generator 对 guidance 的处理方式 |
| Tier 2 false positive | oracle 对错误答案也返回 True | 改进 oracle 问题的判别边界 |

---

## 五、Evolution 策略

### 5.1 保持树形进化

`init_population_size=1`，Gen 1 用单一 seed 的 6 次评估建立可靠 failure profile，Gen 2+ 基于充分失败分析做定向 mutation。Gen 0 预分化（高 temperature 盲 mutation）在少代数场景弊大于利。

### 5.2 Per-Agent Failure Attribution

当前 TDG 只能归因到 F1/F2/F3/F4（protocol 级）。CMAP 可做细粒度归因：

| 失败现象 | 归因 agent | Meta-Architect 修复目标 |
|---------|-----------|----------------------|
| decomposition 遗漏了 rubric 中的关键要求 | Agent 1 | 改进 decompose prompt 的提取粒度 |
| decomposition 完整但 constraints 没覆盖 | Agent 2 | 改进约束转化策略 |
| constraints 完整但 answer 没遵守 | Agent 3 | 改进 answer 对 constraints 的主动利用 |
| constraints 不足但 critique 也没发现 | Agent 4 | 改进 critique 策略 |
| constraints 不足但 critique 成功补上 | 系统正常 | **涌现能力的证据** |

### 5.3 关键参数建议

| 参数 | 建议值 | 原因 |
|------|--------|------|
| `tasks_per_eval` | 20+ | 降低评估方差，16 任务仍不足 |
| `calibration_tasks` | 6 | 固定锚点防止跨代不可比 |
| `task_overlap_ratio` | 0.5 | 部分重叠提供连续性 |
| `worker_temperature` | 0 | 降低 LLM 非确定性 |
| `elite_count` | 2 | 保留多样性同时聚焦 |
| `selection_score` penalty | 降低 FP penalty 从 0.6 → 0.2 | 防止 accuracy=18% 被 accuracy=6% 压制 |

---

## 六、涌现的量化指标

为了验证 CMAP 产生了真正的涌现能力，需要追踪：

1. **`patch_effectiveness`**：Agent 4 生成的 patch tests 中，实际检测到 answer 问题的比例。若 > 0 且稳定，说明 critique 产生了 constraints 没有的新验证能力。

2. **`accuracy_with_critique - accuracy_without_critique`**：开启和关闭 Agent 4 的 accuracy 差异。差值随进化代数增大 → 进化在优化 agent 间协作，涌现能力在增强。

3. **`constraint_attribution_accuracy`**：decomposition 中列出的要求，最终有多少比例被 Tier 1 或 Tier 2 tests 覆盖。

---

## 七、Ablation 实验方案

在 seed protocol 上，依次关闭每个 agent：

| 配置 | 描述 | 验证假设 |
|------|------|---------|
| Full CMAP | 全部 4 个 agent | baseline |
| No critique | 关闭 Agent 4 | critique 的增量贡献 |
| No decomposition | Agent 2 直接从 raw context 生成 constraints | decomposition 的价值 |
| No constraints-to-answer | Agent 3 不看 constraints（等价于当前 TDG） | 主动遵守 vs 被动修复 |
| Direct prompting | 无 agent，直接回答 | TDG 系列 vs baseline |

---

## 八、实现优先级（给 Codex）

1. **`BaseCMAPProtocol`**：定义 4 个抽象方法 + fixed runtime + `_sanitize_decomposition()`
2. **Seed protocol**：手写 4 个 method 的初始 prompt，质量决定进化起点
3. **三层 constraint document**：在 runtime 中解析 TIER 1/2/3，将 `GENERATION_GUIDANCE` 注入 Agent 3 的 system prompt
4. **修改 `MetaArchitect`**：增加 CMAP mode prompt template，包含 4-agent 角色描述和 per-agent failure attribution
5. **修改 failure summary**：增加 phase attribution 指标（decomposition completeness, constraint coverage, critique effectiveness）
6. **修复已知 bug**（与架构无关，但影响进化稳定性）：
   - `textwrap.dedent` 改为 placeholder + 后插值
   - selection score 排名改为 accuracy 优先的分层排序
   - `elite_count > population_size` 加校验
7. **跑 ablation**：先确认架构本身有效，再接入 evolution loop
8. **Evolution**：复用现有 GA 框架，只改 protocol loader 和 evaluation 部分

---

## 九、风险与缓解

| 风险 | 严重度 | 缓解方案 |
|------|--------|---------|
| decomposition 质量差导致信息丢失 | 高 | Agent 2 同时接收 raw context 和 decomposition 作为双源输入；Agent 4 直接访问 raw context 做独立审查 |
| 4 个 method 导致 token 成本过高 | 中 | decomposition 限制输出长度（结构化 Python 比叙述文本紧凑）；constraints 复用 decomposition 避免重复处理 context |
| Agent 3 学会"应付"constraints 而非理解 context | 中 | Agent 4 基于 context（不是 constraints）审查，能检测表面合规但实质错误的答案 |
| 搜索空间增大导致进化收敛慢 | 中 | Per-agent failure attribution 让 architect 只改需要改的 method；保持树形进化集中 mutation 预算 |
| LLM 非确定性使 decomposition 每次不同 | 高 | temperature=0 + decomposition prompt 强调确定性结构化输出格式 |
| Tier 2 oracle 本身不稳定 | 中 | oracle 问题聚焦单一属性（比 judge 全局评估更可靠）；oracle 失败可归因并改进问题措辞 |
