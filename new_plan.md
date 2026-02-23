# Cognitive Focus Protocol (CFP)

# Multi-Agent 认知焦点方法论：通过显式注意力管理解决长上下文理解

> 基于头脑风暴整理，2026-02-23
> 前序文档：`future_plan.md`（CMAP 初版设计）

---

## 一、思想基础：网眼之外

### 1.1 核心洞察

世界的本质是"溢出的"——它永远比我们的精确化工具能捕捉的更多。每一次定义都同时是一次遗漏（Derrida 的 différance）。科学、法律、经济学，全都是在世界上投下一张网，网眼之外的东西，我们假装它不存在。

这个哲学洞察直接映射到 LLM 系统的核心困境：

- **语言塑造认知**（Sapir-Whorf）：LLM 能"思考"什么，取决于我们给它什么样的 context 和 prompt。没有对应的概念框架，LLM 就无法把相应的信息从混沌中抽离出来——就像文盲不知道偏微分方程不是因为笨，而是认知工具箱里没有"函数""极限"这些前置概念。

- **前语言直觉存在但不可传递**：LLM 的参数中编码了大量"直觉"（parametric knowledge），但这些直觉如果不被显式化为 code 或结构化表达，就无法被验证、修正和迭代。

- **语言化是文明加速的引擎**：一个人悟到的东西如果不能语言化就会消失，但一旦语言化了，它就能被质疑、被修正、被发展。对 agent 系统同理——agent 间的知识必须以可执行的形式（code）外化，才能被积累和进化。

### 1.2 对系统设计的指导原则

| 哲学原理 | 工程推论 |
|---------|---------|
| 每次精确化都是一次遗漏 | 单一 agent 的任何输出都有盲区，系统必须有多重"网"来互补 |
| 语言划定可思考的边界 | Decomposition 的输出格式决定了下游 agent 能"想到"什么 |
| 不同语言揭示不同现实 | 不同的表征格式（Python dict / test function / NL guidance）天然提供视角多样性 |
| 说不出的东西无法迭代 | Agent 间只通过 code 通信——确保所有中间知识可验证、可进化 |
| 扩张认知的同时承认溢出 | 系统不追求完美覆盖，而追求最大化"被捕获的关键信息比例" |

---

## 二、核心研究假设（从 CMAP 升级）

### 2.1 原假设（CMAP）

> 当 code 作为 multi-agent 系统中所有 agent 之间的通信介质时，agent 间的协作会产生涌现能力——系统整体表现显著超过任何单个 agent 的能力上界。

### 2.2 新假设（认知焦点方法论）

> **长上下文任务的瓶颈不是模型能力不够，而是注意力分配不当。通过显式化注意力分配、将注意力优先级翻译为可执行验证、并用多视角校正遗漏，可以在不增加模型能力的情况下显著提升长上下文任务的表现。**

关键区别：

- CMAP 假设的核心词是"涌现"——强调 multi-agent 协作产生超越个体的能力
- 新假设的核心词是"注意力分配"——更精确地定位了问题本质，且**任务无关**（不绑定特定评测集）

两者不矛盾，而是层级关系：认知焦点是底层方法论，涌现是方法论正确运行后的可观测效果。

---

## 三、认知焦点方法论：三层原理

### 3.1 显式注意力分配（Explicit Attention Allocation）

**问题**：LLM 处理长上下文时，注意力隐式分配在 transformer 的 attention weights 中，不可观测、不可控。当 context 很长时，关键信息被淹没的概率随长度增长。

**解决**：在生成答案之前，用一个独立步骤生成结构化的**注意力地图**（Attention Map）。这一步的核心价值是把 LLM 内部不可观测的注意力过程，外化为可检查、可修正的 code 产物。

**关键**：注意力地图不是"摘要"。摘要只保留信息，注意力地图还标注**认知风险**——哪些地方模型最容易犯错、哪些信息之间存在张力、哪些是 LLM 先验知识可能覆盖上下文的危险区。

### 3.2 可执行的注意力验证（Executable Focus Verification）

**问题**：即使有了注意力地图，answer generator 可能仍然"知道该关注什么但没做到"——理解约束和遵守约束是两件事。

**解决**：将注意力优先级翻译为可执行的测试。不是"我觉得答案应该关注 X"，而是 `assert "X" in answer`。测试失败 = 注意力偏移的确定性信号。

**三层约束谱系**（沿用 CMAP 设计，重新定位）：

```
完全可 code 化          oracle 可桥接            不可机械化
      ↓                     ↓                      ↓
  Tier 1: 硬约束        Tier 2: 语义约束       Tier 3: 生成指导
  (assert 语句)          (_oracle 桥接)          (NL 注入 prompt)
  信号确定               信号有轻微噪声           无执行反馈
```

这三层不是三种测试，而是三种不同精确度的"语言"在同一个通信协议中共存——对应不同粒度的"网眼"。

### 3.3 多视角注意力校正（Multi-Perspective Focus Correction）

**问题**：注意力地图的编写者也有盲区。用一种视角投下的网，总有漏网之鱼。

**解决**：用不同的视角重新审查注意力分配。视角多样性的来源有两个：

- **信息不对称**（谁在什么时间点看到什么）：critique agent 看到了 answer 的具体形态，能发现前序 agent 无法预见的问题
- **表征多样性**（同样的信息通过不同的格式处理）：输出 Python dict 时被迫做分类枚举，输出 test function 时被迫做边界判断——格式本身就是视角

**核心论点：表征格式就是视角。** 不需要不同的模型或额外的 LLM 调用来获得视角多样性，只需要强制不同的输出格式。格式逼出不同的思考方式。

---

## 四、涌现的双源机制：信息不对称 + 视角多样性

### 4.1 为什么需要区分两个机制

在当前设计中，Agent 4（critique）相对于 Agent 2（constraints）既有不同的信息（看到了 answer），又有不同的角色（审查者 vs 生成者）。两者纠缠在一起，无法分辨哪个在驱动涌现。

理解这个区分对工程实现有直接影响：
- 信息不对称的设计空间很小（DAG 拓扑排列组合有限）
- 视角多样性的设计空间大得多（每个 agent 的输出格式、角色设定、正/负面思维方向都可调），且很多旋钮几乎免费（改 prompt 即可）

### 4.2 两个机制解决不同层次的问题

| 认知层级 | 含义 | 信息不对称能解决？ | 视角多样性能解决？ |
|---------|------|-----------------|-----------------|
| Known knowns | 约束已写出，答案已遵守 | - | - |
| Known unknowns | 知道有些东西难测，但缺信息 | **能**（Agent 4 看到 answer 后补测试） | 辅助 |
| Unknown unknowns | 所有 agent 的"语言"都没覆盖 | 不能（信息再多，同一透镜看不到） | **能**（不同透镜覆盖不同维度） |

**结论**：信息不对称给深度（在特定方向挖得更深），视角多样性给广度（覆盖更多认知维度）。资源有限时，视角多样性的 ROI 更高，因为它的设计空间更大且很多手段几乎免费。

### 4.3 视角多样性的免费工程杠杆

#### 杠杆 1：对偶视角的 decomposition

当前 decomposition 只有正面提取。加入负面/陷阱视角几乎不增加成本：

```python
# 正面视角：context 说了什么
WHAT_CONTEXT_SAYS = [...]

# 负面视角：context 刻意没说什么
WHAT_CONTEXT_DOES_NOT_SAY = [...]

# 陷阱视角：从回答者的认知偏差出发
TEMPTING_ASSUMPTIONS = [
    "Reader will assume financial data has dates — but this dataset has NO date column",
    "Reader will assume correlation implies causation for tariff events",
]
```

`TEMPTING_ASSUMPTIONS` 不是在描述 context 包含什么，而是在描述"一个粗心的回答者会犯什么错"——从回答者的认知偏差出发而非从文本内容出发。这是一种完全不同的视角，但只是 prompt 里多几行指令。

#### 杠杆 2：对抗性视角

让 critique agent 分为两步：

1. **攻击步**：给定 constraints_code，尝试构造一个表面通过所有测试但实质错误的 adversarial answer
2. **防御步**：如果构造成功了，说明 constraints 有漏洞——针对漏洞生成 patch test

Agent 2 写 constraints 时是防守者心态（"怎么确保答案正确"），攻击步是攻击者心态（"怎么骗过这些测试"）。同样的 constraints_code，防守者看到保障，攻击者看到漏洞。成本：一次额外 LLM 调用。

#### 杠杆 3：跨表征翻译作为错误检测

每次信息跨越表征边界（NL → Python data → test code → NL answer → patch code），都是自动的错误检测机会。就像中文翻译成英文再翻译回来，回译差异揭示原文歧义。

具体做法：在 Agent 2 生成 constraints 后，加一个轻量级回译校验——从 test code 中提取"这些测试在检查什么"的自然语言摘要，与 attention_map 做比对。覆盖率差异就是明确信号。

### 4.4 涌现机制分离实验

若要严格验证两个机制的贡献，可做以下 ablation：

| 配置 | 信息不对称 | 视角多样性 | 实现方式 |
|------|-----------|-----------|---------|
| A: 完整设计 | 有 | 有 | Agent 4 看到 answer + 用 critique 视角 |
| B: 仅信息 | 有 | 无 | Agent 4 看到 answer + 用和 Agent 2 相同的 prompt 视角 |
| C: 仅视角 | 无 | 有 | Agent 4 不看 answer + 用对抗性审查视角独立重写 constraints |
| D: 无 Agent 4 | 无 | 无 | baseline |

B vs C 的比较直接回答"哪个更重要"。不需要改架构，只需要改 Agent 4 的 prompt 和输入。

---

## 五、架构设计：认知焦点协议（Cognitive Focus Protocol）

### 5.1 核心设计变化（相对于 CMAP 初版）

| 维度 | CMAP 初版 | CFP 新版 | 变化原因 |
|------|----------|---------|---------|
| Agent 1 输出 | 平坦的事实列表 | **注意力地图**（含冲突风险、跨引用、认知陷阱标注） | 注意力管理而非信息提取 |
| 下游 agent 输入 | 每个 agent 都收到 full context | Agent 1 是唯一深读全文的节点，下游只收到**注意力地图 + 相关段落** | 消除注意力浪费 |
| 流水线深度 | 固定 4 agent | Agent 1 额外输出 `pipeline_depth` 建议，简单任务可短路 | 适应不同任务复杂度 |
| 进化目标 | 4 个 agent 的 prompt 内容 | **Decomposition schema**（用什么类别、什么粒度切割 context） | 统一进化目标，消解耦合 |
| Critique 机制 | 直接审查 | **攻击-防御双步**（先构造 adversarial answer 暴露漏洞，再补 patch） | 引入对抗性视角多样性 |

### 5.2 信息流设计

```
context ──→ [Agent 1] build_attention_map(context, query)
  query        │
               ▼
         attention_map  ──→  pipeline_depth_decision
               │                    │
               │              ┌─ simple: skip to Agent 3
               │              └─ complex: full pipeline
               │
               ├──→ [Agent 2] generate_constraints(attention_map, relevant_spans)
               │                      │
               │                      ▼
               │              constraints_code
               │                      │
               │                      ├──→ [回译校验] extract_test_summary(constraints_code)
               │                      │        vs attention_map → coverage_gap_signal
               │                      │
               │                      ├──→ [Agent 3] generate_answer(
               │                      │         query, constraints_code,
               │                      │         generation_guidance, relevant_spans)
               │                      │              │
               │                      │              ▼
               │                      │            answer
               │                      │              │
               │                      ▼              ▼
               │              [Agent 4a] construct_adversarial(constraints_code, context, query)
               │                      │
               │                      ▼
               │              adversarial_answer（暴露 constraints 漏洞）
               │                      │
               │                      ▼
               │              [Agent 4b] critique_and_patch(
               │                   context, query, answer, constraints_code,
               │                   adversarial_example)
               │                      │
               │                      ▼
               └──────────── patch_code
                                      │
                                      ▼
                     Fixed Runtime: verify → repair if needed
```

**关键原则**：

1. **Agent 1 是唯一的全文读者**——它的注意力地图决定了下游 agent 的认知边界
2. **Agent 间通过 code 通信**——自然语言只出现在最终 answer 和 Tier 3 guidance 中
3. **Agent 4 有对抗性双步**——先攻击（暴露漏洞）再防御（补 patch）
4. **流水线深度自适应**——简单任务不需要完整 4-agent 链

### 5.3 Agent 1：`build_attention_map(context, query) -> str`

**职责升级**：从"信息提取"变为"注意力管理"。不只是提取 context 说了什么，还要标注认知风险、跨引用关系、和 LLM 参数先验的冲突点。

**输出格式**（注意力地图）：

```python
# === 任务元信息 ===
TASK_TYPE = "financial_analysis"
PIPELINE_DEPTH = "full"  # "simple" | "full" — 供 runtime 决定是否短路

# === 注意力焦点（按认知风险排序） ===
ATTENTION_PRIORITIES = [
    {
        "item": "Dataset has no date column",
        "source_location": "paragraph 5",
        "risk_level": "critical",
        "risk_type": "parametric_override",  # LLM 先验与 context 冲突
        "risk_detail": "LLM strongly expects date columns in financial data",
    },
    {
        "item": "250 rows of S&P 500 data with Open/High/Low/Close/Volume",
        "source_location": "paragraph 3",
        "risk_level": "medium",
        "risk_type": "scope_boundary",  # 回答者可能越界推断
        "risk_detail": "Only 5 columns available, no sector/ticker info",
    },
]

# === 认知陷阱（回答者最可能犯的错误） ===
TEMPTING_ASSUMPTIONS = [
    "Assuming financial data has dates → fabricating date ranges",
    "Assuming correlation between volume spikes and external events implies causation",
    "Assuming S&P 500 data represents the entire market",
]

# === 正面要求 ===
WHAT_CONTEXT_SAYS = [
    "Provide R-squared for any regression",
    "Discuss correlation vs causation when mentioning external events",
]

# === 负面要求（context 刻意没说的 / 留白） ===
WHAT_CONTEXT_DOES_NOT_SAY = [
    "No start date, no end date, no time period specification",
    "No sector breakdown or individual stock data",
]

# === 跨段引用（信息散布在不同位置，需要整合） ===
CROSS_REFERENCES = [
    {
        "topic": "volume anomaly analysis",
        "locations": ["paragraph 7", "paragraph 12", "table 2"],
        "integration_note": "Three separate mentions must be synthesized",
    },
]

# === 输出格式要求 ===
OUTPUT_FORMAT = {
    "structure": "analytical sections with headers",
    "must_include": ["quantified observations", "regression analysis", "anomaly discussion"],
}
```

**与 CMAP 初版的关键差异**：

1. `ATTENTION_PRIORITIES` 带有 `risk_level` 和 `risk_type` —— 不只是"这很重要"，而是"为什么重要，模型在这里最可能犯什么错"
2. `TEMPTING_ASSUMPTIONS` —— 从回答者的认知偏差视角出发，这是纯粹的视角多样性贡献
3. `WHAT_CONTEXT_DOES_NOT_SAY` —— 负面视角，标注留白和缺失
4. `CROSS_REFERENCES` —— 标注散布信息的整合需求，对长上下文特别重要
5. `PIPELINE_DEPTH` —— 隐式路由功能，消解了单独路由器的需要

**任务自适应**：不同评测集的注意力焦点不同，但都可以通过调整 `risk_type` 的类别来适配：

| 评测类型 | 主要 risk_type | 对应的注意力地图重点 |
|---------|---------------|-------------------|
| CL-bench（有冲突） | `parametric_override` | 标注 LLM 先验与 context 的冲突 |
| SCROLLS / QuALITY | `information_scatter` | 标注信息散布位置和整合需求 |
| Multi-doc | `cross_document_conflict` | 标注文档间矛盾和互补 |
| 大海捞针 | `signal_buried_in_noise` | 标注关键细节在噪声中的位置 |
| 长文档摘要 | `theme_priority` | 标注主题层级和相对重要性 |

**进化如何作用于此**：decomposition schema 的类别都是进化的对象。进化可以增加新类别、移除无效类别、调整粒度——搜索"什么样的认知语言让注意力分配最有效"。

### 5.4 Agent 2：`generate_constraints(attention_map, relevant_spans) -> str`

**输入变化**：不再接收 full context，而是接收注意力地图 + 由 runtime 根据注意力地图裁剪的相关段落。

**输出结构**（四层）：

```python
# ═══════════════════════════════════════════════════
# TIER 0: ATTENTION VALIDATION（验证注意力地图本身）
# ═══════════════════════════════════════════════════

def test_attention_completeness(attention_map, context):
    """context 中的关键信息段是否都被 attention_map 覆盖了"""
    ...

def test_attention_relevance(attention_map, query):
    """attention_map 中的每条是否都和 query 相关"""
    ...

# ═══════════════════════════════════════════════════
# TIER 1: HARD CONSTRAINTS（确定性 code）
# 失败 → 立即触发 repair，信号可靠
# ═══════════════════════════════════════════════════

def test_no_date_assumption(answer):
    """[02cc] No assumed start date when dataset lacks dates"""
    forbidden = ["started in", "beginning of", "from january", "since q1"]
    hits = [p for p in forbidden if p in answer.lower()]
    assert not hits, f"Assumed date: {hits}"

# ═══════════════════════════════════════════════════
# TIER 2: SEMANTIC CONSTRAINTS（oracle 桥接）
# 失败 → repair，信号有轻微噪声
# ═══════════════════════════════════════════════════

def test_causal_boundary_discussed(answer):
    """[02cc] Must discuss correlation vs. causation limits"""
    return _oracle(
        "Does this financial analysis explicitly acknowledge the limits of "
        "inferring causation from correlation when discussing external factors?",
        answer, bool
    )

# ═══════════════════════════════════════════════════
# TIER 3: GENERATION GUIDANCE（NL，不执行，注入 answer generator）
# ═══════════════════════════════════════════════════

GENERATION_GUIDANCE = """
STYLE REQUIREMENTS (apply throughout your response):
...
"""
```

每个测试函数的 docstring 标注对应 attention_map 中的哪一条，便于 failure attribution。

### 5.5 Agent 3：`generate_answer(query, constraints_code, generation_guidance, relevant_spans) -> str`

**关键变化**：输入中**不包含 full context**，只包含 relevant_spans + constraints_code + Tier 3 guidance。Agent 3 的注意力完全由 Agent 1 的注意力地图引导。

**Prompt 结构**：
```
System: "You are generating an answer that must pass ALL of the following
executable Python tests. Read each test carefully. Pay special attention
to TEMPTING_ASSUMPTIONS — these are traps you are likely to fall into."

<constraints>
{constraints_code}
</constraints>

<generation_guidance>
{tier_3_guidance}
</generation_guidance>

<relevant_context>
{relevant_spans}  ← 注意力地图裁剪后的相关段落，不是全文
</relevant_context>

User: {query}
```

### 5.6 Agent 4：攻击-防御双步 critique

#### Agent 4a：`construct_adversarial(constraints_code, context, query) -> str`

**职责**：给定 constraints_code，构造一个**表面通过所有 Tier 1 测试但实质错误的 adversarial answer**。

如果无法构造成功的 adversarial answer（所有 constraints 都足够严密），这一步可以短路。

#### Agent 4b：`critique_and_patch(context, query, answer, constraints_code, adversarial_example) -> str`

**职责**：综合以下信息生成补丁测试：
1. 原始 context（独立审查，不依赖 decomposition，避免继承 Agent 1 盲区）
2. 当前 answer 的具体内容
3. adversarial_example 暴露的 constraints 漏洞
4. 已有 constraints_code

**输出**：patch_code（补丁测试函数）

### 5.7 Fixed Runtime

```python
def run(self, context, query, max_retries=1, messages_raw=None):
    # ═══ Phase 1: Attention Allocation ═══
    attention_map = sanitize_as_python(
        self.build_attention_map(context, query)
    )
    relevant_spans = extract_relevant_spans(context, attention_map)
    pipeline_depth = attention_map.get("PIPELINE_DEPTH", "full")

    # ═══ Phase 2: Constraint Specification ═══
    constraints_code = sanitize(
        self.generate_constraints(attention_map, relevant_spans)
    )
    constraint_tests = extract_test_names(constraints_code)
    generation_guidance = extract_tier3_guidance(constraints_code)

    # 回译校验：constraints 是否覆盖了 attention_map 的所有优先项
    coverage_gaps = cross_check_coverage(attention_map, constraints_code)

    # ═══ Phase 3: Focused Generation ═══
    answer = self.generate_answer(
        query, constraints_code, generation_guidance,
        relevant_spans, messages_raw
    )

    # ═══ Phase 4: Multi-Perspective Correction ═══
    if pipeline_depth == "full":
        # 4a: 攻击步 — 暴露 constraints 漏洞
        adversarial = self.construct_adversarial(
            constraints_code, context, query
        )

        # 4b: 防御步 — 生成补丁测试
        patch_code = sanitize(
            self.critique_and_patch(
                context, query, answer, constraints_code, adversarial
            )
        )
        patch_tests = extract_test_names(patch_code)
    else:
        patch_code = ""
        patch_tests = []

    # ═══ Phase 5: Verify & Repair ═══
    all_code  = constraints_code + "\n\n" + patch_code
    all_tests = constraint_tests + patch_tests

    test_results = run_tests(all_code, all_tests, answer)

    if any_failed(test_results):
        failed_original = [t for t in constraint_tests if not test_results[t]]
        failed_patch    = [t for t in patch_tests    if not test_results[t]]
        answer = repair(
            context, query, answer, all_code,
            failed_original, failed_patch
        )

    # ═══ Metadata for Evolution ═══
    return SandboxResult(answer=answer, metadata={
        "attention_map": attention_map,
        "pipeline_depth": pipeline_depth,
        "coverage_gaps": coverage_gaps,
        "constraint_count": len(constraint_tests),
        "patch_count": len(patch_tests),
        "token_usage_per_agent": measure_token_usage(),
        "phase_attribution": {
            "attention_map_coverage": compute_coverage(attention_map, constraints_code),
            "attention_map_precision": compute_precision(attention_map, query),
            "constraint_coverage": len(constraint_tests) / max(
                len(attention_map.get("ATTENTION_PRIORITIES", [])), 1
            ),
            "critique_additions": len(patch_tests),
            "critique_caught_violations": len(
                [t for t in patch_tests if not test_results.get(t, True)]
            ),
            "adversarial_success": adversarial is not None,
        }
    })
```

---

## 六、进化策略

### 6.1 核心变化：单一进化目标消解耦合

**问题**：如果引入独立的路由器和能力库，三者同时进化会形成强耦合——改了路由，同样的能力库得分变了；改了能力库，路由决策失效。搜索空间指数膨胀。

**解决**：Decomposition schema 就是路由，就是注意力地图，就是能力选择。

Agent 1 的输出结构隐式决定下游所有 agent 的行为：

```
进化目标（唯一）            下游效果（因果跟随，无需独立进化）
      │
decomposition schema ──→ 输出了 PARAMETRIC_OVERRIDE_RISKS？
      │                    → Agent 2 自动生成 anti-hallucination tests
      │                    → Agent 3 知道哪里不能信自己的直觉
      │
      │                  输出了 CROSS_REFERENCES？
      │                    → Agent 2 生成跨段一致性测试
      │                    → Agent 3 在答案中做信息整合
      │
      │                  输出了 PIPELINE_DEPTH = "simple"？
      │                    → Runtime 跳过 Agent 4（短路，节省 token）
```

进化只搜索一个空间（decomposition schema + 各 agent 的 prompt），不是三个耦合的空间。

### 6.2 保持树形进化

`init_population_size=1`，Gen 1 用单一 seed 的多次评估建立可靠 failure profile，后续基于充分失败分析做定向 mutation。

### 6.3 高密度进化信号

**当前问题**：fitness 信号太粗（一个 task 只有 pass/fail），需要很多轮进化积累足够信号。

**认知焦点方法论自然带来更密的信号**：

```python
evaluation_signal = {
    # 粗信号（已有）
    "task_accuracy": 0 or 1,

    # 细粒度信号（新增 — 每个都可操作）
    "attention_map_coverage": 0.85,     # 关键信息被注意力地图覆盖的比例
    "attention_map_precision": 0.72,    # 注意力地图中与 query 真正相关的比例
    "constraint_coverage": 0.80,        # 注意力优先项被转化为测试的比例
    "answer_focus_adherence": 0.65,     # answer 遵循注意力焦点的程度
    "critique_discovery_rate": 0.30,    # critique 发现的新问题比例
    "adversarial_escape_rate": 0.15,    # adversarial answer 能骗过多少测试

    # 归因信号（定位修复方向）
    "bottleneck_agent": "agent_1",
    "bottleneck_category": "missed CROSS_REFERENCES in paragraph 7-12",
}
```

**信号密度高 → 每轮信息量大 → 总轮数少。** Meta-Architect 收到的不再是"accuracy 掉了，不知道为什么"，而是"attention_map_coverage 从 0.85 掉到 0.60，因为新 mutation 删掉了 CROSS_REFERENCES 类别"——修复方向明确。

### 6.4 Per-Agent Failure Attribution（升级版）

| 失败现象 | 归因 | 可操作的修复方向 |
|---------|------|----------------|
| attention_map 遗漏了关键信息 | Agent 1 | 改进 decomposition prompt 的提取粒度/类别 |
| attention_map 完整但 constraints 没覆盖 | Agent 2 | 改进约束转化策略 |
| constraints 完整但 answer 没遵守 | Agent 3 | 改进 answer 对 constraints 的利用方式 |
| constraints 不足，adversarial answer 逃逸了 | Agent 4a 暴露 | 收紧 constraints 或改进 Agent 2 |
| constraints 不足，critique 也没发现 | Agent 4b | 改进 critique 策略 |
| constraints 不足，但 critique 成功补上 | **系统正常** | **涌现能力的证据** |
| 注意力地图标注了高风险但 answer 仍犯错 | Agent 3 | 改进 prompt 对风险标注的强调方式 |
| 回译校验发现 coverage gap | Agent 2 | 自动反馈给 Agent 2 或 repair 阶段 |

### 6.5 进化参数

| 参数 | 建议值 | 原因 |
|------|--------|------|
| `tasks_per_eval` | 20+ | 降低评估方差 |
| `calibration_tasks` | 6 | 固定锚点防止跨代不可比 |
| `task_overlap_ratio` | 0.5 | 部分重叠提供连续性 |
| `worker_temperature` | 0 | 降低 LLM 非确定性 |
| `elite_count` | 2 | 保留多样性同时聚焦 |
| `selection_score` | accuracy 优先分层排序 | 防止惩罚项反转排名 |

---

## 七、有机能力生长（阶段二愿景）

### 7.1 概念

当进化积累足够多代数据后，系统应该能从"进化 prompt 文本"升级为"生长和修剪能力树"：

- **生长（萃取）**：从多代最佳 protocol 中提取反复出现的模式，固化为可复用的技能模块
- **修剪（压缩）**：标记多代未被选中的 mutation 方向为低价值搜索区域
- **打磨（精炼）**：识别跨代稳定保留的 pattern，固化为 seed 的核心部分

### 7.2 技能库数据结构（概念性）

```python
SKILL_LIBRARY = {
    "legal_decomposition": {
        "trigger": "task involves legal analysis",
        "pattern": "add BURDEN_OF_PROOF, STATUTORY_FRAMEWORK to attention_map",
        "origin": "Gen 4, protocol sha=xxx",
        "success_rate": 0.82,
        "times_used": 14,
        "status": "crystallized",  # 不再被 mutation 修改
    },
    "temporal_anti_hallucination": {
        "trigger": "dataset lacks explicit temporal markers",
        "pattern": "add date-assumption prohibition to TEMPTING_ASSUMPTIONS",
        "origin": "Gen 2, protocol sha=yyy",
        "success_rate": 0.91,
        "times_used": 23,
        "status": "core",  # 已合并进 seed
    },
    "chess_move_analysis": {
        "trigger": "task involves game/strategy analysis",
        "pattern": "require explicit pros/cons for each option",
        "origin": "Gen 5, protocol sha=zzz",
        "success_rate": 0.45,
        "times_used": 3,
        "status": "dormant",  # 使用太少，休眠
    },
}
```

### 7.3 为什么阶段二要等阶段一

Bootstrapping 问题：需要先有一个能跑的系统产出进化数据，才能从中萃取能力。固定流水线是能力树的"种子"——先让种子发芽（阶段一），再让它自然生长（阶段二）。

---

## 八、Ablation 实验方案

### 8.1 架构 ablation

| 配置 | 描述 | 验证假设 |
|------|------|---------|
| Full CFP | 完整架构 | baseline |
| No attention map | Agent 2 直接从 raw context 生成 constraints | 注意力地图的价值 |
| No critique | 关闭 Agent 4 | 多视角校正的增量贡献 |
| No adversarial | 关闭 Agent 4a，保留 Agent 4b | 对抗性视角的价值 |
| No constraints-to-answer | Agent 3 不看 constraints | 主动遵守 vs 被动修复 |
| Full context to all | 所有 agent 收到 full context（不裁剪） | 注意力裁剪 vs 冗余传递 |
| Direct prompting | 无 agent，直接回答 | agent 系统 vs baseline |

### 8.2 涌现机制分离实验

（见第四章 4.4 节）

### 8.3 跨评测集实验

在至少 2 个不同评测集（如 CL-bench + SCROLLS）上运行 CFP，观察：
- 进化是否在不同评测集上产生不同的 decomposition schema
- 方法论三层原理在不同任务类型上是否都有效
- 通用 seed vs 专用 seed 的性能差异

---

## 九、涌现的量化指标

### 9.1 原有指标（沿用）

1. **`patch_effectiveness`**：Agent 4 生成的 patch tests 中实际检测到问题的比例
2. **`accuracy_with_critique - accuracy_without_critique`**：critique 的增量贡献
3. **`constraint_attribution_accuracy`**：decomposition 中的要求被测试覆盖的比例

### 9.2 新增指标

4. **`attention_map_quality`**：注意力地图与 rubric 关注点的重合度（事后对比）
5. **`adversarial_escape_rate`**：adversarial answer 骗过 constraints 的比例（越低越好）
6. **`focus_efficiency`**：accuracy / total_tokens_consumed —— 认知焦点的终极指标，单位 token 产出的准确率
7. **`schema_evolution_velocity`**：进化过程中 decomposition schema 的变化速度（快速收敛 → 找到了好的认知语言；持续漂移 → 还在搜索）
8. **`cross_benchmark_transfer`**：在一个评测集上进化的 schema，迁移到另一个评测集的 zero-shot 表现

---

## 十、实现优先级

### Phase 0：修复已知 bug（不依赖架构变化）

1. `textwrap.dedent` 改为 placeholder + 后插值
2. selection score 排名改为 accuracy 优先的分层排序
3. `elite_count > population_size` 加校验

### Phase 1：CFP 核心架构

4. **`BaseCFPProtocol`**：定义 evolved methods + fixed runtime + `extract_relevant_spans()`
5. **Seed protocol**：手写注意力地图 prompt + 三层约束 prompt + answer prompt + critique prompt
6. **注意力地图格式**：实现 `build_attention_map` 输出格式验证和解析
7. **Runtime 中的 relevant_spans 裁剪**：根据注意力地图从 context 中提取相关段落
8. **Pipeline depth 自适应**：根据 `PIPELINE_DEPTH` 字段决定是否短路

### Phase 2：视角多样性机制

9. **对偶视角 decomposition**：在 seed prompt 中加入 `TEMPTING_ASSUMPTIONS` 和 `WHAT_CONTEXT_DOES_NOT_SAY`
10. **对抗性双步 critique**：实现 Agent 4a + Agent 4b
11. **回译校验**：实现 constraints → NL summary → 与 attention_map 比对的 coverage gap 检测

### Phase 3：进化接入

12. **修改 `MetaArchitect`**：增加 CFP mode prompt template + per-agent failure attribution
13. **高密度进化信号**：实现 `evaluation_signal` 完整指标体系
14. **Ablation**：先确认架构本身有效，再接入 evolution loop
15. **跑进化**：复用现有 GA 框架，改 protocol loader 和 evaluation 部分

### Phase 4：通用化和能力生长（待 Phase 3 积累数据后）

16. **跨评测集实验**：在第二个评测集上运行 CFP
17. **技能萃取**：从多代进化数据中提取可复用 pattern
18. **技能库**：建立 SKILL_LIBRARY，让 Meta-Architect 组合已有技能

---

## 十一、风险与缓解

| 风险 | 严重度 | 缓解方案 |
|------|--------|---------|
| 注意力地图质量差导致下游全部偏移 | 高 | Agent 4b 直接访问 raw context 做独立审查；回译校验提供自动 coverage gap 信号 |
| relevant_spans 裁剪过激丢失关键信息 | 高 | 初期保守裁剪（保留 70%+ context）；Agent 4b 始终可访问 full context 作为安全网 |
| 5 个 LLM 调用 token 成本过高 | 中 | Pipeline depth 自适应（简单任务跳过 Agent 4）；注意力裁剪减少下游输入长度 |
| Adversarial 构造步骤太难 | 中 | 如果失败率 > 80%，简化为"列举 constraints 漏洞"而非完整构造 adversarial answer |
| 搜索空间增大导致进化收敛慢 | 中 | 单一进化目标消解耦合；高密度信号减少所需轮数 |
| LLM 非确定性使注意力地图不稳定 | 高 | temperature=0 + prompt 强调确定性结构化输出 |
| 方法论对短上下文无意义 | 低 | PIPELINE_DEPTH = "simple" 自动短路 |

---

## 十二、与 CMAP 初版的完整对照

| 维度 | CMAP 初版 | CFP 新版 |
|------|----------|---------|
| 核心假设 | 涌现（agent 协作超越个体） | 认知焦点（注意力分配决定表现） |
| Agent 1 定位 | 信息提取器 | **注意力管理器** |
| Agent 1 输出 | 平坦事实列表 | 带风险标注的注意力地图 |
| 下游 agent 输入 | 每个 agent 都读 full context | 仅 Agent 1 读全文，下游读裁剪后的 relevant_spans |
| 视角多样性 | 意外获得（不同输出格式） | **刻意最大化**（对偶视角、对抗性双步、跨表征回译） |
| Critique 机制 | 单步审查 | **攻击-防御双步** |
| 流水线深度 | 固定 4 agent | **自适应**（Agent 1 决定深度） |
| 进化目标 | 4 个 prompt | **Decomposition schema**（单一目标，消解耦合） |
| 进化信号 | accuracy（粗） | **多维度细粒度信号** |
| 任务通用性 | 绑定 CL-bench | **任务无关**（不同 risk_type 适配不同评测集） |
| 远期愿景 | 无 | **有机能力树**（萃取、修剪、打磨） |

---

## 十三、一句话总结

**长上下文任务的瓶颈是注意力分配。CFP 通过将注意力分配显式化为可进化的 code 产物，让 multi-agent 系统在有限的认知资源下把焦点放在最关键的地方——而进化过程本身，就是在搜索"什么样的认知语言能让注意力分配最有效"。**
