# 接口 / 函数说明文档（基于当前代码）

> 代码基线：`/Users/liuxuan/Desktop/context_learning`
> 覆盖范围：`infer.py`、`eval.py`、`run_baselines.py`、`run_evolution.py`、`core/`、`benchmarks/`、`baselines/`

## 1. 核心数据结构

### 1.1 `benchmarks.base.TaskRecord`
- **用途**：统一任务记录结构，贯穿数据加载、协议执行、评测与失败分析。
- **关键字段**：
  - 输入侧：`task_id`、`context`、`query`、`messages_raw`、`rubrics`、`metadata`
  - 执行侧：`model_output`、`reasoning_trace`、`verification_passed`
  - token侧：`tokens_used`、`prompt_tokens`、`completion_tokens`
  - 评测侧：`score`、`eval_detail`、`failure_mode`

### 1.2 `core.base_protocol.ProtocolResult`
- **用途**：Legacy 协议（`BaseProtocol`）统一返回。
- **关键字段**：`answer`、`confidence`、`reasoning_trace`、`verification_passed`、`tokens_used`、`prompt_tokens`、`completion_tokens`、`metadata`

### 1.3 `core.base_sandbox_protocol.SandboxResult`
- **用途**：CaS / TDG 统一返回（扩展自 `ProtocolResult`）。
- **新增字段**：`sandbox_code`、`solver_code`、`execution_output`、`execution_success`、`compilation_success`

### 1.4 其他结构体
- `core.sandbox_executor.ExecutionResult`：沙箱执行结果（`success/output/error/namespace`）
- `core.protocol_loader.ValidationError`：代码校验阶段错误封装（`stage/message/fixable`）
- `core.meta_architect.ParentPerformance`：父协议性能摘要（F1-F4 + 编译/执行成功率）
- `core.evolution_loop.EvolutionConfig`：进化超参数集合（含 mutation 预算与适应度权重）
- `core.failure_classifier.FailureFeedback`：失败模式分析结果包
- `core.compiler_library.CompilerStrategy`：编译策略库记录单元

---

## 2. CLI 脚本接口

## 2.1 `infer.py`

### `main()`
- **功能**：批量读取 JSONL 任务，调用 OpenAI 兼容接口推理，增量写回输出 JSONL。
- **主要参数**：`--model --input --output --env-file --base-url --api-key --workers --max-samples --retry-delay --api-timeout --max-tokens`
- **关键行为**：
  1. `load_env_file` + `first_env/env_float/env_int` 解析配置
  2. 断点续跑：按输出文件 `idx` 跳过已完成样本
  3. 串行或线程池并发执行 `process_single_case`
  4. 每个成功样本用 `append_jsonl` 立即落盘

### 其他函数
- `call_openai_api(client, messages, model, ...) -> (response_text, error)`：带重试 API 调用
- `process_single_case(args) -> (idx, result, error)`：单样本处理
- `load_jsonl/save_jsonl/append_jsonl`：JSONL I/O
- `get_timestamp/log`：日志工具

## 2.2 `eval.py`

### `main()`
- **功能**：对模型输出执行 LLM-as-judge 二值评分，并生成汇总统计。
- **主要参数**：`--input --output --judge-model --env-file --base-url --api-key --workers --max-retries --api-timeout --max-tokens`
- **关键行为**：
  1. 读取待评测输出并按 `idx` 断点续跑
  2. 调用 `process_single_item`（串行/并发）
  3. 样本级结果增量写入输出
  4. `calculate_statistics` 汇总 solving rate

### 其他函数
- `build_rubrics_text(rubrics)`：rubric 列表格式化
- `call_judge_api(...)`：构造严格评分 prompt，返回 judge 原始 JSON 字符串
- `process_single_item(args)`：处理空输出、API失败、JSON解析失败
- `calculate_statistics(output_path)`：统计 `score_0/score_1/solving_rate`

## 2.3 `run_baselines.py`

### `main()`
- **功能**：运行基线协议并输出评测结果。
- **协议注册**：`naive / cot / react / cas_seed / cas_naive / cas_pydantic`
- **关键调用链**：
  1. 构造 `worker_client`、`judge_client`
  2. 从注册表实例化协议类
  3. `run_protocol_on_benchmark(...)`
  4. `print_metrics(...)`

## 2.4 `run_evolution.py`

### `main()`
- **功能**：Evo-Protocol 主入口（`legacy / cas / tdg`）。
- **关键能力**：
  - 读取 YAML/JSON 配置 + CLI 覆盖（`resolve_cli_or_config`）
  - 支持失败分类器、attention drift、selection 温度/惩罚、sandbox/protocol timeout 等参数
  - 构造 Loader（`ProtocolLoader` / `SandboxProtocolLoader` / `TDGProtocolLoader`）
  - 构造 `MetaArchitect`、`EvolutionEngine` 并执行 `engine.run(initial_code)`
  - 可选 best protocol final eval

### 辅助函数
- `load_initial_code(path, mode)`：加载初始协议代码（文件或默认 seed）
- `load_experiment_config(path, allow_missing)`：加载实验配置（JSON/YAML）

---

## 3. Benchmark 接口

## 3.1 `benchmarks.base.BaseBenchmark`（抽象）
- `load_tasks(data_path, split="all") -> list[TaskRecord]`
- `evaluate(record, judge_client=None) -> TaskRecord`
- `get_metrics(records) -> dict[str, float]`（默认 overall + 分类别 accuracy）

## 3.2 注册机制
- `register_benchmark(name)`：装饰器注册
- `get_benchmark(name, **kwargs)`：按名实例化

## 3.3 `benchmarks.cl_bench.CLBenchmark`
- `load_tasks`：读取 CL-bench JSONL，提取 `context/query/messages/rubrics/metadata`
- `evaluate`：复用 `eval.py` rubric 评测
- `_apply_split`：按 `context_category` 确定性分层切分

---

## 4. 协议抽象接口

## 4.1 `core.base_protocol.BaseProtocol`（Legacy）

### 必须实现
- `perception(context) -> dict`
- `cognition(query, perceived_info) -> str`
- `verification(answer, context) -> bool`

### 已实现
- `run(context, query, max_retries=2) -> ProtocolResult`
  - 固定流程：Perception -> Cognition -> Verification
  - 验证失败时注入 `retry_feedback` 重试
  - 最终失败输出空 answer 并标记 `output_blocked`
- `_call_llm(messages, temperature=0.0)`
  - API 调用 + 调用预算保护 + `prompt/completion/total` token 计数

## 4.2 `core.base_sandbox_protocol.BaseCaSCompiler`（CaS）

### 必须实现（可进化）
- `compile_sandbox(context) -> str`
- `generate_solver(query, sandbox_schema) -> str`

### 已实现（不可变运行时）
- `run(context, query, max_retries=2) -> SandboxResult`
  - 编译环境代码 -> 执行校验 -> 生成求解代码 -> 执行
  - 语法修复、失败反馈重试、oracle fallback
- `_make_oracle_fn()`：注入 `_oracle(prompt, return_type)`
- `_sanitize_generated_code()`：AST 清洗（去 `_oracle` 占位定义/赋值等）
- `_derive_dynamic_call_budget(...)`：长上下文动态扩容 LLM 调用预算

## 4.3 `core.base_tdg_protocol.BaseTDGCompiler`（TDG）

### 必须实现（可进化）
- `compile_tests(context, query) -> str`
- `generate_answer(context, query, messages_raw=None) -> str`

### 已实现（不可变运行时）
- `run(context, query, max_retries=2, messages_raw=None) -> SandboxResult`
  - 生成测试 -> 生成答案 -> 跑测试 -> repair loop
  - 测试不可用时降级为直接答案返回
- `_run_tests/_build_test_runner/_repair_answer`：测试执行与答案修复
- `_sanitize_generated_code()`：测试代码清洗（会删除调用危险 builtin 的 `test_*`）

---

## 5. 加载 / 校验 / 超时执行接口

## 5.1 `core.protocol_loader.ProtocolLoader`
- `validate(code)`：`syntax -> lint(ruff) -> security(AST)`
- `load_from_code(code)`：校验并动态加载 `BaseProtocol` 子类
- `run_with_timeout(protocol, context, query)`：优先子进程硬超时，失败时回退线程超时
- 进程间序列化：保留 `tokens_used + prompt_tokens + completion_tokens`

## 5.2 `SandboxProtocolLoader`
- 面向 `BaseCaSCompiler`
- 额外 contract 校验：
  - 必须实现 `compile_sandbox/generate_solver`
  - 禁止覆写 `run/_call_llm/_make_oracle_fn`
- 进程间序列化同样保留 prompt/completion token 字段

## 5.3 `TDGProtocolLoader`
- 面向 `BaseTDGCompiler`
- 额外 contract 校验：
  - 必须实现 `compile_tests/generate_answer`
  - 两个方法都必须调用 `self._call_llm`
  - 禁止调用未定义 `self.xxx()` 辅助方法
- `_ALLOWED_SELF_ATTRS` 已包含 `_task_prompt_tokens/_task_completion_tokens` 等运行时字段

---

## 6. 执行与评估接口

## 6.1 `core.evaluator.run_protocol_on_benchmark(...)`
- **用途**：通用执行器（Legacy/CaS/TDG 三态统一）
- **关键能力**：
  - 自动构建上下文载荷（普通/role-preserving CaS/role-preserving TDG）
  - 自动选择 loader
  - 支持多线程并发（协议对象克隆）
  - 执行结果写回 `tokens_used/prompt_tokens/completion_tokens`
  - 可选写出 JSONL 结果

## 6.2 `core.evaluator.print_metrics(records, benchmark_name, ...)`
- 调 benchmark 的 `get_metrics` 并打印

## 6.3 `core.sandbox_executor`
- `validate_sandbox_code(code)`：AST 安全校验
- `execute_sandbox_code(code, oracle_fn=None, timeout=30)`：核心沙箱执行
- `execute_compilation_code / execute_query_code`：兼容接口

---

## 7. 进化与自修复接口

## 7.1 `core.meta_architect.MetaArchitect`
- `build_prompt(...)`：按 mode 组装 mutation prompt
- `mutate(...)`：调用 `generate_with_repair` 生成候选协议
- 新增温度透传参数：`architect_temperature`、`repair_temperature`

## 7.2 `core.self_repair.generate_with_repair(...)`
- 多候选代码提取（fenced/json/转义字符串）
- 失败后按 `ValidationError` 构造 repair prompt 迭代修复
- TDG 含局部 contract 自动降级修复逻辑
- 底层 `_call_architect(...)` 支持：
  - `chat` / `responses` / `auto` 模式切换（含 Codex 模型优先 Responses）
  - token 上限自适应回退
  - request nonce 防重复失败

## 7.3 `core.evolution_loop.EvolutionEngine`

### 主接口
- `run(initial_code) -> dict`
  - 种群评估（同 SHA 候选只评估一次，结果复用）
  - 精英保留
  - 父代选择（`archive.select(tau, alpha)`）
  - 并发 mutation + 去重 + 预算控制
  - adaptive architect temperature（重复率高时升温，接受新颖子代后降温）
  - 落盘运行日志

### 关键内部能力
- `_evaluate_protocol`：逐任务执行 + 评分 + failure trace
- `_compute_selection_score`：accuracy-first 排序打分
- `_summarize_failures`：代级失败统计摘要（含 attention drift / token 效率 / TDG 指标）
- `_measure_attention_drift`：上下文漂移估计（judge 或启发式）
- `_append_jsonl`：带锁写日志，避免并发写入冲突

---

## 8. 失败分析与工具接口

## 8.1 `core.failure_classifier`
- `build_failure_feedback(record, llm_client=None, ...) -> dict`
  - 推断失败模式 F1/F2/F3/F4
  - 可启用 LLM 分类器补充 `root_cause/repair_actions`
  - 支持 CaS/TDG 上下文增强字段（stage、test_pass_rate 等）
- `classify_failure_mode(...)`：向后兼容模式分类接口

## 8.2 `core.archive.ProtocolArchive`
- 协议代码与元数据归档
- 评分更新（含 evaluation_history）
- 软选择采样（softmax + 访问惩罚）

## 8.3 `core.compiler_library.CompilerLibrary`
- 持久化管理编译策略（按 domain 检索最优策略）

## 8.4 `core.env_utils`
- `load_env_file / first_env / env_float / env_int`

## 8.5 `core.token_tracker`
- `TokenTracker.record/summary/reset` + 全局 `TRACKER`

---

## 9. Baseline 协议接口

### Legacy 基线
- `NaiveProtocol`：单轮直接回答
- `CoTProtocol`：推理式回答 + 轻验证
- `ReActProtocol`：证据抽取 -> 生成回答 -> 验证

### CaS 基线
- `SeedCaSCompiler`：较完整上下文编译与 solver 生成
- `NaiveCaSCompiler`：扁平 `facts` 字典编译
- `PydanticCaSCompiler`：Pydantic 编译 + validator 约束

### TDG 基线
- `SeedTDGCompiler`：生成测试函数 + 生成答案（支持多轮 messages）

---

## 10. 输出文件格式（关键）

### 推理输出（`infer.py`）
```json
{"idx": 0, "messages": [...], "model_output": "...", "rubrics": [...]} 
```

### 评测输出（`eval.py`）
```json
{
  "idx": 0,
  "messages": [...],
  "model_output": "...",
  "rubrics": [...],
  "grading_rationale": "...",
  "requirement_status": ["yes", "no"],
  "score": 0
}
```

### 基线/进化评估输出（`core.evaluator`）
```json
{
  "idx": 0,
  "messages": [...],
  "model_output": "...",
  "rubrics": [...],
  "score": 1,
  "metadata": {
    "prompt_tokens": 123,
    "completion_tokens": 45
  },
  "failure_mode": "F2"
}
```
