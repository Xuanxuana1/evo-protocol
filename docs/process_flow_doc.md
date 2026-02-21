# 流程文档（基于当前代码实现）

> 目标：描述该仓库从推理、评测到进化搜索的端到端执行流程与关键分支。

## 1. 总体流程

```text
数据集(JSONL)
   -> 推理 (infer.py)
   -> 模型输出(JSONL)
   -> 评测 (eval.py / benchmark.evaluate)
   -> 指标(score/solving_rate)

进化模式（run_evolution.py）
   -> 初始协议代码
   -> EvolutionEngine: 评估 -> 选择 -> 变异 -> 修复 -> 迭代
   -> 最佳协议
   -> 可选 final eval
```

---

## 2. `infer.py` 推理流程

## 2.1 主路径
1. 解析 CLI 参数，加载 `.env`。
2. 解析模型名、timeout、max_tokens、API key/base_url。
3. 读取输入 JSONL。
4. 若输出文件已存在：按 `idx` 识别已完成样本，断点续跑。
5. 构造 pending task 列表。
6. 单线程或线程池执行 `process_single_case`。
7. 每个成功结果立即 `append_jsonl`。
8. 输出成功/失败统计。

## 2.2 单样本处理
1. 读取 `messages`。
2. 调用 `call_openai_api`（带重试）。
3. 成功返回 `idx/messages/model_output/rubrics`。
4. 失败返回 `error`。

## 2.3 失败分支
- API 异常 -> 重试 -> 最终失败。
- 样本缺少 `messages` -> 直接失败。

---

## 3. `eval.py` 评测流程

## 3.1 主路径
1. 解析 CLI 参数并加载 `.env`。
2. 读取待评测 JSONL（通常来自 `infer.py`）。
3. 若输出文件存在：按 `idx` 断点续跑。
4. 串行/并发执行 `process_single_item`。
5. 结果增量写入输出 JSONL。
6. `calculate_statistics` 统计 solving rate。

## 3.2 单样本评分
1. 空输出直接记 `score=0`。
2. `build_rubrics_text` 格式化 rubric。
3. `call_judge_api` 调 judge 模型（严格二值评分指令）。
4. 解析 judge JSON：
   - 成功 -> 写入 `grading_rationale/requirement_status/score`
   - 失败 -> 触发重评；最终失败记 `score=0`

## 3.3 关键设计点
- all-or-nothing 二值制。
- 支持中断后续跑。
- JSON 解析失败不会导致整体崩溃。

---

## 4. `run_baselines.py` 流程

1. 读取参数和环境变量，创建 `worker_client/judge_client`。
2. 从 `BASELINE_REGISTRY` 实例化协议。
3. 调 `run_protocol_on_benchmark`：加载任务 -> 执行协议 -> benchmark 评分。
4. `print_metrics` 输出指标。

---

## 5. `run_evolution.py` 流程

## 5.1 启动阶段
1. 解析 CLI + 可选配置文件（YAML/JSON）。
2. 通过 `resolve_cli_or_config` 合并配置（CLI 显式参数优先）。
3. 创建 OpenAI 客户端（worker/architect/judge）。
4. 按 mode 选择 loader：
   - `legacy` -> `ProtocolLoader`
   - `cas` -> `SandboxProtocolLoader`
   - `tdg` -> `TDGProtocolLoader`
5. 创建 `MetaArchitect`、`ProtocolArchive`、`EvolutionConfig`、`EvolutionEngine`。

## 5.2 运行阶段
1. 加载 `initial_code`，进入 `engine.run(initial_code)`。
2. 每代执行：
   - 候选评估
   - 精英保留
   - 选择父代并变异
   - 校验/修复/去重
   - 填充新种群
3. 输出 `best_sha / best_fitness / best_accuracy`。

## 5.3 收尾阶段
1. 可选对最佳协议做 final eval。
2. 输出 summary JSON。
3. 打印关键指标与日志路径。

---

## 6. `EvolutionEngine.run` 代内流程

## 6.1 候选评估（两阶段）
1. **并行阶段**：同一代中按 `eval_key`（优先 SHA）只评估一次。
2. **串行阶段**：按种群顺序回填候选结果，复用同 SHA 评估缓存并写日志。
3. 每个候选更新：`fitness/mode_accuracy/failures/eval_summary/selection_score`。
4. 同步写入 archive 元数据和 candidate trace。

## 6.2 排序与保留
1. 按 `_candidate_rank_key` 排序（accuracy-first）。
2. 选 elite。
3. 更新 `best_overall`。
4. 记录代级统计（均值、唯一 SHA 比例、代码多样性）。

## 6.3 变异生成
1. `archive.select(tau, alpha)` 抽父代。
2. `meta_architect.mutate(...)` 并发生成子代。
3. 校验通过后归档；重复 SHA 丢弃。
4. 触发拒绝 streak / 时间预算 / 尝试预算上限时提前停止。
5. 不足种群用 elite 克隆补齐。

## 6.4 Adaptive 架构温度（新）
- 当重复子代比例高时，提高 architect temperature（探索更多变异）。
- 接受新颖子代后，按 cooldown 回落温度。
- 相关参数来自环境变量（如 `EVO_ARCH_TEMP_STEP`、`EVO_ARCH_TEMP_MAX`、`EVO_ARCH_TEMP_DUP_RATE_TRIGGER`）。

---

## 7. `_evaluate_protocol` 任务级流程

1. 复制 `TaskRecord`（避免污染原始样本）。
2. 构造模式相关输入：
   - legacy: `_build_effective_context`
   - cas: `_build_cas_inputs`
   - tdg: `_build_tdg_inputs`
3. `loader.run_with_timeout(...)` 执行协议。
4. 写回输出与 token 统计：`tokens_used/prompt_tokens/completion_tokens`。
5. `benchmark.evaluate` 得到 `score`。
6. 可选 attention drift 评估（支持按任务独立 RNG 采样率控制）。
7. `score < 1` 时：调用 `build_failure_feedback`，记录 failure sample。
8. 聚合返回：accuracy、编译/执行成功率、TDG test 指标、false positive、sanitized test drop 等。

---

## 8. CaS 运行时流程（`BaseCaSCompiler.run`）

```text
compile_sandbox(context)
  -> sanitize code
  -> execute env_code (compile check)
     -> fail: syntax-repair once -> still fail => blocked
  -> retry loop:
       generate_solver(query + feedback)
       -> sanitize
       -> execute env + solver
       -> success and FINAL_ANSWER => return
       -> else feedback for next round
  -> oracle fallback direct answer
  -> still fail => blocked
```

关键点：
- 可进化方法只有 `compile_sandbox/generate_solver`。
- `_oracle(prompt, return_type)` 为受控神经感知接口。
- 记录 `prompt_tokens/completion_tokens` 并纳入结果。

---

## 9. TDG 运行时流程（`BaseTDGCompiler.run`）

```text
compile_tests(context, query)
  -> sanitize + compile check (+ optional syntax repair)
generate_answer(context, query, messages_raw)
  -> empty => oracle fallback
if tests unavailable:
  -> return answer (degrade path)
else:
  run tests
  retry repair_answer on failed tests
  return answer + test metrics
```

关键点：
- 最坏情况降级为直接回答，不会因测试生成失败整体失效。
- 运行时产出 `test_pass_rate/raw_test_count/sanitized_test_drop_count` 等指标。

---

## 10. Loader 校验与执行流程

## 10.1 校验链
- `syntax` -> `lint(ruff)` -> `security(AST)` -> `contract(mode-specific)`

## 10.2 安全策略
- 禁止高风险导入（`os/subprocess/socket/...`）
- 禁止危险 builtin（`eval/exec/open/...`）
- 禁止 `while`（避免失控循环）

## 10.3 超时执行
- 优先子进程硬超时（`fork + terminate/kill`）
- 回退线程超时
- 结果跨进程序列化会保留 `prompt_tokens/completion_tokens`

---

## 11. 自修复与架构调用流程（`core.self_repair`）

1. `_call_architect` 发起初次生成或修复请求。
2. 根据模型/配置在 `chat` 与 `responses` API 间选择（`ARCHITECT_API_MODE=auto/chat/responses`）。
3. 遇到 token 上限错误时自动下调 token limit 重试。
4. 为每次请求注入 nonce，降低重复请求失败概率。
5. `generate_with_repair` 在校验失败时构造 repair prompt 多轮修复。
6. TDG 场景额外有本地 contract 降级修复路径。

---

## 12. 失败分析流程（`build_failure_feedback`）

1. 从 record 提取 query/answer/rubrics/eval_detail/metadata。
2. 先走启发式 `_infer_default_mode` + `_heuristic_feedback`。
3. 若提供 LLM classifier，再调用模型细化 `mode/root_cause/repair_actions`。
4. 输出结构：`mode/confidence/root_cause/repair_actions/unsatisfied_rubrics/stage/source`。

模式语义：
- `F1` 参数知识覆盖上下文（Parametric Override）
- `F2` 上下文检索/导航失败（Context Navigation Failure）
- `F3` 推理链断裂（Reasoning Breakdown）
- `F4` 归纳失败（Induction Failure）

---

## 13. 日志与产物流

- `archive/protocol_*.py`：协议代码归档
- `archive/protocol_*_meta.json`：协议元数据
- `archive/_logs/generation_trace.jsonl`：代级事件
- `archive/_logs/candidate_trace.jsonl`：候选评估轨迹
- `archive/_logs/failure_trace.jsonl`：失败样本
- `archive/_logs/task_trace.jsonl`：任务级时序
- `outputs/*.jsonl`：推理/评测/基线输出
- `outputs/evolution_result.json`：进化总结

> 注：`EvolutionEngine._append_jsonl` 已使用互斥锁，保证并发日志写入安全。
