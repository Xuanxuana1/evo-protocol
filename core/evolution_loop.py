"""Main Evo-Protocol optimization loop (evaluate -> select -> mutate)."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from statistics import mean
from pathlib import Path
from typing import Any, Union

from benchmarks.base import TaskRecord, get_benchmark
from core.archive import ProtocolArchive
from core.base_protocol import BaseProtocol
from core.base_sandbox_protocol import BaseCaSCompiler, SandboxResult
from core.base_tdg_protocol import BaseTDGCompiler
from core.env_utils import env_float, env_int
from core.failure_classifier import build_failure_feedback
from core.meta_architect import MetaArchitect, ParentPerformance
from core.protocol_loader import ProtocolLoader, SandboxProtocolLoader, TDGProtocolLoader


@dataclass
class EvolutionConfig:
    """Config for evolutionary search."""

    generations: int = 30
    population_size: int = 8
    elite_count: int = 3
    tasks_per_evaluation: int = 50
    failure_samples_per_mutation: int = 5
    enable_failure_classifier: bool = True
    failure_classifier_model: str = "gpt-4o"
    enable_attention_drift: bool = True
    attention_drift_model: str = "gpt-4o-mini"
    attention_drift_sample_rate: float = 1.0
    max_repair_attempts: int = 2
    seed: int = 42
    mode: str = "cas"
    init_population_size: int = 1
    sandbox_timeout_seconds: int = 30
    selection_tau: float = 0.5
    selection_alpha: float = 0.5
    fitness_weights: dict[str, float] = field(default_factory=lambda: {
        "answer_correctness": 0.8,
        "execution_success": 0.1,
        "compilation_success": 0.1,
    })


class EvolutionEngine:
    """Population-based protocol evolution engine."""

    def __init__(
        self,
        benchmark_name: str,
        data_path: str,
        protocol_loader: Union[ProtocolLoader, SandboxProtocolLoader, TDGProtocolLoader],
        meta_architect: MetaArchitect,
        archive: ProtocolArchive,
        config: EvolutionConfig | None = None,
        split: str = "train",
        benchmark_kwargs: dict[str, Any] | None = None,
        judge_client=None,
        failure_classifier_client=None,
    ) -> None:
        self.benchmark = get_benchmark(benchmark_name, **(benchmark_kwargs or {}))
        self.tasks = self.benchmark.load_tasks(data_path, split=split)
        self.loader = protocol_loader
        self.meta_architect = meta_architect
        self.archive = archive
        self.config = config or EvolutionConfig()
        self.judge_client = judge_client
        self.failure_classifier_client = failure_classifier_client or judge_client
        self.config.attention_drift_sample_rate = max(
            0.0,
            min(1.0, float(self.config.attention_drift_sample_rate)),
        )
        self.config.selection_tau = max(1e-6, float(self.config.selection_tau))
        self.config.selection_alpha = max(0.0, float(self.config.selection_alpha))
        self.rng = random.Random(self.config.seed)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = self.archive.dir / "_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.generation_trace_path = self.logs_dir / "generation_trace.jsonl"
        self.candidate_trace_path = self.logs_dir / "candidate_trace.jsonl"
        self.failure_trace_path = self.logs_dir / "failure_trace.jsonl"
        self.task_trace_path = self.logs_dir / "task_trace.jsonl"

    @property
    def _is_cas(self) -> bool:
        return self.config.mode == "cas"

    @property
    def _is_tdg(self) -> bool:
        return self.config.mode == "tdg"

    def run(self, initial_code: str) -> dict[str, Any]:
        """Run full evolution from an initial protocol source string."""

        protocol, error = self.loader.load_from_code(initial_code)
        if protocol is None:
            raise ValueError(f"Initial protocol failed validation: {error.message if error else 'unknown'}")

        init_population_size = int(getattr(self.config, "init_population_size", 0) or 0)
        if init_population_size <= 0:
            init_population_size = self.config.population_size
        init_population_size = max(1, min(init_population_size, self.config.population_size))

        init_sha = self.archive.save(initial_code, generation=0, score=None)
        base_candidate = {
            "sha": init_sha,
            "code": initial_code,
            "protocol": protocol,
            "fitness": 0.0,
            "mode_accuracy": {},
            "failures": [],
        }
        population = [dict(base_candidate)]
        for _ in range(init_population_size - 1):
            cloned_protocol, _ = self.loader.load_from_code(initial_code)
            population.append(
                {
                    "sha": init_sha,
                    "code": initial_code,
                    "protocol": cloned_protocol if cloned_protocol is not None else protocol,
                    "fitness": 0.0,
                    "mode_accuracy": {},
                    "failures": [],
                }
            )

        history: list[dict[str, Any]] = []
        self._append_jsonl(
            self.generation_trace_path,
            {
                "run_id": self.run_id,
                "event": "run_start",
                "time": self._now_iso(),
                "generations": self.config.generations,
                "population_size": self.config.population_size,
                "init_population_size": init_population_size,
                "elite_count": self.config.elite_count,
                "tasks_per_evaluation": self.config.tasks_per_evaluation,
                "benchmark": getattr(self.benchmark, "name", "unknown"),
                "mode": self.config.mode,
                "selection_tau": self.config.selection_tau,
                "selection_alpha": self.config.selection_alpha,
            },
        )

        for generation in range(1, self.config.generations + 1):
            print(f"\n[Gen {generation}] evaluating {len(population)} protocols (mode={self.config.mode})...")
            sampled_tasks = self._sample_tasks(self.config.tasks_per_evaluation)

            for idx, candidate in enumerate(population, start=1):
                fitness, mode_accuracy, failures, eval_summary = self._evaluate_protocol(
                    protocol=candidate["protocol"],
                    generation=generation,
                    candidate_index=idx,
                    candidate_total=len(population),
                    candidate_sha=candidate["sha"],
                    sampled_tasks=sampled_tasks,
                )
                candidate["fitness"] = fitness
                candidate["mode_accuracy"] = mode_accuracy
                candidate["failures"] = failures
                candidate["eval_summary"] = eval_summary
                candidate["answer_accuracy"] = float(eval_summary.get("accuracy", 0.0))
                candidate["selection_score"] = self._compute_selection_score(
                    fitness=fitness,
                    eval_summary=eval_summary,
                )
                self.archive.update_evaluation(
                    sha=candidate["sha"],
                    generation=generation,
                    score=float(candidate["selection_score"]),
                    failure_summary=eval_summary,
                )
                candidate_log = {
                    "run_id": self.run_id,
                    "generation": generation,
                    "candidate_index": idx,
                    "population_size": len(population),
                    "sha": candidate["sha"],
                    "fitness": fitness,
                    "accuracy": candidate["answer_accuracy"],
                    "selection_score": candidate["selection_score"],
                    "num_failures": len(failures),
                    "failure_mode_counts": eval_summary.get("failure_mode_counts", {}),
                    "avg_tokens_per_task": eval_summary.get("avg_tokens_per_task", 0.0),
                    "attention_drift_mean": eval_summary.get("attention_drift_mean"),
                    "attention_drift_high_rate": eval_summary.get("attention_drift_high_rate"),
                    "top_root_causes": eval_summary.get("top_root_causes", [])[:3],
                    "time": self._now_iso(),
                }
                if self._is_cas or self._is_tdg:
                    candidate_log["compilation_success_rate"] = eval_summary.get("compilation_success_rate", 0.0)
                    candidate_log["execution_success_rate"] = eval_summary.get("execution_success_rate", 0.0)
                if self._is_tdg:
                    candidate_log["avg_test_pass_rate"] = eval_summary.get("avg_test_pass_rate", 0.0)

                self._append_jsonl(self.candidate_trace_path, candidate_log)
                print(
                    f"[Gen {generation}] candidate {idx}/{len(population)} "
                    f"sha={candidate['sha']} fitness={fitness:.4f} "
                    f"accuracy={candidate['answer_accuracy']:.4f} "
                    f"failures={len(failures)}"
                )
                self._append_failure_samples(generation, candidate["sha"], failures)

            ranked = sorted(population, key=self._candidate_rank_key, reverse=True)
            elites = ranked[: max(1, self.config.elite_count)]
            best = elites[0]
            diversity = self._compute_population_diversity(ranked)
            gen_record = {
                "generation": generation,
                "best_sha": best["sha"],
                "best_fitness": best["fitness"],
                "best_accuracy": float(best.get("answer_accuracy", 0.0)),
                "best_selection_score": float(best.get("selection_score", 0.0)),
                "mean_fitness": mean([item["fitness"] for item in ranked]),
                "unique_protocols": len({item["sha"] for item in ranked}),
                "code_diversity": diversity["code_diversity"],
                "unique_sha_ratio": diversity["unique_sha_ratio"],
                "best_failure_summary": best.get("eval_summary", {}),
            }
            history.append(gen_record)
            self._append_jsonl(
                self.generation_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "generation_end",
                    "time": self._now_iso(),
                    **gen_record,
                },
            )
            print(
                f"[Gen {generation}] best={gen_record['best_fitness']:.4f} "
                f"best_acc={gen_record['best_accuracy']:.4f} "
                f"mean={gen_record['mean_fitness']:.4f} "
                f"diversity={gen_record['code_diversity']:.4f}"
            )

            if generation == self.config.generations:
                break

            new_population = [dict(item) for item in elites]
            mutation_attempts = 0
            max_attempts = self.config.population_size * 6
            existing_child_shas = {str(item.get("sha", "")) for item in new_population if str(item.get("sha", ""))}
            elite_by_sha: dict[str, dict[str, Any]] = {}
            for item in elites:
                sha = str(item.get("sha", "")).strip()
                if sha and sha not in elite_by_sha:
                    elite_by_sha[sha] = item

            for _ in range(max_attempts):
                if len(new_population) >= self.config.population_size:
                    break
                mutation_attempts += 1
                selected_shas = self.archive.select(
                    k=1,
                    tau=self.config.selection_tau,
                    alpha=self.config.selection_alpha,
                    candidate_shas=list(elite_by_sha.keys()),
                )
                selected_sha = selected_shas[0] if selected_shas else ""
                parent = elite_by_sha.get(selected_sha) if selected_sha else None
                if parent is None:
                    parent = self.rng.choice(elites)
                parent_perf = self._build_parent_performance(
                    parent.get("mode_accuracy", {}),
                    float(parent.get("answer_accuracy", parent["fitness"])),
                    parent.get("eval_summary"),
                )
                failures = parent.get("failures", [])
                sampled_failures = self.rng.sample(
                    failures,
                    k=min(len(failures), self.config.failure_samples_per_mutation),
                ) if failures else []

                child_protocol, child_code_or_err = self.meta_architect.mutate(
                    loader=self.loader,
                    generation=generation,
                    mutation_attempt=mutation_attempts,
                    parent_code=parent["code"],
                    parent_performance=parent_perf,
                    failure_examples=sampled_failures,
                    max_repair_attempts=self.config.max_repair_attempts,
                )
                if child_protocol is None:
                    self._append_jsonl(
                        self.generation_trace_path,
                        {
                            "run_id": self.run_id,
                            "event": "mutation_rejected",
                            "time": self._now_iso(),
                            "generation": generation,
                            "parent_sha": parent["sha"],
                            "reason": str(child_code_or_err)[:320],
                        },
                    )
                    continue

                child_code = child_code_or_err
                child_sha = self.archive.save(
                    child_code,
                    generation=generation,
                    parent_sha=parent["sha"],
                )
                if child_sha in existing_child_shas:
                    self._append_jsonl(
                        self.generation_trace_path,
                        {
                            "run_id": self.run_id,
                            "event": "mutation_duplicate",
                            "time": self._now_iso(),
                            "generation": generation,
                            "parent_sha": parent["sha"],
                            "child_sha": child_sha,
                        },
                    )
                    continue
                existing_child_shas.add(child_sha)
                new_population.append(
                    {
                        "sha": child_sha,
                        "code": child_code,
                        "protocol": child_protocol,
                        "fitness": 0.0,
                        "selection_score": 0.0,
                        "answer_accuracy": 0.0,
                        "mode_accuracy": {},
                        "failures": [],
                    }
                )

            for _ in range(self.config.population_size - len(new_population)):
                new_population.append(dict(self.rng.choice(elites)))

            population = new_population

        best_final = max(population, key=self._candidate_rank_key)
        self._append_jsonl(
            self.generation_trace_path,
            {
                "run_id": self.run_id,
                "event": "run_end",
                "time": self._now_iso(),
                "best_sha": best_final["sha"],
                "best_fitness": best_final["fitness"],
                "best_accuracy": float(best_final.get("answer_accuracy", 0.0)),
                "best_selection_score": float(best_final.get("selection_score", 0.0)),
            },
        )
        return {
            "run_id": self.run_id,
            "best_sha": best_final["sha"],
            "best_fitness": best_final["fitness"],
            "best_accuracy": float(best_final.get("answer_accuracy", 0.0)),
            "best_selection_score": float(best_final.get("selection_score", 0.0)),
            "history": history,
            "mode": self.config.mode,
            "log_files": {
                "generation_trace": str(self.generation_trace_path),
                "candidate_trace": str(self.candidate_trace_path),
                "failure_trace": str(self.failure_trace_path),
                "task_trace": str(self.task_trace_path),
            },
        }

    def _evaluate_protocol(
        self,
        protocol,
        generation: int,
        candidate_index: int,
        candidate_total: int,
        candidate_sha: str,
        sampled_tasks: list[TaskRecord],
    ) -> tuple[float, dict[str, float], list[dict[str, Any]], dict[str, Any]]:
        """Run protocol on sampled train tasks and compute fitness/failure traces."""

        sampled = sampled_tasks
        scores: list[float] = []
        failure_mode_counter: Counter[str] = Counter()
        failures: list[dict[str, Any]] = []
        token_usage: list[int] = []
        attention_drift_values: list[float] = []
        compilation_success_count = 0
        execution_success_count = 0
        test_pass_rate_values: list[float] = []

        for task_idx, task in enumerate(sampled, start=1):
            record = TaskRecord(
                task_id=task.task_id,
                context=task.context,
                query=task.query,
                messages_raw=task.messages_raw,
                rubrics=task.rubrics,
                metadata=dict(task.metadata),
            )
            task_timer = perf_counter()
            task_info = {
                "generation": generation,
                "candidate_index": candidate_index,
                "candidate_total": candidate_total,
                "sha": candidate_sha,
                "task_index": task_idx,
                "task_total": len(sampled),
                "task_id": record.task_id,
                "context_category": str(record.metadata.get("context_category", "unknown")),
                "sub_category": str(record.metadata.get("sub_category", "unknown")),
            }

            print(
                f"[Gen {generation}] cand {candidate_index}/{candidate_total} "
                f"task {task_idx}/{len(sampled)} start id={record.task_id}"
            )
            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "task_start",
                    "time": self._now_iso(),
                    **task_info,
                },
            )

            if self._is_cas:
                effective_context, effective_query = self._build_cas_inputs(record)
            elif self._is_tdg:
                effective_context, effective_query = self._build_tdg_inputs(record)
            else:
                effective_context = self._build_effective_context(record)
                effective_query = record.query
            worker_timer = perf_counter()
            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "worker_start",
                    "time": self._now_iso(),
                    **task_info,
                },
            )
            result = self.loader.run_with_timeout(protocol, context=effective_context, query=effective_query) if not self._is_tdg else self.loader.run_with_timeout(protocol, context=effective_context, query=effective_query, messages_raw=record.messages_raw)
            worker_seconds = perf_counter() - worker_timer
            if result is None:
                record.model_output = ""
                record.reasoning_trace = ["execution failed or timed out"]
                record.verification_passed = False
                worker_status = "timeout_or_error"
                if self._is_cas or self._is_tdg:
                    record.metadata["compilation_success"] = False
                    record.metadata["execution_success"] = False
                    record.metadata["compilation_failed"] = False
                    record.metadata["execution_output"] = "execution failed or timed out"
                    record.metadata["execution_traceback"] = "execution failed or timed out"
                    if self._is_tdg:
                        record.metadata["test_pass_rate"] = 0.0
            else:
                record.model_output = result.answer
                record.reasoning_trace = list(result.reasoning_trace)
                record.verification_passed = bool(result.verification_passed)
                record.tokens_used = int(result.tokens_used)
                record.metadata.update(result.metadata)
                worker_status = "ok"

                # CaS/TDG-specific: capture sandbox result fields
                if (self._is_cas or self._is_tdg) and isinstance(result, SandboxResult):
                    if result.compilation_success:
                        compilation_success_count += 1
                    if result.execution_success:
                        execution_success_count += 1
                    record.metadata["compilation_success"] = bool(result.compilation_success)
                    record.metadata["execution_success"] = bool(result.execution_success)
                    record.metadata["compilation_failed"] = not bool(result.compilation_success)
                    record.metadata["sandbox_code"] = result.sandbox_code[:2000]
                    record.metadata["solver_code"] = result.solver_code[:2000]
                    execution_output = result.execution_output[:500]
                    record.metadata["execution_output"] = execution_output
                    if result.execution_success:
                        record.metadata.pop("execution_traceback", None)
                    else:
                        record.metadata["execution_traceback"] = execution_output
                    # TDG-specific: capture test_pass_rate
                    if self._is_tdg:
                        tpr = float(result.metadata.get("test_pass_rate", 0.0))
                        record.metadata["test_pass_rate"] = tpr
                        test_pass_rate_values.append(tpr)

            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "worker_end",
                    "time": self._now_iso(),
                    "worker_status": worker_status,
                    "worker_seconds": round(worker_seconds, 3),
                    "tokens_used": int(record.tokens_used),
                    **task_info,
                },
            )

            token_usage.append(int(record.tokens_used))

            judge_timer = perf_counter()
            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "judge_start",
                    "time": self._now_iso(),
                    **task_info,
                },
            )
            self.benchmark.evaluate(record, judge_client=self.judge_client)
            judge_seconds = perf_counter() - judge_timer
            score = float(record.score or 0.0)
            scores.append(score)
            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "judge_end",
                    "time": self._now_iso(),
                    "judge_seconds": round(judge_seconds, 3),
                    "score": score,
                    **task_info,
                },
            )

            attention_drift = self._measure_attention_drift(record, effective_context=effective_context)
            if attention_drift is not None:
                attention_drift_values.append(float(attention_drift.get("score", 0.0)))
                self._append_jsonl(
                    self.task_trace_path,
                    {
                        "run_id": self.run_id,
                        "event": "attention_drift",
                        "time": self._now_iso(),
                        "attention_drift_score": float(attention_drift.get("score", 0.0)),
                        "attention_drift_source": str(attention_drift.get("source", "unknown")),
                        "attention_drift_rationale": str(attention_drift.get("rationale", ""))[:200],
                        **task_info,
                    },
                )

            if score < 1:
                classifier_timer = perf_counter()
                self._append_jsonl(
                    self.task_trace_path,
                    {
                        "run_id": self.run_id,
                        "event": "failure_classifier_start",
                        "time": self._now_iso(),
                        **task_info,
                    },
                )
                failure_feedback = self._build_failure_feedback(record)
                classifier_seconds = perf_counter() - classifier_timer
                failure_mode = str(failure_feedback.get("mode", "F3") or "F3")
                record.failure_mode = failure_mode
                failure_mode_counter[failure_mode] += 1
                self._append_jsonl(
                    self.task_trace_path,
                    {
                        "run_id": self.run_id,
                        "event": "failure_classifier_end",
                        "time": self._now_iso(),
                        "classifier_seconds": round(classifier_seconds, 3),
                        "failure_mode": failure_mode,
                        "classifier_source": str(failure_feedback.get("source", "unknown")),
                        **task_info,
                    },
                )

                failure_record: dict[str, Any] = {
                    "task_id": record.task_id,
                    "context_category": str(record.metadata.get("context_category", "unknown")),
                    "sub_category": str(record.metadata.get("sub_category", "unknown")),
                    "query": record.query,
                    "answer": record.model_output or "",
                    "score": score,
                    "failure_mode": failure_mode,
                    "trace": record.reasoning_trace,
                    "verification_passed": record.verification_passed,
                    "tokens_used": record.tokens_used,
                    "failure_feedback": failure_feedback,
                    "judge_summary": {
                        "rationale": failure_feedback.get("judge_rationale", ""),
                        "unsatisfied_rubrics": failure_feedback.get("unsatisfied_rubrics", []),
                    },
                }

                # CaS/TDG-specific failure fields
                if self._is_cas or self._is_tdg:
                    failure_record["compilation_failed"] = bool(record.metadata.get("compilation_failed", False))
                    failure_record["sandbox_error"] = (
                        str(record.metadata.get("execution_output", ""))[:500]
                        if not record.metadata.get("execution_success", True)
                        else ""
                    )
                    failure_record["execution_traceback"] = str(record.metadata.get("execution_traceback", ""))[:500]
                    failure_record["sandbox_code_snippet"] = record.metadata.get("sandbox_code", "")[:500]
                if self._is_tdg:
                    failure_record["test_pass_rate"] = float(record.metadata.get("test_pass_rate", 0.0))

                failures.append(failure_record)

            task_seconds = perf_counter() - task_timer
            self._append_jsonl(
                self.task_trace_path,
                {
                    "run_id": self.run_id,
                    "event": "task_end",
                    "time": self._now_iso(),
                    "task_seconds": round(task_seconds, 3),
                    "score": score,
                    "tokens_used": int(record.tokens_used),
                    "verification_passed": (
                        bool(record.verification_passed) if record.verification_passed is not None else None
                    ),
                    "failure_mode": record.failure_mode,
                    **task_info,
                },
            )
            print(
                f"[Gen {generation}] cand {candidate_index}/{candidate_total} "
                f"task {task_idx}/{len(sampled)} done score={score:.0f} "
                f"tokens={int(record.tokens_used)} elapsed={task_seconds:.1f}s"
            )

        total_tasks = float(len(sampled) or 1)
        mode_acc = {
            mode: max(0.0, 1.0 - float(failure_mode_counter.get(mode, 0)) / total_tasks)
            for mode in ("F1", "F2", "F3", "F4")
        }

        # Compute fitness: weighted multi-objective for CaS, simple accuracy for legacy
        answer_correctness = sum(scores) / len(scores) if scores else 0.0
        compilation_rate = compilation_success_count / total_tasks if (self._is_cas or self._is_tdg) else 1.0
        execution_rate = execution_success_count / total_tasks if (self._is_cas or self._is_tdg) else 1.0
        avg_test_pass_rate = mean(test_pass_rate_values) if test_pass_rate_values else 0.0

        if self._is_tdg:
            w = self.config.fitness_weights
            fitness = (
                w.get("answer_correctness", 0.55) * answer_correctness
                + w.get("test_pass_rate", 0.25) * avg_test_pass_rate
                + w.get("execution_success", 0.15) * execution_rate
                + w.get("compilation_success", 0.05) * compilation_rate
            )
        elif self._is_cas:
            w = self.config.fitness_weights
            fitness = (
                w.get("answer_correctness", 0.6) * answer_correctness
                + w.get("execution_success", 0.2) * execution_rate
                + w.get("compilation_success", 0.2) * compilation_rate
            )
        else:
            fitness = answer_correctness

        eval_summary = self._summarize_failures(
            failures=failures,
            num_tasks=len(sampled),
            fitness=fitness,
            answer_correctness=answer_correctness,
            token_usage=token_usage,
            attention_drift_values=attention_drift_values,
            compilation_success_count=compilation_success_count,
            execution_success_count=execution_success_count,
            is_cas=self._is_cas or self._is_tdg,
            is_tdg=self._is_tdg,
            test_pass_rate_values=test_pass_rate_values,
        )
        return fitness, mode_acc, failures, eval_summary

    @staticmethod
    def _build_effective_context(record: TaskRecord) -> str:
        """Merge system context with prior turns for multi-turn tasks."""

        context = record.context or ""
        if len(record.messages_raw) <= 2:
            return context

        prior_turns: list[str] = []
        for message in record.messages_raw[1:-1]:
            role = str(message.get("role", "")).upper()
            content = str(message.get("content", ""))
            prior_turns.append(f"[{role}] {content}")

        if not prior_turns:
            return context

        suffix = "\n".join(prior_turns)
        return f"{context}\n\n---\nPrior conversation:\n{suffix}" if context else suffix

    @staticmethod
    def _build_cas_inputs(record: TaskRecord) -> tuple[str, str]:
        """Build role-preserving CaS context/query payloads."""

        normalized_messages = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in record.messages_raw
        ]
        messages_json = json.dumps(normalized_messages, ensure_ascii=False)
        base_context = str(record.context or "")
        context_sections = [
            "RAW_MESSAGES_JSON (role-preserving):\n"
            f"{messages_json}",
        ]
        if base_context and not EvolutionEngine._context_covered_by_messages(base_context, normalized_messages):
            context_sections.append(
                "SYSTEM_CONTEXT_EXTRACT:\n"
                f"{base_context}"
            )
        context_payload = "\n\n".join(context_sections)
        query_payload = (
            "LAST_USER_QUERY:\n"
            f"{str(record.query or '')}\n\n"
            "Use all constraints from RAW_MESSAGES_JSON when producing FINAL_ANSWER."
        )
        return context_payload, query_payload

    @staticmethod
    def _build_tdg_inputs(record: TaskRecord) -> tuple[str, str]:
        """Build TDG context/query payloads preserving full context without CaS framing."""

        normalized_messages = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in record.messages_raw
        ]
        messages_json = json.dumps(normalized_messages, ensure_ascii=False)
        base_context = str(record.context or "")
        context_sections = [
            "RAW_MESSAGES_JSON (role-preserving):\n"
            f"{messages_json}",
        ]
        if base_context and not EvolutionEngine._context_covered_by_messages(base_context, normalized_messages):
            context_sections.append(
                "SYSTEM_CONTEXT_EXTRACT:\n"
                f"{base_context}"
            )
        context_payload = "\n\n".join(context_sections)
        query_payload = (
            "LAST_USER_QUERY:\n"
            f"{str(record.query or '')}\n\n"
            "Use all constraints from the conversation when producing your answer."
        )
        return context_payload, query_payload

    @staticmethod
    def _context_covered_by_messages(base_context: str, messages: list[dict[str, str]]) -> bool:
        compact_context = re.sub(r"\s+", " ", str(base_context or "")).strip()
        if not compact_context:
            return True
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            if role != "system":
                continue
            compact_content = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
            if not compact_content:
                continue
            if compact_context in compact_content or compact_content in compact_context:
                return True
        return False

    def _sample_tasks(self, sample_size: int) -> list[TaskRecord]:
        if sample_size >= len(self.tasks):
            return list(self.tasks)
        return self.rng.sample(self.tasks, sample_size)

    @staticmethod
    def _build_parent_performance(
        mode_accuracy: dict[str, float],
        overall: float,
        eval_summary: dict[str, Any] | None = None,
    ) -> ParentPerformance:
        compilation_rate = 1.0
        execution_rate = 1.0
        if eval_summary:
            compilation_rate = float(eval_summary.get("compilation_success_rate", 1.0))
            execution_rate = float(eval_summary.get("execution_success_rate", 1.0))

        return ParentPerformance(
            overall=overall,
            f1=mode_accuracy.get("F1", overall),
            f2=mode_accuracy.get("F2", overall),
            f3=mode_accuracy.get("F3", overall),
            f4=mode_accuracy.get("F4", overall),
            compilation_success_rate=compilation_rate,
            execution_success_rate=execution_rate,
        )

    def _compute_selection_score(self, fitness: float, eval_summary: dict[str, Any]) -> float:
        """Accuracy-first score used for ranking/parent selection."""

        accuracy = float(eval_summary.get("accuracy", 0.0))
        if not self._is_cas and not self._is_tdg:
            return accuracy + 1e-3 * float(fitness)

        execution_rate = float(eval_summary.get("execution_success_rate", 0.0))
        compilation_rate = float(eval_summary.get("compilation_success_rate", 0.0))
        test_pass_rate = float(eval_summary.get("avg_test_pass_rate", 0.0))
        # Lexicographic preference: accuracy dominates, runtime metrics break ties.
        return (
            accuracy
            + 1e-3 * float(fitness)
            + 1e-4 * execution_rate
            + 1e-5 * compilation_rate
            + 1e-6 * test_pass_rate
        )

    @staticmethod
    def _candidate_rank_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        """Sort key that prioritizes answer accuracy over proxy runtime metrics."""

        summary = item.get("eval_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        accuracy = float(item.get("answer_accuracy", summary.get("accuracy", 0.0)))
        fitness = float(item.get("fitness", 0.0))
        execution_rate = float(summary.get("execution_success_rate", 1.0))
        compilation_rate = float(summary.get("compilation_success_rate", 1.0))
        test_pass_rate = float(summary.get("avg_test_pass_rate", 0.0))
        return (accuracy, fitness, execution_rate, compilation_rate, test_pass_rate)

    def _build_failure_feedback(self, record: TaskRecord) -> dict[str, Any]:
        """Create structured failure analysis for mutation feedback."""

        classifier_client = self.failure_classifier_client if self.config.enable_failure_classifier else None
        return build_failure_feedback(
            record=record,
            llm_client=classifier_client,
            model=self.config.failure_classifier_model,
        )

    def _measure_attention_drift(
        self,
        record: TaskRecord,
        effective_context: str | None = None,
    ) -> dict[str, Any] | None:
        """Estimate context-faithfulness drift (0=faithful, 1=parametric override)."""

        if not self.config.enable_attention_drift:
            return None
        if self.config.attention_drift_sample_rate <= 0:
            return None
        if self.rng.random() > self.config.attention_drift_sample_rate:
            return None

        answer = str(record.model_output or "").strip()
        context = str(effective_context if effective_context is not None else record.context or "")
        query = str(record.query or "")

        if not answer:
            return {
                "score": 1.0,
                "source": "heuristic_empty",
                "rationale": "Empty output treated as maximum drift.",
            }

        if self.judge_client is None:
            return self._heuristic_attention_drift(answer=answer, context=context)

        timeout_seconds = env_float(
            [
                "ATTENTION_DRIFT_API_TIMEOUT_SECONDS",
                "JUDGE_API_TIMEOUT_SECONDS",
                "OPENAI_API_TIMEOUT_SECONDS",
                "OPENAI_API_TIMEOUT",
                "API_TIMEOUT_SECONDS",
            ],
            default=45.0,
        )
        max_tokens = env_int(
            ["ATTENTION_DRIFT_MAX_TOKENS", "OPENAI_MAX_TOKENS", "MAX_TOKENS"],
            default=512,
        )

        prompt = (
            "Rate context faithfulness drift for the answer on a 0-1 scale.\n"
            "0 means fully grounded in provided context.\n"
            "1 means answer ignores context and relies on outside priors.\n"
            "Return JSON only with keys: drift, rationale.\n\n"
            f"Query:\n{query[:1200]}\n\n"
            f"Context excerpt:\n{context[:3000]}\n\n"
            f"Answer:\n{answer[:1800]}\n"
        )

        try:
            response = self.judge_client.chat.completions.create(
                model=self.config.attention_drift_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
            )
            raw = str(response.choices[0].message.content or "").strip()
            parsed = self._extract_json_from_text(raw)
            if isinstance(parsed, dict):
                drift = self._parse_drift_score(parsed.get("drift"))
                if drift is not None:
                    return {
                        "score": drift,
                        "source": "judge",
                        "rationale": str(parsed.get("rationale", ""))[:300],
                    }

            numeric = self._parse_drift_score(raw)
            if numeric is not None:
                return {
                    "score": numeric,
                    "source": "judge_numeric",
                    "rationale": "",
                }
        except Exception as exc:
            heuristic = self._heuristic_attention_drift(answer=answer, context=context)
            heuristic["source"] = "heuristic_fallback"
            heuristic["rationale"] = f"Judge failed: {str(exc)[:200]}"
            return heuristic

        return self._heuristic_attention_drift(answer=answer, context=context)

    @staticmethod
    def _heuristic_attention_drift(answer: str, context: str) -> dict[str, Any]:
        """Lexical-overlap fallback for attention drift when judge is unavailable."""

        answer_tokens = set(re.findall(r"[A-Za-z0-9_]{3,}", answer.lower()))
        context_tokens = set(re.findall(r"[A-Za-z0-9_]{3,}", context.lower()))
        if not answer_tokens:
            return {"score": 1.0, "source": "heuristic_empty", "rationale": "Answer has no lexical tokens."}
        overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
        drift = 1.0 - overlap
        return {
            "score": float(max(0.0, min(1.0, drift))),
            "source": "heuristic_overlap",
            "rationale": "Drift estimated from answer-context lexical overlap.",
        }

    @staticmethod
    def _extract_json_from_text(raw: str) -> dict[str, Any] | None:
        """Extract first JSON object from plain/fenced model output."""

        text = str(raw or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _parse_drift_score(raw: Any) -> float | None:
        """Parse and clamp drift score to [0,1]."""

        if isinstance(raw, (float, int)):
            return float(max(0.0, min(1.0, float(raw))))

        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return float(max(0.0, min(1.0, float(text))))
        except (TypeError, ValueError):
            pass

        match = re.search(r"([01](?:\.\d+)?)", text)
        if not match:
            return None
        try:
            return float(max(0.0, min(1.0, float(match.group(1)))))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _summarize_failures(
        failures: list[dict[str, Any]],
        num_tasks: int,
        fitness: float,
        answer_correctness: float,
        token_usage: list[int],
        attention_drift_values: list[float],
        compilation_success_count: int = 0,
        execution_success_count: int = 0,
        is_cas: bool = False,
        is_tdg: bool = False,
        test_pass_rate_values: list[float] | None = None,
    ) -> dict[str, Any]:
        """Build compact, generation-level failure summary for archive metadata."""

        mode_counter: Counter[str] = Counter()
        root_cause_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        rubric_counter: Counter[str] = Counter()
        verification_failed = 0
        compilation_failure_count = 0
        execution_failure_count = 0
        top_compilation_errors: Counter[str] = Counter()
        top_execution_errors: Counter[str] = Counter()

        for item in failures:
            mode_counter[str(item.get("failure_mode", "unknown"))] += 1
            if item.get("verification_passed") is False:
                verification_failed += 1

            # CaS-specific counters
            if is_cas:
                if item.get("compilation_failed"):
                    compilation_failure_count += 1
                    sandbox_err = str(item.get("sandbox_error", ""))[:100]
                    if sandbox_err:
                        top_compilation_errors[sandbox_err] += 1
                exec_tb = str(item.get("execution_traceback", ""))
                if exec_tb:
                    execution_failure_count += 1
                    top_execution_errors[exec_tb[:100]] += 1

            feedback = item.get("failure_feedback", {})
            if isinstance(feedback, dict):
                root_cause = str(feedback.get("root_cause", "")).strip()
                if root_cause:
                    root_cause_counter[root_cause] += 1

                actions = feedback.get("repair_actions", [])
                if isinstance(actions, list):
                    for action in actions:
                        text = str(action).strip()
                        if text:
                            action_counter[text] += 1

                unsatisfied = feedback.get("unsatisfied_rubrics", [])
                if isinstance(unsatisfied, list):
                    for rubric in unsatisfied:
                        text = str(rubric).strip()
                        if text:
                            rubric_counter[text] += 1

        avg_tokens = (sum(token_usage) / len(token_usage)) if token_usage else 0.0
        max_tokens = max(token_usage) if token_usage else 0
        num_failures = len(failures)
        drift_count = len(attention_drift_values)
        drift_mean = (sum(attention_drift_values) / drift_count) if drift_count else None
        drift_high_rate = (
            sum(1 for value in attention_drift_values if float(value) >= 0.67) / drift_count
            if drift_count
            else None
        )

        total = max(1, num_tasks)
        summary: dict[str, Any] = {
            "num_tasks": int(num_tasks),
            "num_failures": int(num_failures),
            "fitness": float(fitness),
            "failure_rate": float(1.0 - answer_correctness),
            "accuracy": float(answer_correctness),
            "avg_tokens_per_task": float(avg_tokens),
            "max_tokens_per_task": int(max_tokens),
            "token_efficiency": float(answer_correctness / avg_tokens) if avg_tokens > 0 else 0.0,
            "fitness_token_efficiency": float(fitness / avg_tokens) if avg_tokens > 0 else 0.0,
            "verification_failed_count": int(verification_failed),
            "attention_drift_mean": drift_mean,
            "attention_drift_high_rate": drift_high_rate,
            "attention_drift_measured_tasks": int(drift_count),
            "failure_mode_counts": dict(mode_counter),
            "top_root_causes": [text for text, _ in root_cause_counter.most_common(5)],
            "top_repair_actions": [text for text, _ in action_counter.most_common(8)],
            "top_unsatisfied_rubrics": [text for text, _ in rubric_counter.most_common(8)],
        }

        if is_cas:
            compilation_failure_count = max(0, total - int(compilation_success_count))
            execution_failure_count = max(0, total - int(execution_success_count))
            summary["compilation_success_rate"] = float(compilation_success_count / total)
            summary["execution_success_rate"] = float(execution_success_count / total)
            summary["compilation_failure_count"] = int(compilation_failure_count)
            summary["execution_failure_count"] = int(execution_failure_count)
            summary["top_compilation_errors"] = [e for e, _ in top_compilation_errors.most_common(5)]
            summary["top_execution_errors"] = [e for e, _ in top_execution_errors.most_common(5)]

        if is_tdg and test_pass_rate_values is not None:
            avg_tpr = (sum(test_pass_rate_values) / len(test_pass_rate_values)) if test_pass_rate_values else 0.0
            summary["avg_test_pass_rate"] = float(avg_tpr)

        return summary

    @staticmethod
    def _compute_population_diversity(population: list[dict[str, Any]]) -> dict[str, float]:
        """Measure code-space diversity for one generation."""

        codes = [str(item.get("code", "")) for item in population]
        shas = [str(item.get("sha", "")) for item in population]
        size = len(codes)
        if size <= 1:
            return {
                "code_diversity": 0.0,
                "unique_sha_ratio": 1.0 if size == 1 else 0.0,
            }

        token_sets = [set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)) for code in codes]
        distances: list[float] = []
        for left in range(size):
            for right in range(left + 1, size):
                a = token_sets[left]
                b = token_sets[right]
                union = len(a | b)
                if union == 0:
                    distances.append(0.0)
                else:
                    distances.append(1.0 - (len(a & b) / union))

        diversity = (sum(distances) / len(distances)) if distances else 0.0
        unique_sha_ratio = len({sha for sha in shas if sha}) / max(1, size)
        return {
            "code_diversity": float(diversity),
            "unique_sha_ratio": float(unique_sha_ratio),
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_failure_samples(self, generation: int, sha: str, failures: list[dict[str, Any]], limit: int = 5) -> None:
        """Write a few failure samples per candidate for quick debugging."""

        if not failures:
            return

        for item in failures[:limit]:
            feedback = item.get("failure_feedback", {})
            record: dict[str, Any] = {
                "run_id": self.run_id,
                "generation": generation,
                "sha": sha,
                "task_id": item.get("task_id"),
                "context_category": item.get("context_category", "unknown"),
                "sub_category": item.get("sub_category", "unknown"),
                "score": item.get("score"),
                "failure_mode": item.get("failure_mode"),
                "query": str(item.get("query", ""))[:300],
                "answer": str(item.get("answer", ""))[:300],
                "root_cause": str(feedback.get("root_cause", ""))[:300] if isinstance(feedback, dict) else "",
                "repair_actions": feedback.get("repair_actions", [])[:3] if isinstance(feedback, dict) else [],
                "time": self._now_iso(),
            }
            if self._is_cas or self._is_tdg:
                record["compilation_failed"] = item.get("compilation_failed", False)
                record["sandbox_error"] = str(item.get("sandbox_error", ""))[:200]
                record["execution_traceback"] = str(item.get("execution_traceback", ""))[:200]
            if self._is_tdg:
                record["test_pass_rate"] = float(item.get("test_pass_rate", 0.0))
            self._append_jsonl(self.failure_trace_path, record)
