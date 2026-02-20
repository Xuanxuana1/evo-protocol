"""Main Evo-Protocol optimization loop (evaluate -> select -> mutate)."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from statistics import mean
from pathlib import Path
from typing import Any

from benchmarks.base import TaskRecord, get_benchmark
from core.archive import ProtocolArchive
from core.base_protocol import BaseProtocol
from core.env_utils import env_float, env_int
from core.failure_classifier import build_failure_feedback
from core.meta_architect import MetaArchitect, ParentPerformance
from core.protocol_loader import ProtocolLoader


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


class EvolutionEngine:
    """Population-based protocol evolution engine."""

    def __init__(
        self,
        benchmark_name: str,
        data_path: str,
        protocol_loader: ProtocolLoader,
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
        self.rng = random.Random(self.config.seed)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = self.archive.dir / "_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.generation_trace_path = self.logs_dir / "generation_trace.jsonl"
        self.candidate_trace_path = self.logs_dir / "candidate_trace.jsonl"
        self.failure_trace_path = self.logs_dir / "failure_trace.jsonl"
        self.task_trace_path = self.logs_dir / "task_trace.jsonl"

    def run(self, initial_code: str) -> dict[str, Any]:
        """Run full evolution from an initial protocol source string."""

        protocol, error = self.loader.load_from_code(initial_code)
        if protocol is None:
            raise ValueError(f"Initial protocol failed validation: {error.message if error else 'unknown'}")

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
        while len(population) < self.config.population_size:
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
                "elite_count": self.config.elite_count,
                "tasks_per_evaluation": self.config.tasks_per_evaluation,
                "benchmark": getattr(self.benchmark, "name", "unknown"),
            },
        )

        for generation in range(1, self.config.generations + 1):
            print(f"\n[Gen {generation}] evaluating {len(population)} protocols...")
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
                self.archive.update_evaluation(
                    sha=candidate["sha"],
                    generation=generation,
                    score=fitness,
                    failure_summary=eval_summary,
                )
                candidate_log = {
                    "run_id": self.run_id,
                    "generation": generation,
                    "candidate_index": idx,
                    "population_size": len(population),
                    "sha": candidate["sha"],
                    "fitness": fitness,
                    "num_failures": len(failures),
                    "failure_mode_counts": eval_summary.get("failure_mode_counts", {}),
                    "avg_tokens_per_task": eval_summary.get("avg_tokens_per_task", 0.0),
                    "attention_drift_mean": eval_summary.get("attention_drift_mean"),
                    "attention_drift_high_rate": eval_summary.get("attention_drift_high_rate"),
                    "top_root_causes": eval_summary.get("top_root_causes", [])[:3],
                    "time": self._now_iso(),
                }
                self._append_jsonl(self.candidate_trace_path, candidate_log)
                print(
                    f"[Gen {generation}] candidate {idx}/{len(population)} "
                    f"sha={candidate['sha']} fitness={fitness:.4f} "
                    f"failures={len(failures)}"
                )
                self._append_failure_samples(generation, candidate["sha"], failures)

            ranked = sorted(population, key=lambda item: item["fitness"], reverse=True)
            elites = ranked[: max(1, self.config.elite_count)]
            best = elites[0]
            diversity = self._compute_population_diversity(ranked)
            gen_record = {
                "generation": generation,
                "best_sha": best["sha"],
                "best_fitness": best["fitness"],
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
                f"mean={gen_record['mean_fitness']:.4f} "
                f"diversity={gen_record['code_diversity']:.4f}"
            )

            if generation == self.config.generations:
                break

            new_population = [dict(item) for item in elites]
            mutation_attempts = 0
            max_attempts = self.config.population_size * 6

            while len(new_population) < self.config.population_size and mutation_attempts < max_attempts:
                mutation_attempts += 1
                parent = self.rng.choice(elites)
                parent_perf = self._build_parent_performance(parent.get("mode_accuracy", {}), parent["fitness"])
                failures = parent.get("failures", [])
                sampled_failures = self.rng.sample(
                    failures,
                    k=min(len(failures), self.config.failure_samples_per_mutation),
                ) if failures else []

                child_protocol, child_code_or_err = self.meta_architect.mutate(
                    loader=self.loader,
                    generation=generation,
                    parent_code=parent["code"],
                    parent_performance=parent_perf,
                    failure_examples=sampled_failures,
                    max_repair_attempts=self.config.max_repair_attempts,
                )
                if child_protocol is None:
                    continue

                child_code = child_code_or_err
                child_sha = self.archive.save(
                    child_code,
                    generation=generation,
                    parent_sha=parent["sha"],
                )
                new_population.append(
                    {
                        "sha": child_sha,
                        "code": child_code,
                        "protocol": child_protocol,
                        "fitness": 0.0,
                        "mode_accuracy": {},
                        "failures": [],
                    }
                )

            while len(new_population) < self.config.population_size:
                new_population.append(dict(self.rng.choice(elites)))

            population = new_population

        best_final = max(population, key=lambda item: item["fitness"])
        self._append_jsonl(
            self.generation_trace_path,
            {
                "run_id": self.run_id,
                "event": "run_end",
                "time": self._now_iso(),
                "best_sha": best_final["sha"],
                "best_fitness": best_final["fitness"],
            },
        )
        return {
            "run_id": self.run_id,
            "best_sha": best_final["sha"],
            "best_fitness": best_final["fitness"],
            "history": history,
            "log_files": {
                "generation_trace": str(self.generation_trace_path),
                "candidate_trace": str(self.candidate_trace_path),
                "failure_trace": str(self.failure_trace_path),
                "task_trace": str(self.task_trace_path),
            },
        }

    def _evaluate_protocol(
        self,
        protocol: BaseProtocol,
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

            effective_context = self._build_effective_context(record)
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
            result = self.loader.run_with_timeout(protocol, context=effective_context, query=record.query)
            worker_seconds = perf_counter() - worker_timer
            if result is None:
                record.model_output = ""
                record.reasoning_trace = ["execution failed or timed out"]
                record.verification_passed = False
                worker_status = "timeout_or_error"
            else:
                record.model_output = result.answer
                record.reasoning_trace = list(result.reasoning_trace)
                record.verification_passed = bool(result.verification_passed)
                record.tokens_used = int(result.tokens_used)
                record.metadata.update(result.metadata)
                worker_status = "ok"

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
                failures.append(
                    {
                        "task_id": record.task_id,
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
                )

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
        fitness = sum(scores) / len(scores) if scores else 0.0
        eval_summary = self._summarize_failures(
            failures=failures,
            num_tasks=len(sampled),
            fitness=fitness,
            token_usage=token_usage,
            attention_drift_values=attention_drift_values,
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

    def _sample_tasks(self, sample_size: int) -> list[TaskRecord]:
        if sample_size >= len(self.tasks):
            return list(self.tasks)
        return self.rng.sample(self.tasks, sample_size)

    @staticmethod
    def _build_parent_performance(mode_accuracy: dict[str, float], overall: float) -> ParentPerformance:
        return ParentPerformance(
            overall=overall,
            f1=mode_accuracy.get("F1", overall),
            f2=mode_accuracy.get("F2", overall),
            f3=mode_accuracy.get("F3", overall),
            f4=mode_accuracy.get("F4", overall),
        )

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
        token_usage: list[int],
        attention_drift_values: list[float],
    ) -> dict[str, Any]:
        """Build compact, generation-level failure summary for archive metadata."""

        mode_counter: Counter[str] = Counter()
        root_cause_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        rubric_counter: Counter[str] = Counter()
        verification_failed = 0

        for item in failures:
            mode_counter[str(item.get("failure_mode", "unknown"))] += 1
            if item.get("verification_passed") is False:
                verification_failed += 1

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

        return {
            "num_tasks": int(num_tasks),
            "num_failures": int(num_failures),
            "failure_rate": float(1.0 - fitness),
            "accuracy": float(fitness),
            "avg_tokens_per_task": float(avg_tokens),
            "max_tokens_per_task": int(max_tokens),
            "token_efficiency": float(fitness / avg_tokens) if avg_tokens > 0 else 0.0,
            "verification_failed_count": int(verification_failed),
            "attention_drift_mean": drift_mean,
            "attention_drift_high_rate": drift_high_rate,
            "attention_drift_measured_tasks": int(drift_count),
            "failure_mode_counts": dict(mode_counter),
            "top_root_causes": [text for text, _ in root_cause_counter.most_common(5)],
            "top_repair_actions": [text for text, _ in action_counter.most_common(8)],
            "top_unsatisfied_rubrics": [text for text, _ in rubric_counter.most_common(8)],
        }

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
            record = {
                "run_id": self.run_id,
                "generation": generation,
                "sha": sha,
                "task_id": item.get("task_id"),
                "score": item.get("score"),
                "failure_mode": item.get("failure_mode"),
                "query": str(item.get("query", ""))[:300],
                "answer": str(item.get("answer", ""))[:300],
                "root_cause": str(feedback.get("root_cause", ""))[:300] if isinstance(feedback, dict) else "",
                "repair_actions": feedback.get("repair_actions", [])[:3] if isinstance(feedback, dict) else [],
                "time": self._now_iso(),
            }
            self._append_jsonl(self.failure_trace_path, record)
