"""Benchmark-agnostic protocol evaluation utilities."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from benchmarks.base import TaskRecord, get_benchmark
from core.base_protocol import BaseProtocol
from core.protocol_loader import ProtocolLoader


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


def _clone_protocol(protocol: BaseProtocol) -> BaseProtocol:
    """Clone protocol for parallel workers to avoid shared mutable state."""

    try:
        return protocol.__class__(protocol.llm, protocol.model)
    except Exception:
        return protocol


def run_protocol_on_benchmark(
    protocol: BaseProtocol,
    benchmark_name: str,
    data_path: str,
    split: str = "test",
    output_path: Optional[str] = None,
    benchmark_kwargs: Optional[dict] = None,
    judge_client=None,
    workers: int = 1,
) -> list[TaskRecord]:
    """Execute a protocol across benchmark tasks and optionally persist results."""

    benchmark = get_benchmark(benchmark_name, **(benchmark_kwargs or {}))
    tasks = benchmark.load_tasks(data_path=data_path, split=split)
    loader = ProtocolLoader(protocol.llm, protocol.model)

    def process(record: TaskRecord) -> TaskRecord:
        local_protocol = protocol if workers == 1 else _clone_protocol(protocol)
        context = _build_effective_context(record)
        result = loader.run_with_timeout(local_protocol, context=context, query=record.query)

        if result is None:
            record.model_output = ""
            record.reasoning_trace = ["Protocol execution failed or timed out."]
            record.verification_passed = False
            record.tokens_used = 0
        else:
            record.model_output = result.answer
            record.reasoning_trace = list(result.reasoning_trace)
            record.verification_passed = bool(result.verification_passed)
            record.tokens_used = int(result.tokens_used)
            record.metadata.update(result.metadata)

        benchmark.evaluate(record, judge_client=judge_client)
        return record

    results: list[TaskRecord] = []
    if workers <= 1:
        for task in tqdm(tasks, desc=f"{benchmark_name}:{split}"):
            results.append(process(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{benchmark_name}:{split}"):
                results.append(future.result())

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            for record in results:
                file_obj.write(
                    json.dumps(
                        {
                            "idx": record.metadata.get("idx"),
                            "messages": record.messages_raw,
                            "model_output": record.model_output,
                            "rubrics": record.rubrics,
                            "score": record.score,
                            "metadata": record.metadata,
                            "failure_mode": record.failure_mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return results


def print_metrics(records: list[TaskRecord], benchmark_name: str, benchmark_kwargs: Optional[dict] = None) -> None:
    """Print aggregate metrics using benchmark-specific logic."""

    benchmark = get_benchmark(benchmark_name, **(benchmark_kwargs or {}))
    metrics = benchmark.get_metrics(records)
    print(f"\n=== {benchmark_name} metrics ===")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
