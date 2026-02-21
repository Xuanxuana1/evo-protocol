"""Benchmark abstractions for Evo-Protocol experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskRecord:
    """Universal task container used by loading, protocol execution, and evaluation."""

    task_id: str
    context: str
    query: str
    messages_raw: list[dict[str, Any]] = field(default_factory=list)
    rubrics: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    model_output: Optional[str] = None
    reasoning_trace: list[str] = field(default_factory=list)
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verification_passed: Optional[bool] = None

    score: Optional[float] = None
    eval_detail: dict[str, Any] = field(default_factory=dict)
    failure_mode: Optional[str] = None


class BaseBenchmark(ABC):
    """Pluggable benchmark interface for context-learning tasks."""

    name: str = "base"

    @abstractmethod
    def load_tasks(self, data_path: str, split: str = "all") -> list[TaskRecord]:
        """Load raw benchmark data and return task records."""

    @abstractmethod
    def evaluate(self, record: TaskRecord, judge_client=None) -> TaskRecord:
        """Evaluate a record and populate score/eval metadata."""

    def get_metrics(self, records: list[TaskRecord]) -> dict[str, float]:
        """Default metrics: overall accuracy and per-category accuracy."""

        scored = [record for record in records if record.score is not None]
        if not scored:
            return {"accuracy": 0.0, "total": 0}

        total = len(scored)
        overall = sum(float(record.score) for record in scored) / total
        metrics: dict[str, float] = {"accuracy": overall, "total": total}

        by_category: dict[str, list[float]] = {}
        for record in scored:
            category = record.metadata.get("context_category", "unknown")
            by_category.setdefault(category, []).append(float(record.score))

        for category, values in by_category.items():
            metrics[f"accuracy/{category}"] = sum(values) / len(values)

        return metrics


BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {}


def register_benchmark(name: str):
    """Register a benchmark class by name."""

    def decorator(cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
        BENCHMARK_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """Instantiate a registered benchmark implementation."""

    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(sorted(BENCHMARK_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown benchmark {name}. Available: {available}")
    return BENCHMARK_REGISTRY[name](**kwargs)
