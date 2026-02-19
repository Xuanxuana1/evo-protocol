"""Benchmark registry exports."""

from benchmarks.base import BaseBenchmark, TaskRecord, get_benchmark, register_benchmark
from benchmarks.cl_bench import CLBenchmark

__all__ = [
    "BaseBenchmark",
    "TaskRecord",
    "get_benchmark",
    "register_benchmark",
    "CLBenchmark",
]
