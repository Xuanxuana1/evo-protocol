"""CLI to run baseline protocols on CL-bench via the new abstraction layer."""

from __future__ import annotations

import argparse

from openai import OpenAI

from baselines import CoTProtocol, NaiveProtocol, ReActProtocol
from core.env_utils import env_float, first_env, load_env_file
from core.evaluator import print_metrics, run_protocol_on_benchmark


BASELINE_REGISTRY = {
    "naive": NaiveProtocol,
    "cot": CoTProtocol,
    "react": ReActProtocol,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline protocols")
    parser.add_argument("--baseline", choices=sorted(BASELINE_REGISTRY), default="naive")
    parser.add_argument("--benchmark", default="cl-bench")
    parser.add_argument("--data-path", default="data/CL-bench.jsonl")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="outputs/baseline_output.jsonl")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-timeout", type=float, default=None, help="Per-request API timeout in seconds")
    args = parser.parse_args()

    load_env_file(args.env_file)

    worker_model = args.worker_model or first_env(["BASELINE_WORKER_MODEL", "OPENAI_MODEL", "MODEL"]) or "gpt-4o-mini"
    judge_model = args.judge_model or first_env(["BASELINE_JUDGE_MODEL", "OPENAI_JUDGE_MODEL", "JUDGE_MODEL"]) or "gpt-5.1"
    base_url = args.base_url or first_env(["OPENAI_BASE_URL", "BASE_URL", "OPENAI_API_BASE"])
    api_key = args.api_key or first_env(["OPENAI_API_KEY", "API_KEY"])
    api_timeout = (
        float(args.api_timeout)
        if args.api_timeout and args.api_timeout > 0
        else env_float(
            ["OPENAI_API_TIMEOUT_SECONDS", "OPENAI_API_TIMEOUT", "API_TIMEOUT_SECONDS"],
            default=90.0,
        )
    )
    if not api_key:
        raise ValueError("Please provide --api-key or set OPENAI_API_KEY/API_KEY")

    client_kwargs = {"api_key": api_key, "timeout": api_timeout}
    if base_url:
        client_kwargs["base_url"] = base_url

    worker_client = OpenAI(**client_kwargs)
    judge_client = OpenAI(**client_kwargs)

    protocol_cls = BASELINE_REGISTRY[args.baseline]
    protocol = protocol_cls(worker_client, worker_model)

    records = run_protocol_on_benchmark(
        protocol=protocol,
        benchmark_name=args.benchmark,
        data_path=args.data_path,
        split=args.split,
        output_path=args.output,
        benchmark_kwargs={"judge_model": judge_model},
        judge_client=judge_client,
        workers=args.workers,
    )
    print_metrics(records, args.benchmark, benchmark_kwargs={"judge_model": judge_model})


if __name__ == "__main__":
    main()
