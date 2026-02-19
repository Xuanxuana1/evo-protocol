"""CLI entrypoint for running Evo-Protocol evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from baselines.naive import NaiveProtocol
from benchmarks import get_benchmark
from core.archive import ProtocolArchive
from core.env_utils import env_float, first_env, load_env_file
from core.evaluator import run_protocol_on_benchmark
from core.evolution_loop import EvolutionConfig, EvolutionEngine
from core.meta_architect import MetaArchitect
from core.protocol_loader import ProtocolLoader


def load_initial_code(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")

    import inspect

    return inspect.getsource(NaiveProtocol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evo-Protocol evolution loop")
    parser.add_argument("--benchmark", default="cl-bench", help="Benchmark registry key")
    parser.add_argument("--data-path", default="data/CL-bench.jsonl", help="Benchmark dataset path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], help="Data split")
    parser.add_argument("--initial-code", default=None, help="Path to initial protocol .py")
    parser.add_argument("--worker-model", default=None, help="Model used inside protocols")
    parser.add_argument("--architect-model", default=None, help="Model used for protocol mutation")
    parser.add_argument("--judge-model", default=None, help="Judge model for rubric evaluation")
    parser.add_argument("--failure-classifier-model", default=None, help="Model for failure-mode analysis")
    parser.add_argument("--disable-failure-classifier", action="store_true", help="Disable LLM-based failure analysis")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tasks-per-eval", type=int, default=20)
    parser.add_argument("--archive-dir", default="archive")
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/evolution_result.json", help="Where to save final summary")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip final evaluation of best protocol")
    parser.add_argument("--final-eval-split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--final-eval-workers", type=int, default=1)
    parser.add_argument("--final-eval-output", default="outputs/evo_best_protocol_eval.jsonl")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key (fallback to OPENAI_API_KEY)")
    parser.add_argument("--api-timeout", type=float, default=None, help="Per-request API timeout in seconds")
    args = parser.parse_args()

    load_env_file(args.env_file)

    worker_model = args.worker_model or first_env(["EVO_WORKER_MODEL", "OPENAI_MODEL", "MODEL"]) or "gpt-4o-mini"
    architect_model = args.architect_model or first_env(["EVO_ARCHITECT_MODEL", "ARCHITECT_MODEL"]) or "gpt-4o"
    judge_model = args.judge_model or first_env(["EVO_JUDGE_MODEL", "OPENAI_JUDGE_MODEL", "JUDGE_MODEL"]) or "gpt-5.1"
    failure_classifier_model = (
        args.failure_classifier_model
        or first_env(["EVO_FAILURE_CLASSIFIER_MODEL", "FAILURE_CLASSIFIER_MODEL"])
        or "gpt-4o"
    )

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
    architect_client = OpenAI(**client_kwargs)
    judge_client = OpenAI(**client_kwargs)

    archive = ProtocolArchive(args.archive_dir)
    loader = ProtocolLoader(worker_client, worker_model)
    architect = MetaArchitect(architect_client, architect_model=architect_model)

    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        tasks_per_evaluation=args.tasks_per_eval,
        enable_failure_classifier=not args.disable_failure_classifier,
        failure_classifier_model=failure_classifier_model,
        max_repair_attempts=args.max_repair_attempts,
        seed=args.seed,
    )

    engine = EvolutionEngine(
        benchmark_name=args.benchmark,
        data_path=args.data_path,
        split=args.split,
        protocol_loader=loader,
        meta_architect=architect,
        archive=archive,
        config=config,
        benchmark_kwargs={"judge_model": judge_model},
        judge_client=judge_client,
        failure_classifier_client=judge_client,
    )

    initial_code = load_initial_code(args.initial_code)
    summary = engine.run(initial_code=initial_code)
    log_files = summary.get("log_files", {})
    if isinstance(log_files, dict) and log_files:
        print("\nIntermediate logs:")
        for key in sorted(log_files):
            print(f"  {key}: {log_files[key]}")

    if not args.skip_final_eval:
        best_sha = str(summary["best_sha"])
        best_code = archive.get_code(best_sha)
        best_protocol, load_error = loader.load_from_code(best_code)
        if best_protocol is None:
            err_msg = load_error.message if load_error else "unknown"
            raise RuntimeError(f"Failed to load best protocol {best_sha}: {err_msg}")

        benchmark_kwargs = {"judge_model": judge_model}
        records = run_protocol_on_benchmark(
            protocol=best_protocol,
            benchmark_name=args.benchmark,
            data_path=args.data_path,
            split=args.final_eval_split,
            output_path=args.final_eval_output,
            benchmark_kwargs=benchmark_kwargs,
            judge_client=judge_client,
            workers=args.final_eval_workers,
        )
        benchmark = get_benchmark(args.benchmark, **benchmark_kwargs)
        metrics = benchmark.get_metrics(records)
        summary["final_eval"] = {
            "split": args.final_eval_split,
            "output_path": args.final_eval_output,
            "num_records": len(records),
            "metrics": metrics,
        }
        print("\nFinal evaluation metrics:")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evolution completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
