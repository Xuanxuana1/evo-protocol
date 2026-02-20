"""CLI entrypoint for running Evo-Protocol evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def load_experiment_config(path: str | None, allow_missing: bool = False) -> dict[str, Any]:
    """Load JSON/YAML experiment config file."""

    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")

    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        if allow_missing:
            # Keep CLI backwards-compatible even when PyYAML is unavailable.
            print(f"[Warn] PyYAML not available, skipping config load: {path}")
            return {}
        raise RuntimeError("YAML config requires PyYAML. Install `pyyaml` or use a JSON config.") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evo-Protocol evolution loop")
    default_config_path = "configs/evolution.yaml"
    parser.add_argument(
        "--config",
        default=default_config_path,
        help=f"Path to YAML/JSON experiment config (default: {default_config_path})",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore config file and use only CLI args + env vars",
    )
    parser.add_argument("--benchmark", default="cl-bench", help="Benchmark registry key")
    parser.add_argument("--data-path", default="data/CL-bench.jsonl", help="Benchmark dataset path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], help="Data split")
    parser.add_argument("--initial-code", default=None, help="Path to initial protocol .py")
    parser.add_argument("--worker-model", default=None, help="Model used inside protocols")
    parser.add_argument("--architect-model", default=None, help="Model used for protocol mutation")
    parser.add_argument("--judge-model", default=None, help="Judge model for rubric evaluation")
    parser.add_argument("--failure-classifier-model", default=None, help="Model for failure-mode analysis")
    parser.add_argument("--attention-drift-model", default=None, help="Model for attention-drift scoring")
    parser.add_argument("--disable-failure-classifier", action="store_true", help="Disable LLM-based failure analysis")
    parser.add_argument("--disable-attention-drift", action="store_true", help="Disable attention-drift metric scoring")
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
    parser.add_argument("--protocol-timeout", type=int, default=300, help="Protocol hard timeout in seconds")
    parser.add_argument("--attention-drift-sample-rate", type=float, default=1.0, help="Sample rate in [0,1] for attention-drift scoring")
    args = parser.parse_args()

    config_doc = (
        {}
        if args.no_config
        else load_experiment_config(
            args.config,
            allow_missing=(args.config == default_config_path),
        )
    )
    benchmark_cfg = config_doc.get("benchmark", {}) if isinstance(config_doc.get("benchmark", {}), dict) else {}
    evolution_cfg = config_doc.get("evolution", {}) if isinstance(config_doc.get("evolution", {}), dict) else {}
    evaluation_cfg = config_doc.get("evaluation", {}) if isinstance(config_doc.get("evaluation", {}), dict) else {}

    def resolve_cli_or_config(arg_name: str, config_value: Any) -> Any:
        cli_value = getattr(args, arg_name)
        flag = f"--{arg_name.replace('_', '-')}"
        provided = any(token == flag or token.startswith(f"{flag}=") for token in sys.argv[1:])
        if provided:
            return cli_value
        return cli_value if config_value is None else config_value

    load_env_file(args.env_file)

    benchmark_name = str(resolve_cli_or_config("benchmark", benchmark_cfg.get("name")))
    data_path = str(resolve_cli_or_config("data_path", benchmark_cfg.get("data_path")))
    split = str(resolve_cli_or_config("split", benchmark_cfg.get("split")))
    generations = int(resolve_cli_or_config("generations", evolution_cfg.get("generations")))
    population_size = int(resolve_cli_or_config("population_size", evolution_cfg.get("population_size")))
    elite_count = int(resolve_cli_or_config("elite_count", evolution_cfg.get("elite_count")))
    tasks_per_eval = int(resolve_cli_or_config("tasks_per_eval", evolution_cfg.get("tasks_per_evaluation")))
    archive_dir = str(resolve_cli_or_config("archive_dir", evolution_cfg.get("archive_dir")))
    max_repair_attempts = int(resolve_cli_or_config("max_repair_attempts", evolution_cfg.get("max_repair_attempts")))
    seed = int(resolve_cli_or_config("seed", evolution_cfg.get("seed")))
    final_eval_split = str(resolve_cli_or_config("final_eval_split", evaluation_cfg.get("final_eval_split")))
    final_eval_workers = int(resolve_cli_or_config("final_eval_workers", evaluation_cfg.get("workers")))
    protocol_timeout = int(resolve_cli_or_config("protocol_timeout", evolution_cfg.get("timeout_seconds")))

    worker_model = (
        resolve_cli_or_config("worker_model", evolution_cfg.get("worker_model"))
        or first_env(["EVO_WORKER_MODEL", "OPENAI_MODEL", "MODEL"])
        or "gpt-4o-mini"
    )
    architect_model = (
        resolve_cli_or_config("architect_model", evolution_cfg.get("meta_architect_model"))
        or first_env(["EVO_ARCHITECT_MODEL", "ARCHITECT_MODEL"])
        or "gpt-4o"
    )
    judge_model = (
        resolve_cli_or_config("judge_model", evaluation_cfg.get("judge_model"))
        or first_env(["EVO_JUDGE_MODEL", "OPENAI_JUDGE_MODEL", "JUDGE_MODEL"])
        or "gpt-5.1"
    )
    failure_classifier_model = (
        resolve_cli_or_config("failure_classifier_model", evolution_cfg.get("failure_classifier_model"))
        or first_env(["EVO_FAILURE_CLASSIFIER_MODEL", "FAILURE_CLASSIFIER_MODEL"])
        or "gpt-4o"
    )
    attention_drift_model = (
        resolve_cli_or_config("attention_drift_model", evaluation_cfg.get("attention_drift_model"))
        or first_env(["EVO_ATTENTION_DRIFT_MODEL", "ATTENTION_DRIFT_MODEL"])
        or "gpt-4o-mini"
    )
    attention_drift_sample_rate = float(
        resolve_cli_or_config("attention_drift_sample_rate", evaluation_cfg.get("attention_drift_sample_rate"))
    )
    attention_drift_sample_rate = max(0.0, min(1.0, attention_drift_sample_rate))
    disable_failure_classifier = bool(args.disable_failure_classifier)
    disable_flag = "--disable-failure-classifier"
    disable_flag_provided = any(
        token == disable_flag or token.startswith(f"{disable_flag}=")
        for token in sys.argv[1:]
    )
    if not disable_flag_provided:
        disable_failure_classifier = not bool(evolution_cfg.get("enable_failure_classifier", True))
    disable_attention_drift = bool(args.disable_attention_drift)
    disable_attention_flag = "--disable-attention-drift"
    disable_attention_flag_provided = any(
        token == disable_attention_flag or token.startswith(f"{disable_attention_flag}=")
        for token in sys.argv[1:]
    )
    if not disable_attention_flag_provided:
        disable_attention_drift = not bool(evaluation_cfg.get("enable_attention_drift", True))

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

    archive = ProtocolArchive(archive_dir)
    loader = ProtocolLoader(worker_client, worker_model, timeout_seconds=protocol_timeout)
    architect = MetaArchitect(architect_client, architect_model=architect_model)

    config = EvolutionConfig(
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        tasks_per_evaluation=tasks_per_eval,
        enable_failure_classifier=not disable_failure_classifier,
        failure_classifier_model=failure_classifier_model,
        enable_attention_drift=not disable_attention_drift,
        attention_drift_model=attention_drift_model,
        attention_drift_sample_rate=attention_drift_sample_rate,
        max_repair_attempts=max_repair_attempts,
        seed=seed,
    )

    engine = EvolutionEngine(
        benchmark_name=benchmark_name,
        data_path=data_path,
        split=split,
        protocol_loader=loader,
        meta_architect=architect,
        archive=archive,
        config=config,
        benchmark_kwargs={"judge_model": judge_model},
        judge_client=judge_client,
        failure_classifier_client=judge_client,
    )

    if config_doc:
        print(f"Loaded config: {args.config}")
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
            benchmark_name=benchmark_name,
            data_path=data_path,
            split=final_eval_split,
            output_path=args.final_eval_output,
            benchmark_kwargs=benchmark_kwargs,
            judge_client=judge_client,
            workers=final_eval_workers,
        )
        benchmark = get_benchmark(benchmark_name, **benchmark_kwargs)
        metrics = benchmark.get_metrics(records)
        summary["final_eval"] = {
            "split": final_eval_split,
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
