from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.archive import ProtocolArchive
from core.base_tdg_protocol import BaseTDGCompiler
from core.evolution_loop import EvolutionConfig, EvolutionEngine


class _DummyLoader:
    def load_from_code(self, code: str):
        return object(), None

    def run_with_timeout(self, protocol, **kwargs):  # pragma: no cover - not used in these tests
        return None


class _DummyMetaArchitect:
    def __init__(self) -> None:
        self.calls = 0

    def mutate(self, **kwargs):
        self.calls += 1
        parent_code = str(kwargs.get("parent_code", ""))
        mutation_attempt = int(kwargs.get("mutation_attempt", 0))
        code = f"{parent_code}\n# child-{mutation_attempt}\n"
        return object(), code


def _build_engine(tmp_dir: str, *, config: EvolutionConfig | None = None) -> EvolutionEngine:
    archive = ProtocolArchive(str(Path(tmp_dir) / "archive"))
    return EvolutionEngine(
        benchmark_name="cl-bench",
        data_path="data/CL-bench.jsonl",
        split="train",
        protocol_loader=_DummyLoader(),
        meta_architect=_DummyMetaArchitect(),
        archive=archive,
        config=config or EvolutionConfig(),
        benchmark_kwargs={"split_seed": 42},
    )


class _AlwaysPassTDGCompiler(BaseTDGCompiler):
    def compile_tests(self, context: str, query: str) -> str:
        return (
            "def test_non_empty(answer):\n"
            "    assert isinstance(answer, str)\n"
        )

    def generate_answer(self, context: str, query: str, messages_raw: list = None) -> str:
        return "normal answer"


class EvolutionStabilityTests(unittest.TestCase):
    def test_sample_tasks_uses_overlap_and_calibration(self) -> None:
        with TemporaryDirectory() as td:
            config = EvolutionConfig(
                tasks_per_evaluation=20,
                task_overlap_ratio=0.5,
                calibration_tasks_per_evaluation=5,
            )
            engine = _build_engine(td, config=config)

            sampled_gen1 = engine._sample_tasks(20, generation=1)
            meta_gen1 = dict(engine._last_task_sample_meta)
            sampled_gen2 = engine._sample_tasks(20, generation=2)
            meta_gen2 = dict(engine._last_task_sample_meta)

            self.assertEqual(len(sampled_gen1), 20)
            self.assertEqual(len(sampled_gen2), 20)
            self.assertEqual(meta_gen1["calibration_count"], 5)
            self.assertEqual(meta_gen2["calibration_count"], 5)
            self.assertGreaterEqual(meta_gen2["overlap_count"], 5)
            calibration_ids = set(engine._calibration_task_ids)
            self.assertTrue(calibration_ids.issubset(set(meta_gen1["task_ids"])))
            self.assertTrue(calibration_ids.issubset(set(meta_gen2["task_ids"])))

    def test_selection_score_stabilization_blends_with_history(self) -> None:
        with TemporaryDirectory() as td:
            config = EvolutionConfig(elite_score_current_weight=0.7)
            engine = _build_engine(td, config=config)

            first = engine._stabilize_selection_score("sha123", 0.6, generation=1)
            second = engine._stabilize_selection_score("sha123", 0.2, generation=2)
            _ = engine._stabilize_selection_score("sha123", 0.2, generation=2)

            self.assertAlmostEqual(first, 0.6, places=8)
            self.assertAlmostEqual(second, 0.32, places=8)
            self.assertEqual(len(engine._selection_history_by_sha["sha123"]), 2)

    def test_init_population_diversify_creates_multiple_shas(self) -> None:
        with TemporaryDirectory() as td:
            config = EvolutionConfig(
                init_population_diversify=True,
                init_population_mutation_attempts=1,
            )
            engine = _build_engine(td, config=config)
            initial_code = "class Seed:\n    pass\n"
            init_sha = engine.archive.save(initial_code, generation=0)

            population = engine._build_initial_population(
                initial_code=initial_code,
                initial_protocol=object(),
                init_sha=init_sha,
                init_population_size=4,
            )

            self.assertEqual(len(population), 4)
            self.assertGreater(len({str(item["sha"]) for item in population}), 1)

    def test_adversarial_validation_surfaces_weak_tests(self) -> None:
        compiler = _AlwaysPassTDGCompiler(llm_client=object(), model_name="dummy")
        result = compiler.run(context="ctx", query="question", max_retries=0)

        self.assertTrue(result.metadata["tests_compiled"])
        self.assertEqual(result.metadata["adversarial_total_tests"], 1)
        self.assertAlmostEqual(result.metadata["adversarial_test_pass_rate"], 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
