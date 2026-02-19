"""CL-bench implementation for the benchmark abstraction layer."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from benchmarks.base import BaseBenchmark, TaskRecord, register_benchmark


@register_benchmark("cl-bench")
class CLBenchmark(BaseBenchmark):
    """CL-bench adapter with deterministic stratified split and judge-based scoring."""

    CATEGORY_TO_MODE = {
        "Domain Knowledge Reasoning": "F1",
        "Rule System Application": "F2",
        "Procedural Task Execution": "F3",
        "Empirical Discovery & Simulation": "F4",
    }

    def __init__(
        self,
        split_ratio: dict[str, float] | None = None,
        split_seed: int = 42,
        judge_model: str = "gpt-5.1",
    ) -> None:
        self.split_ratio = split_ratio or {"train": 0.7, "val": 0.15, "test": 0.15}
        self.split_seed = split_seed
        self.judge_model = judge_model

    def load_tasks(self, data_path: str, split: str = "all") -> list[TaskRecord]:
        """Load CL-bench JSONL and optionally return one split."""

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        records: list[TaskRecord] = []
        with path.open("r", encoding="utf-8") as file_obj:
            for idx, line in enumerate(file_obj):
                line = line.strip()
                if not line:
                    continue

                raw = json.loads(line)
                messages = raw.get("messages", [])
                context, query = self._extract_context_and_query(messages)
                metadata = dict(raw.get("metadata", {}))
                category = metadata.get("context_category", "")

                records.append(
                    TaskRecord(
                        task_id=str(metadata.get("task_id", idx)),
                        context=context,
                        query=query,
                        messages_raw=messages,
                        rubrics=raw.get("rubrics", []),
                        metadata={
                            **metadata,
                            "idx": idx,
                            "gravity_type": self.CATEGORY_TO_MODE.get(category, "F1"),
                        },
                    )
                )

        if split == "all":
            return records

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: all, train, val, test")

        return self._apply_split(records, split)

    def evaluate(self, record: TaskRecord, judge_client=None) -> TaskRecord:
        """Evaluate one record with CL-bench rubric logic from eval.py."""

        if not record.model_output or not record.model_output.strip():
            record.score = 0
            record.eval_detail = {"reason": "Empty model output"}
            record.failure_mode = record.metadata.get("gravity_type")
            return record

        if judge_client is None:
            record.score = 0
            record.eval_detail = {"reason": "judge_client is required for rubric grading"}
            record.failure_mode = record.metadata.get("gravity_type")
            return record

        from eval import build_rubrics_text, call_judge_api

        rubrics_text = build_rubrics_text(record.rubrics)
        raw_judge = call_judge_api(judge_client, self.judge_model, rubrics_text, record.model_output)

        if not raw_judge:
            record.score = 0
            record.eval_detail = {"reason": "Judge API failed"}
            record.failure_mode = record.metadata.get("gravity_type")
            return record

        try:
            parsed = json.loads(raw_judge)
            score = int(parsed.get("Overall Score", 0))
            record.score = 1 if score == 1 else 0
            record.eval_detail = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            record.score = 0
            record.eval_detail = {"reason": "Judge parse failure", "raw": raw_judge[:500]}

        if record.score == 0:
            record.failure_mode = record.metadata.get("gravity_type")

        return record

    def _extract_context_and_query(self, messages: list[dict[str, Any]]) -> tuple[str, str]:
        """Extract context and last user query from chat-style messages."""

        context_parts: list[str] = []
        query = ""
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role == "system" and content:
                context_parts.append(content)
            elif role == "user" and content:
                query = content

        context = "\n\n".join(context_parts).strip()
        return context, query.strip()

    def _apply_split(self, records: list[TaskRecord], split: str) -> list[TaskRecord]:
        """Deterministic stratified split by context_category without extra dependencies."""

        groups: dict[str, list[TaskRecord]] = {}
        for record in records:
            category = str(record.metadata.get("context_category", "unknown"))
            groups.setdefault(category, []).append(record)

        rng = random.Random(self.split_seed)
        split_records: dict[str, list[TaskRecord]] = {"train": [], "val": [], "test": []}

        train_ratio = float(self.split_ratio.get("train", 0.7))
        val_ratio = float(self.split_ratio.get("val", 0.15))
        test_ratio = float(self.split_ratio.get("test", 0.15))
        total_ratio = train_ratio + val_ratio + test_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            train_ratio, val_ratio, test_ratio = (
                train_ratio / total_ratio,
                val_ratio / total_ratio,
                test_ratio / total_ratio,
            )

        for category_records in groups.values():
            local_records = list(category_records)
            rng.shuffle(local_records)
            total = len(local_records)

            train_end = int(round(total * train_ratio))
            val_end = train_end + int(round(total * val_ratio))

            train_slice = local_records[:train_end]
            val_slice = local_records[train_end:val_end]
            test_slice = local_records[val_end:]

            # Ensure no split is empty on tiny categories when possible.
            if total >= 3:
                if not train_slice:
                    train_slice.append(test_slice.pop() if test_slice else val_slice.pop())
                if not val_slice:
                    val_slice.append(test_slice.pop() if test_slice else train_slice.pop())
                if not test_slice:
                    test_slice.append(val_slice.pop() if val_slice else train_slice.pop())

            split_records["train"].extend(train_slice)
            split_records["val"].extend(val_slice)
            split_records["test"].extend(test_slice)

        return split_records[split]
