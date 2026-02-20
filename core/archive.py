"""SHA-indexed protocol archive with lineage metadata."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Optional


class ProtocolArchive:
    """Store protocol source code and metadata for each generation."""

    def __init__(self, archive_dir: str = "archive") -> None:
        self.dir = Path(archive_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for meta_file in self.dir.glob("protocol_*_meta.json"):
            try:
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                sha = str(metadata.get("sha", ""))
                if sha:
                    self.db[sha] = metadata
            except json.JSONDecodeError:
                continue

    def save(
        self,
        code: str,
        generation: int,
        parent_sha: Optional[str] = None,
        score: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save protocol code and metadata, then return its short SHA."""

        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
        code_path = self.dir / f"protocol_{sha}.py"
        meta_path = self.dir / f"protocol_{sha}_meta.json"

        if sha in self.db:
            metadata = self.db[sha]
            metadata["generation"] = int(min(int(metadata.get("generation", generation)), int(generation)))
            if parent_sha and not metadata.get("parent_sha"):
                metadata["parent_sha"] = parent_sha
            if score is not None:
                metadata["score"] = float(score)
            seen = metadata.setdefault("seen_generations", [])
            if int(generation) not in seen:
                seen.append(int(generation))
                seen.sort()
            if extra:
                metadata.update(extra)
            if not code_path.exists():
                code_path.write_text(code, encoding="utf-8")
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return sha

        code_path.write_text(code, encoding="utf-8")
        metadata = {
            "sha": sha,
            "generation": int(generation),
            "parent_sha": parent_sha,
            "score": score,
            "visit_count": 0,
            "seen_generations": [int(generation)],
        }
        if extra:
            metadata.update(extra)

        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.db[sha] = metadata
        return sha

    def get_code(self, sha: str) -> str:
        """Load archived source code by SHA."""

        path = self.dir / f"protocol_{sha}.py"
        return path.read_text(encoding="utf-8")

    def get_meta(self, sha: str) -> dict[str, Any]:
        """Get metadata dictionary for a SHA."""

        if sha not in self.db:
            raise KeyError(f"Unknown protocol SHA: {sha}")
        return self.db[sha]

    def update_score(self, sha: str, score: float) -> None:
        """Update protocol score metadata."""

        self.db[sha]["score"] = float(score)
        self._write_meta(sha)

    def update_evaluation(
        self,
        sha: str,
        generation: int,
        score: float,
        failure_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update evaluation metadata, including per-generation failure summary."""

        if sha not in self.db:
            raise KeyError(f"Unknown protocol SHA: {sha}")

        meta = self.db[sha]
        score_value = float(score)
        meta["score"] = score_value
        meta["eval_count"] = int(meta.get("eval_count", 0)) + 1
        meta["last_evaluated_generation"] = int(generation)
        meta["best_score"] = max(float(meta.get("best_score", score_value)), score_value)

        if failure_summary is not None:
            meta["last_failure_summary"] = failure_summary
            history = meta.setdefault("evaluation_history", [])
            entry = {
                "generation": int(generation),
                "score": score_value,
                "num_tasks": int(failure_summary.get("num_tasks", 0)),
                "num_failures": int(failure_summary.get("num_failures", 0)),
                "failure_rate": float(failure_summary.get("failure_rate", 0.0)),
                "token_efficiency": float(failure_summary.get("token_efficiency", 0.0)),
                "attention_drift_mean": failure_summary.get("attention_drift_mean"),
                "attention_drift_high_rate": failure_summary.get("attention_drift_high_rate"),
                "attention_drift_measured_tasks": int(failure_summary.get("attention_drift_measured_tasks", 0)),
                "failure_mode_counts": dict(failure_summary.get("failure_mode_counts", {})),
                "top_root_causes": list(failure_summary.get("top_root_causes", [])),
            }
            existing_idx = next(
                (idx for idx, item in enumerate(history) if int(item.get("generation", -1)) == int(generation)),
                None,
            )
            if existing_idx is not None:
                history[existing_idx] = entry
            else:
                history.append(entry)
                history.sort(key=lambda item: int(item.get("generation", 0)))
            if len(history) > 30:
                del history[:-30]

        self._write_meta(sha)

    def increment_visit(self, sha: str) -> None:
        """Increment protocol visit count for selection penalty."""

        self.db[sha]["visit_count"] = int(self.db[sha].get("visit_count", 0)) + 1
        self._write_meta(sha)

    def select(self, k: int = 5, tau: float = 0.5, alpha: float = 0.5) -> list[str]:
        """Sample protocols by softmax(score - visit_penalty)."""

        candidates = [(sha, meta) for sha, meta in self.db.items() if meta.get("score") is not None]
        if not candidates:
            return []

        baseline = min(float(meta["score"]) for _, meta in candidates)
        logits: list[float] = []
        shas: list[str] = []

        for sha, meta in candidates:
            reward = float(meta["score"]) - baseline
            normalized = 1.0 / (1.0 + math.exp(-reward))
            penalty = alpha * math.log1p(float(meta.get("visit_count", 0)))
            logits.append((normalized - penalty) / max(tau, 1e-6))
            shas.append(sha)

        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        selected = random.choices(shas, weights=weights, k=min(k, len(shas)))

        for sha in selected:
            self.increment_visit(sha)

        return selected

    def top_k(self, k: int) -> list[str]:
        """Return top-k SHAs ranked by score descending."""

        ranked = sorted(
            ((sha, meta) for sha, meta in self.db.items() if meta.get("score") is not None),
            key=lambda item: float(item[1]["score"]),
            reverse=True,
        )
        return [sha for sha, _ in ranked[:k]]

    def _write_meta(self, sha: str) -> None:
        path = self.dir / f"protocol_{sha}_meta.json"
        path.write_text(json.dumps(self.db[sha], ensure_ascii=False, indent=2), encoding="utf-8")
