"""Compiler library: stores evolved compilation strategies indexed by domain."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CompilerStrategy:
    """A single evolved compilation strategy."""

    key: str
    code: str
    domain: str
    data_structures: list[str] = field(default_factory=list)
    fitness: float = 0.0
    failure_modes_addressed: list[str] = field(default_factory=list)
    generation: int = 0
    parent_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompilerLibrary:
    """Persistent store for evolved compilation strategies.

    Strategies are saved as ``{key}.py`` (source code) and ``{key}_meta.json``
    (metadata) inside the library directory. The library accumulates the
    Meta-Agent's evolutionary discoveries -- e.g., physics needs graphs,
    legal needs FSMs, etc.
    """

    def __init__(self, library_dir: str = "compiler_library") -> None:
        self.dir = Path(library_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CompilerStrategy] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load all strategy metadata from disk."""

        for meta_path in self.dir.glob("*_meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                key = meta.get("key", meta_path.stem.replace("_meta", ""))
                code_path = self.dir / f"{key}.py"
                code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
                self._index[key] = CompilerStrategy(
                    key=key,
                    code=code,
                    domain=meta.get("domain", ""),
                    data_structures=meta.get("data_structures", []),
                    fitness=float(meta.get("fitness", 0.0)),
                    failure_modes_addressed=meta.get("failure_modes_addressed", []),
                    generation=int(meta.get("generation", 0)),
                    parent_key=meta.get("parent_key"),
                )
            except Exception:
                continue

    def save_strategy(
        self,
        key: str,
        code: str,
        domain: str,
        data_structures: list[str] | None = None,
        fitness: float = 0.0,
        failure_modes_addressed: list[str] | None = None,
        generation: int = 0,
        parent_key: str | None = None,
    ) -> CompilerStrategy:
        """Save a compilation strategy to the library."""

        strategy = CompilerStrategy(
            key=key,
            code=code,
            domain=domain,
            data_structures=data_structures or [],
            fitness=fitness,
            failure_modes_addressed=failure_modes_addressed or [],
            generation=generation,
            parent_key=parent_key,
        )

        code_path = self.dir / f"{key}.py"
        meta_path = self.dir / f"{key}_meta.json"
        code_path.write_text(code, encoding="utf-8")
        meta_path.write_text(
            json.dumps(strategy.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._index[key] = strategy
        return strategy

    def get_strategy(self, key: str) -> Optional[CompilerStrategy]:
        """Retrieve a strategy by key."""

        return self._index.get(key)

    def get_best_for_domain(self, domain: str, k: int = 3) -> list[CompilerStrategy]:
        """Retrieve top-k strategies for a given domain, sorted by fitness."""

        candidates = [
            s for s in self._index.values()
            if s.domain.lower() == domain.lower()
        ]
        candidates.sort(key=lambda s: s.fitness, reverse=True)
        return candidates[:k]

    def get_all_domains(self) -> list[str]:
        """List all unique domains in the library."""

        return sorted({s.domain for s in self._index.values() if s.domain})

    def get_all_strategies(self) -> list[CompilerStrategy]:
        """List all strategies sorted by fitness."""

        strategies = list(self._index.values())
        strategies.sort(key=lambda s: s.fitness, reverse=True)
        return strategies

    def summary(self) -> dict[str, Any]:
        """Return a summary of the library contents."""

        domains = self.get_all_domains()
        return {
            "total_strategies": len(self._index),
            "domains": domains,
            "strategies_per_domain": {
                domain: len(self.get_best_for_domain(domain, k=999))
                for domain in domains
            },
            "best_fitness": max(
                (s.fitness for s in self._index.values()),
                default=0.0,
            ),
        }
