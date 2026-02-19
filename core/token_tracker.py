"""Global token usage tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class TokenTracker:
    """Thread-safe token usage aggregator."""

    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for one API call."""

        with self._lock:
            entry = self.usage.setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                },
            )
            entry["prompt_tokens"] += int(prompt_tokens)
            entry["completion_tokens"] += int(completion_tokens)
            entry["total_tokens"] += int(prompt_tokens) + int(completion_tokens)
            entry["calls"] += 1

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of all token usage counters."""

        with self._lock:
            total = sum(entry["total_tokens"] for entry in self.usage.values())
            return {"per_model": dict(self.usage), "total_tokens": total}

    def reset(self) -> None:
        """Reset all usage counters."""

        with self._lock:
            self.usage.clear()


TRACKER = TokenTracker()
