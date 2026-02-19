"""Small helpers for loading key/value pairs from .env-style files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def load_env_file(path: str | None, override: bool = False) -> bool:
    """Load environment variables from a .env-style file.

    Supported line formats:
      - KEY=value
      - export KEY=value
      - optional single/double quotes around values
    """

    if not path:
        return False

    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return False

    loaded_any = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
        loaded_any = True

    return loaded_any


def first_env(keys: Iterable[str]) -> Optional[str]:
    """Return first non-empty env value from candidate keys."""

    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def env_float(keys: Iterable[str], default: float) -> float:
    """Parse first non-empty env value as float, otherwise return default."""

    raw = first_env(keys)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def env_int(keys: Iterable[str], default: int) -> int:
    """Parse first non-empty env value as int, otherwise return default."""

    raw = first_env(keys)
    if raw is None:
        return int(default)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)
