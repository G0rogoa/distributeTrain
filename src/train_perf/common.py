from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{kind}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys - value.keys()
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")


def require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_state() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}
