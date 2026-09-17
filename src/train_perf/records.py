from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import git_state, run_id, utc_now, write_json
from .result_schema import validate_manifest


@dataclass
class RunRecord:
    experiment_type: str
    results_dir: Path
    config: dict[str, Any]
    kind: str
    command: list[str] = field(default_factory=list)
    run_id: str = field(init=False)
    root: Path = field(init=False)
    raw: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_id = run_id(self.kind)
        self.root = self.results_dir / self.run_id
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True)

    def write_log(self, name: str, content: str) -> str:
        path = self.raw / name
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.root))

    def write_metrics(self, rows: list[dict[str, Any]]) -> str:
        path = self.root / "metrics.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path.name

    def finish(
        self,
        status: str,
        *,
        failure_reason: str | None = None,
        exit_code: int | None = None,
        raw_logs: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at_utc": utc_now(),
            "experiment_type": self.experiment_type,
            "status": status,
            "failure_reason": failure_reason,
            "exit_code": exit_code,
            "config": self.config,
            "command": self.command,
            "git": git_state(),
            "raw_logs": raw_logs or [],
        }
        if extra:
            manifest.update(extra)
        validate_manifest(manifest)
        write_json(self.root / "manifest.json", manifest)
        return manifest

    def finish_exception(self, exc: BaseException, *, exit_code: int | None = None) -> dict[str, Any]:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log = self.write_log("exception.log", trace)
        return self.finish(
            "failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            exit_code=exit_code,
            raw_logs=[log],
        )
