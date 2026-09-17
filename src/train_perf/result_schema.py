from __future__ import annotations

from typing import Any

VALID_STATUS = {"success", "failed", "skipped"}
REQUIRED = {"schema_version", "run_id", "created_at_utc", "experiment_type", "status", "config", "raw_logs"}


def validate_manifest(value: dict[str, Any]) -> None:
    missing = REQUIRED - value.keys()
    if missing:
        raise ValueError(f"manifest missing keys: {sorted(missing)}")
    if value["status"] not in VALID_STATUS:
        raise ValueError(f"invalid status: {value['status']}")
    if not isinstance(value["config"], dict):
        raise ValueError("config must be an object")
    if value["status"] != "success" and not value.get("failure_reason"):
        raise ValueError("non-success manifest requires failure_reason")
    if value["schema_version"] != 1:
        raise ValueError("unsupported schema_version")


def comparable_key(manifest: dict[str, Any]) -> tuple[Any, ...]:
    cfg = manifest["config"]
    if manifest["experiment_type"] == "nccl_all_reduce":
        return ("nccl", tuple(cfg["gpu_uuids"]), cfg["dtype"], cfg["in_place"])
    if manifest["experiment_type"] == "single_gpu_train":
        return (
            "train",
            cfg["model_config"],
            cfg["sequence_length"],
            cfg["micro_batch_size"],
            cfg["precision"],
        )
    return (manifest["experiment_type"],)
