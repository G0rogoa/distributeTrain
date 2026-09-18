from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .common import load_yaml, require_keys, require_positive_int
from .records import RunRecord

ROW = re.compile(
    r"^\s*(?P<size>\d+)\s+(?P<count>\d+)\s+(?P<type>\S+)\s+(?P<redop>\S+)\s+(?P<root>-?\d+)\s+"
    r"(?P<out_time>[\d.]+)\s+(?P<out_algbw>[\d.]+)\s+(?P<out_busbw>[\d.]+)\s+(?P<out_err>\S+)\s+"
    r"(?P<in_time>[\d.]+)\s+(?P<in_algbw>[\d.]+)\s+(?P<in_busbw>[\d.]+)\s+(?P<in_err>\S+)\s*$"
)


def _number(value: str) -> float:
    if value.lower() in {"n/a", "na", "-"}:
        raise ValueError(f"missing numeric NCCL value: {value}")
    return float(value)


def parse_nccl_output(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        item = match.groupdict()
        for in_place, prefix in ((False, "out"), (True, "in")):
            errors = _number(item[f"{prefix}_err"])
            rows.append(
                {
                    "bytes": int(item["size"]),
                    "count": int(item["count"]),
                    "dtype": item["type"],
                    "reduction": item["redop"],
                    "in_place": in_place,
                    "time_us": _number(item[f"{prefix}_time"]),
                    "algbw_gbps": _number(item[f"{prefix}_algbw"]),
                    "busbw_gbps": _number(item[f"{prefix}_busbw"]),
                    "errors": errors,
                }
            )
    if not rows:
        raise ValueError("NCCL output contains no complete result rows")
    if any(row["errors"] != 0 for row in rows):
        raise ValueError("NCCL correctness check reported non-zero errors")
    return rows


def query_gpu_uuids(indices: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    mapping = {int(a.strip()): b.strip() for a, b in (line.split(",", 1) for line in result.stdout.splitlines())}
    return [mapping[index] for index in indices]


def validate_config(config: dict[str, Any]) -> None:
    require_keys(config, {"binary", "gpu_sets", "types", "min_bytes", "max_bytes", "step_factor",
                          "warmup_iterations", "iterations", "repeats", "timeout_seconds"}, "NCCL config")
    require_positive_int(config["warmup_iterations"], "warmup_iterations")
    require_positive_int(config["iterations"], "iterations")
    require_positive_int(config["repeats"], "repeats")
    require_positive_int(config["timeout_seconds"], "timeout_seconds")
    if not config["gpu_sets"] or any(len(set(group)) < 2 for group in config["gpu_sets"]):
        raise ValueError("every NCCL gpu_set must contain at least two unique GPU indices")
    if not config["types"]:
        raise ValueError("types must not be empty")


def inspect_help(binary: Path) -> str:
    result = subprocess.run([str(binary), "--help"], text=True, capture_output=True, check=False, timeout=30)
    output = result.stdout + result.stderr
    required = ("--iters", "--warmup_iters", "--check", "--datatype")
    missing = [flag for flag in required if flag not in output]
    if missing:
        raise RuntimeError(f"nccl-tests help is missing required options: {missing}")
    return output


def execute(config_path: str, results_dir: str, *, preflight_only: bool = False) -> bool:
    config = load_yaml(config_path)
    validate_config(config)
    binary = Path(config["binary"])
    if not binary.is_file():
        raise FileNotFoundError(f"nccl-tests binary not found: {binary}")
    help_output = inspect_help(binary)
    failures = 0
    for gpu_set in config["gpu_sets"]:
        indices = [int(value) for value in gpu_set]
        uuids = query_gpu_uuids(indices)
        for dtype in config["types"]:
            for repeat in range(int(config["repeats"])):
                if preflight_only and repeat > 0:
                    continue
                run_config = {**config, "gpu_indices": indices, "gpu_uuids": uuids, "dtype": dtype,
                              "repeat": repeat, "in_place": "separate_metrics", "preflight": preflight_only}
                command = [
                    str(binary), "-b", str(config["min_bytes"]),
                    "-e", str(config.get("preflight_max_bytes", "1M") if preflight_only else config["max_bytes"]),
                    "-f", str(config["step_factor"]), "-g", str(len(indices)), "-d", str(dtype),
                    "-w", str(1 if preflight_only else config["warmup_iterations"]),
                    "-n", str(5 if preflight_only else config["iterations"]), "-c", "1",
                ]
                record = RunRecord("nccl_all_reduce", Path(results_dir), run_config, "nccl", command)
                record.write_log("binary-help.log", help_output)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, indices))
                nccl_lib = binary.parent.parent / ".nccl-prefix" / "lib"
                if nccl_lib.is_dir():
                    existing = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = f"{nccl_lib.resolve()}:{existing}" if existing else str(nccl_lib.resolve())
                result = None
                try:
                    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False,
                                            timeout=int(config["timeout_seconds"]))
                    stdout_log = record.write_log("stdout.log", result.stdout)
                    stderr_log = record.write_log("stderr.log", result.stderr)
                    if result.returncode:
                        raise RuntimeError(f"nccl-tests exited with {result.returncode}")
                    metrics = parse_nccl_output(result.stdout)
                    record.write_metrics(metrics)
                    record.finish("success", exit_code=0,
                                  raw_logs=["raw/binary-help.log", stdout_log, stderr_log])
                except Exception as exc:
                    failures += 1
                    record.finish_exception(exc, exit_code=getattr(result, "returncode", None))
    return failures == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible nccl-tests AllReduce baselines")
    parser.add_argument("--config", default="configs/nccl_baseline.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--preflight", action="store_true", help="Run one short correctness pass per combination")
    args = parser.parse_args()
    if not execute(args.config, args.results_dir, preflight_only=args.preflight):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
