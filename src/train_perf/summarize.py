from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(results_dir: str, output: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in Path(results_dir).glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "success" or manifest["experiment_type"] not in {"single_gpu_train", "nccl_all_reduce"}:
            continue
        metrics_path = path.parent / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            metric = json.loads(line)
            cfg = manifest["config"]
            if manifest["experiment_type"] == "single_gpu_train":
                key = ("train", cfg["model_config"], cfg["sequence_length"], cfg["micro_batch_size"],
                       cfg["precision"], cfg["gpu_uuid"])
                groups[key].append(metric)
            else:
                key = ("nccl", tuple(cfg["gpu_uuids"]), cfg["dtype"], metric["in_place"], metric["bytes"])
                groups[key].append(metric)
    rows = []
    for key, values in groups.items():
        metric_name = "input_tokens_per_s" if key[0] == "train" else "busbw_gbps"
        numbers = [float(value[metric_name]) for value in values]
        mean = statistics.fmean(numbers)
        rows.append({"group": repr(key), "metric": metric_name, "count": len(numbers),
                     "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers),
                     "cv_percent": statistics.stdev(numbers) / mean * 100 if len(numbers) > 1 and mean else None,
                     "review_variance": len(numbers) > 1 and mean != 0 and statistics.stdev(numbers) / mean > 0.05})
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["group", "metric", "count", "median", "min", "max", "cv_percent", "review_variance"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_results(results_dir: str, figures_dir: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    nccl: dict[tuple[str, bool], list[tuple[int, float, float]]] = defaultdict(list)
    for path in Path(results_dir).glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "success" or manifest.get("experiment_type") != "nccl_all_reduce":
            continue
        metrics_path = path.parent / "metrics.jsonl"
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            metric = json.loads(line)
            label = f"{len(manifest['config']['gpu_uuids'])} GPU"
            nccl[(label, metric["in_place"])].append((metric["bytes"], metric["time_us"], metric["busbw_gbps"]))
    if not nccl:
        return []
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for column, ylabel, filename in ((1, "Latency (us)", "nccl_latency.png"),
                                     (2, "busbw (GB/s)", "nccl_busbw.png")):
        fig, axis = plt.subplots(figsize=(8, 5))
        for (label, in_place), values in sorted(nccl.items()):
            by_size: dict[int, list[float]] = defaultdict(list)
            for value in values:
                by_size[value[0]].append(value[column])
            xs = sorted(by_size)
            axis.plot(xs, [statistics.median(by_size[x]) for x in xs], marker=".",
                      label=f"{label}, {'in' if in_place else 'out'}-of-place")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Message bytes per GPU")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = target / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize comparable successful benchmark runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    print(json.dumps({"summary": summarize(args.results_dir, args.output),
                      "figures": plot_results(args.results_dir, args.figures_dir)}, indent=2))


if __name__ == "__main__":
    main()
