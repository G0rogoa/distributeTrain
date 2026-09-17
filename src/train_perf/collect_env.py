from __future__ import annotations

import argparse
import os
import json
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .records import RunRecord

ALLOWED_ENV = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_ALGO", "NCCL_PROTO",
)

COMMANDS = {
    "gpu_inventory": ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,mig.mode.current,power.limit,clocks.current.sm", "--format=csv,noheader"],
    "gpu_compute_capability": ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"],
    "gpu_processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
    "topology": ["nvidia-smi", "topo", "-m"],
    "nvlink": ["nvidia-smi", "nvlink", "--status"],
    "os_release": ["uname", "-a"],
    "linux_distribution": ["cat", "/etc/os-release"],
    "cpu_numa": ["lscpu"],
    "memory": ["free", "-h"],
    "python": ["python", "--version"],
    "nvcc": ["nvcc", "--version"],
    "compiler": ["c++", "--version"],
    "nsys": ["nsys", "--version"],
    "ncu": ["ncu", "--version"],
    "nccl_tests_commit": ["git", "-C", "third_party/nccl-tests", "rev-parse", "HEAD"],
    "nccl_tests_linked_libraries": ["ldd", "third_party/nccl-tests/build/all_reduce_perf"],
}


def capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        return {"command": shlex.join(command), "status": "available" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": shlex.join(command), "status": "unavailable", "returncode": None,
                "stdout": "", "stderr": str(exc)}


def collect() -> dict[str, Any]:
    values = {name: capture(command) for name, command in COMMANDS.items()}
    torch_code = (
        "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
        "'nccl':torch.cuda.nccl.version() if torch.cuda.is_available() else None,"
        "'cuda_available':torch.cuda.is_available(),'cudnn':torch.backends.cudnn.version(),"
        "'devices':[{'logical_index':i,'name':torch.cuda.get_device_name(i),"
        "'uuid':str(torch.cuda.get_device_properties(i).uuid),'capability':torch.cuda.get_device_capability(i)}"
        " for i in range(torch.cuda.device_count())]}))"
    )
    values["pytorch"] = capture(["python", "-c", torch_code])
    values["host"] = {"platform": platform.platform(), "machine": platform.machine()}
    values["allowed_environment"] = {key: os.environ[key] for key in ALLOWED_ENV if key in os.environ}
    values["note"] = "The CUDA version shown by nvidia-smi is driver capability, not the PyTorch runtime or nvcc toolkit version."
    return values


def render_summary(values: dict[str, Any]) -> str:
    lines = ["# Environment Summary", "", f"Host platform: `{values['host']['platform']}`", ""]
    for name in ("gpu_inventory", "topology", "nvlink", "python", "pytorch", "nvcc", "compiler", "nsys", "ncu"):
        item = values[name]
        lines.extend([f"## {name.replace('_', ' ').title()}", "", f"Status: `{item['status']}`", ""])
        output = item["stdout"].strip() or item["stderr"].strip() or "No output"
        lines.extend(["```text", output, "```", ""])
    lines.append(values["note"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect read-only GPU, topology, host, and software facts")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    record = RunRecord("environment", Path(args.results_dir), {}, "environment")
    values = collect()
    (record.root / "environment.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (record.root / "summary.md").write_text(render_summary(values), encoding="utf-8")
    record.finish("success", raw_logs=["environment.json", "summary.md"])
    print(record.root)


if __name__ == "__main__":
    main()
