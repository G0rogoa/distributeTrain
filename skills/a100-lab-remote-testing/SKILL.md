---
name: a100-lab-remote-testing
description: Connect to the private a100-lab server and validate distributeTrain environment, NCCL baselines, and single-GPU training. Use for this repository's remote GPU testing, not deployment or unrelated DistServe work.
---

# A100 Lab Remote Testing

Use this skill for remote validation of the `distributeTrain` repository. It records connection mechanics and safety constraints; it does not grant permission to occupy shared GPUs.

## Connection and project

- Connect only through SSH alias `a100-lab`; keep host, hop, and credentials in SSH configuration.
- Local checkout is the source of truth. The remote project root is `/data/zhangshenqiang/distributeTrain`.
- Run Python commands non-interactively with `/opt/anaconda3/bin/conda run -n zsq ...`.
- `zhangshenqiang` is the project owner's personal account and `zsq` is the owner's personal Conda environment. Use that environment directly; do not create or clone another Conda environment.
- Keep all project code, build trees, downloaded source, and generated artifacts under directories owned by `zhangshenqiang`, such as the project root or a dedicated subdirectory beneath the `zsq` environment. Do not write into another user's directory.
- The upstream repository is `https://github.com/G0rogoa/distributeTrain.git`.
- Do not expose keys, credentials, full environments, unnecessary process arguments, or other users' data.

Read-only connection checks:

```bash
ssh a100-lab date --iso-8601=seconds
ssh a100-lab /opt/anaconda3/bin/conda run -n zsq python --version
ssh a100-lab nvidia-smi topo -m
```

## Source delivery

Develop and test source in the local checkout. Push reviewed commits to `https://github.com/G0rogoa/distributeTrain.git`, then clone or fast-forward the remote project checkout to an explicitly reported commit. Do not edit remote source independently and do not use `scp` to overlay a working tree. Use file transfer only for sanitized experiment artifacts when the active task requires it. Never synchronize credentials, caches, raw unrelated results, or another project's runtime state.

## GPU launch gate

Before every GPU launch, query both aggregate state and compute allocations:

```bash
ssh a100-lab nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
ssh a100-lab nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits
```

- GPU 0 is excluded from opportunistic work. A required five-GPU measurement must wait for explicit authorization and an idle window.
- Low utilization is not proof of availability. Do not inspect or terminate another user's process.
- Bind any services to `127.0.0.1` and manage only processes launched by the active test.
- Use only the GPUs required by the test. Stop on correctness errors, nonzero exit, or timeout, and clean up only owned child processes.
- Formal measurements must not overlap with other benchmarks on the same host.

## Repository validation

Install project dependencies into the existing `zsq` environment without creating a new environment or changing drivers or global CUDA/NCCL. Use the repository README for current commands. Preserve raw stdout/stderr and manifests under `results/<run_id>/`; do not present examples or preflight observations as benchmark measurements.

For NCCL, use the pinned official `nccl-tests` build and inspect its actual help before relying on flags. For training, keep profiler runs separate from throughput measurements. Report code delivery and server acceptance as separate statuses.

## Reporting

Report source revision and dirty state, synchronization, GPU UUIDs, test status, retained evidence paths, unavailable tooling, failures, and cleanup. Never call a waiting or skipped run complete, and do not extrapolate AllReduce bandwidth into training throughput.
