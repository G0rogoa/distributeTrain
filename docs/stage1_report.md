# Stage 1 Report

## Measured conclusions

Code delivery is complete for the local-first stage-one harness. Server acceptance is not yet complete. This document must only be populated from retained run artifacts under `results/`; configuration values and examples are not measurements.

## Current read-only preflight

- Connection alias: `a100-lab` (reachable on 2026-09-17).
- Conda environment: `zsq`, Python 3.11.15, PyTorch 2.8.0+cu128, PyTorch NCCL 2.27.3.
- CUDA toolkit: unavailable (`nvcc` was not found in the environment or `/usr/local/cuda`). PyTorch's bundled CUDA runtime does not replace the compiler needed by nccl-tests.
- Inventory: five 80 GB A100 GPUs. GPU 0 and GPU 1 are PCIe variants; GPU 2-4 report SXM4.
- Topology: GPU 0-1 report NV12; all other pairs report PIX. This mixed inventory requires topology-specific pair results and must not be inferred from model names alone.
- Resource state at preflight: GPU 0 had multiple compute allocations and is excluded. GPU 1-4 had only small existing allocations; availability was not inferred from utilization alone.

These observations are connection preflight only, not retained environment acceptance evidence. Run `collect-env` on the server to produce the authoritative artifact.

## NCCL AllReduce

Not yet measured. Official nccl-tests v2.17.5 source was pinned at commit `0bb567cc0218312bd1c6e5641f4ea68e7219e5bb`, but the server build stopped because `nvcc` is unavailable. The resource-authorized matrix has been revised to low-occupancy 2-GPU and 3-GPU combinations; correctness checks, repetitions, and plots remain pending. This is a scoped server acceptance result, not the original 2/4/5-GPU matrix.

## Single-GPU Transformer

Not yet measured. Smoke and formal sequence-length 512/2048 runs remain pending.

## Evidence and limitations

- Code-delivery status: complete. Local parser, schema, configuration, CLI, environment collector, and end-to-end runner checks pass. The local machine lacks PyTorch, so its model correctness test is explicitly skipped and must be rerun after checkout in `zsq`.
- Server-acceptance status: incomplete.
- Real-text validation is optional and has not been run.
- Profiling is separate from throughput measurement and has not been run.

## Next stage

After stage-one evidence is complete, use it to design a DDP scaling experiment. Do not infer training scaling from the AllReduce baseline alone.
