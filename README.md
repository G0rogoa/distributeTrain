# distributeTrain Stage 1

Reproducible environment capture, NCCL AllReduce baselines, and a single-GPU BF16 Llama training baseline. Stage 1 does not implement custom CUDA operators or distributed training.

## Setup

Use a CUDA-capable environment with a compatible PyTorch installation. The A100 lab uses the owner's existing `zsq` Conda environment; do not create or clone another environment:

```bash
/opt/anaconda3/bin/conda run -n zsq python -m pip install -e '.[report,test]'
```

Dependencies are installed into the selected environment; the scripts do not modify the driver, CUDA toolkit, or system NCCL.

The development workflow is local-first: merge or commit reviewed code locally, push it to `https://github.com/G0rogoa/distributeTrain.git`, and check out that exact commit on the lab server. Do not edit or overwrite remote source independently.

## Environment evidence

```bash
collect-env --results-dir results
```

The collector records each command, return code, stdout, and stderr. Missing optional tools are marked unavailable. Its environment-variable capture is allowlisted.

## NCCL baseline

Review GPU availability and edit only the GPU sets in `configs/nccl_baseline.yaml` as needed. Build the pinned official nccl-tests revision and inspect the installed binary's help before a formal run:

```bash
bash scripts/build_nccl_tests.sh
third_party/nccl-tests/build/all_reduce_perf --help
run-nccl-bench --config configs/nccl_baseline.yaml --results-dir results
run-nccl-bench --preflight --config configs/nccl_baseline.yaml --results-dir results
```

The checked-in matrix uses low-occupancy 2-GPU and 3-GPU combinations and excludes GPU 0. Recheck availability immediately before every run. The runner uses one process controlling N GPUs; this is a communication baseline, not one-process-per-GPU DDP.

## Training baseline

Start with a small correctness smoke run, then run the fixed formal matrix. Change GPU, sequence length, batch size, and result directory through CLI/configuration rather than source edits:

```bash
run-train-bench --smoke --gpu 1 --sequence-length 512 --micro-batch-size 1
run-train-bench --config configs/train_baseline.yaml --gpu 1 --results-dir results
profile-train --gpu 1 --sequence-length 512 --steps 3 --results-dir results
```

The measured step includes zeroing gradients, forward/loss, backward, and optimizer update. Throughput counts input tokens. Parameters and optimizer state remain FP32 while forward/backward use BF16 autocast. Hugging Face performs the causal-label shift internally. Profiling is a separate short diagnostic and its trace must not be mixed into throughput measurements.

## Tests and summary

```bash
python -m unittest discover -s tests -v
python -m pytest
summarize-results --results-dir results --output results/summary.csv
```

The summary command also writes NCCL latency and bus-bandwidth figures when matplotlib is installed. If pytest is unavailable, report that explicitly and retain the unittest result. CPU tests validate parsers and schemas but do not count as GPU acceptance. Results are immutable per-run directories; failed and incomparable runs are excluded from performance aggregation.

See `docs/stage1_report.md` for measured findings and remaining acceptance work.
