from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from .common import load_yaml, require_keys, require_positive_int
from .records import RunRecord


def build_model(model_config: dict[str, Any], attention_backend: str = "sdpa"):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(**model_config)
    config._attn_implementation = attention_backend
    return LlamaForCausalLM(config)


def verify_one_update(model, optimizer, tokens, autocast_dtype) -> tuple[float, bool]:
    import torch
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=autocast_dtype):
        loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()
    optimizer.step()
    changed = not torch.equal(before, next(model.parameters()).detach())
    return float(loss.detach()), changed


def validate_config(base: dict[str, Any]) -> None:
    require_keys(base, {"model_config", "mode", "sequence_lengths", "micro_batch_sizes", "warmup_steps",
                        "measurement_steps", "repeats", "seed", "learning_rate", "betas", "eps",
                        "weight_decay", "gradient_accumulation", "precision", "attention_backend",
                        "activation_checkpointing", "torch_compile", "allow_tf32"}, "training config")
    for key in ("warmup_steps", "measurement_steps", "repeats", "gradient_accumulation"):
        require_positive_int(base[key], key)
    if base["gradient_accumulation"] != 1:
        raise ValueError("stage-one baseline requires gradient_accumulation=1")
    if base["precision"] != "bf16_autocast":
        raise ValueError("stage-one baseline requires precision=bf16_autocast")
    if base["activation_checkpointing"] or base["torch_compile"]:
        raise ValueError("activation checkpointing and torch.compile must remain disabled")


def optimizer_state_dtypes(optimizer) -> list[str]:
    import torch
    return sorted({str(value.dtype) for state in optimizer.state.values() for value in state.values()
                   if isinstance(value, torch.Tensor)})


def run_one(base: dict[str, Any], model_cfg: dict[str, Any], sequence_length: int,
            batch_size: int, repeat: int, gpu: int, record: RunRecord) -> Path:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU execution is not a GPU acceptance test")
    torch.cuda.set_device(gpu)
    torch.manual_seed(int(base["seed"]) + repeat)
    torch.backends.cuda.matmul.allow_tf32 = bool(base["allow_tf32"])
    model = build_model(model_cfg, str(base["attention_backend"])).cuda().train()
    parameter_count = sum(p.numel() for p in model.parameters())
    kwargs = dict(lr=float(base["learning_rate"]), betas=tuple(base["betas"]), eps=float(base["eps"]),
                  weight_decay=float(base["weight_decay"]))
    optimizer_mode = "fused"
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        optimizer_mode = "standard"
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    generator = torch.Generator(device="cuda").manual_seed(int(base["seed"]) + repeat)
    tokens = torch.randint(0, int(model_cfg["vocab_size"]), (batch_size, sequence_length),
                           device="cuda", generator=generator)
    loss, changed = verify_one_update(model, optimizer, tokens, torch.bfloat16)
    if not math.isfinite(loss) or not changed:
        raise RuntimeError(f"correctness preflight failed: loss={loss}, parameter_changed={changed}")
    for _ in range(int(base["warmup_steps"])):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warmup_loss = model(input_ids=tokens, labels=tokens).loss
        warmup_loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(int(base["measurement_steps"]))]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(int(base["measurement_steps"]))]
    start = time.perf_counter()
    final_loss = None
    for step in range(int(base["measurement_steps"])):
        starts[step].record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final_loss = model(input_ids=tokens, labels=tokens).loss
        final_loss.backward()
        optimizer.step()
        ends[step].record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    steps = int(base["measurement_steps"])
    event_times = [begin.elapsed_time(end) for begin, end in zip(starts, ends)]
    metric = {
        "wall_time_s": elapsed, "mean_step_time_ms": elapsed * 1000 / steps,
        "input_tokens_per_s": batch_size * sequence_length * steps / elapsed,
        "input_tokens": batch_size * sequence_length * steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_loss": float(final_loss.detach()), "loss_finite": bool(torch.isfinite(final_loss)),
        "parameter_updated": changed,
        "cuda_step_time_ms_median": float(torch.tensor(event_times).median()),
        "cuda_step_time_ms_min": min(event_times),
        "cuda_step_time_ms_max": max(event_times),
        "timing_scope": "wall clock synchronized around full measurement interval; CUDA events per full step",
    }
    record.write_metrics([metric])
    props = torch.cuda.get_device_properties(gpu)
    gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in model.parameters()
                              if parameter.grad is not None})
    record.config.update(gpu_uuid=str(props.uuid), gpu_name=props.name, parameter_count=parameter_count,
                         parameter_dtype=str(next(model.parameters()).dtype), gradient_dtypes=gradient_dtypes,
                         optimizer_state_dtypes=optimizer_state_dtypes(optimizer),
                         autocast_dtype="torch.bfloat16", optimizer_mode=optimizer_mode,
                         labels="input_ids; LlamaForCausalLM performs the causal shift internally")
    record.finish("success")
    print(record.root)
    return record.root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-GPU Llama causal-LM training baseline")
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use the small model and one warmup/measurement step")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    args = parser.parse_args()
    base = load_yaml(args.config)
    if args.smoke:
        base.update(model_config="configs/model_small.yaml", warmup_steps=1, measurement_steps=1, repeats=1)
    validate_config(base)
    model_cfg = load_yaml(base["model_config"])
    sequences = [args.sequence_length] if args.sequence_length else base["sequence_lengths"]
    batches = [args.micro_batch_size] if args.micro_batch_size else base["micro_batch_sizes"]
    for sequence in sequences:
        for batch in batches:
            for repeat in range(int(base["repeats"])):
                run_config = {**base, "model_config": model_cfg, "sequence_length": int(sequence),
                              "micro_batch_size": int(batch), "repeat": repeat,
                              "gpu_logical_index": args.gpu}
                record = RunRecord("single_gpu_train", Path(args.results_dir), run_config, "train")
                try:
                    run_one(base, model_cfg, int(sequence), int(batch), repeat, args.gpu, record)
                except Exception as exc:
                    record.finish_exception(exc)
                    raise


if __name__ == "__main__":
    main()
