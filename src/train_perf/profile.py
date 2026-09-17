from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_yaml
from .records import RunRecord
from .train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short, separate PyTorch CUDA profiler diagnostic")
    parser.add_argument("--model-config", default="configs/model_small.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the profiler diagnostic")
    torch.cuda.set_device(args.gpu)
    config = vars(args).copy()
    record = RunRecord("training_profile", Path(args.results_dir), config, "profile")
    try:
        model_cfg = load_yaml(args.model_config)
        model = build_model(model_cfg, "sdpa").cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        tokens = torch.randint(0, model_cfg["vocab_size"],
                               (args.micro_batch_size, args.sequence_length), device="cuda")
        trace = record.raw / "trace.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for _ in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=tokens, labels=tokens).loss
                loss.backward()
                optimizer.step()
                profiler.step()
        profiler.export_chrome_trace(str(trace))
        kernels = []
        for event in profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25).splitlines():
            if event.strip():
                kernels.append(event)
        (record.root / "operators.txt").write_text("\n".join(kernels) + "\n", encoding="utf-8")
        props = torch.cuda.get_device_properties(args.gpu)
        record.config.update(gpu_uuid=str(props.uuid), gpu_name=props.name)
        record.finish("success", raw_logs=["raw/trace.json", "operators.txt"])
        print(record.root)
    except Exception as exc:
        record.finish_exception(exc)
        raise


if __name__ == "__main__":
    main()
