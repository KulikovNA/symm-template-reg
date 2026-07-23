#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from _common import build_model, build_real_batch, move_to_device, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one model forward's parameter and CUDA memory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device is None:
        print(json.dumps({"status": "skipped", "reason": "CUDA is not available"}, indent=2))
        return 0
    config, _, batch, lengths = build_real_batch(args.config, 2)
    model = build_model(config["model"]).to(device).eval()
    batch = move_to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(batch)
    payload = {
        "status": "ok",
        "device": str(device),
        "observed_lengths": lengths,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

