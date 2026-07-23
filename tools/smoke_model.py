#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from _common import build_model, build_real_batch, move_to_device, resolve_device, tensor_shapes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real variable-size model forward")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device is None:
        print(json.dumps({"status": "skipped", "reason": "CUDA is not available"}, indent=2))
        return 0
    torch.manual_seed(0)
    config, dataset, batch, lengths = build_real_batch(args.config, args.num_samples)
    model = build_model(config["model"]).to(device).eval()
    batch = move_to_device(batch, device)
    with torch.no_grad():
        prediction = model(batch)
    prediction.validate()
    payload = {
        "status": "ok",
        "device": str(device),
        "dataset_samples": len(dataset),
        "observed_lengths": lengths,
        "different_observed_lengths": len(set(lengths)) > 1,
        "symmetry_available": prediction.symmetry_available.tolist(),
        "finite": all(
            bool(torch.isfinite(value).all())
            for value in prediction.as_dict().values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
        "output_shapes": tensor_shapes(prediction),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

