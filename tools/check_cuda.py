#!/usr/bin/env python3
"""Report whether the pure-PyTorch baseline can use CUDA."""

from __future__ import annotations

import json
import os

import torch


def main() -> int:
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    result = {
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": devices,
        "baseline": "pure_pytorch_no_custom_extensions",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
