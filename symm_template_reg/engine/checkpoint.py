"""Strict checkpoint save/load and reproducibility manifests."""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch


def system_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    epoch: int,
    best_metric: float,
    manifest: Mapping[str, Any],
    is_best: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_payload = {
        **dict(manifest),
        "checkpoint_path": str(destination.resolve()),
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "strict_load_status": "saved_for_strict_load",
        "system": system_info(),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if is_best:
        best = destination.parent / "best.pt"
        shutil.copyfile(destination, best)
        shutil.copyfile(
            manifest_path, best.with_suffix(best.suffix + ".manifest.json")
        )
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


__all__ = ["load_checkpoint", "save_checkpoint", "system_info"]
