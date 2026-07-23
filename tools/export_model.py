#!/usr/bin/env python3
"""Экспортировать active model state_dict и проверяемый JSON-манифест."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.models import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = apply_overrides(load_config(args.config), args.cfg_options)
    model = build_model(config["model"]).cpu().eval()
    source = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    destination = output / "coordinate_guided_surface_v3_state_dict.pt"
    torch.save(model.state_dict(), destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "format": "pytorch-state-dict",
        "model_type": type(model).__name__,
        "source_checkpoint": str(source),
        "state_dict": destination.name,
        "sha256": digest,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forbidden_legacy_keys": [
            key
            for key in model.state_dict()
            if any(
                token in key.lower()
                for token in model.checkpoint_forbidden_tokens()
            )
        ],
    }
    if manifest["forbidden_legacy_keys"]:
        raise AssertionError(manifest["forbidden_legacy_keys"])
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
