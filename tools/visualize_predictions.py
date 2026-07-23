#!/usr/bin/env python3
"""Export deterministic all-view PLY diagnostics for one strict checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.checkpoint import load_checkpoint  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402
from symm_template_reg.visualization.prediction_debug import (  # noqa: E402
    export_prediction_visualizations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.cfg_options)
    output = Path(args.output_dir).expanduser().resolve()
    register_all_modules()
    dataset_cfg = deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_cache_dir"] = output.parent / (
        output.name + "_cache"
    )
    for key in ("fragment_mesh_filter", "observed_filter", "symmetry_region_activity"):
        if key in config["data"]:
            dataset_cfg[key] = deepcopy(config["data"][key])
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, _ = load_faces840_manifest(args.manifest, config, dataset)
    record_indices = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    indices = [record_indices[sample["sample_id"]] for sample in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    device = resolve_device(args.device)
    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)
    paths = export_prediction_visualizations(
        model,
        dataset,
        indices,
        collate,
        device,
        epoch=0,
        output_dir=output,
        config={
            **config["debug_visualization"],
            "single_fragment_layout": len(indices) == 10,
            "pose_query_ranking": config["loss"].get("pose_query_ranking", {}),
            "symmetry_region_activity": config["data"].get(
                "symmetry_region_activity", {}
            ),
        },
    )
    print(json.dumps({"output_dir": args.output_dir, "files": len(paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
