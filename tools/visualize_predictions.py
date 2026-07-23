#!/usr/bin/env python3
"""Сохранить active q_aux, exact projection и предсказанную регистрацию в PLY."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.evaluation.active_coordinate import evaluate_active_sample  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402
from symm_template_reg.visualization.ply import write_colored_ply  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--fragment-id", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    config = load_config(args.config)
    dataset_cfg = deepcopy(config["data"]["validation"])
    dataset_cfg.update(
        dataset_root=args.dataset_root,
        split=args.split,
        selector={
            "scene_ids": [args.scene_id],
            "frame_ids": [args.frame_id],
            "fragment_ids": [args.fragment_id],
            "max_samples": 1,
        },
        boundary_augmentation={"enabled": False},
        index_cache_dir=str(output / "dataset_index"),
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = collate([dataset[0]])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    moved = move(batch)
    model = build_model(config["model"]).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    with torch.no_grad():
        prediction = model(moved)
        result = evaluate_active_sample(
            q_aux_O=prediction.correspondence_points_O[0],
            valid_mask=prediction.observed_valid_mask[0],
            target_O=moved["gt"]["points_O_corresponding"][0],
            observed_C=moved["observed"]["points_C"][0],
            vertices_O=moved["template_mesh_vertices_O"][0],
            faces=moved["template_mesh_faces"][0],
            equivalent_pose=moved["gt"]["T_C_from_O"][0],
            procrustes=model.weighted_procrustes,
            candidate_k=int(config.get("evaluation", {}).get("candidate_k", 16)),
        )
    mask = prediction.observed_valid_mask[0]
    q_aux = prediction.correspondence_points_O[0, mask]
    projected = result["projected_points_O"]["exact_global"]
    predicted_pose = result["T_C_from_O"]["exact_global"]
    registered = transform_points(
        invert_transform(predicted_pose)[None],
        moved["observed"]["points_C"][0, mask][None],
    )[0]
    vertices = moved["template_mesh_vertices_O"][0].cpu().numpy()
    faces = moved["template_mesh_faces"][0].cpu().numpy()
    write_colored_ply(
        output / "template.ply", vertices, [180, 180, 180], faces=faces
    )
    write_colored_ply(
        output / "q_aux_raw.ply",
        q_aux.cpu().numpy(),
        [30, 210, 80],
    )
    write_colored_ply(
        output / "q_aux_projected_exact.ply",
        projected.cpu().numpy(),
        [220, 40, 210],
    )
    write_colored_ply(
        output / "registered_observed_fragment.ply",
        registered.cpu().numpy(),
        [255, 145, 20],
    )
    summary = {
        "sample_id": batch["sample_id"][0],
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "pose_source": "q_aux -> exact mesh projection -> uniform WeightedProcrustes",
        "files": [
            "template.ply",
            "q_aux_raw.ply",
            "q_aux_projected_exact.ply",
            "registered_observed_fragment.ply",
        ],
    }
    (output / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
