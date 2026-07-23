#!/usr/bin/env python3
"""Run one optimizer step for isolated B1 and B2 local heads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss  # noqa: E402
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import (  # noqa: E402
    SurfaceConstrainedCorrespondenceHeadV2,
)


def _head(device: torch.device, *, exact: bool):
    return SurfaceConstrainedCorrespondenceHeadV2(
        embed_dim=8,
        num_patches=2,
        top_k_patches=2,
        local_candidates=2,
        teacher_forcing_initial_probability=1.0,
        teacher_forcing_final_probability=1.0,
        deduplicate_local_candidates=True,
        inject_all_valid_triangles=True,
        teacher_force_exact_triangle=exact,
        max_local_candidate_total=4,
        sort_owned_faces_by_distance=True,
    ).to(device).train()


def _inputs(device: torch.device):
    vertices = torch.tensor(
        [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]], device=device
    )
    faces = torch.tensor([[0,1,2],[1,3,2]], device=device)
    observed = torch.randn(1, 3, 8, device=device)
    template = torch.randn(1, 4, 8, device=device)
    template_points = vertices[None]
    target = torch.tensor([[[.2,.2,0.],[.8,.2,0.],[.8,.8,0.]]], device=device)
    return dict(
        observed_features=observed,
        template_features=template,
        template_points=template_points,
        observed_mask=torch.ones(1, 3, dtype=torch.bool, device=device),
        template_mask=torch.ones(1, 4, dtype=torch.bool, device=device),
        template_mesh_vertices_O=[vertices],
        template_mesh_faces=[faces],
        teacher_forcing_target_points_O=target,
    ), target


def run_step(device: torch.device, exact: bool) -> dict:
    torch.manual_seed(0)
    head = _head(device, exact=exact)
    inputs, target = _inputs(device)
    trainable = head.barycentric_head.parameters() if exact else head.fine_query.parameters()
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    result = head(**inputs)
    if exact:
        loss = (result["points_O"] - target).square().mean()
        name = "B2"
    else:
        auxiliary = result["auxiliary"]
        loss = multi_positive_softmax_loss(
            auxiliary["fine_local_logits"][0],
            auxiliary["valid_triangle_local_mask"][0],
        )
        name = "B1"
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return {"substage": name, "loss": float(loss.detach()), "finite": bool(torch.isfinite(loss))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": args.device, "results": [run_step(torch.device(args.device), False), run_step(torch.device(args.device), True)]}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all(row["finite"] for row in payload["results"]) else 2


if __name__ == "__main__": raise SystemExit(main())
