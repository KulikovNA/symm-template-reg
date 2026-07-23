#!/usr/bin/env python3
"""Run one finite optimizer step for every V4 correspondence head."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from symm_template_reg.models.heads import (  # noqa: E402
    CanonicalCoordinateRegressionControl,
    SoftCoarseLocalSurfaceCorrespondenceHead,
    SurfaceConstrainedCorrespondenceHeadV2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); torch.manual_seed(0)
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], device=device)
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], device=device)
    results = {}
    for head in (
        SurfaceConstrainedCorrespondenceHeadV2(embed_dim=8, num_patches=4, top_k_patches=2, local_candidates=2),
        SoftCoarseLocalSurfaceCorrespondenceHead(embed_dim=8, nearest_triangle_candidates=2),
        CanonicalCoordinateRegressionControl(embed_dim=8, hidden_dim=16),
    ):
        head = head.to(device); optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        observed = torch.randn(1, 8, 8, device=device); template_features = torch.randn(1, 4, 8, device=device)
        output = head(observed, template_features, vertices[None], torch.ones((1, 8), dtype=torch.bool, device=device), torch.ones((1, 4), dtype=torch.bool, device=device), template_mesh_vertices_O=[vertices], template_mesh_faces=[faces])
        loss = output["points_O"].square().mean(); optimizer.zero_grad(); loss.backward(); optimizer.step()
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(value).all()) for value in head.parameters())
        results[type(head).__name__] = {"loss": float(loss.detach()), "finite": finite}
    print(json.dumps({"device": str(device), "heads": results}, indent=2))
    return 0 if all(row["finite"] for row in results.values()) else 2


if __name__ == "__main__": raise SystemExit(main())
