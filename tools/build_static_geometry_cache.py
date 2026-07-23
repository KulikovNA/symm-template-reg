#!/usr/bin/env python3
"""Build and audit the coordinate-only cache for the fixed 4x4 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.static_geometry_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION, build_static_geometry, save_static_geometry_cache,
    static_geometry_cache_key, validate_static_cache_configuration,
)
from symm_template_reg.models.backbones.simple_point_encoder import as_padded_points  # noqa: E402


def _sha_tensors(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _template_topology(batch, device):
    vertices = batch["template_mesh_vertices_O"][0].to(device)
    faces = batch["template_mesh_faces"][0].to(device=device, dtype=torch.long)
    triangles = vertices[faces]
    edges = torch.stack((
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 1],
        triangles[:, 0] - triangles[:, 2],
    ), 1)
    cross = torch.linalg.cross(edges[:, 0], -edges[:, 2], dim=-1)
    normals = cross / torch.linalg.vector_norm(cross, dim=-1, keepdim=True).clamp_min(1e-12)
    undirected = torch.cat((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    undirected = undirected.sort(-1).values.unique(dim=0)
    return {
        "template_triangle_normals": normals,
        "template_triangle_edge_lengths": torch.linalg.vector_norm(edges, dim=-1),
        "template_triangle_aabb_min": triangles.amin(1),
        "template_triangle_aabb_max": triangles.amax(1),
        "template_adjacency_edges": undirected,
        "template_mesh_aabb": torch.stack((vertices.amin(0), vertices.amax(0))),
        "template_surface_samples": vertices,
    }


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config, manifest, manifest_file_sha, _, samples, collate, model = load_multifragment_context(
        args.config, args.manifest, output.parent / (output.name + "_loader"), device
    )
    del model
    validate_static_cache_configuration(
        enabled=True, augmentations=config.get("augmentations"),
        frozen_feature_cache=config.get("frozen_feature_cache"),
    )
    host_batch = collate(samples)
    batch = move_to_device(host_batch, device)
    observed_points, observed_mask = as_padded_points(batch["observed"])
    template_points, template_mask = as_padded_points(batch["template"])
    geometry_config = {
        "observed_encoder_neighbors": 12, "template_encoder_neighbors": 12,
        "observed_tokens": 256, "template_tokens": 512,
        "geometry_neighbors": 8, "fine_neighbors": 32,
    }
    mesh_sha = _sha_tensors(
        [host_batch["template_mesh_vertices_O"][0], host_batch["template_mesh_faces"][0]]
    )
    key = static_geometry_cache_key(
        manifest_sha256=manifest.get("manifest_sha256", manifest_file_sha),
        observed_points=observed_points, observed_mask=observed_mask,
        template_points=template_points, template_mask=template_mask,
        template_mesh_sha256=mesh_sha, geometry_config=geometry_config,
        point_selection_policy="shell_only",
    )
    started = time.perf_counter()
    online = build_static_geometry(
        observed_points, observed_mask, template_points, template_mask, **geometry_config
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - started
    # Recompute independently with the production primitive for an online/cache audit.
    comparison = build_static_geometry(
        observed_points, observed_mask, template_points, template_mask, **geometry_config
    )
    integer_equal = {
        name: bool(torch.equal(online[name], comparison[name]))
        for name in online if not online[name].is_floating_point()
    }
    float_max_abs = {
        name: float((online[name] - comparison[name]).abs().max())
        for name in online if online[name].is_floating_point()
    }
    online.update(_template_topology(batch, device))
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key_sha256": key,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_sha256": manifest_file_sha,
        "template_mesh_sha256": mesh_sha,
        "geometry_config": geometry_config,
        "point_selection_policy": "shell_only",
        "dtype": str(observed_points.dtype),
        "sample_count": len(samples),
        "contains_learned_features": False,
        "cached_fields": sorted(online),
        "build_seconds": build_seconds,
    }
    tensor_path, manifest_path = save_static_geometry_cache(output, online, metadata)
    passed = all(integer_equal.values()) and max(float_max_abs.values(), default=0.0) <= 1e-7
    audit = {
        "audit_passed": passed,
        "integer_indices_bitwise_equal": integer_equal,
        "float_descriptor_max_abs_difference": float_max_abs,
        "float_tolerance": 1e-7,
        "cache_path": str(tensor_path),
        "manifest_path": str(manifest_path),
    }
    (output / "static_geometry_cache_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({**metadata, **audit}, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
