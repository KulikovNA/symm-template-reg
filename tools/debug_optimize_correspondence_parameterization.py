#!/usr/bin/env python3
"""Optimize five correspondence parameterizations directly, without a network."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.geometry import (  # noqa: E402
    barycentric_points,
    closest_points_on_triangle_mesh,
    nearest_triangles_on_mesh,
)
from symm_template_reg.models.pose import WeightedProcrustes  # noqa: E402
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance  # noqa: E402
from symm_template_reg.models.geometry.point_ops import farthest_point_indices  # noqa: E402
from _correspondence_diagnostics import actual_template_anchors, build_dataset, manifest_samples, statistics_mm  # noqa: E402

METHODS = (
    "free_q",
    "surface_gt_patch_barycentric",
    "surface_gt_injected_predicted_patch",
    "surface_predicted_topk",
    "legacy_soft_local_surface",
)


def _ids(count: int, maximum: int) -> torch.Tensor:
    return torch.linspace(0, count - 1, min(count, maximum)).long()


def _prepare(sample, max_points, device):
    q_all = sample["gt"]["points_O_corresponding"].to(device)
    p_all = sample["observed"]["points_C"].to(device)
    ids = _ids(len(q_all), max_points).to(device)
    q_gt, p = q_all[ids], p_all[ids]
    vertices = sample["template"]["points_O"].to(device)
    faces = sample["template"]["faces"].to(device)
    triangles, centroids = vertices[faces], vertices[faces].mean(1)
    coarse_points = actual_template_anchors(sample, 512).to(device)
    patch_indices, patch_mask = farthest_point_indices(
        coarse_points[None], torch.ones((1, len(coarse_points)), dtype=torch.bool, device=device), 64
    )
    patch_points = coarse_points[patch_indices[0, patch_mask[0]]]
    patch_to_face = torch.cdist(patch_points.float(), centroids.float())
    face_owner = patch_to_face.argmin(0)
    maximum_owned = max(
        int(face_owner.eq(patch_id).sum()) for patch_id in range(len(patch_points))
    )
    candidate_count = min(len(faces), max(32, maximum_owned))
    patch_faces = torch.stack([
        torch.cat((
            (owned := torch.nonzero(face_owner.eq(patch_id), as_tuple=False).flatten()),
            patch_to_face[patch_id].argsort()[
                ~torch.isin(patch_to_face[patch_id].argsort(), owned)
            ],
        ))[:candidate_count]
        for patch_id in range(len(patch_points))
    ])
    nearest = closest_points_on_triangle_mesh(q_gt, vertices, faces)
    gt_face = nearest["face_ids"]
    gt_patch = face_owner[gt_face]
    return {
        "sample": sample, "ids": ids, "q_gt": q_gt, "p": p,
        "vertices": vertices, "faces": faces, "triangles": triangles,
        "centroids": centroids, "patch_points": patch_points,
        "coarse_points": coarse_points, "patch_faces": patch_faces,
        "nearest": nearest, "gt_face": gt_face, "gt_patch": gt_patch,
    }


def _run(method, prepared, temperature, start, steps, device):
    sample = prepared["sample"]; ids = prepared["ids"]
    q_gt = prepared["q_gt"]; p = prepared["p"]
    vertices = prepared["vertices"]; faces = prepared["faces"]
    triangles = prepared["triangles"]; centroids = prepared["centroids"]
    patch_points = prepared["patch_points"]; coarse_points = prepared["coarse_points"]
    patch_faces = prepared["patch_faces"]; nearest = prepared["nearest"]
    gt_face = prepared["gt_face"]; gt_patch = prepared["gt_patch"]
    generator = torch.Generator(device=device).manual_seed(
        100000 * int(sample["frame_id"]) + 100 * start + int(temperature * 100)
    )
    parameters: list[torch.nn.Parameter] = []
    state = {}
    if method == "free_q":
        state["q"] = torch.nn.Parameter(
            q_gt + torch.randn(q_gt.shape, generator=generator, device=device) * .005
        )
        parameters.append(state["q"])
    elif method == "surface_gt_patch_barycentric":
        state["bary"] = torch.nn.Parameter(
            temperature * (
                nearest["barycentric"].clamp_min(1e-5).log()
                + torch.randn(
                    nearest["barycentric"].shape,
                    generator=generator,
                    device=device,
                ) * .1
            )
        )
        parameters.append(state["bary"])
    else:
        width = len(coarse_points) if method == "legacy_soft_local_surface" else len(patch_points)
        state["coarse"] = torch.nn.Parameter(
            temperature
            * torch.randn((len(ids), width), generator=generator, device=device)
            * .05
        )
        state["fine"] = torch.nn.Parameter(
            torch.randn(
                (
                    len(ids),
                    (4 * patch_faces.shape[-1])
                    if method != "legacy_soft_local_surface"
                    else min(32, len(faces)),
                ),
                generator=generator,
                device=device,
            ) * (.05 * temperature)
        )
        state["bary"] = torch.nn.Parameter(
            temperature * (
                nearest["barycentric"].clamp_min(1e-5).log()
                + torch.randn((len(ids), 3), generator=generator, device=device) * .1
            )
        )
        parameters.extend(state.values())

        # Capacity is a representability audit, not a cold-start training test.
        # Use supervised warm starts plus independently seeded perturbations;
        # the five parameterizations are still optimized in every run.
        with torch.no_grad():
            row = torch.arange(len(q_gt), device=device)
            if method == "legacy_soft_local_surface":
                state["coarse"].copy_(
                    temperature
                    * (-torch.cdist(q_gt.float(), coarse_points.float()) / .002)
                    + state["coarse"]
                )
                q_coarse = torch.softmax(state["coarse"] / temperature, -1) @ coarse_points
                candidate = nearest_triangles_on_mesh(
                    q_coarse.detach(), vertices, faces, min(32, len(faces)),
                    point_chunk_size=256,
                )["face_ids"]
                state["fixed_candidate"] = candidate
            else:
                state["coarse"][row, gt_patch] += 8.0 * temperature
                topk = state["coarse"].topk(4, -1).indices
                candidate = patch_faces[topk].flatten(1, 2)
                if method == "surface_gt_injected_predicted_patch":
                    included = candidate.eq(gt_face[:, None]).any(-1)
                    candidate[:, -1] = torch.where(
                        included, candidate[:, -1], gt_face
                    )
            matches = candidate.eq(gt_face[:, None])
            fallback = torch.linalg.vector_norm(
                centroids[candidate] - q_gt[:, None], dim=-1
            ).argmin(-1)
            fine_target = torch.where(
                matches.any(-1), matches.float().argmax(-1), fallback
            )
            state["fine"][row, fine_target] += 8.0 * temperature

    def decode():
        extra = q_gt.new_zeros(())
        candidate_recall = q_gt.new_tensor(1.0)
        if method == "free_q":
            return state["q"], extra, candidate_recall
        if method == "surface_gt_patch_barycentric":
            q = barycentric_points(triangles[gt_face], torch.softmax(state["bary"] / temperature, -1))
            return q, extra, candidate_recall
        if method == "legacy_soft_local_surface":
            q_coarse = torch.softmax(state["coarse"] / temperature, -1) @ coarse_points
            candidate = state["fixed_candidate"]
            fallback_target = torch.linalg.vector_norm(
                centroids[candidate] - q_gt[:, None], dim=-1
            ).argmin(-1)
            gt_matches = candidate.eq(gt_face[:, None])
            candidate_recall = gt_matches.any(-1).float().mean()
            local_target = torch.where(
                gt_matches.any(-1), gt_matches.float().argmax(-1), fallback_target
            )
            selected = state["fine"].argmax(-1)
            row = torch.arange(len(q_gt), device=device)
            selected_face = candidate[row, selected]
            extra = F.smooth_l1_loss(q_coarse / .002, q_gt / .002) + F.cross_entropy(
                state["fine"] / temperature, local_target
            )
        else:
            topk = state["coarse"].topk(4, -1).indices
            candidate = patch_faces[topk].flatten(1, 2)
            if method == "surface_gt_injected_predicted_patch":
                included = candidate.eq(gt_face[:, None]).any(-1)
                candidate = candidate.clone()
                candidate[:, -1] = torch.where(included, candidate[:, -1], gt_face)
            fallback_target = torch.linalg.vector_norm(
                centroids[candidate] - q_gt[:, None], dim=-1
            ).argmin(-1)
            gt_matches = candidate.eq(gt_face[:, None])
            local_target = torch.where(
                gt_matches.any(-1), gt_matches.float().argmax(-1), fallback_target
            )
            selected = state["fine"].argmax(-1)
            row = torch.arange(len(q_gt), device=device)
            selected_face = candidate[row, selected]
            extra = F.cross_entropy(state["coarse"] / temperature, gt_patch) + F.cross_entropy(
                state["fine"] / temperature, local_target
            )
            candidate_recall = candidate.eq(gt_face[:, None]).any(-1).float().mean()
        q = barycentric_points(
            triangles[selected_face], torch.softmax(state["bary"] / temperature, -1)
        )
        return q, extra, candidate_recall

    if method in {"free_q", "surface_gt_patch_barycentric"}:
        optimizer = torch.optim.Adam(parameters, lr=.001)
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": [state["coarse"], state["fine"]], "lr": .01},
                {"params": [state["bary"]], "lr": .05},
            ]
        )
    completed_steps = 0
    early_threshold_mm = {
        "free_q": .08,
        "surface_gt_patch_barycentric": .45,
        "surface_gt_injected_predicted_patch": .45,
        "surface_predicted_topk": 1.9,
        "legacy_soft_local_surface": 1.9,
    }[method]
    for step in range(steps):
        optimizer.zero_grad()
        q, auxiliary_loss, _ = decode()
        distance = torch.linalg.vector_norm(q - q_gt, dim=-1)
        loss = (q - q_gt).square().mean() / (.002 ** 2) + auxiliary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
        completed_steps = step + 1
        if completed_steps >= 10 and completed_steps % 10 == 0:
            with torch.no_grad():
                check_q, _, _ = decode()
                check_p95 = float(torch.quantile(
                    torch.linalg.vector_norm(check_q - q_gt, dim=-1), .95
                ) * 1000.)
            if check_p95 < early_threshold_mm:
                break

    with torch.no_grad():
        q, _, candidate_recall = decode()
        mask = torch.ones((1, len(q)), dtype=torch.bool, device=device)
        weights = torch.full((1, len(q)), 1 / len(q), device=device)
        solution = WeightedProcrustes().solve(q[None].float(), p[None].float(), weights, mask)
        pose = solution["transform"][0]
        gt_pose = sample["gt"]["T_C_from_O"].to(device)
        correspondence = torch.linalg.vector_norm(q - q_gt, dim=-1)
        aligned = (pose[:3, :3] @ q.T).T + pose[:3, 3]
        alignment = torch.linalg.vector_norm(aligned - p, dim=-1)
        surface = (
            closest_points_on_triangle_mesh(q, vertices, faces)["distances"]
            if method == "free_q" else torch.zeros(len(q), device=device)
        )
        rotation = float(torch.rad2deg(rotation_geodesic_distance(pose[:3, :3], gt_pose[:3, :3])))
        translation = float(torch.linalg.vector_norm(pose[:3, 3] - gt_pose[:3, 3]) * 1000)
        corr = statistics_mm(correspondence)
    return {
        "frame_id": int(sample["frame_id"]), "method": method,
        "temperature": temperature, "random_start": start, "steps": completed_steps,
        "optimized_points": len(ids), "correspondence_p50_mm": corr["p50_mm"],
        "correspondence_p95_mm": corr["p95_mm"], "rotation_error_deg": rotation,
        "translation_total_mm": translation,
        "alignment_p95_mm": statistics_mm(alignment)["p95_mm"],
        "predicted_to_template_surface_p95_mm": statistics_mm(surface)["p95_mm"],
        "gt_triangle_in_candidate_set_fraction": float(candidate_recall),
        "correspondence_rank": int(solution["rank"][0]),
        "procrustes_rank_valid": bool(solution["rank_valid"][0]),
    }


def _passes(row, corr, pose):
    return row["correspondence_p95_mm"] < corr and row["rotation_error_deg"] < pose and row["translation_total_mm"] < pose and row["correspondence_rank"] == 3


def _write_progress(output: Path, rows: list[dict], completed: int, expected: int) -> None:
    """Persist partial results so a long audit remains observable and recoverable."""
    if rows:
        with (output / "surface_parameterization_capacity_progress.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    progress = {
        "completed_runs": completed,
        "expected_runs": expected,
        "fraction": completed / expected if expected else 1.0,
        "last_result": rows[-1] if rows else None,
    }
    (output / "surface_parameterization_capacity_progress.json").write_text(
        json.dumps(progress, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame", type=int, choices=(4, 8), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--random-starts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500); parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0, .5, .2, .1])
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    expected = (
        "surface_parameterization_capacity.csv",
        "surface_parameterization_capacity_summary.json",
        "surface_parameterization_capacity_report.md",
    )
    if any((output / name).exists() for name in expected):
        raise FileExistsError(f"capacity audit outputs already exist in {output}")
    _, dataset = build_dataset(args.config, output, shell_only=True)
    _, samples = manifest_samples(dataset, args.manifest, frames=(args.frame,)); sample = samples[0]
    prepared = _prepare(sample, args.max_points, device)
    rows = []
    expected_runs = sum(
        (1 if method == "free_q" else len(args.temperatures)) * args.random_starts
        for method in args.methods
    )
    _write_progress(output, rows, 0, expected_runs)
    for method in args.methods:
        temperatures = [1.0] if method == "free_q" else args.temperatures
        for temperature in temperatures:
            for start in range(args.random_starts):
                print(f"capacity frame={args.frame} method={method} temperature={temperature} start={start + 1}/{args.random_starts}", flush=True)
                row = _run(method, prepared, temperature, start, args.steps, device)
                rows.append(row)
                _write_progress(output, rows, len(rows), expected_runs)
                print(
                    "capacity result "
                    f"p95={row['correspondence_p95_mm']:.6f}mm "
                    f"rot={row['rotation_error_deg']:.6f}deg "
                    f"trans={row['translation_total_mm']:.6f}mm "
                    f"steps={row['steps']}",
                    flush=True,
                )
    best = {method: min((row for row in rows if row["method"] == method), key=lambda row: row["correspondence_p95_mm"] + row["rotation_error_deg"] + row["translation_total_mm"]) for method in args.methods}
    gates = {
        "free_capacity_passed": any(_passes(row, .1, .1) for row in rows if row["method"] == "free_q"),
        "gt_patch_barycentric_capacity_passed": any(_passes(row, .5, .5) for row in rows if row["method"] == "surface_gt_patch_barycentric"),
        "gt_injected_predicted_patch_capacity_passed": any(_passes(row, .5, .5) for row in rows if row["method"] == "surface_gt_injected_predicted_patch"),
        "full_predicted_topk_capacity_passed": any(_passes(row, 2.0, 2.0) for row in rows if row["method"] == "surface_predicted_topk"),
        "legacy_soft_local_surface_capacity_passed": any(_passes(row, 2.0, 2.0) for row in rows if row["method"] == "legacy_soft_local_surface"),
    }
    summary = {
        "audit_passed": all(gates.values()),
        "frame_id": args.frame,
        "random_start_count": args.random_starts,
        "temperatures": args.temperatures,
        "maximum_steps": args.steps,
        "random_start_policy": "independently seeded supervised warm starts; early stop only after the method-specific physical gate",
        **gates,
        "best_by_parameterization": best,
    }
    with (output / "surface_parameterization_capacity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "surface_parameterization_capacity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "surface_parameterization_capacity_report.md").write_text("# Surface parameterization capacity\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2)); return 0 if summary["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
