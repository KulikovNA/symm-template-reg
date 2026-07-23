#!/usr/bin/env python3
"""Build and validate a deterministic shell-only coordinate-view manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.datasets.fragment_mesh_filter import sha256_file  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.engine.single_fragment import (  # noqa: E402
    manifest_content_sha256,
    validate_single_fragment_manifest_payload,
)
from symm_template_reg.engine.view_ladder import subset_view_manifest  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.registry import DATASETS, build_from_cfg  # noqa: E402


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _write_manifest(path: Path, payload: dict) -> str:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def build_coordinate_view_manifest(
    source_manifest: str | Path,
    frames: list[int],
    output: str | Path,
    *,
    shell_only: bool,
    config_path: str | Path = "configs/debug/coordinate_guided_surface_v2/views02.py",
) -> dict:
    if not shell_only:
        raise ValueError("coordinate-guided manifests must be explicitly --shell-only")
    requested = list(map(int, frames))
    prepared = {
        (4, 5, 2, 8): {"parent": (4, 8), "label": "four-view"},
        (4, 5, 2, 8, 0, 1, 6, 9): {
            "parent": (4, 5, 2, 8),
            "label": "eight-view",
        },
        tuple(range(10)): {
            "parent": (4, 5, 2, 8, 0, 1, 6, 9),
            "label": "ten-view-scratch",
        },
    }
    contract_spec = prepared.get(tuple(requested))
    if contract_spec is None:
        raise ValueError(
            "prepared coordinate stages require frames 4 5 2 8 or "
            "4 5 2 8 0 1 6 9, or 0 1 2 3 4 5 6 7 8 9 in that exact order"
        )
    source_path = Path(source_manifest).expanduser().resolve()
    sidecar = source_path.with_suffix(source_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"source manifest SHA sidecar is missing: {sidecar}")
    expected_file_sha = sidecar.read_text(encoding="ascii").split()[0]
    if sha256_file(source_path) != expected_file_sha:
        raise ValueError("source manifest file SHA256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("manifest_sha256") != manifest_content_sha256(source):
        raise ValueError("source manifest internal SHA256 mismatch")
    validate_single_fragment_manifest_payload(source, expected_samples=10)
    payload = subset_view_manifest(source, requested)
    payload.update(WARNING_FLAGS)
    if len(requested) == 10:
        payload.update(
            initialization_mode="scratch",
            pretrained_checkpoint=None,
        )
    payload["registration_point_selection"] = "shell_only"
    payload["point_selection_policy"] = "deterministic_shell_only_all_points"
    for sample in payload["samples"]:
        sample["registration_point_selection"] = "shell_only"
    payload["manifest_sha256"] = manifest_content_sha256(payload)

    config = load_config(config_path)
    dataset_cfg = deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_filter"] = deepcopy(config["data"]["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(config["data"]["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        config["data"].get("symmetry_region_activity", {})
    )
    destination = Path(output).expanduser().resolve()
    dataset_cfg["fragment_mesh_cache_dir"] = str(
        destination.parent / ".fragment_mesh_metadata_cache"
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    # Validate the subset against the exact runtime dataset contract before it
    # is persisted.  The temporary file is not needed: the source is already
    # validated, and the records below exercise point selection directly.
    indices = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    sample_reports = []
    for manifest_sample in payload["samples"]:
        sample_id = str(manifest_sample["sample_id"])
        sample = dataset[indices[sample_id]]
        labels = sample["observed"]["surface_labels"]
        shell_count = int(labels.eq(0).sum())
        fracture_count = int(labels.ne(0).sum())
        if fracture_count != 0 or shell_count != int(labels.numel()):
            raise ValueError(f"shell-only point contract failed for {sample_id}")
        if sample["gt"].get("T_C_from_O") is None or sample["gt"].get("T_W_from_C") is None:
            raise ValueError(f"pose transforms are missing for {sample_id}")
        group = manifest_sample.get("effective_symmetry_group", {})
        if str(group.get("type")) != "C" or int(group.get("order", -1)) != 2:
            raise ValueError(f"effective group is not C2 for {sample_id}: {group}")
        if manifest_sample.get("data_contract_errors"):
            raise ValueError(f"data contract errors for {sample_id}")
        manifest_sample["shell_point_count"] = shell_count
        manifest_sample["fracture_point_count"] = 0
        manifest_sample["points_passed_to_model"] = shell_count
        sample_reports.append(
            {
                "sample_id": sample_id,
                "frame_id": int(manifest_sample["frame_id"]),
                "shell_point_count": shell_count,
                "fracture_point_count": 0,
                "effective_symmetry_group": "C2",
                "T_C_from_O_present": True,
                "T_W_from_C_present": True,
            }
        )
    payload["manifest_sha256"] = manifest_content_sha256(payload)
    contract = validate_single_fragment_manifest_payload(
        payload, expected_samples=len(requested), min_num_faces=840
    )
    parent_frames = set(contract_spec["parent"])
    current_frames = set(requested)
    nested = parent_frames < current_frames
    if not nested:
        raise ValueError(
            f"{contract_spec['label']} manifest is not a strict superset of "
            f"{sorted(parent_frames)}"
        )
    file_sha = _write_manifest(destination, payload)
    report = {
        **WARNING_FLAGS,
        "manifest": str(destination),
        "manifest_file_sha256": file_sha,
        "manifest_sha256": payload["manifest_sha256"],
        "source_manifest": str(source_path),
        "source_manifest_file_sha256": expected_file_sha,
        "frames": requested,
        "stage_label": contract_spec["label"],
        "nested_parent_frames": list(contract_spec["parent"]),
        **({
            4: {"nested_two_view_frames": [4, 8]},
            8: {"nested_four_view_frames": [4, 5, 2, 8]},
            10: {"nested_eight_view_frames": [4, 5, 2, 8, 0, 1, 6, 9]},
        }[len(requested)]),
        "nested_contract_passed": nested,
        "shell_only": True,
        "deterministic_point_selection": True,
        "train_validation_same_samples": True,
        "fragment_mesh_sha256": payload["fragment_mesh_sha256"],
        "contract": contract,
        "samples": sample_reports,
        "manifest_validation_passed": True,
        **(
            {"initialization_mode": "scratch", "pretrained_checkpoint": None}
            if len(requested) == 10 else {}
        ),
    }
    report_path = destination.with_name(destination.stem + "_validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if len(requested) in {8, 10}:
        prefix = "eight_view" if len(requested) == 8 else "ten_view"
        title = "Eight-view" if len(requested) == 8 else "Ten-view scratch"
        audit_path = destination.with_name(f"{prefix}_manifest_audit.json")
        audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [
            f"# {title} shell-only manifest audit",
            "",
            f"- validation passed: `{report['manifest_validation_passed']}`",
            f"- nested parent-view contract: `{report['nested_contract_passed']}`",
            f"- one physical fragment SHA256: `{report['fragment_mesh_sha256']}`",
            "- effective group: `C2` for every sample",
            "- fracture points: `0` for every sample",
            "",
            "| frame | shell points | T_C_from_O | T_W_from_C |",
            "|---:|---:|:---:|:---:|",
        ]
        lines.extend(
            f"| {row['frame_id']} | {row['shell_point_count']} | "
            f"{row['T_C_from_O_present']} | {row['T_W_from_C_present']} |"
            for row in sample_reports
        )
        destination.with_name(f"{prefix}_manifest_audit.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    # Re-open through the production validator after writing so file hashes,
    # sample IDs, and dataset filtering are checked together.
    validate_config = deepcopy(config)
    validate_config["data"]["train_manifest"] = str(destination)
    load_and_validate_manifest(str(destination), validate_config, dataset)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--shell-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/debug/coordinate_guided_surface_v2/views02.py",
    )
    args = parser.parse_args()
    register_all_modules()
    report = build_coordinate_view_manifest(
        args.source_manifest,
        args.frames,
        args.output,
        shell_only=args.shell_only,
        config_path=args.config,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
