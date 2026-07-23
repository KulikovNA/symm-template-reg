#!/usr/bin/env python3
"""Write point/active symmetry-region distribution for one strict manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import apply_overrides, load_config  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.single_fragment import region_class_distribution  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.registry import DATASETS, build_from_cfg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg-options", nargs="*")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.cfg_options)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    dataset_cfg = deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_cache_dir"] = output / "cache"
    data = config["data"]
    for key in ("fragment_mesh_filter", "observed_filter", "symmetry_region_activity"):
        if key in data:
            dataset_cfg[key] = deepcopy(data[key])
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, _ = load_faces840_manifest(args.manifest, config, dataset)
    indices = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    report = region_class_distribution(
        (dataset[indices[sample["sample_id"]]] for sample in manifest["samples"]),
        max_class_weight=float(
            config["loss"].get("observed_region_loss", {}).get(
                "max_class_weight", 5.0
            )
        ),
    )
    (output / "region_class_distribution.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "region_class_distribution.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "region_id", "point_frequency", "class_weight",
            "active_positive_samples", "active_valid_samples", "active_pos_weight",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in report["region_ids"]:
            writer.writerow({
                "region_id": name,
                "point_frequency": report["point_frequency"][name],
                "class_weight": report["inverse_sqrt_frequency_weights"][name],
                "active_positive_samples": report["active_positive_samples"][name],
                "active_valid_samples": report["active_valid_samples"][name],
                "active_pos_weight": report["active_pos_weight"][name],
            })
    print(json.dumps({"output_dir": str(output), "absent_point_classes": report["absent_point_classes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
