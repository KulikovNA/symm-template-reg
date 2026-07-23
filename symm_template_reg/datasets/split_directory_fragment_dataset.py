"""Production train/val/test dataset with automatic deterministic indexing."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .boundary_augmentation import BoundaryMaskAugmentation
from .fragment_template_dataset import (
    FragmentTemplateRegistrationDataset,
    _symmetry_targets,
)
from .template_contract import inspect_template_contract
from .template_repository import TemplateRepository

try:
    from symm_template_reg.registry import DATASETS
except (ImportError, AttributeError):
    DATASETS = None


def _register(cls: type[Any]) -> type[Any]:
    if DATASETS is not None and DATASETS.get(cls.__name__) is None:
        DATASETS.register_module(cls)
    return cls


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalized_ids(
    values: Sequence[Any] | None, prefix: str, width: int
) -> set[Any] | None:
    if values is None:
        return None
    normalized: set[Any] = set()
    for value in values:
        if prefix and isinstance(value, int):
            normalized.add(f"{prefix}_{value:0{width}d}")
        else:
            normalized.add(value)
    return normalized


@_register
class SplitDirectoryFragmentDataset(FragmentTemplateRegistrationDataset):
    """Scan one split without a user manifest.

    The index order is always ``split/scene/frame/fragment``. Fragment and
    frame identifiers remain metadata and are never appended to model features.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        selector: Mapping[str, Any] | None = None,
        scene_ids: Sequence[str | int] | None = None,
        frame_ids: Sequence[int] | None = None,
        fragment_ids: Sequence[int] | None = None,
        max_samples: int | None = None,
        min_num_faces: int = 840,
        min_observed_shell_points: int = 128,
        max_observed_shell_points: int = 4096,
        point_sampling: str | None = None,
        fragment_mesh_filter: Mapping[str, Any] | None = None,
        index_cache_dir: str | Path | None = None,
        boundary_augmentation: Mapping[str, Any] | None = None,
        random_seed: int = 0,
        template_fine_points: int | None = 4096,
        template_coarse_points: int | None = 1024,
        **kwargs: Any,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        root = Path(dataset_root).expanduser().resolve()
        if not all((root / name).is_dir() for name in ("train", "val", "test")):
            raise FileNotFoundError(
                f"{root} must contain physically separated train/val/test directories"
            )
        self.production_dataset_root = root
        self.split = split
        self.template_contract = inspect_template_contract(root)
        template_repository = TemplateRepository(
            root / split / "models",
            fine_points=template_fine_points,
            coarse_points=template_coarse_points,
            recompute_normals=True,
        )
        filter_config = {
            "enabled": True,
            "min_num_faces": int(min_num_faces),
            "exclude_entire_fragment": True,
            "missing_mesh_policy": "error",
            "manifest_mismatch_policy": "error",
            "cache_metadata": index_cache_dir is not None,
        }
        if fragment_mesh_filter is not None:
            filter_config.update(dict(fragment_mesh_filter))
        sampling = point_sampling or (
            "random_up_to_max" if split == "train" else "farthest_point_up_to_max"
        )
        augmentation = dict(boundary_augmentation or {})
        if split != "train" and bool(augmentation.get("enabled", False)):
            raise ValueError(f"boundary augmentation is forbidden for split={split}")
        self.boundary_augmentation_config = augmentation
        self.boundary_augmentation = (
            BoundaryMaskAugmentation(augmentation)
            if bool(augmentation.get("enabled", False))
            else None
        )
        self.random_seed = int(random_seed)
        self.epoch = 0
        self._scene_asset_cache: dict[str, dict[str, Any]] = {}
        cache_root = (
            Path(index_cache_dir).expanduser().resolve()
            if index_cache_dir is not None
            else root / ".index_cache_disabled"
        )
        super().__init__(
            dataset_root=root,
            split=split,
            observed_filter={
                "min_observed_points": int(min_observed_shell_points),
                "max_observed_points": int(max_observed_shell_points),
                "point_policy": sampling,
            },
            fragment_mesh_filter=filter_config,
            fragment_mesh_cache_dir=cache_root / "fragment_mesh_metadata",
            random_seed=int(random_seed),
            template_repository=template_repository,
            template_fine_points=template_fine_points,
            template_coarse_points=template_coarse_points,
            registration_point_selection="shell_only",
            max_samples=None,
            **kwargs,
        )
        selected = dict(selector or {})
        explicit = {
            "scene_ids": scene_ids,
            "frame_ids": frame_ids,
            "fragment_ids": fragment_ids,
            "max_samples": max_samples,
        }
        for key, value in explicit.items():
            if value is not None:
                if key in selected and selected[key] != value:
                    raise ValueError(f"conflicting selector value for {key}")
                selected[key] = value
        unknown = set(selected).difference(
            {"scene_ids", "frame_ids", "fragment_ids", "max_samples"}
        )
        if unknown:
            raise ValueError(f"unknown selector fields: {sorted(unknown)}")
        scene_filter = _normalized_ids(selected.get("scene_ids"), "scene", 6)
        frame_filter = _normalized_ids(selected.get("frame_ids"), "", 0)
        fragment_filter = _normalized_ids(selected.get("fragment_ids"), "", 0)
        records = [
            record
            for record in self._records
            if (scene_filter is None or record.scene_id in scene_filter)
            and (frame_filter is None or record.frame_id in frame_filter)
            and (fragment_filter is None or record.fragment_id in fragment_filter)
        ]
        selected_max = selected.get("max_samples")
        if selected_max is not None:
            records = records[: int(selected_max)]
        self._records = [
            replace(record, sample_id=f"{split}/{record.sample_id}")
            for record in records
        ]
        if not self._records:
            raise ValueError(f"selector produced no samples for split={split}")
        self.selector_config = {
            key: value for key, value in selected.items() if value is not None
        }
        self.index_fingerprint_payload = self._fingerprint_payload(filter_config)
        encoded = json.dumps(
            self.index_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.index_fingerprint = hashlib.sha256(encoded).hexdigest()
        self.index_cache_path: Path | None = None
        if index_cache_dir is not None:
            self.index_cache_path = cache_root / (
                f"dataset_index_{self.index_fingerprint}.json"
            )
            _atomic_json(
                self.index_cache_path,
                {
                    "fingerprint": self.index_fingerprint,
                    "fingerprint_payload": self.index_fingerprint_payload,
                    "split": split,
                    "sample_count": len(self._records),
                    "samples": [
                        {
                            "sample_id": record.sample_id,
                            "scene_id": record.scene_id,
                            "frame_id": record.frame_id,
                            "fragment_id": record.fragment_id,
                            "visible_points_path": str(record.visible_points_path),
                            "fragment_mesh_sha256": (
                                record.fragment_mesh_metadata.sha256
                            ),
                        }
                        for record in self._records
                    ],
                    "index_report": self.index_report,
                },
            )
        self.index_report.update(
            {
                "split": split,
                "split_qualified_sample_ids": True,
                "selector": self.selector_config,
                "accepted_frame_observations_after_selector": len(self._records),
                "dataset_index_fingerprint": self.index_fingerprint,
                "dataset_index_path": (
                    str(self.index_cache_path)
                    if self.index_cache_path is not None
                    else None
                ),
                "template_contract": self.template_contract,
            }
        )

    def _fingerprint_payload(
        self, filter_config: Mapping[str, Any]
    ) -> dict[str, Any]:
        npz_metadata = []
        for path in sorted({record.visible_points_path for record in self._records}):
            stat = path.stat()
            npz_metadata.append(
                {
                    "path": str(path.relative_to(self.production_dataset_root)),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        template_row = self.template_contract["splits"][self.split]
        return {
            "dataset_root": str(self.production_dataset_root),
            "split": self.split,
            "scene_list": sorted({record.scene_id for record in self._records}),
            "npz_metadata": npz_metadata,
            "fragment_mesh_hashes": sorted(
                {
                    record.fragment_mesh_metadata.sha256
                    for record in self._records
                }
            ),
            "template_sha256": template_row["template_sha256"],
            "sidecar_sha256": template_row["sidecar_sha256"],
            "filter_config": dict(filter_config),
            "observed_filter": dict(self.observed_filter_config),
            "selector": self.selector_config,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _frame_assets(self, record: Any) -> dict[str, Any]:
        scene_dir = record.visible_points_path.parent.parent
        cached = self._scene_asset_cache.get(record.scene_id)
        if cached is None:
            camera = json.loads(
                (scene_dir / "camera_info.json").read_text(encoding="utf-8")
            )
            gt = json.loads(
                (scene_dir / "gt_annotations.json").read_text(encoding="utf-8")
            )
            cached = {
                "scene_dir": scene_dir,
                "camera": camera,
                "frames": {
                    int(frame["frame_id"]): frame for frame in gt.get("frames", [])
                },
            }
            self._scene_asset_cache[record.scene_id] = cached
        frame = cached["frames"][record.frame_id]
        return {**cached, "frame": frame}

    def _augmentation_seed(self, sample_id: str) -> int:
        payload = (
            f"{self.random_seed}|{self.epoch}|{sample_id}".encode("utf-8")
        )
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def _apply_boundary_augmentation(
        self, index: int, base: dict[str, Any]
    ) -> dict[str, Any]:
        if self.boundary_augmentation is None:
            return {
                "base": base,
                "debug": None,
                "metadata": {
                    "augmentation_applied": False,
                    "augmentation_mode": "none",
                    "epoch": self.epoch,
                },
            }
        record = self._records[index]
        assets = self._frame_assets(record)
        frame = assets["frame"]
        scene_dir = assets["scene_dir"]
        camera = assets["camera"]
        with np.load(record.visible_points_path, allow_pickle=False) as arrays:
            fragment = np.asarray(arrays["fragment_id"]) == record.fragment_id
            shell = fragment & (np.asarray(arrays["surface_label"]) == 0)
            shell_points = np.asarray(arrays["points_C"][shell], dtype=np.float32)
            shell_targets = np.asarray(arrays["points_O"][shell], dtype=np.float32)
            shell_uv = np.stack(
                (
                    np.asarray(arrays["u"][shell], dtype=np.int64),
                    np.asarray(arrays["v"][shell], dtype=np.int64),
                ),
                axis=-1,
            )
        depth_path = scene_dir / str(
            frame.get("depth", f"depth/frame_{record.frame_id:06d}.png")
        )
        instance_path = scene_dir / str(
            frame.get(
                "instance_mask",
                f"instance_masks/frame_{record.frame_id:06d}.png",
            )
        )
        surface_path = scene_dir / str(
            frame.get(
                "surface_mask",
                f"surface_masks/frame_{record.frame_id:06d}.png",
            )
        )
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth *= float(camera.get("depth_scale_m", 0.001))
        instance_mask = np.asarray(Image.open(instance_path))
        surface_mask = np.asarray(Image.open(surface_path))
        instance_value = int(record.gt_fragment["instance_mask_value"])
        shell_mask = (instance_mask == instance_value) & (surface_mask == 1)
        template = base["template"]
        result = self.boundary_augmentation.apply(
            shell_points_C=shell_points,
            shell_targets_O=shell_targets,
            shell_uv=shell_uv,
            shell_mask=shell_mask,
            instance_mask=instance_mask,
            surface_mask=surface_mask,
            depth_m=depth,
            intrinsics=np.asarray(camera["K"], dtype=np.float32),
            instance_value=instance_value,
            T_C_from_O=np.asarray(record.gt_fragment["T_C_from_O"], dtype=np.float32),
            template_vertices_O=template["points_O"].cpu().numpy(),
            template_faces=template["faces"].cpu().numpy(),
            seed=self._augmentation_seed(record.sample_id),
            epoch=self.epoch,
        )
        selected = self.selector.indices(
            result["points_C"],
            sample_seed=self._augmentation_seed(record.sample_id) & 0x7FFFFFFF,
        )
        points_C = torch.from_numpy(result["points_C"][selected].copy())
        targets_O = torch.from_numpy(result["target_points_O"][selected].copy())
        source_labels = torch.from_numpy(
            result["source_labels"][selected].astype(np.int64, copy=True)
        )
        base["observed"].update(
            {
                "points_C": points_C,
                "surface_labels": source_labels,
                "valid_mask": torch.ones(len(points_C), dtype=torch.bool),
            }
        )
        symmetry_targets = _symmetry_targets(
            targets_O,
            base["gt"]["T_C_from_O"],
            template.get("symmetry_metadata"),
            self.symmetry_region_activity,
        )
        base["gt"].update(
            {
                "points_O_corresponding": targets_O,
                "overlap_labels": torch.ones(len(points_C), dtype=torch.bool),
                **symmetry_targets,
            }
        )
        result["metadata"]["final_point_count_before_max_sampling"] = int(
            result["metadata"]["final_point_count"]
        )
        result["metadata"]["final_point_count"] = len(points_C)
        base["meta"]["num_observed_points"] = len(points_C)
        return {
            "base": base,
            "debug": result["debug"],
            "metadata": result["metadata"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        augmented = self._apply_boundary_augmentation(index, sample)
        sample = augmented["base"]
        record = self._records[index]
        mesh = record.fragment_mesh_metadata
        template_row = self.template_contract["splits"][self.split]
        sample.update(
            {
                "split": self.split,
                "points_C": sample["observed"]["points_C"],
                "target_points_O": sample["gt"]["points_O_corresponding"],
                "valid_mask": sample["observed"]["valid_mask"],
                "T_C_from_O": sample["gt"]["T_C_from_O"],
                "T_W_from_C": sample["gt"]["T_W_from_C"],
                "fragment_mesh_path": str(mesh.mesh_path),
                "fragment_mesh_sha256": mesh.sha256,
                "fragment_num_faces": mesh.num_faces,
                "fragment_surface_area": mesh.surface_area_m2,
                "fragment_bbox": {
                    "min": list(mesh.bbox_min),
                    "max": list(mesh.bbox_max),
                    "diagonal_m": mesh.bbox_diagonal_m,
                },
                "effective_symmetry_group": sample["gt"].get(
                    "effective_symmetry_group"
                ),
                "active_symmetry_regions": sample["gt"].get(
                    "active_symmetry_regions"
                ),
                "template_reference": template_row["template_path"],
                "template_sha256": template_row["template_sha256"],
                "symmetry_sidecar_reference": template_row["sidecar_path"],
                "symmetry_sidecar_sha256": template_row["sidecar_sha256"],
                "augmentation_metadata": augmented["metadata"],
                "data_contract_errors": [],
            }
        )
        sample["meta"].update(
            {
                "split": self.split,
                "dataset_index_fingerprint": self.index_fingerprint,
                "augmentation_metadata": sample["augmentation_metadata"],
                "data_contract_errors": sample["data_contract_errors"],
            }
        )
        return sample

    def augmentation_preview(self, index: int) -> dict[str, Any]:
        """Return one normal sample plus debug geometry for the preview CLI."""

        base = super().__getitem__(index)
        result = self._apply_boundary_augmentation(index, base)
        sample = result["base"]
        return {
            "sample": sample,
            "debug": result["debug"],
            "metadata": result["metadata"],
        }

    def write_index_artifacts(self, output_dir: str | Path) -> Path:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "dataset_index_fingerprint.json"
        _atomic_json(
            path,
            {
                "fingerprint": self.index_fingerprint,
                "fingerprint_payload": self.index_fingerprint_payload,
                "index_report": self.index_report,
            },
        )
        return path


__all__ = ["SplitDirectoryFragmentDataset"]
