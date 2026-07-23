"""Dataset for synthetic fragment-to-template registration samples."""

from __future__ import annotations

import json
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from symm_template_reg.config import canonical_point_policy

from .structures import DatasetSampleRecord
from .fragment_mesh_filter import (
    FragmentMeshFilter,
    FragmentMeshMetadata,
    REQUIRED_TEST_SPLIT_FLAGS,
    scan_fragment_mesh_metadata,
)
from .template_repository import TemplateRepository
from .transforms import ObservedPointSelector

try:
    from symm_template_reg.registry import DATASETS
except (ImportError, AttributeError):  # package bootstrap remains importable
    DATASETS = None


def _register_dataset(cls: type[Any]) -> type[Any]:
    if DATASETS is not None and DATASETS.get(cls.__name__) is None:
        DATASETS.register_module(cls)
    return cls


def resolve_split_root(root: str | Path, split: str | None = None) -> Path:
    """Resolve either a direct split path or a common root plus split name."""

    root = Path(root).expanduser().resolve()
    if split is not None:
        candidate = root / split
        if candidate.is_dir():
            root = candidate
        elif root.name != split:
            raise FileNotFoundError(f"split {split!r} not found below {root}")
    if not (root / "models").is_dir() or not any(root.glob("scene_*")):
        raise FileNotFoundError(
            f"{root} is not a dataset split (expected models/ and scene_*/ directories)"
        )
    return root


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _symmetry_targets(
    points_O: torch.Tensor,
    T_C_from_O: torch.Tensor,
    metadata: Any,
    activity_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the optional symmetry package while keeping the dataset standalone."""

    if metadata is None:
        return {
            "active_symmetry_regions": None,
            "effective_symmetry_group": None,
            "equivalent_T_C_from_O": None,
        }
    from symm_template_reg.models.symmetry.targets import build_fragment_symmetry_targets

    activity = dict(activity_config or {})
    return build_fragment_symmetry_targets(
        points_O,
        metadata,
        base_pose=T_C_from_O,
        min_points=int(activity.get("min_points", 1)),
        min_fraction=float(activity.get("min_fraction", 0.0)),
        assignment_tolerance_m=float(activity.get("boundary_tolerance_m", 1e-6)),
    ).to_dataset_dict()


@_register_dataset
class FragmentTemplateRegistrationDataset(Dataset[dict[str, Any]]):
    """One item is one visible fragment in one camera frame.

    The source NPZ contains every visible fragment in a frame.  Indexing groups
    its rows by ``fragment_id`` and never assumes a fixed point count.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        root: str | Path | None = None,
        data_root: str | Path | None = None,
        split: str | None = None,
        observed_policy: str | None = None,
        input_policy: str | None = None,
        min_observed_points: int = 128,
        max_observed_points: int | None = 4096,
        fragment_mesh_filter: Mapping[str, Any] | None = None,
        observed_filter: Mapping[str, Any] | None = None,
        symmetry_region_activity: Mapping[str, Any] | None = None,
        fragment_mesh_cache_dir: str | Path = "work_dirs/cache",
        voxel_size_m: float = 0.002,
        random_seed: int = 0,
        template_fine_points: int | None = 4096,
        template_coarse_points: int | None = 1024,
        template_repository: TemplateRepository | None = None,
        registration_point_selection: str = "all_fragment_points",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_samples: int | None = None,
    ) -> None:
        selected_root = dataset_root if dataset_root is not None else root
        selected_root = selected_root if selected_root is not None else data_root
        if selected_root is None:
            raise TypeError("one of dataset_root, root or data_root is required")
        self.dataset_root = resolve_split_root(selected_root, split)
        self.split = split or self.dataset_root.name
        legacy_policy = input_policy or observed_policy
        observed_config = {
            "min_observed_points": min_observed_points,
            "max_observed_points": max_observed_points,
            "point_policy": legacy_policy or "precomputed_dataset_points",
        }
        if observed_filter is not None:
            unknown = set(observed_filter).difference(observed_config)
            if unknown:
                raise ValueError(f"unknown observed_filter fields: {sorted(unknown)}")
            filter_config = dict(observed_filter)
            if legacy_policy is not None and "point_policy" in filter_config:
                if canonical_point_policy(legacy_policy) != canonical_point_policy(
                    filter_config["point_policy"]
                ):
                    raise ValueError(
                        "conflicting observed point policies: observed_filter.point_policy="
                        f"{filter_config['point_policy']!r}, observed_policy={legacy_policy!r}"
                    )
            observed_config.update(filter_config)
        requested_point_policy = canonical_point_policy(observed_config["point_policy"])
        self.min_observed_points = int(observed_config["min_observed_points"])
        self.max_observed_points = (
            int(observed_config["max_observed_points"])
            if observed_config["max_observed_points"] is not None
            else None
        )
        self.observed_filter_config = {
            **observed_config,
            "point_policy": requested_point_policy,
        }
        self.symmetry_region_activity = {
            "min_points": int((symmetry_region_activity or {}).get("min_points", 1)),
            "min_fraction": float((symmetry_region_activity or {}).get("min_fraction", 0.0)),
            "boundary_tolerance_m": float(
                (symmetry_region_activity or {}).get("boundary_tolerance_m", 1e-6)
            ),
        }
        self.selector = ObservedPointSelector(
            policy=requested_point_policy,
            min_points=self.min_observed_points,
            max_points=self.max_observed_points,
            voxel_size_m=float(voxel_size_m),
            random_seed=int(random_seed),
        )
        self.transform = transform
        from symm_template_reg.models.pose.pose_codec import PoseCodec

        self.pose_codec = PoseCodec()
        self.template_repository = template_repository or TemplateRepository(
            self.dataset_root / "models",
            fine_points=template_fine_points,
            coarse_points=template_coarse_points,
        )
        if registration_point_selection not in {"all_fragment_points", "shell_only"}:
            raise ValueError(
                "registration_point_selection must be all_fragment_points or shell_only"
            )
        self.registration_point_selection = registration_point_selection
        self.fragment_mesh_filter = FragmentMeshFilter(fragment_mesh_filter)
        self.fragment_metadata_by_id, self.fragment_mesh_cache_report = (
            scan_fragment_mesh_metadata(
                self.dataset_root,
                filter_config=self.fragment_mesh_filter.config,
                cache_dir=fragment_mesh_cache_dir,
            )
        )
        self.fragment_filter_decisions = self.fragment_mesh_filter.filter_fragments(
            self.fragment_metadata_by_id
        )
        self._records, index_report = self._build_index(max_samples=max_samples)
        self.index_report = index_report
        self.skipped_too_small = int(index_report["rejected_observed_points_too_few"])
        self._observed_lengths_cache: list[int] | None = None
        if not self._records:
            raise ValueError(
                f"no usable samples in {self.dataset_root}; "
                f"min_observed_points={self.min_observed_points}"
            )

    @property
    def sample_records(self) -> tuple[DatasetSampleRecord, ...]:
        return tuple(self._records)

    @property
    def observed_lengths(self) -> list[int]:
        if self._observed_lengths_cache is None:
            if self.selector.policy == "all_points" or self.max_observed_points is None:
                lengths = [record.num_observed_points for record in self._records]
            elif self.selector.policy != "voxel_downsample":
                lengths = [
                    min(record.num_observed_points, self.max_observed_points)
                    for record in self._records
                ]
            else:
                lengths = []
                for index, record in enumerate(self._records):
                    with np.load(record.visible_points_path, allow_pickle=False) as arrays:
                        mask = arrays["fragment_id"] == record.fragment_id
                        if self.registration_point_selection == "shell_only":
                            mask &= arrays["surface_label"] == 0
                        points = np.asarray(arrays["points_C"][mask], dtype=np.float32)
                    lengths.append(len(self.selector.indices(points, sample_seed=index)))
            self._observed_lengths_cache = lengths
        return list(self._observed_lengths_cache)

    def _build_index(
        self, *, max_samples: int | None
    ) -> tuple[list[DatasetSampleRecord], dict[str, Any]]:
        records: list[DatasetSampleRecord] = []
        rejected_samples: list[dict[str, Any]] = []
        total_observations = 0
        rejected_physical = 0
        rejected_observed = 0
        for scene_dir in sorted(self.dataset_root.glob("scene_*")):
            gt_path = scene_dir / "gt_annotations.json"
            if not gt_path.is_file():
                continue
            gt = _json(gt_path)
            scene_id = str(gt.get("scene_id", scene_dir.name))
            scene_meta = _json(scene_dir / "scene_meta.json") if (scene_dir / "scene_meta.json").is_file() else {}
            object_model_rel = scene_meta.get("object_model")
            if object_model_rel:
                object_model_id = Path(str(object_model_rel)).stem
            else:
                fragment_meta = _json(scene_dir / "fragments" / "fragment_annotations.json")
                object_model_id = Path(str(fragment_meta["object_model"])).stem
            for frame in gt.get("frames", []):
                frame_id = int(frame["frame_id"])
                npz_path = scene_dir / str(
                    frame.get("visible_points", f"visible_points/frame_{frame_id:06d}.npz")
                )
                if not npz_path.is_file():
                    continue
                with np.load(npz_path, allow_pickle=False) as arrays:
                    if "fragment_id" not in arrays or "points_C" not in arrays:
                        raise KeyError(f"{npz_path} misses fragment_id or points_C")
                    registration_rows = np.ones(len(arrays["fragment_id"]), dtype=bool)
                    if self.registration_point_selection == "shell_only":
                        if "surface_label" not in arrays:
                            raise ValueError(
                                f"shell_only registration requires surface_label: {npz_path}"
                            )
                        registration_rows &= arrays["surface_label"] == 0
                    fragment_ids, counts = np.unique(
                        arrays["fragment_id"][registration_rows], return_counts=True
                    )
                T_C_from_W = frame.get("T_C_from_W")
                annotations = {
                    int(fragment["fragment_id"]): {
                        **fragment,
                        "T_C_from_W": T_C_from_W,
                    }
                    for fragment in frame.get("fragments", [])
                }
                for fragment_id, count in zip(fragment_ids.tolist(), counts.tolist()):
                    fragment_id = int(fragment_id)
                    count = int(count)
                    total_observations += 1
                    metadata_key = (scene_id, fragment_id)
                    if metadata_key not in self.fragment_filter_decisions:
                        raise KeyError(
                            f"observation {scene_id}/fragment_{fragment_id:04d} has no physical mesh metadata"
                        )
                    decision = self.fragment_filter_decisions[metadata_key]
                    sample_id = f"{scene_id}/frame_{frame_id:06d}/fragment_{fragment_id:04d}"
                    if not decision.accepted:
                        rejected_physical += 1
                        rejected_samples.append(
                            {
                                "sample_id": sample_id,
                                "scene_id": scene_id,
                                "frame_id": frame_id,
                                "fragment_id": fragment_id,
                                "num_observed_points": count,
                                "rejection_reason": "physical_fragment_rejected",
                                "detail": ";".join(decision.reasons),
                            }
                        )
                        continue
                    if count < self.min_observed_points:
                        rejected_observed += 1
                        rejected_samples.append(
                            {
                                "sample_id": sample_id,
                                "scene_id": scene_id,
                                "frame_id": frame_id,
                                "fragment_id": fragment_id,
                                "num_observed_points": count,
                                "rejection_reason": "observed_points_too_few",
                                "detail": f"{count} < {self.min_observed_points}",
                            }
                        )
                        continue
                    if fragment_id not in annotations:
                        raise KeyError(
                            f"fragment {fragment_id} in {npz_path} has no GT annotation"
                        )
                    records.append(
                        DatasetSampleRecord(
                            sample_id=sample_id,
                            scene_id=scene_id,
                            frame_id=frame_id,
                            fragment_id=fragment_id,
                            fragment_key=decision.metadata.fragment_key,
                            object_model_id=object_model_id,
                            visible_points_path=npz_path,
                            num_observed_points=count,
                            gt_fragment=annotations[fragment_id],
                            scene_meta=scene_meta,
                            fragment_mesh_metadata=decision.metadata,
                        )
                    )
        full_accepted = len(records)
        if max_samples is not None:
            records = records[: int(max_samples)]
        report = {
            **self.fragment_mesh_cache_report,
            **self.fragment_mesh_filter.to_report(),
            "observed_filter": dict(self.observed_filter_config),
            "total_frame_observations": total_observations,
            "accepted_frame_observations_before_max_samples": full_accepted,
            "accepted_frame_observations": len(records),
            "rejected_because_physical_fragment": rejected_physical,
            "rejected_observed_points_too_few": rejected_observed,
            "rejected_samples": rejected_samples,
            "max_samples": max_samples,
        }
        return records, report

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        with np.load(record.visible_points_path, allow_pickle=False) as arrays:
            fragment_mask = arrays["fragment_id"] == record.fragment_id
            if self.registration_point_selection == "shell_only":
                if "surface_label" not in arrays:
                    raise ValueError("shell_only registration requires surface_label")
                fragment_mask &= arrays["surface_label"] == 0
            fragment_rows = np.flatnonzero(fragment_mask)
            points_C_all = np.asarray(arrays["points_C"][fragment_rows], dtype=np.float32)
            selected_local = self.selector.indices(points_C_all, sample_seed=index)
            rows = fragment_rows[selected_local]
            points_C = torch.from_numpy(np.asarray(arrays["points_C"][rows], dtype=np.float32).copy())
            points_O = (
                torch.from_numpy(np.asarray(arrays["points_O"][rows], dtype=np.float32).copy())
                if "points_O" in arrays
                else None
            )
            surface_labels = (
                torch.from_numpy(np.asarray(arrays["surface_label"][rows], dtype=np.int64).copy())
                if "surface_label" in arrays
                else None
            )
            source_fields = tuple(arrays.files)
        T_C_from_O = torch.as_tensor(record.gt_fragment["T_C_from_O"], dtype=torch.float32)
        T_C_from_W_value = record.gt_fragment.get("T_C_from_W")
        T_C_from_W = (
            torch.as_tensor(T_C_from_W_value, dtype=torch.float32)
            if T_C_from_W_value is not None
            else None
        )
        T_W_from_C = torch.linalg.inv(T_C_from_W) if T_C_from_W is not None else None
        pose_context = self.pose_codec.context(
            points_C.unsqueeze(0), torch.ones((1, len(points_C)), dtype=torch.bool)
        )
        encoded_pose = self.pose_codec.encode_transform(
            T_C_from_O.unsqueeze(0),
            pose_context.observed_centroid_C,
            pose_context.observed_scale,
        )[0]
        cached_template = self.template_repository.get(record.object_model_id)
        # Samples own their tensors so transforms cannot corrupt the repository cache.
        template = {
            key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for key, value in cached_template.items()
        }
        symmetry_metadata = template.get("symmetry_metadata")
        symmetry_targets = (
            _symmetry_targets(
                points_O,
                T_C_from_O,
                symmetry_metadata,
                self.symmetry_region_activity,
            )
            if points_O is not None
            else {
                "active_symmetry_regions": None,
                "effective_symmetry_group": None,
                "equivalent_T_C_from_O": None,
            }
        )
        sample: dict[str, Any] = {
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "frame_id": record.frame_id,
            "fragment_id": record.fragment_id,
            "object_model_id": template["object_model_id"],
            "observed": {
                "points_C": points_C,
                "normals_C": None,
                "surface_labels": surface_labels,
                "valid_mask": torch.ones(len(points_C), dtype=torch.bool),
            },
            "template": template,
            "gt": {
                "T_C_from_O": T_C_from_O,
                "T_W_from_C": T_W_from_C,
                "pose_parameters_normalized": encoded_pose,
                "observed_centroid_C": pose_context.observed_centroid_C[0],
                "observed_scale": pose_context.observed_scale[0],
                "T_C_from_F": (
                    torch.as_tensor(record.gt_fragment["T_C_from_F"], dtype=torch.float32)
                    if "T_C_from_F" in record.gt_fragment
                    else None
                ),
                "points_O_corresponding": points_O,
                # Shell points lie on the digital-twin surface; fracture points do not.
                "overlap_labels": surface_labels.eq(0) if surface_labels is not None else None,
                **symmetry_targets,
            },
            "meta": {
                "coord_unit": "m",
                "coordinate_convention": "BOP/OpenCV: X right, Y down, Z forward",
                "matrix_convention": "column-vector homogeneous transforms",
                "symmetry_available": symmetry_metadata is not None,
                "symmetry_sidecar_path": template.get("symmetry_sidecar_path"),
                "observed_policy": self.selector.policy,
                "registration_point_selection": self.registration_point_selection,
                "symmetry_region_activity": dict(self.symmetry_region_activity),
                "num_observed_points_raw": record.num_observed_points,
                "num_observed_points": len(points_C),
                "npz_fields": source_fields,
                "visible_points_path": str(record.visible_points_path),
                "fragment_mesh": {
                    "num_vertices": record.fragment_mesh_metadata.num_vertices,
                    "num_faces": record.fragment_mesh_metadata.num_faces,
                    "surface_area_m2": record.fragment_mesh_metadata.surface_area_m2,
                    "bbox_min": list(record.fragment_mesh_metadata.bbox_min),
                    "bbox_max": list(record.fragment_mesh_metadata.bbox_max),
                    "bbox_diagonal_m": record.fragment_mesh_metadata.bbox_diagonal_m,
                    "mesh_path": str(record.fragment_mesh_metadata.mesh_path),
                    "sha256": record.fragment_mesh_metadata.sha256,
                    "passed_training_size_filter": True,
                },
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def write_filter_artifacts(self, output_dir: str | Path) -> Path:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "fragment_filter_config.json").write_text(
            json.dumps(self.fragment_mesh_filter.config, indent=2) + "\n",
            encoding="utf-8",
        )
        decisions = list(self.fragment_filter_decisions.values())

        def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        fragment_fields = [
            "scene_id", "fragment_id", "fragment_key", "num_faces", "num_vertices",
            "surface_area_m2", "bbox_diagonal_m", "rejection_reasons",
            "observations_removed", "mesh_path", "mesh_sha256",
        ]
        observation_counts: dict[tuple[str, int], int] = {}
        for rejected in self.index_report["rejected_samples"]:
            if rejected["rejection_reason"] == "physical_fragment_rejected":
                key = (str(rejected["scene_id"]), int(rejected["fragment_id"]))
                observation_counts[key] = observation_counts.get(key, 0) + 1
        fragment_rows = []
        for decision in decisions:
            metadata = decision.metadata
            fragment_rows.append(
                {
                    "scene_id": metadata.scene_id,
                    "fragment_id": metadata.fragment_id,
                    "fragment_key": metadata.fragment_key,
                    "num_faces": metadata.num_faces,
                    "num_vertices": metadata.num_vertices,
                    "surface_area_m2": metadata.surface_area_m2,
                    "bbox_diagonal_m": metadata.bbox_diagonal_m,
                    "rejection_reasons": ";".join(decision.reasons),
                    "observations_removed": observation_counts.get(
                        (metadata.scene_id, metadata.fragment_id), 0
                    ),
                    "mesh_path": str(metadata.mesh_path),
                    "mesh_sha256": metadata.sha256,
                }
            )
        write_csv(
            destination / "accepted_fragments.csv",
            fragment_fields,
            [row for row, decision in zip(fragment_rows, decisions) if decision.accepted],
        )
        write_csv(
            destination / "rejected_fragments.csv",
            fragment_fields,
            [row for row, decision in zip(fragment_rows, decisions) if not decision.accepted],
        )
        sample_fields = [
            "sample_id", "scene_id", "frame_id", "fragment_id",
            "num_observed_points", "rejection_reason", "detail",
        ]
        accepted_sample_rows = [
            {
                "sample_id": record.sample_id,
                "scene_id": record.scene_id,
                "frame_id": record.frame_id,
                "fragment_id": record.fragment_id,
                "num_observed_points": record.num_observed_points,
                "rejection_reason": "",
                "detail": "",
            }
            for record in self._records
        ]
        write_csv(destination / "accepted_samples.csv", sample_fields, accepted_sample_rows)
        write_csv(
            destination / "rejected_samples.csv",
            sample_fields,
            list(self.index_report["rejected_samples"]),
        )
        summary = dict(self.index_report)
        summary.pop("decisions", None)
        summary.pop("rejected_samples", None)
        (destination / "fragment_filter_summary.json").write_text(
            json.dumps({**REQUIRED_TEST_SPLIT_FLAGS, **summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination


__all__ = ["FragmentTemplateRegistrationDataset", "resolve_split_root"]
