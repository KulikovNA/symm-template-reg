"""Strict debug sample-manifest construction and training-time validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from symm_template_reg.datasets.fragment_mesh_filter import (
    REQUIRED_TEST_SPLIT_FLAGS,
    sha256_file,
)


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def validate_debug_training_flags(config: Mapping[str, Any]) -> None:
    for key, expected in REQUIRED_TEST_SPLIT_FLAGS.items():
        if config.get(key) is not expected:
            raise ValueError(f"debug test-split contract requires {key} = true")


def _filter_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    if isinstance(data, Mapping) and isinstance(data.get("fragment_mesh_filter"), Mapping):
        return dict(data["fragment_mesh_filter"])
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping) and isinstance(dataset.get("fragment_mesh_filter"), Mapping):
        return dict(dataset["fragment_mesh_filter"])
    raise ValueError("config has no data.fragment_mesh_filter section")


def load_and_validate_manifest(
    manifest_path: str | Path,
    config: Mapping[str, Any],
    dataset: Any,
) -> tuple[dict[str, Any], str]:
    validate_debug_training_flags(config)
    path = Path(manifest_path).expanduser().resolve()
    digest = manifest_sha256(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.is_file():
        raise FileNotFoundError(f"manifest SHA256 sidecar is missing: {digest_path}")
    expected_digest = digest_path.read_text(encoding="ascii").strip().split()[0]
    if digest != expected_digest:
        raise ValueError("sample manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample manifest root must be an object")
    for key, expected in REQUIRED_TEST_SPLIT_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"manifest requires {key} = true")
    config_filter = _filter_config(config)
    manifest_filter = payload.get("fragment_filter")
    if not isinstance(manifest_filter, Mapping):
        raise ValueError("manifest has no fragment_filter object")
    if manifest_filter.get("min_num_faces") != config_filter.get("min_num_faces"):
        raise ValueError(
            "manifest fragment face threshold does not match config min_num_faces"
        )
    if bool(manifest_filter.get("enabled")) != bool(config_filter.get("enabled")):
        raise ValueError("manifest fragment filter enabled state does not match config")
    accepted = {
        key
        for key, decision in dataset.fragment_filter_decisions.items()
        if decision.accepted
    }
    records = {record.sample_id: record for record in dataset.sample_records}
    for sample in payload.get("samples", []):
        sample_id = str(sample["sample_id"])
        if sample_id not in records:
            raise ValueError(f"manifest sample is not in the filtered Dataset: {sample_id}")
        record = records[sample_id]
        key = (record.scene_id, record.fragment_id)
        if key not in accepted:
            raise ValueError(f"rejected physical fragment appears in manifest: {key}")
        metadata = dataset.fragment_metadata_by_id[key]
        if sample.get("fragment_mesh_sha256") != metadata.sha256:
            raise ValueError(f"fragment mesh SHA256 changed for {sample_id}")
        if int(sample.get("fragment_num_faces", -1)) != metadata.num_faces:
            raise ValueError(f"fragment mesh face count changed for {sample_id}")
    return payload, digest


__all__ = [
    "load_and_validate_manifest",
    "manifest_sha256",
    "validate_debug_training_flags",
]
