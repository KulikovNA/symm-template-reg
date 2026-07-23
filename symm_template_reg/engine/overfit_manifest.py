"""Strict validation for the shared faces840 train/validation manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from symm_template_reg.datasets.fragment_mesh_filter import sha256_file
from symm_template_reg.config import canonical_point_policy
from symm_template_reg.engine.single_fragment import (
    validate_single_fragment_manifest_payload,
)
from symm_template_reg.engine.multifragment_overfit import (
    validate_multifragment_manifest_payload,
)


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_overfit_flags(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    for key, expected in WARNING_FLAGS.items():
        if config.get(key) is not expected or (
            isinstance(experiment, Mapping) and experiment.get(key) is not expected
        ):
            raise ValueError(f"faces840 overfit requires {key} = true")


def load_faces840_manifest(
    path: str | Path,
    config: Mapping[str, Any],
    dataset: Any,
) -> tuple[dict[str, Any], str]:
    validate_overfit_flags(config)
    manifest_path = Path(path).expanduser().resolve()
    if "<HASH>" in str(manifest_path):
        raise ValueError("faces840 train_manifest still contains unresolved <HASH>")
    file_digest = sha256_file(manifest_path)
    digest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not digest_sidecar.is_file():
        raise FileNotFoundError(f"manifest SHA256 sidecar is missing: {digest_sidecar}")
    expected_file_digest = digest_sidecar.read_text(encoding="ascii").split()[0]
    if file_digest != expected_file_digest:
        raise ValueError("faces840 manifest file SHA256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in WARNING_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"faces840 manifest requires {key} = true")
    internal_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if payload.get("manifest_sha256") != internal_digest:
        raise ValueError("faces840 manifest internal SHA256 mismatch")
    data = config["data"]
    manifest_filter = dict(payload.get("fragment_filter", {}))
    configured_filter = dict(data["fragment_mesh_filter"])
    # Phase policy does not alter which samples are stored in this already
    # filtered manifest.  V2 deliberately tightens validation to ``exclude``.
    manifest_filter["validation_policy"] = configured_filter.get(
        "validation_policy"
    )
    if manifest_filter != configured_filter:
        raise ValueError("faces840 manifest fragment filter differs from config")
    manifest_observed = dict(payload.get("observed_filter", {}))
    configured_observed = dict(data["observed_filter"])
    if "point_policy" in manifest_observed:
        manifest_observed["point_policy"] = canonical_point_policy(
            manifest_observed["point_policy"]
        )
    if "point_policy" in configured_observed:
        configured_observed["point_policy"] = canonical_point_policy(
            configured_observed["point_policy"]
        )
    if manifest_observed != configured_observed:
        raise ValueError("faces840 manifest observed filter differs from config")
    manifest_type = str(payload.get("manifest_type", ""))
    samples = payload.get("samples")
    if manifest_type == "single_fragment_overfit":
        if not bool(data.get("single_fragment_contract", False)):
            raise ValueError(
                "single-fragment manifest requires data.single_fragment_contract=True"
            )
        contract = validate_single_fragment_manifest_payload(
            payload,
            expected_samples=int(data.get("expected_selected_samples", 10)),
            min_num_faces=int(configured_filter.get("min_num_faces", 840)),
        )
        configured_scene = data.get("scene_id")
        configured_fragment = data.get("fragment_id")
        if configured_scene is not None and str(configured_scene) != contract["scene_id"]:
            raise ValueError("single-fragment manifest scene_id differs from config")
        if configured_fragment is not None and int(configured_fragment) != contract["fragment_id"]:
            raise ValueError("single-fragment manifest fragment_id differs from config")
    elif manifest_type == "four_fragments_four_views_overfit":
        if not bool(data.get("multifragment_contract", False)):
            raise ValueError(
                "four-fragment manifest requires data.multifragment_contract=True"
            )
        validate_multifragment_manifest_payload(
            payload,
            min_num_faces=int(configured_filter.get("min_num_faces", 840)),
        )
        if int(data.get("expected_selected_samples", -1)) != 16:
            raise ValueError("four-fragment config must expect exactly 16 samples")
    else:
        expected_counts = {
            "physical_fragments_total": 40,
            "accepted_physical_fragments": 36,
            "rejected_physical_fragments": 4,
            "observations_total": 400,
            "accepted_observations": 360,
            "rejected_observations": 40,
        }
        for key, expected in expected_counts.items():
            if int(payload.get(key, -1)) != expected:
                raise ValueError(f"faces840 manifest {key} must equal {expected}")
        if not isinstance(samples, list) or len(samples) != 360:
            raise ValueError("faces840 manifest must contain exactly 360 samples")
    assert isinstance(samples, list)
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("faces840 manifest contains duplicate sample IDs")
    if payload.get("train_sample_ids") != sample_ids:
        raise ValueError("train_sample_ids differ from manifest samples")
    if payload.get("validation_sample_ids") != sample_ids:
        raise ValueError("validation must use the same samples as train")
    if str(data.get("validation_manifest")) != "same_as_train":
        raise ValueError("config validation_manifest must be same_as_train")
    records = {record.sample_id: record for record in dataset.sample_records}
    accepted_keys = {
        key
        for key, decision in dataset.fragment_filter_decisions.items()
        if decision.accepted
    }
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id not in records:
            raise ValueError(f"manifest sample absent from filtered Dataset: {sample_id}")
        record = records[sample_id]
        key = (record.scene_id, record.fragment_id)
        if key not in accepted_keys:
            raise ValueError(f"rejected fragment appears in faces840 manifest: {key}")
        metadata = record.fragment_mesh_metadata
        if sample.get("fragment_mesh_sha256") != metadata.sha256:
            raise ValueError(f"fragment mesh SHA changed for {sample_id}")
        if int(sample.get("fragment_num_faces", -1)) != metadata.num_faces:
            raise ValueError(f"fragment face count changed for {sample_id}")
    first = dataset.template_repository.get(records[sample_ids[0]].object_model_id)
    template_path = Path(str(first["mesh_path"])).resolve()
    sidecar_path = Path(str(first["symmetry_sidecar_path"])).resolve()
    checks = (
        ("template_path", str(template_path)),
        ("template_sha256", sha256_file(template_path)),
        ("symmetry_sidecar_path", str(sidecar_path)),
        ("symmetry_sidecar_sha256", sha256_file(sidecar_path)),
    )
    for key, expected in checks:
        if payload.get(key) != expected:
            raise ValueError(f"faces840 manifest {key} changed")
    return payload, file_digest


__all__ = [
    "WARNING_FLAGS",
    "load_faces840_manifest",
    "validate_overfit_flags",
]
