from __future__ import annotations

import unittest

import hashlib
import json
import tempfile
from pathlib import Path

from symm_template_reg.config import load_config
from symm_template_reg.engine.single_fragment import validate_single_fragment_manifest_payload
from symm_template_reg.engine.view_ladder import subset_view_manifest
from tests.test_single_frame_manifest import source_manifest
from tools.build_view_ladder_manifests import build_view_ladder


class ViewLadderManifestTest(unittest.TestCase):
    def test_progressive_subsets_preserve_one_mesh_and_requested_order(self) -> None:
        for frames in ([4, 8], [4, 5, 2, 8], list(range(10))):
            payload = subset_view_manifest(source_manifest(), frames)
            report = validate_single_fragment_manifest_payload(
                payload, expected_samples=len(frames)
            )
            self.assertEqual([item["frame_id"] for item in payload["samples"]], frames)
            self.assertEqual(report["fragment_mesh_sha256"], "mesh")

    def test_configs_keep_deterministic_fp32_pose_only_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name, queries, auxiliary in (
            ("k1_pose_only.py", 1, 0.0),
            ("k8_pose_only.py", 8, 0.0),
            ("k8_pose_only_aux_decoder.py", 8, 0.5),
        ):
            config = load_config(root / "configs/debug/view_ladder" / name)
            self.assertEqual(config["model"]["pose_head"]["num_queries"], queries)
            self.assertEqual(
                config["loss"]["pose_decoder_auxiliary_weight"], auxiliary
            )
            self.assertFalse(config["train"]["amp"])
            self.assertEqual(config["train"]["scheduler"]["type"], "constant")
            self.assertEqual(config["data"]["train_batch_size"], 1)
            self.assertEqual(config["dataset"]["random_seed"], 0)

    def test_builder_refuses_to_overwrite_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            encoded = (json.dumps(source_manifest(), indent=2) + "\n").encode()
            source.write_bytes(encoded)
            source.with_suffix(".json.sha256").write_text(
                f"{hashlib.sha256(encoded).hexdigest()}  source.json\n"
            )
            output = root / "ladder"
            build_view_ladder(source, output)
            with self.assertRaises(FileExistsError):
                build_view_ladder(source, output)


if __name__ == "__main__":
    unittest.main()
