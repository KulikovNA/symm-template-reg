import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


TOOL = Path(__file__).resolve().parents[1] / "tools" / "recheck_patch_classifier_stage.py"
SPEC = importlib.util.spec_from_file_location("recheck_patch_classifier_stage", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PatchStageRecheckTest(unittest.TestCase):
    def test_recheck_keeps_source_signatures_unchanged(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "old"
            (run / "checkpoints").mkdir(parents=True)
            capacity = root / "capacity.json"
            capacity.write_text(json.dumps({"capacity_ok": True}))
            leakage = root / "leakage.json"
            leakage.write_text(json.dumps({"target_leakage_detected": False}))
            config = {
                "stage_gate_dependencies": {
                    "parameterization_capacity_path": str(capacity),
                    "parameterization_capacity_required_field": "capacity_ok",
                },
                "target_leakage_policy": {"audit_path": str(leakage)},
            }
            (run / "resolved_config.json").write_text(json.dumps(config))
            (run / "checkpoints" / "best.pth").write_bytes(b"checkpoint")
            (run / "checkpoints" / "best_metrics.json").write_text("{}")
            (run / "stage_gate.json").write_text("{}")
            (run / "final_summary.json").write_text(json.dumps({"status": "ok"}))
            output = root / "new"
            output.mkdir()
            metrics = {
                "single_owner_top1_accuracy": 0.9,
                "valid_patch_set_top1_accuracy": 0.96,
                "valid_patch_set_top4_recall": 1.0,
                "valid_patch_set_in_candidate_set_fraction": 1.0,
                "unique_predicted_patches": 3,
                "most_popular_patch_fraction": 0.4,
                "nonfinite_detected": False,
            }
            with patch.object(MODULE, "audit_checkpoint", return_value=(metrics, [])):
                result = MODULE.recheck(
                    run, run / "checkpoints" / "best.pth", output, torch.device("cpu")
                )
            self.assertTrue(result["source_run_unchanged"])
            self.assertTrue(result["candidate_stage_passed"])


if __name__ == "__main__":
    unittest.main()
