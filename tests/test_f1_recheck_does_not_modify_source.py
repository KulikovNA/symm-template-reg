import json, tempfile, unittest
from pathlib import Path
from tools.recheck_f1_coordinate_stage import recheck


class F1RecheckIntegrityTest(unittest.TestCase):
    def test_source_hashes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); run = root / "run"; (run / "checkpoints").mkdir(parents=True)
            leakage = root / "leakage.json"; leakage.write_text(json.dumps({"target_leakage_detected": False}))
            payloads = {
                "stage_gate.json": {}, "final_summary.json": {},
                "coordinate_metrics.json": {"epoch": 1, "aux_coordinate_p95_mm": 1.01, "aux_coordinate_rmse_mm": .52, "fine_feature_collision_fraction": 0, "fine_feature_variance": .1},
                "fine_feature_metrics.json": {},
                "resolved_config.json": {"target_leakage_policy": {"audit_path": str(leakage)}, "fine_stage_gate": {}},
                "checkpoints/best_metrics.json": {},
            }
            for name, value in payloads.items(): (run / name).write_text(json.dumps(value))
            (run / "checkpoints/best.pth").write_bytes(b"fixed")
            result = recheck(run, root / "out")
            self.assertFalse(result["stage_passed"])
            self.assertTrue(json.loads((root / "out/source_integrity.json").read_text())["source_unchanged"])


if __name__ == "__main__": unittest.main()

