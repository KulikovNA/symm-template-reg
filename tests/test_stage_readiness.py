from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.check_stage_readiness import check_stage_readiness


class StageReadinessTest(unittest.TestCase):
    def test_pose_stage_ready_only_with_metric_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "best_oracle_pose.pth"
            checkpoint.write_bytes(b"checkpoint")
            (checkpoint_dir / "best_manifest.json").write_text(
                json.dumps({"checkpoint_path": str(checkpoint)})
            )
            (checkpoint_dir / "best_metrics.json").write_text(
                json.dumps({"metrics": {"eval/oracle_topK_pose_success_5deg_5mm": 0.9}})
            )
            report = check_stage_readiness("pose_only", root)
            self.assertTrue(report["ready_for_next_stage"])


if __name__ == "__main__":
    unittest.main()
