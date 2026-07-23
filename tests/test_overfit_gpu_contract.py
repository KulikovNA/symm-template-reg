from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.config import load_config
from symm_template_reg.engine.history import STANDARD_FIELDS, TrainingHistory
from symm_template_reg.engine.overfit_trainer import (
    _is_improvement,
    _select_manifest_samples_by_scene,
)
from symm_template_reg.visualization.prediction_debug import select_debug_samples


class Faces840ConfigContractTest(unittest.TestCase):
    def test_config_is_explicit_best_only_shared_test_overfit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/debug/test_overfit_faces840_gpu.py")
        self.assertTrue(config["debug_training_on_test_split"])
        self.assertTrue(config["train_and_validation_use_same_samples"])
        self.assertTrue(config["results_are_not_final_evaluation"])
        self.assertEqual(config["data"]["fragment_mesh_filter"]["min_num_faces"], 840)
        self.assertEqual(config["data"]["validation_manifest"], "same_as_train")
        self.assertEqual(config["train"]["eval_interval_epochs"], 2)
        self.assertEqual(config["train"]["debug_visualization_interval_epochs"], 50)
        self.assertTrue(config["train"]["save_best_only"])
        self.assertFalse(config["train"]["save_periodic_checkpoints"])
        self.assertFalse(config["train"]["save_final_checkpoint"])
        self.assertEqual(
            config["experiment"]["work_dir_root"],
            "/home/nikita/disser/fragment-template-registration-lab/work_dirs",
        )

    def test_scene000000_config_inherits_v2_and_selects_one_scene(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/debug/test_overfit_faces840_scene000000_gpu.py"
        )
        self.assertEqual(config["data"]["scene_ids"], ("scene_000000",))
        self.assertEqual(config["data"]["expected_selected_samples"], 40)
        self.assertIsNone(config["data"]["max_train_samples"])
        self.assertIsNone(config["data"]["max_validation_samples"])
        self.assertEqual(config["train"]["max_epochs"], 101)
        self.assertEqual(config["loss"]["pose_query_ranking"]["type"], "soft_quality")

    def test_manifest_scene_selection_is_explicit(self) -> None:
        samples = [
            {"sample_id": "zero/a", "scene_id": "scene_000000"},
            {"sample_id": "one/a", "scene_id": "scene_000001"},
            {"sample_id": "zero/b", "scene_id": "scene_000000"},
        ]
        selected = _select_manifest_samples_by_scene(
            samples, ("scene_000000",)
        )
        self.assertEqual(
            [sample["sample_id"] for sample in selected], ["zero/a", "zero/b"]
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            _select_manifest_samples_by_scene(samples, ("scene_999999",))

    def test_min_delta_is_required_for_best_replacement(self) -> None:
        self.assertTrue(_is_improvement(1.0, None, mode="min", min_delta=1e-6))
        self.assertFalse(_is_improvement(0.9999995, 1.0, mode="min", min_delta=1e-6))
        self.assertTrue(_is_improvement(0.999998, 1.0, mode="min", min_delta=1e-6))


class HistoryContractTest(unittest.TestCase):
    def test_jsonl_is_incremental_and_has_standard_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            history = TrainingHistory(
                temporary,
                "run",
                {
                    "enabled": True,
                    "filename": "history/history.jsonl",
                    "flush_every_record": True,
                    "fsync": False,
                    "save_epoch_csv": True,
                },
            )
            history.record("run_start", epoch=0, global_step=0, phase="setup")
            first_size = history.path.stat().st_size
            history.record("train_epoch", epoch=1, global_step=2, phase="train")
            self.assertGreater(history.path.stat().st_size, first_size)
            rows = [json.loads(line) for line in history.path.read_text().splitlines()]
            self.assertEqual([row["record_type"] for row in rows], ["run_start", "train_epoch"])
            self.assertTrue(set(STANDARD_FIELDS).issubset(rows[0]))
            self.assertTrue(history.write_epoch_csv().is_file())


class DebugSelectionContractTest(unittest.TestCase):
    def test_fixed_debug_selection_spans_multiple_scenes(self) -> None:
        class FakeDataset:
            def __getitem__(self, index: int):
                scene = f"scene_{index % 4:06d}"
                return {
                    "sample_id": f"{scene}/frame_{index:06d}/fragment_0000",
                    "scene_id": scene,
                    "fragment_id": 0,
                    "observed": {"points_C": [None] * (index + 1)},
                    "gt": {"effective_symmetry_group": {"type": "C", "order": 1 + index % 2}},
                    "meta": {
                        "num_observed_points_raw": 100 + index,
                        "fragment_mesh": {"num_faces": 840 + index},
                    },
                }

        indices, entries = select_debug_samples(
            FakeDataset(), list(range(16)), count=8, seed=0
        )
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(set(indices)), 8)
        self.assertGreaterEqual(len({entry["scene_id"] for entry in entries}), 3)


if __name__ == "__main__":
    unittest.main()
