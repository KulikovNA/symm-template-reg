"""Faces840 V2 GPU overfit on scene_000000 only.

Training, validation, and fixed debug visualizations use the same 40 accepted
observations from the zero-indexed scene.  The full strict faces840 manifest is
still validated before its scene_000000 subset is selected.
"""

_base_ = ["test_overfit_faces840_gpu.py"]

experiment = dict(
    name="test_overfit_faces840_scene000000_gpu_v2",
)

data = dict(
    scene_ids=("scene_000000",),
    expected_selected_samples=40,
    max_train_samples=None,
    max_validation_samples=None,
)
