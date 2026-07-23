"""Stage 05 (manual only): all four accepted scene_000000 fragments, 40 views."""

_base_ = ["04_k8_joint_finetune.py"]

experiment = dict(name="scene000000_05_all_fragments_joint")

data = dict(
    train_manifest=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
        "test_faces840_all_9e91dfb58d07.json"
    ),
    single_fragment_contract=False,
    expected_selected_samples=40,
    scene_id=None,
    fragment_id=None,
    scene_ids=("scene_000000",),
)

train = dict(
    max_optimizer_steps=4000,
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.0),
)

debug_visualization = dict(single_fragment_layout=False, num_samples=8)

stage = dict(
    name="scene_all_fragments_joint",
    checkpoint_filename="best_scene_top1.pth",
    requires_init_stage="joint_finetune",
    trainable_module_prefixes=None,
)
