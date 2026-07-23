_base_ = ["01_k1_direct_equal_exposure.py"]

experiment = dict(name="conditioned_v2_07_k1_curriculum")
curriculum = dict(
    enabled_only_via_manual_runner=True,
    levels=(1, 2, 4, 10),
    restore_optimizer=False,
    initialization="model_only",
)
stage = dict(name="conditioned_v2_k1_curriculum", checkpoint_filename="best_k1_curriculum.pth")
