"""Optional compatible-weight control; prepared only, never launched automatically."""

_base_ = ["four_fragments_four_views_scratch.py"]

experiment_type = "four_fragments_four_views_warmstart_control"
initialization_mode = "warm_start"
pretrained_checkpoint = "<TEN_VIEW_FRAGMENT0002_BEST_PTH>"

experiment = dict(
    name="coordinate_guided_surface_v3_four_fragments_four_views_warmstart_control",
    experiment_type="four_fragments_four_views_warmstart_control",
    initialization_mode="warm_start",
    pretrained_checkpoint="<TEN_VIEW_FRAGMENT0002_BEST_PTH>",
)

stage = dict(strict_initialization=False)

