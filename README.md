# symm-template-reg

Modular research baseline for direct, symmetry-aware registration of a visible
fragment to an explicit object template. Components are built from Python dict
configs through project-local registries; neighbouring repositories are audit and
architectural-reference inputs only, never runtime dependencies.

## What iteration one provides

- real synthetic-dataset loader with one `(scene, frame, fragment)` per sample;
- variable-size packed batches as the default, with an explicit padded adapter;
- cached template meshes, vertex normals, and configurable fine/coarse samples;
- optional template symmetry sidecars (`Cn` and continuous `SO2`);
- pure-PyTorch point encoding and bounded-token self/cross interaction;
- `K=8` direct SE(3) hypotheses plus overlap, visibility, correspondence,
  confidence, region, uncertainty, and insufficient-information outputs;
- symmetry-aware pose-set loss and dependency-free polynomial rectangular assignment;
- timestamped template/fragment symmetry visualization with an internal RGB PLY writer;
- real CPU forward and one finite optimizer-step smoke path.

This is an inspectable baseline, not a trained quality claim. PTv3, DFAT focus
attention, and PointDSC consistency are optional/deferred and are not used by the
baseline config.

## Installation

Use the already-active `fracs` environment. Do not replace PyTorch, CUDA,
torchvision, or NumPy.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
python -m pip install -e . --no-deps
```

The runtime requirements are only NumPy and PyTorch. No Open3D, SciPy, `plyfile`,
custom C++/CUDA extension, or adjacent checkout is required.

## Reproduce the checks

```bash
python -m compileall symm_template_reg tools tests configs
python tools/inspect_environment.py \
  --third-party-root /home/nikita/disser/fragment-template-registration-lab
python tools/inspect_dataset.py \
  --dataset-root /home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test \
  --out-dir work_dirs/dataset_inspection/2026-07-08-test
python tools/smoke_dataset.py \
  --config configs/symm_template_reg_baseline.py --num-samples 8
python tools/smoke_model.py \
  --config configs/symm_template_reg_baseline.py --device cpu
python tools/smoke_train_step.py \
  --config configs/symm_template_reg_baseline.py --device cpu
python -m unittest discover -s tests -v
```

## Controlled debug training on the test split

These commands are debugging/overfit checks only. Their results are explicitly
marked `results_are_not_final_evaluation=true`. First audit physical fragment
sizes; do not enable training until `min_num_faces` has been selected manually:

```bash
python tools/audit_fragment_mesh_sizes.py \
  --dataset-root /home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test \
  --output-dir work_dirs/fragment_size_audit/2026-07-08-test

# After setting the chosen threshold in the debug configs:
python tools/build_debug_manifests.py \
  --config configs/debug/tiny_overfit_pose_only.py \
  --output-dir work_dirs/debug_manifests

python tools/train.py \
  --config configs/debug/tiny_overfit_pose_only.py \
  --device cpu --max-steps 5
```

CUDA availability and the strict no-fallback CUDA path can be checked with
`python tools/check_cuda.py`. The complete coordinate, symmetry, assignment,
filtering, and inference contract is in
[docs/TRAINING_CONTRACT.md](docs/TRAINING_CONTRACT.md).

CUDA tools return a documented skip when `torch.cuda.is_available()` is false:

```bash
python tools/smoke_model.py \
  --config configs/symm_template_reg_baseline.py --device cuda
python tools/smoke_train_step.py \
  --config configs/symm_template_reg_baseline.py --device cuda
```

## Configuration and registries

The baseline is [configs/symm_template_reg_baseline.py](configs/symm_template_reg_baseline.py).
It builds all production components through these registries:

`MODELS`, `BACKBONES`, `ATTENTION`, `GEOMETRY_MODULES`, `MATCHERS`,
`HEADS`, `LOSSES`, `POSE_MODULES`, `SYMMETRY_MODULES`, `DATASETS`, and
`COLLATE_FUNCTIONS`.

The data root may be a direct split directory, as in the baseline, or a common
root plus `split="train"`, `"val"`, or `"test"`.

## Model output contract

For padded maxima `No`, `Nt`, configured pose queries `K`, uncertainty width `U`,
and region capacity `R`, `RegistrationPrediction` contains:

| field | shape |
|---|---|
| `pose_hypotheses` | `[B, K, 4, 4]` |
| `pose_logits` | `[B, K]` |
| `pose_uncertainty` | `[B, K, U]` |
| `observed_overlap_logits` | `[B, No]` |
| `template_visibility_logits` | `[B, Nt]` |
| `correspondence_points_O` | `[B, No, 3]` |
| `correspondence_confidence` | `[B, No]` |
| `observed_region_logits` | `[B, No, R]` |
| `active_region_logits` | `[B, R]` |
| `insufficient_information_logit` | `[B, 1]` |
| `observed_valid_mask`, `template_valid_mask` | `[B, No]`, `[B, Nt]` |

Dense padded outputs always carry masks; padding is not supervision. Direct pose
hypotheses are the inference output. Correspondences remain auxiliary and
diagnostic—there is no mandatory numerical pose solver.

## Symmetry

A sidecar lives beside a template as `<template-stem>.symmetry.json`. Its schema
is [schemas/template_symmetry.schema.json](schemas/template_symmetry.schema.json),
with an example in [examples](examples/). If absent, the loader returns no
metadata, sets `symmetry_available=false`, and does not silently assert `C1`.
Symmetry-region losses are then disabled; ordinary single-pose loss is only the
documented smoke/debug fallback.

The current real sidecar can be inspected without duplicating target math via
`tools/debug_symmetry_visualization.py`; see
[docs/SYMMETRY_DEBUG.md](docs/SYMMETRY_DEBUG.md).

## Audit and data documentation

- [Dataset format](docs/DATASET_FORMAT.md)
- [Third-party audit](docs/THIRD_PARTY_AUDIT.md)
- [Environment compatibility](docs/ENVIRONMENT_COMPATIBILITY.md)
- [Module porting plan](docs/MODULE_PORTING_PLAN.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Complete iteration-one report](docs/ITERATION1_REPORT.md)
- [Symmetry debug visualization](docs/SYMMETRY_DEBUG.md)
- [Debug training contract](docs/TRAINING_CONTRACT.md)
- [CUDA environment diagnosis](docs/CUDA_ENVIRONMENT_FIX.md)
- [faces840 controlled GPU overfit](docs/TEST_OVERFIT_FACES840.md)

Every influenced clean-room module records the exact reference repository,
commit, original paths, license, and project-local changes in its header. The
machine-readable inventory is `third_party_modules.json`.
