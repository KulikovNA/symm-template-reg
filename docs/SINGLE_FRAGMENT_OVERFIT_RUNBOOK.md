# Single-fragment staged overfit runbook

This is debug training on the test split. Train and validation intentionally use
the same observations; none of these metrics is a final evaluation result.

Repository and fixed inputs:

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

DATASET=/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test
MANIFEST=/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json
```

The audit found that fragments `0`, `1`, `2`, and `3` all pass the six required
binary criteria. Fragment `2` is used as the explicit working example from the
request, not as an automatically selected winner.

## A. Audit candidates

The command below refuses to overwrite existing report files. Use a new output
directory when repeating it.

```bash
python tools/audit_single_fragment_candidates.py \
  --dataset-root "$DATASET" \
  --scene-id scene_000000 \
  --min-fragment-faces 840 \
  --min-observed-points 128 \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/single_fragment_audit_new
```

Inspect `single_fragment_candidates.md`, CSV, and JSON. Do not silently switch
fragment IDs.

## B. Build the selected manifest

The checked manifest already exists at `$MANIFEST`. To rebuild it after a
dataset change, use a new output directory and then update the config/variable:

```bash
python tools/build_single_fragment_manifest.py \
  --dataset-root "$DATASET" \
  --scene-id scene_000000 \
  --fragment-id 2 \
  --min-fragment-faces 840 \
  --min-observed-points 128 \
  --max-observed-points 4096 \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests
```

Expected checked artifact:

```text
/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json
```

It contains 10 different frame IDs, one `fragment_id`, and one physical mesh
SHA256.

## C. Compile and test

```bash
python -m compileall symm_template_reg tools tests configs
python -m unittest discover -s tests -v
```

## D. Region target audit

Run this before Stage 03, using a new output directory:

```bash
python tools/audit_region_distribution.py \
  --config configs/debug/single_fragment/03_k8_regions_only.py \
  --manifest "$MANIFEST" \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/single_fragment_region_audit_new
```

For fragment 2, every point class (`band_00` through `band_03`) is present, but
every active-region target is positive in all 10 views. Therefore active-set
accuracy alone is not evidence that the head learned negative classes or
generalizes.

## E. Stage 01: K=8 pose only

```bash
python tools/train.py \
  --config configs/debug/single_fragment/01_k8_pose_only.py \
  --device cuda \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train"
```

Copy the printed `run_dir` exactly:

```bash
STAGE01_RUN=/home/nikita/disser/fragment-template-registration-lab/work_dirs/<printed-stage01-run-id>
```

Evaluate and check the manual gate:

```bash
python tools/evaluate.py \
  --config configs/debug/single_fragment/01_k8_pose_only.py \
  --checkpoint "$STAGE01_RUN/checkpoints/best_oracle_pose.pth" \
  --manifest "$MANIFEST" \
  --device cuda

python tools/check_stage_readiness.py \
  --stage pose_only \
  --run-dir "$STAGE01_RUN"
```

Do not start ranking unless
`eval/oracle_topK_pose_success_5deg_5mm >= 0.9` and the checkpoint exists.

## F. Optional K=1 diagnostic

Run only if Stage 01 fails after 3000 optimizer steps. It is diagnostic only and
must never initialize K=8.

```bash
python tools/train.py \
  --config configs/debug/single_fragment/optional_k1_pose_diagnostic.py \
  --device cuda \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train"
```

## G. Stage 02: K=8 categorical ranking only

`--init-checkpoint` loads model weights only. It deliberately does not restore
optimizer, scheduler, scaler, or counters.

```bash
python tools/train.py \
  --config configs/debug/single_fragment/02_k8_ranking_only.py \
  --device cuda \
  --init-checkpoint "$STAGE01_RUN/checkpoints/best_oracle_pose.pth" \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train"

STAGE02_RUN=/home/nikita/disser/fragment-template-registration-lab/work_dirs/<printed-stage02-run-id>

python tools/evaluate.py \
  --config configs/debug/single_fragment/02_k8_ranking_only.py \
  --checkpoint "$STAGE02_RUN/checkpoints/best_top1_ranking.pth" \
  --manifest "$MANIFEST" \
  --device cuda

python tools/check_stage_readiness.py \
  --stage ranking_only \
  --run-dir "$STAGE02_RUN"
```

The gate is `eval/top1_query_is_oracle >= 0.9`. Spearman is defined as
`correlation(pose_logit, -pose_cost)`, so positive is good. To audit a manual
evaluation, copy its printed output directory and run:

```bash
STAGE02_EVAL=/home/nikita/disser/fragment-template-registration-lab/work_dirs/<stage02-run-id>/manual_evaluations/<printed-eval-id>

python tools/audit_ranking_targets.py \
  --per-sample-metrics "$STAGE02_EVAL/per_sample_metrics.csv" \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/ranking_audit_new
```

## H. Stage 03: region heads only

```bash
python tools/train.py \
  --config configs/debug/single_fragment/03_k8_regions_only.py \
  --device cuda \
  --init-checkpoint "$STAGE02_RUN/checkpoints/best_top1_ranking.pth" \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train"

STAGE03_RUN=/home/nikita/disser/fragment-template-registration-lab/work_dirs/<printed-stage03-run-id>

python tools/check_stage_readiness.py \
  --stage regions_only \
  --run-dir "$STAGE03_RUN"
```

Only `symmetry_head` is trainable. Pose coordinates and ranking logits stay
fixed. Interpret point-region macro F1/confusion in addition to active-set
accuracy because this fragment has no negative active targets.

## I. Stage 04: joint fine-tune

```bash
python tools/train.py \
  --config configs/debug/single_fragment/04_k8_joint_finetune.py \
  --device cuda \
  --init-checkpoint "$STAGE03_RUN/checkpoints/best_regions.pth" \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train"

STAGE04_RUN=/home/nikita/disser/fragment-template-registration-lab/work_dirs/<printed-stage04-run-id>

python tools/evaluate.py \
  --config configs/debug/single_fragment/04_k8_joint_finetune.py \
  --checkpoint "$STAGE04_RUN/checkpoints/best_joint_top1.pth" \
  --manifest "$MANIFEST" \
  --device cuda

python tools/check_stage_readiness.py \
  --stage joint_finetune \
  --run-dir "$STAGE04_RUN"
```

All model parameters are trainable. Cross-view consistency is metric-only;
`cross_view_consistency_weight=0.0`.

Export all ten views manually if required:

```bash
python tools/visualize_predictions.py \
  --config configs/debug/single_fragment/04_k8_joint_finetune.py \
  --checkpoint "$STAGE04_RUN/checkpoints/best_joint_top1.pth" \
  --manifest "$MANIFEST" \
  --device cuda \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/single_fragment_visualization_new
```

Expected layout includes `per_view/frame_000000` through `frame_000009`, one
`reference/fragment_0002`, and `cross_view/`.

## J. Stage 05: all four fragments in scene 000000

Start manually only after Stage 04 is ready:

```bash
python tools/train.py \
  --config configs/debug/single_fragment/05_scene000000_all_fragments_joint.py \
  --device cuda \
  --init-checkpoint "$STAGE04_RUN/checkpoints/best_joint_top1.pth"
```

This uses 40 observations and tests discrimination between four physical
geometries. It is not started automatically.

## K. Resume versus initialization

Use `--resume` only inside the same stage/config. It restores model, optimizer,
scheduler, AMP scaler, `batch_step`, `optimizer_step`, and `samples_seen`.

Use `--init-checkpoint` between stages. `--init-modules prefix ...` optionally
restricts strict model-only initialization to explicit module prefixes.

## L. Files to send after each stage

Always send:

- `resolved_config.json`
- `run_manifest.json`
- `history/history.jsonl`
- `history/epoch_metrics.csv`
- `checkpoints/best_metrics.json`
- `stage_summary.json`
- `gradient_summary.json`

Additionally:

- Stage 01: `oracle_pose_metrics.json`, `cross_view_consistency.json`
- Stage 02: `ranking_diagnostics.json`, `ranking_target_statistics.json`
- Stage 03: `region_class_distribution.json`, `region_confusion_matrix.json`, `region_metrics.json`
- Stage 04: `top1_vs_oracle_summary.json`, `cross_view_consistency.json`, `effective_group_metrics.json`

Create one compact archive without PLY or checkpoint weights:

```bash
ARCHIVE=/home/nikita/disser/fragment-template-registration-lab/work_dirs/single_fragment_stage_reports.tar.gz

find "$STAGE01_RUN" "$STAGE02_RUN" "$STAGE03_RUN" "$STAGE04_RUN" \
  -type f \
  \( \
    -name resolved_config.json -o \
    -name run_manifest.json -o \
    -name history.jsonl -o \
    -name epoch_metrics.csv -o \
    -name best_metrics.json -o \
    -name stage_summary.json -o \
    -name gradient_summary.json -o \
    -name oracle_pose_metrics.json -o \
    -name cross_view_consistency.json -o \
    -name ranking_diagnostics.json -o \
    -name ranking_target_statistics.json -o \
    -name region_class_distribution.json -o \
    -name region_confusion_matrix.json -o \
    -name region_metrics.json -o \
    -name top1_vs_oracle_summary.json -o \
    -name effective_group_metrics.json \
  \) -print0 | tar --null --create --gzip --file "$ARCHIVE" --files-from=-
```

The file selection has no `*.ply`, `*.pth`, or `*.pt` patterns.
