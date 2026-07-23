# Conditioned pose v2 runbook

This workflow separates insufficient sample exposure from architectural
collapse. It does not enable ranking, region supervision, a larger K, PTv3,
DFAT, PointDSC, or automatic curriculum execution.

## What changed

- `train_budget.mode=sample_exposures` plans complete epochs and records exact
  exposure counts for every sample in `training_budget.json`.
- Early-stopping patience does not advance before every sample reaches 750
  exposures.
- Rotation context is geometry-only. Centroid and scale enter only the
  translation context.
- A grouped batch can contain multiple views of the same physical fragment.
- Cross-view world consistency and pairwise pose response are optional losses.
- Base pose can come from direct context, Weighted Procrustes, or bounded direct
  correction on top of Procrustes.
- K8 residuals are limited to 15 degrees and 10 mm and are not allowed to act as
  arbitrary absolute-pose queries.

## Common paths

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export VL=$WD/manifests/view_ladder
export V4=$VL/frames04_05_02_08.json
export V10=$VL/all_10_views.json
export V1=$VL/frame04_only.json
export V2=$VL/frames04_08.json
export TAG=$(date +%Y%m%d_%H%M%S)
```

## Exact commands

1. Audit budgets before training:

```bash
python tools/audit_training_budget.py \
  --config configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py \
  --manifest-dir "$VL" \
  --output-dir "$WD/conditioned_v2_budget_audit_$TAG"
```

2. Audit target leakage on the actual forward contract:

```bash
python tools/audit_target_leakage.py \
  --config configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py \
  --manifest "$V4" \
  --device cuda \
  --max-samples 4 \
  --output-dir "$WD/conditioned_v2_target_leakage_audit_$TAG"
```

Without `--checkpoint`, this enforces target invariance, sample permutation
equivariance, and geometry sensitivity. Repeat it with
`--checkpoint /path/to/best.pth` after training to additionally require that
zeroing either geometry stream worsens GT correspondence RMSE.

3. Direct K1, four views, seed 0:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py \
  --manifest "$V4" --seeds 0 --device cuda \
  --output-dir "$WD/conditioned_v2_direct_v4_seed0_$TAG"
```

4. Direct K1, four views, seeds 0-2:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_direct_v4_seeds012_$TAG"
```

5. Cross-view consistency K1, four views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/02_k1_cross_view_consistency.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_cross_view_v4_seeds012_$TAG"
```

6. Procrustes base K1, four views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/03_k1_procrustes_base.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_procrustes_v4_seeds012_$TAG"
```

7. Hybrid Procrustes base K1, four views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/04_k1_hybrid_procrustes_base.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_hybrid_v4_seeds012_$TAG"
```

8. Select the best base path only after all three-seed runs finish:

```bash
python tools/select_best_base_path.py \
  --candidate direct="$WD/conditioned_v2_direct_v4_seeds012_$TAG/per_run_summary.csv" \
  --candidate cross_view="$WD/conditioned_v2_cross_view_v4_seeds012_$TAG/per_run_summary.csv" \
  --candidate procrustes="$WD/conditioned_v2_procrustes_v4_seeds012_$TAG/per_run_summary.csv" \
  --candidate hybrid="$WD/conditioned_v2_hybrid_v4_seeds012_$TAG/per_run_summary.csv" \
  --output "$WD/conditioned_v2_base_selection_$TAG.json"
```

Set `BEST_K1_CONFIG` and `BEST_K1_CHECKPOINT` from that report. The selection
tool returns exit code 2 if no candidate passes every readiness gate; in that
case do not advance to K8.

```bash
export BEST_K1_CONFIG=configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py
export BEST_K1_CHECKPOINT=/absolute/path/to/the/selected/best_k1_checkpoint.pth
```

9. Selected K1 base, ten views, equal exposure:

```bash
python tools/run_view_ladder.py \
  --config "$BEST_K1_CONFIG" --manifest "$V10" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_best_k1_v10_seeds012_$TAG"
```

Before K8, define overrides matching the selected base source:

```bash
# direct or cross-view:
K8_CFG_OPTIONS=()
# Procrustes instead:
# K8_CFG_OPTIONS=(model.base_pose_source=weighted_procrustes "model.weighted_procrustes={'type':'WeightedProcrustes'}")
# hybrid instead:
# K8_CFG_OPTIONS=(model.base_pose_source=procrustes_plus_direct_residual "model.weighted_procrustes={'type':'WeightedProcrustes'}" model.base_pose_head.output_mode=bounded_correction)
```

10. Bounded K8 residual, four views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/05_k8_bounded_residual.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$BEST_K1_CHECKPOINT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder sample_context_aggregator base_pose_head correspondence_head point_weight_head \
  --output-dir "$WD/conditioned_v2_k8_bounded_v4_seeds012_$TAG" \
  --cfg-options "${K8_CFG_OPTIONS[@]}"
```

11. Bounded K8 residual, ten views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/conditioned_pose_v2/05_k8_bounded_residual.py \
  --manifest "$V10" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$BEST_K1_CHECKPOINT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder sample_context_aggregator base_pose_head correspondence_head point_weight_head \
  --output-dir "$WD/conditioned_v2_k8_bounded_v10_seeds012_$TAG" \
  --cfg-options "${K8_CFG_OPTIONS[@]}"
```

12. Optional manual curriculum ablation:

```bash
python tools/run_view_curriculum.py \
  --config configs/debug/conditioned_pose_v2/07_k1_curriculum.py \
  --manifests "$V1" "$V2" "$V4" "$V10" \
  --seeds 0 1 2 --device cuda \
  --output-dir "$WD/conditioned_v2_curriculum_seeds012_$TAG"
```

13. Compact archive without checkpoints, PLY galleries, or caches:

```bash
find "$WD" -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.jsonl' \) \
  -path '*conditioned_v2*' -print0 | \
  tar --null -czf "$WD/conditioned_pose_v2_compact_$TAG.tar.gz" --files-from=-
```

## Files to send for analysis

- `conditioned_v2_base_selection_*.json`;
- every `per_run_summary.csv` and `view_scaling_curve.csv`;
- each run's `training_budget.json`, `final_summary.json`, and
  `checkpoints/best_metrics.json`;
- the best evaluation's `per_sample_metrics.csv` and
  `context_conditioning_diagnostics.json`;
- `target_leakage_audit.json` from the structural audit and from the selected
  trained checkpoint;
- K8 `query_assignment_diagnostics.json` and readiness result;
- `curriculum_summary.json` only if the optional curriculum was run.

The first files to inspect are the selection report, then each
`training_budget.json`, and finally `context_conditioning_diagnostics.json`.
