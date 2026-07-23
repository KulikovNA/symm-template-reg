# Correspondence → Weighted Procrustes runbook

The stopped direct K1 run is a failed legacy baseline.  Its established
diagnosis is `rotation_context_collapse_and_static_base_rotation`: translation
is learned, but pooled direct rotation does not respond to the four input
views.  Do not resume it, and do not advance to K8, ranking, or regions before
the K1 Procrustes gate passes.

All commands below create outputs only under `work_dirs`.  Start in a fresh
shell with:

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export DIRECT_RUN=$WD/conditioned_v2_01_k1_direct_equal_exposure_frames04_05_02_08_seed0_20260719_101204
export V4=$WD/manifests/view_ladder/frames04_05_02_08.json
export V10=$WD/manifests/view_ladder/all_10_views.json
export TAG=$(date +%Y%m%d_%H%M%S)
```

## Exact commands

1. Analyze the stopped direct run:

```bash
python tools/analyze_training_plateau.py \
  --run-dir "$DIRECT_RUN" \
  --output-dir "$WD/correspondence_pose_v1_plateau_$TAG"
```

2. Audit the direct rotation feature path:

```bash
python tools/audit_rotation_feature_path.py \
  --run-dir "$DIRECT_RUN" \
  --output-dir "$WD/correspondence_pose_v1_rotation_path_$TAG"
```

3. Run the real-sample oracle Procrustes preflight:

```bash
python tools/test_oracle_correspondence_pose.py \
  --config configs/debug/correspondence_pose_v1/00_oracle_procrustes.py \
  --manifest "$V4" --device cpu \
  --output-dir "$WD/correspondence_pose_v1_oracle_$TAG"
```

4. Audit target leakage and matching counterfactuals before training:

```bash
python tools/audit_target_leakage.py \
  --config configs/debug/correspondence_pose_v1/01_correspondence_only_4views.py \
  --manifest "$V4" --device cuda --max-samples 4 \
  --output-dir "$WD/correspondence_pose_v1_leakage_untrained_$TAG"

export UNTRAINED_LEAKAGE_AUDIT=$WD/correspondence_pose_v1_leakage_untrained_$TAG/target_leakage_audit.json
```

5. Train correspondence-only on four views, seed 0:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/01_correspondence_only_4views.py \
  --manifest "$V4" --seeds 0 --device cuda \
  --output-dir "$WD/correspondence_pose_v1_corr_v4_seed0_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$UNTRAINED_LEAKAGE_AUDIT"
```

5a. Read the actual checkpoint path from the completed seed-0 summary and run
the trained audit.  Do not use `/absolute/path/to/...` literally:

```bash
export CORR_SUMMARY=$WD/correspondence_pose_v1_corr_v4_seed0_$TAG/per_run_summary.csv
export CORR_CKPT=$(python -c "import csv,sys; print(next(csv.DictReader(open(sys.argv[1])))['best_checkpoint'])" "$CORR_SUMMARY")
test -f "$CORR_CKPT" && echo "Using checkpoint: $CORR_CKPT"

python tools/audit_target_leakage.py \
  --config configs/debug/correspondence_pose_v1/01_correspondence_only_4views.py \
  --manifest "$V4" --checkpoint "$CORR_CKPT" \
  --device cuda --max-samples 4 \
  --output-dir "$WD/correspondence_pose_v1_leakage_trained_$TAG"

export TRAINED_LEAKAGE_AUDIT=$WD/correspondence_pose_v1_leakage_trained_$TAG/target_leakage_audit.json
```

The trained audit must report no target leakage, permutation equivariance,
worse correspondence RMSE after zeroing either geometry stream, and a
non-collapsed correspondence context.

6. Train correspondence-only on four views, seeds 0–2:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/01_correspondence_only_4views.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --output-dir "$WD/correspondence_pose_v1_corr_v4_seeds012_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$UNTRAINED_LEAKAGE_AUDIT"
```

Only continue if p95 ≤ 2 mm, RMSE ≤ 1 mm, all views are finite, effective
correspondence count is at least 16, and the trained leakage audit passes.

7. Train Procrustes base on four views, seed 0, from the correspondence-only
   checkpoint using model-only initialization:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/02_procrustes_base_4views.py \
  --manifest "$V4" --seeds 0 --device cuda \
  --init-checkpoint "$CORR_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_procrustes_v4_seed0_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

8. Repeat Procrustes base for seeds 0–2:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/02_procrustes_base_4views.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$CORR_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_procrustes_v4_seeds012_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

All seeds must reach success 5°/5 mm ≥ 0.9, rotation response ≥ 0.5, static
fraction 0, world-axis spread ≤ 10°, and no target leakage.

9. Run the bounded hybrid only after step 8 passes:

```bash
export PROC_CKPT=/absolute/path/to/best_procrustes_base.pth
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/03_hybrid_bounded_residual_4views.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$PROC_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_hybrid_v4_seeds012_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

10. Optional cross-view ablation; it is not an orientation source:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/04_procrustes_cross_view_ablation.py \
  --manifest "$V4" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$PROC_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_cross_view_ablation_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

11. Expand the proven K1 Procrustes base to ten views:

```bash
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/05_procrustes_base_10views.py \
  --manifest "$V10" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$PROC_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_procrustes_v10_seeds012_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

12. Run the ten-view bounded hybrid only after step 11 passes:

```bash
export PROC10_CKPT=/absolute/path/to/best_procrustes_base_10views.pth
python tools/run_view_ladder.py \
  --config configs/debug/correspondence_pose_v1/06_hybrid_bounded_residual_10views.py \
  --manifest "$V10" --seeds 0 1 2 --device cuda \
  --init-checkpoint "$PROC10_CKPT" \
  --init-modules observed_encoder template_encoder interaction_transformer dual_stream_geometry_encoder correspondence_head point_weight_head \
  --output-dir "$WD/correspondence_pose_v1_hybrid_v10_seeds012_$TAG" \
  --cfg-options target_leakage_policy.audit_path="$TRAINED_LEAKAGE_AUDIT"
```

13. Make a compact archive without checkpoints, PLY galleries, or caches:

```bash
find "$WD" -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' -o -name '*.jsonl' -o -name '*.npz' \) \
  \( -path '*correspondence_pose_v1*' -o -path "$DIRECT_RUN/*" \) -print0 | \
  tar --null -czf "$WD/correspondence_pose_v1_compact_$TAG.tar.gz" --files-from=-
```

## Compact files to send for analysis

- `plateau_analysis.json`, `plateau_analysis.csv`, and `plateau_report.md`;
- `rotation_feature_path_summary.json`, pairwise matrices NPZ, and report;
- `oracle_procrustes_results.json` and report;
- both untrained and trained `target_leakage_audit.json` reports;
- every stage's `per_run_summary.csv`, `training_budget.json`,
  `final_summary.json`, and `checkpoints/best_metrics.json`;
- best-evaluation `per_sample_metrics.csv`,
  `context_conditioning_diagnostics.json`, and correspondence summary;
- any `diagnostic_failure.json` and `failure_report.md`.
