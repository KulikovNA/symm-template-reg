# Conditioned pose runbook

The new path predicts one sample-conditioned base pose and only camera-frame
residuals from K mode embeddings. Ranking and region stages remain disabled.
Every command is explicit; no command advances to the next stage automatically.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

WORK=/home/nikita/disser/fragment-template-registration-lab/work_dirs
MANIFEST="$WORK/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json"
LADDER="$WORK/manifests/view_ladder"
STAMP=$(date +%Y%m%d_%H%M%S)
```

## 1. Compile and tests

```bash
python -m compileall -q symm_template_reg tools configs tests
python -m unittest discover -s tests -q
```

## 2. Legacy conditioning intervention audit

```bash
python -u tools/audit_pose_input_conditioning.py \
  --run "$WORK/single_fragment_01_k8_pose_only_20260717_125815" \
  --manifest "$MANIFEST" --frames 4 8 6 5 --device cuda \
  --output-dir "$WORK/legacy_pose_input_conditioning_audit_$STAMP"
```

## 3-5. Conditioned K1 on individual views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/frame04_only.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_frame04_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/frame08_only.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_frame08_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/frame06_only.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_frame06_$STAMP"
```

Stop unless every `one_frame_summary.json` has `criterion_passed=true`.

## 6. Conditioned K1, two views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/frames04_08.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_2views_$STAMP"
```

## 7. Conditioned K1, four views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_4views_$STAMP"
```

## 8. Four-view context-conditioning gates

Run a counterfactual audit for each seed selected in the four-view summary:

```bash
while IFS=, read -r seed run_dir; do
  python -u tools/audit_pose_input_conditioning.py \
    --run "$run_dir" --manifest "$LADDER/frames04_05_02_08.json" \
    --frames 4 5 2 8 --device cuda \
    --output-dir "$WORK/conditioned_k1_4views_conditioning_seed${seed}_$STAMP"
done < <(
  tail -n +2 "$WORK/conditioned_k1_4views_$STAMP/per_run_summary.csv" \
    | cut -d, -f8,9 | tr -d '\r'
)

python tools/check_conditioning_gates.py \
  --run-summary "$WORK/conditioned_k1_4views_$STAMP/per_run_summary.csv" \
  --conditioning-summaries \
    "$WORK/conditioned_k1_4views_conditioning_seed0_$STAMP/conditioning_summary.json" \
    "$WORK/conditioned_k1_4views_conditioning_seed1_$STAMP/conditioning_summary.json" \
    "$WORK/conditioned_k1_4views_conditioning_seed2_$STAMP/conditioning_summary.json" \
  --output-dir "$WORK/conditioned_k1_4views_gate_$STAMP"
```

Do not start K8 unless `conditioning_gate_summary.json` contains
`all_seeds_passed=true`.

## 9. Conditioned K8 residual hypotheses, four views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/02_base_k8_residual_pose_only.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k8_4views_$STAMP"
```

## 10. Conditioned K1 and K8, ten views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/01_base_k1_pose_only.py \
  --manifest "$LADDER/all_10_views.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k1_10views_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/02_base_k8_residual_pose_only.py \
  --manifest "$LADDER/all_10_views.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k8_10views_$STAMP"
```

## 11. Correspondence ablation

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/04_base_k8_correspondence.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k8_correspondence_4views_$STAMP"
```

## 12. Weighted-SVD consistency ablation

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/conditioned_pose/05_base_k8_correspondence_svd_consistency.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "$WORK/conditioned_k8_svd_4views_$STAMP"
```

## 13. Compact archive

Remove from `REPORT_DIRS` any stage which was intentionally not run.

```bash
REPORT_DIRS=(
  "$WORK/conditioned_k1_frame04_$STAMP"
  "$WORK/conditioned_k1_frame08_$STAMP"
  "$WORK/conditioned_k1_frame06_$STAMP"
  "$WORK/conditioned_k1_2views_$STAMP"
  "$WORK/conditioned_k1_4views_$STAMP"
  "$WORK/conditioned_k1_4views_gate_$STAMP"
  "$WORK/conditioned_k8_4views_$STAMP"
  "$WORK/conditioned_k1_10views_$STAMP"
  "$WORK/conditioned_k8_10views_$STAMP"
  "$WORK/conditioned_k8_correspondence_4views_$STAMP"
  "$WORK/conditioned_k8_svd_4views_$STAMP"
)

RUN_DIRS=()
for report_dir in "${REPORT_DIRS[@]}"; do
  if test -f "$report_dir/per_run_summary.csv"; then
    while IFS= read -r run_dir; do RUN_DIRS+=("$run_dir"); done \
      < <(tail -n +2 "$report_dir/per_run_summary.csv" | cut -d, -f9 | tr -d '\r')
  fi
done

ARCHIVE="$WORK/conditioned_pose_reports_$STAMP.tar.gz"
find "${REPORT_DIRS[@]}" "${RUN_DIRS[@]}" \
  "$WORK/legacy_pose_input_conditioning_audit_$STAMP" \
  -type f \( -name '*.json' -o -name '*.csv' -o -name '*.jsonl' -o -name '*.md' \) \
  -print0 | sort -zu | tar --null -czf "$ARCHIVE" --files-from=-
```

## Decision gates

- K1 one-frame failure: `pose_head_implementation_problem`.
- K1 one-frame success but 2/4-view failure with healthy context response:
  `feature_representation_or_viewpoint_problem`; only then enable correspondence.
- K1 four-view success: base conditioning is fixed.
- K1 success but K8 residual failure: `residual_multi_hypothesis_problem`.
- K8 oracle success but later top-1 failure: only then return to ranking.

## Files to send for analysis

Send the compact archive, or at minimum:

- every `per_run_summary.csv`, `view_scaling_curve.csv` and
  `one_frame_summary.json`;
- `conditioning_gate_summary.json`, all `conditioning_summary.json` and
  `conditioning_interventions.csv`;
- each run's `checkpoints/best_metrics.json`, `resolved_config.json`,
  `history/history.jsonl`, `query_assignment_diagnostics.json` and the best
  evaluation's `query_assignment_matrix.csv`/`per_sample_metrics.csv`.

PLY visualizations and checkpoint weights are not needed for the first metric
review unless a geometry or strict-load problem is being investigated.
