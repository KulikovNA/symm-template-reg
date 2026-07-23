# Pose view-ladder runbook

These are debug runs on the test split. Train and validation intentionally use
the same samples; results are not final evaluation metrics. Every invocation is
explicit and the runner never starts the next ladder level automatically.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

MANIFEST=/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json
LADDER=/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/view_ladder
STAMP=$(date +%Y%m%d_%H%M%S)
```

## 1. Direct pose-parameter gate

```bash
python tools/debug_optimize_pose_parameters.py \
  --manifest "$MANIFEST" \
  --frames 4 8 6 \
  --num-starts 16 \
  --steps 3000 \
  --device cpu \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/direct_pose_optimization_$STAMP"
```

Stop if any frame has fewer than 15 successful starts.

## 2–7. One-frame K1/K8, seeds 0/1/2

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/frame04_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame04_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/frame04_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame04_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/frame08_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame08_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/frame08_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame08_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/frame06_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame06_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/frame06_only.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame06_$STAMP"
```

Do not continue unless every one-frame `one_frame_summary.json` has
`criterion_passed=true`.

## 8. Two views: frames 4 + 8

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/frames04_08.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_2views_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/frames04_08.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_2views_$STAMP"
```

## 9. Four views: frames 4 + 5 + 2 + 8

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_4views_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/frames04_05_02_08.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_4views_$STAMP"
```

## 10. All ten views

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k1_pose_only.py \
  --manifest "$LADDER/all_10_views.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_10views_$STAMP"

python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only.py \
  --manifest "$LADDER/all_10_views.json" --seeds 0 1 2 \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_10views_$STAMP"
```

## 11. Auxiliary decoder ablation

Run only after the K8 final-layer one-frame baseline completes:

```bash
python tools/run_view_ladder.py --device cuda \
  --config configs/debug/view_ladder/k8_pose_only_aux_decoder.py \
  --manifest "$LADDER/frame04_only.json" --seeds 0 1 2 \
  --baseline-summary "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame04_$STAMP/per_run_summary.csv" \
  --output-dir "/home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_aux_frame04_$STAMP"
```

## 12. Compact archive

Set `REPORT_DIRS` to the explicit summary directories created above. The loop
also reads the corresponding training-run directories from each
`per_run_summary.csv`, so the archive includes per-eval query matrices without
including meshes or checkpoints:

```bash
DIRECT_DIR=/home/nikita/disser/fragment-template-registration-lab/work_dirs/direct_pose_optimization_<STAMP>
AUDIT_DIR=/home/nikita/disser/fragment-template-registration-lab/work_dirs/k1_k8_view_audit_<STAMP>

REPORT_DIRS=(
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame04_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame04_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame08_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame08_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_frame06_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_frame06_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_2views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_2views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_4views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_4views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k1_10views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_10views_<STAMP>
  /home/nikita/disser/fragment-template-registration-lab/work_dirs/view_ladder_k8_aux_frame04_<STAMP>
)

ARCHIVE=/home/nikita/disser/fragment-template-registration-lab/work_dirs/pose_view_ladder_reports.tar.gz

RUN_DIRS=()
for report_dir in "${REPORT_DIRS[@]}"; do
  while IFS= read -r run_dir; do
    RUN_DIRS+=("$run_dir")
  done < <(tail -n +2 "$report_dir/per_run_summary.csv" | cut -d, -f9 | tr -d '\r')
done

SOURCE_DIRS=("$DIRECT_DIR" "$AUDIT_DIR" "$LADDER" "${REPORT_DIRS[@]}" "${RUN_DIRS[@]}")

find "${SOURCE_DIRS[@]}" -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.jsonl' -o -name '*.md' \) \
  -print0 | sort -zu | tar --null --create --gzip --file "$ARCHIVE" --files-from=-
```

The archive command does not select PLY, PTH or PT files.
