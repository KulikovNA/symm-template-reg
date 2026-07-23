# Local correspondence Stage B runbook

Работать из корня репозитория. Каждый STOP обязателен. Старые runs и
checkpoints используются read-only; все новые результаты создаются в
`work_dirs`.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export STAGE_A4=$WD/v4_patch_classifier_frame04_20260720_113703
export STAGE_A8=$WD/v4_patch_classifier_frame08_20260720_115206
export FAILED_B4=$WD/v4_local_barycentric_gt_patch_frame04_20260720_125942
export M4=$WD/manifests/joint_correspondence_pose/fragment0002_frame04_only.json
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Аудит failed Stage B contract

```bash
python tools/audit_local_triangle_target_contract.py \
  --run-dir "$FAILED_B4" \
  --checkpoint "$FAILED_B4/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --output-dir "$WD/local_triangle_contract_$TAG"
```

Аудит старого run ожидаемо может вернуть exit code 2: это означает найденный
contract failure, а не падение инструмента. Проверить
`local_triangle_target_contract_summary.json`.

## 2. Аудит triangle ambiguity

```bash
python tools/audit_triangle_target_ambiguity.py \
  --run-dir "$FAILED_B4" \
  --checkpoint "$FAILED_B4/checkpoints/best.pth" \
  --manifest "$M4" \
  --triangle-target-tolerance-m 0.00015 \
  --device cuda \
  --output-dir "$WD/triangle_target_ambiguity_$TAG"
```

## 3. Frozen-feature capacity

Полный диагностический запуск, не являющийся обучением основной модели:

```bash
python tools/audit_fine_feature_capacity.py \
  --checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --steps 200 \
  --max-points 1024 \
  --output-dir "$WD/fine_feature_capacity_$TAG"
```

Короткий smoke использует `--steps 2 --max-points 64`, но его диагноз нельзя
считать capacity-результатом.

## 4. B1 frame 4 — только triangle classifier

Запускать только после просмотра трёх аудитов выше:

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4_local/01_triangle_classifier_gt_patch_frame04.py \
  --device cuda \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --work-dir "$WD"
```

Назначить напечатанный путь и упаковать:

```bash
export B1_4=$WD/<НАПЕЧАТАННЫЙ_B1_FRAME4_RUN>
python tools/package_correspondence_head_stage.py \
  --input "$B1_4" \
  --output "$WD/$(basename "$B1_4")_compact_report.tar.gz"
```

**STOP.** Передать отчёт на внешний анализ. B2 не запускать до подтверждения
`stage_passed=true`.

## 5. B2 frame 4 — exact GT triangle + barycentric

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4_local/02_barycentric_gt_triangle_frame04.py \
  --device cuda \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --work-dir "$WD"
```

Упаковать аналогично и **STOP**.

## 6. B3 frame 4 — predicted triangle + barycentric

Только после отдельно пройденных B1 и B2. Сначала объединить model-only веса:

```bash
export B2_4=$WD/<ПРОЙДЕННЫЙ_B2_FRAME4_RUN>
export MERGED_B3_INIT=$WD/b3_frame04_init_$TAG.pth

python tools/merge_local_stage_checkpoints.py \
  --stage-a "$STAGE_A4/checkpoints/best.pth" \
  --b1 "$B1_4/checkpoints/best.pth" \
  --b2 "$B2_4/checkpoints/best.pth" \
  --output "$MERGED_B3_INIT"

python tools/train.py \
  --config configs/debug/correspondence_head_v4_local/03_triangle_plus_barycentric_frame04.py \
  --device cuda \
  --init-checkpoint "$MERGED_B3_INIT" \
  --work-dir "$WD"
```

Упаковать и **STOP**.

## 7. B4 frame 4 — full local correspondence

Только после B3 gate:

```bash
export B3_4=$WD/<ПРОЙДЕННЫЙ_B3_FRAME4_RUN>

python tools/train.py \
  --config configs/debug/correspondence_head_v4_local/04_full_local_correspondence_frame04.py \
  --device cuda \
  --init-checkpoint "$B3_4/checkpoints/best.pth" \
  --work-dir "$WD"
```

Упаковать и **STOP**.

## 8. Frame 8 и дальнейшие стадии

Только после внешнего подтверждения B4 frame 4 повторить B1→B2→B3→B4 с
конфигами `*_frame08.py`, каждый раз с package + STOP. Scheduled teacher
forcing и two-view разрешены только после прохождения всех четырёх substages
на обоих frames.

## 9. Короткие проверки

```bash
python -m compileall -q symm_template_reg tools tests
python -m unittest discover -s tests -v

python tools/smoke_local_correspondence_substages.py \
  --device cuda \
  --output "$WD/local_b1_b2_cuda_smoke_$TAG.json"
```

## Файлы для передачи после каждого substage

- `stage_gate.json`, `final_summary.json`, `diagnostic_failure.json` при провале;
- `best_evaluation/`, `history/history.jsonl`, `checkpoints/best_metrics.json`;
- B1: `triangle_classifier_metrics.json`, `random_baseline.json` и оба target-аудита;
- B2: `barycentric_metrics.json`, `canonical_coordinate_metrics.json`;
- B3/B4: correspondence, pose и Procrustes compact reports;
- compact `tar.gz`, созданный `package_correspondence_head_stage.py`.

PLY/PTH/PT/NPY/NPZ в compact archive не включаются.
