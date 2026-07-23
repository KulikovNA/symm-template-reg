# Correspondence Head V4 runbook

Все команды выполняются из корня `symm-template-reg`. Каждый STOP обязателен:
следующую стадию нельзя запускать автоматически. Все новые результаты пишутся
только в `work_dirs`; исходный run и его checkpoint остаются read-only.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export STAGE_A4=$WD/v4_patch_classifier_frame04_20260720_100142
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Recheck существующего Stage A frame 4

```bash
python tools/recheck_patch_classifier_stage.py \
  --run-dir "$STAGE_A4" \
  --checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --device cuda \
  --output-dir "$WD/v4_patch_classifier_frame04_recheck_$TAG"

export RECHECK_A4=$WD/v4_patch_classifier_frame04_recheck_$TAG
cat "$RECHECK_A4/candidate_stage_gate.json"
cat "$RECHECK_A4/top1_quality_gate.json"
```

Recheck не обучает модель. Он считает точный ближайший template triangle,
single-owner patch, полный набор patches, содержащих triangle, и проверяет, что
размер/mtime ключевых файлов исходного run не изменились.

Candidate gate требует одновременно:

- `valid_patch_set_top4_recall >= 0.995`;
- `valid_patch_set_in_candidate_set_fraction >= 0.995`;
- больше одного предсказываемого patch и popular fraction `< 0.8`;
- отсутствие nonfinite и target leakage;
- успешный capacity audit.

`top1_quality_gate.json` — только диагностика. Его провал не блокирует
teacher-forced Stage B, если candidate gate пройден.

**STOP при `candidate_stage_passed=false`.** Не запускать frame 8 или Stage B.
Старый checkpoint обучался с legacy single-target semantics; после исправления
loss нужно повторить Stage A frame 4:

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4/00_patch_classifier_frame04.py \
  --device cuda \
  --work-dir "$WD"
```

Затем назначить новый `STAGE_A4`, снова выполнить recheck и остановиться для
анализа gate.

## 2. Optional refinement frame 4

Разрешён только когда `candidate_stage_passed=true`, но
`top1_quality_passed=false`, и coarse ranking действительно требуется улучшить.
Это не обязательное условие Stage B. Инициализация только model-only:

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4/00_patch_classifier_frame04_refine.py \
  --device cuda \
  --work-dir "$WD" \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth"
```

Не использовать `--resume`: refinement имеет новый optimizer, `lr=1e-4`,
constant scheduler и максимум 500 epochs.

## 3. Stage A frame 8

Только после успешного frame 4 candidate gate:

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4/00_patch_classifier_frame08.py \
  --device cuda \
  --work-dir "$WD"
```

Назначить напечатанный путь:

```bash
export STAGE_A8=$WD/<НАПЕЧАТАННЫЙ_RUN_FRAME08>
export TAG8=$(date +%Y%m%d_%H%M%S)

python tools/recheck_patch_classifier_stage.py \
  --run-dir "$STAGE_A8" \
  --checkpoint "$STAGE_A8/checkpoints/best.pth" \
  --device cuda \
  --output-dir "$WD/v4_patch_classifier_frame08_recheck_$TAG8"

python tools/package_correspondence_head_stage.py \
  --input "$STAGE_A8" \
  --input "$WD/v4_patch_classifier_frame08_recheck_$TAG8" \
  --output "$WD/v4_patch_classifier_frame08_report_$TAG8.tar.gz"
```

**STOP. Передать архив на внешний анализ.** Stage B не запускать в той же
последовательности команд.

## 4. Teacher-forced local Stage B frame 4

Только после внешнего подтверждения обоих Stage A candidate gates (frame 4 и
frame 8). Coarse classifier заморожен; GT valid patch и точный GT triangle
принудительно включаются в local candidates.

```bash
python tools/train.py \
  --config configs/debug/correspondence_head_v4/01_local_barycentric_gt_patch_frame04.py \
  --device cuda \
  --work-dir "$WD" \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth"
```

Stage B gate:

- correspondence p95 `<= 0.5 mm`;
- barycentric reconstruction p95 `<= 0.5 mm`;
- exact GT triangle candidate recall `= 1.0`;
- correspondence rank `= 3` и Procrustes rank valid;
- rotation `<= 0.5°`, translation `<= 0.5 mm`.

Назначить `STAGE_B4` и упаковать:

```bash
export STAGE_B4=$WD/<НАПЕЧАТАННЫЙ_RUN_STAGE_B4>
export TAGB=$(date +%Y%m%d_%H%M%S)

python tools/package_correspondence_head_stage.py \
  --input "$STAGE_B4" \
  --output "$WD/v4_local_barycentric_frame04_report_$TAGB.tar.gz"
```

**STOP.** Передать отчёт на внешний анализ. Frame 8 Stage B до этого не
запускать.

## Файлы для передачи после frame 8

- `candidate_stage_gate.json`;
- `top1_quality_gate.json`;
- `stage_gate.json`;
- `patch_target_ambiguity_summary.json`;
- `patch_target_ambiguity_report.md`;
- `patch_target_ambiguity_per_point.csv`;
- `recheck_summary.json` и `final_summary.json`;
- `best_evaluation/evaluation_summary.json`;
- `best_evaluation/per_sample_metrics.csv`;
- `checkpoints/best_metrics.json`;
- `history/history.jsonl`;
- compact archive, созданный `package_correspondence_head_stage.py`.

Checkpoint/PTH и PLY в compact archive намеренно не включаются.

## Короткая проверка среды

```bash
python -m compileall -q symm_template_reg tools tests
python -m unittest discover -s tests -v
python tools/smoke_correspondence_heads.py --device cuda
```

CUDA smoke выполняется в пользовательском окружении с доступной NVIDIA GPU.
