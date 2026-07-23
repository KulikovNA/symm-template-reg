# Fine correspondence feature runbook

Все команды выполняются из репозитория. Каждый output-каталог должен быть новым.
Старые Stage A/B runs и checkpoints не изменяются.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export STAGE_A4="$WD/v4_patch_classifier_frame04_20260720_113703"
export M4="$WD/manifests/joint_correspondence_pose/fragment0002_frame04_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Full capacity V2, frame 4

Это длинный convergence-аудит: четыре triangle probes, normalized coordinate probe,
три learning rates, до 1500 шагов, evaluation каждые 50 шагов.

```bash
python tools/audit_fine_feature_capacity_v2.py \
  --checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --output-dir "$WD/fine_feature_capacity_v2_frame04_$TAG"
```

Gate: `coordinate p95 <= 1 mm` или одновременно triangle top-1 `>= 0.95`
и top-4 `>= 0.995`. Если ни один probe не проходит, frozen features недостаточны.

## 2. Barycentric capacity, frame 4

Probe получает точный GT-треугольник и предсказывает только barycentric weights.

```bash
export TAG=$(date +%Y%m%d_%H%M%S)
python tools/audit_barycentric_feature_capacity.py \
  --checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --output-dir "$WD/barycentric_feature_capacity_frame04_$TAG"
```

Gate: canonical-coordinate p95 `<= 0.5 mm`. Этот аудит диагностический и сам
по себе не разрешает F2/F3.

## 3. F1 coordinate control, frame 4

Запускать после просмотра обоих полных отчётов. По умолчанию обучаются только
fine adapter и auxiliary coordinate head. Stage A загружается model-only;
новые ключи остаются корректно инициализированными.

```bash
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/01_fine_adapter_coordinate_control_frame04.py \
  --device cuda \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --work-dir "$WD"
```

Gate F1: p95 `<= 1 mm`, RMSE `<= 0.5 mm`, leakage отсутствует, feature variance
не схлопнута.

Если adapter-only F1 не проходит после полноценного обучения, один раз запустить
policy B с LR `3e-4` для новых модулей и `3e-5` для последнего interaction layer
и двух fine projections:

```bash
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/01_fine_adapter_coordinate_control_policy_b_frame04.py \
  --device cuda \
  --init-checkpoint "$STAGE_A4/checkpoints/best.pth" \
  --work-dir "$WD"
```

Если policy B тоже не проходит, диагноз:
`fine_feature_representation_failure`. F2 запускать нельзя.

## 4. Package F1 and STOP

Подставить фактический новый F1 run. В архив попадают только JSON, CSV, JSONL и
MD; PLY/PTH/PT/NPY/NPZ исключаются.

```bash
export F1_RUN="$WD/<fine_adapter_coordinate_control_frame04_TIMESTAMP>"
python tools/package_correspondence_head_stage.py \
  --input "$F1_RUN" \
  --output "$WD/$(basename "$F1_RUN")_report.tar.gz"
```

**STOP.** Перед F2 передать для анализа:

- `stage_gate.json`;
- `final_summary.json`;
- `resolved_config.json`;
- `checkpoints/best_metrics.json`;
- `history/history.jsonl`;
- `coordinate_metrics.json`;
- `fine_feature_metrics.json`;
- `gradient_summary.json`.

## 5. F2 only after F1 PASS

F2 должен инициализироваться из успешного F1 checkpoint, не из Stage A.

```bash
export F1_BEST="$F1_RUN/checkpoints/best.pth"
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/02_fine_adapter_triangle_frame04.py \
  --device cuda \
  --init-checkpoint "$F1_BEST" \
  --work-dir "$WD"
```

Gate F2: valid-set top-1 `>= 0.95`, top-4 `>= 0.995`, candidate recall `= 1`,
target-index mismatch `= 0`, no collapse. Затем package и **STOP**.

## 6. F3 only after F2 PASS

```bash
export F2_RUN="$WD/<fine_adapter_triangle_frame04_TIMESTAMP>"
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/03_fine_adapter_triangle_plus_barycentric_frame04.py \
  --device cuda \
  --init-checkpoint "$F2_RUN/checkpoints/best.pth" \
  --work-dir "$WD"
```

Gate F3: correspondence p95 `<= 0.5 mm`, barycentric p95 `<= 0.5 mm`, rank `= 3`,
Procrustes valid. Затем package и **STOP**.

Frame 8 конфиги подготовлены, но их нельзя запускать до PASS F3 на frame 4.
Также этот runbook не разрешает B2/B3/B4 и two-view эксперименты.

## Tensor contract

- `coarse_patch_features`: `[B, 64, 256]`, sampled template encoder features до
  fine adapter; используются только coarse patch classifier/candidate builder.
- `fine_point_features`: `[B, N_dense, 256]`, fusion исходного dense observed
  encoder feature до FPS, интерполированного cross-conditioned interaction и
  трёх масштабов локальной C-frame geometry.
- `fine_triangle_features`: `[B, F, 256]`, triangle-local O-frame geometry плюс
  coarse owner-patch feature и fine template interaction feature.

Fine head отклоняет конфигурацию без dense adapter и candidate-conditioned head.
Observed local descriptors строятся только внутри C, triangle descriptors —
только внутри O. Вычитание raw observed coordinates C из triangle coordinates O
в реализации отсутствует.

## Smoke-команда для разработчика

Для ровно одной эпохи недостаточно переопределить `train.max_epochs`: нужно
также ограничить отдельный budget.

```bash
--cfg-options train.max_epochs=1 train_budget.epochs=1
```

Такой smoke проверяет только forward/backward/optimizer/reporting и не является
результатом сходимости.
