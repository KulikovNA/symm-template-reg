# Coordinate-guided surface runbook

Этот runbook фиксирует безопасную последовательность для frame 4. Команды не
изменяют существующий F1 run: каждый recheck/audit получает новый каталог, а
refinement загружает только веса через `--init-checkpoint`.

## Исходные данные

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export F1="$WD/fine_adapter_coordinate_control_frame04_20260721_092049"
export M4="$WD/manifests/joint_correspondence_pose/fragment0002_frame04_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Recheck существующего F1

```bash
python tools/recheck_f1_coordinate_stage.py \
  --run-dir "$F1" \
  --output-dir "$WD/fine_adapter_coordinate_control_frame04_recheck_$TAG"
```

У recheck должны быть `run_status="ok"`, отдельный
`stage_readiness`, явные thresholds и реальные имена failed checks. Исходный
каталог F1 проверяется по хешам до и после операции.

## 2. Отдельный pose audit для q_aux

```bash
python tools/audit_aux_coordinate_pose.py \
  --checkpoint "$F1/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --output-dir "$WD/aux_coordinate_pose_frame04_$TAG"
```

Этот audit строит позу непосредственно из `q_aux` и наблюдаемых точек. Метрики
основной legacy triangle/barycentric ветки к этой позе не относятся.

## 3. Exact surface projection audit

```bash
python tools/audit_coordinate_guided_surface_projection.py \
  --checkpoint "$F1/checkpoints/best.pth" \
  --manifest "$M4" \
  --device cuda \
  --output-dir "$WD/coordinate_guided_projection_frame04_$TAG"

cat "$WD/coordinate_guided_projection_frame04_$TAG/coordinate_projection_gate.json"
```

Проверяются три разных режима: global mesh (только диагностика), GT-patch
(teacher-forced диагностика) и predicted candidates (единственный
inference-valid режим). Во всех режимах точка вычисляется точной проекцией на
выбранный треугольник, а barycentric coordinates — аналитически.

## 4. Решение по gate

Если `projection_gate_passed=true`, остановиться и упаковать отчёт. Raw
`q_aux p95=1.012 мм` сам по себе не блокирует surface path, если проекция
проходит gate.

Если gate не пройден, разрешён только F1 refinement:

```bash
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/01_fine_adapter_coordinate_control_frame04_refine.py \
  --device cuda \
  --init-checkpoint "$F1/checkpoints/best.pth" \
  --work-dir "$WD"
```

Не использовать `--resume`: это новый run с model-only initialization. После
обучения назначить новый `F1` на созданный каталог, задать новый `TAG`, повторить
шаги 2 и 3 и снова остановиться для анализа.

## 5. Learned triangle fallback — только после повторного failure

Только если predicted-candidate projection всё ещё не проходит gate после F1
refinement:

```bash
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/02_coordinate_guided_triangle_frame04.py \
  --device cuda \
  --init-checkpoint "$F1/checkpoints/best.pth" \
  --work-dir "$WD"
```

Это fallback candidate-conditioned triangle classifier. После выбора
треугольника он использует exact projection и analytic barycentric coordinates.
Learned barycentric primary path сохранён только как legacy ablation со статусом
`failed_frozen_feature_barycentric_capacity_on_frame04` и не является default в
новых fine-конфигах.

## 6. Упаковка компактного отчёта

Команда для текущих результатов:

```bash
python tools/package_coordinate_guided_surface_report.py \
  --input "$WD/fine_feature_capacity_v2_frame04_20260721_091941" \
  --input "$WD/barycentric_feature_capacity_frame04_20260721_091941" \
  --input "$F1" \
  --input "$WD/aux_coordinate_pose_frame04_$TAG" \
  --input "$WD/coordinate_guided_projection_frame04_$TAG" \
  --output "$WD/coordinate_guided_surface_frame04_$TAG.tar.gz"
```

Архив содержит только JSON, CSV, JSONL и MD; PLY, PTH, PT, NPY и NPZ не
включаются.

## 7. Frame 8 — не запускать до frame-4 pass

Подготовлены:

- `01_fine_adapter_coordinate_control_frame08.py`;
- `01_fine_adapter_coordinate_control_frame08_refine.py`.

Первый frame-8 run инициализируется через `--init-checkpoint` из:

```text
/home/nikita/disser/fragment-template-registration-lab/work_dirs/v4_patch_classifier_frame08_20260720_115206/checkpoints/best.pth
```

Последовательность та же: coordinate control → projection audit → optional
refinement → triangle fallback только при необходимости. F2, frame 8, full
backbone и последующие стадии автоматически не запускаются.

## STOP

После каждого `coordinate_projection_gate.json` — STOP. Никакая следующая
стадия не начинается без отдельного анализа этого файла.
