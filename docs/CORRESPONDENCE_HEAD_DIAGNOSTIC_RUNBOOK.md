# Correspondence-head diagnostic runbook

Рабочий каталог репозитория:

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
```

Все команды ниже пишут только в `/home/nikita/disser/fragment-template-registration-lab/work_dirs`.
Старый run `single_pose_uniform_correspondence_views02_fragment0002_views02_seed0_20260719_152751`
используется только для чтения. Четыре view автоматически не запускаются.

Аудиты 2–4 уже были выполнены при подготовке реализации. Готовые результаты можно
просто проверить:

```bash
cat /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/point_contract/registration_point_contract_summary.json
cat /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/template_resolution/template_resolution_summary.json
cat /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/current_head_checkpoint_final/soft_correspondence_head_summary.json
```

Если требуется именно повторный запуск, один раз в текущем shell создайте новый root.
Audit-инструменты намеренно не перезаписывают существующие каталоги:

```bash
RERUN_ROOT=$(mktemp -d /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_rerun_XXXXXX)
echo "$RERUN_ROOT"
```

## 1. Compile и tests

```bash
python -m compileall -q symm_template_reg tools configs tests
python -m unittest discover -s tests -v
```

## 2. Point-contract audit

```bash
python tools/audit_registration_point_contract.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame04.py \
  --manifest /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json \
  --output-dir "$RERUN_ROOT/point_contract"
```

Не обучать, если `registration_point_contract_summary.json:audit_passed` не равен `true`.

## 3. Template-resolution audit

```bash
python tools/audit_template_correspondence_resolution.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame04.py \
  --manifest /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json \
  --output-dir "$RERUN_ROOT/template_resolution"
```

## 4. Current-head audit на старом best checkpoint

```bash
python tools/audit_soft_correspondence_head.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame04.py \
  --manifest /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json \
  --run-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/single_pose_uniform_correspondence_views02_fragment0002_views02_seed0_20260719_152751 \
  --device cuda \
  --output-dir "$RERUN_ROOT/current_head_checkpoint"
```

## 5. Full direct-capacity audit, frame 4

```bash
python tools/debug_optimize_correspondence_parameterization.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame04.py \
  --manifest /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json \
  --frame 4 --device cuda --random-starts 8 --steps 300 --max-points 512 \
  --methods free_q soft_global_anchors topk_local_soft triangle_barycentric \
  --temperatures 1.0 0.5 0.2 0.1 0.05 \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/capacity_full_frame04
```

## 6. Full direct-capacity audit, frame 8

```bash
python tools/debug_optimize_correspondence_parameterization.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame08.py \
  --manifest /home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame08_only.json \
  --frame 8 --device cuda --random-starts 8 --steps 300 --max-points 512 \
  --methods free_q soft_global_anchors topk_local_soft triangle_barycentric \
  --temperatures 1.0 0.5 0.2 0.1 0.05 \
  --output-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/capacity_full_frame08
```

Если free-q проходит, а `soft_global_anchors` не проходит, диагноз —
`soft_global_surface_matching_parameterization_failure`: legacy runs ниже пропускаются,
а работа продолжается с SurfaceHeadV2.

## 7. Legacy current head, one view frame 4

```bash
python tools/train.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame04.py \
  --device cuda \
  --work-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

```bash
RUN_LEGACY04=$(ls -dt /home/nikita/disser/fragment-template-registration-lab/work_dirs/legacy_soft_shell_frame04_* | head -1)
cat "$RUN_LEGACY04/stage_gate.json"
```

**STOP — не переходить дальше без анализа `stage_gate.json`.**

## 8. Legacy current head, one view frame 8

Запускать только если это не запрещено capacity/frame-4 gate.

```bash
python tools/train.py \
  --config configs/debug/joint_correspondence_pose_v3/00_legacy_soft_frame08.py \
  --device cuda \
  --work-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

```bash
RUN_LEGACY08=$(ls -dt /home/nikita/disser/fragment-template-registration-lab/work_dirs/legacy_soft_shell_frame08_* | head -1)
cat "$RUN_LEGACY08/stage_gate.json"
```

**STOP — не переходить дальше без анализа `stage_gate.json`.**

## 9. SurfaceHeadV2, frame 4

```bash
python tools/smoke_train_step.py \
  --config configs/debug/joint_correspondence_pose_v3/01_surface_v2_frame04.py \
  --device cuda --num-samples 1
python tools/train.py \
  --config configs/debug/joint_correspondence_pose_v3/01_surface_v2_frame04.py \
  --device cuda \
  --work-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

```bash
RUN_SURFACE04=$(ls -dt /home/nikita/disser/fragment-template-registration-lab/work_dirs/surface_v2_shell_frame04_* | head -1)
cat "$RUN_SURFACE04/stage_gate.json"
```

**STOP — не переходить дальше без анализа `stage_gate.json`.**

## 10. SurfaceHeadV2, frame 8

```bash
python tools/train.py \
  --config configs/debug/joint_correspondence_pose_v3/02_surface_v2_frame08.py \
  --device cuda \
  --work-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

```bash
RUN_SURFACE08=$(ls -dt /home/nikita/disser/fragment-template-registration-lab/work_dirs/surface_v2_shell_frame08_* | head -1)
cat "$RUN_SURFACE08/stage_gate.json"
```

**STOP — не переходить дальше без анализа `stage_gate.json`.**

## 11. SurfaceHeadV2, two views

Запускать только если оба one-view Surface gates прошли.

```bash
python tools/train.py \
  --config configs/debug/joint_correspondence_pose_v3/03_surface_v2_views02.py \
  --device cuda \
  --work-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs
```

```bash
RUN_SURFACE02=$(ls -dt /home/nikita/disser/fragment-template-registration-lab/work_dirs/surface_v2_shell_views02_* | head -1)
cat "$RUN_SURFACE02/stage_gate.json"
```

**STOP — внешний анализ; четыре view не запускать.**

## 12. Compact reports

Архиватор включает только JSON, CSV, JSONL и MD; PLY/PTH/PT/NPY/NPZ исключены.

```bash
python tools/package_joint_stage_report.py \
  --run-dir /home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719 \
  --output /home/nikita/disser/fragment-template-registration-lab/work_dirs/parameterization_capacity_report.tar.gz
python tools/package_joint_stage_report.py \
  --run-dir "$RUN_SURFACE04" \
  --output /home/nikita/disser/fragment-template-registration-lab/work_dirs/joint_surface_frame04_report.tar.gz
python tools/package_joint_stage_report.py \
  --run-dir "$RUN_SURFACE08" \
  --output /home/nikita/disser/fragment-template-registration-lab/work_dirs/joint_surface_frame08_report.tar.gz
python tools/package_joint_stage_report.py \
  --run-dir "$RUN_SURFACE02" \
  --output /home/nikita/disser/fragment-template-registration-lab/work_dirs/joint_surface_views02_report.tar.gz
```

Если файл архива уже существует, выбрать новое имя: архиватор намеренно не перезаписывает результаты.
