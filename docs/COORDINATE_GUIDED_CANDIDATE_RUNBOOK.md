# Coordinate-guided candidate runbook

Старые runs/checkpoints не изменяются. Каждый audit получает новый output под
`work_dirs`.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export F1_REF="$WD/fine_adapter_coordinate_control_frame04_refine_20260721_105906"
export M4="$WD/manifests/joint_correspondence_pose/fragment0002_frame04_only.json"
export M8="$WD/manifests/joint_correspondence_pose/fragment0002_frame08_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Candidate pipeline

```bash
python tools/audit_triangle_candidate_pipeline.py \
  --checkpoint "$F1_REF/checkpoints/best.pth" --manifest "$M4" --device cuda \
  --output-dir "$WD/triangle_candidate_pipeline_frame04_$TAG"
```

## 2. Exact-global benchmark

```bash
python tools/benchmark_mesh_projection.py \
  --checkpoint "$F1_REF/checkpoints/best.pth" \
  --manifest "$M4" --manifest "$M8" --device cuda \
  --output-dir "$WD/mesh_projection_benchmark_frame04_frame08_$TAG"
```

## 3. Recheck exact-global, old-32 и q_aux shortlists

```bash
python tools/recheck_coordinate_guided_surface.py \
  --checkpoint "$F1_REF/checkpoints/best.pth" --manifest "$M4" --device cuda \
  --candidate-k 16 32 64 128 256 \
  --output-dir "$WD/coordinate_surface_recheck_frame04_$TAG"

cat "$WD/coordinate_surface_recheck_frame04_$TAG/coordinate_surface_stage_gate.json"
```

## 4. Package и STOP

```bash
python tools/package_coordinate_candidate_report.py \
  --input "$F1_REF" \
  --input "$WD/triangle_candidate_pipeline_frame04_$TAG" \
  --input "$WD/mesh_projection_benchmark_frame04_frame08_$TAG" \
  --input "$WD/coordinate_surface_recheck_frame04_$TAG" \
  --output "$WD/coordinate_candidate_frame04_$TAG.tar.gz"
```

После чтения stage gate — **STOP**. Если frame 4 проходит хотя бы exact-global,
correctness считается решённой; shortlist остаётся ускорением и не блокирует
frame 8.

## 5. Frame 8 — только вручную после frame-4 pass

```bash
python tools/train.py \
  --config configs/debug/fine_correspondence_v1/01_fine_adapter_coordinate_control_frame08.py \
  --device cuda \
  --init-checkpoint "$WD/v4_patch_classifier_frame08_20260720_115206/checkpoints/best.pth" \
  --work-dir "$WD"
```

Затем выполнить aux-coordinate audit, exact-global/shortlist recheck для frame 8,
упаковать отчёт и снова **STOP**. Two-view config
`configs/debug/coordinate_guided_surface_v2/views02.py` подготовлен, но не
запускается до успешного анализа обоих однокадровых gates. Learned triangle и
learned barycentric heads в этой последовательности не обучаются.

## TWO-VIEW COORDINATE TRAINING

```bash
export F4="$WD/fine_adapter_coordinate_control_frame04_refine_20260721_105906/checkpoints/best.pth"
export F8="$WD/fine_adapter_coordinate_control_frame08_20260721_115051/checkpoints/best.pth"
export TRANSFER="$WD/coordinate_checkpoint_transfer_$TAG"

python tools/audit_coordinate_checkpoint_transfer.py \
  --checkpoint-frame4 "$F4" --checkpoint-frame8 "$F8" \
  --manifest-frame4 "$M4" --manifest-frame8 "$M8" --device cuda \
  --output-dir "$TRANSFER"

cat "$TRANSFER/selected_two_view_initialization.json"
```

Взять единственный `selected_checkpoint` без усреднения весов:

```bash
export INIT_CHECKPOINT=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$TRANSFER/selected_two_view_initialization.json")

python tools/train.py \
  --config configs/debug/coordinate_guided_surface_v2/views02.py \
  --device cuda --init-checkpoint "$INIT_CHECKPOINT" --work-dir "$WD"
```

После завершения прочитать `stage_gate.json`, задать `TWO_VIEW_RUN` и упаковать:

```bash
python tools/package_two_view_coordinate_report.py \
  --input "$TRANSFER" --input "$TWO_VIEW_RUN" \
  --output "$WD/two_view_coordinate_report_$TAG.tar.gz"
```

Затем **STOP** и внешний анализ. `views02_selective_unfreeze.py` разрешён только
после явного решения по failed fine-only run. Four-view не запускать.

## FOUR-VIEW COORDINATE TRAINING

Этот раздел начинается только после зафиксированного successful two-view run.
Все команды создают новые каталоги; исходный run и его checkpoint не меняются.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export RUN02="$WD/coordinate_guided_surface_v2_views02_20260721_155216"
export CKPT02="$RUN02/checkpoints/best.pth"
export M02="$WD/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json"
export M04="$WD/manifests/coordinate_guided_surface/fragment0002_views04_shell_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
```

### 1. Recheck чистого active path

```bash
python tools/recheck_coordinate_guided_surface.py \
  --checkpoint "$CKPT02" --manifest "$M02" --device cuda \
  --output-dir "$WD/two_view_active_recheck_$TAG"
```

### 2. Построить и проверить nested shell-only manifest

```bash
python tools/build_coordinate_view_manifest.py \
  --source-manifest "$WD/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json" \
  --frames 4 5 2 8 --shell-only --output "$M04"
```

### 3. Before-training initialization audit

```bash
python tools/audit_four_view_initialization.py \
  --checkpoint "$CKPT02" --manifest "$M04" --device cuda \
  --output-dir "$WD/four_view_initialization_$TAG"
```

### 4. Cache equivalence и benchmark

```bash
python tools/audit_frozen_feature_cache.py \
  --config configs/debug/coordinate_guided_surface_v2/views04.py \
  --checkpoint "$CKPT02" --manifest "$M04" --device cuda \
  --output-dir "$WD/frozen_feature_cache_audit_$TAG"

cat "$WD/frozen_feature_cache_audit_$TAG/frozen_feature_cache_audit.json"
```

Cache используется только при `cache_allowed=true` в указанном audit JSON.
При любом failed check trainer автоматически переходит на online path; это не
меняет математический путь.

### 5. Four-view fine-only training

```bash
python tools/train.py \
  --config configs/debug/coordinate_guided_surface_v2/views04.py \
  --device cuda \
  --init-checkpoint "$CKPT02" \
  --cfg-options \
    data.train_manifest="$M04" \
    data.validation_manifest="same_as_train" \
  --work-dir "$WD"
```

После завершения прочитать `stage_gate.json`. Один failed frame блокирует этап.
Автоматически selective-unfreeze не запускать.

### 6. Compact report и STOP

```bash
export RUN04=/absolute/path/printed/by/train

python tools/package_four_view_coordinate_report.py \
  --input "$WD/two_view_active_recheck_$TAG" \
  --input "$M04" \
  --input "$WD/four_view_initialization_$TAG" \
  --input "$WD/frozen_feature_cache_audit_$TAG" \
  --input "$RUN04" \
  --output "$WD/four_view_coordinate_report_$TAG.tar.gz"
```

**STOP и внешний анализ.** `views04_selective_unfreeze.py` только подготовлен.
Его запуск разрешается отдельным решением после классификации failed fine-only
run. Восемь и десять views автоматически не запускать.

## EIGHT-VIEW COORDINATE TRAINING

Это controlled overfit/debug experiment: train и validation содержат те же
восемь samples тестового split. Результат не является финальной оценкой
обобщения. Исходный four-view run и его checkpoint только читаются.

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export RUN04="$WD/coordinate_guided_surface_v2_views04_20260722_105023"
export CKPT04="$RUN04/checkpoints/best.pth"
export SOURCE="$WD/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json"
export M04="$WD/manifests/coordinate_guided_surface/fragment0002_views04_shell_only.json"
export M08="$WD/manifests/coordinate_guided_surface/fragment0002_views08_shell_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
export RECHECK04="$WD/four_view_active_recheck_$TAG"
export INIT08="$WD/eight_view_initialization_$TAG"
export SMOKE08="$WD/eight_view_smoke_$TAG"
export CACHE08="$WD/eight_view_cache_audit_$TAG"
```

### 1. Four-view active-path recheck

```bash
python tools/recheck_coordinate_guided_surface.py \
  --checkpoint "$CKPT04" --manifest "$M04" --device cuda \
  --candidate-k 16 --output-dir "$RECHECK04"
```

### 2. Nested shell-only manifest

Создавать manifest следует только если `$M08` ещё не существует: builder
намеренно запрещает перезапись.

```bash
python tools/build_coordinate_view_manifest.py \
  --source-manifest "$SOURCE" \
  --frames 4 5 2 8 0 1 6 9 --shell-only --output "$M08"

cat "$WD/manifests/coordinate_guided_surface/eight_view_manifest_audit.json"
```

### 3. Initialization audit без обучения

```bash
python tools/audit_eight_view_initialization.py \
  --checkpoint "$CKPT04" --manifest "$M08" --device cuda \
  --output-dir "$INIT08"
```

### 4. Batch-8 CUDA smoke

```bash
python tools/smoke_eight_view_coordinate.py \
  --config configs/debug/coordinate_guided_surface_v2/views08.py \
  --checkpoint "$CKPT04" --manifest "$M08" --batch-size 8 \
  --output "$SMOKE08/cuda_batch8_one_step.json"
```

Если и только если batch 8 завершился CUDA OOM, явно повторить с
`--batch-size 4`; основной training config тогда должен получить
`data.train_batch_size=4 train.gradient_accumulation_steps=2`. Оба режима дают
`effective_views_per_optimizer_step=8`.

### 5. Frozen-feature cache equivalence и benchmark

```bash
python tools/audit_frozen_feature_cache.py \
  --config configs/debug/coordinate_guided_surface_v2/views08.py \
  --checkpoint "$CKPT04" --manifest "$M08" --device cuda \
  --output-dir "$CACHE08"

python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("cache_allowed=",r["cache_allowed"],"speedup=",r["speedup"])' \
  "$CACHE08/frozen_feature_cache_audit.json"
```

Trainer использует cache только при `cache_allowed=true` и совпадении SHA256
checkpoint, manifest, template, sidecar и frozen state. Иначе он явно
переходит на online path.

### 6. Eight-view fine-only training

Новый состав данных означает model-only initialization через
`--init-checkpoint`; `--resume` здесь неверен.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python tools/train.py \
  --config configs/debug/coordinate_guided_surface_v2/views08.py \
  --device cuda \
  --init-checkpoint "$CKPT04" \
  --cfg-options \
    data.train_manifest="$M08" \
    data.validation_manifest="same_as_train" \
    frozen_feature_cache.audit_path="$CACHE08/frozen_feature_cache_audit.json" \
  --work-dir "$WD"
```

Обучаются только `fine_feature_adapter` и
`fine_coordinate_auxiliary_head`. Каждый sample сначала усредняется по своим
точкам, затем восемь sample losses усредняются с весом `1/8`.

### 7. Gates, package и STOP

```bash
export RUN08="$WD/<КАТАЛОГ_НАПЕЧАТАННЫЙ_TOOLS_TRAIN>"

cat "$RUN08/strict_stage_gate.json"
cat "$RUN08/practical_stage_gate.json"
cat "$RUN08/final_summary.json"

python tools/package_eight_view_coordinate_report.py \
  --input "$RECHECK04" \
  --input "$WD/manifests/coordinate_guided_surface/eight_view_manifest_audit.json" \
  --input "$WD/manifests/coordinate_guided_surface/eight_view_manifest_audit.md" \
  --input "$INIT08" --input "$CACHE08" --input "$RUN08" \
  --output "$WD/eight_view_coordinate_report_$TAG.tar.gz"
```

`strict_stage_gate.json` требует correspondence/alignment p95 не более 1 мм.
`practical_stage_gate.json` разрешает переход при p95 не более 1.5 мм,
rotation не более 0.25° и translation не более 0.10 мм; practical pass никогда
не переписывает strict failure.

После упаковки — **STOP и внешний анализ**. Не запускать автоматически
selective-unfreeze, 10 views, остальные fragments, ranking, regions или
learned confidence. Config `views08_selective_unfreeze.py` только подготовлен.
