# symm-template-reg

Минимальный production-код регистрации видимого фрагмента на единственном
известном эталонном mesh с учётом осевой симметрии.

## Текущая архитектура

В registry есть одна модель: `CoordinateGuidedSurfaceRegistrationV3`.
Shell-точки в системе камеры и точки эталона проходят point encoders,
двунаправленное interaction, dual-stream geometry и fine adapter. Голова
предсказывает канонические координаты `q_aux` в системе объекта. На inference
используются exact-global либо `q_aux`-guided K=16 поиск треугольников, точная
проекция и uniform `WeightedProcrustes`. Hard projection не входит в
дифференцируемый training path.

В production отсутствуют pose queries, ranking, learned patch/triangle/
barycentric/confidence/region/overlap heads и direct pose regression.

## Dataset

Корень имеет физически раздельные `train/`, `val/`, `test/`; внутри split
лежат `models/` и `scene_*`. `SplitDirectoryFragmentDataset` автоматически
сканирует все сцены, кадры и наблюдения физических фрагментов без manifest.
Sample ID: `split/scene_XXXXXX/frame_XXXXXX/fragment_XXXX`.

Template `.ply`, `.symmetry.json` и `.meta.json` проверяются между splits.
Raw PLY может отличаться сохранёнными normals, поэтому контракт использует
semantic SHA-256 вершин и faces и отдельно сообщает raw hash. Подробности:
[DATASET_RU.md](docs/DATASET_RU.md).

## Установка

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
python -m pip install -e . --no-deps
```

## Inventory

```bash
python tools/inspect_dataset.py \
  --dataset-root "$FRAG_DATASET_ROOT" \
  --output-dir "$FRAG_WORK_DIR/dataset_inventory"
```

## Boundary augmentation

Только train может локально erode/dilate границу shell mask. Добавляемые
fracture/depth-ring точки проходят depth gate и точную проекцию на template;
GT pose используется только для построения train target и не подаётся модели.
См. [AUGMENTATION_RU.md](docs/AUGMENTATION_RU.md).

## Smoke

```bash
python tools/check_cuda.py
python tools/train.py --config configs/debug/smoke.py --device cuda \
  --from-scratch --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

Smoke ограничен пятью optimizer steps и одним val batch.

## Обучение и resume

```bash
python tools/train.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --device cuda --from-scratch \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"

python tools/train.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --device cuda --resume "<RUN>/checkpoints/latest.pth" \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

`latest.pth` содержит model, optimizer, scheduler, RNG, epoch, позицию batch,
global optimizer step и fingerprints train/val индексов. Подробности и profile:
[TRAINING_RU.md](docs/TRAINING_RU.md).

## Validation и explicit test

```bash
python tools/evaluate.py \
  --config configs/eval/coordinate_guided_surface_v3.py \
  --dataset-root "$FRAG_DATASET_ROOT" --split val \
  --checkpoint "<RUN>/checkpoints/best.pth" --device cuda \
  --output-dir "$FRAG_WORK_DIR/evaluation_val"

python tools/evaluate.py \
  --config configs/eval/coordinate_guided_surface_v3.py \
  --dataset-root "$FRAG_DATASET_ROOT" --split test \
  --checkpoint "<RUN>/checkpoints/best.pth" --device cuda \
  --output-dir "$FRAG_WORK_DIR/evaluation_test"
```

Test индексируется только второй явной командой и всегда получает warning, что
его нельзя использовать для model selection. См. [INFERENCE_RU.md](docs/INFERENCE_RU.md).

## Визуализация и export

`tools/visualize_predictions.py`, `tools/visualize_boundary_augmentation.py` и
`tools/visualize_template_symmetry.py` сохраняют PLY/PNG/JSON diagnostics.
`tools/export_model.py` экспортирует чистый state dict с SHA-256.
`tools/package_training_report.py` создаёт компактный архив без PTH/PT/PLY/
NPY/NPZ.

## Структура

```text
configs/{train,eval,debug}/
docs/
symm_template_reg/{datasets,engine,evaluation,geometry,models,visualization}/
tests/
tools/
```

## Ограничения

Модель рассчитана на один известный template, uniform Procrustes и небольшие
ошибки границы, прошедшие geometry gates. Она не является robust estimator для
произвольных фоновых выбросов. FP16/BF16 не включаются без отдельного
equivalence audit. `test` не используется при подборе модели.


```bash
export LARGE_DATASET_ROOT=/путь/к/полному_dataset
export FRAG_DATASET_ROOT=/путь/к/воркдир

python tools/train.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --device cuda \
  --from-scratch \
  --work-dir "$FRAG_WORK_DIR" \
  --cfg-options \
    data.dataset_root="$LARGE_DATASET_ROOT"
```