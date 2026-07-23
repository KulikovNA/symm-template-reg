# Four fragments × four views scratch overfit

Это контролируемый overfit на 16 наблюдениях одной сцены и одного эталона, а не
оценка generalization. Один набор весов видит четыре физически разные геометрии
`fragment_0000..0003` в общих кадрах `2, 4, 5, 8`. Train и validation намеренно
совпадают; результаты нельзя использовать как финальную test-метрику.

## Переменные

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export DATASET=/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test
export MANIFEST="$WD/manifests/multifragment_overfit/scene000000_fragments0000_0003_frames0002_0004_0005_0008_shell_only.json"
export CONFIG=configs/debug/coordinate_guided_surface_v3/four_fragments_four_views_scratch.py
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Обязательные pre-training gates

```bash
python tools/audit_four_fragments_four_views.py \
  --dataset-root "$DATASET" --scene-id scene_000000 \
  --fragment-ids 0 1 2 3 --frame-ids 2 4 5 8 --min-num-faces 840 \
  --output-dir "$WD/four_fragments_four_views_audit_$TAG"

python tools/audit_overfit_identifiability.py \
  --manifest "$MANIFEST" \
  --output-dir "$WD/four_fragments_four_views_identifiability_$TAG"

python tools/audit_multifragment_active_parameter_graph.py \
  --config "$CONFIG" --manifest "$MANIFEST" --device cuda \
  --output-dir "$WD/four_fragments_four_views_active_graph_$TAG"

python tools/smoke_multifragment_batch.py \
  --config "$CONFIG" --manifest "$MANIFEST" --device cuda \
  --output-dir "$WD/four_fragments_four_views_smoke_$TAG"
```

Продолжать можно только если data audit имеет `selection_passed=true`, в
identifiability audit нет `non_identifiable_training_pair`, ID-counterfactual
прошёл, active graph не содержит trainable-параметров без gradient и smoke выбрал
один из режимов 16×1, 8×2, 4×4, 2×8. Если выбран не 16×1, перенесите напечатанные
`actual_batch_size` и `gradient_accumulation_steps` через `--cfg-options`; fallback
нельзя выбирать молча.

## 2. Scratch training

Для выбранного 16×1 режима:

```bash
python tools/train.py \
  --config "$CONFIG" --device cuda --from-scratch \
  --cfg-options \
    data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train" \
    data.train_batch_size=16 \
    train.gradient_accumulation_steps=1 \
  --work-dir "$WD"
```

Запрещены `--resume` и `--init-checkpoint`. Модель использует только геометрию,
а не fragment/frame ID. Каждый sample сначала получает собственный loss, затем
берётся среднее: веса sample/fragment/frame равны 1/16, 1/4 и 1/4.

## 3. Проверка результата

```bash
export RUN4X4="$WD/<НАПЕЧАТАННЫЙ_RUN_DIR>"
cat "$RUN4X4/strict_surface_gate.json"
cat "$RUN4X4/practical_surface_gate.json"
cat "$RUN4X4/pose_placement_gate.json"
cat "$RUN4X4/final_summary.json"
```

- strict: correspondence/alignment ≤1 мм, rotation ≤0.25°, translation ≤0.10 мм;
- practical: correspondence/alignment ≤2.5 мм, rotation ≤1°, translation ≤0.50 мм;
- pose placement: rotation ≤1°, translation ≤0.50 мм, surface membership p95 ≤0.1 мм.

Во всех трёх дополнительно нужны rank=3, K16 recall ≥0.995 и fallback=0 для
каждого из 16 samples. Один плохой sample блокирует gate. Смотрите также
`best_evaluation/per_fragment_metrics.csv`, `per_frame_metrics.csv` и
`world_metrics_per_fragment.csv`. PLY нужны только для визуальной проверки:
четыре цвета fragment ID, predicted против GT в каждом frame и четыре world poses
одного физического fragment.

## 4. Компактный архив и обязательный STOP

```bash
python tools/package_four_fragments_four_views_report.py \
  --input "$WD/four_fragments_four_views_audit_$TAG" \
  --input "$WD/four_fragments_four_views_identifiability_$TAG" \
  --input "$WD/four_fragments_four_views_active_graph_$TAG" \
  --input "$WD/four_fragments_four_views_smoke_$TAG" \
  --input "$RUN4X4" \
  --output "$WD/four_fragments_four_views_report_$TAG.tar.gz"
```

Архив содержит только JSON/CSV/JSONL/MD, без PTH/PLY/PT/NPY/NPZ. После упаковки
обязательный **STOP**: не запускать warm-start или новый config до анализа gates,
худшего sample, худшего fragment и худшего frame.

