# Runbook производительности: 4 фрагмента × 4 ракурса

Этот эксперимент всегда использует 16 samples: фрагменты `0000…0003` и кадры
`000002, 000004, 000005, 000008`. Каждый sample имеет вес `1/16`. Полный run
запускается только с `seed=0`, `--from-scratch`, без checkpoint.

## Переменные

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export MANIFEST="$WD/manifests/multifragment_overfit/scene000000_fragments0000_0003_frames0002_0004_0005_0008_shell_only.json"
export BASE_CFG=configs/debug/coordinate_guided_surface_v3/four_fragments_four_views_scratch.py
export FAST_CFG=configs/debug/coordinate_guided_surface_v3/four_fragments_four_views_scratch_optimized_fp32.py
export TAG=$(date +%Y%m%d_%H%M%S)
```

## 1. Исторический baseline 14.7 с

Исторические измерения находятся в
`coordinate_guided_surface_v3_four_fragments_four_views_scratch_20260722_202532`.
Не продолжайте этот run. Его `history.jsonl` показывает 14.65–14.70 с при
`data_time≈0.08 с`. Текущий код на том же config воспроизводит около 1.35–1.68 с,
поэтому прежние 14.7 с нельзя честно воспроизвести без сохранённого тогда
profiler trace и снимка исходников. Старый standalone smoke синхронизировал CUDA,
но использовал другой SDP backend, поэтому его 5.18 с нельзя напрямую сравнивать
с trainer.

Короткая проверка реального trainer (не полный эксперимент):

```bash
python tools/train.py --config "$BASE_CFG" --device cuda --from-scratch \
  --cfg-options train_budget.epochs=3 train.max_epochs=3 \
    train.max_optimizer_steps=3 train.evaluate_before_training=false \
    train.visualize_before_training=false train.eval_interval_optimizer_steps=0 \
    train.debug_visualization_interval_optimizer_steps=0 \
  --work-dir "$WD"
```

## 2. Профиль шага

```bash
python tools/profile_training_step.py \
  --config "$BASE_CFG" --manifest "$MANIFEST" --device cuda \
  --warmup-steps 10 --measure-steps 20 \
  --output-dir "$WD/multifragment_profile_baseline_$TAG"
```

Профиль использует CUDA Events и явную синхронизацию. Полный trace остаётся
рядом с профилем, но не включается в компактный архив.

## 3. Static geometry cache

```bash
python tools/build_static_geometry_cache.py \
  --config "$BASE_CFG" --manifest "$MANIFEST" --device cuda \
  --output-dir "$WD/static_geometry_cache_$TAG"
```

В кэше разрешены только координатные kNN/FPS/interpolation indices, masks,
необучаемые расстояния и topology. Learned encoder/attention/fine features и
`q_aux` не кэшируются. При augmentation кэш запрещён validation-ошибкой.

## 4. Строгий fp32 equivalence gate

```bash
python tools/audit_training_optimization_equivalence.py \
  --baseline-config "$BASE_CFG" --optimized-config "$FAST_CFG" \
  --manifest "$MANIFEST" --device cuda --steps 20 \
  --output-dir "$WD/multifragment_fp32_equivalence_$TAG"
```

Продолжать можно только при `audit_passed=true`. Проверяются `q_aux`, все loss,
C2/C4 selection, Procrustes R/t, gradients, clipped AdamW update и 20 шагов.

## 5. Batch modes и padding

```bash
python tools/benchmark_training_batch_modes.py \
  --config "$FAST_CFG" --manifest "$MANIFEST" --device cuda \
  --warmup-steps 10 --measure-steps 20 \
  --output-dir "$WD/multifragment_batch_benchmark_$TAG"
```

Проверяются `16×1`, `8×2`, `4×4`, `2×8`. Для micro-batches используется
size-bucketing; optimizer step всё равно получает сумму 16 sample losses / 16.
Выбирается самый быстрый режим, прошедший gradient/update equivalence.

## 6. Optimized fp32 и короткие 20 шагов

Config `four_fragments_four_views_scratch_optimized_fp32.py` не меняет модель,
loss, число точек, seed или физические метрики. AMP и `torch.compile` выключены.

```bash
python tools/train.py --config "$FAST_CFG" --device cuda --from-scratch \
  --cfg-options train_budget.epochs=20 train.max_epochs=20 \
    train.max_optimizer_steps=20 train.evaluate_before_training=false \
    train.visualize_before_training=false train.eval_interval_optimizer_steps=0 \
    train.debug_visualization_interval_optimizer_steps=0 \
  --work-dir "$WD"
```

## 7. Полное обучение с нуля

Выполнять только после успешных cache, batch и 20-step equivalence audit:

```bash
python tools/train.py \
  --config "$FAST_CFG" --device cuda --from-scratch \
  --cfg-options data.train_manifest="$MANIFEST" \
    data.validation_manifest="same_as_train" \
  --work-dir "$WD"
```

Запрещены `--resume` и `--init-checkpoint`. Старый partial run не используется.

## 8. Упаковка и обязательный STOP

```bash
python tools/package_training_performance_report.py \
  --input-dir "$WD/multifragment_profile_baseline_<TAG>" \
  --input-dir "$WD/static_geometry_cache_<TAG>" \
  --input-dir "$WD/multifragment_batch_benchmark_<TAG>" \
  --input-dir "$WD/multifragment_fp32_equivalence_<TAG>" \
  --output "$WD/four_fragments_four_views_performance_<TAG>.tar.gz"
```

Архив содержит только JSON/CSV/JSONL/MD. Веса, PLY/NPY/NPZ/PT и полный trace
исключены.

**STOP:** автоматический запуск полного 8000-step обучения в диагностическом
workflow запрещён. После отчёта и архива передайте их для внешнего анализа и
явно решите, запускать ли полный run.

## Optional ablations

AMP проверяется только после passing fp32 audit командой
`tools/audit_mixed_precision_equivalence.py`; основной config от этого не
меняется. `torch.compile` также остаётся выключенным: текущий graph имеет breaks
на packed→padded, runtime SHA256 и scatter/index_put, а Inductor отклоняет
`fill_diagonal_` над view-chain в fine diagnostics.
