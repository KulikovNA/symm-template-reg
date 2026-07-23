# Ten-view clean V3 scratch runbook

## Назначение

Это controlled-overfit на десяти ракурсах одного физического `fragment_0002`
(frames `0..9`). Train и validation намеренно используют одни и те же samples.
Результат показывает обучаемость активной архитектуры, но не является оценкой
обобщения.

Scratch-run принципиально отличается от прежней последовательности
`1 view -> 2 views -> 4 views -> 8 views`: он не загружает model weights,
optimizer, scheduler или counters из checkpoint. Обязателен флаг
`--from-scratch`; `--resume`, `--init-checkpoint` и `--init-modules` с ним
взаимоисключаются.

## Чистый active graph

`CoordinateGuidedSurfaceRegistrationV3` содержит только путь:

```text
observed shell points -> observed_encoder --------------------------+
                                                                    |
template surface points -> template_encoder ------------------------+
        -> bidirectional interaction_transformer                    |
        -> dual_stream_geometry_encoder(matching_only)              |
        -> dense_observed_fine_projection                           |
        -> fine_template_projection + template_context_projection   |
        -> fine_feature_adapter(observed_only)                       |
        -> canonical_coordinate_head -> normalized q_aux            |
        -> object-bbox decode -> raw q_aux in O                      |
        -> parameter-free uniform WeightedProcrustes -> T_C_from_O
```

Exact-global и q-guided K16 mesh projection выполняются только при evaluation,
gate, visualization и inference. Через hard projection gradient не идёт.

В active graph отсутствуют absolute/residual pose queries, pose ranking/direct
pose head, patch classifier, learned triangle classifier, learned barycentric
head, active/observed-region heads, insufficient-information head, learned
correspondence-confidence head и старые overlap/visibility heads. Legacy-код
остаётся в репозитории только для воспроизводимости прежних config.

## Scratch initialization

При seed `0` модель применяет следующие правила до первого optimizer step:

- `Linear`: Xavier uniform, bias zero;
- `Conv1d`: Kaiming uniform для ReLU, bias zero;
- `LayerNorm`: scale one, bias zero;
- последний `q_aux` Linear: normal с `std=1e-3`, bias zero.

`initialization_summary.json` содержит SHA256 исходного `state_dict`, seed,
правила и пустой список checkpoint sources. До обучения также сохраняются
`scratch_initialization_per_sample.csv` и
`scratch_initialization_summary.json`. Permutation audit проверяет, что
перестановка десяти входов только переставляет соответствующие outputs.

## Loss и optimizer

Loss сначала считается отдельно на каждом sample, затем десять sample losses
усредняются с одинаковым весом. Фактические base weights/scales:

| component | weight | scale |
|---|---:|---:|
| coordinate mean | 1.0 | raw normalized-coordinate loss |
| coordinate top-10% tail | 0.5 | raw normalized-coordinate loss |
| raw-q Procrustes rotation | 0.25 | 1 degree |
| raw-q Procrustes translation | 0.25 | 0.001 m |
| raw-q visible alignment | 0.25 | 0.001 m |

Один общий допустимый symmetry element `S` выбирается совместно для всех пяти
компонентов. На epochs `0..249` pose/alignment weights линейно растут от нуля;
с epoch `250` используются полные веса. Модули во время loss warmup не
замораживаются.

AdamW использует `weight_decay=0`, gradient clipping `max_norm=1`. Encoders,
interaction и geometry имеют LR `1e-4`; dense/fine projections, adapter и
coordinate head — `3e-4`. Scheduler линейно разогревает LR первые 100 optimizer
steps, затем сохраняет его постоянным.

## Подготовка и короткие аудиты

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export SOURCE_MANIFEST="$WD/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json"
export M10="$WD/manifests/coordinate_guided_surface/fragment0002_views10_shell_only.json"
export TAG=$(date +%Y%m%d_%H%M%S)
```

Manifest builder намеренно отказывается перезаписывать существующий output.
Следующая команда нужна один раз, если `$M10` ещё отсутствует:

```bash
python tools/build_coordinate_view_manifest.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --frames 0 1 2 3 4 5 6 7 8 9 \
  --shell-only \
  --output "$M10"
```

Проверка реального autograd graph всех десяти samples:

```bash
python tools/audit_active_parameter_graph.py \
  --config configs/debug/coordinate_guided_surface_v3/views10_scratch_full.py \
  --manifest "$M10" \
  --device cuda \
  --output-dir "$WD/ten_view_active_graph_$TAG"
```

CUDA smoke перебирает `10x1`, затем только при OOM явно пробует `5x2` и `2x5`.
Первый прошедший режим записывается вместе с peak memory и benchmark:

```bash
python tools/smoke_ten_view_scratch.py \
  --config configs/debug/coordinate_guided_surface_v3/views10_scratch_full.py \
  --manifest "$M10" \
  --device cuda \
  --output-dir "$WD/ten_view_scratch_smoke_$TAG"
```

Не переносите fallback из smoke в full config молча. Проверьте
`ten_view_scratch_smoke.json`; если выбран не `batch_size=10`, явно задайте в
копии нового config соответствующие batch и accumulation. Число views на один
optimizer step всегда должно оставаться 10. Frozen-feature cache запрещён;
кэшируется только неизменяемая геометрия.

## Полное обучение с нуля

Запускать только после passing manifest, active-graph и CUDA smoke:

```bash
python tools/train.py \
  --config configs/debug/coordinate_guided_surface_v3/views10_scratch_full.py \
  --device cuda \
  --from-scratch \
  --cfg-options \
    data.train_manifest="$M10" \
    data.validation_manifest="same_as_train" \
  --work-dir "$WD"
```

В этой команде не должно быть `--init-checkpoint` или `--resume`. Основной
budget — 6000 epochs/optimizer steps, eval каждые 50 epochs, visualization
каждые 250. Best checkpoint минимизирует худший из десяти sample scores, а не
среднее. Warm-start config только подготовлен и до анализа scratch-run не
запускается.

## Проверка результата

Подставьте напечатанный trainer каталог:

```bash
export RUN10="$WD/<НАПЕЧАТАННЫЙ_RUN_DIR>"

cat "$RUN10/strict_surface_gate.json"
cat "$RUN10/practical_surface_gate.json"
cat "$RUN10/pose_placement_gate.json"
cat "$RUN10/stage_gate.json"
cat "$RUN10/final_summary.json"
```

Gates независимы:

- strict surface: p95 correspondence/alignment `<=1 mm`, rotation `<=0.25°`,
  translation `<=0.10 mm`;
- practical surface: p95 correspondence/alignment `<=2 mm`, rotation `<=0.5°`,
  translation `<=0.5 mm`;
- pose placement: rotation `<=0.5°`, translation `<=0.5 mm`, surface membership
  p95 `<=0.1 mm`.

Каждый gate дополнительно требует rank 3, K16 recall `>=0.995`, fallback 0 на
каждом из десяти кадров. Один плохой кадр блокирует соответствующий общий gate;
strict failure никогда не переписывается как success.

## Компактный архив и обязательный STOP

```bash
python tools/package_ten_view_scratch_report.py \
  --input "$WD/ten_view_active_graph_$TAG" \
  --input "$WD/ten_view_scratch_smoke_$TAG" \
  --input "$RUN10" \
  --output "$WD/ten_view_scratch_report_$TAG.tar.gz"
```

Архив содержит только JSON/CSV/JSONL/MD: manifest audit, active graph,
initialization, smoke/benchmark, resolved config, history, best/per-frame
metrics, три gates, world matrices, conditioning/gradient audits и diagnosis.
Он исключает PTH/PLY/PT/NPY/NPZ.

После получения архива — **STOP**. Не запускать автоматически warm-start,
no-warmup ablation, bf16 variant или новую архитектуру. Сначала передать на
внешний анализ:

- `ten_view_scratch_report_$TAG.tar.gz`;
- `active_parameter_graph.json` и report MD;
- `ten_view_scratch_smoke.json` и training benchmark;
- `initialization_summary.json` и scratch baseline summary;
- `resolved_config.json`, `history.jsonl`, `best_metrics.json`;
- три gate JSON, `stage_gate.json`, `final_summary.json`;
- per-sample exact-global/K16 CSV;
- world-frame matrix CSV и conditioning/gradient JSON/CSV;
- diagnostic failure JSON, если gate не пройден.
