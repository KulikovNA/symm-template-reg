# Debug configs

- `smoke.py` — весь active graph, ровно 5 optimizer steps и один val batch.
- `tiny_overfit.py` — selector одной сцены/кадров/fragments, максимум 100 steps;
  запускается только явной командой пользователя.
- `four_fragments_four_frames_overfit.py` — 16 observations одной train-сцены:
  fragments `0–3`, frames `2,4,5,8`; validation использует те же observations
  без augmentation. Один optimizer step соответствует одной полной эпохе.
- `augmentation_preview.py` — параметры визуализации boundary augmentation,
  без обучения.

```bash
python tools/train.py --config configs/debug/smoke.py --device cuda \
  --from-scratch --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

Selectors задаются непосредственно config-полями `scene_ids`, `frame_ids`,
`fragment_ids`, `max_samples`; manifest не используется.

```bash
python tools/train.py \
  --config configs/debug/four_fragments_four_frames_overfit.py \
  --device cuda --from-scratch \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```
