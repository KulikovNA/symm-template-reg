# Обучение

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
export FRAG_DATASET_ROOT=/path/to/dataset
export FRAG_WORK_DIR=/path/to/work_dirs

python tools/profile_training.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --dataset-root "$FRAG_DATASET_ROOT" --device cuda \
  --output-dir "$FRAG_WORK_DIR/training_profile"

python tools/train.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --device cuda --from-scratch \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

Defaults: FP32, AdamW; encoder/interaction LR `1e-4`, fine/coordinate path
`3e-4`, weight decay `1e-4`, gradient clip `1.0`, cosine schedule, 1000-step
LR и loss warmup, effective batch 16, максимум 150 epochs или 100000 optimizer
steps — что наступит раньше.

Resume:

```bash
python tools/train.py \
  --config configs/train/coordinate_guided_surface_v3.py \
  --device cuda --resume "<RUN>/checkpoints/latest.pth" \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

Пять шагов smoke:

```bash
python tools/train.py --config configs/debug/smoke.py \
  --device cuda --from-scratch \
  --cfg-options data.dataset_root="$FRAG_DATASET_ROOT" \
  --work-dir "$FRAG_WORK_DIR"
```

Полное обучение автоматически никаким tool не запускается.
