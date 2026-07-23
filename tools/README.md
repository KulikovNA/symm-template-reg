# Tools

Все outputs направляйте в `FRAG_WORK_DIR`; dataset tools не изменяют dataset.

- `train.py` — scratch/resume production training. Аргументы: config, device,
  work-dir, `--from-scratch` либо `--resume`.
- `evaluate.py` — явный val/test; test не участвует в model selection.
- `inspect_dataset.py` — schema, inventory, template/leakage contracts.
- `profile_training.py` — по одному forward/backward без optimizer step для
  batch 2/4/8/16; сохраняет throughput/memory report.
- `visualize_boundary_augmentation.py` — PNG/PLY/JSON preview одной sample.
- `visualize_template_symmetry.py` — regions, legend и hypotheses gallery.
- `visualize_predictions.py` — raw q_aux, exact projection и registered cloud.
- `export_model.py` — чистый state dict и SHA-256 manifest.
- `package_training_report.py` — компактный `.tar.gz`; PTH/PT/PLY/NPY/NPZ
  исключаются.
- `audit_active_repository_graph.py` — keep/delete dependency audit.
- `check_cuda.py` — read-only CUDA availability.

Примеры всех основных команд находятся в `docs/TRAINING_RU.md` и root README.
Tools не запускают полное обучение сами: budget задаёт выбранный config.
