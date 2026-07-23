# Тесты

Оставлены тесты production graph:

- automatic Dataset, schema/index, split isolation, filtering и shell-only;
- boundary erosion/dilation, geometry gates, target alignment и GT leakage;
- symmetry sidecar/groups/targets;
- canonical head, fine adapter, active gradients и отсутствие legacy keys;
- exact-global/K16 projection и Weighted Procrustes;
- registry с единственной production model;
- упаковка compact report.

Запуск:

```bash
export FRAG_DATASET_ROOT=/path/to/dataset
python -m unittest discover -s tests -v
```

Real-dataset tests читают несколько samples и не изменяют dataset. Ни один тест
не запускает полное обучение; CUDA-проверка выполняется отдельно коротким smoke.
