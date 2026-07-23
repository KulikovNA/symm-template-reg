# Cleanup report

До удаления созданы branch `archive/pre_minimal_coordinate_guided_v3`, tag
`pre_minimal_coordinate_guided_v3` и полный Git bundle рядом с репозиторием.
Dependency audit выполнен до cleanup и повторён после extraction активных
частей. Из рабочего дерева удалялись только пути из соответствующего
`delete_manifest.json`.

Удалены staged configs, runbooks, legacy tools/tests, старые detectors, heads,
losses и неиспользуемые evaluation/engine modules. Сохранены production Dataset,
V3 model/loss, exact/K16 geometry, Procrustes, symmetry parser/resolver,
train/eval/export/visualization/report tools и legal notices.
