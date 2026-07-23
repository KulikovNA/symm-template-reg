# Production Dataset

`SplitDirectoryFragmentDataset(dataset_root, split)` не принимает manifest.
Он детерминированно сканирует `split/scene_*/visible_points/frame_*.npz`,
сопоставляет GT и физический mesh, затем создаёт observation на каждый
`scene/frame/fragment`.

По умолчанию принимаются физические fragments с `num_faces >= 840`,
observations с не менее 128 shell points; до модели остаётся максимум 4096.
Непрошедший physical fragment исключается целиком во всех его кадрах.
Train поддерживает random/FPS, val/test — только детерминированную политику.

Fingerprint включает root, split, сцены, NPZ metadata, fragment hashes,
semantic template hash, sidecar hash, filters и selectors. Кэш индекса пишется
только в work directory. Dataset не изменяет исходные данные.

`inspect_dataset.py` сохраняет schema, counts, accepted/rejected statistics,
template contract и leakage audit. Одинаковый template между splits ожидаем;
совпадение physical fragment SHA-256 сообщается как leakage.
