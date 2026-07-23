# Архитектура

Единственная trainable model — `CoordinateGuidedSurfaceRegistrationV3`.

```text
observed shell C + template surface O
→ point encoders → interaction → dual-stream geometry
→ dense projections → fine local adapter
→ canonical coordinate head → q_aux(O)
```

Training оптимизирует `CleanCoordinatePoseLossV3`. Каждая sample сначала
редуцируется по своим точкам, затем samples усредняются с одинаковым весом.
Один symmetry element выбирается совместно для coordinate и pose terms.

Inference выполняет точную проекцию на mesh (global либо K=16 shortlist),
аналитические barycentric coordinates и uniform Weighted Procrustes.
Проекция не участвует в gradient path. Все координаты, SVD, Procrustes и
physical metrics остаются FP32.
