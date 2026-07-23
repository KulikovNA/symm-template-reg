# Inference и evaluation

`q_aux` проецируется exact-global либо через q_aux-guided K=16 shortlist.
После точной point-to-triangle projection uniform Weighted Procrustes оценивает
`T_C_from_O`.

Evaluation сохраняет metrics по sample, physical fragment и scene:
q_aux/projected p50/p95, alignment, rotation, translation, rank, K16 recall,
fallback fraction и runtime. Best checkpoint определяется только по
`validation/p90_physical_score`; worst sample — только diagnostic.

`--split test` требуется указать явно. Test summary всегда содержит
`test_results_must_not_be_used_for_model_selection=true`.
