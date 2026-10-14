# Optimization

`selection/` owns candidate pools, objective optimization and JSON grid execution.
`semantic_buckets/` owns bucket construction, refinement, sliding utility scoring,
physical membership ledgers and explicit publication. `dili_score/` is the unchanged
standalone deterministic record scorer.

See [selection](selection/README.md) for direct/indirect policies and prepared L1 card
caches, and [semantic buckets](semantic_buckets/README.md) for construction and scoring.
Experiment settings are Bash recipes; Python owns scientific algorithms, validation,
execution and artifact handling. Prediction/evaluation remains in `predict/`.
