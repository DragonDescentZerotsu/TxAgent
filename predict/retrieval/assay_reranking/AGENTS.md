# Assay-reranking boundary

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This package prepares reusable assay-transfer scores before a reasoning harness runs.
Harnesses consume finalized caches and must not call these checkpoints online.

Only two model lineages are supported: task-specific V9 for direct gold-label transfer
and BBB V19.1 for record-level transfer. Add no compatibility fallback, alternate
checkpoint, or copied historical prompt here.

`runtime.py` owns the pinned task-to-role mapping and cache contract; `v9.py` owns its
prompt and Morgan train-neighbor prefilter; `v19_1.py` owns its record-level prompt.
`progressive_levels.py` is the independent offline BBB cache builder for the frozen
historical level overlay. Keep direct and indirect scientific contracts separate.

The revised BBB progressive cache keeps conditioned benchmark votes in L1. L2 is the
union of historical near-direct L2 and historical L1 rows that did not vote; L3-L5 use
the historical v5 passive, efflux, and influx row assignments. L1 reuses the V9 gold
cache at Morgan width 75. L2-L5 use the V19.1 checkpoint, the same V19.1 template, and
native source context; the small direct-BBB projection extension is explicit because
those records are outside the checkpoint's numeric-indirect training domain.

The level cache uses current normalized-v7 Stage-06 retrieval eligibility and only
borrows family assignment from the historical v5 overlay. It is therefore not a replay
of the older progressive index. Candidate selection is independently Morgan top-75 per
level after parent-identity exclusion, with no similarity floor.

Generated candidates and scores live under `predict/retrieval/cache/assay_reranking/`
and are ignored by Git. They are reusable retrieval inputs, not paper outputs. Prefer
adding a clear section to an existing owner over adding a small helper module.
