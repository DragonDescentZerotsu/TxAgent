# Assay-reranking boundary

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

This package prepares reusable assay-transfer scores before a reasoning harness runs.
Harnesses consume finalized caches and must not call these checkpoints online.

Only two model lineages are supported: task-specific V9 for direct gold-label transfer
and task-specific BBB/Bioavailability/Skin V19.1 for record-level transfer. Add no
compatibility fallback, alternate
checkpoint, or copied historical prompt here.

`runtime.py` owns the pinned task-to-role mapping and cache contract; `v9.py` owns its
prompt and Morgan train-neighbor prefilter; `v19_1.py` owns its record-level prompt.
`progressive_levels.py` is the independent offline BBB/Bioavailability/Skin cache
builder for the current Stage 3 level sidecar. Keep direct and indirect scientific
contracts separate.

Current progressive membership is assigned from normalized-v7 Stage 3 records by
`data.processing.evidence_library.versions.v7.progressive_level_mapping`. The sidecar
uses current vote-ledger membership rather than the stored `retrieval_source_id` tag.
BBB has L1-L5, Bioavailability has L1-L6, and Skin has L1-L4. Skin exposure records
outside the current progressive curve retain a null level instead of being forced in.

For BBB, L1 contains current direct voters and is globally excluded from this score
cache. L2 is current near-direct evidence plus former-L1 records that do not currently
vote; L3-L5 are current passive, efflux, and influx assignments. Do not globally remove
L2-L5 records merely because their parent occurs in validation/test or another query.
Every retained L2-L5 molecule instead passes per-query `scaffold_disjoint` exclusion.

Build the BBB L2-L5 cache from Stage 3, not Stage 6/7/8 held-out views. Construct fresh
parent-molecule identities and Morgan fingerprints, then take independent top-75 pools
per level with no similarity floor. Globally omit the 260 pinned unresolved-endpoint
records and the 5 pinned rows without a resolvable parent identity. The latter cannot be
Morgan-ranked or checked under `scaffold_disjoint`; do not fall back to raw SMILES. Keep
every other nonnumeric or assay-transfer-ineligible L2-L5 row, and retain its explicit
out-of-domain flag in the level sidecar.

The 260 endpoint exclusions have no usable source endpoint and therefore never entered
the assay-transfer prompt contract. They may be reconsidered only through a reviewed,
source-specific endpoint mapping; do not synthesize `missing_endpoint` prompts.

L1 reuses the V9 gold cache at Morgan width 75. L2-L5 use the BBB V19.1 checkpoint and
the V19.1 raw/source-field prompt projection. Copy the retrieved record's assay context
to the hidden-value query side and use the normalized-v7 parent SMILES for the known
molecule. The direct-BBB projection extension is explicit OOD because L2 direct-source
records were outside numeric-indirect checkpoint training.

Build Bioavailability L2-L6 under the same Stage 3, top-75, no-floor, and per-query
`scaffold_disjoint` policy. Globally exclude L1 and the one pinned row without a
resolvable parent identity. Its `hf_bioavailability` L2 nonvoters use an explicit OOD
projection extension because that direct source was outside V19.1 indirect training.
Build Skin L3-L4 with its task-specific V19.1 checkpoint under the same Stage 3,
top-75, no-floor, and per-query `scaffold_disjoint` policy. L4 is the distinct-hazard
phototoxicity/irritation/local-damage family, not sensitization-label evidence. The
`direct_skin_reaction` L3 projection extension is explicit OOD because that source was
outside V19.1 numeric-indirect training. There is no checkpoint fallback.

Generated candidates and scores live under `predict/retrieval/cache/assay_reranking/`
and are ignored by Git. They are reusable retrieval inputs, not paper outputs. Prefer
adding a clear section to an existing owner over adding a small helper module.
