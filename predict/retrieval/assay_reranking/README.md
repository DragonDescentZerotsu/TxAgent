# Assay-reranking releases

`CACHE_LAYOUT.md` describes cache ownership, active aliases, and the one-time
artifact verification receipt. `MODEL_CONTRACTS.md` records model and training
prompt revisions. Tracked `flat_v5_manifests/` files identify portable cache and
evidence payloads; release indexes and their manifests are the runtime source of
truth.

The active BBB/Oral Gold-v1 reader uses independent task/split/level caches:
100 frozen scaffold-disjoint L1 context cards and parent-disjoint physical L2+
UIDs. At L1, use the card's frozen ordered UID membership rather than choosing
one representative. Later levels select indexed UIDs independently. Prompt
preparation reads the selected indexes and projection rows; full database hashes,
SQLite integrity scans, and source equivalence belong to publication or restore.

V27 safety and Skin successors select a shared parent universe across later
levels and retain every physical UID. The Gold Carcinogens and TDC Ames top-40
successors are opt-in; partial snapshots remain explicit and may contain missing
assay scores. Cache releases never impute missing scores or describe a partial
snapshot as fully scored. The opt-in Gold Carcinogens validation selector may use
the mean of a query's observed scores for selection only; selected unscored records
remain scoreless in the prompt and the immutable cache.

Canonical TDC reads `data/gold_labels/TDC/manifest.json` for all six tasks.
Its `retrieval` fields bind the selected frozen L1/L2+ rankings under
`data/caches/assay_reranking/active/flat_v5/tdc/<task>/`; `selection.direct_cache`
binds the new full-molecule L1 cards separately. Parent-grouped scores do not
apply to individual cards. Existing exact-score snapshots remain frozen and
missing card scores stay explicit. See [the TDC contract](../../../optimization/TDC_CONTRACT.md).
