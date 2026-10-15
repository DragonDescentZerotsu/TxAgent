# Context selection and semantic grouping

This owner turns pinned retrieval candidates into ordered context/UID selection manifests consumed by the prediction harness. Paths here are relative to `optimization/`.

- `selection/pool.py` assembles canonical cards/records and exact scores; `objectives.py` owns math, `policies.py` owns direct/indirect contracts and `search.py` executes JSON grids.
- Direct/L1 parent normalization is filtering-only. Rank and score full molecules; count distinct canonical context cards and preserve their authoritative binary labels and physical members.
- Direct/L1 construction and selection require scaffold-disjoint candidates. Only indirect L2+ supports both parent-disjoint and scaffold-disjoint cache sets; never weaken L1 when switching the indirect policy.
- Indirect selection is joint across L2+, using physical UIDs and ledger-bound semantic projections. Exact assay scores remain query/card-specific; missing scores remain missing.
- `selection/record_scores.py` owns source-bound DILI caches; keep geometric relevance, frozen direct controls and declared diversity requirements.
- `semantic_buckets/inputs.py` resolves selected maps, weights and ledgers. `dili_score/` remains the offline scorer.
- Objectives combine declared relevance and diversity terms; emitted selections bind query identities, source hashes, objective profiles and ordered records for later inference.

- Select on the declared validation cohort and freeze parameters before test inference.
- Expose selection controls through shared selection CLIs; Bash recipes instantiate sweeps. All context optimizers export ordered physical UIDs, including frozen L1 card member records, and validate candidate-universe membership and parent/scaffold disjointness before export.
- Preserve ordered UIDs, direct/indirect budgets and query/label identities; reject missing or incompatible inputs.
- Follow [TDC_CONTRACT.md](TDC_CONTRACT.md) and [README.md](README.md); keep selection independent of inference prompt prose.
- Completed selection outputs are immutable; temporary experiments belong in `/tmp`.

- Prediction metadata uses one shared batch/FINAL contract with named versions and query UIDs, read through `predict.harnesses.branches.manifests`. Paths locate runtime inputs; they and content fingerprints are not experiment identity. Per-query state belongs in existing checkpoints, never duplicated manifests. Preserve sparse scientific overrides and reject incompatible or unknown identities before reuse or inference.
- Explicitly approved metadata migrations may convert historical prediction headers and operational indexes. Preserve frozen scientific payloads and leave compressed FINAL bundles untouched unless a push is requested.

Use `--benchmark tdc` and its canonical manifest. Direct search uses `selection.direct_cache`; indirect search uses `retrieval.later`. Never attach parent-aggregated L1 scores to full-molecule cards. Missing card scores stay missing until separately scored.

Indirect selection resolves the same named retrieval cache set as prediction. TDC direct search retains the separate canonical `selection.direct_cache` binding.
