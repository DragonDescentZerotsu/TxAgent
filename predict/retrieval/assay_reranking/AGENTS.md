# Assay reranking

This owner builds and reads frozen candidate rankings and assay-transfer scores. Resolve YAML → bundle manifest/receipt → release index → level database → evidence projection.

- `runtime.py` resolves cache profiles; `*.yaml` binds task/level sources; `flat_v5_manifests/` pins bundles.
- `artifact_bundle.py` checks bundle receipts stored at repository `data/caches/assay_reranking/active/flat_v5/receipts/`.
- `ranked_uid_retrieval.py` selects ranked panels and hydrates physical UIDs from evidence-owned projections; builders publish their indexes and per-level payloads.
- `prompts/` contains assay-transfer scoring templates; prediction templates live in `../../prompts/`.
- Payloads live at repository `data/caches/assay_reranking/active/flat_v5/<benchmark>/<task>/<role>/<method>/<release>/<variant>/`; compact direct ranking sources also live in `../cache/assay_reranking/active/`.
- `../policies.py` defines parent/scaffold disjointness. DILI example below the payload root: `tdc/dili/l2plus/RELEASE_INDEX.json`.
- Keep missing cached scores and partial coverage explicit; select exact releases through manifests, not directory-name guesses.

- Follow [CACHE_LAYOUT.md](CACHE_LAYOUT.md) and the selected bundle manifest and receipt.
- Reuse shared retrieval readers for optimized and manual physical UID selections. Preserve L1 context labels and ordered voter membership; reject invalid candidate membership or disjointness. Specific retrieval comparisons belong in Bash recipes calling the appropriate capability CLI.
- Readers use indexed lookups and existing receipts; publication or explicit verification owns full hashes and integrity checks.
- Reject absent UIDs, incompatible query identities and score/model/template drift before selection or inference.
- Never rescore or rewrite a published cache in place; specialized workflows retain their pinned request contracts.

Canonical TDC cache bindings and task policies live in `data/gold_labels/TDC/manifest.json`, rather than a second versioned YAML. Keep frozen retrieval L1 and new optimizer `l1_cards/` separate; their scores have different scientific identities.
