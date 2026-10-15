# Data publication

This owner builds benchmark labels and experimental evidence consumed by retrieval and prediction. Paths here are relative to `data/`.

- `processing/paths.py` resolves canonical roots and evidence releases; `processing/gold_labels/conditioned_benchmark.py` resolves benchmark splits.
- `processing/gold_labels/` publishes `gold_labels/<Task>/<release>/scaffold/` and canonical TDC `gold_labels/TDC/<Task>/`.
- `processing/evidence_library/versions/` owns release-specific construction, task policies and prompts; `processing/evidence_library/shared/v1/` supplies the pinned shared kernel.
- `evidence_libraries/<task>/<release>/` contains canonical records, level mappings and retrieval projections that cache readers hydrate.
- `processing/` owns builds; `gold_labels/`, `evidence_libraries/` and `caches/` own active releases. `artifacts/` and `legacy/` are audit-only.

- Resolve Gold/evidence through `CURRENT` and release manifests. TDC uses `gold_labels/TDC/manifest.json`: Oral/Skin frozen scaffold splits; BBB/Ames/DILI/Carcinogens frozen original-source splits. No TDC version dispatch.
- Preserve immutable releases when science changes; TDC keeps one canonical configuration and preserves historical inputs under `legacy/tdc/`. Preserve source rows, reviewed UID decisions, labels, level assignments and previous receipts.
- Hash large files and check SQLite integrity at publication, restore or explicit verification, never during ordinary indexed reads.
- Retrieval cache renames are recorded in `caches/assay_reranking/relocations.json` and resolved through the shared path reader. Preserve scientific payload bytes and content IDs; deleted obsolete releases must never redirect to scientifically corrected scores.
- See [gold-label processing](processing/gold_labels/README.md) and the [V10 construction contract](processing/evidence_library/versions/v10/README.md).
- Share build mechanics, but preserve version-specific scientific policies and stages. Bash recipes specify experiment inputs; use `predict.api_client.transport` for SDK construction and the shared API pool for provider routing.

Gold voter membership must come from uncapped source votes with canonical physical `source_row_uid` values. Card summary IDs and examples are capped and cannot reconstruct membership. Preserve vote units and member weights, and verify every member against the selected evidence release before publication. A physical row can belong to distinct frozen Skin condition cards; membership is a card-to-UID relation. Conflicting frozen counts or unresolved UIDs block publication. See [Gold voter provenance](../docs/look/gold_voter_provenance.md).

Voter protection starts at source import: verify that every physical voter is present before Stage 1, then preserve it through cleaning and deduplication. Skin intentionally retains its broader reviewed L1 set; extend it with missing replayed voters without changing other level assignments. Its corrected Gold v1 follows the pinned source replay and original molecule table, with the previous summaries archived in the correction receipt.

Before advancing evidence `CURRENT`, publish its matching semantic release and UID ledger. Existing indirect maps and weights may carry forward only after verifying that their physical rows and level assignments are unchanged.
