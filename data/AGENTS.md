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
- See [gold-label processing](processing/gold_labels/README.md) and the [V10 construction contract](processing/evidence_library/versions/v10/README.md).
- Share build mechanics, but preserve version-specific scientific policies and stages. Bash recipes specify experiment inputs; use `predict.api_client.transport` for SDK construction and the shared API pool for provider routing.
