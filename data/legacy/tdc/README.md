# Historical TDC inputs

Current inputs are configured by `data/gold_labels/TDC/manifest.json`.
This directory preserves previous cohorts and byte-exact metadata for reading
completed experiments. It never selects the current dataset.

- `labels/`: noncanonical label editions and original publication headers.
- `metadata/`: original small cache headers, mirrored by their former repository paths.
- `configs/`: the 19 retired edition-specific cache YAMLs.
- `code/`: historical benchmark construction and exact-score indexing sources.
- `relocations.json`: exact metadata locations and moved payload prefixes.

`data.processing.paths.resolve_legacy_path` consults the explicit map when a
recorded path is absent. Relative paths inside archived headers retain their
original base. There are no filesystem aliases. Scientific payloads and completed
prediction results are not rewritten or relabeled.

The scaffold-disjoint cache migration also preserves its original versioned
headers here. Active bindings use the single `tdc` namespace; database bytes,
payload content IDs and frozen query cohorts are unchanged. The migration audit
is `outputs/analysis/retrieval_cache/tdc_versionless_20261015/`.
