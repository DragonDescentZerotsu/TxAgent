# Canonical TDC selection

The only current TDC configuration is `data/gold_labels/TDC/manifest.json`.
Oral and Skin use the frozen scaffold-conditioned cohorts. BBB, Ames, DILI and
Carcinogens use the frozen original-source cohorts. Label files live directly
under `data/gold_labels/TDC/<Task>/`; no CLI selects an edition.

The task's `retrieval` fields bind the preserved frozen rankings. Its
`selection.direct_cache` binds full-molecule L1 cards, with parent identity used
only for exclusion. These are different candidate identities. Existing
parent-grouped assay scores cannot score individual full-molecule cards.
Canonical card assay scores are currently uncomputed and remain missing.
Indirect records may retain their exact frozen query/UID scores.

`optimization.selection.search` performs direct or indirect searches with
explicit objective grids, validation cohorts and budgets. `compose` combines
matching frozen panels. Every result exports ordered physical `source_row_uid`
members through `predict.retrieval.selection`; membership, query identity and
parent/scaffold exclusions are checked before export and prediction. The flat
harness accepts the optimizer manifest through `--uids-file` and validates its
configured cache identities before hydration.

Use validation to select settings; freeze them before test prediction. Preserve
missing scores, source identities and prompt provenance. Neither this migration
nor card preparation reruns models or changes the frozen label rows. Historical
headers and relocation metadata live in `data/legacy/tdc/`; completed prediction
artifacts remain unchanged.
