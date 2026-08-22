# ClinTox clinical-trial-failure canonical source QA

- contract: `clintox_clinical_trial_failure.v1`
- status: `benchmark_ready`
- source rows: 1,509
- labeled molecular parents: 1,428
- Y=0/Y=1: 1,324/104
- AACT/FDA source-role overlap parents: 65

## Definition

Y=1 requires membership in the frozen AACT-derived toxicity-failed-trial list. Y=0 requires membership in the FDA-approved comparator list and absence from the positive source after molecular-parent normalization. Starling prose and mechanistic toxicity records do not vote.

## Completeness and validity

- identity audit reconciliation: 1,509 rows
- invalid/unresolved source structures: 4
- joined-reference common parent label matches/mismatches: 1,421/0
- source-only/reference-only parents: 7/0

## Starling evidence coverage (not labels)

- benchmark parents with clintox_base_v1 evidence: 1,128
- Starling rows used to create labels: 0

## Leakage boundary

The canonical coverage table is descriptive only. Before retrieval-based evaluation, valid/test parents must be removed from every ClinTox evidence library and query-time retrieval must remain parent-disjoint.
