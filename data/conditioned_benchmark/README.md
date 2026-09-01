# Conditioned Benchmark data

This directory is the single active gold dataset for BBB_Martins,
Bioavailability_Ma, ClinTox, and Skin_Reaction.

- `manifest.json`: task roots, target definitions, construction entrypoints,
  split contracts, and split sizes.
- `migration_receipt.json`: exact source-path and row/hash equivalence audit.
- `<Task>/{scaffold,random}/{train,valid,test}.jsonl`: evaluation inputs.
- `<Task>/{scaffold,random}/*_molecule_condition_labels.jsonl`: detailed provenance.
- `random_split_receipt.json`: exact row-equivalence, seed, coverage,
  held-out vote-support objectives, and parent/scaffold-overlap audit for the
  parent-grouped quality-stratified random split.

The active path intentionally has no task-specific generation suffix. Internal
contract evolution is recorded in the manifest, not by accumulating competing
evaluation directories.

The complete construction contract is maintained in
`tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md`. In particular,
random allocation keeps whole parents together, enforces exact row counts and
three-way condition coverage, and for voter-based tasks optimizes held-out
multi-vote support before label or condition balance. Minimal train/valid/test
files are not sufficient provenance on their own.
