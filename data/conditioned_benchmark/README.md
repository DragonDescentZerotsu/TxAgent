# Conditioned Benchmark data

This directory is the single active gold dataset for BBB_Martins,
Bioavailability_Ma, ClinTox, and Skin_Reaction.

- `manifest.json`: task roots, target definitions, and split sizes.
- `migration_receipt.json`: exact source-path and row/hash equivalence audit.
- `<Task>/{scaffold,random}/{train,valid,test}.jsonl`: evaluation inputs.
- `<Task>/{scaffold,random}/*_molecule_condition_labels.jsonl`: detailed provenance.
- `random_split_receipt.json`: exact row-equivalence, seed, coverage,
  held-out vote-support objectives, and parent/scaffold-overlap audit for the
  parent-grouped quality-stratified random split.

The active path intentionally has no task-specific generation suffix. Internal
contract evolution is recorded in the manifest, not by accumulating competing
evaluation directories.
