# Conditioned Benchmark data

This directory is the single active gold dataset for BBB_Martins,
Bioavailability_Ma, ClinTox, and Skin_Reaction.

- `manifest.json`: task roots, target definitions, and split sizes.
- `migration_receipt.json`: exact source-path and row/hash equivalence audit.
- `<Task>/scaffold/{train,valid,test}.jsonl`: evaluation inputs.
- `<Task>/scaffold/*_molecule_condition_labels.jsonl`: detailed provenance.

The active path intentionally has no task-specific generation suffix. Internal
contract evolution is recorded in the manifest, not by accumulating competing
evaluation directories.
