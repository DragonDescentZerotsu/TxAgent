# Conditioned Benchmark data

This is the active benchmark root for BBB_Martins, Bioavailability_Ma, ClinTox,
Skin_Reaction, Ames, DILI, and Carcinogens.

DILI uses Starling-only gold_v4: 4,024 molecule-condition rows, with
train/valid/test 3,220/402/402 in scaffold and random. Carcinogens uses the same
source-only contract: 4,692 rows (3,578 positive / 1,114 negative), with
3,754/469/469 in both schemes and five organism conditions: rodent, human, dog,
monkey and rabbit. Neither benchmark includes TDC labels.

The final retrieval sources are frozen in `data/starling_data/<task>/retrieval_final/`.
Their source consolidation preserves these gold labels and splits; publication and
restoration receipts are in `data/starling_data/retrieval_final_cleanup.json`.
Large detailed-label and vote JSONL files restore byte-for-byte through
`rebuild_current_starling_retrieval restore-records --tasks dili carcinogens`.
Completed model, baseline and diagnostic results are indexed in
`tools/chembl_tool/paper_experiments/current_conditioned_results.json`.

These two tasks use the explicit `new_task_tautomer_identity.v2` leakage boundary;
gold retains stereochemical identity. DILI scaffold valid/test names were exchanged
on 2026-09-08, and both tasks' evaluation sets have been inspected. Full identity
provenance remains in `data/starling_data/new_tasks_gold_audit/TAUTOMER_IDENTITY_REPAIR.md`.

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
