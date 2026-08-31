# Reviewed condition labels: provenance note

The active evaluation contract and paths are documented in
[`CONDITIONED_BENCHMARK.md`](CONDITIONED_BENCHMARK.md).

External-condition rows were admitted only after task-specific semantic review,
parent-condition voting, and scaffold-safe split allocation. The final accepted
rows and their review ledgers now live beside each task under:

```text
data/conditioned_benchmark/<Task>/scaffold/
```

Important distinctions:

- `train.jsonl`, `valid.jsonl`, and `test.jsonl` are the active benchmark.
- `*_molecule_condition_labels.jsonl` contains label and source provenance.
- `source_condition_review.jsonl` contains accepted/rejected source review.
- `migration_receipt.json` maps former internal lineage names to the single
  active suite and proves whether evaluated split rows are identical.

Do not use the former `processed_starling_context_conditioned_selected_vN`
directories as runner inputs. They are migration sources only.
