# Full-test upstream-V2 recovery snapshot

The [all-profile table](all_profile_metrics.tsv) joins successful predictions by query index from original batches and disjoint recovery slices. [Best by task](best_by_task.tsv) ranks profiles by macro-F1; [the compact view](best_macro_f1_by_task.tsv) includes coverage counts. A profile is complete only when every test query has a successful result.

Local-flash recoveries use a different served model ID and omit server-enforced JSON format; their merged scores are operational and provisional, not original-model-only replicates. The active Carcinogens recovery may still be incomplete at this snapshot.
