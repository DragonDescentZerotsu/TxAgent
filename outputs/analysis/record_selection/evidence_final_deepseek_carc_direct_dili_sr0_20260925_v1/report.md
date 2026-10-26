# DeepSeek FINAL additions: Carcinogens direct and Gold DILI sr=0

Two complete test arms were added without replacing existing FINAL entries. The DeepSeek V4 TDC-v2 Carcinogens upstream-v3 direct panel scored **0.817392 macro-F1** on 56 queries. The DeepSeek V4 Gold-v1 DILI `sr=0` control scored **0.680085 macro-F1** on 402 queries. All predictions and per-query traces are present and hash-verified. `results.tsv` contains the numeric table; `provenance.json` pins the final artifacts and optimization snapshots.

The Carcinogens direct profile was chosen on validation. The DILI zero-informativeness control was evaluated after viewing test ablations and is labeled exploratory; the validation-selected DILI arm and `canonical_dili` remain unchanged. The Gold DILI exact-count macro-F1 is 0.6800854037 (reported as 0.680085); the historical harness rounds each class F1 first and reports 0.680086 for these same predictions. The further Gold DILI diversity ablation at 0.693906 is excluded.

The previously finalized DeepSeek V4 stronger-direct Carcinogens mixed arm remains at 0.790611; the TDC-v2 DILI mixed arm remains at 0.821346. Existing Luna Carcinogens and DILI arms were verified and retained.
