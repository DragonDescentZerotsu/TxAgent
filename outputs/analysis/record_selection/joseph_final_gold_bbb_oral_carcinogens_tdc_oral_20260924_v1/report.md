# Canonical Joseph full-test results and traces

The [final collection](../../../paper/assay_transfer_harness/joseph/final/collection.json) contains seven complete arms and 2,390 successful test queries. Each arm has predictions, metrics, per-query runs and traces, a combined trace, and hash-pinned provenance. The numerical results are in [results.tsv](results.tsv).

Gold results include 46 locally recovered queries served as `deepseek-v4-flash-0731`; the other Gold queries were served as `deepseek-ai/DeepSeek-V4-Flash-0731`. These are complete mixed-route results, not exact-model-only replicates. The per-query route and source run are in each arm's `query_provenance.tsv`.

TDC Oral direct+indirect uses 87 queries from the recovery leaf and 41 valid predecessor runs. Its aggregate metrics were recomputed from the 128-query union. Gold Carcinogens direct uses the agreed complete `ga075_mc000_label000` profile.

The source batches remain unchanged. The final collection was verified after installation against each copied file hash and each prediction-to-trace link.
