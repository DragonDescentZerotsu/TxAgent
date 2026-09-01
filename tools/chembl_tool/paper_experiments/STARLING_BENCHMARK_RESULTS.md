# Historical Starling benchmark index

This former experiment-by-experiment ledger has been retired from the active
documentation surface. It had grown to mix obsolete dataset versions, prompt
probes, router/RL no-go studies, timeout retries, and current results in one
file, which made lineage mistakes likely.

Use the following sources instead:

- current benchmark creation and split constraints:
  `tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md`;
- current paper result status and last-complete retained references:
  `tools/chembl_tool/paper_experiments/RESULTS.md`;
- machine-readable artifact roots, hashes, and freshness:
  `tools/chembl_tool/paper_experiments/current_conditioned_results.json`;
- one-shot family and append-only progressive protocols:
  `tools/chembl_tool/paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`;
- retained artifact cleanup boundary:
  `tools/chembl_tool/paper_experiments/TRACE_RETENTION.md`.

The detailed historical ledger remains available in Git history. Its old
numbers must not be copied into a current table without reconstructing their
benchmark, split, source, model, visibility, prompt, and retrieval-index
contracts from the original manifests.

The historical experiment families still relevant to the paper are represented
by exactly one last-complete reference in `current_conditioned_results.json`.
All other historical sections were method-development records, not active
pipeline contracts.
