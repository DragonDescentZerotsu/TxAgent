# Full-depth validation matrix

> Historical runbook retained for reference. Do not launch this matrix unless
> its versions and execution settings match the current repository contract.

For the active rolling launch, use `--rolling-from <prepared-matrix-root>`,
`--execution-mode throughput`, and a separate `--output-root`. Rolling is an
explicit uninterrupted resume and therefore skips the pilot while keeping its
live-tree traces private until promotion. A 128-descriptor queue
feeds one global scheduler, with L1 priority and immediate dependent
stage admission. Saved task evidence is read in 16-query batches; tasks absent
from the saved matrix are prepared in 16-query batches while inference runs.
No full-matrix preparation barrier is used on this path. Both output roots are
locked to exclude duplicate launchers. Progress and exact response/attempt caches
remain under the original matrix's `inference/`; `rolling.json` records the active
scheduler separately from the original frozen request-key namespace.

Successful responses are never resent for an identical rendered request and
generation contract. Concurrent duplicates wait for one shared request. Wire
failures and invalid outputs retain bounded retry histories; ambiguous interrupted
requests cannot provide a remote exactly-once guarantee. New task batch outputs
are linked into the common condition query tree for full-cohort summaries.

The matrix uses `runner.py` to prepare evidence, `prompt.py` to render each stage,
and `inference.query_steps` to yield a validated model-call job and consume its response;
the ordinary runner drives the same iterator synchronously. The matrix shares
exact requests, persists validated responses, and resumes each consumer with its
own citation mapping. No prompt assets are modified.

Default sweep: BBB L1–L5 and oral L1–L6, Morgan and assay transfer, pools `all` and
`assay-transfer-trained`, and 10/25/50 new records at each later level. L1 remains
10 molecules × up to 10 records each. Cached query priors and tool summaries stay
visible. L5 remains full-library Morgan in both modes and both pools.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:/tmp/txagent_prompt_validation_deps \
python -u -m predict.harnesses.progressive.matrix \
  --study baseline \
  --batch-id baseline-v8-20260914 \
  --prompt-version reranked_progressive_l1_simple_v12_references_v1 \
  --max-level 1 \
  --record-pools assay-transfer-trained \
  --records-per-level 50
```

`--study` is the normal output interface. A condition is written to a compact
leaf such as `baseline/morgan/k10_m0_20260914_1200/`; the leaf `run.json`
contains prompt, provider, prior, task, cache, and shape provenance. Different
methods and M values never share a leaf. They share validated responses through
`_batches/<batch-id>/inference/`, and `_batches/<batch-id>/runs.json` lists the
leaves. `--output-root` remains only for resuming an existing legacy matrix and
cannot be combined with `--study`.

An organized study rejects a prompt without an immutable reasoning-reference
contract. Current successors are
`reranked_progressive_l1_simple_v12_references_v1`,
`reranked_progressive_l1_context_order_only_v1_references_v1`,
`reranked_progressive_l1_context_v2_references_v1`,
`reranked_progressive_l1_context_l2_semantic_v1_references_v1`, and
`reranked_progressive_l1_context_l2_weighted_v1_references_v1`. Their inherited
prompt modifiers remain composable. The contract does not require references or
change inference: it gives rendering and diagnostics one exact identifier index.

The latest organization/diagnostics validation used
`/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/bin/python`
on epyc-1-6, not node002.
Use `--tasks bbb_martins` or `--tasks bioavailability_ma` to prepare and run one
task independently. Do not run multiple launchers against the same endpoints
at once; task selection does not create another global request budget.
Use the same command and output root to resume. Changed matrix/code/prompt/input
contracts fail closed. A root-level advisory lock prevents duplicate launchers.
Attempt budgets persist across restarts; exhausted requests require investigation,
not silently reset budgets. An unresolved pilot prevents the full inference phase.
The first three samples per dataset-method child run are published and stop at
`awaiting_review`; `--continue-after-pilot` runs the remainder.

All unique L1 jobs enter the queue before descendants. Each successful stage
immediately unlocks its consumers, without a batch barrier. Duplicate waiters use
no inference slots. The current default is the declared 834-call aggregate across
the three endpoint capacities (128, 450, and 256); the parser rejects larger values.
Model and thinking settings are fixed in the execution receipt; a dedicated
environment variable contains the nonsecret credential `EMPTY`. HTTP read timeout
is disabled, while connection, write, and connection-pool waits remain bounded.
Server-side failures are still possible. Ten actual attempts are allowed per
unique stage, across transport and schema failures, with delayed retries.

`matrix_execution.json` in each run leaf is the inference receipt; its runner
manifest records preparation configuration. `inference/progress.json` gives live
queue and provider counts, `pilot.json` records the pilot gate, and `status.json`
is written after final summaries. Canonical responses and attempt receipts remain
under `inference/responses/`; condition outputs retain source-response links.
The inference completion report counts logical chains, not independent replicas.

Final performance tables are `macro_f1.tsv`, `accuracy.tsv`, and
`completion_reuse_evidence.tsv`. Each completed leaf also contains
`visible_evidence.tsv`, both `neighborhood_label_mix.*.tsv` tables, both
`reasoning_reference_coverage.*.tsv` tables, `diagnostics_manifest.json`, and
`report.md`. A run may report 0/K explicit reasoning coverage; missing diagnostic
inputs prevent completion. Identifier matching is case-insensitive. Historical
duplicate molecule labels are reported as undefined molecule coverage while
their unambiguous card/group coverage is retained. Missing BBB L6 is `NA`;
incomplete stages remain failures, never manufactured predictions.
