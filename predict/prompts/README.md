# Prediction prompts

All six tasks have `tasks/<task>/profiles.yaml`. Each contains the default and
historical native profiles, their schemas/instructions, and the canonical flat
task contract. Profile names describe scientific prompt contracts; they are
independent of benchmark storage.

`flat/final_full_flat/` contains the default system/user Jinja templates, level
descriptions and provenance. `query_prior/` contains shared native/generic prior
templates and generic prior instructions. `variants/` holds deliberate prompt
experiments, with task overrides rather than copied six-task definitions.

`__init__.py` loads these assets and renders prior messages; the flat harness
uses the same loader. Rendering, evidence processing, validation and inference
remain Python. Prompt prose and output schemas live here. Retrieval-scoring,
semantic-ranking and evidence-construction prompts stay with their capabilities.

Historical flat bundles are read explicitly from `archive/branches/prompts/`.
They retain their frozen contracts; there is no old-path symlink directory.

## Frozen paper baseline

`variants/paper_baseline_v1/` preserves the paper baseline system/user templates,
six native task/output contracts and the two TDC overrides (Ames and Carcinogens).
It retains task-specific prediction fields, confidence, supportive/contradictory
card IDs, prediction basis, bounded claims/gaps and decision summaries. These
are distinct from the current FINAL `claims` / `final_prediction` contract.
The source/asset hashes are recorded in the bundle's `provenance.json`.

`paper_baseline.py` renders frozen `prompt_payload` objects and supplies response
validation to the existing `predict.utils.response` functions. Exact field
deduplication is retained; older full-field payloads remain readable. It neither
selects evidence nor regenerates priors. Historical task definitions are explicit
and do not inherit the current default profiles.

From the repository root, verify a restored original batch without model calls:

```bash
python -m predict.harnesses.branches --verify-paper-baseline /path/to/frozen/batch
```

The command reads `experiment_manifest.json`, `runs/<task>/query_*/` and the
saved task summaries. It checks request/manifest bindings, exact re-rendering,
record aliases, raw response parsing, native-label validation, saved decisions
and metrics. It prints a report and does not modify the batch. This verifies
saved artifacts; it does not rebuild candidates or independently re-audit gold
labels, providers or costs.

Rebuild candidates from a restored paper batch and its hash-bound source inputs:

```bash
python -m predict.harnesses.branches \
  --prepare-paper-baseline /path/to/frozen/batch \
  --source-root /path/to/restored/paper/repository \
  --run-name paper-baseline-recheck --query-indices 0 10 20
```

Omit `--query-indices` to prepare the entire source cohort. Preparation makes no
model or tool-service calls. It writes a new batch under
`outputs/predictions/branches/paper_baseline_v1/<run-name>/` and reports ordered
record/request differences in `preparation_comparison.json`. Never reuse old
predictions for changed requests. The source batch and historical scores remain
unchanged. Changing a source, prior, implementation or configuration requires a
new run name; concurrent preparation/execution of one run is rejected.

The source manifest supplies the setting (None, Direct or Direct+Indirect),
128-candidate limits, record budgets, selector/seed, split and generation settings.
`predict/retrieval/paper_baseline.py` partitions L1/L2+ before shared identity
filtering and preserves the original TDC train-label merge and heldout filter.
`optimization/selection/record_budget.py` preserves Neighbor-fill, molecule-cap
and group-balanced ordering; source records are projected through the existing
minimal-evidence helper. The frozen query, condition and cached prior are retained.
The new parent rules apply during candidate filtering before ranking/budget fill;
DILI/Carcinogens retain their declared scaffold topology. No split is regenerated.

Execute a prepared batch with the shared provider pool (this calls the model):

```bash
python -m predict.harnesses.branches \
  --run-paper-baseline outputs/predictions/branches/paper_baseline_v1/paper-baseline-recheck \
  --provider-config /path/to/provider.json --parallelism 8
```

To move prepared inputs to another checkout or machine, export a fresh execution
bundle with the same CLI, without rebuilding retrieval indexes:

```bash
python -m predict.harnesses.branches \
  --bundle-paper-baseline outputs/predictions/branches/paper_baseline_v1/paper-baseline-recheck \
  --run-name paper-baseline-portable
```

Copy the entire `outputs/predictions/branches/paper_baseline_v1/paper-baseline-portable/`
directory to the destination. Execution uses its relative source snapshot and
evaluation files; it does not open the original repository. Original source hashes,
messages, priors, record order, labels and generation settings remain verified.
Export requires a new directory and includes only the selected prepared requests,
selection traces and their validation inputs. It excludes predictions, provider
configuration, credentials and logs, so its first execution makes new model calls.
To resume an already running portable batch, move that whole batch directly instead
of exporting it again. Historical path strings remain in provenance; this command
does not anonymize a bundle or make it ready for public distribution.

Prepared execution can use a small environment (verified with Python 3.12):

```bash
python -m venv .venv-paper
.venv-paper/bin/python -m pip install -r predict/requirements-paper-baseline.txt
.venv-paper/bin/python -m predict.harnesses.branches \
  --run-paper-baseline /path/to/paper-baseline-portable \
  --provider-config /path/to/provider.json --env-file /path/to/provider.env --parallelism 8
```

These dependencies cover prepared LLM execution, not retrieval rebuilding or ML
training. The latter still require their documented data and environments.

Provider model and extra request parameters must match the frozen manifest.
Temperature, reasoning effort and token budget come from that manifest. The paper
protocol explicitly retains `max_tokens` and rejects context overflow rather
than silently shrinking the budget; current FINAL defaults are unchanged.
Requests, cohorts, priors and existing responses are checked before dispatch.
The shared JSON validator supplies up to four total attempts with the original repair messages and native
output fields. Successful saved outputs are resumed without a model call;
failed outputs can be retried by rerunning the command. Summary coverage and
metrics distinguish failed queries. A transport exception aborts the command;
completed query files remain resumable.
Ordinary API retries are explicit execution controls, separate from the frozen
scientific inputs. For example, add `--parallelism 128 --retry-race-width 6
--max-stage-requeues 2 --retry-delay-s 1`. The first call uses width 1; a format
repair, failed stage or interrupted failed query can use the configured width.
The first response passing the complete native validator wins; losing HTTP
requests are cancelled and awaited. Race and requeue receipts stay with the query.
Defaults are width 1 and zero stage requeues. Successful queries are never rerun.
Races cannot use the synchronous spend ledger; that combination fails before calls.

OpenRouter Batch uses the same execution command and a provider configuration
with `max_failovers: 0`, one provider with `transport: "openrouter_batch"`, and
optional `batch_options` (`max_requests`, `max_bytes`, `flush_seconds`,
`poll_seconds`, `submit_interval_seconds`). The provider must use
`https://openrouter.ai/api/v1` and the exact frozen model ID, without `:batch`.
Generation settings still come from the frozen manifest. Batch requires race
width 1, no automatic POST retries and no synchronous spend
ledger. Unsupported transports and mixed/failover Batch pools are rejected.
Stage requeues remain opt-in and bounded. If all four native-format attempts were
invalid, a restart can first recover those same replies; an explicit
`--max-stage-requeues 1` allows another stage after that replay. An uncertain
submission still stops the journal; requeues never override that protection.

Batch requests are queued before waiting for results. `batch_transport/` holds
the durable journal: completed responses and in-flight jobs recover without
resubmission, and confirmed terminal failures can be retried while keeping
successful siblings. An uncertain POST remains `submitting` and blocks restart
until reconciled; never delete that journal to force a retry. Stopping the local
client does not cancel paid remote jobs. Move the entire run, including its
journal, to resume elsewhere; exporting a fresh input bundle intentionally does
not copy jobs or predictions. The migrated Batch protocol and retry paths have
offline acceptance coverage, not a new paid endpoint-availability check.

Matched Direct-3 and standalone prior generation are not yet integrated into this mode. It does not
replace the default FINAL scheduler or its named-cache protocol. Offline transport
and recovery tests are not a fresh model-performance replication. Candidate
rebuilding still requires the original full retrieval inputs; the portable bundle
only makes already prepared execution self-contained. Full data distribution,
anonymization and full-cohort validation remain release gates.
