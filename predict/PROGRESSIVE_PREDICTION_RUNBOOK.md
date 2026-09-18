# Progressive prediction runbook

> Historical runbook retained for reference. Do not use its version, model, or
> launch settings in place of the current contract in the repository `AGENTS.md`.

Use this guide to run any compatible progressive prompt, retrieval mode, or
execution mode against the immutable V3 retrieval cache. Run commands from the
authoritative checkout at
`/vast/projects/myatskar/design-documents/joseph/TxAgent`.

## Infrastructure

- Candidate endpoints: `predict/api_client/providers/current_endpoints.json`
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Capacity: selected at launch from healthy exact-model candidates
- Cache: `predict/retrieval/assay_reranking/ranked_evidence_retrieval_parent_v1.yaml`
- Traces: `outputs/paper/live/`
- Results: `outputs/paper/assay_transfer_harness/joseph/<study>/<method>/<run>/`

The candidate inventory contains transport settings only. It does not select the
task, prompt, retrieval mode, query-prior condition, or live/throughput mode.
The launcher probes candidates concurrently, excludes unavailable or wrong-model
entries, and proceeds whenever at least one compatible endpoint remains. It
records requested and effective parallelism plus all probe outcomes. Do not run
a second matrix against the same selected capacity while one is active.

## Choose the experiment

Set these values for each batch. A batch may compare methods and K/M values;
pool, record cap, and query-prior mode stay singular so the compact run leaf is
unambiguous.

```bash
TXAGENT_STUDY=contrastive
TXAGENT_BATCH=contrastive-$(date +%Y%m%d-%H%M)
TXAGENT_PROMPT=replace_with_prompt_version
TXAGENT_HARNESS=reranked-progressive-v2
TXAGENT_TASKS=(bbb_martins bioavailability_ma)
TXAGENT_MODES=(assay-transfer morgan)
TXAGENT_PRIORS=(cached)
TXAGENT_POOLS=(assay-transfer-trained)
TXAGENT_RECORD_CAPS=(50)
TXAGENT_L1_COUNTS=(10)
TXAGENT_L1_MIN_CONTRASTS=(0 1 2 3)
TXAGENT_MAX_LEVEL=1
TXAGENT_EXECUTION=throughput
TXAGENT_REVIEW_ARGS=(--publish-review-traces)
```

Choose mode and harness together:

| Desired retrieval | Harness | Mode value |
|---|---|---|
| Native assay-transfer and/or Morgan | `reranked-progressive-v2` | `assay-transfer`, `morgan`, or both |
| Symmetric top-5 + top-5 joint | `reranked-progressive-v2` | `joint` |
| Asymmetric top-3 assay + top-7 Morgan joint | `reranked-progressive-v4` | `joint` |

V4 accepts only joint mode with `TXAGENT_L1_COUNTS=(10)`. A prompt bundle may
also require a specific maximum level or query-prior setting; the runner checks
those constraints and fails closed. For a no-prior condition, the matrix
resolves the selected prompt plus its composable `_no_query_prior` modifier.

### Structured private-reasoning comparison

`reranked_progressive_l1_simple_v14` is the free-reasoning control and
`reranked_progressive_l1_simple_v15` is its grammar-enforced variant. Both use
the same SGLang tokenized-completion transport, render identical system and
user messages, support BBB and Bioavailability, and require
`TXAGENT_MAX_LEVEL=1`. V15 alone applies a query-specific EBNF grammar that
requires one bounded paragraph per visible molecule, names all records for the
molecule in its heading, and ends with one global-synthesis paragraph before
the normal JSON answer. It does not require a separate sentence or slot for
every record.

Run v14 and v15 as separate batches with otherwise identical variables and
different batch IDs. The launcher automatically fails closed unless every
configured endpoint advertises the exact model, SGLang is ready with the
`xgrammar` backend and `deepseek-v4` reasoning parser, no completion template is
installed, and tokenizing the chat messages enters `<think>`. Each level's
`request.json` records the rendered grammar and its SHA-256; v14 records a null
grammar. Shared response keys include the rendered grammar, so responses cannot
cross the control/treatment boundary.

For live review, use:

```bash
TXAGENT_EXECUTION=live
TXAGENT_REVIEW_ARGS=()
```

Live mode publishes three samples and pauses for review. Throughput mode runs
without that gate; adding `--publish-review-traces` publishes three samples
while the full run continues.

## Launch

The command below consumes only the choices above:

```bash
"$TXAGENT_PYTHON" -m predict.harnesses.progressive.matrix \
  --study "$TXAGENT_STUDY" \
  --batch-id "$TXAGENT_BATCH" \
  --tasks "${TXAGENT_TASKS[@]}" \
  --reranking-modes "${TXAGENT_MODES[@]}" \
  --harness-version "$TXAGENT_HARNESS" \
  --record-pools "${TXAGENT_POOLS[@]}" \
  --records-per-level "${TXAGENT_RECORD_CAPS[@]}" \
  --l1-molecules "${TXAGENT_L1_COUNTS[@]}" \
  --l1-min-contrasts "${TXAGENT_L1_MIN_CONTRASTS[@]}" \
  --parallelism 768 \
  --provider-pool-config predict/api_client/providers/current_endpoints.json \
  --trace-root outputs/paper/live \
  --execution-mode "$TXAGENT_EXECUTION" \
  "${TXAGENT_REVIEW_ARGS[@]}" \
  --prompt-version "$TXAGENT_PROMPT" \
  --query-prior-modes "${TXAGENT_PRIORS[@]}" \
  --prior-root outputs/paper/legacy/starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/runs_deployment_visible_parent_disjoint \
  --max-level "$TXAGENT_MAX_LEVEL" \
  --max-tokens 131072 \
  --request-timeout-s 3600 \
  --assay-transfer-cache predict/retrieval/assay_reranking/ranked_evidence_retrieval_parent_v1.yaml
```

Modes, prompt versions, query-prior modes, retrieval widths, and K/M values in
one invocation share one rolling queue and become separate sibling leaves linked
by the batch ID. If a requested comparison axis is absent from the CLI, extend
the matrix before launching instead of using shell loops or separate launchers.
Non-contrastive modes are recorded as M=0 because they make no minimum
label-balance guarantee.

## Cache contract

The V3 cache is read-only during prompt preparation. A run must not rebuild
source mappings, fingerprints, rankings, scaffold checks, or database integrity
scans. The cache YAML supplies locations only; the selected harness continues
to own stage and ranking rules. Joint panels are reconstructed from the native
rankings and are never stored as another cache.

## Operate and verify

```bash
"$TXAGENT_PYTHON" -m predict.live list
"$TXAGENT_PYTHON" -m predict.live show RUN_ID
"$TXAGENT_PYTHON" -m predict.live continue RUN_ID
"$TXAGENT_PYTHON" -m predict.live cancel RUN_ID
```

Use the same command and batch ID to resume an interrupted matrix; immutable
settings are checked before reuse. Do not infer completion from process exit or
raw output count alone. A complete matrix has `status.json` with zero failed
chains plus one complete `diagnostics_manifest.json` per run leaf. Each leaf
also has performance tables, the visible-evidence index, L1 label-mix tables,
reasoning-reference coverage tables, and `report.md`. Prompt, provider, cache,
and execution receipts are recorded under `_batches/<batch-id>/` and in each
leaf's `run.json` and `matrix_execution.json`.

If prompt text changes, create a new prompt version and output root. Never
compare or resume by version label alone: verify the prompt hashes recorded in
the run manifest.
