# Semantic buckets

This directory is the single owner of the BBB and Oral semantic-bucket lifecycle:
generation code, prompts, policies, selected releases, review provenance, retired
generations, audits, and tests.

## Active release

`artifacts.py` resolves the selected V10 data from
`releases/<task>/v10/manifest.json`. Both tasks select
`gold_v1_protected_incremental_v1`. Retrieval eligibility is the separately
versioned `gold_v1_protected_incremental_v1_weighted_v2` sidecar.

The V2 policy is `policies/semantic_weighted_top10_v2.json`; V1 remains frozen
for historical reproduction.

## Lifecycle

- `prompts/`: pairwise ranking, grouping, readout, and eligibility templates.
- `releases/`: active manifest-selected BBB and Oral artifacts.
- `provenance/`: complete incremental request schedules, prompts, responses, and ledgers.
- `history/`: physically relocated pre-Gold-v2 semantic generations.
- `policies/`: immutable expert-weight policies and pinned ranking hashes.
- `audits/`: machine-readable TSVs and joined scientific reports.
- `tests/`: resolver, schema, identity, provenance, and policy tests.

Pairwise ranking and bucket eligibility operate on compact bucket identities and
representative canonical profiles. They do not see the query molecule or full
downstream inference records. Expert weights were assigned manually; there is no
separate weight-assignment LLM prompt.

Run the focused checks with:

```bash
pytest -q semantic_buckets/tests
python -m semantic_buckets.audit_policy_v2
```

## Sliding weight assignment

`sliding_weight_assignment.py` prepares the reviewed 12-candidate bootstrap and
anchored seven-candidate sliding schedule for every ranked BBB and Oral bucket.
It renders readable canonical profiles and representative records without
showing bucket IDs, ranks, or counts to the model. Prepared runs and resumable
request ledgers live directly under
`provenance/semantic_weight_sliding_v5/<run-id>/` on `/vast`. V2 added the fixed
seven-band utility rubric. V5 keeps that prompt bundle and uses OpenRouter
`deepseek/deepseek-v4-pro` for each task-level seed call, then local
`deepseek/deepseek-v4-flash-0731` for anchored calls. The active V5 run began on
local Flash services and resumed through OpenRouter; endpoint migrations are
recorded separately from scientific identity. New anchored requests use the
explicit mixed profile from `data.processing.openrouter_provider_pool`, which
admits only gold-qualified 0731/V4.1 Flash model-provider routes, refreshes live
price and p50 throughput before dispatch when its snapshot is one hour old, and
records the exact snapshot hash with each request. The reusable pool remains
0731-only unless a consumer explicitly enables mixed Flash models.
V1 through V3 are retained as never-executed predecessors; the V4 seed attempt
is retained as a failed validation ledger.

Preparation makes no model requests:

```bash
python -m semantic_buckets.sliding_weight_assignment prepare --run-id <run-id>
```

After reviewing the rendered prompts, execute with the exact
`prompt_review_manifest_sha256` printed by preparation:

```bash
python -m semantic_buckets.sliding_weight_assignment run \
  --run-id <run-id> \
  --approved-review-sha256 <sha256>
```

`publish-candidate` validates complete coverage and writes a compact candidate
table and report. It does not change the active V2 policy or retrieval releases.

## V6 calibration

`calibrate_weights_v6.py` waits for a hash-published V5 candidate, sorts each
task-level by its V5 weights, and directly recalibrates the larger of 32 buckets
or half of that task-level. The remaining rows retain their V5 score with an
explicit carry-forward status. V5 weights, rationales, and request provenance
remain separate columns in every V6 artifact for later auditing.
After the 12-bucket Pro seed, V6 advances two chains of seven candidates per
round and carries the last two outputs from each chain forward as four anchors.

Preparation is inference-free and requires the immutable V5 run identity:

```bash
python -m semantic_buckets.calibrate_weights_v6 prepare \
  --run-id <v6-run-id> --v5-run-id <complete-v5-run-id>
```

Completed task-levels may be calibrated while the remaining V5 chains continue.
Pass one `--task-level task:L#` option per completed level. Preparation verifies
complete V5 coverage and freezes those rows plus the V5 manifest inside the V6
shard, so later V5 checkpoints cannot change its inputs. The shard's request rows
are reusable when the complete V6 run is assembled.

After prompt review, use `run --approved-review-sha256 <sha256>` and then
`publish-candidate`. Publication writes the full candidate, change table,
task-level rank/change summary, largest revisions, and detailed plus summarized
comparisons with the 90 manual policy weights; it does not activate the candidate.
