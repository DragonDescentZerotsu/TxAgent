# Semantic buckets

This is the sole owner, imported as `optimization.semantic_buckets`. Maintained code
follows the scientific operations; experiment variations are Bash recipes.

| File | Responsibility |
| --- | --- |
| `inputs.py` | Selected releases, physical membership and task-native intrinsic atoms. |
| `bucket_cards.py` | Representative scientific cards, task targets and Jinja rendering. |
| `bucket_construction.py` | Source-local split/merge algorithms and decision validation. |
| `bucket_refinement.py` | Fresh family audits and partitioned successor proposals. |
| `bucket_scoring.py` | Initial batched scoring, sorted sliding rescoring and replica aggregation. |
| `build.py` | Construction preparation, execution, review and establishment. |
| `execution.py` | Shared completion execution, concurrency, retries and replica collection. |
| `storage.py` | Queues, checkpoints, immutable bindings and receipts. |
| `ledger.py` | Physical UID census, reviewed membership means and pending worklists. |
| `publication.py` | Reviewed materialization, verification and explicit selection. |

`prompts/` remains a directory of Jinja templates, grouped as `bucket_construction/`,
`bucket_refinement/` and `bucket_scoring/`. Task-specific clinical rubrics are optional
scoring inputs; fixed-neighbor calibration and per-level selection are retired.

## Construct and refine

Construction takes pinned V10 evidence, a retrieval index and a fresh run root.
Task-native policies preserve source/level boundaries and exclude unit/species/
population fields from semantic split choices. Approval binds prompts, code and inputs.

```bash
TASK=dili RELEASE_ROOT=/path/evidence RETRIEVAL_INDEX=/path/index.json \
  RUN_ROOT=/path/fresh-construction bash experiments/semantic_buckets/construct.sh
# STAGE=run requires --approved-review-sha256 HASH.
# STAGE=establish requires --review reviewed-decisions.json.
```

`refine.sh` supports `prepare`, `run-pass1`, `prepare-pass2`, `run-pass2`, `collect`.
It emits fresh audits/proposals for explicit review. Collection does not select maps.
`publication materialize` applies reviewed atom-level merges/splits to physical records;
inherited scores are seeds, not new reviewed scores.

## Score

A declared world contains `level, semantic_bucket_id` in the retrieval universe.
`bucket_cards.py` prepares frozen world/card descriptors from reviewed construction
outputs using the preserved deterministic representative sampling. Its optional
`--enriched-records` retains the richer Ames/DILI/Carcinogens representative contract.
A descriptor binds task `target`, `world`, `payloads`, optional `levels`, `source_labels`
and an optional `rubric` Jinja path.

```bash
python -m optimization.semantic_buckets.bucket_cards --task dili \
  --world world.parquet --semantic-root /path/reviewed/semantic_run --output /path/fresh-inputs
INPUT_MANIFEST=/path/fresh-inputs/inputs.json RUN_ROOT=/path/fresh-score PASSES=3 \
  bash experiments/scoring/sliding_auto.sh
python -m optimization.semantic_buckets.bucket_scoring run --root /path/fresh-score
python -m optimization.semantic_buckets.bucket_scoring collect --root /path/fresh-score
```

`--passes N` counts the initial pass; default 2. Pass 1 scores independent batches.
Every later pass sorts the preceding complete pass by score, breaking ties by bucket
ID, and executes sliding windows. Each candidate displays its immediately preceding
pass weight as revisable context; pass 1 has no previous scores. Newly scored tail
anchors from completed windows in the current pass are locked for subsequent windows.
Later completions cannot change an anchor decision. Every scheduled candidate finishes before the next pass.
Fixed neighboring anchors are not supported.

Manual/automatic chains, overlap, advancement quorum, batch sizes, replica fanout/
averaging and provider configurations remain explicit controls. Automatic chaining
balances all requested task-levels against one longest-chain reference. Later passes
reuse the declared pass-2 execution settings. `--through-pass` permits stage inspection;
resume retains completed work and rejects changed sources, templates, code or routes.
`--prompt-dir`, `--max-tokens` and `--reasoning-effort` parameterize scientific recipes;
medium reasoning retains the approved Luna contract. Both local and OpenRouter use
shared transport/pools. Receipts record actual served models/providers and late replicas.
Collection exports final-pass weights as an unselected candidate.

For a selected-release refresh, use `bucket_cards --task TASK --release CURRENT --output DIR`.
It reconstructs canonical atoms and representatives without regrouping reviewed buckets.
`bucket_scoring prepare --seed-weights FILE --passes 2` freezes published weights as
stage 1 and makes model calls only for the new sliding stage. Seeds set ordering
and appear as revisable previous-pass candidate weights in the first sliding pass;
anchors always come from completed windows in the current pass. Resume validates
the seed, cards, input descriptor, code, prompts and provider configuration.

```bash
export RUN_ROOT=/vast/path/to/fresh-oral-semantic-run
STAGE=prepare bash experiments/scoring/selected_refresh.sh
STAGE=run bash experiments/scoring/selected_refresh.sh
STAGE=publish bash experiments/scoring/selected_refresh.sh
```

The recipe defaults to Oral and Luna medium, averaging two replicas per window and
at most four automatic chains per task-level. Set `FANOUT`, `AVERAGE` and `MAX_CHAINS`
to override those controls; fewer chains allow longer sequential calibration.
For high reasoning, prepare with `REASONING_EFFORT=high` and
`PROVIDER_CONFIG=predict/api_client/providers/gpt6_luna_openrouter_standard_high_64.json`.
`publication refresh-weights` verifies complete
bucket and indirect UID coverage before selecting the new generation. New releases
declare `semantic_weight_resolution: reviewed_membership_mean.v1`: the shared reader
joins the complete UID ledger with bucket weights in memory and preserves the mean
over distinct reviewed memberships. No per-retrieval weight projection or extension
is needed. L1 stays direct and receives no semantic weights. Old projection-backed
releases remain readable; frozen maps, payloads and evidence CURRENT are preserved.

## Selected artifacts

Evidence CURRENT chooses a named evidence release; its semantic manifest selects maps,
weights and ledgers. Previously published scientific payloads remain immutable.
`assets/`, maintained retrieval policies, pending worklists and required older published
selection inputs remain. `cleanup/` records the earlier intermediate cleanup and this
refactor's verification. Publish a reviewed successor explicitly using `publication build`,
`verify` and `select`; never overwrite frozen payloads or evidence CURRENT.
