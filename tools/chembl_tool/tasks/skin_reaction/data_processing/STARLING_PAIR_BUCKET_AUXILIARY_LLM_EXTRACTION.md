# Skin_Reaction embedding-bucket auxiliary reconciliation

The executable builder is `build_embedding_bucket_mapping.py`; its data-driven prompt registry is
`auxiliary_value_prompts.json`, and `auxiliary_mapping_helpers/reconciliation.py` owns the final
mapping schema and validation. Distinct values are embedded locally with MiniLM, grouped into
neighbourhoods of roughly 100, and sent to the LLM one cluster per request. Every open-vocabulary
prompt requires the model to consider the neighbourhood together, reuse exactly the same label for
equivalent values, and choose the smallest scientifically defensible bucket set. This is neither one
call nor one independent extraction per distinct value.

The earlier `starling_skin_embedding_bucket_mapping.v2` prompt lineage was rejected before publication:
although its transport batched roughly 100 values, its instructions described independent per-value
extractions and produced fragmented labels such as human occupations in the species field. Version v3
rebuilds every LLM cluster and retains the rejected cache and ledger only as a spend/provenance audit.

The final `globally_reconciled_auxiliary_value_mapping.json` contains per-source output sections used
by `starling_auxiliary_metadata.py`. Context and species stratify pair buckets; sensitization also
uses a reconciled endpoint concept, and direct skin reactions may use a reconciled explicit severity
grade when no locally parseable grade exists.

## Scope and planned calls

Null-like values never reach the model. Ordinary fields retain the historical target of roughly 100
values per neighborhood. Phototoxicity `assay_method` targets 250 and additionally hard-caps every
request at 250 labels; uneven MiniBatchKMeans groups are split deterministically when necessary. Its
listed call count is therefore a lower bound. The reviewed `skin_exposure.study_design` table costs no
calls.

| source | input | output | distinct | calls |
|---|---|---|---:|---:|
| `direct_skin_reaction` | `assay_or_test` | `global_context` | 7,173 | 72 |
| `direct_skin_reaction` | `species_or_population` | `global_species_context` | 11,030 | 111 |
| `direct_skin_reaction` | `effect_metric` | `global_severity_grade` | 10,441 | 105 |
| `sensitization_aop` | `assay_type` | `global_context` | 6,291 | 63 |
| `sensitization_aop` | `assay_type` | `global_species_context` | 6,291 | 63 |
| `sensitization_aop` | `endpoint_or_target` | `global_endpoint_context` | 7,349 | 74 |
| `phototoxicity_irritation_local_damage` | `assay_method` | `global_context` | 157,737 | 937 actual (631 primary) |
| `phototoxicity_irritation_local_damage` | `evidence_system` | `global_species_context` | 87 | 1 |
| `skin_exposure` | `study_design` | `global_context` | 78 | 0 |
| `skin_exposure` | `skin_source` | `global_species_context` | 22,240 | 223 |
| **total** |  |  |  | **1,649 actual** |

The recommended execution order is the four non-phototoxicity sources first (711 calls), then the
phototoxicity source (938 calls on the frozen inventory). Exact Lloyd KMeans is used for the ordinary inventories;
MiniBatchKMeans is used only for the 157,737-value phototoxicity assay inventory.

## Semantic boundaries

- Open-vocabulary labels are reconciled across each embedding neighbourhood. Equivalent values in the
  same neighbourhood must share one label; independent cluster-level ontologies are aligned further by
  the reviewed alias pass in `reconciliation.py`.
- Species labels require explicit support in the value. A dedicated tissue-origin field may name the
  species directly; assay knowledge alone is not enough. Human clinical, demographic, occupational,
  and nationality descriptions normalize to the base species `human`, never separate population labels.
- Sensitization endpoint aliases merge wording variants such as cysteine depletion and cysteine
  peptide depletion. Cysteine, lysine, and GSH remain distinct, as do CD86, CD54, and IL-8.
- Severity is deliberately narrow: explicit 0–4, plus ladders, or explicit mild/moderate/strong/severe
  wording may map to a grade. A generic positive result, percentage, count, range, or reaction without
  an ordinal grade maps to null.
- `study_design_reviewed_mapping.json` remains the reviewed source of truth for the 78 study-design
  values; its uncovered inputs fail the build.

## Token budget, cache, and resume

The default budget is 500,000 provider-reported total tokens **per invocation**. Every response,
including validation retries, is added under a lock to a lifetime JSON ledger grouped by
source/output. Once the invocation budget is reached, no queued worker starts a new request; already
in-flight requests finish and their valid mappings are cached. The command exits with status 75.

Each extraction has a JSONL cache keyed by prompt, model, reasoning effort, and exact cluster members.
Completed source snapshots and cluster caches survive a budget stop or crash. Re-running the same
command serves completed clusters from cache at zero new cost. Caches are deleted only after all four
sources validate and the final mapping is atomically published. If a provider response omits token
usage, the valid response is still cached, but the run stops with status 76 because the budget can no
longer be audited. A ledger also records and validates its prompt version, so a rejected prompt lineage
cannot silently resume into a replacement run.

Non-retryable provider failures such as an invalid key or exhausted credit stop the shared worker queue
and exit with status 77. At most the already in-flight worker set can reach the provider; queued clusters
do not each repeat the rejected request.

The API key is read only from the environment variable named by `--api-key-env`. The approved provider
is the standard OpenAI client configured by `/data1/joseph/therapeutic-tuning/distillation/keys.py`,
injected into the child environment as `OPENAI_API_KEY`. Its value is never written to commands, code,
caches, ledgers, manifests, or traces.
The current v3 defaults are model `gpt-5.4-mini` and reasoning effort `low`. The already-complete direct
and sensitization source snapshots retain their audited `gpt-5.4` identity; final publication accepts
them only when `--compatible-snapshot-model gpt-5.4` is explicit. Changing a model still changes every
new cache and source-snapshot identity.

## Commands

Use the project environment for every command:

```bash
# Free inventory check. This does not initialize the API client or embedding model.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.skin_reaction.data_processing.build_embedding_bucket_mapping \
  --dry-run

# Core phase: 711 planned calls.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.skin_reaction.data_processing.build_embedding_bucket_mapping \
  --sources direct_skin_reaction sensitization_aop skin_exposure \
  --api-key-env OPENAI_API_KEY --token-budget 500000

# Phototoxicity phase. Repeat after any status-75 stop with a refreshed key.
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.skin_reaction.data_processing.build_embedding_bucket_mapping \
  --sources phototoxicity_irritation_local_damage \
  --api-key-env OPENAI_API_KEY --token-budget 500000 \
  --compatible-snapshot-model gpt-5.4
```

Once all source snapshots exist, either phase command finalizes the same output. The result has the
shape consumed by the auxiliary attacher:

```json
{"mapping_version": "...", "sources": {"<source>": {"<output_name>": {
  "source_columns": ["..."], "mapping": {"<json-tuple-key>": "<value>"}}}}}
```
