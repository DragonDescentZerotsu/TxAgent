# In-distribution assay-transfer: design & investigation

Status: implemented; validation cache generation uses the runbook below.
Scope: Bioavailability_Ma assay-transfer reranking + how records are presented to the LLM.

## Problem

The assay-transfer reranker currently **reconstructs** its scoring records from TxAgent's
evidence library, in `reranking/build_assay_transfer_rerank_catalog.py::_index_example_record`:

- `canonical_endpoint_key` is synthesized as `index.{concept}.{slug(endpoint)}.{unit}`
  (e.g. `index.Fg.efflux_or_secretory_transport.not_reported`).
- the scalar is regex-scraped from prose via `_first_number`.
- units frequently default to `not_reported`.

The assay-transfer model (`jiosephlee/assay-transfer-tool`) was trained in
`/data1/joseph/starling_assay_transfer` on **normalized** records with real canonical keys,
normalized values/units, and typed categoricals. So the reconstructed prompts are **out of
distribution**, and the transfer scores that drive reranking (and the `assay_transfer_tool`
group prompt) are unreliable — worst for qualitative Fg/Fh endpoints.

### Concrete mismatch

The model reads an `endpoint:` line and a `known value:` line. Authoritative vs TxAgent:

| axis | authoritative (trained on) | TxAgent (reconstructed) |
|---|---|---|
| endpoint prefix | `q2/q3/q4` source collection | `index.Fa/Fg/Fh/oral_*` |
| endpoint subtype | canonicalized (`transporter_substrate_status`, `efflux_ratio`, `papp`) | raw slug (`efflux_or_secretory_transport`) |
| unit | normalized (`cm_per_second`, `dimensionless`, `percent`, `categorical`) | `cm_s`, or `not_reported` |
| direction | present (`absorptive`, `secretory_over_absorptive`) | dropped |
| value | normalized `scalar_value` | `_first_number()` regex on prose |

The raw `efflux_or_secretory_transport` family is split upstream into **distinct, typed**
endpoints — `q3.transporter_substrate_status` (`categorical`/`binary_category`),
`q3.efflux_ratio` (`dimensionless`), `q3.apparent_permeability` (`cm_per_second`) — that
TxAgent collapses into one bucket and forces to a float (a categorical "substrate" → `None`,
a "−1.5 ± 9.3 %" prose string → a meaningless `-1.5`).

## How the upstream handles the "weird" data (for reference)

- **Uncertainty (`8 ± 0.5`)**: `pipeline/source_normalization/scalar.py` parses mean +
  variation into structured fields (`scalar_value`, `variation_value`, interval bounds).
  The prompt's value line shows only the normalized mean; raw `± ` narrative survives in
  `extra_details`/`support_text`.
- **Categorical (`P-gp substrate`)**: kept as a validated `categorical_value` (allowlist)
  under a `binary_category` endpoint; in scalar-endpoint prompts, substrate status appears
  as a **context** line, not the value.
- **Pair template (v6_5)**: query record + retrieval record share the same fully-canonical
  endpoint key; the retrieval value is shown, the query value hidden; task is
  "(A) transfer / (B) not transfer". Real example endpoint key:
  `q3.intestinal_transport.efflux_ratio.dimensionless_ratio.secretory_over_absorptive`.

## Decision

Stop reconstructing. Add an **in-distribution mode** that sources the pipeline from the
starling normalized dataset **start to finish** — retrieval pool, scoring records, and LLM
presentation — so the model is scored on exactly its training-format inputs and the LLM
sees the same rich, real records.

## Key facts (verified)

- **Authoritative per-record source: the eligible v6.5 training records**
  `/data1/joseph/starling_assay_transfer/datasets/eligible/assay_transfer_soft_evidence_v6_5/records.parquet`
  (~139k rows). Carries `assay_concept` (oral_bioavailability/oral_exposure/Fa/Fg/Fh) **and
  the 16 `context_<field>` columns already in template form** (e.g.
  `context_measured_process = 'efflux_or_secretory_transport'`) + `scalar_value`,
  `unit_basis`, `canonical_endpoint_key`, `endpoint_subtype`, `metric_type`, `child_id`.
  Rendering from these `context_*` fields reproduces the training prompt **exactly**.
- **Do NOT use `datasets/base/hf_cleaned/`** — that export overwrote the raw endpoint column
  (`gut_wall_process` = the canonical key), so `context_measured_process` would render the
  `q3.…` key instead of `efflux_or_secretory_transport`. The builder prefers `context_*` +
  `assay_concept` when present and only falls back to the `compose_v3` raw-column mapping for
  bare records (tests). Caught via the committed scoring-pair example.
- Narrative for LLM presentation: eligible records expose `context_extra_details` (used as
  `support_text`); the fuller `support_text` sentence lives in the canonical base
  (`datasets/base/canonical_endpoints_v1/*/records.parquet`, joinable by `record_id`) if a
  richer presentation is wanted later.
- Concept grouping already matches TxAgent's branches: `assay_concept ∈
  {oral_bioavailability, oral_exposure, Fa, Fg, Fh}` (`pipeline/v6_intern.py`) →
  `Observed.direct_oral_bioavailability` / `Observed.oral_auc_cmax_exposure` / `Fa.…` /
  `Fg.…` / `Fh.…`. (`q2=fa`, `q3=fg`, `q4=fh`.)
- Exact scoring template already vendored in TxAgent:
  `reranking/assay_transfer_templates/assay_transfer_v6_5_intern_default.jinja` +
  `AssayTransferPromptRenderer`. The v6_5 query side copies retrieval context with the value
  hidden (`copy_retrieval_assay_context_value_hidden.v1`), so a novel query molecule needs
  only its SMILES — reusable as-is.
- One raw extraction can yield several normalized records (scalar-splitting); `rerank()`
  already scores each and keeps the best (`transfer_winning_record_id`).
- The only join needed is **intra-starling** (normalized record → raw source via
  `parent_provenance_id`); no fragile cross-dataset (TxAgent↔starling) match.

## Architecture

Gate with `--retrieval-source starling_in_distribution` in `run_reasoning_pipeline.py`.
When enabled, sourced entirely from `hf_cleaned`:

1. **Loader/import** of the starling normalized dataset; canonicalize SMILES consistently
   with the reranker (`standardize_smiles`, using `datasets/_source/final_smiles_mapping_v2.parquet`
   where needed). New builder under `tasks/bioavailability_ma/reranking/`.
2. **Fingerprint retrieval index over the starling molecules** — built directly from
   `hf_cleaned` (Morgan fingerprints + the molecule's normalized records grouped by
   `assay_concept` → the five `group_id`s). This is NOT TxAgent's old aggregated evidence
   library (`build_starling_factor_evidence_library`), which is not used in this mode.
   May reuse the generic `build_neighbor_index` / `fingerprint_metadata` helpers; the
   records are the raw `hf_cleaned` rows.
3. **Retrieval**: top-k per group over the in-distribution pool via
   `common/experiment_retrieval.py::retrieve_experiment_view`.
4. **Scoring (the fix)**: build the reranker catalog from the **real normalized records**
   (canonical_endpoint_key / scalar_value / unit_basis / context straight from `hf_cleaned`)
   instead of `_index_example_record`; render with the existing `AssayTransferPromptRenderer`
   (v6_5). `AssayTransferCachedReranker.rerank()` reused unchanged.
5. **LLM presentation (group + final)**: present the neighbor's normalized record(s) with
   full metadata via the `group_prompt_render` field-policy path (new `in_distribution.*`
   record-type policy). Optional enrichment via `parent_provenance_id` → raw source for any
   field `hf_cleaned` drops. `transfer_winning_record` carries the normalized record, so no
   cross-dataset join.

Legacy / Morgan / existing reranker paths are untouched; this is additive behind the flag.

## Independent axes (do not conflate)

- **Data sourcing** (this doc): starling normalized `hf_cleaned` vs TxAgent's reconstructed
  catalog. This is the correctness fix.
- **Retrieval mechanism** (separate ablation): Morgan-fingerprint retrieval vs
  assay-transfer-tool retrieval (`--retrieval-reranker {none,assay_transfer}`). Both arms
  run over the same in-distribution pool; we ablate both.
- **Presentation granularity**: default = winning normalized record, raw (matches the
  `assay_transfer_tool` format); "all records" is a field-policy/mode toggle.

## Verification

1. Endpoint-key sanity: scored `endpoint:` lines read `q2/q3/q4.…` with normalized units and
   real `scalar_value` (not `index.*` / `not_reported`).
2. Score sanity: on a known query, compare in-distribution scores vs old reconstructed
   scores on the same neighbors; qualitative Fg/Fh records no longer forced to spurious
   scalars.
3. `pytest tests/chembl_tool/tasks/bioavailability_ma -q` green (env
   `/data1/joseph/miniconda3/envs/txagent-glm/bin/python`); add tests for the new
   loader/grouping and that catalog records carry real canonical keys.
4. Recompile `prompt_audit` examples for an in-distribution run; group prompt shows rich
   normalized records.

## Implementation status

- [x] **Stage 1 — hf_cleaned loader + in-distribution evidence/index builder.**
  `reranking/build_starling_in_distribution_library.py` (+ tests). Maps
  `canonical_endpoint_key` prefix (`q1/q2/q3/q4`) → the five `group_id`s; emits one
  evidence row per normalized record (real `canonical_endpoint_key`/`scalar_value`/
  `unit_basis` + `starling_record` + provenance); builds the Morgan neighbor index.
  Verified on real data (real `q3.…` keys, no `index.*`).
- [x] **Stage 2 — in-distribution catalog record + exact v6_5 render.**
  `in_distribution_catalog_record()` builds a catalog record from the normalized fields
  (template context mirrors `compose_v3._context`; metric/threshold via `_v3_scoring_profile`;
  catalog id = `child_id` so scalar-split records don't collide; categorical/no-scalar →
  unscoreable). Verified: renders through `AssayTransferPromptRenderer(v6_5)` to an
  in-distribution prompt (real key + normalized value/unit + substrate context, no `index.*`).
- [x] **Stage 3a — materialize the catalog.** The builder writes
  `starling_in_distribution_catalog.jsonl` (metadata + `assay_record` rows); verified
  loadable by `AssayTransferCatalog` (template_id/hash/uniqueness checks pass).
- [x] **Stage 4 — retrieval-source wiring.** `experiment_config.STARLING_IN_DISTRIBUTION`
  (+ `STARLING_RETRIEVAL_SOURCES`); guards in `run_reasoning_pipeline` /
  `assay_transfer_prompt_policy` accept `starling_in_distribution`. The **Morgan-retrieval
  arm runs end-to-end with no GPU** (retrieval over the in-distribution pool + rich
  presentation).
- [x] **Stage 5 — LLM presentation.** Every top-k assay-transfer record uses the shared
  `minimal_evidence.v1` / `morganfingerprint.record` field policy; the transfer score and
  record-level top-k selection are the only retriever-specific presentation elements. The
  same molecule may appear in multiple ranked entries. The in-distribution catalog carries
  the raw `support_text`, so each normalized selected record
  presents the real endpoint, value, unit, context, and narrative. Verified. Reconstructed
  catalogs keep an empty narrative when none exists.
- [x] **Stage 3b — candidate manifest + score precompute.** Precompute consumes the
  immutable `starling_in_distribution_catalog.jsonl`, freezes only the condition manifest,
  and scores all identity-eligible survivors from the Morgan top 100 using the v6.5 prompt.

## Runbook

```bash
PY=/data1/joseph/miniconda3/envs/txagent-glm/bin/python
# 1) Build the in-distribution index + catalog (no GPU)
$PY -m tools.chembl_tool.tasks.bioavailability_ma.reranking.build_starling_in_distribution_library \
  --out-dir outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution

# 2) Morgan-retrieval arm end-to-end (no GPU): rich in-distribution records, no scores
$PY -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline \
  --retrieval-source starling_in_distribution --experiment-mode full_mechanism \
  --index <out-dir>/starling_in_distribution_neighbor_index.pkl \
  --retrieval-reranker none --group-prompt-format morganfingerprint ...

# 3) assay_transfer arm — after Stage 3b (candidate manifest + GPU precompute over
#    starling_in_distribution_catalog.jsonl): add
#    --retrieval-reranker assay_transfer --rerank-catalog <out-dir>/starling_in_distribution_catalog.jsonl
#    --rerank-cache <scores.sqlite3> --group-prompt-format assay_transfer_tool
```

## Open items

- Exact flag surface (`--retrieval-source starling_in_distribution` vs a dedicated
  `--in-distribution-transfer`).
- Whether to snapshot/import `hf_cleaned` into TxAgent's `data/` or reference the starling
  repo path via config.
- Which raw fields (if any) the LLM presentation should pull via `parent_provenance_id`
  beyond what `hf_cleaned` already carries.
