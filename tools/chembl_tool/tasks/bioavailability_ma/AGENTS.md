# Bioavailability_Ma paper-path notes

本目录只保留可进入论文主方法的简洁 pipeline：通用 evidence contract、molecule-level retrieval、
single/group/final LLM reasoning 和 task ontology。历史 Fa/Fg/Fh full expert policy、deterministic
force/block/rescue、fallback calibration、postprocess 和 test-error-driven evolution 已从 `main` 删除。

旧实现的完整快照保存在：

```text
branch: archive/bioavailability-full-expert-policy-20260710
commit: 14803c2
```

不要从该 archive 向 paper path 恢复 class-changing policy。

## Task contract

```text
input:
  drug: query SMILES
  Y: 0/1 evaluation label

label:
  Y=1 -> high, oral bioavailability F >= 20%
  Y=0 -> low, oral bioavailability F < 20%
```

新的 Starling-held-out benchmark 由 `starling_benchmark.py` 构建：只接受可确认 human context 的
direct oral F；百分数与明确 fraction 统一到 percent，跨 20% 的 range、relative comparison、
非 human、population 不明或 `qualifying_conditions` 非空的记录都不进入 gold label。
parent-level 0/1 冲突分子不做多数票。
这里的 benchmark label conversion 与下文禁止的 inference-time Starling label policy 是两回事；
它不能进入 LLM prompt 或改变有效 prediction。

当前 frozen build 位于：

```text
data/processed_starling/Bioavailability_Ma/random/
data/processed_starling/Bioavailability_Ma/scaffold/
```

共有 1,862 个 binary parents；两种 split 的 test target 均为 372。公共构建/审计协议见
`tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md`。正式运行前必须按各 split 的
`test_molecule_labels.jsonl` 分别重建 train-only retrieval index。

Exact-query evidence 默认关闭。Neighbor retrieval 是 evidence prefetch，不是 LLM function tool。

## Paper pipeline

```text
source rows
  -> minimal_evidence.v1
  -> molecule-level aggregation and fingerprint index
  -> operational or parent-disjoint identity filtering
  -> source-local groups mapped into direct/mechanism families
  -> parallel single-molecule and mechanism-family reasoning
  -> final LLM synthesis
  -> structured-output validation only
```

允许的工程防护：

- canonical SMILES 和 molecule-level aggregation；
- exact-query exclusion；
- source/provenance 保留；
- single branch 的 `molecule_properties`；
- group branch 的 `mmp_structure_compare` / `properties_compare`；
- JSON schema validation 和有 trace 的 bounded retry（当前默认最多 4 次总尝试）；
- trace、batch resume 和 metrics。

禁止添加：

- `force_high` / `force_low`；
- final prediction override；
- 针对某个 test molecule 或 failure pattern 的 blocker/rescue；
- Starling-specific label policy；
- valid/test-selected postprocess；
- 将内部 `evidence_direction` / `evidence_strength` 直接发送给 LLM。

## Minimal evidence contract

所有 ChEMBL、Starling 和未来 source row 在 LLM prompt 中统一使用：

```text
tools/chembl_tool/common/evidence_contract.py
contract_version: minimal_evidence.v1
```

Contract 包含：

```text
source
molecule
group
endpoint + measurement
evidence/context text
annotations: evidence_role, scope, transferability, uncertainty
quality
provenance
representative examples
```

其中 `transferability=not_assessed` 是 retrieval-time 默认值；query-specific transferability 必须由
group LLM 根据结构比较和 evidence context 判断。Contract 不包含 threshold vote 或 label recommendation。

## Data sources

Paper-facing source/group mapping 的唯一配置入口：

```text
tools/chembl_tool/tasks/bioavailability_ma/experiment_config.py
```

它声明 ChEMBL/Starling 的 direct groups 和 5 个 mechanism families；不得在 runner 或 source adapter 中
复制该 mapping。

ChEMBL evidence library：

```text
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/
  bioavailability_molecule_evidence.jsonl
  bioavailability_neighbor_index.pkl
```

Starling task data：

```text
data/starling_data/bioavailability_ma/
  Oral_AUC-Cmax_Exposure/extractions.parquet
  Fa/extractions.parquet
  Fg/extractions.parquet
  Fh/extractions.parquet
```

Starling factor builder：

```text
tools/chembl_tool/tasks/bioavailability_ma/build_starling_factor_evidence_library.py
```

Starling gold benchmark adapter：

```text
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
```

前者构建 inference-time evidence/index；后者只实现 direct human oral F 的 binary label adapter。
二者不能互相替代。

Starling factor builder 使用 shared profile ingestion：

```text
tools/chembl_tool/common/starling/evidence_library.py
```

Task wrapper 只声明 column mapping 和 group/role：

| Source | Group | Evidence role |
|---|---|---|
| HF Oral Bioavailability | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| Oral_AUC-Cmax `bioavailability` rows | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| Other Oral_AUC-Cmax rows | `Observed.oral_auc_cmax_exposure` | `surrogate_proxy` |
| Fa parquet | `Fa.absorption_solubility_permeability` | `mechanistic_factor` |
| Fg parquet | `Fg.gut_wall_efflux_intestinal_metabolism` | `mechanistic_factor` |
| Fh parquet | `Fh.hepatic_clearance_metabolic_stability` | `mechanistic_factor` |

Fa/Fg/Fh 是 task ontology，不是 deterministic classifier。Final prediction 仍由 LLM 根据 group outputs
综合得出。

## Source ingestion rules

- Parquet column 差异通过 `StarlingSourceProfile` 配置解决，不为每个 parquet 写独立 parser。
- 缺少 SMILES 或 RDKit 无法解析的 row 不进入结构 retrieval，并计入 source stats。
- SMILES 在聚合前 canonicalize；同一 canonical molecule 跨 profile 使用相同 source molecule id。
- 多条 source row 聚合成一个 molecule/group evidence row；代表性 examples 保留 endpoint、value、unit、
  context 和 support text 的绑定。
- PMID/DOI 可保留在 raw internal row 中，但不能进入 LLM-visible minimal evidence。
- Direct numeric outcome 只有在 endpoint 单一、unit 一致时才生成 aggregate measurement；proxy/mechanism
  evidence 保留 examples，不把异构数值混成一个 synthetic value。

## Layered normalized-record library

The side-by-side layered Starling builder is:

```text
tools/chembl_tool/tasks/bioavailability_ma/build_normalized_starling_evidence_library.py
```

The policy-decoupled v5 builder persists every boundary before molecule aggregation:

```text
starling_normalized_v5/
  01_cleaned_records.parquet
  02_normalized_records.parquet
  records.parquet
  duplicates.parquet
  organization_exclusions.parquet
  scalar_distribution_audit.parquet
  context_canonicalization_policy.json
  molecule_family_evidence.jsonl
  neighbor_index.pkl
  manifest.json
```

Cleaning is meaning-preserving. Normalization consumes `01_cleaned_records.parquet` directly and emits exactly
one pre-deduplication record for every `cleaned_record_id`. It owns the atomic `canonical_measurement` /
`canonical_unit` pair and scalar metadata; organization owns within-source deduplication and retrieval
eligibility. Cleaning also promotes cleaned source-facing context columns and
`source_smiles` to top-level columns. Endpoint identity has three explicit layers:

```text
endpoint_name
  -> spacing_and_spelling_endpoint
  -> canonical_endpoint
```

`endpoint_name` preserves the cleaned source value. The middle layer applies only an explicit audited
spacing/spelling correction and preserves case, with status, reason, and version provenance.
`canonical_endpoint` is then derived mechanically by casefolding and treating whitespace, underscores,
hyphens, and Unicode dash variants as equivalent separators. Other punctuation and meaningful symbols are
preserved. Report context, assay context, and units never rewrite endpoint identity. Original endpoint,
measurement, unit, SMILES, and source context are authoritative for presentation and remain both top-level
and in `source_payload_json`.

Measurement normalization has one authoritative path. Cleaning preserves the source-facing
`measurement_text` and `unit_text`; normalization removes an exact repeated source-unit suffix, parses one
complete atomic numeric expression, folds explicit notation once, canonicalizes the unit once, applies the
reviewed endpoint conversion, and then evaluates factual domain validity. No alternate raw-unit argument or
hidden `source_payload_json` fallback participates in normalization or pair-invariant validation.

`is_absolute_and_continuous` means an explicit finite point value. It includes valid percentage, ratio,
fold, permeability and other scalar endpoint measurements; it does not mean "non-ratio" or specifically
absolute oral F. Bounds, ranges, qualitative text and ambiguous compound measurements remain evidence with a
null scalar. `normalization_validity_status` records policy-independent structure, parsing, unit and physical
domain validity. The existing `finite_scalar_value` and `canonical_unit` are the policy-ready comparison
input; v5 does not persist labeling policy, distance, threshold or comparison-value duplicates.

Joint and compound measurements are never split. Papp/efflux, AUC/Cmax ratios, Vmax/Km, statistics,
vector-valued series, condition series, dissolution-model parameters, clock times, and formulation ratios each
remain one intact record. If the complete row cannot be parsed as one measurement it receives no finite scalar,
but remains retrieval-eligible when its structure and mechanism family resolve. Later scalar assay-transfer
datasets must select only records with non-null `absolute_and_continuous_value`.

The v1-v4 directories are historical audit artifacts. The v5 code does not reproduce or rewrite them.

Scientific notation is folded only when explicit, whether the factor appears in the source unit or in an
atomic source measurement. `2.5` with unit `×10^-6 cm/s` and `2.5×10^-6 cm/s` with unit `cm/s` both become
`0.0000025 cm/s`. Shared-factor point estimates such as `175 ± 19 ×10^-6` scale the value and variation
atomically. Bounds, ranges, compounds, conflicting factors, and OCR-compressed `×106` forms are never promoted
to scalars; ambiguous notation is retained and marked rather than guessed. Notation is provenance, not a
permeability comparison stratum.

Endpoint-specific conversion is controlled by `starling_normalization_policy.py` and selected only by
`canonical_endpoint`. Unit dimensions may select a reviewed representation, such as mass- versus molar-AUC,
but cannot alter endpoint identity. When no reviewed conversion applies, the mechanically normalized
measurement, unit, and status are retained. Every nonempty unit remains valid and defines a distinct
`source_id + canonical_endpoint + canonical_unit` comparison stratum. Metric-domain gates retain invalid
rows for provenance but exclude them from assay-transfer pairing. The builder still fails on endpoint-inventory
drift in full builds.

The versioned source-aware pair-bucket sidecar is the context-comparability contract. Its required source
fields are report type for `direct_hf`, oral dose for `oral_exposure`, assay system for `fa` and `fg`, and
species plus assay system for `fh`. Null-like mapped fields use `__unknown__`; unknown matches unknown.
Every fact-valid scalar record belongs to exactly one bucket, and `source_id` prevents cross-source pairing.
The sidecar does not enumerate pairs, create labels, set thresholds, or produce modeling datasets.
All context canonicalization now occurs in the normalize stage. Dose uses quantity kind, basis, factor-of-two
magnitude bin, and regimen; species uses base taxon with sex/strain/model qualifiers preserved separately;
assay systems use reviewed platform/protocol/modifier classes with conservative protected-token fuzzy
matching and exact `unmapped:` fallbacks. It is a standalone derived artifact and is not built automatically:

Dose regimen has one deterministic resolver. `/d`, `/day`, `per day`, `QD`/`q.d.`, daily, and
BID/TID/QID forms are repeated-dose signals; a calendar reference such as `day 1` is not. Text containing
both explicit single-dose and repeated-dose signals uses the distinct
`conflicting_single_and_repeated` regimen and cannot share either bucket. Permeability direction recognizes
the A/B and AP/BL abbreviations, arrows, `to`, and spelled apical/basolateral forms. A record containing both
directions remains mixed and receives no scalar-transfer policy.

```text
starling_normalized_v5/pair_buckets/
  pair_bucket_records.parquet
  pair_bucket_metadata.json
```

The bucket key is the readable JSON tuple
`source_id + canonical_endpoint + canonical_unit + persisted source-aware fields`. The sidecar never reads
`source_payload_json`, normalizes text, applies fuzzy matching, assigns a policy, or re-evaluates metric
domains. Original context and displayed values remain unchanged and can be joined by `normalized_record_id`.

Endpoint-specific distance and labeling policies are a standalone downstream artifact:

```text
starling_normalized_v5/endpoint_policies/v1/
  endpoint_policy_assignments.parquet
  endpoint_policy_registry.json
  endpoint_policy_metadata.json
```

Each fact-valid supported row receives an endpoint-specific policy key. v1 inherits the former v4 distance
functions and cutoffs without calibration. Rows formerly rejected as `missing_comparison_policy` remain
unassigned with `unsupported_assay_transfer_semantics`; they remain retrieval evidence but cannot enter
assay-transfer labeling. Effective assay-transfer eligibility requires both a context bucket and an assigned
endpoint policy. This artifact does not enumerate or label molecular pairs.

Endpoint policy v2 is an additional standalone artifact; v1 is immutable:

```text
starling_normalized_v5/endpoint_policies/v2/
  endpoint_policy_assignments.parquet
  endpoint_policy_registry.json
  endpoint_policy_metadata.json
```

Every assignment row contains `normalized_record_id`, `measurement_subtype`,
`endpoint_policy_key`, and `policy_assignment_status`. Keys have the form
`bioavailability_ma/<canonical_endpoint>/<measurement_subtype>/v2`. V2 has no generic
positive-scalar fallback: it explicitly distinguishes bounded percentages/fractions,
exposure metrics, Tmax, half-life, permeability direction, clearance, solubility mode,
dimensionless ratios, and resolved kinetic parameters. Mixed or under-specified semantics
remain unassigned. Percentage dissolution/stability requires a resolved time context;
directional permeability and efflux comparisons use direction-specific subtypes.
Finite negative permeability coordinates reported as `log(cm/s)` or `log10(cm/s)` are factually valid.
Bioavailability v2 uses the reviewed convention that bare permeability `log` is log10 and compares these
already-transformed values by absolute coordinate difference; it never applies log10 a second time.
Transformed metrics outside this reviewed permeability case remain valid evidence but unassigned.

Every assigned endpoint policy resolves exactly one named threshold profile using the frozen
precedence `endpoint+subtype override -> subtype-prefix override -> subtype override -> policy-family
default`. The registry publishes the selected profile, resolution reason, profile version, raw thresholds,
normalization anchor, and normalized cutoffs. AUC, Cmax, other exposure, Tmax, half-life, rate constants,
hepatic clearance, intrinsic clearance, equilibrium/kinetic solubility, ratios, and resolved Km/Vmax each
have explicit semantic profiles. Profiles may currently share the same reviewed numeric thresholds; the
separate names prevent unrelated endpoints from being silently coupled in later reviewed revisions.
Papp and Peff use the wide log profile (3-fold transfer / 10-fold not-transfer primary) because permeability
systems are heterogeneous. Already-log10 permeability coordinates use identity coordinate distance with the
same wide thresholds and `log10(20)` permissive far anchor. Pair-bucket matching still independently enforces
direction and assay-system comparability.

V2 freezes primary and sensitivity thresholds before model evaluation. Its calibration
audit excludes the union of random and scaffold test parents and emits aggregate
diagnostics only: assignment/exclusion counts, observational same-parent dispersion,
cross-parent distance summaries, aggregate record and unique-molecule pair yields,
coverage, and sensitivity-label stability. It does not materialize training pairs, read
benchmark outcomes, modify modeling datasets, or fit thresholds.

V2 also publishes `endpoint_policy_distance_normalization.v1`, one invariant distance
score per pair on a clipped 0--100 scale. Each policy family is normalized linearly in
its resolved endpoint/subtype profile's reviewed raw comparison space, with 100 anchored at that profile's
permissive `not_transfer_min`. The same score is used for strict, primary, and permissive
sensitivity analyses; only their normalized cutoffs differ. Pair labels, deadbands,
stability calculations, and inclusive boundary behavior continue to use full-precision
raw distance. Raw and normalized values are stored at full floating-point precision,
while reports and LLM-visible text render normalized scores to two decimal places.

The downstream artifacts can be rebuilt without rewriting normalized v5:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments_v2

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.audit_starling_pair_bucket_distributions \
  --policy-version v2

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.audit_starling_endpoint_policy_v2_calibration
```

Staged normalized-v5 rebuilds publish each stage only after its temporary outputs
have been written and validated. A successful `clean`, `normalize`, or `organize`
rebuild deletes every active downstream artifact, including pair buckets,
endpoint-policy v1/v2, analyses, evidence JSONL, and the neighbor index as
applicable. Failed stage computation preserves the prior coherent build. Stage
resume also verifies that the recorded immediate-upstream input hash still
matches the current upstream artifact; a self-consistent but stale downstream
stage cannot be resumed. Endpoint-policy v1 remains frozen in definition but is
intentionally deleted, not regenerated, when `records.parquet` is rebuilt.

The local v6.5 identifier-to-SMILES mapping must be staged at:

```text
data/starling_data/_shared/final_smiles_mapping_v2.parquet
sha256: 98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b
```

Build with the local project environment:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_normalized_starling_evidence_library \
  --workers 128
```

Use `--from-stage` and `--through-stage` with `clean|normalize|organize|index` to inspect or restart a hashed
stage. The legacy factor/direct builders and v1-v3 artifacts remain unchanged. Experiments must opt into
`starling_normalized_v5/neighbor_index.pkl` explicitly until its full artifacts and held-out exclusion
variants have been audited.

## Commands

构建 Starling index：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full \
  --workers 32
```

用通用 pipeline 跑 Starling source：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl \
  --api-key-env GLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --batch-id bioavailability_ma_paper_starling_<date>
```

正式 paper run 只使用上述 `outputs/paper/` index。`outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/`
下的 task-level builder 默认目录只用于临时开发，不得把历史 index 复制或软链接到正式实验路径；运行前应检查
meta 中 `index_version`、`include_direct_hf`、`scope`、`evidence_content` 和五个稳定 group ID。

2026-07-23 strict-hop availability census 见
`outputs/chembl_tool/tasks/bioavailability_ma/distance_expansion/analysis/hop_availability_census/`。C 之外的
experimental pKa、LogD/LogP、PPB/Fu 可组成 H1 candidate union（59,049 parents；parent-disjoint >=1 coverage
98.44%），但没有合格 H2。pKa/LogD/LogP 与 query `molecule_properties` tool 语义重叠；PPB/Fu 只支持
hepatic clearance 而非 direct absolute F。它们只能作为独立 distance/relevance 设计候选，不得修改现有
paper matrix。

API key 只能通过环境变量或未提交的本地 env file 提供，不能写入代码、manifest、命令示例或 git。

## Tests

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m pytest \
  tests/chembl_tool/common \
  tests/chembl_tool/tasks/bioavailability_ma -q
```

至少覆盖：

- legacy ChEMBL row -> `minimal_evidence.v1`；
- profile-driven parquet column mapping；
- missing/invalid SMILES stats；
- molecule-level aggregation和 exact-query exclusion；
- direct/proxy role split；
- JSON required-field/value/tool validation retry；
- API key 不进入 tracked files。

## Evaluation boundary

历史 Bioavailability test set 已在旧 expert-policy 迭代中被反复检查，不能作为论文的 untouched final
test。Paper result 应使用新 holdout、重新冻结的 split 或外部 evaluation。Archive branch 的历史 metrics
不能作为当前 simplified paper pipeline 的结果。

## Cached assay-transfer reranking

The optional `assay_transfer` retrieval reranker is restricted to the five Starling families declared in
`experiment_config.STARLING`. It is disabled by default. Candidate selection is frozen as Tanimoto top 100,
then the shared identity exclusion policy, then every survivor, then cached model reranking, then the existing
final `top_k_per_group`. It never scans or backfills below the raw top 100.

The current imported cache is the retrieval-agnostic `cache_v2` store under
`evidence_library/assay_transfer_rerank/`. It contains an append-only SQLite score store and its provenance
artifacts. Score identity includes only the exact prompt hash, immutable model revision, scoring contract,
and template hash. Catalog, split, source composition, and identity policy are audit/join metadata and never
change score identity.

A separate soft-checkpoint cache is preserved at
`evidence_library/assay_transfer_rerank/cache_soft_5fc06af6/`. It contains all 45,057 unique prompts in the
frozen prepared-HF validation catalog and was built on GPUs 0--5 with
`jiosephlee/assay-transfer-tool-soft@5fc06af66b490575e8eb32231d96aeada26794c6`. Its immutable
`VERSION.json` records the exact catalog/manifest hashes, scoring contract, model runtime, devices, and row
count. The original `cache_v2` remains unchanged. Soft-checkpoint validation uses these scores only to rank
neighbors; it does not pass the scores or their semantics to reasoning prompts.

The soft checkpoint's custom tokenizer requires `transformers==4.57.6` and
`huggingface-hub==0.36.0`. The local ignored `.runtime/transformers_4_57_6/` overlay supplies those versions
for cache inference without changing the project environment. Precompute commands for this checkpoint must
prepend that directory to `PYTHONPATH`; reasoning runs only read SQLite and do not need the overlay.

Both direct-bioavailability source schemas are normalized: HF Oral Bioavailability records use
`oral_bioavailability_value_percent`, while direct rows from Oral_AUC-Cmax use
`reported_value`/`reported_units`. This normalization is only for assay-transfer prompts; it does not change
the Starling evidence indexes or ordinary Morgan retrieval.

The versioned `v6_5_query_context_copy` prompt profile vendors the exact
`assay_transfer_v6_5_intern/default.jinja` contract (SHA-256
`e30f995988db7214cae4b170a2c36f3a5fd61b6bee6188b39ce54377fa67f5bd`). It renders the retrieval
record with its canonical SMILES, scalar measurement, unit, canonical endpoint, assay concept, and complete
assay context. The query uses canonical SMILES and copies the retrieval endpoint/concept/context while never
receiving a scalar or known-value field. `legacy_v3` remains the default so existing catalogs, cache identities,
and runs are unchanged. Cache and reasoning CLIs select the new contract explicitly with
`--assay-transfer-template-profile v6_5_query_context_copy`.

The corrected `v6_5_query_context_copy_no_extra_details` profile uses the same immutable v6.5 Jinja and
canonical endpoint contract, but copies every retrieval assay-context field to the query except
`extra_details`. Retrieval-side `extra_details` remains visible and the query always renders
`extra details: not specified`. Its cache provenance records
`copy_retrieval_assay_context_except_extra_details_value_hidden.v1`, and corrected artifacts live under
`evidence_library/starling_in_distribution/v6_5_no_query_extra_details/`; do not reuse the historical
`validation_parent_disjoint_r100_c100_min0_v6_5/` condition as the corrected cache.

For the frozen prepared-HF validation manifest at similarity floor 0.30, there are 9,863 record-level prompt
references. Exact v6.5 rendering collapses 39 duplicate references into 9,824 unique retrieval-agnostic prompt
hashes; the older v6 template produced 9,844 unique hashes. The v6.5 SQLite cache therefore stores 9,824 scores
while covering every record reference. Its `VERSION.json` records both counts and the reason for the difference.

## Hosted GLM validation concurrency

These instructions apply directly to `run_reasoning_batch.py`, `run_reasoning_pipeline.py`, and all
Bioavailability hosted-GLM launch configurations.

### Concurrency target

- Treat 256 as the maximum number of outstanding GLM completion requests, not as a target for molecule
  subprocesses or a multi-prompt request payload.
- For a fresh `full_mechanism` Bioavailability batch with the current five mechanism families, default to
  `--parallelism 48 --group-workers 5`. This permits approximately 240 simultaneous group requests and leaves
  headroom below the 256-request API limit.
- Calculate effective fan-out as
  `min(parallelism, unfinished_molecules) * min(group_workers, active_group_count)`. Do not report
  `parallelism * group_workers` when fewer groups exist. For example, `32 * 8` has an effective group-stage
  ceiling of 160 when only five mechanism families are active.
- For a different experiment view, choose the largest safe molecule parallelism that targets 224--240
  effective requests without exceeding 256. Set `group_workers` to the number of concurrently active group
  branches unless there is a measured reason to use less.
- Do not default new hosted-GLM runs to serial execution. If the observed endpoint or node cannot sustain the
  target, reduce concurrency in measured 20--25% steps and record the errors and final setting.

### Launch and resume requirements

- Use the local `txagent-glm` environment and inject the ignored local API key through `LITELLM_API_KEY`;
  never print or persist the key.
- Use `--skip-existing` whenever resuming so completed molecule outputs are preserved. A partial molecule
  without `final_reasoning_output.json` may be rerun.
- Preserve separate retrieval, group, final, and batch output directories for each k value or condition.
  Never reuse final predictions across k values.
- Archive a genuinely failed attempt before a clean retry. An intentional concurrency restart is a resume,
  not a failed-attempt archive.
- Keep score-hidden runs score-hidden: assay-transfer scores may rank neighbors but must not appear in any
  GLM-visible prompt.

### Preflight and monitoring

Before issuing GLM requests, verify:

- the tool service health endpoint is healthy;
- SQLite `quick_check` is `ok` and the required score count and immutable provenance match;
- the frozen catalog, candidate manifest, model revision, template profile, and template hash match;
- the node has enough available memory for the requested molecule-process count, because each molecule
  process loads retrieval state;
- the configured effective GLM fan-out is no greater than 256.

During the run, monitor completed outputs, active molecule processes, HTTP 429/5xx responses, timeouts,
validation retries, and memory. Concurrency should remain near the target when healthy. Endpoint throttling,
repeated transport failures, or memory pressure are valid reasons to reduce it; document the reason rather
than silently reverting to serial execution.

After completion, require 64/64 outputs and zero failures, then run the task's retrieval, identity,
similarity-floor, group-size, below-k attribution, prompt-visibility, and provenance audits.

Prompt-audit tooling lives in the `prompt_audit/` subpackage: `view_run.py` (trace/prompt viewer),
`dump_prompt_examples.py` (example generator), and the committed `prompt_audit/prompt_examples/`. The examples
are one per stage (single, group, final), with the group stage in all three formats (`group.legacy`,
`group.morganfingerprint`, `group.assay_transfer_tool`). They are **compiled prompts only** (no model output),
generated from one frozen real datapoint (`tests/.../fixtures/prompt_examples/fixture.json`) by
`python -m tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples --write`, and pinned by
the golden test `tests/.../test_prompt_examples.py` so they cannot drift from the prompt code — regenerate after
any prompt change. Never add hidden labels, API credentials, or non-visible provenance to an example, and never
add assay-transfer scores to a **score-hidden** example; the `assay_transfer_tool` format legitimately shows its
by-design visible transfer score.

The group text-format instruction blocks are editable `.txt` files under `prompt_instructions/`
(`morganfingerprint.txt`, `assay_transfer_tool.txt`; one instruction per line, `#` comments ignored), loaded by
`group_prompt_render.load_instructions`. Which metadata fields appear is likewise editable in
`group_prompt_field_policy.py`.

Use `--group-prompt-instructions-file <path>` for a versioned instruction ablation instead of replacing the
default file. The batch runner fingerprints the selected UTF-8 file, passes the launch-time SHA-256 guard to
every molecule process, and records the resolved path, hash, and instruction count in batch and run manifests.
Missing, empty, unreadable, or mid-run modified files fail before a mixed-prompt condition can be produced.

Use `--group-output-schema legacy|assay-transfer` to select the group response contract independently of the
prompt layout. `legacy` remains the default. The evidence-centric `assay-transfer` profile is permitted only
with `--group-prompt-format assay_transfer_tool`; it replaces rigid transferability/direction enums and
molecule-ID/similarity/tool fields with free-text assay-transfer assessment, bioavailability implications,
record rank/endpoint/likelihood evidence, confidence, and caveats. The selected profile and immutable schema
hash must be recorded in each run manifest.

When a reasoning-only ablation reuses the exact retrieval configuration of an existing batch, pass
`--rerank-preflight-source-batch <batch-dir>` to reuse that batch's completed deterministic cache preflight.
The source manifest must match the current indices, cache/catalog/manifest, profile, identity policy, threshold,
and k; actual retrieval replay and post-run threshold/group-size audits remain required.

Cached assay-transfer reranking code lives in the `reranking/` subpackage
(`tasks/bioavailability_ma/reranking/`: rerank, prompt policy, catalog/precompute builders, and the scoring
`assay_transfer_templates/`).

Assay-transfer scoreability filtering and neighbor selection are independent of prompt visibility. A score-hidden
ablation may use the cached scores to select the same ordered neighbors while omitting all transfer scores and
score-policy semantics from LLM-visible group and final prompts.

Attached numeric source-record examples become candidate-scoped prompt records; qualitative-only candidates
remain in the retained pool but do not receive fabricated prompts. Scoring mirrors training evaluation:
append `(A)` and `(B)` with no leading space, find their first divergent token, and softmax the two next-token
logits.

### Assay-transfer inference memory and batch size

The pinned `jiosephlee/assay-transfer-tool@9515603b...` checkpoint is a Qwen3 causal LM with 36 layers,
hidden size 4096, and about 8.2B BF16 parameters. Its four weight shards occupy 16.40 GB decimal
(15.28 GiB), which is the main fixed per-worker GPU cost because precompute loads one independent model
replica per GPU.

Measured cache inference with BF16 and batch size 32 peaked at 29.76--31.00 GiB per full B200 worker. The
293,021-prompt four-worker MIG90 run peaked at 22.11--23.54 GiB per worker and completed inference in 2,041
seconds. GPU memory is not determined by batch size alone:
approximately 15.3 GiB is fixed weights, while the remaining allocator/runtime, activations, padded tokens,
KV state, and output logits grow with both batch size and the longest prompt in that batch. Consequently,
do not extrapolate VRAM as purely linear in batch size.

For future 90 GB MIG or 180 GB full-B200 runs, use batch size 64 as the next default and retain the 64 GiB
free-memory preflight. First verify batch 64 on a bounded query subset and record the reported per-device peak.
Do not raise beyond 64 without a new measurement because prompt-length padding can change the peak. If a worker
OOMs, the append-only SQLite writer preserves already committed batches; rerun with batch size 32 to resume only
the missing scores.

### Runtime profiles: node002 and VAST/SLURM

Machine paths are centralized in `reranking/runtime_profile.sh`. Select one profile instead of editing Python
or launch scripts:

```bash
# node002: /data1 checkout + txagent-glm; direct execution
TXAGENT_RUNTIME_PROFILE=node002 \
  bash tools/chembl_tool/tasks/bioavailability_ma/reranking/build_in_distribution_library.sh
TXAGENT_RUNTIME_PROFILE=node002 \
  bash tools/chembl_tool/tasks/bioavailability_ma/reranking/precompute_in_distribution_validation.sh

# VAST: shared checkout/cache; scheduler wrappers select vast_slurm automatically
sbatch tools/chembl_tool/tasks/bioavailability_ma/reranking/slurm/build_in_distribution_library.sbatch
sbatch tools/chembl_tool/tasks/bioavailability_ma/reranking/slurm/precompute_in_distribution_validation_mig90.sbatch
```

The profiles define only non-secret runtime locations: project root, Python, Starling checkout, Hugging Face
cache/offline behavior, devices, workers, and batch size. Override any `TXAGENT_*` value in the environment for
a one-off machine layout. API credentials remain outside profiles. The evidence/index/catalog/cache paths stay
repository-relative, so copying or synchronizing the corresponding `outputs/.../starling_in_distribution/`
tree preserves the condition manifest and cache provenance across machines; otherwise rebuild the artifacts
from the same eligible-record checksum.

Build the candidate-scoped catalog and manifest, then populate only missing scores with independent BF16
model replicas. The validated v2 model provenance is
`jiosephlee/assay-transfer-tool@9515603b1a5c4586e41c221dcdbc5e7487c0c3f5` and scoring contract
`assay_transfer_chat_first_divergent_token_logits.v1`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
  -m tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank \
  --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_five_source_prepared_hf/starling_factor_neighbor_index.pkl \
  --experiment-mode full_mechanism \
  --neighbor-identity-policy parent_disjoint \
  --top-k-per-group 10 \
  --min-similarity 0.0 \
  --rerank-raw-pool-size 100 \
  --rerank-candidate-size 100 \
  --rerank-catalog <prepared_hf_validation_r100_c100_min0/catalog.jsonl> \
  --candidate-manifest <manifest.jsonl> \
  --rerank-cache <cache_v2/scores.sqlite3> \
  --rerank-devices 0,1,2,3,4,5,6,7 \
  --rerank-batch-size 64
```

Reasoning batches only read the cache and run a complete coverage preflight before starting subprocesses:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --retrieval-source starling \
  --experiment-mode full_mechanism \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_five_source_prepared_hf/starling_factor_neighbor_index.pkl \
  --retrieval-reranker assay_transfer \
  --rerank-raw-pool-size 100 \
  --rerank-candidate-size 100 \
  --rerank-catalog <prepared_hf_validation_r100_c100_min0/catalog.jsonl> \
  --rerank-candidate-manifest <prepared_hf_validation_r100_c100_min0/manifest.jsonl> \
  --rerank-cache <cache_v2/scores.sqlite3> \
  --rerank-cache-mode read_only \
  --enable-assay-transfer-scores \
  --top-k-per-group 7
```

Candidate-scoped catalogs require the exact condition manifest at runtime. The requested k is preserved;
there is no legacy forced-k5 behavior. With `--enable-assay-transfer-scores`, each selected score is rounded
to two decimals and exposed only to its group reasoning prompt. Full-precision scores, winning record IDs,
selection ranks, and structural ranks remain audit-only retrieval metadata and do not enter group or final
reasoning prompts.

Use `--assay-transfer-min-score <probability>` to apply an optional inclusive cached-score floor before the
final `--top-k-per-group` truncation. The default is unset so historical retrieval remains unchanged. The
threshold and rejected-record counts are recorded in retrieval, preflight, and batch audit metadata. For the
frozen in-distribution validation condition, `k=3` with a `0.5` floor retains all 960 selected records across
all 320 query-family groups; the minimum retained top-three score is exactly `0.5`.

The portable GLM validation launcher is `reranking/run_in_distribution_glm_validation.sh`; its VAST scheduler
wrapper is `reranking/slurm/run_in_distribution_glm_validation.sbatch`. Runtime profiles expose a separate
`TXAGENT_REASONING_PYTHON` because the VAST cache-inference and resident-tool environments have different
dependency sets. The launcher imports the ignored `keys.py` only when `LITELLM_API_KEY` is absent and never
prints or persists the credential.
