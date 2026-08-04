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
parent-level 0/1 冲突按 accepted source record 计算 70% agreement；同 PMID 多条 record 分别计票，
精确 tie 或 agreement 低于 70% 才拒绝。
这里的 benchmark label conversion 与下文禁止的 inference-time Starling label policy 是两回事；
它不能进入 LLM prompt 或改变有效 prediction。

当前 canonical-direct v2 build 位于：

```text
data/processed_starling/Bioavailability_Ma/random/
data/processed_starling/Bioavailability_Ma/scaffold/
```

共有 2,092 个 binary parents；两种构造方法的 valid/test target 均为 209。公共构建/审计协议见
`tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md`。正式运行前必须按各 split 的
valid+test union `heldout_molecule_labels.jsonl` 分别重建 train-only retrieval index。

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
  Oral_AUC-Cmax_Exposure/extractions.parquet        # immutable upstream
  canonical_direct_v2/
    hf_oral_bioavailability_snapshot.parquet
    direct_source_rows.parquet
    direct_claims.parquet
    cross_source_dedup_audit.parquet
    local_partition_audit.parquet
    merge_manifest.json
  oral_exposure_residual_v2/
    exposure_records.parquet
    partition_manifest.json
  Fa/extractions.parquet
  Fg/extractions.parquet
  Fh/extractions.parquet
```

统一 canonical source 构建入口：

```text
tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
```

原始 HF snapshot 与 local parquet 不原地修改。Local `bioavailability` 行只有出现明确 absolute wording 或
oral/IV anchor 才转入 canonical direct；relative 与没有 absolute anchor 的 ambiguous rows 留在 residual。
跨 HF/local 的同 parent+PMID 近等值 claim 做一对一去重并保留双来源 provenance。Gold builder 与 agent
direct evidence 必须读取同一个 `direct_claims.parquet` SHA-256。

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
| canonical direct v2 claims | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| residual Oral_AUC-Cmax/relative/ambiguous rows | `Observed.oral_auc_cmax_exposure` | `surrogate_proxy` |
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

The policy-decoupled v6 builder persists every boundary before molecule aggregation and attaches the
globally reconciled assay-transfer context sidecar:

```text
starling_normalized_v6/
  01_cleaned/{records.parquet, manifest.json, source_inventory.json, endpoint_inventory.json}
  02_normalized/{records.parquet, manifest.json, endpoint_registry.json,
                 record_validity_policy.json, auxiliary_mapping_manifest.json, source_contract.json}
  03_records/{records.parquet, duplicates.parquet, exclusions.parquet,
              scalar_distribution.parquet, manifest.json}
  04_pair_buckets/{pair_bucket_records.parquet, pair_bucket_metadata.json}
  05_assay_transfer_policy/{pair_bucket_transfer_policy.json.gz}
  06_remove_heldout_overlap/{random,scaffold}/records.parquet
  07_molecule_evidence/{random,scaffold}/
  08_neighbor_index/{random,scaffold}/
  09_audits/
  manifest.json
```

The local numbered directories are directly readable and ignored by Git. Each completed directory is also
packaged independently as deterministic `tar.zst` parts of at most 90 MB under
`artifacts/chembl_tool/tasks/bioavailability_ma/starling_normalized_v6/`, so a new checkout can restore only
the stages it needs. `starling_artifact_store.py` provides `package`, `restore`, `verify-tracked`, and
`verify-local` actions. Parquet compresses columns with Zstd; the outer deterministic archive further
compresses file/container overhead and allows ordinary GitHub tracking without Git LFS.

Cleaning is meaning-preserving. Normalization consumes `01_cleaned/records.parquet` directly and emits exactly
one pre-deduplication record for every `cleaned_record_id`. It owns the atomic `canonical_measurement` /
`canonical_unit` pair and scalar metadata; organization owns within-source deduplication and retrieval
eligibility. Cleaning also promotes cleaned source-facing context columns and
`source_smiles` and every declared raw source column to sparse top-level Parquet columns. Endpoint identity
has three explicit layers:

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
measurement, unit, SMILES, and source context are authoritative for presentation. Compact v6 does not persist
the former duplicate `source_payload_json` serialization.

LLM visibility is governed separately by `source_column_contract.v1`, persisted as
`starling_normalized_v6/02_normalized/source_contract.json`. Every column in every source schema and every column
in the normalized artifact has the Boolean metadata `source_or_simply_cleaned`. All genuine source fields,
including source provenance such as PMID and extraction ID, are eligible; derived canonical, normalization,
policy, scoring, audit, and helper fields are not. Prompt assembly must project the declared source fields
under their original column names and must fail closed rather than substitute canonical fields. The contract
is one shared map, not a repeated Boolean on every record. At index load time, representative evidence IDs are
joined to `03_records/records.parquet`, and the source projection is reconstructed from those declared columns;
this performs no LLM/API call and no normalization. Retrieval
identity/similarity and an optional assay-transfer score are a separate retrieval envelope. Assay-transfer
scoring may consume canonical fields, but its winning-record display must resolve to this source projection.

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
input; v6 does not persist labeling policy, distance, threshold or comparison-value duplicates.

Joint and compound measurements are never split. Papp/efflux, AUC/Cmax ratios, Vmax/Km, statistics,
vector-valued series, condition series, dissolution-model parameters, clock times, and formulation ratios each
remain one intact record. If the complete row cannot be parsed as one measurement it receives no finite scalar,
but remains retrieval-eligible when its structure and mechanism family resolve. Later scalar assay-transfer
datasets must select only records with non-null `absolute_and_continuous_value`.

The v1-v5 directories are historical audit artifacts. The v6 code does not reproduce or rewrite them.
The read-only `audit_starling_v5_v6_migration.py` records both artifact hashes, normalized-record identity
overlap, removal of the heuristic columns, and per-source global attachment status under
`starling_normalized_v6/09_audits/v5_v6_migration/`.

Direct oral-bioavailability evidence is loaded from the single complete pinned source
`data/starling_data/bioavailability_ma/Direct_HF/records.parquet`. It contains all 163,815 rows from
`starling-labs/Oral_Bioavailability` revision `01bbe3ee9cdd3dc081c39973529c9da0c814d465`, with one stable
`source_index` and the twelve original factual columns. It contains no historical kept/dropped partition or
preparation reason. The normalized-v6 builder must not depend on the legacy benchmark JSONLs under
`outputs/chembl_tool/activity_transfer_benchmark/`.

Scientific notation is folded only when explicit, whether the factor appears in the source unit or in an
atomic source measurement. `2.5` with unit `×10^-6 cm/s` and `2.5×10^-6 cm/s` with unit `cm/s` both become
`0.0000025 cm/s`. Shared-factor point estimates such as `175 ± 19 ×10^-6` scale the value and variation
atomically. Bounds, ranges, compounds, conflicting factors, and OCR-compressed `×106` forms are never promoted
to scalars; ambiguous notation is retained and marked rather than guessed. Notation is provenance, not a
permeability comparison stratum.
The folded value, `unit_notation_status`, and `unit_notation_factor` are one atomic normalization result;
source-specific scalar resolvers must preserve all three.

Endpoint-specific conversion is controlled by `starling_normalization_policy.py` and selected only by
`canonical_endpoint`. Unit dimensions may select a reviewed representation, such as mass- versus molar-AUC,
but cannot alter endpoint identity. When no reviewed conversion applies, the mechanically normalized
measurement, unit, and status are retained. Every nonempty unit remains valid and defines a distinct
`source_id + canonical_endpoint + canonical_unit` comparison stratum. Metric-domain gates retain invalid
rows for provenance but exclude them from assay-transfer pairing. The builder still fails on endpoint-inventory
drift in full builds.

The versioned source-aware pair-bucket sidecar is the context-comparability contract. Fa, Fg, and Fh each
use the globally reconciled `global_context` and `global_species_context`; direct HF uses its reviewed report
type and oral exposure has no auxiliary hard boundary. The global fields are joined from the frozen tuple map
after meaning-preserving cleaning. The build fails on missing tuple coverage or conflicting cleaned-key
collisions. Raw context and dose remain source fields, but heuristic assay-system, species, and structured-dose
columns are absent from v6. Null mapped fields use `__unknown__`; unknown matches unknown.
Every fact-valid scalar record belongs to exactly one bucket, and `source_id` prevents cross-source pairing.
The sidecar does not enumerate pairs, create labels, set thresholds, or produce modeling datasets.
The authoritative auxiliary input is
`data_processing/globally_reconciled_auxiliary_value_mapping.json`, version
`starling_auxiliary.globally_reconciled.v1`. It preserves role-bearing species context such as
`dog host; human gene`. The attached global fields are derived comparison metadata and remain excluded from
the source-only LLM projection. No free-text direction, time, absolute/relative, or other derived subtype
creates an additional comparison stratum.

Reviewed metric-prefix reconciliation is part of canonical normalization, not a downstream comparison
projection. `tools/chembl_tool/common/units.py` loads the central versioned
`tools/chembl_tool/common/contextual_unit_policy.json` and accepts a dynamic assay mapping. Every field declared
by a rule must be present and exactly equal; missing differs from explicit null, extra fields are ignored, and
ambiguous rules fail policy validation. Bioavailability always supplies source, canonical endpoint, globally
reconciled assay context, and species context. When a rule matches, `canonical_measurement`, `canonical_unit`,
`finite_scalar_value`, variation, and absolute/continuous value are converted atomically. Source-facing
`measurement_text` and `unit_text` remain unchanged and authoritative for display. Rule ID, policy version,
status, and factor are internal audit fields excluded from the source-only LLM projection. V1 approves only
Caco-2 Fa flux pg/ng reconciliation with unspecified species and human liver-microsome Fh CYP rate pmol/nmol
reconciliation; context-conditioned intestinal-perfusion candidates remain unmerged.

```text
starling_normalized_v6/04_pair_buckets/
  pair_bucket_records.parquet
  pair_bucket_metadata.json

starling_normalized_v6/05_assay_transfer_policy/
  pair_bucket_transfer_policy.json.gz
```

The readable v8 bucket key remains
`source_id + canonical_endpoint + canonical_unit + persisted source-aware fields`.
No downstream policy may add another comparison stratum, refine the key, or rewrite
`03_records`. The authoritative auxiliary surface consists of one public runner,
`data_processing/auxiliary_value_prompts.json` with exactly the original six
Fa/FG/Fh context/species prompts, and the complete frozen
`globally_reconciled_auxiliary_value_mapping.json`. Dose, statistic type,
intestinal site, and other variance candidates receive no LLM mapping.

`build_starling_pair_bucket_transfer_policy.py` produces one combined entry for
every existing pair bucket. One sample is one normalized record, including repeated
records for the same molecule. A bucket first requires at least 25 records. Supported
buckets are then checked against the source-specific raw candidate columns; the
highest-ranked column uses `omega_squared * coverage`, at least two levels, at
least three records per level, and at least 50% coverage. Missing
`qualifying_conditions` is the explicit `__baseline__` category; other missing
candidate values are excluded.

The automatic exclusion gate is:

```text
omega_squared >= 0.20
AND
(median_range / IQR >= 1.00 OR median_range / SD >= 1.35)
```

This is a numerical heterogeneity warning, not a causal claim. A flagged parent is
excluded wholesale; raw candidate values never create child buckets or downstream
labels. Eligible buckets use sample SD with `ddof=1`. Numeric assay transfer is
`abs(left-right) / bucket_SD <= 1`, inclusively, with no endpoint-specific
thresholds or deadband. Ineligible buckets return no transfer label rather than a
negative example.

Each eligible entry also stores 101 local empirical percentile knots. Small buckets
use every unordered record pair; larger buckets use 10,000 deterministic hash-sampled
record pairs. Exact equality maps to zero. These 0--100 values describe unusualness
inside one bucket and are not a shared physical scale across buckets.

The frozen build has 5,221 buckets and 246,385 records. Of these, 365 buckets /
226,913 records meet n=25; the variance gate removes 76 buckets / 27,318 records;
289 buckets / 199,595 records remain assay-transfer eligible. Every eligible
bucket has positive finite sample SD.

The downstream pair-bucket membership and combined transfer policy can be rebuilt
without rewriting normalized-v6 records:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar

/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_transfer_policy
```

The second command performs no LLM call. The compressed policy contains its source
hashes, complete gate contract, exact counts, bucket-local SD curves, and one entry
per v8 key. Downstream consumers join only by `pair_bucket_key`; there is no
record-level endpoint-policy assignment.

Stages 04 and 05 intentionally use the complete unfiltered Stage-03 records. Therefore
held-out Direct-HF measurements contribute to global bucket support, SD, variance-gate,
and percentile statistics. Stage 06 then materializes separate random/scaffold record
views by parent identity, removing matches only from `direct_hf`. Fa, Fg, Fh, and oral
exposure are retained even for held-out molecules. Only these filtered views feed Stage
07 molecule evidence and Stage 08 neighbor indices; no unfiltered evidence/index branch
is published.

Staged normalized-v6 rebuilds publish each stage only after its temporary outputs
have been written and validated. A successful `clean`, `normalize`, or `organize`
rebuild deletes every active downstream artifact, including pair buckets,
the combined assay-transfer policy, analyses, evidence JSONL, and the neighbor index as
applicable. Failed stage computation preserves the prior coherent build. Stage
resume also verifies that the recorded immediate-upstream input hash still
matches the current upstream artifact; a self-consistent but stale downstream
stage cannot be resumed.

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
stage. The legacy factor/direct consumers now reconstruct their views from the same complete Direct-HF
Parquet; v1-v5 outputs remain frozen historical artifacts. Experiments must select either
`starling_normalized_v6/08_neighbor_index/random/` or `.../scaffold/`; `load_index()` detects the
split manifest and joins the corresponding filtered Stage-06 records once at startup. There is no
unfiltered normalized-v6 neighbor index. Normalized-v5 is not rewritten by the v6 builder.

## Commands

Build a new complete-Parquet Starling factor index:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v4 \
  --workers 32
```

Run the shared reasoning pipeline with that index:

```bash
/data1/joseph/miniconda3/condabin/conda run -n txagent-glm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v4/starling_factor_neighbor_index.pkl \
  --api-key-env GLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --batch-id bioavailability_ma_paper_starling_<date>
```

正式 paper run 只使用上述 `outputs/paper/` index。`outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/`
下的 task-level builder 默认目录只用于临时开发，不得把历史 index 复制或软链接到正式实验路径；运行前应检查
meta 中 `index_version`、`canonical_contract_version`、canonical/residual SHA-256、`scope`、
`evidence_content` 和五个稳定 group ID。

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
