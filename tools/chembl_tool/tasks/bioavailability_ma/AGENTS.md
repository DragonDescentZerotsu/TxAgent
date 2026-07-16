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

它使用 shared profile ingestion：

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

## Commands

构建 Starling index：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --workers 32
```

用通用 pipeline 跑 Starling source：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_factor/starling_factor_neighbor_index.pkl \
  --api-key-env GLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --batch-id bioavailability_ma_paper_starling_<date>
```

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
then the shared identity exclusion policy, then at most 50 survivors, then cached model reranking, then the
existing final `top_k_per_group`. It never backfills below the raw top 100.

The current cache is the retrieval-agnostic `flat_v2` store. It contains one stable five-source record
catalog, eight exact condition manifests (validation/test x four/five-source x operational/parent-disjoint),
an audit mapping, and an append-only SQLite score store. Score identity includes only the exact prompt hash,
immutable model revision, scoring contract, and template hash. Catalog, split, source composition, and
identity policy are audit/join metadata and never change score identity.

Both direct-bioavailability source schemas are normalized: HF Oral Bioavailability records use
`oral_bioavailability_value_percent`, while direct rows from Oral_AUC-Cmax use
`reported_value`/`reported_units`. This normalization is only for assay-transfer prompts; it does not change
the Starling evidence indexes or ordinary Morgan retrieval.

Attached numeric source-record examples become candidate-scoped prompt records; qualitative-only candidates
remain in the retained pool but do not receive fabricated prompts. Scoring mirrors training evaluation:
append `(A)` and `(B)` with no leading space, find their first divergent token, and softmax the two next-token
logits.

Build the flat artifacts and migrate the two legacy caches without inference:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.precompute_flat_assay_transfer_cache --prepare
```

Populate only missing scores with two independent BF16 model replicas using the training environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  /vast/projects/myatskar/design-documents/conda_env/open_rlhf_intern/bin/python \
  -m tools.chembl_tool.tasks.bioavailability_ma.precompute_flat_assay_transfer_cache \
  --infer --devices 0,1 --batch-size 16 --force-model-download
```

Run all count, migration, composition, and independent retrieval checks:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.precompute_flat_assay_transfer_cache \
  --verify --strict-retrieval
```

Reasoning batches only read the cache and run a complete coverage preflight before starting subprocesses:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --retrieval-source starling \
  --experiment-mode full_mechanism \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_factor/starling_factor_neighbor_index.pkl \
  --retrieval-reranker assay_transfer \
  --rerank-candidate-manifest outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/assay_transfer_rerank/flat_v2/manifests/test__five_source__operational.jsonl \
  --rerank-cache-mode read_only
```

Flat catalogs require the exact condition manifest at runtime; concept+molecule lookup is intentionally
rejected because it could join fifth-source records into a four-source run. Reranking scores and winning
record IDs remain audit-only retrieval metadata and must not enter group or final reasoning prompts.
