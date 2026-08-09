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

当前 paper-facing canonical-direct v2 build 位于：

```text
data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold/
```

共有 2,092 个 binary parents；train/valid/test 为 1,674/209/209。旧
`data/processed_starling/Bioavailability_Ma/{random,scaffold}` 属于 `record_agreement70_split811_v1`
historical comparison。公共构建/审计协议见
`tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md`。当前 v2 正式运行前必须按 scaffold 的
valid+test union `heldout_molecule_labels.jsonl` 重建 train-only retrieval index；historical v1 的两个 split
仍各自使用对应 union，不能跨 lineage 复用。

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

## Commands

构建 Starling index：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --out-dir outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v2 \
  --workers 32
```

用通用 pipeline 跑 Starling source：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_full_v2/starling_factor_neighbor_index.pkl \
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
