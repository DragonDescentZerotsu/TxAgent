# Current Starling retrieval and collaborator data

更新时间：2026-09-12。

本页是 BBB、Bioavailability、Skin、Ames、DILI 和 Carcinogens progressive retrieval 的唯一当前数据说明。它同时说明如何恢复、重建、
验证和分享每个 level 的 records。日常复现不需要另一个 TxAgent checkout，也不需要在旧的 `v1`、`v2`、
`v5` 或 `v6` 目录中选择输入。

## One current pipeline

```text
repo-packaged Stage-03 canonical records
  -> task voter-purity overlay
  -> record-family catalog (source_group_id -> family_key -> level)
  -> scaffold/random heldout-filtered index
  -> per-query append-only progressive retrieval
```

只有两个名字表示分类 identity：

- `source_group_id`：overlay 中每条 record 的 source-local 分类，也就是当前 `group_id`；
- `family_key`：跨 Starling/ChEMBL 对齐的 biological family。

`level` 只是某个 source config 的累计展示顺序，不能用于跨 source join。旧 catalog/trace 中的
`endpoint_group` 等价于 `family_key`；`family_id` 和 `Mechanism.tier_N` 只是冻结的 legacy output ID。

两个机器可读清单固定整条 lineage：

```text
artifacts/chembl_tool/starling/current_records/manifest.json
tools/chembl_tool/paper_experiments/current_starling_retrieval.json
```

第一个清单保存六个 task 当前唯一 canonical snapshots 的压缩分片、完整 inventory 和 SHA-256。第二个清单
固定 overlay、catalog、scaffold/random index 的路径、参数和 SHA-256。大型 records 恢复到被 Git 忽略的：

```text
outputs/chembl_tool/starling/current_records/<task>/03_records/
```

统一重建入口从该位置读取；找不到或 hash 不符时直接失败，不会回退到个人目录。
Ames 的 source adapter 在 `data/starling_data/ames/canonical_v1/` 生成带审查结果的 canonical records；
其冻结副本已经包含 family assignment，`overlay_is_canonical=true`，无需再生成一份 purity overlay。
统一验证同时检查 source adapter 的 records 与冻结副本字节一致；`restore-records` 会补齐缺失的
大型 canonical records 文件，其余 raw、votes 和 review ledgers 保留在原有 provenance 目录。

这里的 Stage-03 分片只是为了绕开 GitHub 普通 Git 的 100 MB 单文件限制，并由统一入口自动恢复；它不是
交给协作者阅读的“分享包”。协作者实际浏览和交换的是下文 `current_level_records/` 中直接由 Git 跟踪的普通
Parquet 文件，不需要解压，也不需要安装 Git LFS。

## Restore, rebuild, and verify

在项目根目录执行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval \
  restore-records

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval \
  build --workers 8

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.rebuild_current_starling_retrieval \
  verify
```

`build` 重建原三项 overlays，直接采用 Ames、DILI 和 Carcinogens 已审核的 canonical snapshots，构建六个 catalogs 和十二个 split indices，然后执行 hash gate。也可单独运行
`build-overlays`、`build-catalogs` 或 `build-indices`。clean-room 检查可通过
`--tasks dili carcinogens` 仅处理指定任务；使用
`--output-root /local/tmp/<name>` 避免覆盖正式资源。若已恢复的 Stage-03 目录存在但 hash 不符，程序默认拒绝
覆盖；确认该目录可替换后使用 `restore-records --force`。

DILI 和 Carcinogens 的最终 reviewed snapshots 均为 `data/starling_data/<task>/retrieval_final/source/`。
DILI 纳入已完成 v5 的修复；Carcinogens 把 R18 的全部已修改卡片固化到源行。
`publication.json` 记录最终 source/catalog/scaffold/random 校验；旧实验仍绑定其实际输入。
恢复包同时还原超过 Git 单文件限额的 gold/audit JSONL，保持原始路径与字节哈希。

DILI/Carcinogens 同样采用 `overlay_is_canonical=true`，冻结全量 reviewed records，
并打包 actual-voter ledger、record audit 和 unique-SMILES identity cache。统一导出将
`level_assignment_reason` 映射到公共 `source_family_purity_reason`；缺失 pre-overlay group
保持 null，不编造源分类。raw JSON 和永久 UID 不变。gold、TDC 和 split 合同见
[NEW_TASK_SOURCE_DATA.md](../common/starling/NEW_TASK_SOURCE_DATA.md)。

## What is frozen and what changed

Stage-03 是当前可复现边界。它已完成 raw source 清洗、逐 source 列映射、canonicalization、organization 和
deduplication，并保留 `support_text`、measurement、context、assay provenance、duplicates 和 exclusions。
不同 Starling run 的原始 Parquet 没有原位修改；normalization 生成了新的 canonical records。current manifest
还固定生成时 column contracts 的 SHA-256。

原三个任务的 purity 修正也不改 Stage-03。它生成新的 overlay，并用同一个 `canonical_record_id` 记录：

- 新 `group_id`；
- `source_family_original_group_id`；
- `source_family_purity_reason`；
- `source_family_purity_version`。

因此“移动 record”是可审计地重新赋予 `group_id`，不是在文件间剪切行，也不会改 endpoint、value、unit 或
`support_text`。Bioavailability 另有一个明确的 pre-overlay source repair：排除 6 条已复核的 nitrendipine
structure-name mismatch，并从已有结构化字段补充 nondirect assay context；Stage-03 输入本身仍保持不变。

Ames 的原始四批数据、名称核验和 record audit 保留 source-native 格式；导出的 canonical 字段、
family/level 字段和 card-link keys 与其余任务完全相同。Ames 没有 pre-overlay group，导出中的
`source_family_original_group_id` 为 null；其审核历史使用 `record_audit.parquet` 与 payload-hashed review ledgers。

当前 purity 结果为：

| task | current levels | current correction |
|---|---|---|
| BBB | L1 direct voter; L2 near-direct; L3 passive; L4 efflux; L5 influx | L1 是 exact current voter-record membership；关键最终修正将 5,305 条 direct-like nonvoters 移到 L2。|
| Bioavailability | L1 direct voter; L2 nondirect bioavailability; L3 oral AUC/Cmax; L4 Fa; L5 Fg; L6 Fh | 50,279 条 nonvoters 从 L1 移到 L2，10,092 条 accepted vote records 提升到 L1。|
| Skin | L1 direct voter; L2 near-direct outcome/classification; L3 sensitization mechanism | L1 使用稳定 acquisition-row key 做 exact voter membership；L2 保留 nonvoter measured outcomes 及 predicted/defined-approach overall classifications；L3 保留 experimental/predicted AOP 和有实质内容的 unspecified mechanisms。photo/light-dependent、irritation-only、non-contact severe cutaneous reaction、空/无关 records 从三层全部排除。|

逐 record 原因保存在 overlay audit columns；聚合统计和 hard gates 保存在各 overlay 的 `manifest.json`。这些
overlays 不改 gold labels。

Skin v5 的 semantic partition audit 冻结在
`data/starling_data/skin_reaction/canonical_sensitization_v4/`。它由
`tools/chembl_tool/tasks/skin_reaction/build_canonical_starling_source.py` 从两个 immutable raw acquisition
parquet 重建；paper-facing overlay 读取该 audit，但不直接把 canonical direct/AOP parquet 当成 progressive
index。日常重建仍只调用本页上方的统一 `rebuild_current_starling_retrieval.py` 入口。

## What “records available at a level” means

这个短语可能指三种不同对象，分享时必须明确是哪一种。

### 1. Static source-record membership

先筛选 `retrieval_eligible=True`，再用 catalog 的 `source_group_id -> family_key -> level` 映射得到每条 source
record 的 level。同一个 physical assay 可含多个不同 family 的 records，因此可能向多个 levels 提供卡片。

当前 pre-split membership：

| task | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 8,592 | 268,379 | 14,189 | 163,672 | 43,724 | - |
| Bioavailability | 20,538 | 152,350 | 104,174 | 73,913 | 23,511 | 60,986 |
| Skin | 42,435 | 12,522 | 12,010 | - | - | - |
| Ames | 3,333 | 138,571 | 236,309 | 496,880 | 424,491 | - |

DILI/Carcinogens 的七层 pre-split membership 为：

| task | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DILI | 108,436 | 166,498 | 141,940 | 75,971 | 294,048 | 196,583 | 402,992 |
| Carcinogens | 89,617 | 405,813 | 750,569 | 650,547 | 749,469 | 218,187 | 286,938 |

这些不是某个 query 最终看到的卡片数。

### 2. Split-specific indexed representative cards

构建 scaffold/random index 时，按各 task 的冻结 scope 删除 valid+test heldout parents 的 outcome records：
BBB/Bioavailability/Skin 删除 direct family；Ames 删除 `heldout_filter_scope=bacterial_outcome` 的全部 L1/L2；
DILI gold_v4 只删除 `group_id=Group.dili_actual_voter` 的 L1；Carcinogens gold_v4 只删除
`group_id=Group.carcinogenicity_direct` 的 L1。两者都保留 L2，并使用冻结 tautomer identity cache，
所有层在 query-time 应用 scaffold/parent disjoint。
其余机制 records 仍可保留。随后按 physical assay×molecule 聚合，每组最多
确定性保留 3 张 representative record cards。`support_text` 完整保留，`max_support_text_chars=0` 表示不截断。

这是某个 split 实际可供 retrieval 的候选池，不是 overlay 所有行的副本。Git 分享表用
`molecule_id + card_fingerprint_sha256` 保存紧凑 card 引用，并连接到 source membership 取得 endpoint、value、
context 和 support text；完全相同的 visible card 若来自多条 canonical records，ledger 会保留所有匹配
provenance。导出器会硬校验每张 indexed card 都至少连接到一条 source record。

### 3. Per-query model-visible cards

每一层在截至该层的 cumulative family pool 内做全局 Morgan molecule-similarity retrieval，minimum similarity
为 0.3。scaffold 使用 `scaffold_disjoint`，random 使用 `parent_disjoint`。L1 最多选择 10 个 molecules，默认
每个 molecule 最多 4 张卡；后续每层最多新增 3 个 molecules、补充 3 个 active molecules，并给每个被选
molecule 新增最多 2 张卡。已有卡 append-only 保留。

某个 query 最终看到的精确内容只能从对应 run 读取：

```text
<run-root>/<task>/queries/query_idxNNNNN/levels/level_N/prepared.json
```

`active_evidence` 是截至该层的完整模型可见集合，`new_card_ids` 是本层新解锁的卡。

## GitHub collaborator dataset

仓库直接追踪唯一 current snapshot，不要求单独压缩包或 Git LFS：

```text
artifacts/chembl_tool/starling/current_level_records/
  README.md
  manifest.json
  <task>/source_record_level_membership.parquet
  <task>/scaffold/indexed_representative_cards.parquet
  <task>/random/indexed_representative_cards.parquet
```

大于 90 MB 的 source membership 自动保存为同名目录中的普通 Parquet parts，所有任务仍使用同一 schema。
合作者 clone/pull 后可直接用 pandas、DuckDB 或 PyArrow 浏览：读取 manifest 中的 `path`，它可以是
单文件或目录；目录的各 part SHA-256 在 `parts` 中列出。`manifest.json`
记录 row count、逐 level count、文件大小、SHA-256 和 card-link gate。为避免 scaffold/random 重复大段
support text，indexed 表只保存 card 引用，完整 record 字段保存在 source membership 表。per-query cards
仍由具体 run 的 `prepared.json` 表示。

重建分享数据：

```bash
python -m tools.chembl_tool.paper_experiments.export_current_starling_level_records
```

正式输出目录必须完整导出四个任务和两种 split；部分导出须指定独立 `--output-dir`，避免覆盖完整清单。

导出前会核验 canonical records、overlays、catalogs 和八个 indices 的 current hashes。未来 lineage 改变时应
原位重建，并在 manifest/Git history 中记录变化，不能新增 `v2`、`final2` 等平行副本。

## Historical boundary

Stage-03 之前的 raw ingestion 与 normalized-v7 历史实现来自
`origin/joseph@70750c42035d6bde5ce9504211c4e22ac57e27f8`，只说明 frozen snapshot 的来源，不是运行依赖。
若以后从新 raw run 重新生成 Stage-03，必须作为 source-ingestion migration 审查。

早期流程从 source-specific 列名逐步发展到 canonical records；随后从 assay-level earliest-level assignment 改为
record-level family assignment，并通过 BBB v6、Skin v5 和当前 Bioavailability repair/purity 收紧 L1。2026-09-03
进一步将运行时配置收敛为 `source_group_id -> family_key -> level`，并 clean-room 重建所有 downstream artifacts；
全部 contract hashes 一致，因此这轮命名/API 整理没有改变 source membership 或 retrieval rule。
2026-09-04 的 Skin strict-scope migration 确实改变了 gold、family membership 和两个 split-specific indices；随后
最终可复现性检查又识别出 37 条 MDAM nonvoter final outcomes，gold/split 不变但 L2 和两个 indices 再次变化。
matched baselines 仍为 current；scaffold-valid 2/1、4/2、8/4 分别有 1/2/3 个 queries 等待 targeted replay。
精确 selected-surface audit 见 `receipts/skin_scaffold_valid_mdam_family_rebuild.json`。

旧版本号仍可能出现在已完成 run、receipt 和结果路径中，因为它们是 provenance，不是当前输入选项。当前选择
只由上述两个机器可读清单决定；当前实验结果与 freshness 状态只由
`current_conditioned_results.json` 和 `RESULTS.md` 报告。

大型输入校验复用 `common/build_runtime.py` 的 inode/size/mtime/ctime 与同一启动周期绑定的
摘要缓存；level 导出复用该模块的内容寻址 NVMe 输入缓存，避免反复读取网络盘。首次输入仍完整
校验 SHA-256。普通 Parquet 表超过 90 MB 时自动拆成同 schema 分片，包括 split 卡片引用表。

最终 DILI/Carcinogens 源的逐行修改、冻结 gold 哈希和两种索引校验统一见
`data/starling_data/<task>/retrieval_final/publication.json`。旧 gold_v4 首次发布记录
只作为历史来源；最终检索使用本页的 `retrieval_final` 快照。
