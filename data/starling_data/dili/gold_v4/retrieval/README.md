# DILI gold_v4 对应的 L1/L2 records

本 release 与 `data/starling_data/dili/gold_v4/source_votes.jsonl` 精确绑定。
2026-09-09 已通过 `canonical_publication.json` 切换正式 benchmark、records、catalog 和索引。
`data/conditioned_benchmark/DILI` 现在使用 gold_v4；旧输入及检索索引保存在 `previous_active/`。
`manifest.json` 是发布前的校验快照，当前发布状态以 `canonical_publication.json` 和 current registry 为准。

| 层 | 旧 records | 新 records |
|---|---:|---:|
| L1：实际投票代表 | 4,727 | 108,437 |
| L2：非投票的直接相关证据 | 270,203 | 166,495 |

L1 的 source 方向为正 96,228 / 负 12,209；这是源 records 数，非 benchmark molecule-condition 数。
L2 包括同一论文单位的其他支持段落、内部冲突及其他未投票的直接相关记录，不强制赋统一二元标签。
41,751 条非代表支持 records 不等于完全重复项，`source/vote_support_links.parquet` 保留其代表 UID。
构建时对全部原始内容字段、PMID 和分子身份做精确去重，只忽略抽取 UID、extraction_id 和段落位置；
完全重复的副本直接从工作 records 删除，优先保留 actual voter，不降到 L2。此次全量检查发现 0 个完全重复副本。
父分子最后因一致性不足未得到确定 gold，不撤销其已输出 source vote 的 L1 成员身份。
本 gold 的投票单位为论文来源/已核实研究＋分子＋条件并去除精确重复原文；不是全量独立实验核验。

实际变更：105,809 条 L2→L1；2,101 条旧 L1→L2；另 2 条新投票代表从 L4→L1。
除此以外 L3–L7 的归属不变。全 1,645,109 条记录均保留，raw payload、UID、分子身份和原始科学字段不变。
本次没有模型调用，也未改变 gold 标签、条件和 split 分配。

## 文件

- `source/records.parquet`：包含全部层的完整 records；按 `progressive_level` 读取 L1/L2。
- `source/exact_duplicate_removals.jsonl`：被移除副本到保留 UID 的映射（本轮为空）。
- `source/actual_voter_membership.jsonl`：精确 L1 UID、原 payload hash、来源投票信息。
- `source/record_audit.parquet`、`source/membership_changes.jsonl`：全量审计和逐条变更。
- `source/membership_refresh_receipt.json`：原始字段守恒、成员一致性和 SHA-256。
- `tables/source_record_level_membership/`：可直接读取的 Parquet 数据集，包含层、原文和卡片指纹。
- `catalog/`：共享 family catalog；`indices/{scaffold,random}/`：新 heldout union 对应的索引。
- `heldout/{scaffold,random}.jsonl`：给同一 gold heldout cohort 添加显式泄漏身份字段；
  `heldout/receipt.json` 证明有序分子/标签/row IDs 与原文件一致。原 gold 文件未改写。
- `indices/*/identity_validation.json`：完整索引卡片的 heldout 排除检查和七层检索验证。
- `tables/*/indexed_representative_cards.parquet`、`tables/validation.json`：索引卡片引用及全部来源链接校验。

源层总数是 heldout 排除前的数目；实际每种 split 可检索的数量见 release manifest。
每种索引只按 valid+test 的分子泄漏组排除 L1（`group_id=Group.dili_actual_voter`），
与 BBB/Bioavailability/Skin 的 L1 预排除方式一致。L2 不做整个 heldout union 的预排除；
检索时对所有层按当前 query 执行 scaffold-disjoint 或 parent-disjoint。
本 release 不授权沿用旧 predictions。当前运行时使用 L1 index 预排除和 query-time disjoint，
已撤下旧 DILI 专属的全 heldout 文本过滤；源端已确认的身份错误排除保持有效。
新 valid/test progressive/full-flat 四个 runs 见 current_conditioned_results.json 的 `dili_gold_v4_scaffold_4_2`。

## 重现

在当前仓库与既有 vllm 环境执行 `sh data/starling_data/dili/gold_v4/retrieval/rebuild.sh`。
脚本使用共享 source membership、catalog、index、validation 与 export 模块，不发起模型审阅或实验。

## 完成校验

两种 L1-only 索引已通过完整身份与七层检索检查；2,295,637 张索引卡片来源链接全部通过，缺失为 0。
校验要求 heldout L1 为 0，同时确认 heldout L2 仍在索引中，并检查七层 query-time disjoint。

| 范围（代表卡片压缩前的 records） | L1 | L2 |
|---|---:|---:|
| 全源 | 108,437 | 166,495 |
| scaffold 仅 L1 heldout 排除后 | 83,722 | 166,495 |
| random 仅 L1 heldout 排除后 | 69,267 | 166,495 |

聚合后的索引代表卡片数与上述源 records 数不同，详见 `manifest.json` 和各 index validation。
相关 source、identity 和 level 单元测试共 31 项通过；未运行模型实验。
