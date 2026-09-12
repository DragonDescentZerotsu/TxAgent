# DILI Starling-only gold v4

2026-09-09。复用已完成的全 base 正负方向审阅与有限的条件/出处人工核查。
本次构建没有新模型调用，也不使用已取消的部分 provenance review 作为筛选器。

| 单位 / 阶段 | 正类 | 负类 | 合计 |
|---|---:|---:|---:|
| 原始 records 的已审方向 | 156,476 | 20,573 | 177,049 |
| 身份合格的 binary records | 142,581 | 18,213 | 160,794 |
| 去重后可投票的论文/已核实研究单元 | 96,228 | 12,209 | 108,437 |
| 全部已确定 molecule–condition gold | 3,193 | 874 | 4,067 |
| 可三路分配的 benchmark gold | 3,158 | 866 | 4,024 |
| 条件组不足以三路分配，仍保留 gold | 35 | 8 | 43 |

`gold_labels.jsonl` 是完整 gold，单位为分子＋条件，包含 `drug`、`Y`、条件、身份、正负票数、
`split_eligible`。完整来源链位于 `gold_label_provenance.jsonl`；所有 UID 可回到冻结 raw Parquet。
`gold_rare_conditions.jsonl` 保留上述 43 条，不把 split 覆盖问题当成标签不确定。
完整 gold 有 69 个条件组合，split cohort 有 38 个；本次覆盖全 base，不限于旧 gold 的条件集合。

| split | scaffold 正 / 负 | random 正 / 负 |
|---|---:|---:|
| train | 2,511 / 709 | 2,526 / 694 |
| valid | 327 / 75 | 316 / 86 |
| test | 320 / 82 | 316 / 86 |

当前正式 split 位于 `data/conditioned_benchmark/DILI/{scaffold,random}/`；
`source_only_benchmark/DILI/` 是这次 gold 构建的来源副本，不是另一个实验输入入口。
当前检索源已冻结到 `data/starling_data/dili/retrieval_final/`，包含后续 v5 修复。
`retrieval/README.md` 只记录初次 gold_v4 发布时的 L1/L2 变化，不能代表最终源的层级计数。
最新实验和五种 baselines 以共享 `current_conditioned_results.json` 为准。

## 标签规则与审阅边界

- 只使用 Starling base 的已完成方向账本 `data/starling_data/new_tasks_gold_audit/targeted_review_v2/dili_source_labels.parquet`
  （实际仓库路径见 `summary.json`）。不加入任何 TDC label，不用旧正类偏重的 eligibility 规则再挡一次负类。
- 来源论文–分子–条件最多一票；已核实的错误 PMID 修正后合并，相同原文跨 PMID 另作精确去重。
  论文单元内有正负冲突则不投票。默认条件至少 70% 一致，有条件至少 60%，排除 tie。
- **这是论文来源共识 gold，不是已全量核实独立实验的临床专家 gold。** Review/reference assertions 可以参与；
  多篇综述仍可能间接引用同一实验。原文方向沿用上一轮审阅，未被本次条件规则重新判定。
- 默认组聚合常规用药、人群/普通适应证和制剂途径描述；已审 typed 条件及明确的年龄、过量、基线疾病等
  用粗粒度规则归并。未解析的额外限定保留 raw metadata，不作为另一个拒绝门槛，也不生成任意自由文本条件。
  这一步是规则规范化，不能描述成全量逐条人工确认。
- 原来逐条核实的制剂和条件归属修正继续生效；23 条原错误出处发现扩展到同分子/同错误 PMID 的 44 条
  paragraphs。能明确修复的按已核实出处修复，其余保留待定，详见 `applied_citation_corrections.jsonl`。

## 未定标签与排除账本

有可投票来源但跨论文冲突/不足一致性的 molecule–condition：498 个；
所有合格论文单元都存在内部正负冲突、因此无剩余可投票来源的 molecule–condition：22 个。
合计 520 个，分别在 `unclassified_parent_conditions.jsonl`、`only_conflicted_parent_conditions.jsonl`。
这些数目不与 raw records 数混用。原始不确定/混合方向、身份待定、材料/结构问题、错误出处待定等
另见 `record_dispositions.parquet`、`summary.json`，全 189,172 条原始记录均有去向，不删除原文。

## 重现与校验

在仓库根运行，使用现有 vllm 环境：

```sh
PYTHONPATH=. /data1/tianang/anaconda3/envs/vllm/bin/python -m tools.chembl_tool.tasks.dili.build_reviewed_starling build
```

不需要运行 `review`；该入口已由取消 receipt 阻止重启。
`validation.json` 已验证 source label 不变、gold 多数票重算、两种 split cohort/labels 完全一致、
两种 split 的分子泄漏组交集为 0、scaffold split 的 scaffold 交集为 0、全部入选条件三路覆盖，
以及构建当时的 canonical split 字节不变。该 validation 记录初次构建的验证范围；
后续正式发布和 source 清理分别保留独立 receipt。
输入/输出 SHA-256 与构建器 hash 记录在 `summary.json`、各 split summary、`validation.json`。
已取消模型任务的日志和部分输出只保留为历史审计，不参与上述 gold。

2026-09-09 的初次正式发布收据为 `retrieval/canonical_publication.json`。
2026-09-12 的最终 source、catalog 和两种索引见 `../retrieval_final/publication.json`。
日常恢复无需重跑 gold 构建或模型审阅；统一命令见
`tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md`。
