# Bioavailability canonical-direct v2 数据质量报告

生成日期：2026-08-02

## 数据契约与粒度

该版本把固定 revision 的 `starling-labs/Oral_Bioavailability` 与本地
`Oral_AUC-Cmax-Exposure/extractions.parquet` 分成两条互斥的 active 数据线：

- `direct_claims.parquet`：gold builder 与 agent direct evidence 共用的一条 canonical direct-F claim 表；
- `../oral_exposure_residual_v2/exposure_records.parquet`：AUC/Cmax、relative bioavailability 和没有
  absolute 锚点的 ambiguous bioavailability，仅作为 inference-time exposure/context evidence。

原始 HF snapshot 与原始 local parquet 均保留，不原地修改。

## 分区与去重结果

| 检查 | 结果 |
|---|---:|
| HF snapshot rows | 163,815 |
| HF direct source rows（有效结构） | 111,848 |
| Local input rows | 119,192 |
| Local absolute rows 从 residual 移出 | 3,920 |
| Local absolute rows（有效结构，进入 canonical 候选） | 3,712 |
| Local absolute rows（无效结构，隔离到 rejected audit） | 208 |
| Local residual rows | 115,272 |
| 跨 HF/local 一对一 matched claims | 2,480 |
| 去重前 direct source rows | 115,560 |
| canonical direct claims | 113,080 |
| HF-only claims | 109,368 |
| HF+local claims | 2,480 |
| local-only claims | 1,232 |

Local 分区严格满足 `119,192 = 3,712 + 208 + 115,272`。Residual 中
`partition=canonical_absolute_bioavailability` 的行数为 0，canonical claim ID 重复数为 0。

## Local bioavailability 分区

| Partition | Rows | Active role |
|---|---:|---|
| `canonical_absolute_bioavailability` | 3,712 | canonical direct source |
| `canonical_absolute_rejected_structure` | 208 | rejected audit，不进入任何 active dataset |
| `residual_relative_bioavailability` | 1,189 | exposure/context proxy |
| `residual_ambiguous_bioavailability` | 3,533 | exposure/context proxy |
| `residual_non_bioavailability_exposure` | 110,550 | exposure/context proxy |

Local row 只有出现明确 `absolute bioavailability` wording 或 oral/IV anchor，且没有 relative signal，才进入
canonical。这个规则优先 label precision；没有 absolute anchor 的记录即使字段名为 `bioavailability` 也不会进入
gold source。

## Gold benchmark 变化

| Metric | 旧 mixed-source strict | canonical-direct strict | 当前 70% record agreement |
|---|---:|---:|---:|
| binary parents | 1,862 | 1,828 | 2,092 |
| rejected parents | 385 | 370 | 106 |
| random valid / test | — | — / 365 | 209 / 209 |
| scaffold valid / test | — | — / 365 | 209 / 209 |
| random valid / test Y=0,Y=1 | — | — / 99,266 | 58,151 / 58,151 |
| scaffold valid / test Y=0,Y=1 | — | — / 116,249 | 61,148 / 65,144 |

两套当前 split 的 train/valid/test parent identity 两两 overlap 均为 0；scaffold split 的三个 subset
之间 scaffold overlap 也全部为 0。

## Agent evidence 一致性

Gold summary 与 direct/full agent index meta 都记录同一份 canonical source：

```text
direct_claims.parquet SHA-256:
045261cbda785092143eeadd636f78399f7b02f951b23480b16fb8dde22661c5

residual exposure SHA-256:
81c45f9dc035ed2b066b19a569eb80ef1f806d7a0f421956ea92b2dadb779bda

canonical merge manifest SHA-256:
b986a214491f3b04c6f3ae6fc71cbd4eacad08d390b9b62de53f953966b0a257
```

Full agent evidence 包含五个稳定 mechanism groups；canonical direct 与 residual exposure 不再从同一个
parquet 的重叠 profile 重复加载。

## 风险与边界

- Local schema 没有独立的 `bioavailability_report_type`，因此 absolute/relative 分区依赖保守文本规则。
- `residual_ambiguous_bioavailability` 可能包含未明确写出 oral/IV comparator 的真实 absolute F；当前版本选择
  不将其用于 gold，以降低误标风险。
- Claim dedup 是同 parent+PMID 内的一对一跨来源匹配，要求两条 claim 的完整 numeric point/interval 都在
  20% threshold 同一侧，且距离不超过 1 percentage point；跨过 threshold 的 interval 不参与匹配。它不会
  合并同一来源内部的 records。仅当至少一侧没有可解析 numeric value 时，才允许 identical normalized
  support text 作为后备匹配条件。
- 旧 1,862-parent mixed-source 和 1,828-parent strict-conflict benchmark 的 baseline、agent 和图表必须作为
  historical lineage 保留，不得与当前 2,092-parent 版本混表。

机器可读规则、计数、路径与文件哈希以 `merge_manifest.json` 和 residual
`partition_manifest.json` 为准。
