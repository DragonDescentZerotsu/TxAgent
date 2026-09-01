# Bioavailability_Ma Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v4`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：2,091
- record agreement threshold：70%
- valid / test target：209 / 209
- 原始 label-conflict parents：368
- majority 恢复：262
- agreement/tie 拒绝：106

## Splits

| split | train | valid | test | valid Y=0 / Y=1 | test Y=0 / Y=1 | scaffold pairwise overlap |
|---|---:|---:|---:|---:|---:|---:|
| random | 1,673 | 209 | 209 | 58 / 151 | 58 / 151 | 144 |
| scaffold | 1,673 | 209 | 209 | 61 / 148 | 65 / 144 | 0 |

## Source-row rejection

```json
{
  "interpretation_altering_qualifying_conditions": 12911,
  "missing_bioavailability_value": 14,
  "nonhuman_or_unresolved_population": 77589,
  "numeric_interval_crosses_20_percent_threshold": 738,
  "numeric_value_out_of_percent_range": 635,
  "qualitative_value_not_threshold_anchored": 903,
  "relative_not_absolute_bioavailability": 191,
  "unmapped_or_ambiguous_qualitative_value": 1127
}
```

完整 provenance 见 `molecule_labels.jsonl`；所有原始 conflict parent、未达到 agreement 的拒绝
以及 source-row rejection 示例分别见 `conflicting_molecules.jsonl`、
`rejected_parent_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

每种构造方法的 `heldout_molecule_labels.jsonl` 是 valid+test union retrieval 泄漏隔离清单。
现有 full-source Starling index 不能直接用于 valid 或 test。
