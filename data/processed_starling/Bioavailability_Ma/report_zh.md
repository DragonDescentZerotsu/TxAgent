# Bioavailability_Ma Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v2`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：1,862
- test target：372
- 因 parent-level 标签冲突而丢弃：385

## Splits

| split | train | test | test Y=0 / Y=1 | scaffold overlap |
|---|---:|---:|---:|---:|
| random | 1,490 | 372 | 99 / 273 | 94 |
| scaffold | 1,490 | 372 | 113 / 259 | 0 |

## Source-row rejection

```json
{
  "interpretation_altering_qualifying_conditions": 13347,
  "invalid_or_unresolved_smiles": 185,
  "missing_bioavailability_value": 308,
  "non_bioavailability_exposure_measure": 110550,
  "non_direct_oral_bioavailability_report_type": 51570,
  "nonhuman_or_unresolved_population": 82257,
  "numeric_interval_crosses_20_percent_threshold": 742,
  "numeric_value_out_of_percent_range": 808,
  "qualitative_value_not_threshold_anchored": 912,
  "relative_not_absolute_bioavailability": 68,
  "unmapped_or_ambiguous_qualitative_value": 1164,
  "unsupported_bioavailability_unit": 467
}
```

完整 provenance 见 `molecule_labels.jsonl`；冲突分子和 source-row rejection 示例分别见
`conflicting_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

`random/test_molecule_labels.jsonl` 与 `scaffold/test_molecule_labels.jsonl`
分别是两套 retrieval 泄漏隔离清单。现有 full-source Starling index 不能直接用于这些 test。
