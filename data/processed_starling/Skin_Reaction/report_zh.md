# Skin_Reaction Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v2`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：1,900
- test target：380
- 因 parent-level 标签冲突而丢弃：1,021

## Splits

| split | train | test | test Y=0 / Y=1 | scaffold overlap |
|---|---:|---:|---:|---:|
| random | 1,520 | 380 | 129 / 251 | 72 |
| scaffold | 1,520 | 380 | 117 / 263 | 0 |

## Source-row rejection

```json
{
  "inconclusive_outcome": 181,
  "invalid_or_unresolved_smiles": 16731,
  "outside_tdc_skin_sensitization_scope": 5049
}
```

完整 provenance 见 `molecule_labels.jsonl`；冲突分子和 source-row rejection 示例分别见
`conflicting_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

`random/test_molecule_labels.jsonl` 与 `scaffold/test_molecule_labels.jsonl`
分别是两套 retrieval 泄漏隔离清单。现有 full-source Starling index 不能直接用于这些 test。
