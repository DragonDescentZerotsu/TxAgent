# Skin_Reaction Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v4`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：2,456
- record agreement threshold：70%
- valid / test target：245 / 245
- 原始 label-conflict parents：1,021
- majority 恢复：556
- agreement/tie 拒绝：465

## Splits

| split | train | valid | test | valid Y=0 / Y=1 | test Y=0 / Y=1 | scaffold pairwise overlap |
|---|---:|---:|---:|---:|---:|---:|
| random | 1,966 | 245 | 245 | 75 / 170 | 75 / 170 | 129 |
| scaffold | 1,966 | 245 | 245 | 74 / 171 | 71 / 174 | 0 |

## Source-row rejection

```json
{
  "inconclusive_outcome": 181,
  "invalid_or_unresolved_smiles": 16731,
  "outside_tdc_skin_sensitization_scope": 5049
}
```

完整 provenance 见 `molecule_labels.jsonl`；所有原始 conflict parent、未达到 agreement 的拒绝
以及 source-row rejection 示例分别见 `conflicting_molecules.jsonl`、
`rejected_parent_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

每种构造方法的 `heldout_molecule_labels.jsonl` 是 valid+test union retrieval 泄漏隔离清单。
现有 full-source Starling index 不能直接用于 valid 或 test。
