# BBB_Martins Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v4`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：19,425
- record agreement threshold：70%
- valid / test target：500 / 500
- 原始 label-conflict parents：2,623
- majority 恢复：1,532
- agreement/tie 拒绝：1,091

## Splits

| split | train | valid | test | valid Y=0 / Y=1 | test Y=0 / Y=1 | scaffold pairwise overlap |
|---|---:|---:|---:|---:|---:|---:|
| random | 18,425 | 500 | 500 | 139 / 361 | 139 / 361 | 479 |
| scaffold | 18,425 | 500 | 500 | 150 / 350 | 148 / 352 | 0 |

## Source-row rejection

```json
{
  "interpretation_altering_qualifying_conditions": 103340,
  "invalid_or_unresolved_smiles": 1118,
  "no_tdc_compatible_qualitative_or_logbb_label": 38542,
  "within_record_label_conflict": 153
}
```

完整 provenance 见 `molecule_labels.jsonl`；所有原始 conflict parent、未达到 agreement 的拒绝
以及 source-row rejection 示例分别见 `conflicting_molecules.jsonl`、
`rejected_parent_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

每种构造方法的 `heldout_molecule_labels.jsonl` 是 valid+test union retrieval 泄漏隔离清单。
现有 full-source Starling index 不能直接用于 valid 或 test。
