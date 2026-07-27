# BBB_Martins Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v2`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：17,893
- test target：500
- 因 parent-level 标签冲突而丢弃：2,623

## Splits

| split | train | test | test Y=0 / Y=1 | scaffold overlap |
|---|---:|---:|---:|---:|
| random | 17,393 | 500 | 139 / 361 | 219 |
| scaffold | 17,393 | 500 | 122 / 378 | 0 |

## Source-row rejection

```json
{
  "interpretation_altering_qualifying_conditions": 103340,
  "invalid_or_unresolved_smiles": 1118,
  "no_tdc_compatible_qualitative_or_logbb_label": 38542,
  "within_record_label_conflict": 153
}
```

完整 provenance 见 `molecule_labels.jsonl`；冲突分子和 source-row rejection 示例分别见
`conflicting_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

`random/test_molecule_labels.jsonl` 与 `scaffold/test_molecule_labels.jsonl`
分别是两套 retrieval 泄漏隔离清单。现有 full-source Starling index 不能直接用于这些 test。
