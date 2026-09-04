# Skin_Reaction Starling 二分类数据构建报告

- 协议：`starling_binary_benchmark.v4`
- 分子身份：`rdkit_fragment_parent.v1`
- seed：20260723
- 可用二分类 parent：2,386
- record agreement threshold：70%
- valid / test target：238 / 238
- 原始 label-conflict parents：959
- majority 恢复：531
- agreement/tie 拒绝：428

## Splits

| split | train | valid | test | valid Y=0 / Y=1 | test Y=0 / Y=1 | scaffold pairwise overlap |
|---|---:|---:|---:|---:|---:|---:|
| random | 1,910 | 238 | 238 | 72 / 166 | 72 / 166 | 131 |
| scaffold | 1,910 | 238 | 238 | 73 / 165 | 74 / 164 | 0 |

## Source-row rejection

```json
{
  "inconclusive_outcome": 169,
  "integrated_prediction_or_defined_approach": 6,
  "invalid_or_unresolved_smiles": 16172,
  "mechanistic_assay_in_direct_source": 654,
  "out_of_scope_irritation": 793,
  "out_of_scope_noncontact_cutaneous_reaction": 481,
  "out_of_scope_photo_hazard": 786,
  "outside_sensitization_scope": 5049,
  "prediction_only": 21
}
```

完整 provenance 见 `molecule_labels.jsonl`；所有原始 conflict parent、未达到 agreement 的拒绝
以及 source-row rejection 示例分别见 `conflicting_molecules.jsonl`、
`rejected_parent_molecules.jsonl` 与 `source_rejection_examples.jsonl`。

每种构造方法的 `heldout_molecule_labels.jsonl` 是 valid+test union retrieval 泄漏隔离清单。
现有 full-source Starling index 不能直接用于 valid 或 test。
