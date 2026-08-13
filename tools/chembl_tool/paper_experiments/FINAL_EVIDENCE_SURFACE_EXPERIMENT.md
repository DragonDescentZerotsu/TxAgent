# Final evidence surface 诊断（已完成）

## 目标与边界

当前 direct/full-flat/full-mechanism 结果表明 retrieval evidence 相对 none 有价值，但不同组织方式的净增益不稳定。
本实验先隔离一个更具体的问题：raw analog evidence 是否已被检索到，却在 group-to-final 的二级总结中丢失。

本实验只改变 final evidence surface，不改变 retrieval。Metadata census 同轮记录现有 evidence 的描述性字段
覆盖率，但不实现 compatibility selector，也不把 coverage field 变成 relevance score。

当前诊断数据固定为：

```text
dataset: record_supported_v2
split/subset: scaffold-valid
tasks: BBB_Martins, Bioavailability_Ma, Skin_Reaction
visibility: identity_blind
identity policy: parent_disjoint
source condition: starling_full_flat
model: gpt-oss-120b（从 frozen source run manifests 推导并强校验）
```

正式 test 不在本轮执行范围内；valid 结果不能用于继续调 card 长度、字段或 task prompt。

## 模块职责

```text
tools/chembl_tool/common/final_evidence_surface.py
  版本化、task-agnostic card builder。summary_only 是严格 no-op；card surface 只消费已经准备好的
  reasoning retrieval，不做 retrieval、label、task threshold 或 source-specific scoring。

tools/chembl_tool/common/evidence_compatibility.py
  只产生 endpoint/context/measurement/support/uncertainty 的描述性 census inputs；不输出 relevance score，
  也不是待运行 selector。

tools/chembl_tool/paper_experiments/run_final_evidence_surface_experiment.py
  从冻结 source batch 复制 retrieval/single/group，使用公共 global prompt pool 只运行 final。

tools/chembl_tool/paper_experiments/summarize_final_evidence_surface_experiment.py
  全样本配对 macro-F1、bootstrap、McNemar/Holm、flips 和中文 report。

tools/chembl_tool/paper_experiments/audit_starling_evidence_metadata.py
  流式扫描 full Starling evidence libraries，生成 field/group coverage 和 source hash。

tools/chembl_tool/paper_experiments/audit_final_evidence_surface_contract.py
  全量验证 source/target artifact hash、card payload hash、surface provenance 和 identity leak。
```

三个 task runner 只负责把公共 evidence fields 插入各自 final schema；不得复制 card 构建、采样或 hash 逻辑。

## Surface 合同

```text
summary_only
  原始 group_reasoning_outputs 字段和 instructions；不生成 card，不增加 prompt key。

summary_plus_cards
  保留 group_reasoning_outputs，并增加 final_evidence_cards。

cards_only
  不向 final 提供 group_reasoning_outputs，只提供同一 final_evidence_cards。
```

Card v1 固定：

- 最多 12 个 neighbor cards；超过时 deterministic even spacing；
- 每 card 最多 3 条 evidence rows；
- evidence 只保留 source/group/endpoint/text/annotations/quality/provenance 和最多两个 compact examples；
- comparison 只保留现有 prefetched `properties_compare` / `mmp_structure_compare` 文本；
- 不包含 molecule SMILES、名称或内部 rule-derived direction/strength；
- audit 保存 contract version、candidate/retained counts、bytes 和 SHA-256。

Card 与 group summary 来自同一 evidence，`summary_plus_cards` prompt 明确禁止将两者重复计票。

## 可恢复运行

先生成 metadata census：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.audit_starling_evidence_metadata
```

单样本、三任务、两 surface smoke：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_final_evidence_surface_experiment \
  --limit 1 --parallelism 6
```

完整 valid：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.run_final_evidence_surface_experiment \
  --parallelism 128 --max-stage-requeues 1

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.audit_final_evidence_surface_contract

/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.paper_experiments.summarize_final_evidence_surface_experiment
```

Runner 默认 `--skip-existing`，每个成功 final 原子发布；中断后重复同一命令只修复缺失或失败 stage。

## 输出结构

```text
outputs/paper/final_evidence_surface_record_supported_v2_valid_gpt_oss_120b/
  experiment_manifest.json
  metadata_census/
    coverage.json
    field_coverage.tsv
    group_coverage.tsv
    report.md
  runs_identity_blind_parent_disjoint/
    summary_plus_cards/<task>/<batch>/
    cards_only/<task>/<batch>/
  contract_audit/
    audit.json
    report.md
  analysis/
    summary.json
    metrics.tsv
    paired_comparisons.tsv
    surface_diagnostics.tsv
    report.md
```

不得写入 canonical v2 source run root，也不得覆盖旧 `retrieval.json`、group outputs 或 metrics。

## Gate

Smoke gate：

1. source batch 为完整 0-failure artifact；
2. target manifest 指向 source run，surface/version 正确；
3. source 与 target retrieval/single/group 文件内容 hash 相同；
4. identity-blind trace 无 query/neighbor identity leak；
5. final output 包含 card audit，card SHA 可重建；
6. 三个 task 的 structured output 均有效。

完整 valid gate：

1. 每个 task/surface `n_failed_runs=0`；
2. paired sample 数分别为 500/209/245；
3. 报告 paired macro-F1 95% CI、McNemar/Holm、flips、rescue/harm；
4. 同步报告 prompt token/latency 增量；
5. 若 cards 没有稳定增益，冻结 no-go 结论，不调 card 长度、不运行 formal test，也不自动派生 selector。

## 2026-08-07 scaffold-valid 结果

六个 task/surface batch 共 `1,908/1,908` 成功，`n_failed_runs=0`。全量 contract audit 检查同样的
1,908 个 runs，source/target retrieval、single、group hash、card hash 和 identity-blind leak 均为 0 failure。

| task | summary-only | summary + cards | delta (95% paired CI) | cards-only | delta (95% paired CI) |
| --- | ---: | ---: | --- | ---: | --- |
| BBB | 0.6753 | 0.6737 | -0.0016 [-0.0244,+0.0217] | 0.6467 | -0.0285 [-0.0666,+0.0104] |
| Bioavailability | 0.5707 | 0.5718 | +0.0011 [-0.0376,+0.0392] | 0.5780 | +0.0073 [-0.0435,+0.0593] |
| Skin Reaction | 0.5577 | 0.5445 | -0.0132 [-0.0524,+0.0261] | 0.5595 | +0.0018 [-0.0486,+0.0522] |

六个 interval 均跨 0，exact McNemar 经 Holm 校正后均为 `p=1`。添加 cards 将 final mean prompt tokens
提高到 summary-only 的 `3.32x–4.72x`，但没有稳定改善。因此 aggregation compression 不作为当前主因，
也不把 card surface 升级为默认。不得根据本轮 valid 调 card 长度、运行 formal test，或把 metadata census
包装成新的 compatibility selector。

## 停止条件与后续边界

本实验线到此结束。保留 card builder、runner、audit 和 summarizer 只为复现实测结果；默认 agent prompt 继续使用
`summary_only`。Metadata census 只能说明字段是否存在，不能证明 endpoint 与 query task 匹配，也不能证明证据有用。

下一步不做 group drop/add、per-group train reasoning、full-mechanism train trace bank 或 selector 搜索。Skin label-scope
修复与现有 trace 的 final-vs-upstream bottleneck audit 记录在 `ICLR_2027_EXECUTION_PLAN.md` 的近期小计划中。
