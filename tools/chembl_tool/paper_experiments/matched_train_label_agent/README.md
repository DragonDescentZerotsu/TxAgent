# Matched train-label agent experiments

本目录只包含与正式 Starling matrix 隔离的 scaffold-valid 诊断。保留原模块路径是为了旧命令和 artifact
可复现；这里没有任何 formal-test launcher，所有未通过 promotion gate 的实验都不得升级为默认 pipeline。

## 可复用诊断核心

```text
contract.py / materialize.py / run.py / summarize.py
audit.py / compare_full_pool.py
bio_trace_diagnosis.py / bio_trace_report.py / diagnose_bio_unanimous_positive.py
```

这组模块实现 E15 的 exact Morgan top-3 parity、label-visible direct group、paired 汇总，以及促成当前
Bioavailability `f20_evidence_calibrated_v2` 的 deterministic trace diagnosis。E15 本身没有通过跨 task
promotion gate；Bio 的 task-local prompt 修复是单独冻结、单独验证的当前默认版本。

## 已归档的 BBB E16 no-go

```text
bbb_property_compatibility.py
bbb_property_compatibility_audit.py
bbb_property_compatible_contract.py
bbb_property_compatible_experiment.py
bbb_property_compatible_report.py
bbb_property_compatible_trace_diagnosis.py
molecule_property_profiles.py
```

这些文件共同复现 BBB property-compatible selector 的 availability audit、唯一 matched-v3 candidate 和
post-run trace diagnosis。Candidate macro-F1 仅从 0.6696 变为 0.6708，paired CI 跨 0，promotion gate
失败；不运行 formal test，不继续搜索 budget、selector、prompt 或 structured-ledger 变体。为避免破坏历史
module command 和 artifact provenance，文件保留原位置，不再拆包或移动。

当前版本与指标的唯一短索引见上级 `RESULTS.md`；完整统计和固定 artifact 路径见
`STARLING_BENCHMARK_RESULTS.md`。
