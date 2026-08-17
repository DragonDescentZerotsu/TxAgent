# Skin causal-panel seed

这是 E22 的冻结 valid-only no-go diagnostic，不是 production Skin retrieval source，也不读取 test。

代码按数据生命周期分层：

- `skin_contract.py`：编译 outcome-calibrated MIE+downstream-KE cards，并定义实验专用 retrieval config。
- `build_skin_seed.py`：向冻结 MiniMol index 追加 causal group，保持既有 candidate order 和 groups 不变。
- `audit_skin_seed.py`：验证 direct retrieval parity，并 label-blind 地冻结 64-query seed。
- `analyze_skin_seed.py`：合并 direct 与 gated predictions，计算 paired metrics/bootstrap。
- `diagnose_skin_seed.py`：对已完成 traces 做 failure diagnosis，不触发新模型调用。

生产配置 `tools/chembl_tool/tasks/skin_reaction/experiment_config.py` 只注册 `chembl` 和 canonical `starling`。
Seed config 由 `causal_panel_retrieval_config()` 在本包内创建，避免 completed diagnostic 改变正式 pipeline。

结果：99 个 heldout-filtered cards（89 positive、10 negative）；64/64 seed 成功。合并到 245 条 valid 后，
macro-F1 从 direct `0.641036` 降至 `0.615116`，10 flips 为 2 beneficial/8 harmful。55/64 causal branches
为 low transferability，48/64 direction 为 neutral/unclear。该候选不 promotion、不扩展 BBB、不读取 Skin test。

```text
outputs/paper/skin_causal_panel_seed_v1_scaffold_valid_deepseek_v4_pro/
```
