# 普通 Murcko 检索复跑（2026-10-28）

使用 `merge_release_v1` 的 `8f0b8a31` 代码，通过 OpenRouter 的
`deepseek/deepseek-v4-flash-0731` 完成 DILI、Carcinogens 两 benchmark 的
12 个单元、3,069 条有效预测，无缺失。Direct 均为 scaffold-disjoint；三个设置
分别为 Direct 50、Direct 50 + parent-disjoint Indirect 50，以及用户追加核对的
Direct 50 + scaffold-disjoint Indirect 50。最后一项承接 10 月的独立对照。

这组结果单独登记，[冻结论文主表](../baselines.csv)保持原值。train/val/test、标签、
query condition、cached prior、prompt 模板、生成参数及 Neighbor-fill 配方沿用对应
原 manifest。检索使用普通、忽略立体构型但保留键型的 Murcko，并保留已批准的
parent 细化规则；索引里的旧拓扑字段仅保留来源用途。旧分组内实际 parent/Murcko
身份不同的结构在过滤、排序和预算填充前分别建候选，避免用代表结构替另一条记录
作判断。冻结索引未改写，既有预排除记录和外部排名缓存未自动恢复。

## 结果

下表为 **原冻结 Macro-F1 → 本次 Macro-F1**；完整精度、消息变化数量及预测分歧见
[scores.csv](scores.csv)，逐查询标签、预测及请求/响应哈希见
[predictions.csv](predictions.csv)。这里没有重跑 None、其他四项任务或其他骨干，
不据此计算新的六任务论文均值。重新推理存在波动，本次也包含已批准的 parent
细化及实际结构修复，并非只改变 scaffold 的严格单变量对照；CSV 单列了输入
未变时的预测分歧。

| Benchmark | Task | 每设置查询数 | Direct 50 | 50+50，Indirect parent | 50+50，Indirect scaffold |
| --- | --- | ---: | ---: | ---: | ---: |
| conditioned | dili | 402 | 0.5858 → 0.5618 | 0.6114 → 0.6072 | 0.5803 → 0.6414 |
| conditioned | carcinogens | 469 | 0.5278 → 0.5544 | 0.5783 → 0.5653 | 0.5211 → 0.5503 |
| tdc | dili | 96 | 0.6403 → 0.6267 | 0.6515 → 0.6574 | 0.6267 → 0.6609 |
| tdc | carcinogens | 56 | 0.6778 → 0.6468 | 0.6621 → 0.6468 | 0.6813 → 0.5777 |

## 旧输入的小规模对照

TDC Carcinogens 的 scaffold-indirect 出现较明显下降，因此追加核对原输入。
复用 28 条完全相同输入的当前回复，只对其余 28 条旧输入重新推理。完整 56 条
旧输入对照与新 Murcko 的 Macro-F1 均为 0.5777，但有 4 条预测不同；历史分数
为 0.6813，因此这次下降不能直接归因于 Murcko 修改。
这是一次小 cohort 的诊断，不是方差估计或“检索没有影响”的证明，也不替换主矩阵。
配置、指标与逐查询对照见 [control.json](control.json) 和
[control_predictions.csv](control_predictions.csv)。

## 验收与来源

- 3,069 条回复均通过原生 schema、引用和标签解析，并从逐条预测重算指标。
- 255,750 次选中记录通过实际结构检查；conditioned 卡片回查冻结索引中的源结构，TDC 检查记录携带的原始结构。所有适用的 50/50 预算均填满。
- 1,186 个请求相对原实验改变；本轮修复期间已有 156 条成功预测在请求与证据别名完全一致后复用，其余最终预测重新推理。未继承原论文预测来填充新结果。
- 293 项定向回归通过；一个 96-query 批次断网恢复零连接，回复字节不变。既有 9 项 BBB builder 检查仍受外部映射文件缺失阻塞。
- 原论文表格导出再次通过来源哈希检查，仍为 198 个 ML cells、108 个 LLM cells；本目录未混入原表。

[sources.json](sources.json)绑定代码、原实验目录/manifest、cohort、生成配置和新结果哈希。
本目录提供紧凑结果与执行配置；原始索引、prepared 请求和完整模型回复仍需随复现资产
交接，不把此导出标成完整分发包。输入完整性检查拒绝用其他索引或 prior 替代。

## 复跑入口

从 `sources.json` 选择一个 `source_batch`，恢复其冻结输入。使用一个新的运行名，
沿用[共享 baseline CLI](../../../predict/prompts/README.md#frozen-paper-baseline)：

```bash
python -m predict.harnesses.branches \
  --prepare-paper-baseline "$SOURCE_ROOT/$SOURCE_BATCH" \
  --source-root "$SOURCE_ROOT" --retrieval-protocol murcko --run-name "$RUN_NAME"

python -m predict.harnesses.branches \
  --run-paper-baseline "outputs/predictions/branches/paper_baseline_v1/$RUN_NAME" \
  --provider-config docs/paper/murcko_retrieval_20261028/provider.json \
  --env-file "$PROVIDER_ENV" --parallelism 128 \
  --retry-race-width 6 --max-stage-requeues 2 --retry-delay-s 1
```

`PROVIDER_ENV` 提供 `OPENROUTER_API_KEY`；公开配置使用通用凭据变量名，不包含密钥。
若冻结 manifest 的路径在新机器不可用，用文档中的 `--input-map` 替代 `--source-root`。
模型温度、reasoning 和 token 上限由各自原 manifest 提供；不要改用其他方法的默认
provider 配置。执行记录保存 OpenRouter 实际模型 ID；共用池仅接受已审核的 0731 别名。

若要复现旧论文检索，改用 `--retrieval-protocol legacy_paper` 和新的运行名。
该选项恢复旧 parent/scaffold 分组及代表结构行为，并要求重建后的证据和模型输入
与冻结批次一致；本页表格右侧对应 `murcko`，不是 legacy 重跑结果。

协议开关的离线验收见 [protocol_parity.json](protocol_parity.json)：DILI/Carcinogens
两 benchmark 的 12 单元、3,069 条 legacy 输入全部与原冻结实验一致；Murcko
另抽查 12 条重新检索输入，与本页复跑输入一致。171 项相关测试通过，已有
3,069 条 Murcko 运行仍可离线验证。本轮开关验收没有调用模型，也未覆盖其他
四任务的全量重检索；精确论文分数继续使用冻结预测，重新调用模型仍可能波动。
