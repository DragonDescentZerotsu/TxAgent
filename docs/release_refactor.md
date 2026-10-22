# 发布整合状态

本文后续统一使用中文。当前整合分支为 `codex/release-refactor`，以 Joseph
`73bddf2e` 为目录基础，以本地源码快照 `41403cf4` 和已确认的后续修复为功能标准。
最终展示分支仍为 main；当前尚未合并，也尚未完成整篇论文的复现验收。

## 整合原则

- 采用 Joseph 的 `predict/`、`data/processing/`、`optimization/`、`analysis/`
  组织代码；功能、参数和科学定义以已确认的本地版本为准。
- 同一功能只保留一份实现。已有同名函数必须核对行为，不能只改 import 就认定等价。
- 冻结的 train/valid/test、标签、原始结构、历史预测和分数保持不变。新的检索规则
  使用新的准备结果，不能将旧预测重新标成新规则的结果。
- 确定性的输入、特征、候选和请求应逐项一致；随机训练或推理再比较合理的性能波动。
  已批准的检索规则修复单独记录预期差异，不以旧预测逐条一致为目标。
- 结果数值继续从现有登记和汇总来源读取，不在本文另维护一份分数表。

## 第一批：MiniMol 微调、indirect 和 RASAR

| 能力 | 整合后的唯一实现或入口 |
| --- | --- |
| MiniMol 主任务微调与 indirect 联合训练 | `python -m predict.baselines.minimol.run_multitask` |
| 主任务/辅助目标与 heldout parent 排除 | `predict/baselines/minimol/indirect_data.py` |
| 可训练 encoder 和图缓存 | `predict/baselines/minimol/finetune_runtime.py` |
| RASAR 六维 direct 与 6+2B 维 indirect 特征 | `predict/baselines/retrieval_features.py` |
| RF 训练和验证集选参 | `python -m predict.baselines.retrieval_forest` |

这五个模块的算法主体与源码快照一致，仅迁移 import。共用已有的预测头、指标、
条件词表、指纹和 JSON 工具；没有新增根目录 `baselines/` 别名、独立模型客户端或绘图器。

共用条件邻居选择已保留本地“源文件中第一条 molecule-condition 记录”的代表规则，
并支持 `same_condition_then_null_then_all`。这项与 Joseph 原先选择排名最高重复记录
的行为不同，已明确按本地版本修正。

第一批检查：50 项测试、两个 CLI 帮助命令通过；18,656 个 benchmark 分子的 parent
keys 一致，2,700 次条件选择对照一致。覆盖范围为当前六任务 conditioned/TDC scaffold
三路输入，含 Skin/Carcinogens seed-42 替代输入，不包括整个辅助证据库或完整训练。

## 第二批：已确认的身份和检索规则

共用入口为 `predict/retrieval/policies.py`。DILI/Carcinogens 的专用运行时规则收在
`predict/retrieval/task_identity.py`，由共用判断入口按索引显式声明调用。
普通索引检索、分支 Morgan 检索和分支 assay-transfer 检索均在选择候选前应用同一逻辑。

| 规则 | 本轮实现 |
| --- | --- |
| 盐形式与立体构型同时变化 | 比较选定 parent 的连接结构，仍然排除；版本 `fragment_parent_connectivity.v2` |
| 多组分 | 保留本地原有精确 parent/component 匹配；不额外比较任意两个非主组分，不扩展到所有非主组分的立体变体或 scaffold |
| 普通 scaffold-disjoint | 比较选定 parent 的非空 Murcko scaffold，保留键型；无环分子不因空 scaffold 相同而排除 |
| DILI/Carcinogens parent-disjoint | 只细化旧 ElementGraph＋分子式＋净电荷同组候选；完成的去立体互变异构归一不同可继续接受通用 parent 检查；相同、缺结构或枚举未完成仍排除 |
| DILI/Carcinogens scaffold-disjoint | 保留原末端剪枝后的环/连接拓扑规则；先排除相同非空 scaffold，不改成普通 Murcko |
| 单重原子 query | 保留原元素匹配门槛，防止折叠指纹碰撞让无关元素占用检索预算 |

专用规则仅用于声明 `new_task_tautomer_identity.v2` 且 task 为 `dili` 或
`carcinogens` 的索引；缺失或混用候选身份字段时报错。检索输出同时记录旧分组合同、
新 parent 细化版本 `complete_stereo_free_tautomer_within_legacy_group.v1` 和结构资格版本。
没有重写 gold/split 构建器、heldout L1 预排除或已有索引；原来已从 L1 去掉的记录不会
因运行时细化自动恢复。

**适用边界：**这轮接通的是带本地合同声明的索引运行时。Joseph 的命名 SQLite/UID
缓存没有因此重建或补回原先被过滤的候选，也尚未证明与本地候选池、预算和 prompt
一致。不能直接用任意现成 Joseph 缓存宣称完成了论文复现。

## 全部 baseline 的配置对应关系

以下“本地来源”路径均相对于被保留的本地 main 工作目录，仍由
`tools/chembl_tool/paper_experiments/current_conditioned_results.json` 绑定当前结果。
它们不是整合分支中已经可用的默认数据路径；缺失时不得转用 Joseph 的 `CURRENT`。

| baseline | 必须保留的配置 | Joseph 框架下的入口与状态 |
| --- | --- | --- |
| Morgan KNN | 固定 k=3、阈值 0.5、train reference；all-train/condition-first；原始输入及稳定去重 | `predict.baselines.structure_knn.run` 已存在，共用 selector 已对齐；全量邻居和分数待核 |
| MiniMol embedding KNN | 同上；冻结 encoder/checkpoint 与相同 embeddings | `predict.baselines.minimol.run_embedding_knn` 已存在；环境与缓存复用待核 |
| 普通 MiniMol head | 各 task 原超参数、25 epochs、seeds 1–5；val BCE 选 checkpoint，ensemble val Macro-F1 选阈值；无 CV、无 train+valid refit | `predict.baselines.minimol.run_bioavailability_ma` 已存在；外围 CLI/缓存/输出尚有差异 |
| MiniMol 主任务微调 / indirect | 原主训练池、辅助池及权重候选；checkpoint/权重用 val BCE，阈值用 val Macro-F1；辅助池排除完整 valid/test parent/connectivity union，允许 scaffold 重合 | `predict.baselines.minimol.run_multitask` 已迁代码；配置实例与完整训练验收待完成 |
| RASAR / RASAR+indirect | 6 / 6+2B 特征；两种邻居策略；六个 RF 候选、seeds 1–5、每 seed 100 trees；val BCE 选参数、val Macro-F1 选阈值 | `predict.baselines.retrieval_forest` 已迁代码；三训练池的实际特征、donor 和结果待核 |
| DeepSeek / Luna / Gemini naive LLM | 原模型、reasoning、prior、prompt；None / Direct 50 / Direct 50 + Indirect 50；两池均 Neighbor-fill；每设置一次 full-flat | 本地 family runner 尚未完整接入；不能用 Joseph 默认 prompt/缓存直接替代 |
| 预算扫描、Direct-3 等对照 | 各自记录的预算、候选填充、模型与输入；Direct-3 保留固定 Morgan 三邻居，不套用其他检索过滤 | 复用共享入口，具体配置和结果登记尚待迁入 |

ML 的三种训练池为 conditioned、TDC-only、TDC + Starling。当前统一 ML Appendix 为
198 cells；不得只迁第一批模块后就把其他方法标成完成。

配置的权威来源及替代关系：

- 普通 head：`outputs/paper/conditioned_minimol_val_selected_20260921/manifest.json`
  和 `outputs/paper/tdc_minimol_val_selected_20260919/manifest.json`；按各 cell 的原命令与
  超参数迁移，不用一套默认 hidden dimension / learning rate 代替所有任务。
- encoder/indirect：registry `minimol_indirect_assay`、`tdc_minimol_indirect_assay`；
  TDC 套件的配置在 `outputs/paper/tdc_minimol_indirect_assay_20260922/manifest.json`。
- RF：`outputs/paper/retrieval_forest_20260922/manifest.json` 与
  `outputs/paper/retrieval_forest_direct_20260925/manifest.json`；conditioned DILI/Carcinogens
  使用 `same_condition_then_null_then_all`，其他采用各自已记录策略。
- TDC Skin/Carcinogens 使用 seed-42 的替代套件；原始查询由
  `outputs/paper/tdc_seed42_neighbor_fill_20260924/data/benchmark_manifest.json` 绑定。
  不能直接使用早期 manifest 中的 seed-1 输入；对应 mixed 数据也必须跟随替代套件。
- Skin indirect 以 `outputs/paper/skin_indirect_level_mapped_20260925/manifest.json`
  为当前替代来源；RASAR direct 六列与原 donor 保持不变。
- LLM 的当前来源为 registry `naive_llm_baseline`、`gpt6_luna_batch_b50`、
  `gemini38_flash_api_b50`、`tdc_seed42_neighbor_fill` 及各 budget-scaling 登记。
  用户提供的 Ours 分数保持其来源标注；未解决的预算来源差异不能靠迁移补写。

已经确认的配置差异：现有 `experiments/baselines/conditioned.sh` 使用 `--train-all` 和
固定 0.5 阈值，且提供统一默认 head 参数，不能直接作为我们的论文 head 配置。
head CLI 尚缺本地 `--unseen-condition-policy`，embedding runtime 的共享盘发现与
历史缓存定位也尚未完成对齐。这些是下一批需要处理的项目，本轮未默默改变脚本默认值。

## 验证与后续验收

本轮验证（2026-10-22）：

- 120 项身份、检索、导入边界与受影响 ML baseline 测试通过；另有 3 条依赖库弃用提示。
  回归测试覆盖普通索引、分支 Morgan 和分支 assay-transfer 三条选择路径，以及枚举
  未完成、来源结构优先、缺失/混用合同和单原子指纹碰撞。
- 36 个当前 conditioned/TDC scaffold split 文件与第一批保存的哈希一致；原始输入未改。
- 5,884 个 DILI/Carcinogens 分子—任务组合的完整旧分组字段与本地实现一致。
- 3,675 次通用判断和 900 次 DILI/Carcinogens 专用判断与本地确认后的规则一致。
  通用判断包括固定种子的真实 benchmark 抽样和构造边界案例；专用判断使用构造边界案例。
- 差分检查及测试回执保存于本地 refactor 归档的 `20261022_identity/`。这些验证没有
  启动工具服务、训练套件或付费模型推理，也不是完整辅助库或性能复现。

尚未通过完整流程验收：现有 flat 流程测试需要当前环境未安装的 `duckdb`；部分
UID/cache builder 测试在导入时读取本地缺失的 Joseph v10 发布映射文件。
本轮没有伪造这些发布数据或改变环境来绕过检查。已将纯身份测试的 builder 导入
移至实际需要它的测试内，因此纯身份规则可独立验证；依赖真实发布输入的流程仍待验收。
Joseph 的命名缓存、源字段投影和完整 prompt 仍须单独对齐。

后续顺序：

1. 对齐普通 head 的配置、运行环境和输出，再以原命令参数接入三训练池的共用入口。
2. 比较 KNN 邻居、RF 特征/donor 和 ML prepared 输入；确认后进行训练与性能核验。
3. 迁入 LLM 共享流程，核对完整候选池、records 预算、card 去重、prior 与最终请求。
   继续保留必要的限流、重试、恢复和历史 replay 能力。
4. 接齐唯一结果登记、共享导出器和绘图入口，核对匿名发布文件；冻结结果保持原协议。
5. 完整验收后正常合并到 main，保留双方历史；在此之前不强制替换 main。
