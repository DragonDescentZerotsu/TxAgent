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
| Morgan KNN | 固定 k=3、阈值 0.5、train reference；all-train/condition-first；原始输入及稳定去重 | `predict.baselines.structure_knn.run`；36 cells 重跑，邻居与预测一致 |
| MiniMol embedding KNN | 同上；冻结 encoder/checkpoint 与相同 embeddings | `predict.baselines.minimol.run_embedding_knn`；36 cells 使用原缓存重跑，邻居与预测一致 |
| 普通 MiniMol head | 各 task 原超参数、25 epochs、seeds 1–5；val BCE 选 checkpoint，ensemble val Macro-F1 选阈值；无 CV、无 train+valid refit | `predict.baselines.minimol.run_bioavailability_ma`；18 cells 保存模型重放通过，3 cells 按原配置重新训练通过 |
| MiniMol 主任务微调 / indirect | 原主训练池、辅助池及权重候选；checkpoint/权重用 val BCE，阈值用 val Macro-F1；辅助池排除完整 valid/test parent/connectivity union，允许 scaffold 重合 | `predict.baselines.minimol.run_multitask`；21 套来源输入重建一致，36 cells 模型重放标签/F1 一致；完整候选/五种子重训仍未全量执行 |
| RASAR / RASAR+indirect | 6 / 6+2B 特征；两种邻居策略；六个 RF 候选、seeds 1–5、每 seed 100 trees；val BCE 选参数、val Macro-F1 选阈值 | `predict.baselines.retrieval_forest`；21 套特征重建与 72 cells 重放通过；两批累计 6 cells / 180 seed fits 重训通过 |
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

## 第三批：普通 head 配置与实际性能核验

复用 `experiments/baselines/conditioned.sh`，按 `TASK` 选择六任务各自的 head 参数，
按 `TRAINING_POOL=conditioned|tdc_only|mixed` 选择原条件字段。默认改为上述论文
val 选 checkpoint/阈值协议；Joseph 的固定 25 epochs、固定 0.5 模式仍可显式用
`TRAIN_ALL=1` 运行。两种模式都有实际 CLI 回归测试，没有新增正式训练入口。

head 已补齐本地 `--unseen-condition-policy`（默认仍为报错）、逐 epoch 验证 BCE
和验证集预测输出，同时保留 Joseph 的输入/缓存校验、benchmark 元数据与分条件统计。
runtime 支持显式 `TXAGENT_MINIMOL_RUNTIME_ROOT`，未设置时仍使用可移植的
`third_party/minimol`；本轮直接使用已有共享盘依赖，没有安装环境或启动工具服务。

性能核验以当前 198-cell `ml_appendix.csv` 的逐项来源为准，所有新结果写入独立目录
`outputs/analysis/release_refactor/20261022_ml/`，详细回执与执行命令见该目录中文 README。

| 核验 | 已覆盖范围 | 结果 |
| --- | --- | --- |
| KNN 实际 CLI 重跑 | 三训练池、六任务、两种指纹、两种策略，72 cells / 25,992 条 test 预测 | 逐条分数、标签、邻居及相似度完全一致；汇总指标仅有浮点舍入差 |
| 保存 head 重放 | 三训练池六任务，18 cells；重算 valid/test、阈值和 test 指标 | 逐条分数与标签完全一致；不是 18 cells 全部重新训练 |
| head 从随机初始化重训 | conditioned Ames、TDC-only Carcinogens、mixed Skin，各 5 seeds × 25 epochs | 15 次训练；test 分数、阈值、Macro-F1/AUROC 完全一致 |
| RF 准备与重训 | TDC-only Carcinogens，RASAR 与 RASAR+indirect，各 6 候选 × 5 seeds | 特征/donor 数组逐元素一致；60 次训练，选参、标签及指标一致，概率最大差约 5.6×10⁻¹⁷ |
| MiniMol 新编码 | 共享盘原 checkpoint，8 个真实分子，CPU 离线编码 | 8×512 输出；与原 embedding 最大绝对差约 3.5×10⁻⁶，满足既有容差；未全量重建 embeddings |

严格重训使用原 train/valid/test **分别保存的精确 embedding 副本**。按 SMILES 合并
复用不同批次缓存会接受容差内的数值差；试跑中约 2.6×10⁻⁶ 的 embedding 差异曾使
mixed Skin 的一条最终标签变化。恢复逐 split 精确输入后结果完全复现。因此保留通用
SMILES 复用功能，但不将它描述为逐位复现；精确复现用 `EMBEDDING_CACHE_DIR` 指向
独立复制并核对过顺序、标签的缓存，不能覆写历史缓存。Ames KNN 的旧 valid 缓存按已有
valid/test 对调回执映射至当前 test，而非直接依赖旧文件名。

本轮使用现有环境和冻结输入，RF 的旧 prepared manifest 仍含相对旧工作目录的来源，
通过显式加载整合分支模块完成验证；这不是全新环境下的独立发布验收。另有 42 项受影响
head、条件编码、indirect、RF 与导入边界测试通过。

## 第四批：全部 RF 与 encoder/indirect 核验

核验保存于 `outputs/analysis/release_refactor/20261022_indirect/`。本批新增的 72 个 RF
和 36 个 encoder/indirect 报告单元，加上第三批的 KNN/head，已覆盖当前 ML Appendix
全部 **198 cells**。这里的覆盖指来源、准备结果和保存模型重放，不表示全部重新训练。

- 21 套辅助准备结果重新从原 split 与 Parquet 构建：分子顺序、标签、条件张量、
  bucket、归一化目标和 heldout 排除统计一致。旧 payload 中缺少的 `condition_groups`
  仅补充输出元数据，训练张量未变。Skin 旧主任务微调/六维 direct 与新版 indirect
  按各自原来源分别核验，没有用新版辅助池替换旧实验输入。
- 21 套 RF 特征重新构建，两种 direct 策略、auxiliary 和 donor 索引逐元素一致。
  72 个报告单元对应 65 个保存结果目录，全部重新执行 finalize：校验各候选与其缓存的
  valid 分数，重做选参/阈值，使用保存模型重新计算 test；标签、Macro-F1/AUROC 一致。
- 新增 conditioned Skin、mixed Skin 的 condition-first RASAR/RASAR+indirect，
  共 4 cells / 120 seed fits 完整重训。使用重建的辅助目标与特征，从整合目录运行；
  选参、标签及指标一致。复用 Python CLI，仅新增一个固定论文参数的 Bash 配方
  `experiments/baselines/retrieval_forest.sh`；说明统一在该目录 README。
- 36 个 MiniMol encoder/indirect 单元均重新执行 finalize：所有候选 checkpoint 和
  缓存 valid 分数通过校验，辅助权重/阈值不变，重新计算 test。12,996 条报告预测的
  标签及 Macro-F1 全部不变；其中 8 cells 的 AUROC 有数值差，详见下段。
- TDC-only Skin 当前 1,735 张图重新生成，13,880 个字段逐项一致。完整单 seed、
  λ=0.6、25 epochs 的 CPU 对照使用“原代码＋原输入”和“整合代码＋重建输入”，
  每个 epoch 的主任务损失、辅助损失和 valid BCE 完全一致，选中 checkpoint 的
  327 个状态张量与验证概率也完全一致；不代替所有任务、所有辅助权重的五种子重训。

**数值差异核算：**原 encoder 套件在 CUDA 上训练，本轮在 CPU 上重放，概率最大差
为 1.38×10⁻⁶ 以内。8 cells 共 9 对正负样本的并列/排序发生变化，这些样本原分数差
均不超过 1.2×10⁻⁷；逐对贡献精确解释全部 AUROC 差值。最大为 TDC Carcinogens
indirect 的 −0.0010101（约 −0.10 个百分点），分类标签仍不变。用原代码在同一 CPU
环境重放 Oral 和 TDC Carcinogens 的微调/indirect 四个单元，逐条预测与整合代码完全
相同；五个迁移模块的 32 个函数/类主体也一致，仅局部 import 改路径。保留原始差异
回执，不把近似一致写成逐位相同，不用本轮分数替换论文冻结分数。

这些核验使用现有依赖与保存输入；新 RF 配方已在整合目录运行，但完整数据分发、
全新环境安装与 LLM 流程尚未验收。临时核验脚本收入本地源码 ZIP，不作为正式 runner。

## 第五批：结果入口、文档与 LLM 离线预检（2026-10-23）

[结果入口](results.md)现已包含本地论文主比较与原 FINAL collection 的明确区分。
`docs/paper/baselines.csv` 从原两套 `all_scores.csv` 导出 306 cells，其中 ML 198、
naive LLM 108；ML 的训练池和数值逐项与原 `ml_appendix.csv` 对齐。重新读取 294 个
不同预测文件（报告别名仍保留），306 cells 的 Macro-F1、accuracy 和 ML AUROC 均
与原汇总一致。没有执行新训练或推理，也没有将新检索规则冒充为这些旧结果的协议。

公开表仅保存预测文件 SHA-256，不包含个人绝对路径；本地核验回执保存原路径映射。
`docs/paper/sources.json` 绑定原表、导出物、cohort 和来源 registry 的哈希。
`ours_reported.json` 是原 72 个 Ours 分数来源文件的逐字节副本，保留用户提供及未核验
预测/预算的说明。未解决的 Direct=10/50 和 Ours 预算来源问题仍为待核对。
这些文件是冻结主比较的可阅读导出，不是另一个可手改的结果登记中心；原登记仍为
历史实验权威来源，正式 exporter/plotter、预算扫描与后续 scaffold 实验的分发仍待接入。

文档已补齐能力—入口—输入/输出映射，修正根 README 的废弃 CLI 参数、失效文件链接
以及预测说明中 MiniMol 章节的失效锚点。9 个 AGENTS.md 已取消 Git 跟踪并按文件名
全目录忽略；原本存在的工作区文件字节保持不变，稀疏检出未落盘的文件也已本地备份。
这不删除历史提交中的文件，也不代表整合分支已通过全库匿名审查。

LLM 离线预检覆盖 Gemini 的两套 benchmark、六任务、三 settings，共 36 个 batch。
每个 batch 固定抽取首/中/末三个 query，共 108 个：

- 原任务合同、manifest 和 request hash 全部匹配；用原 adapter v2 重渲染后，108 个
  requests 均与冻结消息逐字一致，None/Direct/Direct+Indirect 分别满足 0/0、50/0、50/50。
- 保持相同的模板输入，改用当前 `final_full_flat` 模板：108 个 user 消息完全一致，
  108 个 system 消息全部不同。当前模板增加/改变了 reasoning instructions、semantic
  utility 说明及固定输出要求；不能用默认模板替代原论文 prompt。
- 原 manifest 采用 direct scaffold-disjoint、indirect parent-disjoint，direct/indirect
  候选分子数均为 128。当前默认命名缓存为 scaffold-disjoint parent-100，不能直接替代。

这里只验证冻结样本与模板层差异，没有重建完整候选池，没有证明新 harness 全流程
等价，也没有推论 DeepSeek/Luna 的所有历史请求已验收。下一步在现有 prompt/selection
入口中显式承载原协议，再比较有序 records、字段去重、prior、最终请求和恢复行为。

详细回执、源文件映射与临时核验脚本归档于本地
`outputs/analysis/release_refactor/20261023_docs/`，不新增正式 Python runner。

## 验证与后续验收

第二批身份规则验证（2026-10-22；与上面的第三批性能核验分开）：

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

1. ML 的全部报告单元已完成保存模型/特征核验，保留上述数值与重训覆盖边界；
   后续独立环境复现沿用相同配置，不重复新增 task-specific runner。
2. 将历史路径依赖收敛为可分发的数据/配置入口，验证独立环境；现有环境通过不等于可直接发布。
3. 迁入 LLM 共享流程，核对完整候选池、records 预算、card 去重、prior 与最终请求。
   继续保留必要的限流、重试、恢复和历史 replay 能力。
4. 主比较分数已导出并验证；继续接齐共享导出器、绘图入口和完整实验登记，
   核对匿名发布文件；冻结结果保持原协议。
5. 完整验收后正常合并到 main，保留双方历史；在此之前不强制替换 main。
