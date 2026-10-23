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
| DeepSeek / Luna / Gemini naive LLM | 原模型、reasoning、prior、prompt；None / Direct 50 / Direct 50 + Indirect 50；两池均 Neighbor-fill；每设置一次 full-flat | 冻结协议、候选准备和普通 API 执行已接入专用模式；仍待全 cohort / 在线复现与 Batch、并行竞速恢复验收，不能用默认 prompt/缓存替代 |
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

## 第六批：冻结 LLM 输入输出协议（2026-10-23）

输入输出适配层已迁入 `predict/prompts/paper_baseline.py`，冻结模板和任务定义位于
`predict/prompts/variants/paper_baseline_v1/`。保留六任务各自的预测字段、置信度、
支持/反对/决策依据 card IDs、claims、evidence gaps 和 decision summary；TDC Ames、
Carcinogens 显式保留原专用定义。其他任务共用原定义，不继承当前 FINAL 默认配置。

复用现有模板加载器、原始 JSON 解析和通用 response validation；仅保留原 baseline
需要的引用校验，未迁入整套 progressive 状态机。字段精确去重函数与原实现 AST 相同，
原 full-fields 与 adapter v2 均可读取。模板环境在进程内复用，避免逐 query 重编译。

现有入口 `python -m predict.harnesses.branches --verify-paper-baseline BATCH_ROOT`
提供只读核验，不启动服务、不调用模型、不改写 batch。检查 manifest/request 绑定、
重渲染、card alias、原始回复解析、输出校验、标签映射及保存的指标。
运行方法与边界见 [冻结 baseline 协议](../predict/prompts/README.md#frozen-paper-baseline)。

论文主比较中的 DeepSeek、Luna、Gemini **108 个单元、38,457 条请求/响应**全部完成
离线核验，每个 backbone 各 12,819 条。迁移后的模板从冻结 payload 重渲染，system/user
逐字一致；原生标签映射、Macro-F1、accuracy 与原汇总一致。另将全部原始回复文本
重新交给现有共用 JSON 解析器，38,457 条解析结果也与保存的 content 完全相同。
原始回复解析与全量请求/标签核验分别留有回执，不把这一结果当作新模型推理复现。

780 个合法/错误响应变体的错误列表与旧校验器完全相同；两 benchmark 六任务的
12 组四次尝试测试中，格式修复请求及校验回执也逐字一致。针对性测试 29 项通过，
共用解析/校验测试另有 25 项通过；4 项既有恢复流程测试仍因缺少 `duckdb` 在导入时
失败，不记作通过。本批未安装依赖，也未伪造当前命名缓存的数据。

本地核验回执和临时脚本归档在 `outputs/analysis/release_refactor/20261023_llm/`。
这是冻结 payload/response 的迁移；从原始候选池重建 records、prior 加载、实际推理
参数/并发及完整 checkpoint 恢复仍待接入。该 bundle 暂不作为 fresh FINAL 的
`--prompt` 选项，不能把此处验证描述为完整 LLM runner 已可复现。

## 第七批：候选准备、预算选择与普通 API 执行（2026-10-23）

按用户授权，在现有 `vllm` 环境补装 `duckdb==2.0.0`。随后发现恢复模块导入时提前
读取 TDC 数据清单；已把读取移到真正执行索引构建时。没有伪造数据，也没有启动工具
服务。此前被阻塞的 4 项恢复测试全部通过，同一测试文件共 23 项通过。

现有 `predict.harnesses.branches` 入口增加 `--prepare-paper-baseline` 和
`--run-paper-baseline` 两种显式模式。前者从原 batch 读取冻结的查询、prior 和配置，
重新从 hash 绑定的完整索引生成候选；后者复用公共 provider pool 和 JSON validator
执行新准备的请求。没有复制原 family/progressive runner，也不把当前 FINAL 的
默认缓存、prompt 或 high reasoning 当作论文 baseline。

职责按现有目录组织：

- `predict/retrieval/paper_baseline.py`：原 L1/L2+ 分池、TDC train-label 合并与 heldout
  排除，随后调用已迁入的公共身份过滤。已批准的盐＋立体构型排除、位置异构体细分和
  DILI/Carcinogens 原 scaffold 规则均在预算填充之前应用；benchmark split 不重建。
- `optimization/selection/record_budget.py`：保持原 Neighbor-fill、molecule-cap、
  group-balanced 的排序、种子和短缺记录；六任务共享一份实现。
- `predict/prompts/paper_baseline.py`：复用公共 evidence 投影，从原始 cards 重建消息；
  query、condition、cached prior、任务定义、system/user 和原生输出 schema 保持原协议。
- `predict/harnesses/branches/paper_baseline.py`：新输出目录、来源预检、准备/执行及恢复；
  模型、temperature、reasoning effort、token 上限由原 manifest 读取。公共客户端
  增加显式 token 参数固定选项，保留原 `max_tokens`，超长时报错而非自动缩减预算。
  当前 FINAL 默认行为不变。改动 prior、cohort、标签或已保存请求绑定会在调用模型前拒绝。

离线验证：从真实 cards 重建的 324 个抽查请求覆盖三模型、两 benchmark、六任务和
三个设置，payload、引用编号、消息完全相同；从保存候选重新选择 indirect records
的顺序也全部相同。另有 1,200 组预算选择与原实现逐项一致，108 套原配置生成的 API
请求的字段及数值与旧客户端完全一致。130 项针对性测试通过，覆盖候选过滤/回填、预算、原生输出、
四次尝试的格式修复、断点恢复、调用前拒绝不一致输入、provider 和既有身份规则。
这些检查均没有网络推理，不把已保存模型回复当作新模型结果。

两套 benchmark 的六任务各取首/中/末 query，共 36 个样本从完整冻结索引重新准备；
Direct/Indirect 各 50 条的有序 IDs 与最终消息全部一致。另用实际 CLI 验证 None 的
3 个样本，均为 0/0 且消息相同。这只是抽查，不证明新规则对全 cohort 零影响。
完整索引的实际抽查结果与每条消息差异见本地
`outputs/analysis/release_refactor/20261023_preparation/`；正式命令和限制统一见
[论文 baseline 模式](../predict/prompts/README.md#frozen-paper-baseline)。

当前模式支持从恢复的论文输入执行普通 API 推理；仍需接齐旧 Batch 提交、first-valid
并行竞速重试、自动 stage requeue、匹配 Direct-3 和独立 prior 生成。公开分发的输入包、
全 cohort 的新检索差异审计和新模型性能复现也尚未完成。原始结果、分数和大 split 未修改，
不能据抽查相同就声称所有 query 的检索都不变或已完成在线性能复现。

## 第八批：快速 smoke 验收（2026-10-23）

本轮按用户要求只做小规模验收，没有启动完整实验或重建大索引。使用第七批已经从
完整索引重建的准备产物，两套 benchmark × 六任务各取首条 query，共 12 条，均为
Direct50 + Indirect50。通过实际 CLI、共享 provider pool 和 SDK 请求本机 HTTP
回放端点，初次返回错误格式，随后返回对应的历史原始回复；没有外部模型推理。

- 12 条全部通过：原 system/user 消息及生成参数一致，格式修复后的原生输出与历史
  回复一致；共 24 次本机 HTTP 请求，执行及恢复用时 27.30 秒。
- 12 次恢复均新增 0 次请求，prepared/output/summary 字节不变；额外修改一条标签
  会在请求前被拒绝。60 个受检查的来源文件哈希未变。
- LLM、身份过滤、provider 与恢复测试 130 项通过（4.97 秒）；MiniMol/RASAR 小型
  回归 50 项通过（62.01 秒），包含微型 head 训练、checkpoint 重载和已有 recipe
  的输出检查。后者有 3 条依赖库弃用提示，没有失败。

本地回执、CLI 日志和临时脚本归档位于
`outputs/analysis/release_refactor/20261023_smoke/`。没有增加正式 runner，也没有修改
正式实现。验收输出明确属于历史回复回放，不能作为新模型预测或性能提升的证据。
本轮确认普通 API 执行链路的基本可用性；全 cohort 检索差异、真实线上性能、Batch
及独立环境发布验收仍待完成，不据此直接宣布整合已全部完成。

## 第九批：已准备 LLM 输入的独立搬移与环境验收（2026-10-23）

发现并修复：原普通 API 执行仍依赖旧工作目录中的 manifest、查询和 cohort 文件。
现有 CLI 增加 `--bundle-paper-baseline`，将已准备请求及校验必需的小文件导出为新
目录，执行时使用包内相对路径。复用同一预检函数，原来源 hash、标签、prior、消息和
生成配置继续核验；源 manifest 除实现指纹外的原字段均须保持一致。空、重复、负数或
越界 query indices 被拒绝。打包采用临时目录完成校验后发布，不覆盖已有运行目录。

包仅包含选中的 prepared/evaluation/selection、原请求及原 manifest 快照和 cohort；
不复制预测、provider 配置、凭据或日志，也不复制大检索索引。导出是一次新的执行输入，
不是恢复已有预测；恢复应直接搬动原 portable batch。旧路径文字仍作为来源记录保留，
不能把这种内部复现包直接当作已经匿名审核的公开数据包。

检索依赖改为仅在准备阶段加载。新增最小执行依赖清单
`predict/requirements-paper-baseline.txt`；在全新 Python 3.12 venv 安装并通过 pip check。
复制源码至独立目录，用 Python audit hook 禁止读取原 main 和整合工作目录；未安装
RDKit、torch 或 duckdb。两套 benchmark × 六任务的 12 个 Direct50+Indirect50
样本均通过真实 CLI、本机 HTTP 回放、格式修复和恢复；25 次受限制的 CLI 进程完成，
共 24 次本机请求、0 次外部模型调用，恢复新增请求为 0。执行及恢复 14.58 秒；来源
60 个文件哈希未变。完整包和 smoke 输出合计约 17.5 MiB，而非完整检索数据集。

另修复 provider 配置加载器静默忽略旧 Batch transport 的问题：现在遇到 Batch 或
未知 transport 明确拒绝，避免意外改成普通 API。最小环境中已验证拒绝发生在客户端
构建前，并验证本机回放服务关闭后仍能恢复成功结果。134 项针对性回归测试通过。
文档中“候选准备及普通 API 尚未接入”的过时描述已更新；临时验收脚本只归档，不新增
独立 runner。复现命令统一见 [论文 baseline 模式](../predict/prompts/README.md#frozen-paper-baseline)。

回执、独立环境依赖版本、测试日志和源码归档位于
`outputs/analysis/release_refactor/20261023_portability/`。本轮仅完成已准备 LLM 执行的
独立环境验收，不代表全量检索重建、MiniMol encoder 训练或整个仓库已可独立发布。
普通 API/恢复已接入；旧 Batch、并行 first-valid 竞速、自动 stage requeue、Direct-3、
独立 prior 生成和完整原始数据分发仍需继续对齐。原实验入口及冻结结果未删除。
已重新 fetch 远端，main 为 `0ab66e37`，Joseph 仍为整合基点 `73bddf2e`；本轮未合并或推送。

## 第十批：Batch、可取消竞速与阶段恢复（2026-10-23）

将本地 main 已有的 Batch journal 和 first-valid 竞速实现迁入公共
`predict/api_client/batch.py` 与 `retry.py`，没有复制旧 family/progressive runner。
消息参数构造、回复解析和 provider 身份校验由普通 API、Batch 和异步请求共用；
同一个 `--run-paper-baseline` 入口选择 transport 和执行控制，冻结 prompt、模型、
reasoning、token budget 与输出 schema 不变。108 套真实原配置的请求字段及数值仍与
原客户端完全一致，1,200 组预算选择差分仍通过。

Batch 支持先提交独立请求，再等待及按 custom ID 回收；保留提交前落盘、独占锁、
不确定 POST 禁止自动重提、轮询失败不重提、显式 429 等待重排队和重启恢复。
修复确认终态失败的请求在重启时反复读出同一失败、无法重新提交的问题；成功兄弟
请求继续复用。停止本地客户端不取消远端付费任务。当前只接受单个 Batch provider、
无自动 failover、无同步费用账本；Batch 使用 race width 1。阶段重排队显式配置、默认
关闭，不能覆盖不确定提交的停止保护。若四次格式修复均失败，重启可能先回放这些
旧回复，可用 `--max-stage-requeues 1` 允许回放后的新阶段。未知 transport 继续拒绝，
不静默降级到普通 API。

普通 API 的 `--retry-race-width`、`--max-stage-requeues`、`--retry-delay-s` 已接入。
默认分别为 1、0、0；首次请求宽度 1，后续失败/格式修复可竞速，通过完整原生校验的
首个回复获胜。竞速路径使用真正的异步 HTTP 取消，所有落后连接清理后才返回。
并发仍受公共 provider 和总任务容量约束。显式阶段重排队只处理未成功 query；已完成
输出保持不变，失败、竞速和阶段恢复记录落在原 query 目录。同步费用账本与竞速组合
会在调用前拒绝，没有绕过原费用限制。

验收：153 项针对性测试通过，包含迁入的 Batch/竞速测试和新增终态失败恢复检查。
独立最小环境中，12 组实际准备输入（两 benchmark × 六任务）的 Batch 本地协议模拟
均通过错误格式修复、checkpoint 恢复及删除本地预测后的 journal 恢复；各两次模拟
提交，恢复新增提交为 0。另用真实本机 HTTP 验证两组 CLI 的格式修复/503 重排队，
共 8 次请求、4 条落后连接确认关闭；恢复新增请求为 0。旧目录读取和外网连接均被
测试守卫禁止，未提交真实付费 Batch，也没有重新产生模型性能结果。
另用真实准备输入验证四次格式失败后的显式 Batch 阶段重排队：第 5 次模拟提交成功，
后续 checkpoint/journal 恢复均不再提交。

完整输入清点：当前 12 个任务/benchmark 配置引用 36 个不同文件，均存在；其中
12 个检索索引共约 38.2 GiB。本轮只做路径与大小清点，hash 沿用原 manifest，未重复
扫描大文件或重建索引。MiniMol prepared 的 data/graphs 已按相对目录读取；RASAR
特征的 `inputs/data_dir`、RF 成员路径及部分训练 checkpoint 记录仍需做搬移适配，
不能批量替换冻结 manifest 中的字符串。完整 ML 输入分发尚未验收。

本地回执、模拟日志、源码与依赖边界记录位于
`outputs/analysis/release_refactor/20261023_transports/`；命令统一见
[论文 baseline 模式](../predict/prompts/README.md#frozen-paper-baseline)。代码增加主要
来自迁入旧功能及连接现有公共客户端，不是重复创建任务或模型专用执行入口。
Direct-3、独立 prior 生成、完整输入分发与匿名发布审核仍未完成；当前仍不合并 main。

## 第十一批：RASAR 搬移适配与少量真实调用（2026-10-23）

修复 RASAR 搬走目录后仍读取旧主数据/模型路径的问题，继续使用原
`retrieval_forest {prepare,fit,finalize}` 及 shell recipe，没有新增 runner 或路径框架。
新特征包将实际使用的三份 split 文件原样保存在 `splits/`，使用 v2 相对路径 manifest；
旧 v1 包不改写 manifest，通过已有 `--data-dir` 指定新位置并核对原哈希。
prepare 的 `--auxiliary`（recipe 的 `AUXILIARY`）可定位搬移后的辅助源，同样必须通过
冻结哈希检查。新模型 receipt 仅记录同目录文件名；旧 receipt 的绝对路径保留原字节，
运行时从 receipt 所在目录找模型并校验哈希。选参比较只归一化候选文件位置，其他字段
仍逐项比较；有效的完成结果恢复时不重写预测。

独立目录验收禁止读取原 main 和整合工作目录，同时禁止联网：

- TDC Carcinogens 的 RASAR / RASAR+indirect 共 60 个保存 seed 模型通过恢复与重放，
  各 28 个 valid、56 个 test；标签、阈值、选参及指标与原结果一致。
  RASAR 概率最大差 `5.55e-17`，RASAR+indirect 预测字节一致，没有重新训练。
- 从搬移后的 MiniMol targets / auxiliary / splits 重新准备 280 行特征；四组特征与
  donor 数组逐元素一致，两种 policy、两种 feature set 均核对。新特征包再次移动后
  不指定原数据目录也能加载。原来源目录的 120 个模型/结果文件哈希未变。
- MiniMol prepared 的数据、图缓存和一个真实 seed checkpoint 搬移后通过哈希、
  输入绑定和 28 条 valid 分数检查；历史 checkpoint 参数中的路径仅作 provenance，
  没有改写。此项不是新 encoder 推理或独立依赖安装验收。
- 67 项身份及受影响 ML 测试通过（3 条已有依赖弃用提示）；含新旧布局搬移恢复、
  重新预测，以及输入/模型变更时拒绝执行。recipe 的参数传递另做无训练检查。

按用户授权增加少量 OpenRouter 真实调用，固定
`deepseek/deepseek-v4-flash-0731`，保留所选原实验的完整 prompt、生成参数与 token 预算：
None 的 Skin 样例通过原生输出校验和零调用恢复。Carcinogens Direct+Indirect 样例
约 49,500 输入 tokens，两次返回均用满原 16,384 输出 tokens，分别没有最终 JSON、
只留下不完整结果，均被校验拒绝。异步传输返回正常，但该样例的完整输出验收未通过。
没有为通过 smoke 放宽 schema、缩减证据、增加 token 预算或替换论文结果。
总计 4 次尝试：1 次旧环境凭据 401、原实验凭据下 3 次模型返回；没有继续追加。
三次返回的 `usage.cost` 合计约 $0.00720，仅为接口返回费用，不作为独立账单审计。

回执、精确命令、原始回复和临时验收脚本归档见
`outputs/analysis/release_refactor/20261023_relocation/`；真实模型结果独立保存于
`outputs/predictions/branches/paper_baseline_v1/openrouter-smoke-20261023-project-key/`。
完整检索索引分发、其余独立环境安装、Direct-3、独立 prior、结果登记/绘图接齐及匿名
发布审核继续待验收。真实 Batch endpoint 尚未调用，不能以普通 API smoke 代替。

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

尚未通过完整流程验收：`duckdb` 缺失及 4 项恢复测试的导入阻塞已在第七批解决；
部分 UID/cache builder 测试仍依赖本地缺失的 Joseph v10 发布映射文件。
没有伪造这些发布数据来绕过检查。已将纯身份测试的 builder 导入
移至实际需要它的测试内，因此纯身份规则可独立验证；依赖真实发布输入的流程仍待验收。
Joseph 的命名缓存、源字段投影和完整 prompt 仍须单独对齐。

后续顺序：

1. ML 的全部报告单元已完成保存模型/特征核验，保留上述数值与重训覆盖边界；
   后续独立环境复现沿用相同配置，不重复新增 task-specific runner。
2. 已准备 LLM 执行及 RASAR 输入/模型已支持独立搬移；继续整理完整检索/ML 输入分发及匿名审核，
   不能把小输入包通过当成全库独立复现或公开发布验收。
3. LLM 候选/预算/冻结 prior、普通 API、Batch 和竞速/重排队路径已接入并离线验收；
   继续做全 cohort 差异审计、匹配 Direct-3 和独立 prior 生成，保留原来源和必要能力。
4. 主比较分数已导出并验证；继续接齐共享导出器、绘图入口和完整实验登记，
   核对匿名发布文件；冻结结果保持原协议。
5. 完整验收后正常合并到 main，保留双方历史；在此之前不强制替换 main。
