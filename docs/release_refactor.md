# 发布整合状态

更新：2026-10-28。交接分支 `merge_release_v1`；上一轮代码验收基线为 `013ba960`。
整合结构基于 `73bddf2e`，main 功能来源为快照 `41403cf4` 及已确认的修复。
上一轮代码级 smoke 验收通过，完整复现依赖分发仍未完成。先由合作者在此分支检查代码、
补齐依赖及待核对来源；确认交接完成后再正常合并到 main，当前不替换 main。

本文原位更新当前状态和证据索引，不继续追加逐轮工作日志；已完成待办及时移除，
验证结论保留范围与证据位置。功能/命令见
[代码职责](experiment_code.md#capability-map)、[baseline 配方](../experiments/baselines/README.md)、
[LLM 协议](../predict/prompts/README.md#frozen-paper-baseline)；分数、图表和来源见
[结果入口](results.md)。这几处各自维护一份事实，本文不复制命令或结果表。

## 当前验收清单

| 项目 | 已完成 | 剩余工作与限制 |
| --- | --- | --- |
| main 论文 ML | 198 cells 保存结果核验、模型重放和有限重训，详见下表 | 精确 embedding、辅助源、MiniMol runtime/checkpoint 的完整分发及独立环境验收 |
| main 论文 naive LLM | 108 cells、38,457 份冻结请求/响应核对；DeepSeek 两任务的 12 单元重检索实跑及断网恢复通过 | 完整执行输入的匿名分发；本轮未重跑其他任务或骨干 |
| 已批准检索规则 | 默认共用普通 Murcko，保留按索引合同启用的 parent 细化；论文 baseline 可显式选择 legacy 协议 | 外部冻结排名缓存仍需按新规则重建；BBB v10 映射读取及相关 9 项 builder 验收按用户决定延至合作者交接 |
| 论文输出及 Evidence 来源 | 表格与三张论文 PDF 已复现；五组消融由归档 TSV 绑定 | Ours 原始运行/预算、第六组消融协议和 Direct 10 分子/50 records 差异仍待核对，详见结果入口 |
| 复现分发 | LLM 36 文件哈希、ML 保存目录清单及 transfer 模型本地缓存清点完成 | 清单不等于完整分发；先对应论文实际运行，再闭合模型/排序缓存依赖 |
| 文件内容匿名化 | 文本、模型别名、冻结源码、二进制与压缩包元数据已统一处理；引用、哈希和换目录读取已验证 | Git 作者与历史按要求保留；远端 main 和审稿镜像须在合并/更新后重新检查 |

2026-10-26 已 fetch 核对：远端 main 为 `0ab66e37`、方法分支为 `73bddf2e`，均无
此前整合依据之外的新提交；方法分支完整包含于整合历史。main 另有 35 个尚未成为
整合分支祖先的提交，包含已按论文范围迁入的 baseline/源数据及旧目录清理；合并时
须保留已验收的新结构与双方历史，不能自动恢复退休入口。GATEWAY 数据读取、权限
及恢复继续留给合作者交接；它们不阻塞本轮代码 smoke，但完整公开复现仍需要这些依赖。

## 范围与不变项

- 只迁移当前论文正文、附录和图表所需能力。main 提供 ML/naive LLM baselines；
  Evidence 提供 transfer、context optimization、Ours 与消融，保留其其他既有功能。
- naive LLM 保留三骨干、两 benchmark、六任务的 None / Direct50 / Direct50+Indirect50，
  两池均 Neighbor-fill。12 cells 用 adapter v1、96 用 v2，均保留原消息和原生输出。
- Direct-3 **LLM**、完整历史预算扫描和 progressive/Joint 不新增迁移入口；ML 的固定
  k=3 及 RASAR 三个直接邻居特征继续保留。None 的旧 `molecule_cap` 字段仅为空输入标记。
- split、标签、原始结构、历史预测/分数不变；新检索输出使用新运行身份。冻结 prior
  已嵌入执行包，不另建生成器。Evidence FINAL 默认模板不能替代 main 的论文 prompt。
- 工具服务与 Joint 保持停止；不因复现依赖检查自动下载模型、重建历史实验或全量推理。

## Baseline 配置与验收

三种 ML 训练池为 conditioned、TDC-only、TDC + Starling。下表是既有验收的汇总，
不是本次重新运行的结果；具体入口和参数只在对应能力文档维护。

| 能力 | 已验证范围 | 边界 |
| --- | --- | --- |
| Morgan / MiniMol embedding KNN | 72 cells / 25,992 条 test 预测；分数、标签、邻居、相似度一致 | 使用冻结 embedding；原固定 k=3、阈值 0.5 |
| MiniMol head | 18 cells 保存模型重放；3 cells × 5 seeds × 25 epochs 重训一致 | task 原参数、val BCE checkpoint、ensemble val Macro-F1 阈值；Evidence 固定 epoch/0.5 模式仍可显式选择 |
| MiniMol 微调 / indirect | 21 套准备输入一致；36 cells 重放分类标签/F1 一致；一个真实 seed 的 25 epochs 训练逐步一致 | 不是全候选/全种子重训；CPU/CUDA 数值差见下文 |
| RASAR / RASAR+indirect | 21 套特征/候选 donor 一致；72 cells 重放；6 cells / 180 seed fits 重训通过 | 保留 6 / 6+2B 特征及 val 选参/阈值；搬移另核验 60 个保存 seed 模型 |
| naive LLM | 38,457 次冻结消息重渲染、回复解析和标签/指标核对；4,273 个 prior 对应 2,054 个缓存源文件 | 保存输入/输出验收，不是新模型性能；TDC prior 原生成 preparation manifest 未独立追溯 |
| 执行与搬移 | 12 组 API/Batch 本机模拟、格式修复、零新增请求恢复；36 个 None 组合最小环境搬移；3 个完整索引的单 query 重建 | 完整索引 smoke 为 conditioned Skin、TDC Skin/DILI；Batch 未调用真实远端 |

MiniMol 精确重训必须保留每个 split 的原 embedding 顺序和字节；按 SMILES 复用
其他批次的近似缓存曾引起一条标签变化。8 个真实分子的重新编码最大误差约
3.5×10⁻⁶，仅证明容差内一致。Encoder 在 CPU 上重放原 CUDA 模型时，8 cells 的
AUROC 因近并列样本排序改变；最大差 −0.0010101，分类标签不变。同一 CPU 上原代码
与迁移代码预测一致；差分回执保留，不用新分数覆盖论文。

2026-10-26 从提交建立完整干净 checkout，新建 Python 3.12 venv，仅安装既有
`predict/requirements-paper-baseline.txt`。TDC Skin 的一个查询经当前规则重新准备后，
50 条 Direct + 50 条 Indirect 的 ID、顺序和模型消息与原实验相同；导出可搬移执行包后，
使用固定 DeepSeek V4 Flash 0731 一次调用通过完整 JSON 校验。沿用 temperature=0、
原 reasoning 设置和 16,384 输出 tokens；接口报告费用 $0.008270857。随后禁止网络连接
恢复成功，输出字节不变、零新增调用。这是功能检查，不估计性能，也不更新论文分数。
此前 Carcinogens 的两次 token 耗尽失败保留在内部回执，本次未改配置重试该样例。

同一干净 checkout 的 9 个公开 CLI 入口、七张派生表和三张论文 PDF 生成成功；DILI
独立包在新环境安装后，于仓库外运行文档示例通过。275 项针对性回归通过，覆盖
baseline、身份规则、上下文选择、模型传输、运行恢复和结果导出；57 个主要文档本地
链接有效。这不是全库所有测试通过，也不包含 GPU 全量训练或不可读外部数据的验收。

此次修复仅保留原冻结 v5 协议名的精确摘要兼容，避免匿名命名使旧 baseline 文件被
拒绝；清单和消息哈希校验不变，未知 v5 名称仍拒绝。修复后用干净 checkout 逐条复核
原始 108 个单元、38,457 份请求/响应，原生标签和保存指标全部一致。另删除与现有模块 README 布局
冲突的过时测试，修正 CLI 测试的 mock 位置，取消跟踪最后一个内部技能软链接。
生产代码未新增入口或执行框架。

### 冻结配置定位

下列路径相对保留的原 main 工作目录，由
`tools/chembl_tool/paper_experiments/current_conditioned_results.json` 绑定，
不是整合分支默认输入。缺失时不能改用 Evidence `CURRENT`。

| 内容 | 原始来源 / 当前替代 |
| --- | --- |
| Head | `outputs/paper/conditioned_minimol_val_selected_20260921/manifest.json`、`tdc_minimol_val_selected_20260919/manifest.json`（同级目录） |
| Encoder / indirect | registry `minimol_indirect_assay`、`tdc_minimol_indirect_assay`；TDC manifest 在 `outputs/paper/tdc_minimol_indirect_assay_20260922/` |
| RF | `outputs/paper/retrieval_forest_20260922/manifest.json`、`retrieval_forest_direct_20260925/manifest.json`（同级目录）；conditioned DILI/Carcinogens 保留 `same_condition_then_null_then_all` |
| TDC Skin/Carcinogens | `outputs/paper/tdc_seed42_neighbor_fill_20260924/data/benchmark_manifest.json`，替代 seed-1，mixed 输入同步跟随 |
| Skin auxiliary | `outputs/paper/skin_indirect_level_mapped_20260925/manifest.json`；旧主任务微调/direct 与新版 indirect 分别使用原来源 |
| LLM | registry `naive_llm_baseline`、`gpt6_luna_batch_b50`、`gemini38_flash_api_b50`、`tdc_seed42_neighbor_fill`，以及主表引用的 scaling 50/50 单元 |

配置由逐 cell 的原 manifest 决定；目录含 scaling 不代表可删除其论文单元，也不意味
必须迁入整个历史扫描。完整命令与超参数不在状态页再维护一份。

## 检索合同与依赖边界

当前规则统一在[检索合同](../predict/retrieval/assay_reranking/CACHE_LAYOUT.md#current-exclusion-rules)
维护，该文档已按实际代码修正过时的“比较所有组分 scaffold”描述。

2026-10-28 用户明确要求 DILI/Carcinogens 检索也采用保留键型的普通 Murcko。
默认共享判断不再使用专用拓扑排除，保留已批准的 parent 细化及冻结 split；此前
“已统一”的表述仅适用于通用实现，不代表当时两任务的 scaffold 定义已相同。
旧论文结果及 10 月 indirect-scaffold 对照仍属于旧拓扑协议。本轮以 `8f0b8a31`
代码完成两任务、两 benchmark 的 Direct50、50+50 parent 和 50+50 scaffold，
共 12 个 DeepSeek 单元、3,069 条有效预测；255,750 次选中记录通过实际结构检查，
适用预算全部填满。旧分组代表结构与实际证据身份不一致的问题已在预算前修复；
293 项回归及 96-query 断网恢复通过。模型执行复用共用池已审核的 0731 别名。
新旧分数、逐查询预测、来源绑定及额外 28 条旧输入诊断统一见
[普通 Murcko 复跑记录](paper/murcko_retrieval_20261028/README.md)。该记录单独保留
规则变化和重新推理带来的差异，不覆盖冻结论文分数，也不宣称所有性能逐数一致。

随后按用户要求保留两个显式协议：`--retrieval-protocol murcko` 为上述默认规则；
`legacy_paper` 恢复论文的旧 parent/scaffold 分组、分组代表结构和通用身份判断。
复用同一个准备/执行入口，协议写入 manifest；执行不能临时切换，切换须使用新运行名。
legacy 从完整冻结索引重新检索，并在准备和执行时强制核对原消息、payload、引用
映射及有序证据 ID；不一致即拒绝，不复制原证据冒充重检索。
两种协议均不修改 split、标签、既有预测或 FINAL 的冻结排名缓存；操作见
[baseline 入口](../predict/prompts/README.md#frozen-paper-baseline)。
双协议离线验收已通过，范围及来源统一见
[协议验收记录](paper/murcko_retrieval_20261028/README.md#复跑入口)；未新增模型调用。

历史差分覆盖 36 个 split 文件哈希、5,884 个专用分子身份、3,675 次通用和 900 次
专用判断；无 split/标签改写。本次为修正文档复核现有身份/结构表示测试，69 项通过。
这些运行时规则不会补回旧索引已排除的记录，也不证明 Evidence 命名缓存已按新规则重建。

2026-10-25 的本地清点与 GATEWAY 只读核对：

- 五个 reranker 注册模块共 32 个精确模型版本（含历史角色）：16 个具备权重、配置及
  tokenizer 文件，2 个 snapshot 不完整，14 个未在检查范围内找到。未加载模型、
  全量哈希权重或闭合自定义代码依赖；这不是论文必须发布的模型清单。
- 原样展开 Git 中三份 relocation 映射后，沿现有 resolver 检查默认
  `scaffold_disjoint_parent100_v2`：12 个 task/benchmark 的 24 个 retrieval index
  和 6 个 optimizer direct-card index，合计 30 个，在当前 checkout 和 Git 中均未找到。
  GATEWAY 的源仓库 HEAD 为 `73bddf2e`，catalog 与本地 SHA-256 相同：11 个 Gold 索引
  可读，6 个 TDC 索引确认存在但不可读，其余 13 个因目录权限无法确认文件状态。
  可读索引引用的 75 处下级清单（73 个不同路径）均受权限阻塞，未验证完整缓存。
- BBB v10 `level_mapping/manifest.json` 已在 GATEWAY 确认存在（2,919 bytes，权限 0600），
  当前账号无法读取，因此尚未恢复本地或重跑相关 9 项 builder 测试。按用户决定，
  读取权限、清单/payload 校验及恢复留给合作者交接阶段，不再为此反复重试或扩大搜索。
  另两项旧 V10.4 检查缺外部训练 release manifest，不自动扩展为论文验收任务。

上述是 Evidence 默认路径的可用性核对，不证明论文用了该默认值，也不影响已经核对的
main baseline 输入。GATEWAY 源目录和逐项权限清单保存在内部回执，不重复写入公开说明。

模型评分与读取已算好的排序缓存需要不同依赖，见
[模型合同](../predict/retrieval/assay_reranking/MODEL_CONTRACTS.md)。先绑定论文运行，
再选择恢复入口与匿名分发范围；不替换为名字相近的缓存或虚构模型下载地址。

## 可维护性与匿名发布

本轮检索、执行及双协议支持相对 `2f4be1d8` 的生产 Python 净增 87 行，复用现有入口与
化学身份函数，未新增 runner 或依赖。已撤回论文不用的选择
算法并移除失去用途的单调用包装。2026-10-25 维护审查相对 Evidence 的生产 Python
净增 3,132 行、测试净增 1,551 行；这是当时的体量快照，不是当前实时统计或零膨胀证明。
变更模块中未发现至少六行、函数体完全一致的重复函数；不代表不存在语义重复。

交接文档按职责维护：本文记录状态与限制，代码总览映射能力，模块指南维护命令，
结果页记录分数与来源，FINAL 指南只维护恢复和发布步骤。已合并重复的评分器、LLM
操作和历史结果说明，修正旧客户端路径、内部 skill 引用及硬编码推送分支。
本次检查覆盖 54 份维护文档、92 处本地链接，路径和章节锚点均有效；历史报告、
冻结协议及归档内说明保持其版本含义，不作为当前启动指南。

### 文件内容匿名化（2026-10-25）

公开文件使用中性 `baseline` / `evidence` 命名；私人机器路径改为仓库相对路径或
显式配置，个人模型地址改为带精确 revision 的本地依赖别名。共享 resolver 校验
`artifact.json` 和文件 SHA-256；缺少依赖会明确报错，不会下载同名替代模型。
配置方式只在[模型合同](../predict/retrieval/assay_reranking/MODEL_CONTRACTS.md)维护。
内部指令不发布；live trace 默认读取同目录数据，可通过 `?data=` 指定其他位置。

清理覆盖完整跟踪树、27 套 V7 stage 归档、ClinTox 包、6 个独立 gzip，以及 FINAL
的 268,904 个文件。解码检查 407 个独立 Parquet/SQLite 文件及 73 个嵌套二进制成员；
其中两张 Parquet 仅修改来源定位列，其余列相等。公开文献作者、临床研究机构和疾病
名称属于科学原始数据，保留原文；它们不作为本项目作者身份标记删除。

FINAL 的五份冻结源码、协议命名、45 处源码绑定与 20 处组合指纹已同步更新。
158 份清单的 204,692 行、198 处副本引用重新校验；1,505 处历史来源引用保留原义。
`original_sha256` / `original_assembly_files_sha256` 区分冻结原件与公开副本，
归档根目录 `publication_projection.json` 说明边界。请求/回复正文、预测、标签、
分数及 split 不变；归档 owner/group 和压缩头不再携带私人元数据。

验收使用正式读取器：在仓库外读取 137 个实验、33,368 个 query 身份与 274 次边界
query；17 个 Direct 配置、41 组参数通过，错误哈希仍被拒绝。两个真实保存 trace 的
25 个分析字段仅请求/输出文件哈希不同；请求中仅协议命名改变。独立解压再核验所有成员、归档分卷与上述引用。
这些证明公开副本可以正确定位和校验已分发记录，不等于全部外部模型/数据已分发，
也不替代新推理性能验收。

本轮生产改动集中在既有路径 resolver、网关配置和 trace 页面默认地址；未增加
runner、plotter 或另一套执行框架。2026-10-25 的扩展回归为 281 项通过、32 项缺输入失败（30 项既有 gold/index
缺件和 2 项旧 V10.4 外部训练清单缺件），失败集合没有扩大；这 32 项缺件仍未闭合，与本轮 275 项针对性回归覆盖不同。文件内容匿名化与“完整独立环境复现”是两个验收项，后者的依赖缺口见上表。

## 验收证据

公开结果与协议由本文顶部的能力、baseline 和结果文档统一索引，不在这里重复维护
数值表。原第 1–19 批过程可用 `git show a2d6c4a1:docs/release_refactor.md` 查阅。
内部完整回执按批次保存在 `outputs/analysis/release_refactor/`；匿名化入口为
`20261025_anonymous_release/`，最终 smoke 入口为 `20261026_final_acceptance/`。回执含私人路径映射，仅供维护者审计，不随匿名文件树发布。
冻结原件保留在历史版本和内部归档；当前 Git 作者、历史提交及远端仓库身份不在此次
文件内容匿名化范围。向审稿系统提供当前文件快照时，不应同时导出 `.git` 历史。
