# 发布整合状态

更新：2026-10-25。工作分支 `codex/release-refactor`，目录基于 Joseph `73bddf2e`，
main 功能来源为快照 `41403cf4` 及已确认的修复。尚未 push 或合并 main，未通过完整发布验收。

本文原位更新当前状态和证据索引，不继续追加逐轮工作日志；已完成待办及时移除，
验证结论保留范围与证据位置。功能/命令见
[代码职责](experiment_code.md#capability-map)、[baseline 配方](../experiments/baselines/README.md)、
[LLM 协议](../predict/prompts/README.md#frozen-paper-baseline)；分数、图表和来源见
[结果入口](results.md)。这几处各自维护一份事实，本文不复制命令或结果表。

## 当前验收清单

| 项目 | 已完成 | 剩余工作与限制 |
| --- | --- | --- |
| main 论文 ML | 198 cells 保存结果核验、模型重放和有限重训，详见下表 | 精确 embedding、辅助源、MiniMol runtime/checkpoint 的完整分发及独立环境验收 |
| main 论文 naive LLM | 108 cells、38,457 份冻结请求/响应核对；普通 API、Batch 模拟、恢复、搬移 smoke | 完整输入匿名分发；真实 Direct+Indirect smoke 尚未成功；未做全量新推理 |
| 已批准检索规则 | 共用 parent/scaffold 实现、预算前过滤及有限完整索引准备验收 | 全 cohort 候选差异和新规则性能未验收；BBB v10 映射读取及相关 9 项 builder 验收按用户决定延至合作者交接 |
| 论文输出及 Joseph 来源 | 表格与三张论文 PDF 已复现；五组消融由归档 TSV 绑定 | Ours 原始运行/预算、第六组消融协议和 Direct 10 分子/50 records 差异仍待核对，详见结果入口 |
| 复现分发 | LLM 36 文件哈希、ML 保存目录清单及 transfer 模型本地缓存清点完成 | 清单不等于完整分发；先对应论文实际运行，再闭合模型/排序缓存依赖 |
| 匿名发布 | 已清理部分链接/路径及内部指令；完成当前文本与全部 83 个分卷文件的限定关键词检查 | FINAL 及旧构建清单确认含身份路径，冻结发布副本与模型地址仍待处理；二进制内容尚未全审，提交作者按用户要求不匿名化 |

当前继续不依赖 PARCC 文件的代码/文档清理、匿名检查及离线验收。PARCC 数据读取与
恢复延至合作者交接，不作为当前整理工作的前置条件；最终发布前仍须核对论文所用
模型/缓存及完整依赖，并做独立环境少量验收，再正常合并 main，保留双方 Git 历史。

## 范围与不变项

- 只迁移当前论文正文、附录和图表所需能力。main 提供 ML/naive LLM baselines；
  Joseph 提供 transfer、context optimization、Ours 与消融，保留其其他既有功能。
- naive LLM 保留三骨干、两 benchmark、六任务的 None / Direct50 / Direct50+Indirect50，
  两池均 Neighbor-fill。12 cells 用 adapter v1、96 用 v2，均保留原消息和原生输出。
- Direct-3 **LLM**、完整历史预算扫描和 progressive/Joint 不新增迁移入口；ML 的固定
  k=3 及 RASAR 三个直接邻居特征继续保留。None 的旧 `molecule_cap` 字段仅为空输入标记。
- split、标签、原始结构、历史预测/分数不变；新检索输出使用新运行身份。冻结 prior
  已嵌入执行包，不另建生成器。Joseph FINAL 默认模板不能替代 main 的论文 prompt。
- 工具服务与 Joint 保持停止；不因复现依赖检查自动下载模型、重建历史实验或全量推理。

## Baseline 配置与验收

三种 ML 训练池为 conditioned、TDC-only、TDC + Starling。下表是既有验收的汇总，
不是本次重新运行的结果；具体入口和参数只在对应能力文档维护。

| 能力 | 已验证范围 | 边界 |
| --- | --- | --- |
| Morgan / MiniMol embedding KNN | 72 cells / 25,992 条 test 预测；分数、标签、邻居、相似度一致 | 使用冻结 embedding；原固定 k=3、阈值 0.5 |
| MiniMol head | 18 cells 保存模型重放；3 cells × 5 seeds × 25 epochs 重训一致 | task 原参数、val BCE checkpoint、ensemble val Macro-F1 阈值；Joseph 固定 epoch/0.5 模式仍可显式选择 |
| MiniMol 微调 / indirect | 21 套准备输入一致；36 cells 重放分类标签/F1 一致；一个真实 seed 的 25 epochs 训练逐步一致 | 不是全候选/全种子重训；CPU/CUDA 数值差见下文 |
| RASAR / RASAR+indirect | 21 套特征/候选 donor 一致；72 cells 重放；6 cells / 180 seed fits 重训通过 | 保留 6 / 6+2B 特征及 val 选参/阈值；搬移另核验 60 个保存 seed 模型 |
| naive LLM | 38,457 次冻结消息重渲染、回复解析和标签/指标核对；4,273 个 prior 对应 2,054 个缓存源文件 | 保存输入/输出验收，不是新模型性能；TDC prior 原生成 preparation manifest 未独立追溯 |
| 执行与搬移 | 12 组 API/Batch 本机模拟、格式修复、零新增请求恢复；36 个 None 组合最小环境搬移；3 个完整索引的单 query 重建 | 完整索引 smoke 为 conditioned Skin、TDC Skin/DILI；Batch 未调用真实远端 |

MiniMol 精确重训必须保留每个 split 的原 embedding 顺序和字节；按 SMILES 复用
其他批次的近似缓存曾引起一条标签变化。8 个真实分子的重新编码最大误差约
3.5×10⁻⁶，仅证明容差内一致。Encoder 在 CPU 上重放原 CUDA 模型时，8 cells 的
AUROC 因近并列样本排序改变；最大差 −0.0010101，分类标签不变。同一 CPU 上原代码
与迁移代码预测一致；差分回执保留，不用新分数覆盖论文。

真实 OpenRouter 小样本：None Skin 成功且恢复零调用；Carcinogens Direct+Indirect
两次均耗尽原 16,384 输出 tokens，未通过完整 JSON 校验。未为通过验收调整证据、
预算或 schema，未继续大量调用。历史回复回放成功不能替代此项在线验收。

### 冻结配置定位

下列路径相对保留的原 main 工作目录，由
`tools/chembl_tool/paper_experiments/current_conditioned_results.json` 绑定，
不是整合分支默认输入。缺失时不能改用 Joseph `CURRENT`。

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

历史差分覆盖 36 个 split 文件哈希、5,884 个专用分子身份、3,675 次通用和 900 次
专用判断；无 split/标签改写。本次为修正文档复核现有身份/结构表示测试，69 项通过。
这些运行时规则不会补回旧索引已排除的记录，也不证明 Joseph 命名缓存已按新规则重建。

2026-10-25 的本地清点与 PARCC 只读核对：

- 五个 reranker 注册模块共 32 个精确模型版本（含历史角色）：16 个具备权重、配置及
  tokenizer 文件，2 个 snapshot 不完整，14 个未在检查范围内找到。未加载模型、
  全量哈希权重或闭合自定义代码依赖；这不是论文必须发布的模型清单。
- 原样展开 Git 中三份 relocation 映射后，沿现有 resolver 检查默认
  `scaffold_disjoint_parent100_v2`：12 个 task/benchmark 的 24 个 retrieval index
  和 6 个 optimizer direct-card index，合计 30 个，在当前 checkout 和 Git 中均未找到。
  PARCC 的源仓库 HEAD 为 `73bddf2e`，catalog 与本地 SHA-256 相同：11 个 Gold 索引
  可读，6 个 TDC 索引确认存在但不可读，其余 13 个因目录权限无法确认文件状态。
  可读索引引用的 75 处下级清单（73 个不同路径）均受权限阻塞，未验证完整缓存。
- BBB v10 `level_mapping/manifest.json` 已在 PARCC 确认存在（2,919 bytes，权限 0600），
  当前账号无法读取，因此尚未恢复本地或重跑相关 9 项 builder 测试。按用户决定，
  读取权限、清单/payload 校验及恢复留给合作者交接阶段，不再为此反复重试或扩大搜索。
  另两项旧 V10.4 检查缺外部训练 release manifest，不自动扩展为论文验收任务。

上述是 Joseph 默认路径的可用性核对，不证明论文用了该默认值，也不影响已经核对的
main baseline 输入。PARCC 源目录和逐项权限清单保存在内部回执，不重复写入公开说明。

模型评分与读取已算好的排序缓存需要不同依赖，见
[模型合同](../predict/retrieval/assay_reranking/MODEL_CONTRACTS.md)。先绑定论文运行，
再选择恢复入口与匿名分发范围；不替换为名字相近的缓存或虚构模型下载地址。

## 可维护性与匿名发布

生产代码按能力共用入口，没有新增按任务/模型复制的 runner。已撤回论文不用的选择
算法并移除失去用途的单调用包装。2026-10-25 维护审查相对 Joseph 的生产 Python
净增 3,132 行、测试净增 1,551 行；这是当时的体量快照，不是当前实时统计或零膨胀证明。
变更模块中未发现至少六行、函数体完全一致的重复函数；不代表不存在语义重复。

数据处理总览已改为引用唯一证据库目录说明，补齐 V9/V10 与 shared/v2，移除不存在的
`versions/vN/shared/` 及旧 pilot 命令。BBB 分类器中无调用者的个人路径常量已删除，
其余 AST 不变。两项旧测试错误地把 gold 条件合格当作 L1 投票身份，已改为同时覆盖
voter/nonvoter，17 项分类回归通过；未改变分类规则或新增执行入口。

匿名清理不全局替换冻结路径/模型 ID。当前跟踪树（含稀疏文件）共 8,312 项；同一组
账号/路径关键词在普通文本中命中 464 个候选文件，修复后为 459 个。已清理绘图说明、
gold 说明和 UID 来源说明中的实名署名/仓库链接；保留精确提交与源码路径，五个定位
均可读取。UID 测试命令也已修正。两处 `zstd` 查找改为 PATH 优先、当前 Python 环境
兜底，相关 12 项测试通过，包含真实打包/恢复。没有新增运行入口或改动科学参数。

全部 83 个分卷文件已按实际格式审查：67 个文件组成 29 套 tar 归档；另 16 个是
7 组可独立读取的 Parquet 分片，不能拼接成一个压缩流。分卷/整包及声明的 stage
文件哈希全部符合原清单。检查了 303,384 个归档成员元数据、268,961 个文本文件；
16 个 Parquet 分片的元数据与字符串字段共 4,313,565 行也已扫描，无本组关键词命中。
归档内 73 个二进制文件仅核验哈希，其他未分卷二进制内容仍未完整审核。

明确未解决的匿名问题是：FINAL 归档有 67,628 个文本成员命中，另有 13 份 V7 构建
清单含个人路径；这主要是路径在多份记录中重复，并非同等数量的独立问题。抽查 FINAL
确认涉及 cache/query/manifest 路径字段，另含冻结组装源码；不能据抽查断言全部命中
都只是元数据。ClinTox 历史包的 22 个成员还保留非空 owner/group 名称。原归档字节
均未改写，当前不能匿名发布；后续须生成可审查的发布副本并核对变换范围、来源映射
及科学内容一致性。实名模型地址/协议 ID 需分别处理，不能全局字符串替换。

内部 AGENTS/CLAUDE 及 12 份改名的历史指令已取消跟踪并保留本地原件；Git 历史未
重写。此次未发现归档成员名为 AGENTS/CLAUDE/INSTRUCTIONS 的 Markdown 文件，
但限定关键词与文件名检查不是完整匿名认证。live trace 的 public projection 同样
不保证匿名，使用说明只在 [predict README](../predict/README.md) 维护。

入口文档先前已修正质量评估模块路径、TSV 输出名及凭据路径，现有测试与三行离线
CLI 通过。6 个独立 gzip 的头及 4 个 tar 成员元数据另有此前回执，正文不属于本次
分卷扫描范围。旧检查结论仅按证据索引查阅，不再在这里重复各轮过程。

## 验收证据索引

原第 1–19 批完整叙述可用 `git show a2d6c4a1:docs/release_refactor.md` 查看，不再复制
为另一篇活跃日志。下表指向保留的内部审计目录，非公开下载地址；部分含原路径或
模型账号，不直接发布。除前两行外，目录均在 `outputs/analysis/release_refactor/` 下。
测试数属于各次检查，覆盖可能重叠，不相加为全库测试总数。

| 审计目录 | 内容 |
| --- | --- |
| 共享盘 `refactor/20261021T014255` | main 源码快照、首批迁移与配置保护 |
| 共享盘 `refactor/20261022_identity` | split/身份差分与 120 项相关检查 |
| `20261022_ml` / `20261022_indirect` | 198 ML cells 重放、有限重训、数值差异解释 |
| `20261023_docs` / `20261023_llm` | 306 cells 来源核算、冻结请求/回复全量核对 |
| `20261023_preparation` / `20261023_smoke` | 候选/预算差分、API 本机回放与恢复 |
| `20261023_portability` / `20261023_transports` | 最小环境、Batch/竞速/阶段恢复模拟 |
| `20261023_relocation` | RASAR 搬移及真实 OpenRouter 成功/失败样例 |
| `20261023_retrieval_relocation` | 三个完整索引搬移和单 query 准备 |
| `20261023_paper_scope` / `20261023_figures` | 论文范围收敛、消融归档核对、三 PDF 像素一致 |
| `20261023_priors` / `20261024_release_inputs` | prior 来源、LLM 36 文件哈希、ML 保留清单 |
| `20261025_maintainability` | 重复代码审查、158 项通过及 9 项缺输入失败 |
| `20261025_release_paths` | 配方参数差分、28 项通过、18 个稀疏文件原样恢复及匿名候选扫描 |
| `20261025_model_release` | 模型限定缓存清点、trace 发布 12 项测试 |
| `20261025_docs_refresh` | 合同/文档去重、默认缓存绑定检查、身份测试与链接核验 |
| `20261025_parcc_inputs` | PARCC 同版 catalog、索引/下级清单及 BBB 映射的存在性与权限检查 |
| `20261025_data_docs` | 数据目录说明与链接检查、无用常量删除的 AST 等价及分类回归 |
| `20261025_instruction_cleanup` | 12 份内部指令原件保留、清单一致性与测试代码未变核验 |
| `20261025_release_entrypoints` | 入口文档目标检查、质量评估离线 CLI、独立 gzip/tar 头信息检查 |
| `20261025_release_audit` | 全跟踪树候选清单、83 个分卷的内容/哈希审查、FINAL 身份字段样例、zstd 可移植性与 12 项测试 |
