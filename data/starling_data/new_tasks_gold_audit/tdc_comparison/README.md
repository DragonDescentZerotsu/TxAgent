> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

> Historical comparison: this directory pins the pre-repair v2 gold/split hashes.
> Source contracts v2.1 subsequently applied these24 diagnostic reviews through
> `../v2_repair/`; current counts are in `../REBUILD_V2.md`. No TDC labels were merged.
> Re-running compare.py now would produce a new comparison, not reproduce this snapshot.

# TDC 与 Starling DILI / Carcinogens 分子比对

结论：TDC 的负类确实很多。我们当前 gold 的高度阳性偏向由来源选择、记录重复、严格且仍有错误的过滤，以及条件覆盖选择共同造成。不能再简单解释为“原始数据没有负例”。本次比对没有用 TDC 标签覆盖 gold，也没有重建 split。

## 官方 TDC 数据与统计单位

依据 [TDC 官方 metadata](https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py) 的下载 ID，直接下载 [DILI 原始表](https://dataverse.harvard.edu/api/access/datafile/4259585) 与 [Carcinogens_Lagunin 原始表](https://dataverse.harvard.edu/api/access/datafile/4259570)。原文件和 SHA-256 在 `acquisition.json`；不安装或改变现有环境。

| 原始 TDC 表 | 行数 | 正类 | 负类 | 负类比例 |
|---|---:|---:|---:|---:|
| DILI | 475 | 236 | 239 | 50.32% |
| Carcinogens_Lagunin | 280 | 60 | 220 | 78.57% |

[官方说明](https://tdcommons.ai/single_pred_tasks/tox/)列 Carcinogens 为278，但本次官方下载文件实际280行。共享 RDKit parent 标准化得到278个不同 parents，其中2个 parent 各有一条0和一条1；两对原始 SMILES 本身就完全相同，不是标准化造成的冲突：

- Drug_2(1) 与 Drug_98(0)：`UPZFHUODAYGHDZ-RMKNXTFCSA-N`，Starling 中名为 cytembena。
- Drug_4(1) 与 Drug_131(0)：`TVWOWDDBXAFQDG-QURGRASLSA-N`，Starling 同结构的来源名称包含 azorubine/magenta，应另审名称身份。

这能解释本文件的280行与278个唯一结构的差别；不能据此断言官方网页278的统计过程。DILI 有475行、474个 parent，一组重复 parent 的标签一致。两张表均无解析失败。

以下分子级主比对先按完整 parent 去重，并排除 TDC 内部两组矛盾 parent，不对矛盾标签投票：

| Task | TDC 唯一 parents | 内部标签矛盾 parents | 无矛盾正类 parents | 无矛盾负类 parents |
|---|---:|---:|---:|---:|
| DILI | 474 | 0 | 236 | 238 |
| Carcinogens_Lagunin | 278 | 2 | 58 | 218 |

每个原始 TDC 条目的结果仍全部保存在 `*_molecule_comparison.csv/jsonl`，包括上述冲突和重复。`comparison_summary.json` 同时提供原始行口径与无冲突 parent 口径。

## 对齐方式与覆盖

复用项目 `rdkit_fragment_parent.v1`：去盐、标准化并中和 parent，用完整 InChIKey 做精确匹配；不按名称或字符串 SMILES 猜测。另报 InChIKey 第一段的 connectivity 匹配，放宽立体化学/同位素等区别，仅作缺失诊断，不能授权 exact identity gold。计数是 TDC parent 被覆盖数，不是 Starling records 数；所有 records 包括有身份审核标记的保留行，名称是否正确仍需单独核验。

| Task | TDC label | TDC parents | 任一来源 records 有该 parent | base 有该 parent | 完整 gold 有该 parent | benchmark 有该 parent |
|---|---:|---:|---:|---:|---:|---:|
| DILI | 0 | 238 | 134 | 104 | 26 | 22 |
| DILI | 1 | 236 | 157 | 144 | 90 | 81 |
| Carcinogens_Lagunin | 0 | 218 | 127 | 47 | 10 | 6 |
| Carcinogens_Lagunin | 1 | 58 | 30 | 22 | 3 | 3 |

专门追踪 TDC 负类到当前 gold 阴性的去向：

| Task | 匹配口径 | TDC 负类 parents | 任一来源 records | base | base 已有负向记录 | 通过候选阴性规则 | 通过身份核验的阴性票 | 完整 gold 含阴性条件 | benchmark 含阴性条件 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DILI | 完整 parent identity | 238 | 134 | 104 | 79 | 4 | 4 | 4 | 2 |
| DILI | connectivity 诊断 | 238 | 209 | 179 | 144 | 10 | 9 | 9 | 4 |
| Carcinogens_Lagunin | 完整 parent identity | 218 | 127 | 47 | 29 | 2 | 2 | 2 | 1 |
| Carcinogens_Lagunin | connectivity 诊断 | 218 | 156 | 63 | 40 | 3 | 3 | 3 | 1 |

例如精确对齐后，DILI 的79个 TDC 负类 parent 已有 Starling 负向记录，却仅4个进入完整 gold 阴性；Carcinogens 为29→2。主要损失发生在候选语义规则之前/之中，而不是最后的身份核验。放宽为 connectivity 后仍是144→9和40→3，所以立体结构差异只能解释部分覆盖差距，不能解释全部阴性损失。

## 为什么我们的正类这么多

1. **采集目标偏向有害关系。** 冻结的 DILI extraction guidance 顶层任务为“Find all molecules known to cause drug-induced liver injury (DILI) within PubMed.”，Carcinogens 为“Find all known carcinogens reported within PubMed.”。两者允许抽取明确负向结论，但不是按 TDC 正负分子清单系统检索的对照集合。对最终偏向的贡献是根据指导文本与覆盖数据作出的解释，不能据此单独量化全部选择偏倚。
2. **原始 records 不等于独立分子。** 正向热门分子反复被抽取。按 base parent 看，DILI 有至少一条正向记录的 parent2375个、有负向记录1980个，其中1121个两种方向都有；Carcinogens 分别5947、3699、2082。两方向可来自不同条件，不能直接当作同条件矛盾。原始记录的阳性比例夸大了分子集合的不平衡。
3. **当前阴性合同远窄于 TDC 的分子级分类。** 我们拒绝“未报告风险”、纯酶学随访、单个病例排除归因、局部器官阴性、体外阴性，并对权威分类/综述要求还原原研究。部分拒绝合理；部分会漏掉可还原的真实阴性，如明确的两物种长期阴性或 NTP 阴性分类。上述79→4、29→2显示这些门槛仍严重影响召回，不能由53项软件测试证明语义正确。
4. **条件覆盖进一步缩小阴性集合。** 已有完整 gold 保留225/461个阴性 parent-condition rows，但只有45/60行能进入现有三路覆盖子集。稀疏条件没有删除，但大量合格来源并未贡献当前评估集合。
5. **v2 仍存在阳性来源与条件错误。** 本轮抽查确认，部分综述、旧案例汇总仍按当前 PMID 计成原研究，白血病发生率等结果仍能进入 query condition。此前206条抽样审核并没有验证所有候选；当前 v2 不能称为完成语义验证的 gold。

## 相同分子的标签不一致

针对无冲突的 TDC 负类 parents，按完整 parent 匹配：

| TDC 负类集合 | 我们 gold 仅有阴性条件 | 同时有正/负条件 | 仅有阳性条件 | 不在完整 gold |
|---|---:|---:|---:|---:|
| DILI 的 TDC 负类 | 1 | 3 | 22 | 212 |
| Carcinogens_Lagunin 的 TDC 负类 | 2 | 0 | 8 | 208 |

DILI 的22个仅阳性 parent 包括 aspirin、amitriptyline、citalopram、propafenone、tacrine；Carcinogens 的8个包括 streptozotocin、azathioprine、melatonin、gibberellic acid。它们是待核查的标签差异，不能全部算我们的误标，也不能自动认为 TDC 错误：TDC 是分子级标签，Starling 的剂量、物种、给药条件与时间来源可能不同。

本轮另外逐条阅读24条来源文本：每个任务6条“对应 TDC 负类但 gold 仅阳性”的票，以及6条“已有负向来源却被拒”的记录。完整原文、UID/hash、原始判定和逐条说明位于 `reviewed_samples.jsonl`；这是 Codex 源文本审核，不是真人专家或原始论文全文核验。

| 样例 | 比对与审核结果 |
|---|---|
| Aspirin，DP2 | TDC0；来源明确是5岁儿童过量/中毒后肝昏迷，不能不带条件直接对比普通分子标签。 |
| Amitriptyline / Citalopram，DP4/DP5 | TDC0；所保存病例段落有明确临床损伤、病理及时间线支持，不能为了与TDC一致而直接改0。 |
| Tacrine，DP1 | 正向安全性概述与未指明的旧试验被计成当前 PMID 独立投票；需要修复来源追踪。 |
| Azathioprine / Melatonin，CP3/CP4 | 旧案例/旧研究汇总被计成当前论文投票；leukemia 发生等结果还混进条件。应撤回/修正相关来源票与条件后再重建。 |
| Chlorpheniramine，CR4 | 明确两物种两年阴性，却因物种未拆分被挡住；值得还原原试验或审核为明确 pooled 条件。 |
| Lithocholic acid，CR5 | 明确 NTP negative/no-evidence 分类，被 authoritative-classification basis 门槛整体拒绝；应追原 NTP 研究，而非忽略其阴性证据。 |
| Ketamine，DR5 | 混合综述同时包含酶学信号与60人 CRPS 试验的阴性结果；可拆分、追溯其中阴性臂，不能把整段作为新独立研究。 |
| Temozolomide / Kaempferol，DR3/CR6 | 前者为单例排除归因，后者为体外转化阴性；拒绝当作普通人群/整动物整体阴性有语义依据。 |

在这24条针对性样本中，5条已有阳性票有来源/条件问题，4条拒绝样本有值得追溯的阴性子结论，另有2条体现端点合同差异。抽样刻意富集问题类型，这些比例不是全量错误率估计。

## 建议与交付文件

目前应把 TDC 分子级标签与自建 condition gold 视为不同的 benchmark 合同。若目标是直接比较 TDC performance，应冻结 TDC cohort/labels 单独评估，Starling 作为证据库并遵守 held-out 防泄漏；自建 condition gold 则先修正上述来源/条件问题，再按已经找到的 TDC 负类证据审核召回。调整类别比例本身不能证明标签质量。

- `dili_molecule_comparison.csv`、`carcinogens_molecule_comparison.csv`：可筛选的逐 TDC 条目对照，保留原始 SMILES、label、parent、Starling 名称、各层覆盖和正负数量。
- 同名 `.jsonl`：完整匹配字段，另含 connectivity 对照和原始负向记录的第一道拒绝理由。
- `comparison_summary.json`：全量计数、TDC 内部矛盾与输入哈希。
- `reviewed_samples.jsonl`、`review_manifest.json`：24条逐条语义审核，未直接应用到 gold。
- `validation.json`：计数守恒、匹配包含关系、review hash 及现有 gold/split 输入哈希校验。
- `compare.py`：从仓库根执行 `PYTHONPATH=. python data/starling_data/new_tasks_gold_audit/tdc_comparison/compare.py` 可重现统计，复用已有身份，不重新构建records。
