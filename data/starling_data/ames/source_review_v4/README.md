# Ames 双向逐条证据审核（2026-09-06）

本轮核对现有实际 L1 的可靠性，以及 L2 中被漏收的直接实验。新增逐条阅读 144 条完整抽取记录：96 条 L1（72 条按标签×活化分层，24 条疑点优先）和 48 条 L2（四个来源各 12 条）。与上一轮 144 条样本无重叠，另外复读并批准上一轮六条候选。

审核者为 Codex agent。这不是人类专家标注，也不是对全部论文进行全文复核。144 条均阅读完整提供的记录；重点阅读 30 条记录对应的原论文摘要，并对四篇原论文的相关方法、结果、表格或图注作全文定位核查。另检查了已确认错误的同 PMID/parent 传播记录。不能把分层和疑点优先样本的比例外推为整个库的错误率。

## 已落实的改变

- 15 条记录通过逐条语义审核后进入统一候选流程；最终新增 12 个独立 study-condition 票，3 条已被同研究同条件覆盖。其中 PhIP 和 3-methylindole 的跨来源描述补入已有票的支持来源；quercetin 的描述原已被代表。
- 撤回 25 个旧票：7 个为确认的论文对象错认，18 个为原研究、实验条件或试样身份待核实。后一组留在 L2；这些实验尚未被证明无效。
- 对 39 条具体 source records 应用已审查的身份隔离，36 条此前可检索；原始记录不删除。隔离范围为核对过的 PMID×错误 parent，逐条 payload hash 必须一致。
- 实际 L1 从 3,346 变为 3,333 条，覆盖 1,423 个投票 parent。标签/条件聚合后有 2,474 个样本、1,383 个 parent、73 个条件。新增 3 个样本，移除 16 个；保留样本标签不变。
- 1,299,474 条记录仍可检索（原始记录的 86.03%），L1–L5 分别为 3,333 / 138,568 / 227,291 / 419,791 / 510,491（heldout 删除前）。

## 关键判读依据

| 问题/案例 | 判读与处置 | 原研究 |
|---|---|---|
| OA | 原文为 okadaic acid，抽取错写 ochratoxin A；错误 parent 的票和证据隔离。 | [PMID 1658641](https://pubmed.ncbi.nlm.nih.gov/1658641/) |
| COL | 原文为 collagenase 酶制品，总有机固体剂量；不能投给 colforsin 小分子。 | [PMID 32278234](https://pubmed.ncbi.nlm.nih.gov/32278234/) |
| NIDA | 原文为 N-nitrosoiminodiacetic acid，所给结构为不同的硝基咪唑；名称查 PubChem 通过仍不能授予正确论文对象。 | [PMID 3009488](https://pubmed.ncbi.nlm.nih.gov/3009488/) |
| 2-aminoanthracene 正对照 | 原全文 Table 2/Figs 6–10 是五菌株 +S9 实测；-S9 用了别的正对照。纠正原抽取的 both 为 present，按 PMID 合并为一票。 | [PMID 29632622](https://pubmed.ncbi.nlm.nih.gov/29632622/) |
| 4NQO / Trp-P-1 | 有剂量、菌株、S9 和实际回复菌落数；允许实测正对照投票，不能仅凭“用作正对照”几个字投票。 | [PMID 30268793](https://pubmed.ncbi.nlm.nih.gov/30268793/)、[PMID 2067537](https://pubmed.ncbi.nlm.nih.gov/2067537/) |
| TNX weak positive | 作者明确报告 TA100+S9 弱致突变且无细胞毒性；低效力不自动等于 equivocal。 | [PMID 17360228](https://pubmed.ncbi.nlm.nih.gov/17360228/) |
| 1-nitropyrene-2-ol | +S9 实测阳性独立于机制中 possibly 的猜测；原 -S9 阴性票不覆盖 +S9 条件。 | [PMID 6720027](https://pubmed.ncbi.nlm.nih.gov/6720027/) |
| AMP397 | 两种方法的 +S9 结果均阴性；方法并列本身不阻止该一致的实验臂。 | [PMID 12113769](https://pubmed.ncbi.nlm.nih.gov/12113769/) |
| 3-methylphenanthrene | 高剂量图省略因为细胞毒性，但图注明确另有至少四个非毒剂量阴性；保留原票，不能因截取文字看似矛盾就删掉。 | [PMID 35182162](https://pubmed.ncbi.nlm.nih.gov/35182162/) |
| PubMed Review 标记 | Aspartame 和 135-compound comparative assay 文章实际报告原实验；不按类型标签机械剔除。 | [PMID 29408486](https://pubmed.ncbi.nlm.nih.gov/29408486/)、[PMID 6374443](https://pubmed.ncbi.nlm.nih.gov/6374443/) |

## 投票与条件边界

新增的完整 panel 可以包含明确的 YG1021/YG1024/YG1029 或 TA104 回复突变菌株。不能删掉陌生菌株后伪造简化 panel，也不能把一条 pooled record 拆成多票。四个新 panel 仍不足三个独立 parent 和三个非空 scaffold，因此经过条件支持 gate 后没有增加最终 benchmark 的条件数；实际 voter 身份不会因后续 group gate 撤销。

来源 run 不是决定直接性的依据。所有 source 都可通过具体、可追溯的审核决定进入候选；仍须名称与结构匹配、实验对象正确、明确实测 outcome、可恢复条件。最终一 PMID×parent×exact condition 最多一票，同 study 冲突不投；有外部条件至少 60% 一致性，无条件至少 70%，均排除 tie。

源内容含有直接 Ames 结果、相关预测或混合背景时，不允许降入 L3–L5。没有成为独立票的合格重复描述可以留 L2。prediction-only 不能生成 gold，但可以按其 endpoint 进入检索。

## 当前 splits

| Split | Train 样本 / parent | Val 样本 / parent | Test 样本 / parent |
|---|---:|---:|---:|
| scaffold | 1,926 / 1,102 | 274 / 137 | 274 / 144 |
| random | 1,980 / 1,217 | 247 / 80 | 247 / 86 |

每个 split 均有 73 条件；所有 parent 整组划分，scaffold 方案另约束非空 Bemis–Murcko scaffold 不交叉。2,230/2,474 样本只有一个 study 票；一致性阈值不是最低三篇研究要求。当前 gold 是已检查的 source subset，不能称为完整或已全量人工核验的 Ames benchmark。

## 审计文件与复现

- `sample.jsonl`：本轮 144 条完整原始记录及抽样分层。
- `audit_annotations.jsonl`：逐条判读、原始 payload hash、构建后实际去向。
- `applied_decisions.jsonl`：73 条具体应用动作（15 accept、19 withhold、39 exclude）；同问题的传播记录和重复描述也单列，不等于 73 个独立票。
- `vote_changes.json`：新增/撤回票和已有票来源变化，包含完整条件与身份。
- `previous_source_votes.jsonl`、`previous_benchmark_rows.jsonl`：变更前对照；原始四份 Parquet 和 PubChem 响应保持不变。
- `pubmed_*.xml`、`*_full.xml`：实际取得的公开原始文献响应。只把文章自身 PubmedData/ArticleIdList 当作 PMID/PMCID/DOI，不使用 reference list 内其他文献的 ID。
- `summary.json`：计数、输入/审计文件哈希和限制；最终技术验证看 `revision_receipt.json` 和上级 `dataset_validation.json`、`retrieval_validation.json`。

抽样已从冻结的旧 votes、L2 frame 和既往 sampled IDs 重新运行，144 条输出字节级相同：

```sh
python data/starling_data/ames/source_review_v4/sample_selection.py /local/tmp/ames-review-replayed
```

完整重建仍使用 task 原有入口 `python -m tools.chembl_tool.tasks.ames.build_dataset --phase all --workers 16`；审核决定不绕过 identity、study collapse、condition support 或 heldout filtering。

## 技术验证

`revision_receipt.json` 已写出 `status=passed`。279 项相关测试通过；完整源记录与 scaffold/random 索引的代表卡片来源、source counts、实际 L1 membership、heldout 删除和累计层级可见性检查均通过。两种 split 共六条 query 的 30 个 level 检索检查通过，97 个其他任务的 benchmark 文件哈希保持不变，未运行模型评估。

最终验证对七个完整文本字段完全相同的输入复用同一独立规则扫描结果；每条源记录、每张卡片的来源匹配与所有断言仍执行。`retrieval_validation.json` 记录 1,293,657 次独立计算与 2,414,204 次相同输入复用。审核摘要已补齐验证器所需的 `review_method` 字段，并重新完成全套验证。

回执中 `source_checks.gold_votes_unchanged` 表示同一次校验期间 vote 文件未变；相对上一轮的 12 票新增、25 票撤回分别见 `vote_changes.json` 和明确标记为 false 的 `preservation.source_votes_unchanged_since_retrieval_revision`。

## 144 条逐条判读

| # | Record | 原层 | 审核判断 | 构建后 | 判读理由 |
|---:|---|---|---|---|---|
| 1 | `ames_base:96596` | L1 | retain | L1 实际票 | TA98 -S9 实测阴性；最高剂量35对照12.7接近三倍，保留作者结论并记录需表格复核，不能自设两倍阈值翻成阳性。 |
| 2 | `ames_base:135295` | L1 | pending_source | L2 | 支持文本明确是implying推断-S9阴性；摘要列出了其它阳性菌株但未直接给出该阴性臂，需原表确认。 |
| 3 | `ames_base:217903` | L1 | retain | L1 实际票 | TA98/TA100 -S9三次独立实验未检出突变；原研究摘要直接支持，与+S9阳性臂分开。 |
| 4 | `ames_base:105244` | L1 | retain | L1 实际票 | 19种已测试芳胺全部在TA98/100 -S9阴性，覆盖6-aminochrysene；不是把其它物质阴性外推。 |
| 5 | `ames_base:214973` | L1 | retain | L1 实际票 | dictamnine -S9的TA98计数与溶剂对照比较给出了独立阴性臂，不被本分子+S9阳性覆盖。 |
| 6 | `ames_base:517` | L1 | retain | L1 实际票 | NTA两次独立实验、312.5–5000微克剂量系列四菌株-S9均无生物学显著增加。 |
| 7 | `ames_base:96393` | L1 | retain | L1 实际票 | 日文表1为benzidine本身TA100 -S9计数，不是大鼠尿液混合物读数；背景和五剂量清楚。 |
| 8 | `ames_base:105914` | L1 | retain | L1 实际票 | Acridine(2)在TA1537 -S9明确未增加回复菌落；DNA结合是伴随机制，未替代标签。 |
| 9 | `ames_base:161238` | L1 | retain | L1 实际票 | Orange II未还原、无预孵育的标准平板TA100 -S9明确阴性；反映该实验臂。 |
| 10 | `ames_base:111796` | L1 | retain | L1 实际票 | DAB在TA98 -S9明确报告阴性，data not shown不等于预测；论文其它norharman实验不改变该独立臂。 |
| 11 | `ames_base:26803` | L1 | retain | L1 实际票 | CP原药TA1535 -S9表中a标记有明确阴性定义；未借用病人尿液或代谢物结果。 |
| 12 | `ames_base:93494` | L1 | pending_source | L2 | 文章为3-MCPD/glycidol酯综述，原文抽取称recent experiments，尚未建立独立原研究身份，不能当该PMID新票。 |
| 13 | `ames_base:38070` | L1 | retain | L1 实际票 | Triallate在TA1537/1538 ±S9均阴性，原表2与condition一致。 |
| 14 | `ames_base:257707` | L1 | retain | L1 实际票 | Thioacetamide五个数量级剂量TA98/100 ±S9阴性；果蝇引文为对比，不构成该Ames结果的来源。 |
| 15 | `ames_base:69664` | L1 | pending_source | L2 | Drometrizole安全性评估汇总外部实验，需原试验study标识，不能把综述PMID当独立实验。 |
| 16 | `ames_base:245796` | L1 | quarantine_identity | 不检索 | 原论文1658641明确OA=okadaic acid，抽取错写ochratoxin A并配后者结构；名称匹配PubChem不修复原文对象误认。 |
| 17 | `ames_base:131006` | L1 | retain | L1 实际票 | Itrabin四个菌株/活化臂均接近对照，MR文字0.82–1.15漏了表中1.20但不改变阴性结论。 |
| 18 | `ames_base:125041` | L1 | pending_condition | L2 | 氯仿gas exposure且另有GSH补充臂，不能只用TA100×both_reported混入普通条件，需暴露方式并分开特殊活化。 |
| 19 | `ames_base:53648` | L1 | retain | L1 实际票 | PubMed虽然标Review，摘要明确报告两项本次实验；aspartame预孵育Ames阴性，不能按文献类型机械删除。 |
| 20 | `ames_base:179618` | L1 | retain | L1 实际票 | Patulin TA102 ±S9回复突变明确阴性；不借用其它毒素或终点。 |
| 21 | `ames_base:51157` | L1 | quarantine_identity | 不检索 | 32278234测的是微生物collagenase(COL)酶制品，总有机固体剂量；抽取误认小分子colforsin。 |
| 22 | `ames_base:223138` | L1 | retain | L1 实际票 | 纯TCDD三菌株±S9最高非毒剂量及背景清楚，按作者2.5倍标准阴性；溶解度解释不等于实验无效。 |
| 23 | `ames_base:128609` | L1 | retain | L1 实际票 | Pyrazine三菌株±S9两次实验各三平板，低于细胞毒性范围也无增加，保留实测阴性。 |
| 24 | `ames_base:83290` | L1 | retain | L1 实际票 | Teniposide WP2uvrA ±S9明确未诱导回复菌落；不与哺乳动物实验混淆。 |
| 25 | `ames_base:250784` | L1 | pending_condition | L2 | BaP仅位于底板，上板细菌接受挥发部分；阴性只描述taped-plate暴露，缺暴露condition不能当普通BaP阴性。 |
| 26 | `ames_base:311996` | L1 | retain | L1 实际票 | Chlorogenic acid本身TA100突变在加入S9后完全消失；独立化合物活化臂，不是抑制AFB1结果。 |
| 27 | `ames_base:190316` | L1 | retain | L1 实际票 | 日文明确TA100敏感而TA98无敏感性，TA98阴性有对应活化背景；不能用TA100阳性替代。 |
| 28 | `ames_base:102179` | L1 | retain | L1 实际票 | 8-AcAQ +S9在TA100弱阳性而TA98阴性，当前仅TA98臂正确。 |
| 29 | `ames_base:248490` | L1 | pending_condition | L2 | Safrole阴性使用3-MC诱导rat-liver S13，不是当前普通S9概括；需要激活制备条件。 |
| 30 | `ames_base:215741` | L1 | retain | L1 实际票 | 2,6-dimethylaniline在TA98 +20% rat S9明确阴性，系列成员归属明确。 |
| 31 | `ames_base:234439` | L1 | retain | L1 实际票 | 未还原Congo red TA1538 +S9接近扣背景零值；细菌预还原产物结果没有冒充母体。 |
| 32 | `ames_base:15048` | L1 | retain | L1 实际票 | 三氯甲烷TA98 +S9阴性与原研究题目及摘要对象一致，非混合消毒副产物。 |
| 33 | `ames_base:153405` | L1 | retain | L1 实际票 | BHP在rat-S9液体孵育TA98阴性、TA100阳性，当前TA98臂可辨识。 |
| 34 | `ames_base:101646` | L1 | retain | L1 实际票 | N,N'-diacetylbenzidine在TA102 +rat S9明确阴性，结构及命名一致。 |
| 35 | `ames_base:227906` | L1 | pending_condition | L2 | 原研究摘要明确mouse-liver S9和mouse hepatocyte两种激活体系；现条件未保存物种/细胞制备，需拆出准确体系。 |
| 36 | `ames_base:65049` | L1 | retain | L1 实际票 | Disperse Orange 1在TA100 +S9阴性，明确不同于同菌株-S9阳性。 |
| 37 | `ames_base:44633` | L1 | retain | L1 实际票 | GMA三个菌株-S9实测2–8倍回复突变，超过剂量后毒性不否定较低剂量阳性。 |
| 38 | `ames_base:152879` | L1 | pending_source | L2 | 核酸化学修饰综述中的反应混合物与N4-aminocytidine产量相关，尚缺纯化合物当前实验和原study身份。 |
| 39 | `ames_base:192129` | L1 | pending_source | L2 | 废水识别表中的glyoxal有具体数值，但PMID为鱼色素细胞肿瘤环境研究，需核验数值是本次纯化物还是外部表引用。 |
| 40 | `ames_base:136325` | L1 | retain | L1 实际票 | 原研究测六种亚硝酸酯且均Salmonella阳性；n-amyl nitrite TA100 -S9抽取为对应实验臂。 |
| 41 | `ames_base:38488` | L1 | pending_source | L2 | 主体论文为CHL突变，Salmonella表2是否重新实测以及-S9依据需原表核验，摘要本身只作背景。 |
| 42 | `ames_base:18676` | L1 | retain | L1 实际票 | MNU TA1535 -S9在150微克给180倍增加，明确当前实验测量。 |
| 43 | `ames_base:66746` | L1 | retain | L1 实际票 | Ro60-0213 TA1537 -S9剂量依赖，最低明确阳性250微克/plate；药物系列比较不影响本臂。 |
| 44 | `ames_base:246345` | L1 | quarantine_identity | 不检索 | 原论文NIDA=N-nitrosoiminodiacetic acid；SMILES是硝基咪唑类而非该二羧酸，确证对象-结构冲突。 |
| 45 | `ames_base:224500` | L1 | retain | L1 实际票 | BaP4,5-oxide本身TA1537无哺乳激活阳性，酶水解保护研究不否认化合物单独阳性臂。 |
| 46 | `ames_base:8936` | L1 | retain | L1 实际票 | 虽PubMed标Review，摘要明确对135化合物进行比较实验并计算净回复数；MNNG条目不能按Review标签误删。 |
| 47 | `ames_base:166236` | L1 | pending_source | L2 | 抽取只是MMS的Ames方法和浓度，没有报告单独处理后的明确结果；需原文图表确认，不能因MMS常识阳性投票。 |
| 48 | `ames_base:59546` | L1 | retain | L1 实际票 | 合成N-OH-Trp-P-2 -S9有两剂量实测回复数；melanoidin抑制是另一个臂，基线实际测量可投票。 |
| 49 | `ames_base:160560` | L1 | retain | L1 实际票 | FP12 TA100无S9直接阳性，S9降低活性但原pooled含两臂；保持panel-any-positive语义，不宣称每臂阳性。 |
| 50 | `ames_base:30430` | L1 | retain | L1 实际票 | paepalantine原研究9261922已有三菌株±S9真实票；之前v1综述15720259转引的结果不能额外增加study票。 |
| 51 | `ames_base:187148` | L1 | retain | L1 实际票 | Hydralazine TA100/1537 ±S9实测阳性，其它三个菌株阴性与当前双菌株条件不冲突。 |
| 52 | `ames_base:136354` | L1 | retain | L1 实际票 | iso-amyl nitrite表III ±S9阳性、CAS明确；与n-amyl nitrite是不同分子。 |
| 53 | `ames_base:53407` | L1 | retain | L1 实际票 | TMTD普通激活和无激活TA100阳性；缺NADP抑制为另外的机理臂，当前可限定原正常体系。 |
| 54 | `ames_base:293308` | L1 | retain | L1 实际票 | 硝基蒽醌无S9多个菌株高水平阳性；文中明确全panel±S9被测试，pooled阳性不要求每个臂阳性。 |
| 55 | `ames_base:98513` | L1 | pending_condition | L2 | 原研究TA98使用rabbit鼻/肺/肝S9；当前both_reported漏掉不同组织物种，虽均有直接阳性，需明确激活体系。 |
| 56 | `ames_base:215218` | L1 | retain | L1 实际票 | 1,3,5-trinitrobenzene五菌株均阳性，S9降低幅度；保持原pooled实验单元。 |
| 57 | `ames_base:102896` | L1 | retain | L1 实际票 | 稳定重氮四氟硼酸盐TA1535/1537 ±rat-S9阳性，按存活比例分析仍为直接回复突变终点。 |
| 58 | `ames_base:147611` | L1 | retain | L1 实际票 | 七种明确合成nitronaphthofuran四菌株阳性且TA1535阴性，当前不含TA1535，普通S9降低活性未抹去结果。 |
| 59 | `ames_base:281194` | L1 | retain | L1 实际票 | HMF图中±S9 TA100回复数且讨论明确本次发现致突变；不是单凭assayed方法判阳性。 |
| 60 | `ames_base:228689` | L1 | retain | L1 实际票 | PHPNT TA100 ±S9明确剂量依赖阳性，TA98阴性不在当前condition。 |
| 61 | `ames_base:189811` | L1 | retain | L1 实际票 | 2-aminoacridine TA1538 +S9有多个剂量数值至>1000回复菌落，明确活化臂阳性。 |
| 62 | `ames_base:158578` | L1 | retain | L1 实际票 | Galangin TA98 +S9独立剂量反应阳性；alpha-naphthoflavone抑制为另臂，不能因此删掉单药实验。 |
| 63 | `ames_base:322295` | L1 | retain | L1 实际票 | AFG1 +S9表中100微克明确+，其它剂量-；接受作者判断并保留单剂量阳性的局限，不另设普适阈值。 |
| 64 | `ames_base:176771` | L1 | retain | L1 实际票 | 2-aminofluorene在rat-liver S9本身明确激活阳性；同时列肝细胞的效率比较不改变可独立识别S9臂。 |
| 65 | `ames_base:54188` | L1 | pending_identity | L2 | Miral500CS商品制剂与纯活性物质不是同一试样；摘要确认使用该制剂，需组成/纯物质实验才能作parent票。 |
| 66 | `ames_base:221443` | L1 | retain | L1 实际票 | 1,9-dimethylfluorene TA98 +rat-liver homogenate明确阳性，有本研究化合物归属。 |
| 67 | `ames_base:124988` | L1 | retain | L1 实际票 | Naphthacene两种常规诱导rat-S9下四菌株阳性且计数/单位充分；保留reported pooled activation语义。 |
| 68 | `ames_base:313503` | L1 | retain | L1 实际票 | Compound2名称在记录中明确展开、结构匹配，九个aminoquinoxaline系列均+S9阳性，可定位到该分子。 |
| 69 | `ames_base:170344` | L1 | pending_condition | L2 | 2AA未诱导rat-S9与Aroclor诱导体系活性明显不同，需保存当前control-S9条件而非普通present。 |
| 70 | `ames_base:130316` | L1 | retain | L1 实际票 | Nitrosodi-n-butylamine TA1535 +Aroclor-S9明确实验阳性，癌性比较不作label依据。 |
| 71 | `ames_base:228726` | L1 | retain | L1 实际票 | N-acetyl-o-dianisidine TA98 +S9四平板/剂量且有对照与剂量反应；比母体稍弱并不等于阴性。 |
| 72 | `ames_base:238494` | L1 | pending_condition | L2 | 当前阳性是30%glucose预处理供体S9强化臂，需把供体处理记录进condition后再投票。 |
| 73 | `ames_base:48019` | L1 | retain | L1 实际票 | Pretomanid两个Ames实验涵盖五菌株±S9，沉淀但无明显毒性、正负对照有效，不能仅因沉淀删阴性。 |
| 74 | `ames_base:284272` | L1 | pending_condition | L2 | 6-TOOH的Aroclor与phenobarbital S9在TA100相反；同条件pooled掩盖已知解释性差异，需激活诱导剂条件。 |
| 75 | `ames_base:10222` | L1 | retain | L1 实际票 | CI921 TA98 ±S9非毒剂量接近对照，1400微克完全毒性另记；阴性不是仅由死亡造成。 |
| 76 | `ames_base:98231` | L1 | retain | L1 实际票 | BPF包含在本次八种BPA类似物检测中，TA98/100 ±S9均阴性；化合物归属明确。 |
| 77 | `ames_base:322555` | L1 | retain | L1 实际票 | MRS5698非GLP仍有四菌株、±S9、剂量系列/对照和无剂量趋势；非GLP不是自动低质量。 |
| 78 | `ames_base:186711` | L1 | retain | L1 实际票 | 1-nitroso-2-naphthol是已识别光反应产物本身的-S9测试，并非必须照光的Ames实验；保留化合物直接阴性。 |
| 79 | `ames_base:81872` | L1 | retain | L1 实际票 | 1,4-DMPh两日独立重复TA98/100 +S9明显剂量反应，较高剂量毒性不否定低剂量阳性。 |
| 80 | `ames_base:105657` | L1 | pending_condition | L2 | test_system/extra写TA97而support写TA97a，菌株condition内部冲突，需原methods核定后修正，不能悄悄当同菌株。 |
| 81 | `ames_base:43364` | L1 | retain | L1 实际票 | o-PD TA98 +S9 3967对照22，明确阳性；与p-PD比较和TA100弱阳性没有错绑。 |
| 82 | `ames_base:185950` | L1 | retain | L1 实际票 | UR144 Ames MPF五菌株±S9明确阴性；MPF是实验不是模型prediction，MN/comet阳性不改变Ames阴性。 |
| 83 | `ames_base:190475` | L1 | retain | L1 实际票 | U64273A TA97/98 +S9剂量依赖阳性，TA100未达该研究三倍阈值不纳入阳性菌株条件。 |
| 84 | `ames_base:22358` | L1 | retain | L1 实际票 | 2-dDCB四菌株±S9三剂量不显著且对照完整；阳性对照本身需独立实测记录才能成为其它分子的票。 |
| 85 | `ames_base:81253` | L1 | retain | L1 实际票 | Acrolein diethylacetal TA1535 -S9 10–18对照12，明确定义的阴性实验。 |
| 86 | `ames_base:149313` | L1 | retain | L1 实际票 | Apocynin低/中剂量四菌株±S9无增加；4000微克毒性单独描述，不因高剂量死亡推导全部阴性。 |
| 87 | `ames_base:228744` | L1 | retain | L1 实际票 | DCBZ TA100 +S9两次独立实验接近对照；TA98阳性属于其它condition。 |
| 88 | `ames_base:114308` | L1 | retain | L1 实际票 | 原全文Fig5/6说明高剂量图省略是细胞毒性，但另外至少四个可分析非毒剂量在TablesS2/S3均阴性，澄清源文看似矛盾；保留原阴性票。 |
| 89 | `ames_base:276779` | L1 | retain | L1 实际票 | BaP在不同S9浓度的标准/spiral实验均实测阳性；改变potency不能自动抹去明确总体阳性。 |
| 90 | `ames_base:160699` | L1 | retain | L1 实际票 | NPIP原论文摘要确认本次Ames与gpt实验，源文reviewed字样不是综述引用，TA100/1535活化阳性明确。 |
| 91 | `ames_base:32802` | L1 | retain | L1 实际票 | Glutaraldehyde -S9在MPF及标准pre-incubation均阳性并重复；不能把concordant方法比较当不可用。 |
| 92 | `ames_base:152618` | L1 | retain | L1 实际票 | BaP单独基线三菌株+S9两实验实测，混合物研究里独立单药臂可投票。 |
| 93 | `ames_base:32312` | L1 | retain | L1 实际票 | Wogonin纯度>99%三菌株±S9无剂量依赖回复增加，染色体弱阳性是别的终点。 |
| 94 | `ames_base:256441` | L1 | retain | L1 实际票 | Pyrogallol TA100 -S9实测剂量趋势阳性；357/81不等于所写MI11.2，保留标签但标注数值转录不一致。 |
| 95 | `ames_base:246940` | L1 | retain | L1 实际票 | Verruculogen无S9多个菌株有原表阳性判定；TA100低剂量无生长抑制，非仅毒剂量噪声。 |
| 96 | `ames_base:217285` | L1 | pending_condition | L2 | 原论文采用密闭惰性孵育系统且与标准测试不同，iodoethane阳性需保留密闭暴露条件。 |
| 97 | `ames_base:215914` | L2 | candidate_extended_panel | L1 实际票 | TA98/YG1021/TA100 -S9全部有明确定义的N.D.，YG工程菌本身仍是回复突变，可保留完整panel后投票，不丢弃未知菌株。 |
| 98 | `ames_base:314721` | L2 | candidate_extended_panel | L1 实际票 | PBTA1四菌株+S9具体净回复数、两次独立实验，含YG不应整条删去；保存完整panel不擅自拆票。 |
| 99 | `ames_base:49122` | L2 | pending_source | L2 | Kojic acid TA1537阴性来自安全综述，需追踪原试验和低于毒剂量的结果。 |
| 100 | `ames_base:63913` | L2 | already_represented | L2 | 原全文Table2明确quercetin TA102 -S9阴性，已由ames_base:63902贡献同study-condition一票；补充来源不增加票。 |
| 101 | `ames_base:247454` | L2 | accept_corrected_condition | L1 实际票 | 原全文Table2/Figs6–10确认2AA仅在+S9用于五菌株，非-S9；修正活化为present，24–27实验室结果合并仍只按PMID保留一票。 |
| 102 | `ames_base:246860` | L2 | candidate_richer_condition | L2 | Trans4acetamidostilbene TA100实测阳性但使用hamster-liver S9，需保存激活物种不能混普通rat-S9。 |
| 103 | `ames_base:256101` | L2 | pending_source | L2 | 1,6-diNP有明确-S9参考剂量和抑制比较，但没有单独基线数值/效应明示，需原图确认实测无herb臂。 |
| 104 | `ames_base:247704` | L2 | candidate_extended_panel | L1 实际票 | 3MC三菌株+S9具体回复率、相同比较并非阴性；需保留YG完整panel和实测对照来源。 |
| 105 | `ames_base:219554` | L2 | pending_source | L2 | 4NQO无样本臂定义100%只说明归一化基准，不能单凭正对照名称投票；查原图/表实测回复数。 |
| 106 | `ames_base:15855` | L2 | candidate_richer_condition | L2 | 1,4-dichlorobutene2在mouse/human-S9均有实测回复数，需激活物种/供体诱导和cofactor条件。 |
| 107 | `ames_base:42536` | L2 | candidate_extended_panel | L1 实际票 | MNU TA100/TA104 ±S9实测明确突变且S9不改变结果，未知TA104不能导致整条丢弃，保存全panel。 |
| 108 | `ames_base:256160` | L2 | candidate | L1 实际票 | BaP Aroclor-S9 TA98剂量反应明确，qualifying仅描述正常激活；需核定无污染处理control臂与去重复。 |
| 109 | `ames_v1:39516` | L2 | not_direct_vote | L2 | 主体是酵母D7光诱导ILV回复及抗氧化干预，附带TA102未给独立8MOP标准Ames条件结果。 |
| 110 | `ames_v1:22815` | L2 | pending_identity | L2 | 正文cefpodoxime proxetil与SMILES无proxetil酯不符，而且文章为review；伴随Ames确实阴性但不能绑定该母体结构。 |
| 111 | `ames_v1:22469` | L2 | not_direct_vote | L2 | Saccharomyces D4酵母，作者也未给突变结论，不能据对照近似当Ames阴性。 |
| 112 | `ames_v1:51687` | L2 | pending_source | L2 | Dodecanenitrile伴随Ames阴性来自RIFM综述，缺全菌株/激活/原study，不能用V79条件补Ames。 |
| 113 | `ames_v1:16545` | L2 | quarantine_identity | 不检索 | PAN气体为peroxyacetyl nitrate，SMILES为过氧乙酰类且无氮；主体转基因小鼠也不产生Ames票。 |
| 114 | `ames_v1:26866` | L2 | not_direct_vote | L2 | Vibrio harveyi BB7XM不是当前Salmonella/WP2体系，附带TA100没有结果数值；不能把Vibrio阴性移用。 |
| 115 | `ames_v1:90000` | L2 | not_direct_vote | L2 | Linalool降低tBOOH诱导回复数是保护效应，不是linalool本身阴性或阳性；百分比结果不能换主语。 |
| 116 | `ames_v1:15680` | L2 | not_direct_vote | L2 | TM677选择8-azaguanine抗性是forward mutation，TA1535 his+仅菌株来历。 |
| 117 | `ames_v1:39636` | L2 | not_direct_vote | L2 | Proflavin诱导ara操纵子突变和pseudo-wild-type回复不是Ames氨基酸回复检测。 |
| 118 | `ames_v1:29514` | L2 | pending_source | L2 | CV的TA1535 slight increase是旧研究引用，缺S9/剂量/判定，酵母阴性不能作为CV Ames阴性。 |
| 119 | `ames_v1:61301` | L2 | not_direct_vote | L2 | ENU是线虫unc93回复突变与序列谱，非细菌Ames；术语revertant不能决定直接性。 |
| 120 | `ames_v1:84190` | L2 | not_direct_vote | L2 | Lactose是营养选择条件，FC40适应性lac回复不是lactose致突变实验。 |
| 121 | `ames_v2:61873` | L2 | already_represented | L2 | 原摘要及记录确认PhIP TA98+S9实测回复；frozen PubChem匹配PhIP，但ames_base:105873已覆盖该study-condition，只增加支持来源。 |
| 122 | `ames_v2:501442` | L2 | candidate_richer_condition | L2 | NPYR TA1975有扣自发背景净34回复且100%存活；需要hamster-S9与TA1975完整condition、同位素试样身份。 |
| 123 | `ames_v2:409457` | L2 | pending_source | L2 | Trenbolone伴随TA100剂量反应很像direct，但PMID是会议摘要合集，需具体摘要作者/study标识和-S9臂证据。 |
| 124 | `ames_v2:330981` | L2 | pending_identity | L2 | BGE被展开为epichlorohydrin可疑，需原文名称定义；尚未确证身份错误，不作已确认quarantine。 |
| 125 | `ames_v2:478507` | L2 | pending_source | L2 | Apigenin伴随四菌株±S9阴性来自Czeczot1990，应该追到原PMID去重而非以2007文章新票。 |
| 126 | `ames_v2:425362` | L2 | quarantine_identity | 不检索 | 正文N-OH-AAF带fluorene骨架，而SMILES为单苯环羧酸衍生物；确证骨架不符。 |
| 127 | `ames_v2:279355` | L2 | pending_source | L2 | Benzylidene acetone +S9有327回复/微摩尔但未给Ames菌株；SOS阴性不可填作Ames标签。 |
| 128 | `ames_v2:52673` | L2 | pending_source | L2 | DMBA较BP低的突变活性仅综述ref164比较，缺菌株/原实验/绝对判定。 |
| 129 | `ames_v2:62496` | L2 | pending_source | L2 | Dieldrin三种DNA断裂阴性无Amespanel结果；摘要提Ames测试条件不足以补出直接票。 |
| 130 | `ames_v2:324903` | L2 | pending_source | L2 | Naltrexone其它plate assays阴性很可能包含direct，但缺完整菌株和本次reverse结果表，不能把polA修复指数当回复突变。 |
| 131 | `ames_v2:363607` | L2 | not_direct_vote | L2 | DMSO仅vehicle/control方法用途，没有独立剂量系列和实测化合物阴性结论。 |
| 132 | `ames_v2:302456` | L2 | pending_source | L2 | Rutin有Ames potent revertant induction字样但菌株、激活与剂量不明；TK6条件不可移用。 |
| 133 | `ames_v3:266376` | L2 | quarantine_identity | 不检索 | 7-HBA=7-hydroxymethylbenz[a]anthracene，SMILES只有环上OH没有CH2OH，确证官能团和对象不符。 |
| 134 | `ames_v3:159525` | L2 | candidate_richer_condition | L2 | AFB1 glucose供体S9强化TA100回复与现L1同研究对应，需供体处理condition并去重，不能加普通S9新票。 |
| 135 | `ames_v3:35502` | L2 | not_direct_vote | L2 | BHT提高其它化合物的活性/结合属于modifier；而3,3'-dichlorobenzene名称本身可疑，不据此投BHT标签。 |
| 136 | `ames_v3:93473` | L2 | not_direct_vote | L2 | Emodin抑制IQ活化是secondary机制，不是emodin单独Ames阴性。 |
| 137 | `ames_v3:241677` | L2 | pending_identity | L2 | 结构是AFB1-GSH conjugate，正文主体为AFB1及其形成产物；不可把母体活性绑定代谢物结构。 |
| 138 | `ames_v3:113473` | L2 | not_direct_vote | L2 | Styrene oxide仅给GLC消失速度及GSH干预，未给可独立判定的TA100回复突变实验臂。 |
| 139 | `ames_v3:330770` | L2 | not_direct_vote | L2 | Melatonin机制推测和引用P450变化，不是自身Ames阴性或新的单药实验。 |
| 140 | `ames_v3:11361` | L2 | not_direct_vote | L2 | 离体肺混合代谢物提取液Ames不能投给原始BaP结构；无纯品独立臂。 |
| 141 | `ames_v3:2875` | L2 | already_represented | L2 | 原摘要确认3MI TA100+S9无突变；frozen PubChem匹配3-methylindole，ames_base:26685已覆盖该study-condition，只增加支持来源。 |
| 142 | `ames_v3:218403` | L2 | quarantine_identity | 不检索 | NO-HEX原研究为N-nitrosohexamethyleneimine环状亚硝胺，所给SMILES为开链多酰胺，确证骨架冲突。 |
| 143 | `ames_v3:387709` | L2 | not_direct_vote | L2 | 无突变的是AFB1葡萄糖醛酸/硫酸结合物，不能把阴性投给AFB1母体；水解释放才是另对象/条件。 |
| 144 | `ames_v3:66674` | L2 | pending_source | L2 | IMI为2009人白细胞MN/comet引用，伴随TA菌株结果未给出，需原文；不可复制其S9条件给Ames。 |
