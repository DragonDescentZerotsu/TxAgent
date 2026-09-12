> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI / Carcinogens 初版 gold 审核（2026-09-07）

**结论：初版 gold/split 未通过语义审核，不可用于正式评估，需要重建。** 原始 records 保留完整；问题在 gold 筛选、研究去重和条件归一化，不是源数据只有个位数负例。

## 原始 base 分布

这里统计原始抽取记录，不是独立分子、独立研究或已验证 gold。原始字段并非二分类；不确定、冲突、缺失值单列。正向分组沿用初版映射以便追踪，不能代表因果主张已核实。

| Task | 原始记录 | 初版正向类别 | 明确负向类别 | 其他 | 正∶负（仅这两组） | 负向占全部 | 负向占正负两组 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dili | 189,172 | 104,029 | 15,563 | 69,580 | 6.68∶1 | 8.23% | 13.01% |
| carcinogens | 354,386 | 236,310 | 33,727 | 84,349 | 7.01∶1 | 9.52% | 12.49% |

DILI 正向合计 = established_or_definite 98,209 + highly_probable 630 + probable 5,190；负向 = not_supported 15,563。其余包含 association_signal 36,369、suspected_cause 20,108、possible 11,719、conflicting 1,256，以及缺失/非规范值。

Carcinogens 正向合计 = positive 219,420 + established 16,890；负向 = negative 33,727。其余包含 limited_or_suspected 51,387、hypothetical 10,857、possible 9,684、equivocal_or_conflicting 4,906、inadequate_or_not_classifiable 4,293、probable 3,113，以及缺失/非规范值。

## 每一阶段正负流失

| 阶段及计数单位 | DILI 正 / 负 | Carcinogens 正 / 负 |
|---|---:|---:|
| 原始base记录（仅初版正负类别） | 104,029 / 15,563 | 236,310 / 33,727 |
| 逐条语义/身份/置信度门槛后候选记录 | 4,734 / 160 | 13,344 / 437 |
| 按 PMID-parent-condition 合并后的票 | 4,578 / 157 | 12,199 / 436 |
| 同parent+逐字条件至少两PMID后保留票 | 1,287 / 38 | 1,007 / 9 |
| 提前要求条件覆盖三parents/三scaffolds后的票 | 1,156 / 38 | 606 / 6 |
| 名称结构及再次支持门槛后的source votes | 985 / 30 | 494 / 6 |
| 多数表决/condition coverage后molecule-condition rows | 158 / 9 | 108 / 3 |

注意最后一行切换成聚合后的 molecule-condition row，不能把 30→9 或 6→3 全部称作“删掉原始 records”。

## 已确认的问题

1. **误排整个 DILI 负向证据类别。** 原始 guidance 的 human_evidence_basis 明确规定主张不支持时使用 explicit_negative_evidence。初版 PRIMARY 白名单不包含它。15,563 条原始负向中 8,235 条属于该类；首个 not_primary_human_outcome gate 共挡住 10,394 条负向。这是选择规则错误，但该类仍含无既往报道、单病例排除、转述等，不可直接全部加 gold。
2. **额外双 PMID 硬门槛过严，又没有确保研究独立。** 去重复之后，DILI 负向157→38，Carcinogens436→9。共用builder允许singleton并在split时优先安排高支持质量；这次额外强制双PMID是初版新增的限制。不同PMID可重复转述同一实验，故不能称minimum_independent_pmids为已验证独立研究数。
3. **逐字条件当身份导致碎片化，缺失字段又导致错误合并。** 将整段qualifying_conditions逐字放入condition ID、提前套三parents/三scaffolds，使明确条件实验难以保留；但qualifier=null时，extra中的关键人群、给药途径、疗程、基因背景又丢失。需要语义标准化，不能清空条件来增加数量。
4. **正则既误杀又漏掉限定/来源问题。** D31的suggested只描述风险区间，实测无肝毒性的结论明确，却全段排除；D28的neither hepatotoxicity未被负向正则认出。Carcinogens的15,686条negative在首个负向正则gate被排除，但本次该层样本多为部位限定、零关联或修饰作用，不能把这15,686全部视为可恢复的总体负例。
5. **目标不对称必须明确。** 原始Carcinogens是agent×tumor/outcome claim；初版改成any-site hazard后，一个部位阳性可支持危险性、一个部位阴性却不足以支持总体阴性。这一目标选择会结构性压低负类召回。不能为了正负平衡把部位阴性改成总体阴性，也不能未经说明更换评估目标。
6. **DILI的未观察到事件不等于无致病能力。** 特定短疗程、有限样本中的阴性与明确药物致病病例不能在条件不明时简单多数对冲。DILI负例需要明确研究观察范围、测量及负向主张，而非一个人的无事件就给普遍安全票。

## 逐条语义抽查

共88条：DILI56、Carcinogens32。先按六个预先定义的类别，各task以SHA256(audit_v1|UID)排序取32条，再补全其余24条DILI负票。覆盖当前全部30条DILI和6条Carcinogens负向source votes，以及12条已接受正票和40条被排除/后续流失记录。

**审核人身份说明：这些是 Codex 对完整 source support_text、qualifying_conditions、extra_details 和结构字段的逐条语义阅读，不是人类专家审核，也未核验原始论文全文。** purposive抽样不估计总体错误率；review为候选恢复或待核实的记录均未自动加票。

代表性可复核发现：

- C04/C05：不同PMID 14555420、4018315明确指向Til 1972a同一研究；sodium metabisulfite被重复计成两项支持。
- D06/D43/D44：不同PMID重复RE-LY；D45/D46还用dabigatran etexilate名称记录同一研究，须核对真正投药实体，PubChem名称匹配不能解决这个问题。
- C14：momfluorothrin的78周CD-1小鼠全组织阴性具有完整条件，却在双PMID阶段流失。
- D13：52人最长两年benorylate试验的明确条件内阴性也在双PMID阶段流失。
- D41/D42：一日itraconazole疗程和半年病例被合入同一普通治疗条件。
- C10：PPARγ杂合小鼠的关键基因背景未进条件。
- D49：汇总移植试验的449总人数与分组167+166+166=499不一致，说明高抽取confidence不能代替文本和源研究审核。

## 处理状态与重建要求

已在两个dataset manifests及全局benchmark manifest中标记 blocked_pending_gold_semantic_rebuild。当前votes和split字节保持不变以便追踪本次审核，不能再称为通过语义审核的正式gold。本次没有运行模型，也没有为两个任务发布检索index。

重建顺序：修复source类别解释和逐条claim语义 → 区分阴性observed-outcome与普遍hazard → 抽取并规范化真正改变结论的条件 → 按原始study而非转述PMID去重 → 保留合格singleton及稀疏条件的审核数据层 → 独立执行split可行性/质量约束 → 检查正负coverage与heldout泄漏。身份、归因、真实负向终点和研究去重标准仍然严格；不按目标比例人为改标签。当前抽查不能预测重建后负类分子总量。

计数/审核文件：`counts.json`、`reviewed_records.jsonl`（完整原记录、UID、payload SHA256及逐条结论）、`manifest.json`（审核范围、分层方法、哈希）。

## 抽查明细

| ID | Task | 名称 | PMID | 原阶段 | 审核结论 |
|---|---|---|---|---|---|
| C01 | carcinogens | benzaldehyde | 12848242 | accepted_negative | 方向为大鼠阴性，但没有试验出处/剂量；与C02可能同NTP试验，独立性无法证实，需原研究ID。 |
| C02 | carcinogens | Benzaldehyde | 16835129 | accepted_negative | 转述NTP评估而非当前PMID新实验；与C01须去重；明确阴性方向不能证明两项独立支持。 |
| C03 | carcinogens | Naproxen | 2107193 | accepted_negative | 药品label转述2年大鼠试验，非当前PMID新实验；与C06同样描述，不能凭两篇文章认定两试验。 |
| C04 | carcinogens | Sodium Metabisulfite | 14555420 | accepted_negative | 明确引用Til, Feron, DeGroot 1972a，和C05同一研究；两PMID重复计票。饲料/剂量/三代试验条件须保留。 |
| C05 | carcinogens | sodium metabisulphite | 4018315 | accepted_negative | 明确引用Til et al 1972a，与C04同一研究；thiamine-enriched diet等条件漏入extra，不能再给独立票。 |
| C06 | carcinogens | Naproxen | 8380424 | accepted_negative | 两年naproxen大鼠试验的简要陈述；与C03可能同一label来源，独立性未证实。 |
| C07 | carcinogens | N-ethyl-N-nitrosourea | 22949413 | accepted_positive | ENU鼠乳腺肿瘤的一般性叙述并引文，不足以证明当前PMID独立原试验；激素/繁殖/剂量影响不能忽略。 |
| C08 | carcinogens | Toxaphene | 18941969 | accepted_positive | 明确转述NCI1979和Litton1978，当前PMID不能自动加独立票；Toxaphene确切试样组成也未在提供字段中解决。 |
| C09 | carcinogens | N-nitroso-N-methylurea | 33842308 | accepted_positive | NMU小鼠乳腺癌模型的一般描述，无当前试验设计，不能按structured bioassay直接认证primary voter。 |
| C10 | carcinogens | azoxymethane | 11530334 | accepted_positive | PPARγ杂合基因背景是核心实验条件却qualifier=null；不得汇入普通mouse条件。Recent animal studies也需检查原研究来源。 |
| C11 | carcinogens | benzo[a]pyrene | 10887479 | accepted_positive | 明确benzo[a]pyrene诱导大鼠乳腺瘤，阳性claim支持；提供文本未说明是否当前PMID原实验，独立票需来源复核。 |
| C12 | carcinogens | 2-amino-1-methyl-6-phenylimidazo[4,5-b]pyridine | 8352890 | accepted_positive | extra明确本研究F344大鼠9个结肠肿瘤，支持阳性；model只记rat而漏F344，须确定该模型条件的规范化合同。 |
| C13 | carcinogens | Troglitazone | 12075125 | negative_lost_after_candidates | troglitazone大鼠、47倍暴露内任何肿瘤均无增加，明确全身范围内阴性；需原试验身份核验，不应仅因单PMID和独有字符串条件丢弃。 |
| C14 | carcinogens | momfluorothrin | 30090614 | negative_lost_after_candidates | momfluorothrin 78周CD-1小鼠全组织阴性、剂量和雌雄齐备；缺第二PMID不应否认这项实验。须保留条件及核验源试验。 |
| C15 | carcinogens | trans-1,4-dichlorobutene | 373114 | negative_lost_after_candidates | 皮肤给药小鼠阴性与同物其他途径阳性分开，符合conditioned目标；不得取消途径条件，但单PMID不等于无效实验。 |
| C16 | carcinogens | α-tocopherol | 3919498 | negative_lost_after_candidates | 纯α-tocopherol单独皮下注射无肿瘤，明确与油组合分开；重要配方/给药条件齐备，可进入研究/身份审核，不应一刀切双PMID。 |
| C17 | carcinogens | Imiprothrin | 32229244 | negative_lost_after_candidates | 作者排除超过MTD的肿瘤后认定阴性，数据有初始趋势且依赖复切片分析；需保留限制并逐研究审核，不能当无条件negative。 |
| C18 | carcinogens | OPP-Na | 4034146 | negative_lost_after_candidates | 明确reference27既有2年试验，现PMID不能作为独立新票；找到原研究后可保留负向证据。 |
| C19 | carcinogens | Pimecrolimus | 18302444 | negative_rejected:negative_not_explicit | 只有皮肤肿瘤无影响，不支持any-site总体阴性；词法确有漏检，但放宽正则仍不能授予总体负票。 |
| C20 | carcinogens | isoproterenol (DL-isoproterenol hydrochloride) | 120430 | negative_rejected:negative_not_explicit | DMBA肿瘤过程中的修饰/抑制作用，不是isoproterenol单独全面致癌阴性。 |
| C21 | carcinogens | Nitroso-3,4-dibromopiperidine | 3319273 | negative_rejected:negative_not_explicit | 对照句明确二溴化合物未诱发鼻黏膜肿瘤，但仅一部位；不能当any-site阴性。 |
| C22 | carcinogens | EPTC | 23697536 | negative_rejected:negative_not_explicit | 无显著白血病风险为特定终点的流行病学零结果，不是全癌种无致癌性。 |
| C23 | carcinogens | Calcium carbonate | 7998589 | negative_rejected:negative_not_explicit | chalk职业暴露对膀胱癌的归因反驳，不足以给纯calcium carbonate的所有癌种负票。 |
| C24 | carcinogens | Atrazine | 12448356 | negative_rejected:negative_not_explicit | triazine多重暴露工人前列腺癌死亡无关联；既非确切atrazine单独归因也非总体致癌阴性。 |
| C25 | carcinogens | Radioactive iodine (I-131, RAI) | 17393376 | negative_rejected:negative_not_explicit | RAI后膀胱癌风险不升高，不代表其他癌种阴性；同位素投药身份另需确认。 |
| C26 | carcinogens | 1,8-Dihydroxyanthraquinone | 23262639 | negative_rejected:negative_not_explicit | 小鼠肠道肿瘤阴性、仍有增生且为Mori1986转述；可保留endpoint证据但非any-site独立负票。 |
| C27 | carcinogens | benzoyl peroxide | 3091706 | negative_rejected:site_limited_negative_not_overall_hazard | Panoxyl配方、UV相关双臂、少量非显著皮肤癌；不能直接给benzoyl peroxide全身负票。 |
| C28 | carcinogens | sodium hippurate | 1509119 | negative_rejected:site_limited_negative_not_overall_hazard | 有明确膀胱肿瘤阴性但未报告全组织阴性；促癌模型与长期bioassay需分开，当前不满足any-site负票。 |
| C29 | carcinogens | Di-(2-ethylhexyl) adipate | 7030887 | negative_rejected:site_limited_negative_not_overall_hazard | F344 rat仅肝肿瘤阴性，对比mouse阳性；保留物种和部位，不能直接总体阴性。 |
| C30 | carcinogens | Ethinyl estradiol | 13181359 | negative_rejected:animal_species_missing_or_pooled | 物种是明确hamster却被记为species_missing_or_pooled，理由错误；但实为肾模型与引用旧研究，仍不能直接any-site投票。 |
| C31 | carcinogens | malathione | 19157067 | negative_rejected:animal_species_missing_or_pooled | 仅rodent未明确种、仅肺肿瘤，不能推定rat/mouse或任何部位阴性。 |
| C32 | carcinogens | Arsenic | 8041912 | negative_rejected:animal_species_missing_or_pooled | 猴物种明确，初版只允许rat/mouse而错误记作missing；需审查混合猴种和arsenic确切化学形态，不能直接恢复。 |
| D01 | dili | tofacitinib | 33906853 | accepted_negative | 临床试验明确报告48周无DILI，但限于强直性脊柱炎人群、5 mg BID和随访期；qualifying_conditions为空，不能视作一般人群无风险。 |
| D02 | dili | aprocitentan | 36356632 | accepted_negative | 明确无肝毒性，但限于难治性高血压、背景降压治疗和12.5/25 mg；与D54的PRECISION试验描述高度一致，不能凭不同PMID认定独立。 |
| D03 | dili | Pioglitazone | 16188168 | accepted_negative | 明确T2DM单药治疗下无DILI；应补适用人群。与D04的结论近乎相同，需要核对原试验，不能直接加票。 |
| D04 | dili | Pioglitazone | 11491207 | accepted_negative | 197人试验明确无黄疸/肝毒性，但T2DM单药条件遗漏；与D03需试验去重。 |
| D05 | dili | Cyclosporine | 6582812 | accepted_negative | 只是在一名患者中未见肝毒性，不能作为普遍无DILI风险的一票与致病病例对冲；需个体限定及负例资格审查。 |
| D06 | dili | Dabigatran | 19717844 | accepted_negative | RE-LY内明确负向结果；与D43/D44重复RE-LY，且D45/D46使用dabigatran etexilate名称，须核验投药实体及试验去重。 |
| D07 | dili | Acetaminophen | 26457748 | accepted_positive | 明确acetaminophen-induced急性肝衰竭，但提供文本未说明普通治疗剂量；不能把结构字段therapeutic_use当作已核实。需暴露核验。 |
| D08 | dili | Phenprocoumon | 1963177 | accepted_positive | 明确phenprocoumon再激发、停药改善和肝炎诊断；支持阳性人类DILI claim。病例年龄等不自动构成仅该年龄有效的条件。 |
| D09 | dili | Halothane | 29516349 | accepted_positive | 20例halothane肝炎且有死亡，支持阳性；重复暴露是易感/加重因素，文本并未把全部病例限于重复暴露。 |
| D10 | dili | Diclofenac | 21128314 | accepted_positive | 同时转述Banks FDA病例和MEDAL试验，不能当作该PMID的一项独立primary trial；人群/剂量/排除既往肝病也未入条件。 |
| D11 | dili | Troglitazone | 35162986 | accepted_positive | 市场撤回与既往致死性肝损伤的概述，缺少当前PMID原始病例数据；阳性方向可保留作证据，primary独立票未证实。 |
| D12 | dili | methimazole | 28804147 | accepted_positive | Zou et al reported提示可能转述他人病例，需确认原始研究；methimazole停药、propranolol继续支持归因，不能仅因合并用药排除。 |
| D13 | dili | Benorylate | 8837 | negative_lost_after_candidates | 52人、最长2年benorylate试验明确未检出肝毒性；保留风湿病/制剂/剂量条件，可作为负向候选，不应仅因缺第二PMID消失。 |
| D14 | dili | Rivaroxaban | 19187276 | negative_lost_after_candidates | 四个RECORD试验的总结；负向临床结论明确，但需研究去重、手术/短疗程条件和现有identity rejection复核，不能直接恢复。 |
| D15 | dili | Galantamine | 11129124 | negative_lost_after_candidates | 引用references 85/87的galantamine试验，当前文章不应自动成为新的独立primary vote；需回到原试验。 |
| D16 | dili | Cyclosporine | 2701722 | negative_lost_after_candidates | 明确限定单个移植患者病程的无肝毒性；不宜当一般药物安全负票。应保留记录及患者限定。 |
| D17 | dili | Naltrexone | 17939765 | negative_lost_after_candidates | 长效肌注naltrexone、至少18个月下明确负向结果；制剂/途径不能丢失，需研究身份审查，但双PMID不应是一刀切门槛。 |
| D18 | dili | Febuxostat | 38002074 | negative_lost_after_candidates | MASLD患者40 mg febuxostat试验明确称无肝毒性，定义基于AST/ALT；需冻结该测量/患者条件并确认符合DILI目标，不能升级成全人群安全。 |
| D19 | dili | coumarin | 14692729 | negative_rejected:not_primary_human_outcome | coumarins类别/跨物种代谢论述、旧文引用及缺暴露，不能直接赋给确切coumarin分子为独立人类负票。 |
| D20 | dili | Cholic acid | 23160874 | negative_rejected:not_primary_human_outcome | 胆汁酸治疗讨论中的assertion而非本研究独立阴性结果，且5β-reductase deficiency限定遗漏；保持待原证据审核。 |
| D21 | dili | Cortisone | 13493701 | negative_rejected:not_primary_human_outcome | 没有肝功能指标异常不等于没有脂肪浸润/全部DILI，且为文献转述，不能作为全面负票。 |
| D22 | dili | Metoprolol | 8628501 | negative_rejected:not_primary_human_outcome | 只排除了metoprolol作为该患者坏死原因，文本反而承认既有急性肝炎报道；不能泛化成药物无DILI。 |
| D23 | dili | 177Lu PSMA-617 | 29873291 | negative_rejected:not_primary_human_outcome | 24人、2周期177Lu PSMA-617无肝毒性是明确范围内阴性，但recent study可能转述；需核验放射性投药实体、随访及原研究。 |
| D24 | dili | Duloxetine | 15818148 | negative_rejected:not_primary_human_outcome | 尚无报道是历史知识缺口，不是足够暴露和观察下的实测负向结论。 |
| D25 | dili | Mifepristone | 23956101 | negative_rejected:not_primary_human_outcome | 单次200/600 mg口服mifepristone未关联DILI的断言，有明确条件，但机制解释非独立临床数据；需原始依据。 |
| D26 | dili | Naltrexone | 12878918 | negative_rejected:not_primary_human_outcome | 16周试验无组间差异不等于无DILI，且与acamprosate组合等多臂混合；须审查具体臂及负向终点，不能按basis类别直接全收。 |
| D27 | dili | Acetaminophen | 17076974 | negative_rejected:negative_not_explicit_or_contains_residual_risk | 明确推荐剂量下无临床肝损伤且extra给出具体试验无肝毒性；初版仅匹配少量措辞而漏掉。七试验汇总须按真实研究去重，保留剂量/人群。 |
| D28 | dili | Fluconazole | 21609268 | negative_rejected:negative_not_explicit_or_contains_residual_risk | neither hepatotoxicity nor...是明确否定而被正则漏掉；两试验总结仍需原研究去重，且移植人群和疗程遗漏，不能原样恢复。 |
| D29 | dili | Paracetamol | 31203255 | negative_rejected:negative_not_explicit_or_contains_residual_risk | Cochrane综述错标为clinical_trial，且无显著组间差异不证明无DILI；不能当独立原试验票。 |
| D30 | dili | Efavirenz | 26400998 | negative_rejected:negative_not_explicit_or_contains_residual_risk | 主要终点为APRI变化，伴随转述既有hepatotoxicity文献；不能把替代指标无变化升级为直接DILI阴性。 |
| D31 | dili | Dilmapimod | 26102252 | negative_rejected:support_context_or_secondary_assertion_requires_review | 实测剂量内明确无肝毒性；suggested描述风险区间推断，不否定已观察结果。全段关键词封杀错误；保留创伤、IV剂量和疗程。 |
| D32 | dili | Copper | 36694841 | negative_rejected:support_context_or_secondary_assertion_requires_review | 委员会汇总三项既有铜暴露研究，且承认检测敏感性/疗程不足，部分饮水环境暴露不属药物DILI；不应新增独立primary票。 |
| D33 | dili | Donepezil | 10179700 | accepted_negative_census_extension | 汇总3项donepezil试验，当前PMID不是自动独立试验；50岁以上AD、5–10mg、12–24周未入条件。 |
| D34 | dili | Donepezil | 10793322 | accepted_negative_census_extension | AD开放延长试验负向结果明确；须补人群/随访并核对D36是否同一延长研究，不能用PMID证明独立。 |
| D35 | dili | Donepezil | 15989517 | accepted_negative_census_extension | 161人phaseII、12周donepezil试验与D37可能同研究；AD/剂量条件遗漏，需原试验去重。 |
| D36 | dili | Donepezil | 9452942 | accepted_negative_census_extension | 133人AD延长试验到98周负向结果；不能泛化一般用药，须核对D34同一试验随访。 |
| D37 | dili | Donepezil | 9853200 | accepted_negative_census_extension | phaseII dose-ranging AD试验；与D35研究身份未分清，保留5mg/d和人群。 |
| D38 | dili | Flucloxacillin | 38391538 | accepted_negative_census_extension | 仅该OPAT cohort未见肝毒性；静脉/门诊抗菌疗程条件遗漏，不能与全部治疗下DILI阳性直接对冲。 |
| D39 | dili | Pioglitazone | 11092281 | accepted_negative_census_extension | 本placebo试验负向与所有试验汇总放在同一passage；只能原研究投票，需核对D03/D04重复及适用条件。 |
| D40 | dili | Pemoline | 8708264 | accepted_negative_census_extension | 40名ADHD大学生中实验室检查未见肝毒性，不是一般人群无风险；需限定观察窗口和endpoint充分性。 |
| D41 | dili | Itraconazole | 22514427 | accepted_negative_census_extension | 仅急性念珠菌阴道炎女性的一日itraconazole疗程阴性，不能合并D42的半年疗程为同一condition。 |
| D42 | dili | Itraconazole | 32256604 | accepted_negative_census_extension | 单名播散性组织胞浆菌病患者、半年itraconazole并用激素；无肝毒性不应作普遍负票，条件也与D41明显不同。 |
| D43 | dili | Dabigatran | 21717198 | accepted_negative_census_extension | phaseII/III总结并含RE-LY数值，与D06/D44的RE-LY并不独立；必须原试验ID及投药实体核验。 |
| D44 | dili | Dabigatran | 21864021 | accepted_negative_census_extension | 同名RE-LY同剂量试验重复D06；不同PMID不能增加独立证据计数。AF/肾功能排除条件也未编码。 |
| D45 | dili | Dabigatran etexilate | 20671015 | accepted_negative_census_extension | dabigatran etexilate下RE-LY，与D46同试验；且D06/D43/D44以另一分子名收录，可能形成投药实体重复，需要核验。 |
| D46 | dili | Dabigatran etexilate | 21438804 | accepted_negative_census_extension | 同RE-LY试验110/150mg，重复D45；不能使同一试验满足两项独立支持。 |
| D47 | dili | Cyclosporin A | 2206973 | accepted_negative_census_extension | 26名重症银屑病、7–37个月cyclosporin研究有明确负向claim；人群/时间条件遗漏，不能等同一般无DILI风险。 |
| D48 | dili | Cyclosporine | 8313821 | accepted_negative_census_extension | 16名Crohn患者肠外给药条件下阴性；不得与银屑病/单病例无条件合并，需负例充分性审核。 |
| D49 | dili | mycophenolate mofetil | 9028422 | accepted_negative_census_extension | 汇总两项移植试验、免疫抑制联合用药条件遗漏；并且449总人数与167+166+166=499不一致，原记录需核验，不能当独立准确primary票。 |
| D50 | dili | rosuvastatin | 39579293 | accepted_negative_census_extension | ROAD特定血脂异常人群观察到无临床显著肝毒性，方向可保留，但需人群/观察时间与因果目标定义，不能普遍负向。 |
| D51 | dili | tofacitinib | 30118353 | accepted_negative_census_extension | OPAL Beyond/Broaden多个试验汇总无DILI，不能当单个新独立PMID票；应与D01不同人群明确区分。 |
| D52 | dili | Namodenoson | 34671996 | accepted_negative_census_extension | 60名NAFLD±NASH三臂phaseII阴性；基础肝病条件遗漏，与D53同剂量/样本数研究高度一致。 |
| D53 | dili | Namodenoson | 38672201 | accepted_negative_census_extension | 60名MASLD phaseII、12.5/25mg与D52高度一致；须试验去重，不能以术语更新和PMID差异算独立。 |
| D54 | dili | Aprocitentan | 37566184 | accepted_negative_census_extension | PRECISION与D02相同药物、难治高血压和剂量，需核实原研究去重；当前不能认定独立。 |
| D55 | dili | Zavegepant | 38223948 | accepted_negative_census_extension | recent phaseIII trial的zavegepant总结，须原试验出处；急性偏头痛/鼻给药10mg的条件遗漏。 |
| D56 | dili | Zavegepant | 38938785 | accepted_negative_census_extension | 已完成phaseII/III的总结非自动primary，需原试验ID；鼻给药/急性疗程与10/20mg条件未编码。 |
