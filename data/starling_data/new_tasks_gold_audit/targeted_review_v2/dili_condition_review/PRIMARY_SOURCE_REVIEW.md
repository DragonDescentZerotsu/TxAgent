# DILI 19 个跨来源冲突：原始出处核验

本轮结论：19 个 TDC=0 均与 Xu et al. 2015 作者原始补充表 S2.1 的 CID、结构、标签一致；18 个分子另有作者明确报告肝损伤的原始临床证据，amphetamine 的原记录出处细节仍待核实。这是对原始报告方向的判断，不是把18个默认条件最终gold直接改为1。

原始论文通常没有机器学习用的Y列。下表“1”表示作者报告该药相关肝损伤；它不表示已完成所有正负研究的综合裁决。读取范围包括正文、摘要和可读取的原文扫描片段，逐项记录在primary_source_adjudications.jsonl；不是19篇全文均已取得。

## TDC 的 0 来自哪里

[Xu et al. 2015原始论文](https://doi.org/10.1021/acs.jcim.5b00238)及[作者补充数据](https://acs.figshare.com/articles/dataset/Deep_Learning_for_Drug_Induced_Liver_Injury/2054931)。本地原件为research/ci5b00238_si_002.xlsx；完整行级对照为research/xu2015_conflict_comparison.json。以下行号都是Excel物理行号，D列为标签，C列为分类描述，A/B列为CID/SMILES。

| 分子 | S2.1 / TDC | 同一份补充文件的另一表 | 结论 |
|---|---|---|---|
| Tacrine | 第23行，Negative / 0 | S1.2第8行，Most DILI-concern / 1；S2.2第7行，Positive / 1 | 作者源表内部也存在相反标签，且CID和SMILES一致 |
| Aminoglutethimide | 第34行，Negative / 0 | S1.2第14行，Most DILI-concern / 1；S2.2第10行，Positive / 1 | 同上 |

这证明冲突不是本项目导入时把0/1翻转。补充表还混合了no DILI-concern、NE和Negative等原来源分类；不能把每个0都解释成一项实际观察到无肝损伤的临床实验。本轮不臆测作者为什么在不同表中保留相反值。

## 逐分子结果

| 分子 | 原S2.1行号 / 标签 | 临床原始报告方向与内容 | 原始出处 | 核验限制与条件 |
|---|---|---|---|---|
| 4-aminobenzoic acid | 15 / 0 | 1：原1991病例题名明确为PABA肝毒性；另核对2018、2022原始病例正文，均将肝损伤归因于potassium para-aminobenzoate。 | [1748050](https://pubmed.ncbi.nlm.nih.gov/1748050/)、[PMC6006614](https://pmc.ncbi.nlm.nih.gov/articles/PMC6006614/)、[PMC9428317](https://pmc.ncbi.nlm.nih.gov/articles/PMC9428317/) | 实际制剂为PABA钾盐，按现有去盐parent合同归属；1991病例全文未获取，新病例不冒充原record出处。 |
| Temozolomide | 206 / 0 | 1：2014原始病例明确诊断TMZ相关混合型肝细胞/胆汁淤积性肝炎；2012病例也报告胆汁淤积性肝炎。 | [24452817](https://pubmed.ncbi.nlm.nih.gov/24452817/)、[22394496](https://pubmed.ncbi.nlm.nih.gov/22394496/) | 2014病例发生于化放疗约5周；保留合并治疗和实际结局，不把所引用的其他死亡病例归给本病例。 |
| Aspirin / Acetylsalicylic acid | 38 / 0 | 1：1981类风湿双盲比较试验中，ASA组有严重肝毒性反应；1978儿童病例报告也支持正向。 | [7034196](https://pubmed.ncbi.nlm.nih.gov/7034196/)、[722421](https://pubmed.ncbi.nlm.nih.gov/722421/) | 成人试验ASA最高4 g/day是当时治疗方案；儿童及明确过量记录仍按各自条件保留。 |
| Memantine | 137 / 0 | 1：2008病例题名报告memantine胆汁淤积性肝炎；2019独立病例正文描述肝损伤、RUCAM 8及停药后恢复。 | [18413635](https://pubmed.ncbi.nlm.nih.gov/18413635/)、[31737715](https://pubmed.ncbi.nlm.nih.gov/31737715/) | 2019患者86岁；年龄信息保留，不自动授权默认条件gold。18413627、18413636并非这篇病例。 |
| Phentermine | 174 / 0 | 1：2022五例严重胆汁淤积性DILI病例系列的结果部分明确将phentermine列为肝毒性药物之一。 | [35975142](https://pubmed.ncbi.nlm.nih.gov/35975142/) | 这是作者病例归因；正文未逐一展开phentermine个案的剂量和因果评分，不能声称高确定性因果证明。文献晚于2015 TDC来源表。 |
| Terfenadine | 208 / 0 | 1：1996原始病例题名报告terfenadine胆汁淤积性肝炎；1985原始来信正文片段明确将三次肝炎发作归因于terfenadine，停药后肝功能恢复。 | [8757175](https://pubmed.ncbi.nlm.nih.gov/8757175/)、[2864011](https://pubmed.ncbi.nlm.nih.gov/2864011/)；[原文扫描片段](https://citeseerx.ist.psu.edu/document?doi=6675c4e90846ccbaf1794aa10a031f8f34e8b0ff&repid=rep1&type=pdf) | 1985与1996是不同研究。旧若干PMID挂到同一期其他来信，不能各算一票；1985正文由可检索原文扫描件读取，未取得完整本地PDF。 |
| Bezafibrate | 325 / 0 | 1：2016 DILI临床队列正文在致病药物列表中明确列出bezafibrate。 | [27703513](https://pubmed.ncbi.nlm.nih.gov/27703513/) | 可核实作者归因，非仅根据药名推断；没有该药逐患者因果评分细节。 |
| Loratadine（来源拼作Loratidine） | 129 / 0 | 1：2016原始报告两例loratadine相关肝炎，其中一例再暴露后再次严重转氨酶升高。 | [27293922](https://pubmed.ncbi.nlm.nih.gov/27293922/) | 来源拼写Loratidine对应loratadine；9139569是肝素相关血小板增多论文，不能直接引用。 |
| Propafenone | 185 / 0 | 1：原始报告两例propafenone治疗期间的急性胆汁淤积性肝炎，排除主要替代病因且停药后改善。 | [12643615](https://pubmed.ncbi.nlm.nih.gov/12643615/) | 保留疗程与实际暴露；论文讨论的既往病例不能额外重复投票。 |
| Amitriptyline | 35 / 0 | 1：1988原始病例明确描述amitriptyline相关长期胆汁淤积，并有连续肝活检；另有再暴露相关暴发性肝炎报告。 | [3335290](https://pubmed.ncbi.nlm.nih.gov/3335290/)、[6500194](https://pubmed.ncbi.nlm.nih.gov/6500194/) | 临床病例正向不等于任何治疗都会发生肝损伤。 |
| amphetamine | 230 / 0 | 待定：摘要仅说明20例不同病因的急性肝损伤患者接受弹性成像，未列amphetamine。 | [18098325](https://pubmed.ncbi.nlm.nih.gov/18098325/) | Starling支持文字提到amphetamine，但当前未取得可核实药物清单的正文，无法确认具体化合物、治疗/滥用或剂量；不能由缺全文改成0。 |
| Propofol | 186 / 0 | 1：2020独立原始病例报告短暂propofol镇静后肝炎，肝活检及后续恢复支持作者诊断。 | [27685310](https://pubmed.ncbi.nlm.nih.gov/27685310/)、[33313002](https://pubmed.ncbi.nlm.nih.gov/33313002/) | 旧2016记录PMID指向会议摘要合集，尚不能作为唯一研究ID；2020新病例是独立佐证，不是旧39岁女性病例的正确PMID。短暂暴露信息仍保留。 |
| Phenelzine | 113 / 0 | 1：原始病例描述phenelzine 45 mg/day用药70天后严重急慢性肝损伤及活检异常。 | [3963046](https://pubmed.ncbi.nlm.nih.gov/3963046/) | 8607600实际为体力活动与骨量研究，不能计作phenelzine病例。 |
| Aminoglutethimide | 34 / 0 | 1：1982 Gerber/Miller原始来信报告aminoglutethimide治疗后伴皮疹的胆汁淤积性黄疸，作者判断为药物过敏性肝毒性。 | [7091987](https://pubmed.ncbi.nlm.nih.gov/7091987/)；[原文扫描片段](https://citeseerx.ist.psu.edu/document?doi=abe4593298e78fbb4d47946e6329876942be52f9&repid=rep1&type=pdf) | 原四个挂载PMID均对应其他来信；正文病例为84岁女性并合用hydrocortisone，保留年龄及合用药；只对应一个研究。读取的是原始扫描件可检索正文片段，不是完整本地PDF。 |
| Fluoxetine | 99 / 0 | 1：1999原始报告两例认为由fluoxetine引起的急性肝炎。 | [10405699](https://pubmed.ncbi.nlm.nih.gov/10405699/) | 作者描述的是病例因果归因；正常受试者或试验中的阴性观察仍应保留。 |
| Bupropion | 7 / 0 | 1：2007原始病例正文报告bupropion相关严重肝炎，RUCAM 8，并讨论自身免疫性肝炎鉴别。 | [17877816](https://pubmed.ncbi.nlm.nih.gov/17877816/) | 正文同时引用未见肝脏不良事件的临床试验；这些阴性观察不因病例正向而删除。病例并非排除了所有因果不确定性。 |
| Citalopram | 383 / 0 | 1：正确2008来信题名为citalopram相关药物性肝病；2016临床队列正文明确将citalopram列为DILI致病药物。 | [18344747](https://pubmed.ncbi.nlm.nih.gov/18344747/)、[27703513](https://pubmed.ncbi.nlm.nih.gov/27703513/) | 六个同期期刊其他短文PMID不是六项独立citalopram研究；综述也不追加独立实验票。 |
| Metformin | 140 / 0 | 1：原始病例明确报告metformin相关胆汁淤积或混合型肝损伤，包括活检、排除其他病因及停药恢复。 | [14561576](https://pubmed.ncbi.nlm.nih.gov/14561576/)、[23983487](https://pubmed.ncbi.nlm.nih.gov/23983487/)、[24263160](https://pubmed.ncbi.nlm.nih.gov/24263160/) | 糖尿病和老年条件保留。合用pioglitazone试验、NASH试验的阴性观察不能删除，也不能作为所有情境无DILI的结论。 |
| Tacrine | 23 / 0 | 1：1990多中心安慰剂交叉试验67名患者中，作者报告9例肝炎，结论明确指出治疗剂量tacrine可诱发肝炎。 | [2107926](https://pubmed.ncbi.nlm.nih.gov/2107926/) | 这是原始临床试验；同一试验的多篇描述或二次引用不重复计票。阿尔茨海默病适应证按已批准规则并入默认组。 |

## 引用身份问题

检查了这19个parent的103条旧source votes所挂的97个不同PMID的题名/摘要/书目信息。在本次明确核对的子集中，23条record的PMID实际指向其他文章；逐条UID、payload hash和实际题名已写入citation_mismatches.jsonl。这23条不是23个独立反证，更不应自动改成负类。

- Aminoglutethimide：7091990、7091995、7091998、7092001分别是其他药物/疾病来信；相应肝毒性病例应追到Gerber/Miller的7091987，同一病例不能四票。
- Citalopram：18344739、18344741、18344745、18344746、18344749、18344751不是citalopram肝病来信；对应正确来信为18344747。
- Memantine：18413627是肺炎治疗，18413636是C. difficile治疗；肝炎来信为18413635。
- Terfenadine、Metformin、Loratadine和Phenelzine亦发现挂错文章的记录。尚不能唯一确认替代出处的，账本保留null；不拿新的正向病例冒充旧record的引用。
- Propofol的27685310是会议摘要合集。须用具体摘要题名/作者定位研究，不应把合集PMID直接当成唯一study ID。

## 对后续gold的处理

18个分子有正向原始报告，支持保留这些正向证据，不支持因TDC=0就删掉它们。已核实的真实阴性研究也继续保留。新gold仍需对全部新审阅记录按实际研究、分子、条件去重后重算；旧TDC数据集标签不与临床实验票简单相加。年龄、单次暴露等用户决定保留的条件仍须对应回去。amphetamine暂为出处未核实，不能填0。

本轮已将旧制剂组全部9条票的实际途径/制剂说明修正并保留为metadata，同时并入默认条件。当前20个benchmark条件经映射后为10个键。输出是staged重建输入；正式gold、split、index与实验结果尚未发布或改写。

可复核输出：primary_source_adjudications.jsonl、citation_mismatches.jsonl、primary_source_review_summary.json、formulation_review.jsonl、summary.json。
