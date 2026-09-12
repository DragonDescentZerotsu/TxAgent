# 合并后 19 个跨来源冲突

全部为旧 Starling gold=1 与 TDC=0。TDC 标签逐条回查冻结下载的 dili.tab 原始行，19 项均未发生导入翻转。后续已核查作者2015原始补充表及临床报告，逐分子结果见 [PRIMARY_SOURCE_REVIEW.md](PRIMARY_SOURCE_REVIEW.md)：18个有明确正向临床报告，amphetamine待核实。下表仍展示原始冲突，不代表新gold。

| 分子（沿用来源名称） | Starling | TDC | Starling 旧条件及票数 | TDC 原始行（零起始） |
|---|---:|---:|---|---|
| 4-aminobenzoic acid | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 12 |
| Temozolomide | 1 | 0 | exposure=therapeutic_use+population=human: 2 | 203 |
| Acetylsalicylic acid | 1 | 0 | exposure=therapeutic_use+population=human: 9; exposure=therapeutic_use+population=human+population_context=rheumatoid_arthritis: 1 | 35 |
| Memantine | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 134 |
| Phentermine | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 171 |
| Terfenadine | 1 | 0 | exposure=therapeutic_use+population=human: 7 | 205 |
| Bezafibrate | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 322 |
| Loratidine | 1 | 0 | exposure=therapeutic_use+population=human: 2 | 126 |
| Propafenone | 1 | 0 | exposure=therapeutic_use+population=human: 4 | 182 |
| Amitriptyline | 1 | 0 | exposure=therapeutic_use+population=human: 5 | 32 |
| amphetamine | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 227 |
| Propofol | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 183 |
| Phenelzine | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 110 |
| Aminoglutethimide | 1 | 0 | exposure=therapeutic_use+population=human: 3 | 31 |
| Fluoxetine | 1 | 0 | exposure=therapeutic_use+population=human: 4 | 96 |
| Bupropion | 1 | 0 | exposure=therapeutic_use+population=human: 1 | 4 |
| Citalopram | 1 | 0 | exposure=therapeutic_use+population=human: 9 | 380 |
| Metformin | 1 | 0 | exposure=therapeutic_use+population=human: 7 | 137 |
| Tacrine | 1 | 0 | exposure=therapeutic_use+population=human: 3; exposure=therapeutic_use+population=human+population_context=alzheimer_disease: 4 | 20 |

处理顺序：核对 TDC 来源定义和年代；逐分子检查 Starling 支持文本的对象、方向、真实条件与研究身份；按新 source review 重建研究票。TDC 数据集标签不充当一项独立阴性实验，也不与原始 record 数量做简单多数表决。同一条件下仍不能给出稳定方向时，保留源 records 和冲突账本，该 parent-condition 不作为确定的二元 gold；不能靠重新拆成语义等价的条件隐藏冲突。

制剂问题见 formulation_review.jsonl：9 票中 2 票明确为 XR naltrexone；其余涉及普通肌注、静注、口服起始肌注、ER/SR 烟酸和制剂未明确的描述，已修正描述后按用户要求全部并入默认组，包括真实XR naltrexone；实际制剂/途径仍保留。普通肌注不能直接等于长效注射。

TDC 原始下载 SHA-256: `d3961647ff6df9711b6aa4b9fe59d6cbeb40f9d23dd6e9d51d6656e6e6df5ed8`
