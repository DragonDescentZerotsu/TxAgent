> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI / Carcinogens 当前条件清单与碎片化审计

统计对象：当前冻结的 conditioned_benchmark scaffold（train+valid+test）及 gold_v3/source_only_benchmark/scaffold。最新 source-label 审阅尚未重建条件组，因此本清单不是新审阅结果的条件分布。条件按原始 condition_group 精确分组；每行计数单位是 molecule-condition row，不是原始 record。

无外部条件 no_reported_external_condition 也计作一个组。全量 gold 条件及被覆盖门槛排除的组见同目录 condition_inventory_gold.csv。

## DILI

正式 benchmark：**20 个组，1,034 rows**；其中 Starling source-only 为 19 个组、569 rows，另有无外部条件组。Scaffold 与 random 的 molecule-condition-label 总集合已核对一致。

覆盖筛选前 gold：2,043 个组、2,608 rows；2,011 个组只有 1 row（98.4%），2,022 个组不足 3 rows。当前 20 个组中 11 个含负类。

| # | 原始 condition_group（完整值） | 总数 | 正 | 负 | train | valid | test |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `age_group=pediatric+exposure=therapeutic_use+population=human` | 7 | 7 | 0 | 5 | 1 | 1 |
| 2 | `age_group=pediatric+exposure=therapeutic_use+population=human+reported_condition=pediatric population` | 3 | 3 | 0 | 1 | 1 | 1 |
| 3 | `exposure=overdose_or_supratherapeutic+population=human` | 7 | 7 | 0 | 5 | 1 | 1 |
| 4 | `exposure=therapeutic_use+formulation=long_acting_injectable+population=human` | 7 | 5 | 2 | 5 | 1 | 1 |
| 5 | `exposure=therapeutic_use+population=human` | 460 | 433 | 27 | 391 | 32 | 37 |
| 6 | `exposure=therapeutic_use+population=human+population_context=alzheimer_disease` | 5 | 3 | 2 | 3 | 1 | 1 |
| 7 | `exposure=therapeutic_use+population=human+population_context=atrial_fibrillation` | 8 | 7 | 1 | 5 | 1 | 2 |
| 8 | `exposure=therapeutic_use+population=human+population_context=healthy_volunteers` | 5 | 4 | 1 | 3 | 1 | 1 |
| 9 | `exposure=therapeutic_use+population=human+population_context=hepatitis_b` | 6 | 5 | 1 | 4 | 1 | 1 |
| 10 | `exposure=therapeutic_use+population=human+population_context=hiv` | 5 | 5 | 0 | 3 | 1 | 1 |
| 11 | `exposure=therapeutic_use+population=human+population_context=inflammatory_bowel_disease` | 3 | 3 | 0 | 1 | 1 | 1 |
| 12 | `exposure=therapeutic_use+population=human+population_context=liver_transplant` | 3 | 1 | 2 | 1 | 1 | 1 |
| 13 | `exposure=therapeutic_use+population=human+population_context=metabolic_fatty_liver_disease` | 14 | 11 | 3 | 12 | 1 | 1 |
| 14 | `exposure=therapeutic_use+population=human+population_context=osteoarthritis` | 5 | 5 | 0 | 3 | 1 | 1 |
| 15 | `exposure=therapeutic_use+population=human+population_context=psoriasis` | 7 | 5 | 2 | 5 | 1 | 1 |
| 16 | `exposure=therapeutic_use+population=human+population_context=rheumatoid_arthritis` | 6 | 6 | 0 | 4 | 1 | 1 |
| 17 | `exposure=therapeutic_use+population=human+population_context=type_2_diabetes` | 9 | 7 | 2 | 7 | 1 | 1 |
| 18 | `exposure=therapeutic_use+population=human+regimen=single_dose_or_one_day` | 6 | 6 | 0 | 4 | 1 | 1 |
| 19 | `exposure=therapeutic_use+population=human+reported_condition=older adult (>65 y) patient cohort` | 3 | 3 | 0 | 1 | 1 | 1 |
| 20 | `no_reported_external_condition` | 465 | 234 | 231 | 365 | 53 | 47 |

## Carcinogens

正式 benchmark：**92 个组，1,464 rows**；其中 Starling source-only 为 91 个组、1,188 rows，另有无外部条件组。Scaffold 与 random 的 molecule-condition-label 总集合已核对一致。

覆盖筛选前 gold：9,240 个组、10,800 rows；8,854 个组只有 1 row（95.8%），9,066 个组不足 3 rows。当前 92 个组中 29 个含负类。

| # | 原始 condition_group（完整值） | 总数 | 正 | 负 | train | valid | test |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `model=a/j mouse+route=oral+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 2 | `model=a/j mouse+species=mouse` | 8 | 8 | 0 | 6 | 1 | 1 |
| 3 | `model=aci rat+species=rat` | 4 | 4 | 0 | 2 | 1 | 1 |
| 4 | `model=adolescent female rat+species=rat` | 4 | 4 | 0 | 1 | 2 | 1 |
| 5 | `model=b6c3f1 and cd-1 mouse+species=mouse` | 4 | 4 | 0 | 1 | 2 | 1 |
| 6 | `model=b6c3f1 mouse+route=inhalation+species=mouse` | 11 | 11 | 0 | 9 | 1 | 1 |
| 7 | `model=b6c3f1 mouse+route=oral+species=mouse` | 23 | 20 | 3 | 11 | 4 | 8 |
| 8 | `model=b6c3f1 mouse+species=mouse` | 24 | 24 | 0 | 16 | 4 | 4 |
| 9 | `model=balb/c mouse+species=mouse` | 12 | 12 | 0 | 8 | 2 | 2 |
| 10 | `model=c57bl/6 mouse+route=oral+species=mouse` | 4 | 4 | 0 | 2 | 1 | 1 |
| 11 | `model=c57bl/6 mouse+species=mouse` | 8 | 8 | 0 | 5 | 1 | 2 |
| 12 | `model=cd-1 mouse+species=mouse` | 7 | 7 | 0 | 4 | 1 | 2 |
| 13 | `model=cdf1 mouse+reported_condition=0.02-0.08%25 in diet+route=oral+species=mouse` | 4 | 4 | 0 | 2 | 1 | 1 |
| 14 | `model=cdf1 mouse+route=oral+species=mouse` | 6 | 6 | 0 | 3 | 1 | 2 |
| 15 | `model=cf1 mouse+route=oral+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 16 | `model=f344 rat (female)+route=oral+species=rat` | 4 | 4 | 0 | 2 | 1 | 1 |
| 17 | `model=f344 rat+route=oral+species=rat` | 21 | 18 | 3 | 12 | 5 | 4 |
| 18 | `model=f344 rat+species=rat` | 39 | 38 | 1 | 23 | 9 | 7 |
| 19 | `model=f344/n rat+route=oral+species=rat` | 9 | 9 | 0 | 4 | 1 | 4 |
| 20 | `model=female a/j mouse+species=mouse` | 5 | 5 | 0 | 3 | 1 | 1 |
| 21 | `model=female b6c3f1 mouse+route=oral+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 22 | `model=female mouse+species=mouse` | 8 | 8 | 0 | 6 | 1 | 1 |
| 23 | `model=female nmri mouse+route=dermal_or_topical+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 24 | `model=female rat+route=oral+species=rat` | 10 | 10 | 0 | 4 | 2 | 4 |
| 25 | `model=female rat+species=rat` | 7 | 7 | 0 | 5 | 1 | 1 |
| 26 | `model=female sprague-dawley rat+route=oral+species=rat` | 10 | 10 | 0 | 3 | 4 | 3 |
| 27 | `model=female wistar spf rat+reported_condition=applied on dorsal skin of 20 rats each; observation period of 15 months+route=dermal_or_topical+species=rat` | 4 | 3 | 1 | 1 | 1 | 2 |
| 28 | `model=hairless mouse+reported_condition=requires combined exposure to uva irradiation; photocarcinogenic effect+route=oral+species=mouse` | 4 | 4 | 0 | 2 | 1 | 1 |
| 29 | `model=hamster (mesocricetus auratus)+species=hamster` | 11 | 9 | 2 | 6 | 3 | 2 |
| 30 | `model=male b6c3f1 and cd-1 mouse+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 31 | `model=male b6c3f1 mouse+route=oral+species=mouse` | 8 | 8 | 0 | 5 | 2 | 1 |
| 32 | `model=male f344 rat+route=oral+species=rat` | 21 | 20 | 1 | 12 | 2 | 7 |
| 33 | `model=male f344 rat+species=rat` | 9 | 9 | 0 | 4 | 4 | 1 |
| 34 | `model=male fischer rat+route=oral+species=rat` | 4 | 4 | 0 | 1 | 1 | 2 |
| 35 | `model=male hras128 rat (human c-ha-ras proto-oncogene transgenic)+reported_condition=male hras128 rats; sacrifice at week 20+species=rat` | 3 | 3 | 0 | 1 | 1 | 1 |
| 36 | `model=male mouse+species=mouse` | 9 | 9 | 0 | 6 | 1 | 2 |
| 37 | `model=male rat+route=oral+species=rat` | 10 | 10 | 0 | 3 | 1 | 6 |
| 38 | `model=male rat+species=rat` | 15 | 15 | 0 | 8 | 1 | 6 |
| 39 | `model=male sprague-dawley rat+route=oral+species=rat` | 8 | 8 | 0 | 6 | 1 | 1 |
| 40 | `model=male strain a/j mouse+reported_condition=single intraperitoneal injection of pah in tricaprylin vehicle; lungs harvested 8 months after treatment+route=injection_or_parenteral+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 41 | `model=male wistar rat+route=oral+species=rat` | 9 | 8 | 1 | 6 | 1 | 2 |
| 42 | `model=mouse (mus musculus)+species=mouse` | 8 | 8 | 0 | 4 | 1 | 3 |
| 43 | `model=mouse skin+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 44 | `model=mrc rat+species=rat` | 4 | 4 | 0 | 2 | 1 | 1 |
| 45 | `model=mus musculus (mouse)+route=dermal_or_topical+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 46 | `model=mus musculus (mouse)+route=oral+species=mouse` | 9 | 8 | 1 | 5 | 2 | 2 |
| 47 | `model=mus musculus (mouse)+species=mouse` | 15 | 14 | 1 | 11 | 1 | 3 |
| 48 | `model=neonatal cd-1 mouse+reported_condition=neonatal treatment on days 1–5 at 2 μg/pup·d; evaluated in aged mice+species=mouse` | 5 | 5 | 0 | 1 | 1 | 3 |
| 49 | `model=neonatal male b6c3f1 mouse+reported_condition=neonatal bioassay; i.p. dosing divided 1/7, 2/7, 4/7 given within 24 h of birth and at 8 and 15 days; dmso vehicle+route=injection_or_parenteral+species=mouse` | 4 | 4 | 0 | 2 | 1 | 1 |
| 50 | `model=newborn mouse+route=injection_or_parenteral+species=mouse` | 3 | 1 | 2 | 1 | 1 | 1 |
| 51 | `model=osborne-mendel rat+route=oral+species=rat` | 4 | 3 | 1 | 1 | 1 | 2 |
| 52 | `model=ovarian cancer patients (international case-control study, 114 taml cases)+species=human` | 5 | 5 | 0 | 2 | 1 | 2 |
| 53 | `model=rat (2-year bioassay)+species=rat` | 3 | 3 | 0 | 1 | 1 | 1 |
| 54 | `model=rat (male and female)+route=oral+species=rat` | 4 | 3 | 1 | 2 | 1 | 1 |
| 55 | `model=rat (male and female)+species=rat` | 4 | 4 | 0 | 2 | 1 | 1 |
| 56 | `model=rat (mammary gland)+species=rat` | 5 | 5 | 0 | 1 | 1 | 3 |
| 57 | `model=rat liver+species=rat` | 4 | 4 | 0 | 2 | 1 | 1 |
| 58 | `model=rattus norvegicus (rat)+route=oral+species=rat` | 14 | 14 | 0 | 9 | 2 | 3 |
| 59 | `model=rattus norvegicus (rat)+species=rat` | 28 | 28 | 0 | 20 | 3 | 5 |
| 60 | `model=sd rat+route=oral+species=rat` | 5 | 5 | 0 | 3 | 1 | 1 |
| 61 | `model=sd rat+species=rat` | 3 | 3 | 0 | 1 | 1 | 1 |
| 62 | `model=sprague-dawley female rat+route=oral+species=rat` | 3 | 2 | 1 | 1 | 1 | 1 |
| 63 | `model=sprague-dawley rat+route=oral+species=rat` | 17 | 14 | 3 | 10 | 2 | 5 |
| 64 | `model=sprague-dawley rat+species=rat` | 16 | 15 | 1 | 12 | 2 | 2 |
| 65 | `model=swiss albino mouse+route=oral+species=mouse` | 3 | 2 | 1 | 1 | 1 | 1 |
| 66 | `model=swiss albino mouse+species=mouse` | 5 | 4 | 1 | 3 | 1 | 1 |
| 67 | `model=transplant patients on immunosuppressive therapy+species=human` | 3 | 3 | 0 | 1 | 1 | 1 |
| 68 | `model=wistar rat+route=injection_or_parenteral+species=rat` | 9 | 9 | 0 | 6 | 1 | 2 |
| 69 | `model=wistar rat+route=oral+species=rat` | 11 | 10 | 1 | 9 | 1 | 1 |
| 70 | `model=wistar rat+species=rat` | 12 | 12 | 0 | 7 | 3 | 2 |
| 71 | `no_reported_external_condition` | 276 | 58 | 218 | 121 | 123 | 32 |
| 72 | `reported_condition=administered in the diet+route=oral+species=rat` | 3 | 2 | 1 | 1 | 1 | 1 |
| 73 | `reported_condition=chronic bioassay dosing+species=mouse` | 3 | 3 | 0 | 1 | 1 | 1 |
| 74 | `reported_condition=chronic bioassay dosing+species=rat` | 3 | 3 | 0 | 1 | 1 | 1 |
| 75 | `route=dermal_or_topical+species=mouse` | 13 | 10 | 3 | 7 | 3 | 3 |
| 76 | `route=inhalation+species=mouse` | 10 | 10 | 0 | 8 | 1 | 1 |
| 77 | `route=inhalation+species=rat` | 28 | 26 | 2 | 24 | 3 | 1 |
| 78 | `route=injection_or_parenteral+species=mouse` | 22 | 20 | 2 | 17 | 3 | 2 |
| 79 | `route=injection_or_parenteral+species=rat` | 24 | 24 | 0 | 19 | 2 | 3 |
| 80 | `route=multiple_routes+species=rat` | 7 | 7 | 0 | 4 | 1 | 2 |
| 81 | `route=oral+species=dog` | 3 | 3 | 0 | 1 | 1 | 1 |
| 82 | `route=oral+species=monkey` | 5 | 4 | 1 | 3 | 1 | 1 |
| 83 | `route=oral+species=mouse` | 50 | 45 | 5 | 35 | 7 | 8 |
| 84 | `route=oral+species=rat` | 106 | 97 | 9 | 68 | 10 | 28 |
| 85 | `route=other+species=mouse` | 8 | 8 | 0 | 4 | 2 | 2 |
| 86 | `route=other+species=rat` | 12 | 12 | 0 | 7 | 3 | 2 |
| 87 | `species=dog` | 7 | 7 | 0 | 2 | 2 | 3 |
| 88 | `species=hamster` | 7 | 7 | 0 | 3 | 3 | 1 |
| 89 | `species=human` | 15 | 13 | 2 | 9 | 2 | 4 |
| 90 | `species=monkey` | 6 | 6 | 0 | 2 | 1 | 3 |
| 91 | `species=mouse` | 110 | 106 | 4 | 72 | 9 | 29 |
| 92 | `species=rat` | 169 | 164 | 5 | 110 | 22 | 37 |

## 已核实的碎片化来源

- DILI `age_group=pediatric+exposure=therapeutic_use+population=human`（7 rows）与附加 `reported_condition=pediatric population` 的组（3 rows）被分开。后者在组文本上重复了儿童条件。
- Carcinogens `model=sd rat` 与 `model=sprague-dawley rat` 的写法未统一；`model=female sprague-dawley rat+route=oral+species=rat`（10 rows）与 `model=sprague-dawley female rat+route=oral+species=rat`（3 rows）也被分开。这里只识别 condition 字符串的同义问题，尚未执行源记录核验或合并。
- Carcinogens 同时有 `species=mouse`、`model=mouse (mus musculus)+species=mouse`、`model=mus musculus (mouse)+species=mouse`。通用物种描述也会产生额外组。
- `source_gold_review.source_conditions` 将未规范化的 qualifying_conditions 文本保存为 reported_condition 条件原子；空格和大小写规范化不足以统一语义。研究人数、观察时间、载体等整段描述会形成专属于一篇研究的键。
- 条件键由全部原子组合：物种、模型、性别、途径、剂量、疗程、人群等；任何一项不同或缺失，都可能分成另一组。缺失项不等于确定相同，不能自动合并。
- `fresh_conditioned_benchmark.py` 要求每组至少三个 parents 和三个非空 scaffolds；否则整个组不进入 source-only benchmark。碎片化因而转化为样本排除。

可能真正改变结论的暴露、物种、性别、基因型、途径、疗程等需要保留。应先规范同义词、拆分结构化语义与研究描述，再审核稀疏组的选择要求；本审计未更改 gold、condition、split 或 index。

## 输入 SHA-256

```json
{
  "data/starling_data/dili/gold_v3/source_only_benchmark/scaffold/accepted_parent_conditions_before_group_gate.jsonl": "f3f78a1b46adb3f98663894c69eca3edec2cd50f5b2843ffbd20819070da53d2",
  "data/starling_data/dili/gold_v3/source_only_benchmark/scaffold/accepted_parent_conditions.jsonl": "3a9ab695b23c2cef1d3101bdc04cd425e4030dc9097e600c678258df66fb3ff3",
  "data/conditioned_benchmark/DILI/scaffold/train.jsonl": "c07d606d03e0cf095c5f0ec1074886ad84af102f40937f72eace471e5cffd8de",
  "data/conditioned_benchmark/DILI/scaffold/valid.jsonl": "3e202d8e4fd182a66c1140a6d9d0c6b1af6e3b47b1ec9d3a0024fe0e810d8db9",
  "data/conditioned_benchmark/DILI/scaffold/test.jsonl": "804363fb1a63d61499042b7a55512c929e195d536a67accb1e6df3a99b7c7460",
  "data/conditioned_benchmark/DILI/random/train.jsonl": "b90c12706e18840d8d01c43533b63a9dda78c52746671a0584dc22befb8ae556",
  "data/conditioned_benchmark/DILI/random/valid.jsonl": "b33d55771ab0f1d2823355b4064f3d13a34b2aa011eecc12beb1e88f5c73d77d",
  "data/conditioned_benchmark/DILI/random/test.jsonl": "9d960868394b93d16949e9b302a00057017d70d0935bb53a3059a6a7dac51c7a",
  "data/starling_data/carcinogens/gold_v3/source_only_benchmark/scaffold/accepted_parent_conditions_before_group_gate.jsonl": "3114fd4912be039687cce457b707f808dc2d6f15a6c193fface1f31554a38460",
  "data/starling_data/carcinogens/gold_v3/source_only_benchmark/scaffold/accepted_parent_conditions.jsonl": "cec3eb518bdcc9d8ac831f93d09960c1f0f66ff96f424bd46ae16d6930b47a2d",
  "data/conditioned_benchmark/Carcinogens/scaffold/train.jsonl": "91df0117df87cfe6596eb51ea29ebcb3613cdc1e1931af43f70fcce5d32cfbab",
  "data/conditioned_benchmark/Carcinogens/scaffold/valid.jsonl": "3bbc0d32a856960d70455d2a1eb1c4d9953d57ce611cab04b459ba9fb50a5fc8",
  "data/conditioned_benchmark/Carcinogens/scaffold/test.jsonl": "0f1ac10f2d616211247ca72dd697010e018bc5ab8e7b2786d171cf08dbb220a1",
  "data/conditioned_benchmark/Carcinogens/random/train.jsonl": "d08f320230e187d6bbf8d05b28aff8977387167f5f8aa94ca8582cffa71d006a",
  "data/conditioned_benchmark/Carcinogens/random/valid.jsonl": "ede20db75c29d606289d770bfdf987d5c216bed7edf6c10d7dc8291b71233c12",
  "data/conditioned_benchmark/Carcinogens/random/test.jsonl": "44c6adc84906b02d18fbd72ac83e07c8f21b9d06d3e9326f61b6c14274e756a5"
}
```
