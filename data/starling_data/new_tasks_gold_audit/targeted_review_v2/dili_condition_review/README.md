> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI 条件合并与医学相关性审查

本轮已执行条件映射、六条条件误归属修正、study 去重和 source-only label 重算。输出是本轮审阅目录的重建输入，尚未发布到正式 benchmark。输入为冻结 gold_v3 的 4,727 条 source votes，并非新一轮完整 base 的 20,573 条负向 records。不得混用两者计数。

## 判断原则

条件是否值得单独定义 benchmark query，取决于它是否有依据改变暴露、易感性或结论的适用范围。一般适应证名称不自动创建条件。合并不意味着该病在所有药物下绝对没有影响；原始人群、疗程、剂量、合用药等仍在 raw_record 和条件变更账本中保留。实际风险因素不能用适应证病名代替，例如叶酸补充、肥胖、既存脂肪肝、硫嘌呤合用别嘌醇。

无报告条件、治疗性用药未追加条件、健康志愿者按用户要求合并到 `no_reported_external_condition`。该键在本轮映射中表示不追加 query modifier；不推断所有原始人群健康，也不把未知暴露补成治疗剂量。

## 当前 20 组的处理

| 原条件 | 处理 | 理由 |
|---|---|---|
| 无报告条件；人类治疗性用药；健康志愿者 | 合并至默认组 | 用户明确要求；保留源描述 |
| 两个儿童组 | 合并为一个 pediatric 组 | 重复描述可合并；年龄本身不判定无影响 |
| 阿尔茨海默病、房颤、骨关节炎 | 合并至默认组 | 当前组是适应证/招募人群描述，不足以定义独立 DILI 目标 |
| 类风湿关节炎、银屑病、炎症性肠病 | 仅这些精确病名组并入默认组 | 不用病名替代药物、合用药、叶酸、代谢/肝病背景等具体因素；带额外条件的组不受此映射影响 |
| 过量/超治疗剂量 | 保留 | 改变实际暴露及剂量依赖毒性 |
| 单次给药/一天疗程 | 保留 | 限定累计暴露；短疗程阴性不能无条件代表长期暴露；给药疗程与随访时间须区分 |
| 儿童、>65 岁 | 保留年龄轴 | 年龄作用依赖具体药物，不是统一增加风险；不能据此删除年龄信息 |
| 乙肝、代谢相关脂肪肝 | 确认用药前已经存在时保留 | 既存肝病对部分药物的易感性和 DILI 结局有意义 |
| 2 型糖尿病 | 暂保留 | 代谢背景在部分药物相关肝损伤中有证据，不能判定完全无影响；糖尿病不自动等于脂肪肝 |
| HIV | 确认患者感染时保留 | 免疫状态、CD4、合并病毒性肝炎及治疗方案可能重要；不能把排除 HIV 当作感染 |
| 肝移植 | 仅用药前已移植者保留 | 移植后治疗背景与 DILI 后接受移植是不同语义 |
| 旧长效注射制剂组 | 全部并入默认组，已应用 | 按用户最新要求；9票的实际制剂/途径说明已修正并保留，包括真实XR naltrexone |

以上组级映射使当前 20 个组映射到 10 个键，已包含制剂组9票并入默认组。这不是新 base 全量重建后的最终条件数。

## 已执行与校验

- 2,397 条旧 source votes 的条件键改变；原始 payload 和 record 方向均不变。
- 六条误归属已在 staged source votes 修正：HIV 排除诊断、两个药物引起的脂肪变、另一个疾病对照组的 NAFLD、另一个病因组的 HBV、DILI 后肝移植。
- 合并后 11 个 study-parent-condition 单元产生同向重复，去重后 4,716 票；研究内部新冲突为 0。
- 按现有无条件 70% / 有条件 60% 一致性规则重算，接受 2,565 个 parent-condition：正 2,373、负 192；7 个因冲突/一致性不够未接受。这些是覆盖筛选前的旧 source-only 审计结果。
- 在当前混合 TDC/Starling benchmark 上只做映射检查，发现 122 个重复 parent-condition，其中 19 个存在正负标签差异。没有将这些 gold labels 当作独立研究投票。原始出处核验结果见 [PRIMARY_SOURCE_REVIEW.md](PRIMARY_SOURCE_REVIEW.md)：18个有正向原始报告，amphetamine仍待核实；尚未改写最终gold。
- `formulation_review.jsonl` 逐条记录旧制剂组全部 9 票的审查结论；已全部修正描述并并入默认条件；候选仍属于staged输入。
- 输入 source 和正式 scaffold split 的 SHA-256 已验证未变。所有源票均被去重代表记录或冲突账本覆盖，没有静默删除。

完整分组映射见 `policy.json`，变更见 `condition_changes.jsonl`，汇总见 `summary.json`。复现：

```sh
PYTHONPATH=. /data1/tianang/anaconda3/envs/vllm/bin/python data/starling_data/new_tasks_gold_audit/targeted_review_v2/review_dili_conditions.py
```

## 医学依据

- [AASLD DILI practice guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC9936988/)：年龄的总体效应不完全明确，但存在药物特异性差异；既存肝病、代谢因素和合用药对部分药物有意义。因此上述合并是数据集分组决策，不是无风险声明。
- [丙戊酸相关肝毒性原始研究](https://pubmed.ncbi.nlm.nih.gov/3102998/)：小于两岁并接受多药治疗的儿童具有明确的特殊风险，不能推断整个儿童年龄轴无用。
- [NIH nevirapine 信息](https://clinicalinfo.hiv.gov/en/drugs/nevirapine/patient)：CD4、性别、乙/丙肝和基线肝功能会影响风险；HIV 病名只是粗粒度背景。
- [烟酸制剂随机对照研究](https://pubmed.ncbi.nlm.nih.gov/8309029/)：缓释与速释制剂的肝毒性有实测差异，说明制剂在科学上可能有影响；本轮按用户指定的benchmark分组粒度将制剂组并入默认，原始制剂信息保留。

本次医学依据审查和 supplied record 字段审查不是所有源论文全文核验，也不是新的全量 source-condition 审阅。
