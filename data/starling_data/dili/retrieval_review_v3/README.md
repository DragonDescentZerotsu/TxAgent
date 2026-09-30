> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI 修复优先 source release v3

本轮落实上次 trace 核查，并逐条补读同 PMID 的相关错绑记录，共修改 11 条；原始 1,645,109 条 acquisition records 不删除。原始 raw_record_json、source_smiles 和冻结 voter 标记保留；修正后的正文与身份用于 canonical retrieval fields，完整修订另存 reviewed_record_json 和 content_repairs.json。source_smiles 是原始获取值，不能覆盖修正后的 canonical_smiles 用于检索。

| 处理 | 条数 | 说明 |
|---|---:|---|
| 恢复 L2 | 1 | Emodin 改绑 CID 3220；人体证据缺口不能呈现为人体阴性试验 |
| 修复后退出 DILI 检索 | 2 | Q39 改绑 CID 46946323；K562 机制和 SMMC-7721 IC50 都属于原文抗癌实验，不是肝安全性端点 |
| 修复后退出单分子检索 | 4 | 1 条 IVCA 复方与 3 条 Pep-1-Mito 线粒体材料实验，保存修正的真实对象，不强行归给单一成分 |
| L5 保留并修正角色/剂量 | 1 | Z-VAD.Fmk 是 5 µM modifier，烟雾是 0.18 puff/ml challenge；保留间接机制与非肝体系限制 |
| L4→L3 | 3 | 单层附着、一般超微结构、凋亡恢复；保护作用保留。另一个明确报告胆小管保存的 TEM 卡保持 L4 |

原来属于 L1 的错误 ivacaftor 绑定，现在修正为 IVCA 复方记录但不进入单分子索引。该 source UID 的历史 is_gold_voter=True 及冻结 vote 文件均保留，实际可检索 L1 减少 1 条；修正后的复方不能继承错误结构的 L1 资格。本轮不进行 gold 方向重审、不自动改 benchmark 或拆分复方投票。

Emodin 和 IVCA 已检查全文；Q39、Pep-1-Mito 使用原论文摘要确认身份/对象，详细量化结果仍保留 supplied extraction 的核验限制。其他换层和 modifier 修正依据完整 supplied record。来源与修订包在本目录，原始诊断见 outputs/paper/starling_conditioned_dili_retrieval_review_v2_no_prior_grounded_sim0/post_replay_trace_review/。

source_validation.json 核对全部行：11 条按修订包改变，其他行逐字段相同，raw acquisition 和 voter 标记全部不变，27 个冻结 gold/split/vote 文件哈希不变。代码测试见 tests.log。source/content_repair_receipt.json 绑定 source hash、ledger 和变更清单。

发布复用共享 stage_new_task_retrieval.apply_content_repairs、rebuild_current_starling_retrieval、validate_new_task_retrieval_identity 和 export_current_starling_level_records，仅处理 DILI。publication.json 通过后为当前 source；保持 L1-only heldout 预排除、L2 保留及 query-time disjoint。其他任务和推理规则不变。

新推理入口：outputs/paper/starling_conditioned_dili_retrieval_review_v3_no_prior_grounded_sim0/run_replay.py。先比较 valid/test 全部 804 行七层选卡，只重新准备发生变化的 queries；既有 query 和 common analog 的工具文本固定复用。完整实际 prompt、工具及 progressive 前缀 gate 决定哪些输出可复用。Hosted Flash-0731、每模式并发 256、最大同时两模式、失败六路竞速、max_tokens 20480；gold、prior、similarity、4/2 卡片预算与完整 level plan 不变。结果以 registry 和该 root 的 runtime_progress.json / completion_validation.json 为准；运行完成前不报告新性能。
