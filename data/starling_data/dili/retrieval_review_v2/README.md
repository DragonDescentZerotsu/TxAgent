# DILI 四条 source 问题修复

本轮只处理 valid L2–L4 trace 诊断留下的四个问题，并对错绑结构做了同结构记录检查。源数据、两种 split 索引和共享 DILI 表均已发布并通过校验；凭据见 `publication.json`。此前全库审阅、pilot 和 v1 的 112 条审阅不计入本轮数量。

| 原卡片 | PMID | 确认的问题 | 处理 |
|---|---|---|---|
| `card_14ff7a36d3313355` | 33509280 | 名称和正文为 emodin，绑定结构却是多一个芳香环的 C21H14O6，非 emodin C15H10O5；不是模型凭空改写名称 | 隔离错误绑定，L2→不进入检索；保留原始记录和正文 |
| `card_5f8d40af5e4fc45a` | 28901189 | 冻存后肝细胞超微结构恢复，没有实际胆汁转运/胆小管端点 | L4→L3；保留保护作用、混合处理和物种未注明的限制 |
| `card_3f822630930f676b` | 17466932 | 死亡受体激动剂与 caspase 抑制剂作用下的 HepG2 晚期凋亡形态 | L4→L3；保留肿瘤细胞模型和 modifier 角色 |
| `card_1ed065acd3db495d` | 30560367 | 冻存原代人肝细胞单层附着/形态评分改善 | L4→L3；保留保护性结果、剂量和批次限制；原文 L3 指细胞批次 |

Emodin 核对依据：[PubChem CID 3220](https://pubchem.ncbi.nlm.nih.gov/compound/3220)、[原论文条目](https://pubmed.ncbi.nlm.nih.gov/33509280/)，以及 `identity_evidence.json` 中的两个结构与 InChIKey。当前 source 的相同错绑结构只对应这一条记录。隔离原因是已确认的对象—结构不一致，不是阴性方向或 benchmark 错误；没有把原文“尚无人体肝毒性证据”当作已证明安全，也没有自动把原文重新绑定到另一个 molecule。

三条换层依据为完整 supplied source 的实际 endpoint、受试对象和角色；本轮未逐篇核验这三篇论文全文。三条证据仍参与检索，raw fields、剂量、support text、uncertainty 均保留。其他正确归属但被模型错误使用的 DWN12088/FPH1 等卡片没有据此删除。

`record_review.json` 按 source UID、raw payload hash、原 level 和原文引句绑定四个决定。共修改四行，物理删除零行；原始字段及 voter 身份不变。L1 为 108,437 条，L2 为 166,491，L3 为 141,949，L4 为 76,083；L5–L7 不变。

旧 trace 重叠检查覆盖 valid/test 共 804 行、progressive/full-flat 的七层：错绑 emodin 卡出现在 valid 的 7 行，被 progressive 的 2 行采用为 basis，且两行均在 L2 从对变错（index 291、292）。复读两条 state，模型明确将该 analog 卡视为 query 自身的人体证据，并把缺失证据用于否定风险。错绑也让描述 query 的文本借用另一结构进入 disjoint 检索；隔离修复了这一具体对象归属问题。三张形态卡均只见于 valid index 303，未对应新增错误；换层依据是端点语义，不是分数。test 的旧 trace 没有这四张卡。详见 `trace_overlap_summary.json` / `trace_overlap.jsonl`。出现和引用不等同于反事实因果证明；换层可能改变其他 query 的选卡，不能据旧 trace 断言 test 的新输入不变。

`audit_selected_surfaces.py` 使用现有 retrieval 和 progressive selection 函数核对旧 source 七层重建，再比较全部 804 行的新旧选卡。它不调用工具或 LLM，也不修改 prompt/prior/similarity；输出是证据变化范围，不是完整 prompt 复用凭据。重跑时仍须校验工具、完整模型输入和 progressive 前缀依赖。当前没有 v2 的新 performance，不把 v1 分数改写成修复后的成绩。

发布复用 `stage_new_task_retrieval`、`rebuild_current_starling_retrieval`、`validate_new_task_retrieval_identity` 和 `export_current_starling_level_records`。仅构建 DILI catalog、scaffold/random indices 和共享 DILI 表；保留 L1-only heldout 预排除、L2 保留和 query-time disjoint。旧 source/index、gold/splits/votes 保持可追溯。`frozen_before.json`、`source/record_review_receipt.json`、`publication.json` 保存校验结果；相关既有测试 19 项通过。未改 reasoning 方法，未新增模型调用。

选卡核对结果：valid 8 / 402 行、test 0 / 402 行在至少一层改变。两组 L1 共 804 行的选卡均不变；逐层计数和首个变化层见 `selected_surface_audit_summary.json`。尚未重新推理，不能据此报告改正数或新的 performance。

后续定点重跑已完成；当前结果见 `outputs/paper/starling_conditioned_dili_retrieval_review_v2_no_prior_grounded_sim0/REPORT.md`。本页“尚未推理”描述的是 source 发布阶段，后续只新增上述 94 个受影响层输出。
