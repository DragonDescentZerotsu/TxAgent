# Starling benchmark：历史结果总账

更新时间：2026-08-28。

本文件保留历史 Starling 数据迁移和实验，不再定义 current gold。当前四任务唯一评估入口是
`data/conditioned_benchmark/<Task>/scaffold/`；统一合同见
`common/starling/CONDITIONED_BENCHMARK.md`，机器可读 current metrics 见
`current_conditioned_results.json`。历史 molecule-only、gold-vN、selected-vN、TDC test/valid 和下文旧矩阵
不得作为新的 runner defaults。

三任务当前 progressive best macro-F1 为 BBB `0.7563`、Bioavailability `0.7608`、Skin `0.6119`；matched
MiniMol-head 分别为 `0.6674/0.5872/0.6030`。完整 level 曲线、resource statistics、row ledger、artifact
roots 和唯一绘图入口见 `ASSAY_LEVEL_RETRIEVAL.md`。

## Historical train-only assay-level retrieval scaling（2026-08-17，scaffold-valid）

新 assay-level 版本不使用 direct/mechanism family；它按冻结 biological relevance 排序选择累计
`canonical_assay_context` prefix，在每个 assay 内做 Morgan top-3、Tanimoto `>=0.3` 的 train-reference
retrieval，再把相同 reference molecule 跨 assays 合并为一个 flat branch。Valid 只允许从 train molecules
检索，继续使用 `identity_blind + parent_disjoint`。Schedule 从 5 开始每次乘 4，最后一点固定为全部 eligible
assays。

| task | all assays | all-assay macro-F1 | best prefix | best macro-F1 | historical group best | delta |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 22,820 | 0.716634 | 20,480 | **0.720856** | full-flat 0.676241 | +0.044615 |
| Bioavailability | 1,840 | 0.579121 | 20 | **0.640151** | full-flat 0.677652 | -0.037501 |
| Skin | 12,015 | 0.613003 | 80 | **0.626714** | direct 0.616345 | +0.010369 |

新增 1,797 个 sample-condition 全部成功、0 failure。三个 task-best 的平均 macro-F1 为 `0.662574`，
historical group-best 平均为 `0.656746`。但 none/group reference 来自既有 OpenRouter Flash artifacts，
assay points 来自 PARCC `DeepSeek-V4-Flash-0731`，故 delta 是 descriptive historical comparison，不能写成
endpoint-matched model comparison。完整 pipeline、英文图、retrieval-volume 曲线和维护入口见
`ASSAY_LEVEL_RETRIEVAL.md`。

## Historical direct-only-heldout-filtered assay curve（2026-08-18 checkpoint）

当时的 v2 不再把全部 reference molecules 限制为 train。它从 Stage-03 eligible records 中只删除 heldout
parents 的 benchmark-defining direct rows，保留 non-direct rows；query-time 仍对所有 sources 执行
`identity_blind + parent_disjoint`。三个 index 的 heldout direct overlap 和所有 replay 的 query-parent overlap
均为 0。默认 19-condition schedule 删除与 all 过密的 BBB Top-20,480 和 Bio Top-1,280，并通过完整-batch
skip、跨 prefix single reuse、evidence-equivalent carry-forward 和 stage checkpoint 恢复减少调用。

该 checkpoint 仅 BBB Top-5/20/80/320/1,280/5,120 完整，macro-F1 为
`0.646214/0.641403/0.648371/0.680329/0.724722/0.763566`，每点均 366/366、0 failed；BBB all、Bio 和 Skin
尚未形成完整 metrics，不能做最终 task-best 或 group-level comparison。完整 source-filter 计数、reuse caveat、
命令和 artifact roots 统一见 `ASSAY_LEVEL_RETRIEVAL.md`，避免在总账重复维护运行细节。

## GPT-OSS-120B MiniMol top-5/no-threshold sensitivity（2026-08-16，scaffold-valid）

冻结 launcher 对三个当时的 current lineages 串行调用共享 matrix；MiniMol cosine retrieval 使用
`top_k_per_group=5`、`min_similarity=0`、identity-blind、parent-disjoint，并只运行 Starling direct/full-flat。

| task | direct | full-flat | n | failures |
|---|---:|---:|---:|---:|
| BBB | 0.652832 | **0.690073** | 366 | 0 |
| Bioavailability | 0.594314 | **0.623542** | 209 | 0 |
| Skin | 0.599776 | **0.612770** | 245 | 0 |

该 valid-only sensitivity 不替代默认 top-3/0.3 设置。维护入口为
`run_minimol_valid_matrix_gpt_oss_120b.py`，共享 runtime 和 artifact root 见
`baselines/minimol/README.md`。

## OpenRouter DeepSeek-V4-Flash 对照（2026-08-14，scaffold-valid）

按统一 current-valid 大图中已有的 DeepSeek-V4-Pro 条件逐项运行 OpenRouter
`deepseek/deepseek-v4-flash`；只排除已失败且无推广价值的 BBB residual/recheck diagnostics。所有 fresh
条件保持与 Pro 相同的数据、prompt、identity-blind、parent-disjoint 和 retrieval 设置：BBB/Bio 使用 Morgan，
Skin canonical direct/AOP 使用 MiniMol cosine，均为 top-3、最低相似度 0.3；单一 global prompt pool 的
并发为 128。12 个 fresh batch 全部 `n_successful=n_total`、`n_failed_runs=0`；Skin AOP gate 另有 1 个
model-dependent trigger，其 fresh final 没有造成 prediction flip。

| task | condition | DeepSeek-V4-Pro | DeepSeek-V4-Flash | Flash - Pro |
|---|---|---:|---:|---:|
| BBB | none / direct / full-flat / full-mechanism | 0.6086 / 0.7000 / 0.7136 / **0.7184** | 0.4890 / 0.6510 / **0.6762** / 0.6520 | -0.1196 / -0.0489 / -0.0373 / -0.0663 |
| Bioavailability | none / direct numeric / direct full / full-flat | 0.5019 / 0.6126 / 0.6516 / **0.7346** | 0.5199 / 0.6175 / 0.6301 / **0.6777** | +0.0181 / +0.0049 / -0.0215 / -0.0569 |
| Skin | none / matched-label direct / canonical direct / direct+AOP / AOP-gated | 0.6364 / 0.6043 / **0.6410** / 0.6123 / **0.6410** | 0.5603 / 0.5956 / **0.6163** / 0.5917 / **0.6163** | -0.0762 / -0.0087 / -0.0247 / -0.0207 / -0.0247 |

因此 Flash 的 task-best macro-F1 比 Pro 分别低 `0.0421/0.0569/0.0247`（BBB/Bio/Skin）；只有 Bio
`none` 和 `direct numeric` 两个较弱条件有小幅正 delta，不能把 Flash 视为等价替代。它的 best retrieval
仍明显高于自身 none：BBB `+0.1872`、Bio `+0.1577`、Skin `+0.0561`，说明 evidence 对 Flash 仍有用，
但没有补回相对 Pro 的 reasoning gap。

成功 artifact 中的 7,370 个响应合计记录 39,119,018 input tokens、15,793,923 output tokens 和约
`$7.7412`；运行 key 的 usage 增量约 `$7.9167`，后者还包含 preflight/smoke 以及被中断的 transport-hang
请求。Skin 首轮 600 秒 timeout 出现明显 OpenRouter 长尾；断点恢复只重跑缺失 stage，并将纯 transport
timeout 降至 180 秒，模型、prompt、token 上限和已完成 artifact 均未改变。

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_current_valid_openrouter_deepseek_v4_flash/
outputs/paper/skin_aop_gated_final_v1_scaffold_valid_openrouter_deepseek_v4_flash/
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_multimodel/figures/starling_model_comparison_highres.png
```

统一图继续使用唯一 `plot_starling_model_comparison.py` 入口；Flash 追加到既有 experiment TSV，没有新增
一次性绘图模块。BBB residual/recheck 行已从图中删除，其冻结历史 artifacts 仍保留但专用实现代码不再维护。

## DeepSeek-V4-Pro scaffold-test reference-pool sensitivity（2026-08-13）

这是 valid 开发完成后的 **test post-selection sensitivity**，用于回答将全部已标注 valid 分子加入 reference
pool 后的性能；它不是 train-only formal primary，也不覆盖下文 GPT-OSS/GLM 的冻结结论。Agent 的 canonical
Starling index 删除全部 test parents，但仍保留现有合同允许的 non-heldout residual inference evidence，不能将其
简称为“只含 labeled train+valid”；Morgan/MiniMol KNN 则严格只读取 labeled train+valid JSONL。三 task 继续使用
各自当时的 lineage 与默认 prompt：BBB `experimental_meaningful_cns_access_v2`，Bio/Skin
`record_supported_v2`。执行为 official DeepSeek API、`deepseek-v4-pro`、identity-blind、parent-disjoint、
全局并发 256；BBB/Bio 用 Morgan，Skin 按默认使用 MiniMol cosine，均为 `k=3, min_similarity=0.3`。

| task | none | direct | full-flat | full-mechanism | Morgan KNN | MiniMol KNN | MiniMol trained head |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.5808 | **0.6908** | 0.6854 | 0.6808 | 0.5927 | 0.6396 | 0.6133 |
| Bioavailability | 0.5227 | 0.6502 | **0.7392** | 0.7150 | 0.7230 | 0.6845 | 0.6359 |
| Skin Reaction | **0.6792** | 0.6460 | 0.6097 | 0.5949 | 0.5870 | 0.5172 | 0.4828 |

12 个 agent condition 全部完整：BBB `366/366`、Bio `209/209`、Skin `245/245`，均 0 failure；2,460 个
retrieval condition single outputs 与 frozen none 语义一致，test-parent overlap、parent-policy conflict 和
MiniMol candidate/query audit 均为 0。总实测 API 消耗为 `$59.55`；余额只属于运行时 receipt，不作为稳定文档状态。

为隔离 valid-reference 的贡献，Bio 进一步只重跑删除 valid parents 后 top-k payload 必然变化的样本：direct
14 条、flat 35 条、mechanism 35 条，共 84/84 成功，其余分别复用 195/174/174 条。完整 209 条合并结果为：

| condition | train+valid reference | canonical train-reference | delta |
|---|---:|---:|---:|
| direct | 0.6502 | 0.6416 | -0.0086 |
| full-flat | 0.7392 | 0.7271 | -0.0121 |
| full-mechanism | 0.7150 | 0.7105 | -0.0045 |

三项 train-reference minus train+valid 的 paired-bootstrap 95% CI 分别为 `[-0.0317,+0.0085]`、
`[-0.0318,0.0000]`、`[-0.0311,+0.0218]`，因此 valid reference 带来小幅正向 point estimate，但没有确定性证据。
Bio full-flat 的 canonical train-reference macro-F1 `0.7271`，比严格 labeled-train Morgan KNN `0.6801` 高
`+0.0470`；paired-bootstrap 95% CI 为 `[-0.0342,+0.1290]`，双侧 paired randomization `p=0.2752`。
在预先指定“full-flat 优于 KNN”方向下的单侧 Monte Carlo p-value 为 `0.1380`（1,000,000 swaps，
MC SE `0.000345`）；该方向是在查看结果后追问，必须标为 exploratory，不能作为 confirmatory significance。

KNN 的 reference-pool sensitivity 也单独保存。严格 labeled-train 的 test macro-F1 为 Morgan
`0.5617/0.6801/0.5641`、MiniMol KNN `0.6143/0.6484/0.5662`（BBB/Bio/Skin）；加入 valid 后分别变为
Morgan `0.5927/0.7230/0.5870`、MiniMol `0.6396/0.6845/0.5172`。因此加入 valid 并非跨任务稳定改善：
Skin MiniMol 反而下降。valid 与 test 的 strict train-only KNN 差异 bootstrap CI 也均跨 0。

本轮没有新增 Bio-specific launcher 或独立小图模块。train/train+valid scope 由已有通用入口参数化，选择性重跑
通过 frozen retrieval diff、已有 `--indices` 和 shared prompt pool 完成；结果继续接入唯一总图入口。

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_current_v2_test_train_valid_reference_v1_deepseek_v4_pro/
outputs/paper/starling_benchmark_results_scaffold_test_train_valid_reference_v1_deepseek_v4_pro/
outputs/baselines/structure_knn_test_train_only_reference_v1/
outputs/baselines/minimol_embedding_knn_test_train_only_reference_v1/
outputs/paper/molecular_evidence_agent_starling_scaffold_current_v2_test_train_only_reference_v1_deepseek_v4_pro_bio_selective/analysis/
```

## Current BBB experimental meaningful-CNS-access gold（scaffold-valid complete）

旧 BBB adapter 只读取 `bbb_permeability_label`，没有验证 label 来自体内 direct exposure、PAMPA/细胞模型、
计算预测还是笼统文献陈述，因此旧 19,425-parent gold 是 heterogeneous endpoint mixture。2026-08-09 冻结
`experimental_meaningful_cns_access_v2`：目标是系统给药后实验支持的 meaningful/adequate CNS access vs
restricted/poor access。它既不要求 passive permeation，也不把任何微量 CNS signal 当 positive；被动渗透、
efflux 和 influx 只作为 agent mechanism evidence。CSF 作为独立 proxy family 保留，不宣称等于 brain parenchyma。
Gold 不按 mechanism/endpoint family 配额平衡；group 是否有发挥空间由新 train-only evidence index 的 coverage
audit 判断，不能通过重标 gold 人为制造。

多轮 deterministic source-review tightening 依次修复了 calculated logBB 漏网、`CSF half-life` 被误当 entry、
间接 efficacy inference、intracisternal/mixed delivery、altered barrier、query/analyte/PMID mismatch、
parent/metabolite/total-radioactivity ambiguity、metal-complex parent collapse 和 qualitative/quant-value 方向冲突。
最终结果：

| item | result |
|---|---:|
| frozen source rows | 304,845 |
| accepted experimental outcome rows | 8,273 |
| binary parents Y=0 / Y=1 | 967 / 2,700 |
| scaffold train / valid / test | 2,935 / 366 / 366 |
| valid multi / singleton; Y=0 / Y=1 | 345 / 21; 97 / 269 |
| test multi / singleton; Y=0 / Y=1 | 344 / 22; 97 / 269 |
| identity/scaffold pairwise overlap | 0 / 0 |

accepted-parent evidence family coverage 为 brain tissue 2,160、brain/systemic ratio 938、CSF 563、
PET/autoradiography 661、unbound brain 137、explicit in-vivo BBB outcome 146；family 会在同一 parent 重叠。
三名 `gpt-5.6-sol` reviewer 在 replacement rounds 中人工审计了 366 条 unique source records；最终 294 条
family×label 分层 sample 全部通过，但不能替代 paper freeze 前的双人原文 annotation。build fingerprint 为
`0a864e56c583f79768427dbb8c4f3e17f4fca43b2eab7d2e791ba3bb34c3eb91`。

新旧 BBB 全量 parent 集合有 3,375 个 identity overlap，其中 3,272 个 label 相同，83 个由旧 1 变新 0，
20 个由旧 0 变新 1；更大的变化是新 contract 排除了大量不满足实验 meaningful-access 定义的旧 parents，
而不是大规模翻转共同 parent 的 label。两版 scaffold-valid 只重叠 30 个 parent，因此旧 500-row 与新
366-row error rate 不能作全样本 paired before/after 因果比较。

```text
tools/chembl_tool/tasks/bbb_martins/experimental_meaningful_cns_access_benchmark.py
tools/chembl_tool/common/starling/build_bbb_experimental_meaningful_cns_access.py
tools/chembl_tool/common/starling/audit_bbb_experimental_meaningful_cns_access.py
data/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/
```

所有旧 BBB baseline/agent metrics 均针对旧 gold，不是新任务性能。新 lineage 已用 valid+test union 重建
heldout-parent-filtered direct/full indices：direct 从 29,731 rows 过滤到 28,878，full 从 47,419 过滤到
45,863；两者 732/732 held-out identities 全部匹配，residual overlap 为 0。366 个 valid query 在 frozen
`k=3, similarity>=0.30, parent_disjoint` 下的 usable-neighbor coverage 为 direct 96.7%、passive 46.2%、
efflux 86.3%、influx 43.4%，四组同时有 evidence 的 query 为 109/366（29.8%）。因此 mechanism groups 有
发挥机会，但 passive/influx 并非均匀覆盖；direct/full 两个 index 的 direct-neighbor parity failure 为 0。

GPT-OSS-120B identity-blind scaffold-valid 四条件均为 366/366 成功、0 failure；三类 train-label baseline 也已
按新 train split 重跑：

| method | accuracy | macro-F1 |
|---|---:|---:|
| none | 0.5546 | 0.5526 |
| Starling direct | 0.6667 | 0.6452 |
| Starling full / flat | 0.6913 | **0.6690** |
| Starling full / mechanism | 0.6885 | 0.6665 |
| MiniMol trained head | **0.7268** | 0.6581 |
| Morgan KNN k=3 | 0.7049 | 0.5896 |
| MiniMol cosine KNN k=3 | 0.6885 | 0.6002 |

逐样本配对表明 retrieval 本身有效，但不同 full organization 仍没有稳定差异。`none -> direct` 的 macro-F1
delta 为 `+0.0927`（paired-bootstrap 95% CI `[+0.0412,+0.1435]`；72 rescue/31 harm；McNemar
`p=6.60e-5`）；`direct -> full/flat` 为 `+0.0237`（`[-0.0116,+0.0589]`；28/19；`p=0.2430`）；
`full/flat -> full/mechanism` 为 `-0.0024`（`[-0.0356,+0.0309]`；20/21；`p=1.0`）。Full/flat 相对
MiniMol head 的 macro-F1 高 `1.08 pp`，但 accuracy 低 `3.55 pp`；两个指标交叉，不能把它描述成 agent
已经全面胜过 trained head。

“额外 groups 没覆盖”也不是完整解释：在四个 family 都有 neighbor 的 109 个 query 中，`direct -> full/flat`
恰好 10 rescue/10 harm，`full/flat -> full/mechanism` 为 7/9；在其余 257 个 query 中则分别为 18/9 和
13/12。即使每组都有发挥机会，额外 mechanism evidence 也没有稳定净增益，问题还包括 evidence
方向/transferability 的噪声和 group interpretation，而不仅是 molecule information 是否存在。

Full-mechanism 的 114 个错误中，direct retrieval proxy wrong/inconclusive 为 40/18，direct group 丢失
gold-aligned proxy 为 42，mechanism conflict 为 9，final aggregation 仅 5。该 proxy 是 visible-neighbor
direction 的描述性诊断，不是新的 label policy，也不证明 analog chemical transferability。它说明新 gold 和
task contract 修复了评估含义，但没有把主要瓶颈转移到 final；下一步仍应优先改善 evidence reliability 和
group-level analog transfer，而不是 final-only ICL、更多 raw cards 或 group ablation。
错误明显偏向 false negative：full-mechanism 为 18 FP / 96 FN，且 114 个错误的 single prior 中 77 个为
`low`、32 个 `moderate`、只有 5 个 `high`。即使已把 final contract 改成 meaningful CNS access，agent 仍比
gold 分布更保守；这为“task calibration 尚未完全对齐”提供了 trace evidence，但不能单凭 single prior 证明
某个具体 final 错误的因果来源。
错误率也随 evidence/gold ambiguity 上升：direct top similarity `0.4–0.6` 的 query 为 42/110（38.2%）错误，
`>=0.6` 为 56/201（27.9%）；non-unanimous gold 为 17/43（39.5%），unanimous 为 97/323（30.0%）。
这些是相关性诊断，不足以单独归因，但与“analog transfer 和 record uncertainty 比 final compression 更关键”
的 error-path 分解一致。
Allowed-value validation 覆盖 366 个 single 和 998 个实际 group outputs；其中 1 个 group 首次响应不合法并在
同 setting 第二次成功，最终 invalid completed outputs 为 0。该结果证明 reliability gate 生效，不把这一次
格式修复计作性能方法。

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2/
outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/
outputs/paper/starling_benchmark_results_scaffold_experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/
outputs/paper/starling_trace_failure_cause_audit_experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/
```

## 0. Bioavailability/Skin current；BBB historical record-supported v2 scaffold benchmark

v2 继续使用第一版相同的 70% record-weighted parent label，只改变 scaffold split 分配。其 BBB 行当前只作
historical comparison；Bioavailability/Skin 仍为 current。分配优先保证
scaffold-disjoint，并按 lexicographic MILP 依次最小化 held-out singleton、valid/test singleton imbalance、
label imbalance，最后才最大化第一版 valid molecule 复用。BBB valid/test 上限各 500；Bioavailability
和 Skin 各取总 parent 的 10%。

| task | train / valid / test | valid multi/single | test multi/single | valid/test Y=0,Y=1 | first-version valid reuse |
|---|---:|---:|---:|---:|---:|
| BBB_Martins | 18,425 / 500 / 500 | 500 / 0 | 500 / 0 | 139,361 / 139,361 | 81 |
| Bioavailability_Ma | 1,674 / 209 / 209 | 209 / 0 | 209 / 0 | 58,151 / 58,151 | 110 |
| Skin_Reaction | 1,966 / 245 / 245 | 240 / 5 | 240 / 5 | 73,172 / 73,172 | 72 |

所有 identity/scaffold pairwise overlap 都是 0。Skin 的 10 个 held-out singleton 是精确 245/245 下的
全局最小值，并均衡分配为 valid/test 各 5 个。当前入口与 canonical roots：

```text
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
data/processed_starling_record_supported_v2/
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
```

五个 v2 Starling held-out indices 均通过 `zero_parent_overlap=true` 和
`n_residual_heldout_parent_identities=0`。以下完整 22-condition scaffold matrix 使用 `parent_disjoint`
retrieval，分别运行 GPT-OSS-120B 与 GLM-5.2 的 identity-blind 和 deployment-visible；现作为 historical
combined record-supported-v2/legacy-prompt comparison。
历史复用按 molecule key 和严格 stage contract 审计；不会按旧 query index 搬运。

该 historical GPT-OSS-120B blind/visible 两套矩阵均已完成 22 conditions、6,887/6,887 sample-conditions，失败为 0。
最佳 macro-F1 与三种 train-label baseline 如下：

| task | best 120B blind | blind | best 120B visible | visible | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---|---:|---|---:|---:|---:|---:|
| BBB_Martins | Starling Full / Flat | 0.6753 | Starling Direct | 0.6900 | **0.7226** | 0.5857 | 0.6907 |
| Bioavailability_Ma | Starling Direct / Full | 0.6031 | Starling Full / Flat | **0.6787** | 0.6755 | 0.5856 | 0.6149 |
| Skin_Reaction | Starling Full / Mechanism | 0.6010 | Starling Full / Mechanism | **0.6151** | 0.5749 | 0.5201 | 0.5749 |

结果 roots：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_blind/
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid_gpt_oss_120b_visible/
outputs/baselines/minimol_starling_record_supported_v2/
outputs/baselines/structure_knn_starling_record_supported_v2/
outputs/baselines/minimol_embedding_knn_starling_record_supported_v2/
```

新旧 dataset 的 GPT-OSS-120B 全设置与三种 baseline 由唯一总图入口生成，图中明确标注 v2 的
multi-record-heavy held-out 特征：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/figures/
  gpt_oss_120b_dataset_version_comparison.svg
  gpt_oss_120b_dataset_version_comparison.png
```

Morgan KNN 的 v2 下降不是 train size 变小造成。严格 leak-free 的 old/new train x valid 交叉分解显示：
BBB 总下降 `-0.1270` 中 valid cohort 替换贡献 `-0.1291`；Skin 总下降 `-0.0862` 中 valid cohort 替换贡献
`-0.0505`，同时移除/加入 train rows 共贡献 `-0.0357`；Bioavailability 总下降 `-0.0343` 主要来自新增
train composition（`-0.0218`）。新 valid 的 unique scaffold 数从 BBB/Skin/Bio 的 `277/133/151`
增至 `377/187/175`，Morgan 的负类 recall 分别从 `0.520/0.392/0.361` 降至
`0.324/0.274/0.276`。因此最稳妥的结论是：v2 把 held-out 重心移到 multi-record、更多样且更难迁移的
scaffold，暴露了 Morgan local-neighborhood 的 domain-shift 弱点；不是 label disagreement，也不是 leakage。
可复核诊断位于：

```text
outputs/paper/starling_benchmark_results_scaffold_record_supported_v2_valid/diagnostics/morgan_dataset_shift/
```

GLM blind 首轮在 endpoint/tunnel 中断前完成 5,876/6,887 sample-conditions，仍有 1,011 个失败；其中
BBB 仅余 3 个，Skin 余 314 个，Bioavailability 余 694 个。该 root 尚未通过 zero-failure gate，visible
矩阵尚未启动，因此所有 partial GLM macro-F1 都不得进入正式表或显著性分析。恢复 endpoint 后必须使用
同一 root 的 `--skip-existing` 修复，再启动独立 visible root。

### Final evidence surface / aggregation bottleneck 诊断

为检验不同 retrieval 的有效信息是否在 group-to-final 二级总结中丢失，2026-08-07 对 GPT-OSS-120B
identity-blind `starling_full_flat` scaffold-valid 做了 matched final-only ablation。每个目标 run 的
retrieval、single 和 group artifacts 逐文件复制 frozen source，只有 final prompt surface 和 fresh final call
改变：历史 `summary_only` control、`summary_plus_cards` 和 `cards_only`。Card v1 最多保留 12 个 analog、
每 analog 最多 3 条去身份 evidence rows，并包含已有 prefetched molecular comparison 文本。

六个候选 batch 共 `1,908/1,908` 成功，失败为 0；全量 audit 对 1,908 个 target run 检查 source/target
artifact SHA-256、trace card SHA-256、surface provenance 和 identity leak，结果为 0 failure。

| task | summary-only | summary + cards | delta (95% paired CI) | cards-only | delta (95% paired CI) |
|---|---:|---:|---|---:|---|
| BBB | 0.6753 | 0.6737 | -0.0016 [-0.0244,+0.0217] | 0.6467 | -0.0285 [-0.0666,+0.0104] |
| Bioavailability | 0.5707 | 0.5718 | +0.0011 [-0.0376,+0.0392] | 0.5780 | +0.0073 [-0.0435,+0.0593] |
| Skin Reaction | 0.5577 | 0.5445 | -0.0132 [-0.0524,+0.0261] | 0.5595 | +0.0018 [-0.0486,+0.0522] |

六个 bootstrap interval 均跨 0，exact McNemar 的 Holm-adjusted p 均为 1。Cards 将 final mean prompt
tokens 增加到 summary-only 的 `3.32x–4.72x`，仍没有稳定增益。因此现有结果不支持“raw evidence 已经正确
retrieved、只是 final 看不到”作为主要解释；card surface 不进入默认 pipeline 或 formal test。

同一轮 metadata census 说明 molecule information 不是整体缺失：endpoint/context/scope 覆盖接近 100%；
但 measurement available 仅为 BBB `51.5%`、Bioavailability `90.2%`、Skin `28.8%`，可评估 measurement
consistency 的比例仅 `21.7%/52.7%/20.7%`，可评估 multi-record/PMID agreement 的比例为
`44.4%/58.8%/66.2%`。这些数字只表示 metadata 是否足以开展进一步审计，不证明 evidence utility，
也不触发 compatibility selector。下一步先修正 Skin task scope，再从现有 traces 区分 upstream reasoning
failure 与 final aggregation failure；不继续增加 raw context 或做 group add/drop 搜索。

```text
tools/chembl_tool/common/final_evidence_surface.py
tools/chembl_tool/common/evidence_compatibility.py
tools/chembl_tool/paper_experiments/run_final_evidence_surface_experiment.py
tools/chembl_tool/paper_experiments/audit_final_evidence_surface_contract.py
tools/chembl_tool/paper_experiments/summarize_final_evidence_surface_experiment.py
outputs/paper/final_evidence_surface_record_supported_v2_valid_gpt_oss_120b/
```

### Skin task-alignment 修复与 valid 结果

2026-08-08 使用已有 GPT-OSS-120B scaffold-valid `starling_full_mechanism` traces 做了确定性 audit，未新增
模型调用，也未查看 formal test。源 condition 为 legacy `legacy_skin_reaction_v1`，245 条 valid 中 90 条错误
（30 FP、60 FN）。严格 Tier 1/2 signal gate 要求 group 同时满足 useful、high/moderate transferability、
high/moderate confidence 和明确方向：

| error category | n | 含义 |
|---|---:|---|
| final-recoverable | 6 | 已有无冲突的 gold-aligned Tier 1/2 signal，final 仍选错 |
| upstream conflict | 3 | Tier 1/2 同时出现两个方向的强 signal |
| upstream wrong direction | 39 | Tier 1/2 只支持错误方向 |
| upstream insufficient | 42 | Tier 1/2 没有满足 gate 的方向性 signal |

因此 `84/90 = 93.3%` 错误不是严格 final-only 可恢复，final-recoverable 仅 `6/90 = 6.7%`。另有
`23/90 = 25.6%` 错误把 phototoxicity、irritation/corrosion 或 exposure 作为 final 主 evidence type，说明
legacy scope bug 真实存在；但只有 2 条同时 final-recoverable，不能把越界引用直接等同于 final 能修复。

`sensitization_aligned_v2` 已同步收紧 single/group/final contract，并保留 legacy v1 复现路径和跨 profile
artifact-reuse guard。2026-08-09 的 fresh GPT-OSS-120B scaffold-valid 四条件全部零失败：

| condition | accuracy | macro-F1 |
|---|---:|---:|
| none | 0.5551 | 0.5225 |
| Starling direct | **0.6204** | **0.5725** |
| Starling full / flat | **0.6204** | 0.5698 |
| Starling full / mechanism | 0.6000 | 0.5423 |

`none -> direct` 为 `+0.0500` macro-F1（paired-bootstrap 95% CI `[-0.0224,+0.1195]`）；
`direct -> full/flat` 为 `-0.0027`（`[-0.0638,+0.0605]`）；`full/flat -> full/mechanism` 为
`-0.0276`（`[-0.0891,+0.0330]`）。后两项仍不支持 full organization 带来稳定提升。

同 cohort 的 legacy full-mechanism 与 aligned-v2 fresh run 有 66 个 prediction flips；macro-F1 delta
`-0.0587`（`[-0.1250,+0.0089]`，McNemar `p=0.389`）。由于两边都是 fresh generation，这不是纯 prompt
causal estimate；能直接确认的是 scope compliance：legacy 90 个错误中有 23 个 scope-contaminated，v2 的
98 个错误中为 **0**。v2 error audit 只找到 3/98 final-recoverable；其余为 upstream conflict 2、wrong
direction 43、insufficient 50。修复确实消除了 label-scope bug，却没有提升性能，且进一步排除了 final-only
ICL：当前主瓶颈仍是 retrieval/group signal 的方向与 transferability。

```text
tools/chembl_tool/tasks/skin_reaction/prompt_profiles.py
tools/chembl_tool/paper_experiments/audit_skin_reasoning_bottleneck.py
outputs/paper/skin_reaction_task_alignment_audit_record_supported_v2_valid_gpt_oss_120b/
outputs/paper/skin_reaction_task_alignment_audit_record_supported_v2_valid_gpt_oss_120b_sensitization_aligned_v2/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_bugfix_v2/
```

#### Skin direct-source scope parity 与 negative-transfer gate

后续 source audit 发现历史 Tier-1 evidence profile 没有复用 gold 的 sensitization/contact-allergy scope filter，
会把 photoallergy、irritation、urticaria 等 records 聚合进 direct cards。versioned scoped source 过滤 5,049 条
scope 外 records，direct molecules 从历史 broad collection 收紧为 3,275。245-query retrieval replay 中 37 个
top-3 identity lists 改变，41 个旧 retrieved neighbors 移除、20 个新 neighbors 回填；203 个 visible direct
contexts 因 cards/counts 重建而改变。

clean-index aligned-v2 direct 的 macro-F1 为 `0.5666`，相对 broad-index aligned-v2 direct 的 `0.5725`
delta `-0.0059`（95% CI `[-0.0724,+0.0603]`；55 flips，28/27 reference/candidate-only correct，McNemar
`p=1.0`）。scope parity 必须保留为数据合同修复，但没有通过性能提升 gate。

clean-index direct traces 仍显示 low/moderate negative analog direction 为 6 gold-aligned / 13 gold-opposed，
因此按 frozen continuation gate 只运行一个 `sensitization_negative_transfer_v3` 候选。它将 negative direction
从 25 降到 3，剩余 3 个均对应 Y=0；但 macro-F1 降至 `0.5531`、Y=0 recall 降至 `0.3425`。相对 clean-index
v2 delta `-0.0135`（95% CI `[-0.0882,+0.0612]`）。该候选不 promotion，不运行 full-flat、full-mechanism
或 test；`sensitization_aligned_v2` 继续是默认 prompt profile。

```text
tools/chembl_tool/paper_experiments/audit_skin_direct_scope_retrieval.py
outputs/paper/skin_direct_scope_retrieval_audit_record_supported_v2_valid/
outputs/paper/skin_negative_transfer_v3_audit_record_supported_v2_valid/
```

### BBB/Bio train-ratio tie-break final-only 诊断

2026-08-09 在 current BBB/Bio scaffold-valid `starling_full_flat` 上运行了一个预先收窄的 final-only 候选。
候选先把综合证据声明为 `consistent_positive / consistent_negative / mixed / insufficient`；consistent 状态禁止
prior 覆盖，只有 mixed/insufficient 且两类确实平局时才按 frozen train majority 打破平局，并显式记录
`prior_used`。BBB train 为 2162/2935 Y=1，Bio 为 1213/1674 Y=1；Skin 未运行。

| task | control macro-F1 / acc | candidate macro-F1 / acc | macro-F1 delta (95% paired CI) | flips | prior used | prior corrected / broken |
|---|---:|---:|---|---:|---:|---:|
| BBB | 0.6690 / 0.6913 | 0.6672 / 0.6967 | -0.0017 [-0.0272,+0.0236] | 22 | 10 | 2 / 4 |
| Bioavailability | 0.6117 / 0.6220 | 0.6203 / 0.6364 | +0.0086 [-0.0325,+0.0478] | 19 | 10 | 5 / 2 |

两批合计 575/575 成功、零失败；575 个 candidate manifests 和 1,725 对 copied retrieval/single/group artifacts
全部通过 provenance/SHA parity。BBB prior-triggered rows 净伤害 2 个，Bio 净挽救 3 个；但每 task 只有 10 个
trigger，而且 paired interval 都跨 0。`prior_used=false` rows 仍有 BBB 16、Bio 12 次 flip，说明 evidence-state
合同本身也改变了 final generation，所以总体 delta 不是 class prior 的纯因果效应。预冻结 promotion gate 要求
macro-F1 正增益、bootstrap lower bound > 0 且 accuracy 不下降；两者均 FAIL，不启动 test、不升级默认，也不继续
增加 prior 强度或做 batch quota。

```text
tools/chembl_tool/common/final_decision_prior.py
tools/chembl_tool/paper_experiments/run_train_ratio_prior_experiment.py
tools/chembl_tool/paper_experiments/train_ratio_prior_analysis.py
outputs/paper/train_ratio_tiebreak_v1_scaffold_valid_gpt_oss_120b/
```

### Matched train-label direct agent vs Morgan KNN

为回答“相同 retrieval 下 reasoning 是否比 KNN 多数票更好”，2026-08-09 完成 E15 scaffold-valid 诊断。
每个 query 的两种方法严格共享正式 Morgan KNN top-3 scaffold-train neighbors、rank、similarity 和 train
`Y`；KNN 使用未加权多数票，GPT-OSS-120B direct agent 只额外使用 identity-blind query properties、逐邻居
MMP/property comparison、group reasoning 和 final synthesis，不读取其它 Starling records。

三批共 820/820 成功、0 failure。逐 query 审计确认 retrieval parity mismatch 为 0、820 个 LLM-facing prompt
identity leak 为 0，每个 query 恰有三条 `frozen_benchmark_train_label` evidence。Formal test 未运行。

| task | KNN acc / macro-F1 | matched agent acc / macro-F1 | agent-KNN macro-F1 (95% CI) | agent-only / KNN-only |
|---|---:|---:|---:|---:|
| BBB | 0.7049 / 0.5896 | 0.6557 / 0.6411 | +0.0514 [-0.0156,+0.1202] | 68 / 86 |
| Bioavailability | 0.7177 / 0.5856 | 0.5359 / 0.5261 | -0.0595 [-0.1570,+0.0348] | 31 / 69 |
| Skin Reaction | 0.6204 / 0.5201 | 0.6327 / 0.5610 | +0.0408 [-0.0388,+0.1201] | 34 / 31 |

Agent 在三项任务都更偏向预测 Y=0。BBB 的 Y=0 recall 从 `0.3299` 升到 `0.8557`，但 Y=1 recall 从
`0.8401` 降到 `0.5836`，因此 macro-F1 上升而 accuracy 下降 `4.92 pp`；Skin 的变化较小。Bio 的 Y=0
recall 从 `0.2759` 升到 `0.7069`，但 Y=1 recall 从 `0.8874` 降到 `0.4702`，导致 macro-F1 和 accuracy
同时下降，McNemar `p=0.00018`。

投票强度分层揭示了主要机制。三邻居全票一致时，BBB 的 65 次改票中 64 次为 `1→0`，rescue/harm 为
`17/48`；Bio 的 50 次改票中 48 次为 `1→0`，rescue/harm 为 `9/41`；
在 2:1 分裂时则为 BBB `51/38`、Bio `22/28`。Skin 的全票一致与 2:1 分层分别为 `12/8` 和 `22/23`。
这说明 BBB reasoning 在真正有分歧的局部邻域中可能有一定 adjudication value，但当前 agent 无约束地推翻
强监督一致性的代价更大，而且该信号没有跨 task 泛化。结论为 no-go：不能声称 matched direct agent
优于 KNN，不运行 formal test，也不把该 train-label direct source 升级为正式 pipeline 条件。

同一批 valid molecules 进一步与原 heldout-filtered full-Starling direct 做逐样本配对。Train-only label direct
相对 full-pool direct 的 macro-F1 delta 为 BBB `-0.0041`（95% CI `[-0.0541,+0.0452]`）、Bio
`-0.0316`（`[-0.0980,+0.0338]`）、Skin `-0.0116`（`[-0.0837,+0.0610]`）；三个 point estimate 都偏向
full pool，但区间全部跨 0。Full pool 的 median top-1 similarity 明显更高：BBB/Bio/Skin 为
`0.642/0.580/0.480`，train-only 为 `0.378/0.384/0.370`。其 coverage 反而较低
（`0.967/0.943/0.829` vs train-only 全部 `1.0`），因为原 direct 保持 `similarity>=0.30`，KNN 则固定回填
三个 train neighbors。

因此额外 Starling pool 确实提供了更近的 analog，且 performance point estimate 没有显示净伤害；但当前只在
Bio 有约 3.2 pp 的较大差值，仍未通过 paired uncertainty gate。该比较还同时改变 raw record evidence 与
aggregated train-label evidence、threshold 和 neighbor set，不能把差值纯归因于“数据数量”。最稳妥结论是：
extra Starling data 在 retrieval quality 上有价值，在最终预测上的普遍增益尚未被证明。

```text
tools/chembl_tool/paper_experiments/matched_train_label_agent/
outputs/paper/matched_train_label_direct_agent_v1_scaffold_valid_gpt_oss_120b/analysis/
```

#### BBB meaningful-CNS adjudication profiles（v2/v3；均未 promotion）

针对 BBB matched trace 中大量无约束 `1→0` 改票，2026-08-09 运行了两个单候选、valid-only 的 versioned
contract。两者都把 source outcome direction 与 query transferability 分开，并新增可审计的
`integrated_outcome_state` / `negative_evidence_basis`；历史合同冻结为 `meaningful_cns_access_v1`，旧 manifest
缺字段时归入 v1，跨 profile branch reuse 被拒绝。Retrieval、dataset、model、visibility 和 parent policy 均不变；
formal test 未运行。

v2 `meaningful_cns_adjudication_v2` 要求 fail 必须来自可迁移的 restricted-CNS outcome 或 measured efflux。
它把规则收得过窄：matched valid 的 pass 数从 v1 的 171 增至 322，accuracy 从 0.6557 提高到 0.7295，
但 Y=0 recall 降到 0.2165，macro-F1 从 0.6411 显著降至 0.5652（paired delta `-0.0759`，95% CI
`[-0.1482,-0.0020]`）。因此 v2 立即失败，不进入 full pool/test/default。

v3 `meaningful_cns_adjudication_v3` 保留 transferable-positive protection，但允许在 state 为 mixed/insufficient、
没有 moderate/high transferable positive 且至少两个独立严重屏障一致时，用
`convergent_intrinsic_barriers` 支持 fail。366 条 matched 与 366 条 full-Starling direct 均为 0 failure，全部
首轮通过 validation：

| evidence condition | v1 acc / macro-F1 | v3 acc / macro-F1 | paired macro-F1 delta (95% CI) | v1-only / v3-only correct | v3 Y=0/Y=1 recall |
|---|---:|---:|---:|---:|---:|
| matched train-label direct | 0.6557 / 0.6411 | 0.7022 / **0.6696** | +0.0285 [-0.0141,+0.0709] | 23 / 40 | 0.7320 / 0.6914 |
| full-Starling direct | 0.6667 / **0.6452** | 0.6776 / 0.6416 | -0.0036 [-0.0471,+0.0402] | 30 / 34 | 0.6804 / 0.6766 |

v3 对 matched KNN 的 macro-F1 增益为 `+0.0799`，95% CI `[+0.0098,+0.1499]`，且 accuracy 基本持平；
说明它确实修复了 train-label matched 场景的过度负向推翻。可是 current pipeline 的 full-Starling direct 只发生
64 次改票、净多做对 4 个，macro-F1 没有提高。严格 continuation gate 因此失败：默认继续使用 v1，不运行
formal test，也不再从同一 valid 继续搜索 v4。v2/v3 artifact 完整保留用于复现。

```text
tools/chembl_tool/tasks/bbb_martins/prompt_profiles.py
outputs/paper/matched_train_label_direct_agent_meaningful_cns_adjudication_v2_scaffold_valid_gpt_oss_120b_formal/
outputs/paper/matched_train_label_direct_agent_meaningful_cns_adjudication_v3_scaffold_valid_gpt_oss_120b_formal/
outputs/paper/molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2_valid_gpt_oss_120b_bbb_adjudication_v3/
```

#### BBB property-compatible analog selector（E16；valid-only，未 promotion）

进一步检查确认：如果把 scope-clean direct 定义为同一批 scaffold-train parents 的 frozen gold outcome，便与 E15
matched train-label direct 是同一个数据合同，不能作为新实验重复运行。E16 因此不重新包装 evidence，而是在完全
相同的 2,935 个 BBB train parents 内进行 label-blind retrieval availability audit。它先按正式 Morgan/Tanimoto
取 top-20，再在相对当前 rank-3 similarity 的固定成本预算内，依次按 pH 7.4 ionization-class match、train-only
IQR 标准化的 molecule-properties distance 和 Morgan similarity 选择三个 neighbors。选择过程不读取 train/valid
label；label 只在选择冻结后用于描述性 KNN vote 诊断。Morgan top-3 与正式 baseline 的 366-query parity mismatch
为 0。

属性距离固定使用 neutral fraction、estimated logD、MW、TPSA、HBD 和 rotatable bonds。3,301 个 train+valid
profiles 中 3,300 个由 `molecule_properties.v1` 完整返回；1 个 MolGpKa graph error 显式降级为 RDKit-only，
pKa/logD 保持 missing、ionization class 为 unknown，没有删除样本或填充预测值。

| rank-3 similarity cost budget | changed sets | mean property-distance gain | mean similarity cost | mean same-ion-class delta | KNN vote macro-F1 delta (95% paired CI) | low-transfer subset delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 217/366 | 0.111 | 0.0099 | +0.45 | +0.0438 [-0.0024,+0.0905] | +0.0391 [-0.0218,+0.1039] |
| 0.05 | 297/366 | 0.190 | 0.0251 | +0.77 | +0.0379 [-0.0226,+0.0978] | +0.0500 [-0.0350,+0.1354] |
| 0.10 | 332/366 | 0.224 | 0.0434 | +0.89 | +0.0284 [-0.0347,+0.0902] | +0.0419 [-0.0474,+0.1305] |

最保守的 0.02 budget 已覆盖 v3 判为 low-transferability 的 128/178 个 queries；selected top-3 全部
ionization-class matched 的 query 从 144 增至 237。其描述性 vote macro-F1 从 0.5896 升至 0.6335，accuracy
从 0.7049 升至 0.7322；36 次 vote flip 中 23 次救回、13 次损坏，且 `1->0` / `0->1` 均存在，不是简单改变
class prior。paired interval 仍轻微跨 0，因此不能报告正式 retrieval improvement；但它证明低 transferability
并非完全由 train coverage 不足造成，足以支持一个预冻结的 `0.02` selector-only valid LLM candidate。更宽预算
不继续搜索。

该唯一候选随后按 matched-v3 合同完成 366/366 valid，0 failure、0 retrieval mismatch、0 prompt identity leak；
复用了同 profile 的 frozen single branch，只重新运行受 selector 影响的 group/final。结果没有把 vote 层的正向
signal 转化成 agent 增益：

| condition | accuracy | macro-F1 | Y=0 recall | Y=1 recall |
|---|---:|---:|---:|---:|
| original Morgan top-3 + v3 agent | 0.7022 | 0.6696 | 0.7320 | 0.6914 |
| property-compatible top-3 + v3 agent | 0.7022 | 0.6708 | 0.7423 | 0.6877 |

paired macro-F1 delta 仅 `+0.0012`，95% CI `[-0.0410,+0.0450]`；58 次 agent flip 为 29 rescue / 29 harm。
217 个实际改变 neighbor set 的样本中也恰好是 21 rescue / 21 harm。更关键的是，36 个 retrieval vote 真正改变
的样本只有 5 个发生 agent prediction flip（4 rescue / 1 harm）。149 个样本只是 neighbor membership 相同，
其中 107 个被 selector 重排；真正 neighbor 顺序与 group 输入逐字相同的 rerun control 只有 42 个，仍产生
8 个 flip（5 rescue / 3 harm）。这说明 observed point estimate 与 run-level reasoning variation 同量级，不能
纯归因于 selector。Group transferability 确实被改变了 110 次，整体从 `low/moderate/high=178/180/8` 变为
`166/193/6`，所以不是 selector 没有进入 prompt；瓶颈是当前两阶段 agent 没有稳定地把更兼容的 neighbor set
转成更好的最终判断。

E16 promotion gate 因 paired CI 跨 0 而失败。该 selector 不进入默认 pipeline，不运行 formal test，也不继续
在同一 valid 上搜索权重、预算或 v4 prompt。

进一步的 deterministic paired trace diagnosis（不新增模型调用）定位了四个断点：

1. **有效 treatment 很小。** Selector 只改变 36/366 个 majority votes；这部分 5 次 agent flip 为 4 rescue / 1
   harm，净增加 3 个正确样本。23 个 retrieval-vote rescue 中 agent 原本已经做对 17 个，candidate 做对 20 个，
   所以离线 KNN 的 +10 net vote correctness 不可能原样叠加到 agent。
2. **Compatibility objective 没有成为 group 的稳定 transfer signal。** 191 个 mean property distance 真正改善的
   queries，group transferability 只发生 33 次上调、32 次下调，agent correctness 反而从 139 降到 133。
   Candidate group prompt 展示完整 `properties_compare`，但不展示 selector property distance、ionization-match
   decision 或 original Morgan rank；LLM 需要从几十个 descriptor deltas 自行重建 selector rationale。
3. **两阶段 free-text transport 不稳定。** 42 个 neighbor 顺序和 group request 逐字相同的 control 中，20 个
   group core tuples、23 个 final integrated states 改变，最终产生 8 flips。即使全体中 group core tuple 完全
   相同的 134 个样本，final 仍有 34 次 state change 和 14 flips，说明 final 还会被 reasoning summary/key evidence
   的措辞变化驱动，而不仅是结构化 state。
4. **BBB final 仍过度用 passive barriers 推翻正向 observed outcomes。** Property-compatible top-3 全为 `Y=1`
   的 166 个 queries 中，145 个 gold 也是 Y=1；但 38 个仍被 `convergent_intrinsic_barriers` 判 fail，其中 25 个
   是 false negative、只有 13 个是真正救回。全体 133 个该 basis 的 fail 中，gold 为 Y=1/Y=0 的数量是 72/61；
   `transferable_restricted_cns_outcome` 的 23 个 fail 也只有 11 个 Y=0。相反，`supports_bbb_crossing` group 的
   181 个 queries 有 156 个 gold Y=1，说明当前 negative adjudication 仍未适配 meaningful-CNS-access ontology。

一个“所有 positive-direction 都强制 pass”的无模型 counterfactual 只把 macro-F1 提高 `+0.0074`，95% CI
`[-0.0286,+0.0426]`，不能作为新 prompt 的 promotion 证据。曾讨论过把 group→final 收缩成 structured analog
ledger，但该方向会显著增加 BBB 专项状态和 compiler 复杂度，用户已明确否决，不进入计划或默认代码。
因此 BBB 当前结论是停止继续微调 final prompt、selector 或 transport contract；保留 E16 代码和 artifacts 仅供
历史复现。若未来有独立的新证据或新方法假设，应建立新的 train-only 协议，而不是从本次 valid trace 继续派生规则。

```text
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatibility.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/molecule_property_profiles.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatibility_audit.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_contract.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_experiment.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_report.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/bbb_property_compatible_trace_diagnosis.py
outputs/paper/bbb_property_compatibility_availability_experimental_meaningful_cns_access_v2_valid/
outputs/paper/matched_train_label_direct_agent_bbb_property_compatible_v1_scaffold_valid_gpt_oss_120b/
```

#### BBB DeepSeek direct-anchored residual adjudication（E20；valid-only，已终止）

2026-08-13 在同一 `experimental_meaningful_cns_access_v2` scaffold-valid 366 条、`identity_blind +
parent_disjoint`、Morgan top-3、Tanimoto `>=0.30` 合同下运行 DeepSeek-v4-pro。标准 Full-mechanism 使用
heldout-filtered Starling full index；residual 与 recheck 逐文件复用其 retrieval、single 和 mechanism-family
group artifacts，只重跑 final。Direct 来自相同数据、检索和模型的既有 matched anchor。

| condition | accuracy | macro-F1 | delta macro-F1 vs Direct | paired 95% CI | Direct-only / candidate-only correct |
|---|---:|---:|---:|---|---:|
| Direct | 0.7350 | 0.7000 | — | — | — |
| Standard Full-mechanism | **0.7486** | **0.7184** | +0.0184 | [-0.0121,+0.0496] | 13 / 18 |
| Direct-anchored residual | 0.7377 | 0.7084 | +0.0084 | [-0.0216,+0.0390] | 15 / 16 |
| Residual + override recheck | 0.7404 | 0.7109 | +0.0109 | [-0.0186,+0.0412] | 14 / 16 |

四个 batch 均零失败。Residual 把 366 条中的 181/113/68/4 条 mechanism evidence 分为
`supporting/context_only/none/decisive`，只提出 4 个 override；独立 recheck uphold/reject 各 2 个。
标准 Full-mechanism 的小幅 point gain 主要出现在 direct-anchor state 为 mixed 的 66 条：Direct、standard、
residual 分别做对 35/43/40 条。严格 residual 没有扩大收益，说明把所有非-decisive mechanism 变成
non-voting context 会丢失部分软 adjudication signal。

另有 29 条重新推导的 anchor 与冻结 Direct prediction 不同（14 rescue / 15 harm），说明“要求模型重建
Direct anchor”不是冻结 Direct output 的纯因果 ablation。虽然可以把 frozen Direct output 显式注入再做一次
诊断，但它不能解决当前 outcome evidence 稀缺和 transporter endpoint direction 不明确的问题，预期收益与
generation variation 同量级，因此不再运行。

预冻结 promotion gate 要求 hybrid macro-F1 上升、accuracy 不降且 paired CI lower bound `>=0`；E20 失败。
BBB formal test 未读取，且 BBB 方法开发在此终止。Residual 的 1,098 对、recheck 的 12 对 copied artifacts
全部通过 SHA-256 equality audit。

```text
outputs/paper/bbb_deepseek_residual_adjudication_valid_v1/{protocol.md,analysis/}
```

2026-08-14 代码整理时已删除这条失败诊断专用的 analyzer、BBB-only decision profiles 和测试；通用
`standard/train_ratio` final-decision contract 以及上述 frozen historical artifacts 均保留。标准
Full-mechanism 不是 residual profile，继续由正式通用 pipeline 支持。

#### Bio 三个正类邻居仍被改成 low 的 trace 诊断

针对 Bio 三个 train neighbors 全为 `Y=1` 的 109 个 valid 样本进行了 deterministic trace audit，不新增模型
调用。其中 gold 为 Y=1/Y=0 的数量是 94/15；agent 预测 high/low 为 61/48。48 个 low 决策中只有 9 个是
正确救回，39 个误伤，因此在该条件下“推翻三票正类并预测 low”识别真实 Y=0 的 precision 仅 `18.75%`。

相似度是改票触发器，但不是有效的 low 判别器。top-1 similarity `<0.30`、`0.30–0.39`、`>=0.40` 三档的
真实 Y=1 比例分别为 `84.4%/87.0%/87.0%`，几乎不变；agent 预测 low 的比例却为
`71.9%/47.8%/25.9%`。当前 prompt 把 `0.20–0.40` 定义为 `distant_analog`，并在 group 和 final 明确要求
没有 strong scaffold/mechanism case 时不得将其作为正负证据。结果是三个正类标签常被降成
`neutral_or_unclear`：39 个误伤中 group direction 为 neutral 24 个，group transferability 为 low 28 个。

下游还有明确的非对称默认。39 个误伤中 single prior 并非大多是明确 low，而是
`mixed_or_unclear=28, low=11`；在全部 109 个样本中，`mixed single + neutral group` 有 22 个，最终 20 个
预测 low、2 个预测 high。全体 209 个 valid 样本里，single prior 为 mixed 的 112 个样本实际有 89 个
Y=1（`79.5%`），final 却只预测 47 个 high。这表明 final 把“正类 analog 不够可迁移/信息不足”系统性地
解释为负证据，而没有按 `F>=20%` 这个较低阈值校准。

gold ambiguity 不是主因。39 个误伤中 36 个来自 source records 全票一致，source record 中位数为 6，范围
2–30；代表性误例包含约 `36.7–37%`、约 `70%` 和接近 `100%` 的实验结果。旧 full-Starling direct 只修复
39 个误伤中的 15 个，并且其全体预测同样严重偏 low（125/209），所以更近、带定量/context 的 records 只
部分缓解，根本偏差早已存在于 Bio 的 task calibration 与 evidence adjudication。

冻结诊断为：任务实际判断 `F>=20%`，但 single/final 正在更像判断“是否具有优秀 drug-like exposure”；
MW、logP、QED、Lipinski、推测的代谢/efflux 风险被过度用于证明 `F<20%`。2026-08-09 已将唯一简单候选实现为
versioned Bio prompt profile `f20_evidence_calibrated_v2`：
**低 transferability 只能降低 evidence weight，不能把正类方向改成 neutral/negative；预测 low 必须有明确的
F<20 支持，mixed/insufficient 不得默认 low。** 先在 matched valid 上运行，再检查 full-pool direct 是否同向
改善；不增加 router，不做多候选 ablation，formal test 继续不运行。历史 prompt 冻结为
`legacy_bioavailability_v1`，旧 manifest 自动归入 legacy，跨 profile branch reuse 被拒绝。

2026-08-09 的 GPT-OSS-120B scaffold-valid fresh run 已完成，所有 condition 均为 209/209 成功、0 failed，
`identity_blind + parent_disjoint`，并使用同一份 v2 single analysis。除 `none` 外，legacy/v2 对应 condition 的
`retrieval.json` 逐样本 SHA mismatch 均为 0，因此 paired difference 只来自 prompt contract：

| condition | legacy macro-F1 | v2 macro-F1 | paired Δ (95% bootstrap CI) | legacy/v2 accuracy | v2 Y=0/Y=1 recall |
|---|---:|---:|---:|---:|---:|
| none | 0.4479 | 0.4129 | -0.0350 [-0.1102,+0.0423] | 0.4498 / 0.7033 | 0.0000 / 0.9735 |
| matched train-label direct | 0.5261 | 0.6140 | +0.0879 [-0.0109,+0.1894] | 0.5359 / 0.7703 | 0.2414 / 0.9735 |
| full-Starling direct | 0.5577 | 0.6375 | +0.0798 [-0.0026,+0.1618] | 0.5646 / 0.7273 | 0.4138 / 0.8477 |
| full-flat | 0.6117 | **0.7004** | **+0.0887 [+0.0163,+0.1606]** | 0.6220 / 0.7560 | 0.5862 / 0.8212 |
| full-mechanism | 0.5869 | **0.6844** | **+0.0975 [+0.0230,+0.1694]** | 0.5933 / 0.7416 | 0.5690 / 0.8079 |

这支持原诊断：旧 prompt 的主要问题确实是把 uncertainty/low transferability 系统性转成 `low`。最直接的
证据是 matched 三邻居全为 Y=1 时，agent 的 `1→0` 改票从 48 次降到 0 次。v2 `none` 几乎全预测 high，
说明修复不能作为无证据分类器单独使用；但在 full-flat/full-mechanism 中，实验 evidence 恢复了 Y=0 recall，
同时两项 macro-F1 paired CI 均高于 0。当前 valid gate 支持保留 v2 为 Bio 默认；该设置随后在
2026-08-10 进入一次性 Bio formal-test，见下方独立小节。

```text
tools/chembl_tool/tasks/bioavailability_ma/prompt_profiles.py
tools/chembl_tool/paper_experiments/matched_train_label_agent/diagnose_bio_unanimous_positive.py
outputs/paper/matched_train_label_direct_agent_v1_scaffold_valid_gpt_oss_120b/analysis/bio_unanimous_positive_diagnosis/
outputs/paper/matched_train_label_direct_agent_f20_calibrated_v2_scaffold_valid_gpt_oss_120b/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_f20_calibrated_v2/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_f20_calibrated_v2/analysis/bio_prompt_profile_comparison.json
```

#### Current latest cross-task figure

为直观看 current valid 结果，canonical `plot_starling_model_comparison.py` 现以 single-series 模式汇总三个
task 各自最新且 lineage-matched 的条件：BBB 使用 `experimental_meaningful_cns_access_v2`，Bio 使用
`record_supported_v2 + f20_evidence_calibrated_v2`，Skin 使用
`record_supported_v2 + sensitization_aligned_v2`。每个 panel 的 MiniMol head、Morgan KNN 与 MiniMol cosine-KNN
均来自该 task 自己的 train split；不同 task 仍是不同 dataset，不能把 bar 当作逐样本跨 task pairing。

```text
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/analysis/best_agent_paired_all_baselines.tsv
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_gpt_oss_120b/figures/starling_model_comparison_highres.png
```

当前图已增加 best valid-selected agent 相对三个 train-derived baselines 的 one-sided paired permutation raw
p-value（100,000 permutations）：

| task | MiniMol train-all | Morgan KNN | MiniMol embedding KNN |
|---|---:|---:|---:|
| BBB | 0.376 | 0.015 | 0.027 |
| Bioavailability | 0.309 | 0.020 | 0.061 |
| Skin Reaction | 0.499 | 0.117 | 0.498 |

全部比较均在各 task 自己的同一批 scaffold-valid molecules 上严格配对；不同 task 之间不配对。由于 best agent
和单侧方向都在观察 valid 后确定，这些 raw p-value 只作 exploratory annotation。跨全部 9 项比较做 Holm
校正后没有一项低于 0.05。这些 p-value 仍只描述 valid；2026-08-10 已完成 Bio selected-condition
formal-test，BBB/Skin 尚未运行。

同一唯一总图入口的 current-valid **multi-model** 版本现改为选择每个 task 的最高 DeepSeek-V4-Pro condition，
并在每个 panel 下方对全部三个 train-label baseline 显示 100,000 次单侧 paired-permutation raw p-value：

| task | best DeepSeek condition (macro-F1) | MiniMol train-all | Morgan KNN | MiniMol embedding KNN |
|---|---|---:|---:|---:|
| BBB | full-mechanism (0.7184) | 0.0295 | 0.00046 | 0.00037 |
| Bioavailability | full-flat (0.7346) | 0.1178 | 0.00361 | 0.00781 |
| Skin Reaction | canonical direct (0.6410) | 0.0586 | 0.00285 | 0.0606 |

图中小于 0.001 的值显示为 `p < 0.001`；机器可读 TSV/JSON 保留完整数值、10,000 次 paired-bootstrap CI 和
跨 9 项 Holm 校正。Holm 后 BBB 对 Morgan/MiniMol KNN、Bio 对 Morgan/MiniMol KNN、Skin 对 Morgan 低于
0.05；与 MiniMol train-all 的三项比较以及 Skin 对 MiniMol KNN 均不低于 0.05。由于 DeepSeek condition 和
方向仍由同一 valid set 选择，这些不能解释为 confirmatory test。

```text
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_multimodel/experiment_metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_multimodel/analysis/best_deepseek_agent_paired_all_baselines.{tsv,json,md}
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_multimodel/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_current_latest_valid_multimodel/figures/starling_model_comparison_highres.png
```

2026-08-10 已在完全相同的 current GPT-OSS-120B、scaffold-valid、identity-blind、parent-disjoint 和 task prompt
合同下补齐 ChEMBL direct/full-flat/full-mechanism。9 个 batch 共 `2460/2460` final 完整，`n_failed_runs=0`，
parent-policy conflict 与 below-threshold retained neighbor 均为 0：

| task | ChEMBL direct | ChEMBL full-flat | ChEMBL full-mechanism | best ChEMBL | best Starling |
|---|---:|---:|---:|---:|---:|
| BBB | 0.5354 | 0.5756 | 0.5756 | 0.5756 | 0.6690 |
| Bioavailability | 0.5552 | 0.6098 | 0.5666 | 0.6098 | 0.7004 |
| Skin Reaction | 0.5123 | 0.5471 | 0.5677 | 0.5677 | 0.5725 |

对应 artifact root 为：

```text
outputs/paper/molecular_evidence_agent_chembl_scaffold_current_latest_valid_gpt_oss_120b/
```

这里的 scaffold split 约束 benchmark train/valid/test，不约束外部 ChEMBL corpus。当前 identity policy 不排除
同 scaffold analog。逐 retrieval audit 中，同非空 scaffold neighbor 占比为：BBB direct/flat/mechanism
`8.37%/7.54%/8.87%`，Bio `15.71%/13.80%/14.99%`，Skin `6.12%/5.01%/4.93%`；涉及至少一个
同-scaffold neighbor 的 query 比例分别为 BBB `9.02%/23.22%/23.22%`、Bio
`22.97%/44.02%/44.02%`、Skin `1.22%/5.71%/5.71%`。因此 ChEMBL bar 表示实际 external-evidence
RAG 能力，不能称为与 train-only KNN 完全 matched 的 retrieval comparison。若研究这部分增益，应使用独立
`scaffold_disjoint_external` ablation，不得覆盖本轮 canonical artifact。

#### External scaffold-disjoint diagnostic archive（2026-08-10；overbroad setting）

最初运行把 `scaffold_disjoint` 错误地应用到了 ChEMBL/Starling 的 direct/full-flat/full-mechanism 全矩阵；这不是
计划中的正式 comparison setting。计划合同已更正为：**只有 ChEMBL direct 使用 `scaffold_disjoint`**，Starling
direct 以及两种 source 的 full-flat/full-mechanism 均继续使用 canonical `parent_disjoint`。因此下面 18-batch
结果只保留为诊断性 archive，不进入当前最佳版本、正式 source comparison 或主图。

通用 `scaffold_disjoint` identity policy 本身实现正确：它是 `parent_disjoint` 的严格超集，
额外排除标准化 parent 的**非空** Bemis–Murcko scaffold 相同的候选；无环分子的空 scaffold 不互相排除。
该 overbroad run 中 ChEMBL 与 Starling 使用完全相同的 policy、Morgan/top-3/threshold、query、single analysis、
task prompt、GPT-OSS-120B 和 identity-blind 设置，只改变 retrieval source。结果写入独立 root，未覆盖上一节
canonical runs：

```text
outputs/paper/molecular_evidence_agent_scaffold_disjoint_source_matched_current_valid_gpt_oss_120b/
  runs_identity_blind_scaffold_disjoint/
  analysis/{summary.json,experiment_summary.tsv,paired_comparisons.tsv,report.md}
```

18 个 batch 共 `4920/4920` final 完整且 `n_failed_runs=0`。artifact-level audit 对 23,902 个实际保留 neighbor
重新执行 identity policy，`retrieval_policy_conflicts=0`；identity-blind prompt leak 也为 0。macro-F1 与同 mode
的 Starling-minus-ChEMBL paired delta 如下：

| task | mode | ChEMBL | Starling | delta | paired-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| BBB | direct | 0.5540 | 0.6318 | +0.0778 | [+0.0343,+0.1209] |
| BBB | full-flat | 0.5706 | 0.6466 | +0.0761 | [+0.0324,+0.1204] |
| BBB | full-mechanism | 0.5691 | 0.6308 | +0.0617 | [+0.0172,+0.1064] |
| Bioavailability | direct | 0.5396 | 0.5943 | +0.0548 | [-0.0433,+0.1524] |
| Bioavailability | full-flat | 0.5538 | 0.5630 | +0.0093 | [-0.0919,+0.1111] |
| Bioavailability | full-mechanism | 0.5541 | 0.6136 | +0.0595 | [-0.0375,+0.1546] |
| Skin Reaction | direct | 0.5027 | 0.5633 | +0.0606 | [-0.0198,+0.1401] |
| Skin Reaction | full-flat | 0.5247 | 0.5545 | +0.0299 | [-0.0555,+0.1136] |
| Skin Reaction | full-mechanism | 0.5434 | 0.5518 | +0.0084 | [-0.0690,+0.0849] |

这些数字只能描述 overbroad diagnostic root，不能回答更正后的“ChEMBL direct 去同 scaffold”问题，也不能外推为
“Starling corpus 普遍更好”；两类 library 的 assay scope、curation 与 coverage 仍不同。

与上一节 parent-disjoint fresh runs 的 before/after 只作敏感性诊断，不能把全部变化因果归于 scaffold filter。
ChEMBL 的 LLM-visible retrieval 在 BBB direct/full、Bio direct/full、Skin direct/full 分别改变
`33/85`、`48/92`、`3/14` 个 query；Starling 分别改变 `149/160`、`65/101`、`39/85`。
但 fresh generation 即使 retrieval hash 不变也会翻转预测：例如 Skin ChEMBL direct/flat/mechanism 的全部
`71/83/70` 次翻转中有 `70/79/67` 次发生在 retrieval 未变的 query。Bio Starling full-flat 从 0.7004 降至
0.5630（delta -0.1374，paired CI [-0.2166,-0.0583]），但其 47 次翻转中也有 26 次 retrieval 未变；所以不能把
整段下降解释成排除同 scaffold 的净效应。

为直接量化 fresh-generation 波动，随后在完全相同的 Bio Starling full-flat `parent_disjoint` 输入上累计运行5次。
全部 run 均为209/209成功、0 failed；相对 original 的836个 retrieval JSON、824个实际 group prompt 和836个
final system prompt 比较均完全一致，3个无 neighbor 样本每次都跳过 group。五次 macro-F1 为
`0.7004/0.6098/0.6560/0.6340/0.6216`，均值 `0.6444`、run-level sample SD `0.0357`、min-max
`0.6098–0.7004`；accuracy 均值 `0.7053`、sample SD `0.0329`。原始 `0.7004` 是5次最高值，后续4次均未复现。

逐样本看，140/209 在5次中预测完全一致，36个为4:1 split，33个为3:2 split，即69/209（33.0%）至少改票一次；
任意两轮平均有34.2/209个预测不同（范围30–37）。206个有 group 的样本中94个（45.6%）的结构化
`evidence_direction` 至少变化一次。五轮 majority vote 的 macro-F1 为 `0.6643`，仅作稳定性描述，不注册为新方法。
这些结果说明一次 fresh two-stage run 的波动足以达到约9 macro-F1 points，单次 `0.7004` 只能作为 nominal
single-run best；原 overbroad run 的 before/after 不能用来估计 scaffold filter 的净效应。repeat artifacts 保存在：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_f20_calibrated_v2_full_flat_repeat{1,2,3,4}/
```

#### Bioavailability formal scaffold-test（2026-08-10；首次且唯一一次）

在 prompt、data lineage、retrieval 和 baseline 超参数全部冻结后，首次查看
`record_supported_v2/Bioavailability_Ma/scaffold/test.jsonl`。test 共209条，SHA256 为
`445e1d8bf0635690e1974872116ba5a388f11469e1e793306cc1208f4e9a5b4e`。两个 agent condition 均使用
GPT-OSS-120B、`identity_blind + parent_disjoint`、Morgan retrieval、standard context 和冻结的
`f20_evidence_calibrated_v2`；full-flat/full-mechanism 均为209/209成功、0 failed。三个 baseline 使用同一
1674条 train：Morgan `k=3` unweighted vote、MiniMol embedding cosine `k=3` unweighted vote，以及冻结的
MiniMol train-all 5-member head（25 epochs、fixed threshold 0.5）。五个 prediction artifact 与 test 的 SMILES、
顺序和 label 均逐条完全对齐。MiniMol source commit 为
`41642328affbb35e4950b7b7a9020aa95919b503`，`state_dict.pth` SHA256 为
`a3061e1cdbbd5e544168fbf8c1e01836c8c19ac030891d495a2ae9f29cfe97a7`；完整 config hash 写入机器可读 summary。

| method | macro-F1 | accuracy | AUROC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|
| Starling full-flat agent | 0.6663 | 0.7177 | - | 34/24/35/116 |
| Starling full-mechanism agent | **0.6720** | 0.7177 | - | 36/22/37/114 |
| Morgan KNN | 0.6801 | 0.7656 | 0.7348 | 26/32/17/134 |
| MiniMol embedding KNN | 0.6484 | 0.7512 | 0.7177 | 22/36/16/135 |
| MiniMol trained head | **0.7027** | **0.7943** | **0.7807** | 25/33/10/141 |

下面的 macro-F1 CI 是10,000次 paired bootstrap；`p` 是同一逐样本配对上的 two-sided exact McNemar
accuracy test。两种 agent 组织方式与三个 baseline 的差异均未达到明确统计区分；full-mechanism 相对
full-flat 也只是 `+0.0057 [-0.0569,+0.0701]`，36次 prediction flip 中双方各自救回18条，`p=1.0000`。

| comparison（agent minus baseline） | macro-F1 delta [95% CI] | exact p |
|---|---:|---:|
| full-flat minus Morgan KNN | -0.0138 [-0.1075,+0.0792] | 0.2678 |
| full-flat minus MiniMol embedding KNN | +0.0179 [-0.0738,+0.1112] | 0.4500 |
| full-flat minus MiniMol trained head | -0.0363 [-0.1261,+0.0541] | 0.0519 |
| full-mechanism minus Morgan KNN | -0.0081 [-0.0957,+0.0827] | 0.2678 |
| full-mechanism minus MiniMol embedding KNN | +0.0236 [-0.0603,+0.1106] | 0.4500 |
| full-mechanism minus MiniMol trained head | -0.0306 [-0.1212,+0.0588] | 0.0519 |

因此 test 支持的结论比 nominal valid best 更保守：Bio evidence-calibrated agent 与强 baselines 在当前209条
test 上大致同档，未证明稳定超过 Morgan 或 MiniMol head；它也没有退回旧 prompt 的严重低类偏置。full-flat
test `0.6663` 位于五次 matched valid repeat 的 `0.6098–0.7004` 范围内。test 只运行了冻结的 full-flat、
full-mechanism 和对应 baselines；不得根据该 test 继续选择 prompt、retriever 或阈值。artifact 为：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/
  experiment_matrix_identity_blind_parent_disjoint_bioavailability_ma_774b17aab9c5.json
  analysis/bioavailability_formal_test/summary.json
  runs_identity_blind_parent_disjoint/bioavailability_ma/
    bioavailability_ma__starling_full_flat/
    bioavailability_ma__starling_full_mechanism/
outputs/baselines/structure_knn_starling_record_supported_v2_test/Bioavailability_Ma/scaffold/
outputs/baselines/minimol_embedding_knn_starling_record_supported_v2_test/Bioavailability_Ma/scaffold/
outputs/baselines/minimol_starling_record_supported_v2_test/Bioavailability_Ma/scaffold/
```

### Full-mechanism failure-cause audit（BBB historical；Bio/Skin legacy baseline 与 bugfix 对照）

已有 GPT-OSS-120B scaffold-valid traces 的 deterministic audit 将错误按观察到的最早失败位置互斥分解，
不新增模型调用。BBB/Bioavailability/Skin 的错误数为 150/89/90；其中 final aggregation 仅 5/2/4，绝大多数
落在 retrieval proxy wrong/inconclusive、group 丢失 gold-aligned proxy 或 mechanism conflict。Full-flat 到
full-mechanism flips 为 BBB 62（31 gain/31 harm）、Bio 41（20/21）、Skin 33（21/12），说明 mechanism
organization 本身没有稳定单向收益。Bio group outputs 有 23 个非 canonical direction、Skin 有 1 个；这推动了
shared single/group allowed-value validation，但 format 修复本身不是性能方法。

这里的 `bugfix-v2` 指 2026-08-08 的 shared validation/Skin contract matrix；其中 Bio 仍使用
`legacy_bioavailability_v1`，不能与上一节后续的 `f20_evidence_calibrated_v2` 混称为当前 Bio prompt。
该 run 中 Bio/Skin 的非 canonical group direction 均为 0；Bio 有 15/839、Skin 有 9/818 个
group output 在同 setting retry 后合法，最终 invalid 为 0。Bio full-mechanism 的 85 个错误分解为 direct
retrieval wrong 25、inconclusive 19、direct group 丢失 gold proxy 38、mechanism conflict 2、final aggregation
1；Skin 则为 42/30/25/0/1。Bio 的 direct→full-flat 是本轮唯一明确的额外-family净增益：macro-F1
`+0.0540`（`[+0.0134,+0.0977]`），而 full-flat→mechanism 为 `-0.0248`
（`[-0.0742,+0.0239]`）。这说明额外证据有时有效，但按 mechanism 分支组织本身仍没有带来增益。
Bio 的 none/direct/full-flat/full-mechanism accuracy 为 `0.4498/0.5646/0.6220/0.5933`，macro-F1 为
`0.4479/0.5577/0.6117/0.5869`；全部 209/209 成功。

BBB 这部分 audit 基于已经撤下主线的 heterogeneous old gold，因此只能解释旧 trace，不能用于估计新
`experimental_meaningful_cns_access_v2` 的瓶颈。Direct-neighbor vote 也只是描述性 proxy，不能证明 analog transferability。

```text
tools/chembl_tool/paper_experiments/audit_starling_trace_failure_causes.py
outputs/paper/starling_trace_failure_cause_audit_record_supported_v2_valid_gpt_oss_120b/
outputs/paper/starling_trace_failure_cause_audit_record_supported_v2_valid_gpt_oss_120b_bugfix_v2/
```

曾生成的 exploratory `record_supported_v1` 因 held-out 分布不符合最终设计，数据、indices、agent runs、
baselines 和显著性 artifact 已于 2026-08-07 删除；这里只保留这条 lineage tombstone，不再引用旧路径。

## 第一版 record-agreement benchmark（historical comparison）

上一版 strict-conflict lineage 的机器可读结果和 canonical bar chart：

```text
outputs/paper/starling_benchmark_results/summary.json
outputs/paper/starling_benchmark_results/metrics.tsv
outputs/paper/starling_benchmark_results/report.md
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results/figures/starling_benchmark_overview_highres.png
```

当前 70% record-agreement v4 的 GPT-OSS-20B identity-blind scaffold-valid 与 train-label baseline
对比单独保存在：

```text
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/report.md
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/figures/starling_benchmark_overview_highres.png
```

这一轮只评估 scaffold `valid`，不使用 test。MiniMol head 使用 scaffold train 全量训练、5-member
ensemble、25 epochs 和固定 0.5 threshold；两种 KNN 都只使用 scaffold train labels、`k=3` 和未加权多数票。
结果如下（macro-F1）：

| task | best GPT-OSS-20B condition | best GPT-OSS | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---|---:|---:|---:|---:|
| BBB_Martins | Starling Full / Mechanism | 0.6934 | 0.7108 | **0.7127** | 0.7007 |
| Skin_Reaction | Starling Full / Mechanism | 0.5585 | 0.5900 | 0.6064 | **0.6166** |
| Bioavailability_Ma | ChEMBL Full / Flat | 0.6443 | **0.6577** | 0.6199 | 0.5757 |

GPT-OSS 共 22 个 condition、6,887 个 sample-condition；其中
`Bioavailability_Ma / chembl_full_flat / idx00076` 因输入超过 131,072 context limit 失败，未补跑。
汇总按预先明确的 `count_as_incorrect_opposite_label` policy 将该样本计错：该 condition 的
failure-inclusive confusion matrix 为 TN=40、FP=21、FN=47、TP=101，macro-F1 从只统计成功样本的
0.6476 调整为 0.6443。其余 6,886 个 sample-condition 成功。

同一 frozen scaffold-valid contract 的 GPT-OSS-120B 结果独立保存在：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/report.md
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/figures/starling_benchmark_overview.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/figures/starling_benchmark_overview_highres.png
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

120B 运行只更换 served model；仍使用 scaffold `valid`、`identity_blind + parent_disjoint`、同一 held-out
retrieval index、同一 prompt/tool contract，并复用上表同一批 train-label baselines。launcher 使用 4 个
TP=2 vLLM backends，经 HAProxy 统一暴露，执行形状为 4 个 condition workers x 每 condition 64 requests，
全局 effective concurrency 为 256。最佳 macro-F1 与 20B 对照如下：

| task | best GPT-OSS-120B condition | 120B | best 20B | delta | strongest baseline |
|---|---|---:|---:|---:|---:|
| BBB_Martins | Starling Direct | **0.7138** | 0.6934 | +0.0203 | Morgan KNN 0.7127 |
| Skin_Reaction | Starling Full / Mechanism | 0.5700 | 0.5585 | +0.0115 | MiniMol KNN 0.6166 |
| Bioavailability_Ma | ChEMBL Full / Flat | **0.6498** | 0.6443 | +0.0055 | MiniMol head 0.6577 |

120B 同样完成 22 个 condition、6,887 个 sample-condition。初次运行的 Bioavailability flat branches
包含 49 个 request timeout（`chembl_full_flat` 27 个、`starling_full_flat` 22 个）和一个确定性的 context
failure：`chembl_full_flat/idx00076` 输入 140,665 tokens，超过 131,072 上限。49 个 timeout 使用 600 秒
timeout、最多 50 个实际并发请求全部成功补齐；旧失败目录和日志保存在同一 task root 下的
`retry_archive_20260802_timeout300/`。最终只有 idx00076 按
`count_as_incorrect_opposite_label` policy 计错，6,886 个 sample-condition 成功；重汇总后
`chembl_full_flat` 与 `starling_full_flat` 的正式 macro-F1 分别为 0.6498 和 0.6346。

### GLM-5.2 NVFP4 identity-blind scaffold-valid（2026-08-03 完成）

GLM 在同一 frozen scaffold `valid`、同一 held-out-filtered index 和
`identity_blind + parent_disjoint` contract 上完成 22 个 condition、6,887 个 sample-condition，严格完整性
检查为 `6887/6887`，没有使用 failure-inclusive 计错。机器可读结果位于：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_glm_5_2_nvfp4/
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/report.md
```

最佳 agent macro-F1 为：

| task | best GLM condition | GLM | best GPT-OSS-120B blind | delta |
|---|---|---:|---:|---:|
| BBB_Martins | Starling Full / Flat | **0.7337** | 0.7138 | +0.0199 |
| Skin_Reaction | Starling Direct | **0.6046** | 0.5700 | +0.0346 |
| Bioavailability_Ma | Starling Full / Mechanism | 0.6467 | **0.6498** | -0.0031 |

逐 trace 审计覆盖 22 条件/6,887 样本：query-SMILES trace leak 为 0，prompt structure/identifier leak 为 0。
最初报告的 3 个 name leak 均来自 Starling source name `PER` 与英文介词 `per` 的词边界碰撞；`PER` 已加入
generic-name audit allowlist 后重新审计为 0。全量 28,298 个 retained neighbor slots 均标记为
`structural_analog`，parent-policy conflict 和低于 0.30 threshold 的补位均为 0；对应 held-out index metadata
记录 `zero_parent_overlap=true` 和 residual held-out parent 为 0。

注意 canonical matrix manifest 会被最后一次 selection/repair launcher 原子更新，因此该 root 当前 manifest
显示最后的 `1x1` finalization，而不是整轮历史峰值并发。完整启动/修复形状保存在同 root 的
`run_*` / `repair_*` logs；不要仅凭最后一个 manifest 反推整轮吞吐。

### GPT-OSS deployment-visible + parent-disjoint 补充矩阵（2026-08-03）

两个 GPT-OSS 模型随后在同一 frozen scaffold `valid` 和同一 held-out-filtered retrieval index 上补齐
`deployment_visible + parent_disjoint`。这里的 visible 同时表示 query/neighbor identity 对 LLM 可见，以及
comparison tools 由模型通过 function call 执行；它不是只改变 SMILES 脱敏的单因素消融。结果分别保存在：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_20b_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_visible_parent_disjoint/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b_visible_parent_disjoint/
```

每个模型均完成 22 个 condition、6,887 个 sample-condition。两边唯一最终失败都是
`Bioavailability_Ma / chembl_full_flat / idx00076`：visible prompt 为 151,630 tokens，超过两个 served
model 的 131,072 context limit。该样本无法在不改变冻结 evidence/prompt contract 的情况下修复，故按
`count_as_incorrect_opposite_label` 计错；每个模型其余 6,886 个 sample-condition 成功。两套矩阵的 19 个
retrieval conditions 全量审计均为 parent-policy conflict=0、below-similarity-threshold=0，manifest 均为
`fresh_parent_disjoint=true`、`operational_staging_used=false`。

20B 首轮有 177 个失败 run（失败 group 合计为 177 个 Harmony tool-header parser error、1 个 request
timeout 和上述 1 个 context error；少数 run 同时包含多个失败 group）。先以 32/16/8 的递减全局并发补跑，
最后对单个长尾使用并发 1；所有可重试失败均成功。各轮原始失败 run、日志和 metrics 保存在该 model root
下的 `retry_archive_20260803_*`。120B 首轮只有上述确定性 context failure。20B 服务最终使用 node001 上
8 个单-GPU backend 和 HAProxy；120B 使用 node002 上常驻的 4 个 TP=2 backend 和 HAProxy。

visible 主矩阵的最佳 macro-F1 为：

| task | best 20B visible condition | 20B visible | best 120B visible condition | 120B visible |
|---|---|---:|---|---:|
| BBB_Martins | Starling Direct | 0.6751 | Starling Full / Flat | **0.6926** |
| Skin_Reaction | Starling Full / Mechanism | 0.5525 | Starling Full / Flat | **0.5913** |
| Bioavailability_Ma | Starling Full / Flat | 0.6545 | ChEMBL Full / Mechanism | **0.6703** |

这些数值与 identity-blind 主矩阵共享数据、retrieval identity policy、prompt schema 和模型权重，但 visible
合同还改变了结构可见性与 tool execution，因此图中的 blind/visible 差异应解释为完整部署合同差异，不能只
归因于分子 identity visibility。

GLM 的同合同 `deployment_visible + parent_disjoint` scaffold-valid 矩阵已于 2026-08-04 完成，输出到：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_glm_5_2_nvfp4_visible_parent_disjoint/
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/metrics.tsv
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/summary.json
outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/report.md
```

该运行最初使用静态 condition lanes；在 `6765/6887` 完整样本处安全停止旧 launcher，并迁移到唯一的
`global_prompt_ready_pool.v1`。首次恢复扫描直接还原 6765 个完整样本，把跨 task/condition 的 144 个缺失
group stage 放进同一 128-slot pool；46 分钟后完整样本增至 6879，余下 8 个极端长请求随后以相同模型、prompt、
reasoning 和 validation 合同、仅提高单请求 timeout 完成。迁移审查还发现 staged final 必须保持旧 pipeline 的
`group_id` 排序；受影响的 38 个 Bioavailability mechanism final/trace 已先按 SHA-256 归档，再只重建 final。
归档及审计位于 `scheduler_migration_audit/final_prompt_order_pre_fix_20260804/`。

最终 gate 为 22 个 batch、6887 predictions、6887 run directories、`sum(n_failed_runs)=0`，逐 run 的
task-specific prediction、single、expected group、final 和 trace 完整性错误均为 0；38 个重建 final 的实际 prompt
顺序和归档 hash 也全部通过。旧 static/condition-lane scheduler、`--condition-workers`、batch
`--group-workers` 和 scheduler 选择开关已从正式与 generic paper runner 删除；batch CLI、Starling matrix 和
MiniMol orchestrator 统一复用 global ready pool，不再保留可绕过全局预算的 endpoint fan-out 路径。

GLM visible 汇总已加入同一 canonical blind+visible 总图。它不是 visible-only 小图：图中同时保留
GPT-OSS-20B、GPT-OSS-120B 和 GLM-5.2 NVFP4 的 blind/visible 六套完整 series，覆盖三个 task 的全部
22 个 agent conditions，并保留共享 train-label baselines 和 matched opt-in experiments。各 task 的最佳
GLM visible macro-F1 为：BBB `starling_full_flat` 0.7096、Skin `starling_direct` 0.6208、Bioavailability
`starling_full_mechanism` 0.6832；相对 GLM blind 的同 task 最佳值分别为 -0.0242、+0.0162、+0.0365。

blind/visible 全条件总图入口为：

```bash
python -m tools.chembl_tool.paper_experiments.plot_starling_model_comparison \
  --reference-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b/metrics.tsv \
  --candidate-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_visible_parent_disjoint/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_120b_visible_parent_disjoint/metrics.tsv \
  --comparison-metrics outputs/paper/starling_benchmark_results_scaffold_valid_glm_5_2_nvfp4_visible_parent_disjoint/metrics.tsv \
  --experiment-metrics outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/three_way_metrics.tsv \
  --experiment-metrics outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/figure_metrics.tsv \
  --paired-ci-metrics outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/analysis/best_agent_paired_baseline_bootstrap_ci.tsv \
  --paired-significance-display pvalue \
  --output outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg \
  --png-output outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

这是 Starling model、visibility、baseline 和 matched ablation 的唯一正式总图。完整 model/visibility
summary 通过重复 `--comparison-metrics` 追加；matched experiment 通过重复 `--experiment-metrics` 追加，
不再生成单实验 overview/bar chart。experiment TSV 中的 anchor 必须与 candidate summary 的既有 condition
在 subset、样本数和 macro-F1 上一致，绘图时只用于合同校验、不重复显示。
总图现在通过 `--paired-significance-display pvalue` 在每个 task panel 内只显示 best agent
相对 MiniMol train-all 和 Morgan KNN 的单侧 paired-permutation p-value，检验方向为
`H1: best agent > baseline`，图内不再显示 95% CI。六个检验均未达 `p < 0.05`；且因为
单侧方向和 best agent 均是看到同一 scaffold-valid 结果后确定，该 p-value 展示明确标为
exploratory，不写成预注册 confirmatory 检验。

本轮 baseline 输出根：

```text
outputs/baselines/minimol_starling_valid/<Task>/scaffold/
outputs/baselines/structure_knn_starling_valid/<Task>/scaffold/
outputs/baselines/minimol_embedding_knn_starling_valid/<Task>/scaffold/
```

## 1. 第一版冻结的数据与 label 决策

公共协议和唯一数据构建入口：

```text
tools/chembl_tool/common/starling/STARLING_BENCHMARK_PROTOCOL.md
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
```

当前冻结 v4 规则：

- parent agreement 为 record-weighted `max(n0,n1)/(n0+n1)`，threshold 为 70%，精确 tie 拒绝；
- valid 与 test size 分别为 `min(500, floor(0.1 * n_binary_molecules))`；
- `random` 是固定 seed 的 label-stratified stable-hash split；
- `scaffold` 以 canonical Bemis–Murcko scaffold 为不可拆分 group；
- 两套 split 从同一批 accepted binary parents 独立构建；
- 每套 train/valid/test parent identity 两两零重叠，scaffold 还要求 scaffold 两两零重叠；
- source row 先按 task adapter 独立转成 `0/1/ambiguous`，再按
  `rdkit_fragment_parent.v1` 聚合；
- 同一个 normalized parent 的 accepted rows 同时出现 0 和 1 时，达到 70% 就接受多数 label；
- `within_record_label_conflict` 指同一条 row 内可用信号互相矛盾；这种 row 在 parent 聚合前拒绝。

冻结规模：

| task | binary parents | rejected | valid/test | random valid / test Y=0,Y=1 | scaffold valid / test Y=0,Y=1 |
|---|---:|---:|---:|---:|---:|
| BBB_Martins | 19,425 | 1,091 | 500 / 500 | 139,361 / 139,361 | 150,350 / 148,352 |
| Bioavailability_Ma | 2,092 | 106 | 209 / 209 | 58,151 / 58,151 | 61,148 / 65,144 |
| Skin_Reaction | 2,456 | 465 | 245 / 245 | 75,170 / 75,170 | 74,171 / 71,174 |

Task label：

```text
BBB_Martins:
  Y=1 pass；Y=0 fail；明确 logBB 使用 >= -1 threshold。

Bioavailability_Ma:
  human oral F >= 20% -> Y=1；F < 20% -> Y=0。
  2026-08-01 起使用 `bioavailability_canonical_direct.v2`：固定 HF snapshot 与 local 中明确
  absolute/oral-IV rows 合并，跨来源近等值 claim 去重；relative/ambiguous local rows 只留在 residual
  inference evidence。当前 direct claims SHA-256 为
  `045261cbda785092143eeadd636f78399f7b02f951b23480b16fb8dde22661c5`。

Skin_Reaction:
  Y=1 skin sensitizer；Y=0 non-sensitizer。
  irritation、phototoxicity、generic local damage 和 skin exposure 不构成这个 gold label。

ClinTox:
  不从 Starling toxicity rows 构造 gold split。独立 `clinical_trial_failure_v1` 从冻结
  AACT toxicity-failure positives 与 SWEETLEAD/FDA-approved comparators 重建 source-defined
  parent labels；clinical/mechanistic Starling rows 只用于 retrieval。该 lineage 的 DeepSeek v3
  结果见 `tools/chembl_tool/tasks/clintox/CLINTOX_BENCHMARK.md`，不得混入本表三项
  Starling gold 汇总。
```

此前所有 17,893/1,828/1,900-parent strict-conflict benchmark 的 baseline、agent 和图表，以及更早的
Bioavailability 1,862-parent mixed-source 结果，均为 historical lineage，不得与当前 lineage 混表。
第一版 record-agreement agent roots 为：

```text
outputs/paper/molecular_evidence_agent_starling_random_record_agreement70_split811_v1/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1/
```

当前 record-supported v2 的 canonical index root 和 model-specific valid roots 见第 0 节。

数据与诊断入口：

```text
Bioavailability canonical source:
  tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py

record/PMID provenance distributions:
  tools/chembl_tool/paper_experiments/analyze_starling_parent_provenance.py

50/60/70/80/90% threshold comparison:
  tools/chembl_tool/paper_experiments/analyze_starling_majority_thresholds.py

70% gold split build:
  tools/chembl_tool/common/starling/build_benchmark_datasets.py

valid+test-heldout index build:
  tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py

identity-blind parent-disjoint valid/test matrix:
  tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
```

具体 numerical/unit/free-text 映射以三个 task adapter 为代码真相，汇总规则见
`STARLING_BENCHMARK_PROTOCOL.md`：

```text
tools/chembl_tool/tasks/bbb_martins/starling_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/starling_benchmark.py
tools/chembl_tool/tasks/skin_reaction/starling_benchmark.py
```

## 2. Held-out retrieval 隔离

Gold split 与 retrieval evidence library 是两个步骤。正式 retrieval condition 分别使用对应构造方法的
`heldout_molecule_labels.jsonl` 从 full Starling evidence 中删除全部 valid+test parents，再重建 train-only index；
不能只靠 query-time exact-SMILES exclusion，也不能直接使用 full-source index。

入口：

```text
tools/chembl_tool/common/starling/heldout_index.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
```

当前 v4 formal pipeline condition 使用 `identity_blind + parent_disjoint` fresh-run 制度：

```text
none:
  没有 retrieval，identity policy 标为 not applicable。

retrieval conditions:
  直接从 valid+test-heldout-filtered index 做 fresh parent_disjoint retrieval；
  retained neighbor 与 query parent identity 零重叠，不读 operational reuse plan。
```

2026-08-01 的 random-valid BBB query-only 首次 512 并发压力运行完成 499/500，唯一失败是
single branch transport timeout，不是 structured-output validation 失败。按预定 contingency 将正式
launcher 稍降为 `parallelism=384, group_workers=1`，只重跑该样本后达到 500/500、0 failed。
但 384 并发的 BBB ChEMBL full-flat 又产生 193/500 失败，其中 190 个为 group request timeout；
矩阵随后以较低并发恢复。当前 random-valid 已生成 22 个 condition metrics 和 6,886 个 final artifacts，
严格完整性为 `6885/6887`；剩余两个失败分别位于 Bioavailability `chembl_full_flat` 和
`chembl_full_mechanism`。机器可读 failure-inclusive pipeline summary 位于
`outputs/paper/starling_benchmark_results_random_valid_glm_5_2_nvfp4/`，但该 split 仍未通过 zero-failure
valid gate，也没有匹配的 v4 random-valid train-label baselines，因此不进入 scaffold-valid model comparison
或正式图。当前默认保持单一 `parallelism=128` global prompt pool；完整 valid gate 通过前不启动 test。

## 3. Historical strict-conflict performance

以下数值全部来自上一版 strict-conflict train/test split，不是当前 70%-agreement scaffold benchmark 的结果。
它们仅用于历史复现；第一版 record-agreement 已完成 scaffold-valid 的 GLM、GPT-OSS-20B/120B blind agent matrix、两套
GPT-OSS visible matrix 和 matched MiniMol/Morgan/MiniMol-KNN baselines，结果见本文开头。正式 test 尚未
全部启动；current Bio frozen full-flat/full-mechanism 与三项 baseline 已于2026-08-10运行一次，BBB/Skin
仍未运行。random-valid GLM 保留两个失败，是独立 lineage，不与这里的 scaffold-valid 模型对照混表。
V4 主合同已在 `AGENTS.md` / `EXPERIMENT_PLAN.md` 冻结为
`identity_blind + parent_disjoint` fresh-run、operational staging disabled、endpoint 上限 512/当前
GLM launcher 128；GPT-OSS scaffold-valid 使用独立 model-specific roots 和 4×64 condition lanes。

下表均为完整 test 的 macro-F1。`Starling direct` 对 Bioavailability 指 full direct-F condition；
另有 numeric-only direct-F：random `0.6652`、scaffold `0.6392`。所有表内 formal pipeline、
MiniMol 和 KNN 条件均为 0 failed samples。

| split / task | none | ChEMBL direct | ChEMBL flat | ChEMBL mechanism | Starling direct | Starling flat | Starling mechanism | MiniMol head | Morgan KNN | MiniMol KNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.5992 | 0.6221 | 0.6378 | 0.6420 | 0.7527 | 0.7462 | 0.7422 | **0.7567** | 0.6950 | 0.7314 |
| random / Bioavailability | 0.4677 | 0.6114 | 0.6265 | 0.6152 | 0.6741 | 0.6871 | 0.6991 | 0.7120 | 0.6824 | **0.7256** |
| random / Skin | 0.6223 | 0.6061 | 0.6023 | 0.6303 | **0.6431** | 0.6353 | 0.6300 | 0.6066 | 0.5973 | 0.5905 |
| scaffold / BBB | 0.5788 | 0.6129 | 0.6484 | 0.6439 | 0.6995 | 0.7024 | 0.7005 | **0.7188** | 0.6745 | 0.6946 |
| scaffold / Bioavailability | 0.4529 | 0.6350 | 0.6617 | 0.6712 | 0.6516 | 0.6565 | 0.6743 | 0.6421 | 0.6952 | **0.7028** |
| scaffold / Skin | 0.5884 | 0.5862 | **0.6107** | 0.5918 | 0.5973 | 0.5921 | 0.5836 | 0.6050 | 0.5372 | 0.5892 |

这些是 point estimates，不支持“增加更多 evidence 或 mechanism branches 必然提升”的普遍结论：

- BBB 的 Starling direct/full 明显高于 ChEMBL，但 direct/flat/mechanism 内部差异较小；
- Bioavailability 的 MiniMol embedding KNN 在 random/scaffold 都高于 Morgan KNN 和 MiniMol head；
- Skin 的 Starling direct 到 full mechanism 在两套 split 都下降，见第 8 节；
- random 与 scaffold 是不同 test sets，不能把两者的绝对高低直接解释为方法对 scaffold 的因果效应。

## 4. Historical strict-conflict MiniMol 选择口径

上一版 strict-conflict Starling MiniMol 使用 `--train-all`：

- 使用该 split 的全部 `train.jsonl`；
- 不读取或构造 `valid.jsonl`；
- 每个 ensemble member 固定 epoch；
- test threshold 固定为 `0.5`；
- 不用 test 选择 epoch、hyperparameter 或 threshold；
- 输出中的 validation metrics 为 null。

因此 MiniMol 最终 performance 不是“在 test 上选最好 epoch”，而是冻结训练设置后对 test 评估一次。
入口：

```text
baselines/minimol/run_bioavailability_ma.py --train-all
```

正式产物：

```text
outputs/baselines/minimol_starling/<Task>/<random|scaffold>/
```

## 5. Morgan KNN baseline 的精确定义

入口：

```text
baselines/structure_knn/run.py
```

正式设置：

```text
fingerprint: RDKit Morgan radius=2, 2048 bits, no chirality
candidate pool: 同 split 的 train.jsonl
k: 3
ranking: query-to-train Tanimoto similarity
vote: 三个 neighbor molecules 各一票，未加权多数票
score/AUROC: 三票中的正类比例
```

它不是“top-k records 的 flat vote”。Benchmark builder 已经先把 source records 按 normalized molecule
parent 聚合，冲突 parent 已排除，`train.jsonl` 每个 accepted parent 只有一个 binary label。因此 KNN 的
top 3 是三个 train molecules，每个 molecule 恰好贡献一票；不会让一个有多篇文献/多条 source records
的 molecule 获得更多票。

Bioavailability 的 scaffold KNN macro-F1 `0.6952` 高于 random 的 `0.6824`，但 accuracy 反而更低
（`0.7634` vs `0.7742`），AUROC 几乎相同（`0.7302` vs `0.7295`）。Scaffold test 有更多负类
（113 vs 99），且 negative-class 表现较好，因而 macro-F1 上升。这不是异常，也不能据此声称 scaffold
split 一般比 random 更容易。

正式产物：

```text
outputs/baselines/structure_knn_starling/<Task>/<random|scaffold>/
```

### MiniMol embedding KNN

2026-07-28 新增的 retrieval-only 对照保持 Morgan KNN 的 train pool、`k=3`、未加权多数票和
正类邻居比例 score 不变；只把 ranking feature 从 Morgan fingerprint/Tanimoto 换成正式 MiniMol
baseline 已缓存的 512 维 embedding/L2-normalized cosine similarity。Cache loader 会逐条校验 SMILES、
label 和 row order 与当前 split 完全一致，并在 manifest 中记录 train/test JSONL 与 embedding cache
的 SHA-256。

入口与产物：

```text
baselines/minimol/run_embedding_knn.py
outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold>/
machine-readable method: minimol_embedding_cosine_knn_k3
```

相对 Morgan KNN 的完整 test 结果：

| split / task | macro-F1 | Δ macro-F1 | accuracy | Δ accuracy | AUROC | Δ AUROC |
|---|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.7314 | +0.0363 | 0.8040 | +0.0220 | 0.7693 | +0.0150 |
| scaffold / BBB | 0.6946 | +0.0201 | 0.7740 | +0.0060 | 0.7790 | +0.0606 |
| random / Bioavailability | 0.7256 | +0.0432 | 0.7957 | +0.0215 | 0.7741 | +0.0445 |
| scaffold / Bioavailability | 0.7028 | +0.0076 | 0.7634 | +0.0000 | 0.7301 | -0.0001 |
| random / Skin | 0.5905 | -0.0068 | 0.6526 | +0.0132 | 0.6374 | +0.0196 |
| scaffold / Skin | 0.5892 | +0.0521 | 0.6316 | +0.0368 | 0.6197 | +0.0930 |

这组 point estimates 表明 MiniMol cosine neighborhood 在 5/6 条件上提高 macro-F1，在 6/6 条件上
不降低 accuracy，在 5/6 条件上提高 AUROC；唯一明显不一致的是 Bioavailability scaffold AUROC
几乎持平而略低 `0.0001`，Skin random 则 accuracy/AUROC 提高但 macro-F1 下降 `0.0068`。
这不等于 learned embedding 普遍支配 Morgan：两种 split 的 test 集不同，而且当前只冻结了 `k=3`
和 cosine 一个设置。

该 baseline 已进入 `summarize_starling_benchmark.py` 生成的 `metrics.tsv`、`summary.json`、
`report.md` 和 canonical `starling_benchmark_overview.{svg,png}`，不是只存在于单独实验目录。

## 6. MiniMol embedding agent retrieval（operational + parent-disjoint）

2026-07-30 完成 MiniMol/cosine agent retrieval 的 random/scaffold operational 矩阵。该实验不是
train-label KNN：它保持 evidence source、direct/flat/mechanism organization、top-k、GLM、prompt、
tool execution 和 inference settings 不变，只将 agent 的 neighbor ranking feature 从
Morgan/Tanimoto 换为 L2-normalized MiniMol embedding/cosine。

完整 gate：

```text
conditions: 38/38（random 19，scaffold 19）
sample-conditions: 15,768/15,768 successful
failed: 0
neighbor identity policy: operational
```

失败的 structured-output 分支均按原设置定点重跑；每次失败版本保存在
`outputs/paper/minimol_retrieval_agent_results/failed_attempts/operational/attempt_*/`，没有通过
postprocess 补字段或改写 prediction。

与已有正式 Morgan agent retrieval 的配对 point estimates：

| split / task | paired win / loss / tie | best MiniMol operational condition | MiniMol macro-F1 | best Morgan formal condition | Morgan macro-F1 |
|---|---:|---|---:|---|---:|
| random / BBB | 0 / 5 / 1 | Starling full / flat | 0.7353 | Starling direct | 0.7527 |
| random / Bioavailability | 7 / 0 / 0 | Starling full / flat | 0.7089 | Starling full / mechanism | 0.6991 |
| random / Skin | 2 / 4 / 0 | Starling direct | 0.6267 | Starling direct | 0.6431 |
| scaffold / BBB | 4 / 2 / 0 | Starling direct | 0.7224 | Starling full / flat | 0.7024 |
| scaffold / Bioavailability | 7 / 0 / 0 | Starling full / mechanism | 0.7111 | Starling full / mechanism | 0.6743 |
| scaffold / Skin | 6 / 0 / 0 | Starling full / mechanism | 0.6252 | ChEMBL full / flat | 0.6107 |

38 个 paired retrieval conditions 中 MiniMol operational 为 26 win / 11 loss / 1 tie；平均
`Δ macro-F1 = +0.0111`，中位数 `+0.0098`。最大提高是 scaffold Bioavailability 的
ChEMBL direct（`+0.0549`），最大下降是 random BBB 的 ChEMBL full/mechanism（`-0.0250`）。
Bioavailability 两套 split 的 14/14 paired conditions 都提高；random BBB 则没有提高，
Skin random 也有 4/6 下降。这些异质性不支持“MiniMol feature 普遍支配 Morgan”的结论。

2026-07-30 的 operational-only 配对图只是结果比较，不是纯 retrieval-feature causal attribution：已有 Morgan bars 使用正式
`parent_disjoint` policy，新 MiniMol bars是本轮 `operational` policy，因此同时改变了 feature 和
neighbor identity policy。No-retrieval、MiniMol train-all head、Morgan KNN 和 MiniMol KNN 仅作为同图
context。当时尚不能作 feature-only claim；下述 2026-08-01 matched 结果解决了这项限制。

2026-08-01 已补齐同轮 MiniMol `parent_disjoint`、paired summary 和正式图：

```text
conditions: 38/38（random 19，scaffold 19）
missing conditions: 0
Morgan failed runs: 0
MiniMol failed runs: 0
neighbor identity policy: parent_disjoint（两侧 matched）
bootstrap replicates: 10,000 per condition
```

在 identity policy matched 后，38 个条件仍为 26 win / 11 loss / 1 tie；平均
`Delta macro-F1 = +0.0103`，中位数 `+0.0112`。六个 task/split 的最优条件与上表一致。
10,000 次 paired bootstrap 中有 6 个条件的 95% CI 不跨 0：random BBB ChEMBL full/mechanism
偏向 Morgan；random Bioavailability ChEMBL full/mechanism、scaffold Skin Starling full/mechanism，
以及 scaffold Bioavailability 的 ChEMBL direct、Starling full/flat、Starling full/mechanism 偏向 MiniMol。
这些是 condition-level 未做多重比较校正的结果，仍不支持把平均改善解释成普遍支配。

正式 parent-disjoint 产物：

```text
outputs/paper/minimol_retrieval_agent_results/summary.json
outputs/paper/minimol_retrieval_agent_results/condition_results.tsv
outputs/paper/minimol_retrieval_agent_results/report.md
outputs/paper/minimol_retrieval_agent_results/all_results_comparison.tsv
outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval.svg
outputs/paper/minimol_retrieval_agent_results/figures/minimol_vs_morgan_agent_retrieval_highres.png
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent.svg
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_highres.png
```

其中 `minimol_vs_morgan_agent_retrieval` 是 38 个 feature-only paired 条件；
`starling_benchmark_with_minimol_agent` 将 matched parent-disjoint MiniMol agent bars 与 no-retrieval、
MiniMol train-all head、Morgan KNN、MiniMol KNN 和既有 Morgan agent 结果放在同一张 bar chart。

历史 operational-only 入口与产物仍保留用于 sensitivity audit：

```text
tools/chembl_tool/paper_experiments/run_minimol_retrieval_agent_experiment.py
tools/chembl_tool/paper_experiments/plot_starling_with_minimol_agent.py

outputs/paper/minimol_retrieval_agent_results/operational_all_results_comparison.tsv
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_operational.svg
outputs/paper/minimol_retrieval_agent_results/figures/starling_benchmark_with_minimol_agent_operational_highres.png
```

## 7. Morgan similarity vs query-feature coverage agent retrieval

2026-07-28 完成 coverage selector 的六个 Starling task/split 全样本 matched LLM 对比，全部
`n_failed_runs=0`。两组条件固定使用 Starling full evidence、`full_mechanism`、deployment-visible、
`parent_disjoint`、`top_k_per_group=3`、`min_similarity=0.30`、相同 GLM/single analysis/group reuse 和
decoding；唯一变化是 neighbor selector：按 Morgan/Tanimoto similarity 排序，或在 similarity 不低于
`0.30` 的候选中贪心最大化 query Morgan-bit union coverage。

| split / task | Morgan accuracy | Coverage accuracy | Δ accuracy | Morgan macro-F1 | Coverage macro-F1 | Δ macro-F1 | flips | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random / BBB | 0.7720 | 0.7440 | -0.0280 | 0.7422 | 0.7078 | -0.0344 | 34 | 0.0243 |
| scaffold / BBB | 0.7380 | 0.7200 | -0.0180 | 0.7005 | 0.6862 | -0.0142 | 33 | 0.1628 |
| random / Bioavailability | 0.7231 | 0.7016 | -0.0215 | 0.6991 | 0.6779 | -0.0213 | 28 | 0.1849 |
| scaffold / Bioavailability | 0.6855 | 0.6801 | -0.0054 | 0.6743 | 0.6694 | -0.0049 | 30 | 0.8555 |
| random / Skin | 0.6526 | 0.6447 | -0.0079 | 0.6300 | 0.6200 | -0.0100 | 23 | 0.6776 |
| scaffold / Skin | 0.6211 | 0.6395 | +0.0184 | 0.5836 | 0.6017 | +0.0181 | 25 | 0.2295 |

Coverage 在 6 个条件中有 5 个降低 accuracy 和 macro-F1；唯一上升是 Skin scaffold，但 paired
McNemar 不显著，macro-F1 paired-bootstrap 95% CI `[-0.0058, +0.0426]` 仍跨零。BBB random 的下降
最明确：Morgan-only correct / coverage-only correct 为 `24 / 10`，McNemar `p=0.0243`，macro-F1
delta bootstrap 95% CI `[-0.0612, -0.0094]`。因此当前结果不支持用 pure coverage selector 全局替代
Morgan similarity selector。它仍可作为 hybrid selector 的互补项，但下一版必须同时保留 analog relevance
约束并预先冻结组合规则，不能根据这六个 test 结果调权后再在同一 test 上作确认性结论。

2026-07-31 对全部 retrieval 和 reasoning trace 做了进一步配对审计。`2,504` 个 query 中有
`1,699 (67.9%)` 实际换入至少一个 neighbor，`1,195 (47.7%)` 至少一个 mechanism group 的 top-1
发生变化；跨 query 的平均 replacement 数为 `2.15`，实际变化 query 中为 `3.17`。selector 的确提高了
Morgan query-feature coverage（六条件平均增幅范围 `+0.0357` 至 `+0.0632`）并降低 neighbor-neighbor
Tanimoto（`-0.0977` 至 `-0.1344`），所以负结果不是 selector 没有改变 retrieval。真正发生 set replacement
的样本中 Morgan-only correct / coverage-only correct 为 `85 / 51`，损失集中在换入 evidence 的下游效用。

Trace audit 还发现 `543` 个 query 的 ranked retrieval 完全相同，其中 `27` 个 final prediction 仍发生
flip；这 `27` 个 query 的 final LLM 输入逐字相同，因此属于模型调用波动，不能归因于 selector。当前 group
prompt 会逐 neighbor 提供 whole-molecule Tanimoto、MCS/MMP、property delta 和 assay evidence，但不会把
selector 的 marginal Morgan-bit coverage、query atom/region mapping 或 neighbor 独有覆盖区域传给 LLM。
因此这轮实验验证的是“coverage-selected analog set + 既有 whole-molecule transferability reasoning”，
不能解释为已经完整检验 fragment-wise compositional reasoning。

### 7.1 Coverage-aware reasoning context pilot（2026-08-03 完成）

为直接检验上述输入缺口，新增了与 selector 正交的 `neighbor_context_profile`。`standard` 保持原 prompt；
`coverage_aware` 在不暴露结构/身份的前提下，向每个 mechanism group 增加 query Morgan feature coverage、
由这些 feature 映射得到的 query atom-environment coverage、每个 neighbor 的 marginal/redundant contribution、
累计 coverage 和 marginal connected-region sizes。Prompt 明确要求模型只在 analog 可迁移时组合互补 evidence，
并明确 coverage 不是 fragment causality 或 label vote。

当前 pilot 固定为 Starling scaffold-valid、GPT-OSS-120B、`identity_blind + parent_disjoint`、Starling
`full_mechanism`、`top_k=3`、`min_similarity=0.30` 和 `query_feature_coverage` selector；两边复用同一批冻结的
120B single-molecule analyses，唯一变化是 `standard` vs `coverage_aware` context。控制组和候选组分别写入：

```text
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_coverage_standard/
outputs/paper/molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_valid_gpt_oss_120b_coverage_aware/
```

完整 valid run 两边均为 `954/954` 成功、`n_failed_runs=0`。两边 `954/954` raw retrieval SHA-256
逐 query 相同；threshold violation、非 structural-analog neighbor、query-parent conflict、held-out parent overlap
和 identity-blind prompt leak 均为 `0`。Standard trace 中没有 coverage message；coverage-aware 的
`3,002/3,002` 个有 neighbor 的 group branch 恰好各收到一条 coverage context，其中 `2,998/3,002`
在输出中显式讨论 coverage、complementarity 或 redundancy。由此可确认新输入确实改变了 reasoning pattern，
而不是再次出现“selector 换了 neighbor，但模型不知道为什么”的旧条件。

| task | Standard accuracy | Aware accuracy | Δ accuracy | Standard macro-F1 | Aware macro-F1 | Δ macro-F1 | flips | McNemar p | macro-F1 Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.7040 | 0.7200 | +0.0160 | 0.6796 | 0.6939 | +0.0143 | 54 | 0.3409 | [-0.0140, +0.0428] |
| Bioavailability | 0.6220 | 0.6220 | +0.0000 | 0.6129 | 0.6151 | +0.0022 | 28 | 1.0000 | [-0.0445, +0.0499] |
| Skin Reaction | 0.6000 | 0.5796 | -0.0204 | 0.5633 | 0.5348 | -0.0285 | 35 | 0.4996 | [-0.0809, +0.0220] |

结论是 mixed / no-go for promotion：BBB 有小幅正向 point estimate，Bioavailability 基本不变，Skin 反而下降；
三项 McNemar 均不显著，三个 macro-F1 bootstrap interval 都跨零。因此 coverage-aware context 解决了“LLM
没有使用 coverage 信息”的机制问题，但没有带来跨 task 稳定性能提升，不能替代 standard context，也不据此
进入 Starling test。它保留为 opt-in 插件，后续若继续，应先在 valid 冻结更有针对性的 region representation 或
task-specific evidence transfer gate，再做独立确认。

与同一 GPT-OSS-120B scaffold-valid 的原始 `Morgan selector + standard context` 主线相比，总图新增的
coverage experiment rows 显示：
BBB macro-F1 为 `0.6583 → 0.6796 → 0.6939`，Bioavailability 为
`0.5791 → 0.6129 → 0.6151`，Skin 为 `0.5700 → 0.5633 → 0.5348`（依次为 Morgan-standard、
coverage-standard、coverage-aware）。Morgan-standard vs coverage-aware 的 BBB 增幅为 `+0.0356`，McNemar
`p=0.0265`，macro-F1 bootstrap 95% CI `[+0.0017, +0.0698]`；但 Bioavailability interval 跨零，Skin point
estimate 为负。因此新 prompt 在 BBB 上使 coverage 路线显著超过 Morgan，但不能把这个 task-specific signal
解释成通用方法胜出；核心结论仍是跨 task 不稳定。

代价方面，三 task 合计 group+final total tokens 从 `35,248,740` 增至 `37,667,006`（`+6.86%`）；
同机并发运行下累计 wall latency proxy 增加 `+2.95%`。因此当前 mixed 性能还伴随确定的上下文和生成成本。

新增分析产物：

```text
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/metrics.tsv
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/comparison.json
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/report.md
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/contract_audit.json
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/contract_audit.md
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/three_way_metrics.tsv
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/morgan_vs_coverage_standard/
outputs/paper/coverage_reasoning_context_gpt_oss_120b_scaffold_valid/analysis/morgan_vs_coverage_aware/
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison.svg
outputs/paper/starling_benchmark_results_scaffold_valid_gpt_oss_20b_vs_120b/figures/starling_model_comparison_highres.png
```

### 7.2 Visible MMP-ledger context ablation（2026-08-03 完成）

为让 visible group reasoning 明确看到每个 retrieved neighbor 对应的结构重合信息，新增 opt-in
`coverage_mmp_ledger` profile。该 profile 保持 `query_feature_coverage` selector、raw retrieval、neighbor set、
single-molecule analysis 和 final schema 不变；对每个 selected analog 复用常驻
`mmp_structure_compare` 的 MCS/MMP 文本，并用 Morgan marginal feature 统计组织 rank-by-rank 的互补、冗余
和未覆盖区域。Morgan feature coverage 不解释为 atom coverage；没有 matched-pair transformation 时具体
fragment correspondence 必须标为 unresolved。

当前 matched valid 实验固定为 Starling scaffold、GPT-OSS-120B、`deployment_visible + parent_disjoint`、
Starling `full_mechanism`、`top_k=3`、`min_similarity=0.30` 和 `query_feature_coverage` selector。Standard 与
MMP-ledger 两边分别完成 BBB `500/500`、Bioavailability `209/209`、Skin `245/245`，均为
`n_failed_runs=0`；两边复用相同 frozen single analyses，smoke 中 raw retrieval 逐字节一致。

| task | Standard macro-F1 | MMP ledger macro-F1 | Δ macro-F1 | flips | McNemar p | macro-F1 Δ 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.6843 | 0.6829 | -0.0014 | 66 | 1.0000 | [-0.0346, +0.0311] |
| Bioavailability | 0.6216 | 0.6511 | +0.0295 | 36 | 0.4050 | [-0.0238, +0.0848] |
| Skin Reaction | 0.5642 | 0.5610 | -0.0032 | 33 | 1.0000 | [-0.0575, +0.0517] |

结果仍为 mixed / no-go for promotion：BBB 和 Skin 基本持平，Bioavailability 有正向 point estimate，但三项
paired-bootstrap interval 均跨零，McNemar 也不显著。因此当前证据不支持将 MMP-ledger 升级为默认 prompt；
它继续作为 visible-only 可插拔 ablation 保留。该结果只在 canonical
`starling_model_comparison.{svg,png}` 总图中追加两行 visible experiment，不生成独立 performance figure。

配对统计与总图输入：

```text
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/metrics.tsv
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/comparison.json
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/report.md
outputs/paper/coverage_mmp_ledger_gpt_oss_120b_scaffold_valid/analysis/figure_metrics.tsv
```

入口与产物：

```text
tools/chembl_tool/paper_experiments/summarize_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/analyze_coverage_selector_retrieval_changes.py
tools/chembl_tool/paper_experiments/plot_coverage_selector_llm_matrix.py
outputs/paper/coverage_selector_llm/analysis/metrics.tsv
outputs/paper/coverage_selector_llm/analysis/comparison.json
outputs/paper/coverage_selector_llm/analysis/report.md
outputs/paper/coverage_selector_llm/analysis/retrieval_change_analysis.json
outputs/paper/coverage_selector_llm/analysis/retrieval_change_analysis.tsv
outputs/paper/coverage_selector_llm/analysis/retrieval_change_report.md
outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix.svg
outputs/paper/coverage_selector_llm/analysis/figures/coverage_selector_llm_matrix_highres.png
```

## 8. Identity-blind 补充控制：当前实测状态

Blind 条件隐藏 query/neighbor 的结构、名称和 source ID，并由 harness 预先提供脱敏后的 properties /
comparison tool evidence。它与 deployment-visible 同时改变 identity visibility 和 tool execution，
所以不是“纯 identity effect”；正式 visibility attribution 仍需 matched-prefetch replay。

统一入口：

```text
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
  --benchmark-split random|scaffold
  --visibility-mode identity_blind
  --neighbor-identity-policy operational
```

两套 split 各已有 22 个 condition metrics，但当前还没有达到正式报告 gate：

```text
random:
  22/22 condition artifacts present
  9 failed sample-condition runs，全部在 Bioavailability

scaffold:
  22/22 condition artifacts present
  5 failed sample-condition runs：
    BBB ChEMBL flat 2
    Bioavailability Starling direct-full / flat / mechanism 各 1

BBB random、Skin random/scaffold 以及其余列出的条件为 0 failure。
```

Blind 当前 point estimates 可用于进度诊断，但在 repair 到 `n_failed_runs=0`、完成 query-SMILES leak 和
visibility-contract audit 前，不进入 formal Starling bar chart。完整逐 condition metrics 位于：

```text
outputs/paper/molecular_evidence_agent_starling_random/runs/
outputs/paper/molecular_evidence_agent_starling_scaffold/runs/
```

## 9. Skin_Reaction：label scope、retrieval distance 与性能下降

Skin gold label 是 sensitization/contact allergy，不是所有 adverse skin effects 的并集。Paper-facing
Starling mechanism families 是 evidence-scope expansion，而不是像 oral bioavailability
`F = Fa × Fg × Fh` 那样的层层因果分解：

```text
Tier 1: direct sensitization/contact-allergy anchors，最接近 gold
Tier 2: sensitization AOP key events，仍与 gold 对齐但不是 clinical outcome
Tier 3: phototoxicity、irritation、corrosion、local damage；相关但属于不同 skin hazard
Tier 4: permeability/retention/exposure context；不能单独证明 sensitization
```

Observed Starling macro-F1：

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 0.6431 | 0.6353 | 0.6300 |
| scaffold | 0.5973 | 0.5921 | 0.5836 |

Full-flat 与 full-mechanism 在两种 source、两种 split 上逐 query 的 LLM-visible evidence-row multiset
完全一致（各 380/380 matches，0 mismatch）。因此 flat-to-mechanism 变化来自 evidence organization、
独立 branch reasoning 和 final synthesis，不是 mechanism 看到了更多 rows。

Starling direct-to-mechanism flips：

```text
random:
  38 flips；17 corrected，21 broken；net -4 correct

scaffold:
  29 flips；12 corrected，17 broken；net -5 correct
```

Trace 中反复出现：

- phototoxicity/irritation/local-damage evidence 被提升成 sensitization risk；
- Tier 4 skin exposure support 被误用为 hazard 支持；
- weak AOP signal 或 distant analog narrative 被 branch packaging 放大；
- broad mixed negatives 稀释一个较近的 positive anchor；
- 没有 neighbor 时仍可因 prompt/organization 边界发生 prediction instability。

Token burden 同时明显增加。保存的 successful-response logical token mean：

| split | direct | full flat | full mechanism |
|---|---:|---:|---:|
| random | 27.7k | 77.4k | 96.1k |
| scaffold | 26.9k | 73.4k | 91.6k |

这说明更多 tokens 主要来自更多 evidence rows、group calls 和重复 synthesis，不等于更多
label-aligned information。完整量化与逐 flip artifacts：

```text
outputs/paper/skin_reaction_retrieval_diagnostic/agent_quant_summary.json
outputs/paper/skin_reaction_retrieval_diagnostic/agent_random_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/agent_scaffold_flips.jsonl
outputs/paper/skin_reaction_retrieval_diagnostic/report_flip_summary.csv
outputs/paper/skin_reaction_retrieval_diagnostic/report_trace_examples.csv
```

### Tier 1+2 final-only post-hoc

为隔离 Tier 3/4，复用原 full-mechanism 的 frozen single、Tier 1、Tier 2 branch，删除 Tier 3/4 后只重跑
final。两套均为 380/380、0 failure，760 个 run 的 single/group artifact reuse 和 scope audit 为 0 mismatch：

| split | Tier 1+2 macro-F1 | 相对 full mechanism | 相对 direct |
|---|---:|---:|---:|
| random | 0.6310 | +0.0010 | -0.0121 |
| scaffold | 0.5948 | +0.0112 | -0.0025 |

两套 bootstrap 95% CI 均跨 0。裁掉 Tier 3/4 在 scaffold 有小幅 point-estimate recovery，但没有超过
direct，也没有证明稳定改善；Tier 2 analog noise、branch synthesis 和生成不稳定性仍可能贡献误差。

真正的 final-only scope 过滤入口是：

```text
--final-only-source-batch <source batch>
--final-only-groups Mechanism.tier_1 Mechanism.tier_2
```

普通 `--groups` 只控制 fresh pipeline，不能用于 resume-final artifact filtering。

### Canonical direct/AOP source repair（2026-08-13，valid-only）

为消除 direct/AOP source overlap 和 scope contamination，新增
`skin_sensitization_direct_aop.v3` canonical partition。两个 raw Starling acquisition parquet 保持 immutable；
112,582 条 raw rows 全部进入 row-level partition audit。Current canonical source 为：

| partition | records | unique parent identities | molecule-level evidence rows | allowed endpoint |
|---|---:|---:|---:|---|
| direct | 54,596 | 3,806 | 4,225 | final sensitization/contact-allergy outcome |
| AOP | 11,434 | 1,289 | 1,340 | MIE、KE2、KE3、KE4 only |

AOP acquisition 中 12,083 条有 direct assay anchor 的 records 移入 canonical direct，另有 12 条未标 AOP、但有
明确 direct assay 与 usable label 的 records 也归入 direct；photo hazard、irritation-only、prediction-only/
in-silico、integrated/unresolved records 拒绝。全字段扫描的 photo、in-silico、integrated 和 AOP irritation 命中均
为 0。Direct 中 65 条文字提及 irritation 的记录均同时报告 sensitization/contact-allergy outcome，保留该 direct
outcome，但 irritation 本身不作 label evidence。Direct/AOP source-record overlap 为 0；同一文段同时独立报告
LLNA 与 DPRA/h-CLAT 等不同 endpoint 时允许各自保留，并保留 source-record provenance。

用重建后的 heldout-parent-filtered index 和 MiniMol descriptor 跑 DeepSeek-v4-pro scaffold-valid：

| condition | accuracy | macro-F1 | TN / FP / FN / TP |
|---|---:|---:|---:|
| canonical direct | **0.6939** | **0.6410** | 38 / 35 / 40 / 132 |
| canonical direct+AOP mechanism | 0.6694 | 0.6123 | 35 / 38 / 43 / 129 |

两者均 245/245、0 failure，direct 和 mechanism 分别有 245/245 retrieval、245/490 group rows、245/245 final
与 trace artifacts。Mechanism 相对 direct 有 34 flips（direct-only correct 20、mechanism-only correct 14），
accuracy delta `-0.0245`，macro-F1 delta `-0.0287`，paired-bootstrap 95% CI `[-0.0801,+0.0224]`，McNemar
`p=0.3915`。AOP 清理后 direct/AOP top-3 molecule overlap 从历史 145/245 queries 降为 88/245，但 AOP branch
仍未超过 direct，不能 promotion。正式 artifacts：

```text
data/starling_data/skin_reaction/canonical_sensitization_v3/manifest.json
outputs/paper/minimol_retrieval_features_skin_canonical_v3_record_supported_v2_valid_verified/summary.json
outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_deepseek_v4_pro_skin_canonical_v3_minimol_top3_audited/
```

早期无后缀、`_final` 和 `_verified` run roots 是 full-field scope audit 完成前的中间 artifacts，均已 superseded，
不得作为 canonical v3 最终结果引用。

### Canonical AOP source/topology-gated final-only no-go（2026-08-13，valid-only）

E19 在完全冻结的 canonical direct 与 direct+AOP branch 上做严格 matched final-only integration。非触发样本
逐样本复用原 direct final；只有 direct=`no_risk`、AOP direction=`supports_sensitizer`、group/card
transferability 至少 moderate，且同一 evidence card 同时通过以下两项，才允许 fresh final：

1. raw source text 含 validated AOP method 或明确 skin-sensitization/contact-allergy context，并有 positive signal；
2. query/reference 共享显式 reactive SMARTS family，或共享 source-backed activation route precursor。

245 个 valid 分子中 direct=`risk` 为 167 个；其余 78 个 direct=`no_risk` 中，75 个 AOP 为 useful=false、
neutral/context-only、low transferability 等 non-voting group。剩余 3 个 supportive group 也全部被门控拒绝：
idx61 的模型声称共享 methacrylate，但 query 不含该 alpha,beta-unsaturated carbonyl；idx121 虽有共享
phenol/tyrosinase-quinone route，但来源只是 generic GSH depletion，没有 skin-sensitization context 或 validated
method；idx192 虽共享 alpha,beta-unsaturated carbonyl，但其对应 positive source 同样缺少上述 source context。

因此 `n_triggered=0`、fresh final=0、prediction flips=0；候选严格等于 direct，macro-F1/accuracy 仍为
`0.641036/0.693878`，paired delta 与 95% bootstrap CI 均为 `[0,0]`。245 个 query、single 与 direct-group
artifact parity failure 均为 0。预先冻结的 promotion gate 未通过，不读取或运行 Skin test。审计产物：

```text
outputs/paper/skin_aop_gated_final_v1_scaffold_valid_deepseek_v4_pro/
```

由于该候选没有触发任何样本且未通过 promotion，专用一次性 runner/gate 未进入长期维护代码；结论由上述
冻结 audit artifact 和本节记录保留。

### Outcome-calibrated multi-event causal-panel seed no-go（E22；2026-08-14，valid-only）

E22 不复用 raw AOP card 作为新方法。版本化 compiler 只保留同一 reference parent 上同时满足 direct
sensitization outcome、`MIE_protein_binding`、至少一个 KE2/KE3/KE4，且 event-level 70% agreement、AOP
事件间方向一致并与 direct outcome 一致的 multi-event causal card。Full source 得到 118 cards
（106 positive、12 negative）；排除 valid+test parents 后 index 保留 99（89/10）。新 index 只追加
`Mechanism.sensitization_causal_panel`，MiniMol candidate order、全部旧 group membership/evidence 与 245/245
direct retrieval hash 均保持不变。

零成本 gate 后，按 top causal-card source direction 和 cosine 排名、完全不读取 gold/direct correctness，冻结
64 个 valid query（32 negative-top、32 positive-top）。运行只 fresh 生成 causal group 与 final；single 和
`Mechanism.tier_1` 均 64/64 从 frozen DeepSeek artifacts 复用。64/64 成功、0 failure。非触发样本逐条保留
direct prediction，合并到完整 245 条后的结果为：

| condition | accuracy | macro-F1 | TN / FP / FN / TP |
|---|---:|---:|---:|
| frozen canonical direct | 0.6939 | 0.6410 | 38 / 35 / 40 / 132 |
| causal-panel gated seed | 0.6694 | 0.6151 | 36 / 37 / 44 / 128 |

Delta 为 accuracy `-0.02449`、macro-F1 `-0.02592`；paired-bootstrap macro-F1 95% CI
`[-0.05630,+0.00217]`。10 个 prediction flips 中 2 beneficial、8 harmful。失败不是 direct retrieval drift：
single/direct-group parity 全部通过。Trace 显示 55/64 causal branches 为 low transferability，48/64 direction
为 neutral/unclear；53/64 的三张 card 中最高报告 Morgan Tanimoto 仍小于 0.30。9/10 flips 出现在
neutral/low-transfer branch 后，其中 8 个有害，说明 extra branch 主要改变 final 的 uncertainty/default policy，
而非传入可转移的 causal signal。Heldout index 的 89:10 positive/negative card imbalance 是另一明确上游缺口。

事后只让 moderate/high 且 directional branch 生效会留下 9 条、仅 1 个 beneficial flip，完整 245 条
macro-F1 nominal `+0.00347`；这是读取本次 branch outputs 后的 exploratory sensitivity，不是冻结 gate，不能
promotion。E22 因此停止，不启动 BBB seed、不读取 Skin test。下一次 Starling acquisition 若要继续，必须把
reactive mechanism family/activation route 与 outcome+AOP 一起抽取，并在相同机制 family 内补足实验 negative，
再以 shared reactive route 作为 retrieval eligibility；MiniMol cosine 只能在 eligible family 内排序。

```text
tools/chembl_tool/paper_experiments/skin_causal_panel_seed/
outputs/paper/skin_causal_panel_seed_v1_scaffold_valid_deepseek_v4_pro/{source_build_audit.json,retrieval_audit_summary.json,analysis/}
```

## 10. 主要运行与汇总入口

```text
data builder:
  python -m tools.chembl_tool.common.starling.build_benchmark_datasets

held-out index:
  python -m tools.chembl_tool.paper_experiments.build_starling_benchmark_indices

MiniMol retrieval features:
  python -m tools.chembl_tool.paper_experiments.build_minimol_retrieval_features

formal/blind matrix:
  python -m tools.chembl_tool.paper_experiments.starling_benchmark_matrix

formal summary:
  python -m tools.chembl_tool.paper_experiments.summarize_starling_benchmark

formal figure:
  python -m tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview

MiniMol:
  python -m baselines.minimol.run_bioavailability_ma \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --output-dir <model-and-split-specific-output> \
    --train-all --evaluation-split valid

Morgan KNN:
  python -m baselines.structure_knn.run \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --output-dir <model-and-split-specific-output> \
    --k 3 --evaluation-split valid

MiniMol embedding KNN:
  python -m baselines.minimol.run_embedding_knn \
    --data-dir data/processed_starling/<Task>/<random|scaffold> \
    --embedding-cache-dir <MiniMol-output>/embeddings \
    --output-dir <model-and-split-specific-output> \
    --k 3 --evaluation-split valid

MiniMol embedding agent retrieval:
  python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
  python -m tools.chembl_tool.paper_experiments.plot_starling_with_minimol_agent

generic final-only group filtering:
  tools/chembl_tool/common/task_workflows/reasoning_batch.py
```

默认 reference pool 是 train。Test post-selection sensitivity 必须用隔离 root，并显式配对：index builder
`--heldout-subsets test`、matrix `--reference-pool train_valid`、KNN
`--reference-splits train valid`；matrix 会核验 base-index metadata 确实只排除了 test。Valid evaluation、
valid-only/重复 split 或声明与实际 index 不一致都会在模型调用前拒绝。

MiniMol 代码入口总索引为 `baselines/minimol/README.md`。另有
`baselines/minimol/run_starling_table2.py` 用于外部 Starling 论文 released CSV 的 Table 2 复现，结果见
`baselines/minimol/STARLING_TABLE2_REPRODUCTION.md`；其中包含作者补充的 `n_extractions` weighted-BCE
matched 结果。该 release 的任务定义和 split 与当前 gold 不同，因此不纳入本结果总账、上方 canonical
snapshot 或统一绘图入口。

### 2026-08-02 至 2026-08-04 入口与 artifact contract 更新

- `starling_benchmark_matrix.py` 提供显式 `--output-root`、`--evaluation-subset`、`--timeout-s`、
  `--neighbor-selector`、`--neighbor-context-profile` 和 `--single-analysis-root`。旧 condition lane/phase
  scheduler 已移除；single/group/final branch 统一进入全局 ready queue，frozen single 按 sample 解锁。
- `molecular_evidence_agent.py` 和四个 task runner 透传 selector/context/timeout；fresh
  `deployment_visible + parent_disjoint` 允许写入独立 ablation root，不再要求 operational reuse plan。
- `coverage_reasoning.py` 提供 `standard`、blind-safe `coverage_aware` 和 visible-only
  `coverage_mmp_ledger` 三个 profile；group reuse 现在要求 source/target profile 一致。
- `summarize_starling_benchmark.py` 支持 model-specific pipeline root、valid/test subset、visibility 和
  failure-inclusive policy。显式 v4/valid summary 不再隐式加载 historical test baselines；baseline roots 必须
  明确传入且与同一 lineage/subset 匹配。`summarize_results.py` 仍只用于旧 TDC visibility matrix，不能对
  `runs_identity_blind_parent_disjoint/` 生成 v4 审计；此前生成的空
  `analysis_identity_blind_parent_disjoint/report.md` 不是 canonical 结果。
- 三个 baseline CLI 新增 `--evaluation-split valid|test`。MiniMol 的 valid 评估只允许与 `--train-all`
  配对，并可通过重复 `--reuse-embedding-cache-dir` 按 exact SMILES 复用 molecule-only embeddings；manifest
  不复用 label 字段。KNN/MiniMol metrics 同时保存 `evaluation_split` 和 `n_evaluation`，旧 `n_test` 仅为兼容。
- `plot_starling_model_comparison.py` 是 model、visibility、baseline 和 matched method rows 的唯一总图入口；
  comparison summary 必须 task/split/subset/n/baseline 对齐，experiment TSV 必须提供不会重复绘制的 anchor。
- Resident tool service 新增 `/tools/batch`、bounded workers、process-resident MolGpKa、native-thread cap、
  SQLite/WAL + LRU/single-flight cache；部署与滚动切换规范见 `tools/service/README.md`。
- `watch_glm_tunnel_and_matrix.py` 监控 `/v1/models`、SSH tunnel 和唯一 resumable matrix。完成计数与 batch
  gate 对齐：必须有可归一化的 task prediction、single/final status=ok、准确的 expected group 数和 0 failed
  groups；它不保存或重放 SSH 密码，Duo 仍需用户批准。

### GLM endpoint 默认值与 2026-08-01 benchmark

上述 paper/Starling GLM 入口现在默认使用：

```text
http://127.0.0.1:50000/v1
nvidia/GLM-5.2-NVFP4
reasoning_effort=""（省略 API 参数，保持历史 GLM reasoning contract）
```

本机端口由 `ssh -fNT parcc-glm` 转发到 `dgx008:50000`。旧 LiteLLM 请求名
`zai-org/GLM-5.2-FP8` 实际也解析到 `hosted_vllm/nvidia/GLM-5.2-NVFP4`，因此这里不是两种模型精度的比较。

同一约 3.3k input-token structured-output 请求的 64 并发 smoke：

| runtime contract | valid JSON | wall time | p50 latency | aggregate token/s |
|---|---:|---:|---:|---:|
| old LiteLLM + historical default thinking | 0 / 64 | 16.41 s | 7.99 s | 14,767 |
| direct dgx008 + `reasoning_effort=none`（诊断，不采用） | 64 / 64 | 6.68 s | 3.43 s | 32,221 |

上表的 `2.46x` wall-time、`2.33x` p50 和 `2.18x` aggregate throughput 改善来自关闭 reasoning，
不属于正式 agent 设置，不能作为新默认相对旧端点的速度结论。正式 runner 保持历史
`--disable-thinking --reasoning-effort ""`：不发送 reasoning-effort 参数，但继续接收并保存 GLM reasoning。

为找 endpoint 吞吐上限，另用约 2.1k token/request、短 JSON 输出和共享 prompt 前缀测试直连端点：

| concurrency | valid JSON | wall time | aggregate token/s |
|---:|---:|---:|---:|
| 128 | 128 / 128 | 4.73 s | 57,128 |
| 256 | 256 / 256 | 6.16 s | 87,883 |
| 512 | 512 / 512 | 8.80 s | 122,927 |

这组最高约 `122.9k token/s` 同样使用 `reasoning_effort=none`，只保留为关闭 reasoning 的 endpoint
ceiling 诊断；它不是当前 reasoning-enabled 默认，也不能换算真实 agent pipeline 完成时间。

### Task-local KNN–agent router v2（2026-08-05，scaffold valid）

`router_oof/` 使用各 task 的完整 scaffold train set 构造 5-fold OOF 训练数据，并保持 Morgan `k=3`、
GPT-OSS-120B `identity_blind + parent_disjoint` direct agent 的冻结合同。v2 删除了随 fold reference-pool 大小
漂移的特征以及 k=3 下的确定性冗余特征，分别估计 `P(agent_only_correct)` 与 `P(knn_only_correct)`；只有
train-only nested paired-bootstrap promotion gate 通过时才允许从 KNN 切换到 agent，否则部署为严格 KNN
fallback。v1 artifact 保留，v2 写入独立 `router_v2/` lineage。

BBB OOF agent 的最后一个缺失样本也已补齐，三个 task 均通过 zero-failure/paired-completeness gate：
BBB `18,425/18,425`、Bioavailability `1,674/1,674`、Skin Reaction `1,966/1,966`。冻结 v2 后一次性得到的
valid 结果为：

| task | valid n | promotion gate | KNN acc / macro-F1 | direct agent acc / macro-F1 | deployed router acc / macro-F1 | router - KNN | switch / rescue / harm |
|---|---:|---|---:|---:|---:|---:|---:|
| BBB | 500 | PASS | 0.7740 / 0.7127 | 0.7300 / 0.7138 | **0.7900 / 0.7330** | +0.0160 / +0.0203 | 42 / 25 / 17 |
| Bioavailability | 209 | PASS | 0.7177 / 0.6199 | 0.6507 / 0.6434 | **0.7177 / 0.6339** | +0.0000 / +0.0139 | 24 / 12 / 12 |
| Skin Reaction | 245 | FAIL, KNN fallback | **0.6857 / 0.6064** | 0.5633 / 0.5487 | **0.6857 / 0.6064** | +0.0000 / +0.0000 | 0 / 0 / 0 |

BBB 的 point estimate 同时超过 KNN 与 direct agent；但其 10,000-repeat label-stratified paired bootstrap
accuracy delta 95% CI 为 `[-0.0100, 0.0420]`，macro-F1 delta CI 为 `[-0.0096, 0.0506]`，均跨 0，现阶段
只能解释为 promising signal。Bioavailability 在 accuracy 不变时 macro-F1 增加 1.39 pp，但 CI 也跨 0。
Skin 的 ungated candidate 在 valid 上实际为 0.6531 / 0.5768（12 rescues、20 harms）；promotion gate 将其
挡住，因此 deployed policy 没有重复 v1 的退化。这说明 v2 已解决“坏 router 必须安全退回 KNN”的工程问题，
尚未证明三 task 都能显著优于 KNN。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/valid_evaluation_result_v2.json`。每 task 的 model、feature、
prediction、report 和输入 SHA-256 位于同一 root 下的 `<task>/router_v2/`。这些 valid 结果不得用于回写
feature、threshold 或 promotion policy；formal test 尚未运行。

### Output-aware post-selector v3（2026-08-05，scaffold valid development）

v3 不覆盖 v2，而是在 KNN 与 GPT-OSS-120B direct agent 已产生不同 label 后，只对 disagreement rows 学习
`agent_only_correct` 对 `knn_only_correct`。它保留原始 query、KNN 和 evidence feature，并比较三档输入：
18-feature output/query/KNN、49-feature evidence、70-feature evidence+structured-trace；每档同时比较 Logistic
与小型 HistGBDT。`KNN=0, agent=1` 与 `KNN=1, agent=0` 使用独立 train-only thresholds。Nested OOF promotion
gate 未通过时部署严格回退 KNN，不能在看到 valid 后解锁。

| task | train gate / frozen profile | KNN acc / macro-F1 | candidate acc / macro-F1 | deployed acc / macro-F1 | candidate switch / rescue / harm | oracle acc |
|---|---|---:|---:|---:|---:|---:|
| BBB | PASS / HistGBDT + evidence | 0.7740 / 0.7127 | **0.7760 / 0.7382** | **0.7760 / 0.7382** | 71 / 36 / 35 | 0.9120 |
| Bioavailability | FAIL / HistGBDT + evidence+trace | 0.7177 / 0.6199 | 0.7177 / **0.6499** | 0.7177 / 0.6199 | 28 / 14 / 14 | 0.9091 |
| Skin Reaction | FAIL / HistGBDT + evidence | **0.6857** / 0.6064 | 0.6694 / 0.6123 | **0.6857** / 0.6064 | 42 / 19 / 23 | 0.8122 |

结果支持“原始 evidence features 有用但还不够”的判断：三个 task 的 full-train deployment choice 都包含
evidence，Bioavailability 还选择了 structured trace；BBB nested outer folds 也全部选择 evidence 或
evidence+trace profile。不过 BBB valid 的 71 次切换只有净 1 次 rescue，accuracy 仅比 KNN 高 0.2 pp；其
macro-F1 delta 为 +2.55 pp，但 10,000-repeat paired-bootstrap 95% CI `[-0.0114, 0.0653]` 跨 0。
Bioavailability candidate 的 14 rescues/14 harms 保持 accuracy、改善类别平衡，但 train accuracy uncertainty
未通过预注册 guardrail；Skin candidate harms 多于 rescues且 accuracy 下降。因此 v3 仍未接近 oracle，瓶颈
不是 feature 是否全部保留，而是现有 trace/evidence summary 对“这一次 disagreement 谁正确”的辨别力仍弱。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v3.json`。每 task 的 model、nested OOF
prediction、metrics、manifest、valid feature 和 report 位于 `<task>/post_selector_v3/`。这是经过既有 valid
观察后设计的 development experiment，只能用于方法迭代；formal test 尚未运行，也没有用 valid 调整 v3
模型、profile 或 thresholds。

### Direction-calibrated post-selector v3.1（2026-08-05，scaffold valid development）

v3.1 是在观察 v3 valid 后冻结的独立 development lineage，不覆盖 v3，也不新增 LLM 请求。两个 disagreement
direction 分别选择 Logistic/HistGBDT、feature profile 与 threshold；每个模型使用 fold-heldout sigmoid
calibration 的五组件 ensemble。一个 direction 只有在 train 上至少 route 20 rows 且 agent-win precision 的
one-sided 95% Wilson lower bound `> 0.5` 时才允许启用；overall train promotion 还要求 paired-bootstrap accuracy
delta 的 95% CI lower bound `> 0`。Valid 不拟合、不校准、不调 threshold。

| task | train gate | KNN acc / macro-F1 | v3.1 acc / macro-F1 | delta | switch / rescue / harm | valid accuracy delta 95% CI | held-out evidence gate |
|---|---|---:|---:|---:|---:|---:|---|
| BBB | PASS | 0.7740 / 0.7127 | **0.8060 / 0.7473** | **+0.0320 / +0.0346** | 26 / 21 / 5 | **[+0.0140, +0.0520]** | **PASS** |
| Bioavailability | PASS | 0.7177 / 0.6199 | 0.7225 / 0.6020 | +0.0048 / -0.0180 | 9 / 5 / 4 | [-0.0239, +0.0335] | FAIL |
| Skin Reaction | PASS | 0.6857 / 0.6064 | 0.6980 / 0.5962 | +0.0122 / -0.0102 | 13 / 8 / 5 | [-0.0163, +0.0408] | FAIL |

BBB 是目前唯一同时满足 train promotion、held-out accuracy evidence gate，并在 valid 上同时提高 accuracy 与
macro-F1 的 task。其两个方向都发生了少量选择性切换：`KNN=0, agent=1` 为 14/18 正确，`KNN=1, agent=0`
为 7/8 正确。Bioavailability 与 Skin 的 point-estimate accuracy 小幅上升，但 CI 跨 0 且 macro-F1 分别下降
1.80 pp 和 1.02 pp，因此不能称为稳定改善，也不能因为 valid 结果去调整 frozen policy。Skin 的复杂 selector
在 train 上也没有超过简单的 `route KNN=0/agent=1 only` baseline，提示该 task 当前主要利用输出方向 prior，
而不是学到可迁移的 trace/evidence 条件边界。

Canonical receipt：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v31.json`。每 task 的 calibrated ensemble、
manifest、nested predictions、valid features/predictions/metrics 位于 `<task>/post_selector_v31/`。这是 valid-informed
development experiment；formal test 未运行，不能把 BBB 结果当成最终 test claim。

### Router termination diagnostics（2026-08-05，train-only）

为区分 data-limited 与 signal-limited，后续诊断不再读取 valid/test，也不重新搜索 v3.1 family/profile。首先将
BBB outer-train disagreement supervision 按 direction/fold/target 分层下采样，固定 full-train direction specs，
主比较改为 learned selector 相对不读 feature 的 direction-only OR rule：

| full-equivalent disagreement budget | seeds | realized outer-train n | accuracy delta vs OR | macro-F1 delta vs OR | positive-seed fraction (acc / F1) |
|---:|---:|---:|---:|---:|---:|
| 800 | 5 | 640.0 | -0.10 pp | +0.50 pp | 0.00 / 1.00 |
| 1,600 | 5 | 1,282.4 | +0.05 pp | +0.91 pp | 0.80 / 1.00 |
| 3,200 | 5 | 2,559.2 | +0.49 pp | +2.33 pp | 1.00 / 1.00 |
| 7,130 | 1 deterministic | 5,704.0 | **+0.74 pp** | **+3.37 pp** | 1.00 / 1.00 |

Full-size train OOF 的 learned-vs-OR paired 95% CI 为 accuracy `[+0.45,+1.02] pp`、macro-F1
`[+2.90,+3.82] pp`。因此 BBB 确实存在随训练量上升的 task-local routing signal；在约 800 disagreements 时，
复杂 selector 还不能提高 accuracy。这解释了 Oral/Skin 为什么更难，但不能覆盖既有 scaffold-valid 结论：BBB
valid 上 learned selector 相对 OR 的 accuracy 增量仍为 0，macro-F1 `+0.95 pp` 且 CI 跨 0，说明跨 scaffold
transfer/calibration 仍是独立瓶颈。

条件触发的共享表示实验固定为两个 direction 的 task-balanced Logistic、完整 68-feature generic profile、
task one-hot，以及 task-specific sigmoid calibration/threshold/Wilson gate。每个 target outer/inner heldout fold
还从其它 task training rows 排除了相同 molecule identity 和 fold-group/scaffold：

| task | shared acc / macro-F1 | shared - direction-only OR | shared - frozen task-local | shared-vs-local macro-F1 95% CI |
|---|---:|---:|---:|---:|
| BBB | 0.7607 / 0.6510 | +0.04 / +0.53 pp | **-0.69 / -2.84 pp** | [-3.29,-2.41] pp |
| Bioavailability | 0.7437 / 0.6239 | -0.30 / -0.18 pp | -0.42 / **-1.01 pp** | [-1.81,-0.32] pp |
| Skin Reaction | 0.6846 / 0.5758 | -0.05 / +0.38 pp | +0.05 / +0.22 pp | [-0.39,+0.89] pp |

共享表示没有让任一小 task 相对 task-local 得到可信提升，且显著伤害 Bioavailability macro-F1；transfer
continuation gate 因此为 **FAIL / `stop_router_main_method`**。结论是：更多 task-local data 能帮助 BBB，但现有
query/KNN/evidence/trace representation 的可迁移性不足。这里的 shared model 监督仍是最终 agent-win label，
不是 counterfactual evidence utility。Router 保留为 reliability baseline 和 negative diagnosis，不再继续做
valid-informed family/profile sweep，也不消耗 formal test。下一条方法线应改变监督信号本身，例如
counterfactual evidence add/drop utility，而不是继续调 post-selector。

Canonical receipts：
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_v31_learning_curve_result.json` 与
`outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_v31_transfer_result.json`；task-local job metrics、
compressed predictions、cross-task fold exclusions 和 thresholds 保存在相应 task artifact directory。

## 11. 当前未完成项

- `record_supported_v2` GLM blind 首轮为 5,876/6,887，尚有 1,011 个 endpoint/tunnel 失败待
  `--skip-existing` 修复；GLM visible 尚未启动，二者均未进入正式结果表；
- `record_supported_v2` 的 paired direct-agent significance 尚未在完整 GLM 矩阵上生成；任何 partial
  GLM p-value 都无效；
- GLM random-valid blind 仍有 Bioavailability 两个 ChEMBL full 条件各 1 个失败，尚未通过 zero-failure gate；
- v4 random-valid 的 matched train-label baselines 尚未生成；
- Bio selected-condition formal test 已按冻结合同运行一次；BBB/Skin formal test 均未启动，其中 BBB E20
  已 gate-failed 并停止、不再列为待办，Skin 仍需独立 promotion gate；
- test matched-prefetch 尚未扩展到全部当前 Starling conditions；
- ECFP RF/XGBoost 和 matched-neighbor evidence retrieval-only vote 未完成；
- 用于确认表示选择稳健性的独立第二种 pretrained encoder baseline 未完成；
- source-quality 双人 annotation 未完成；
- GLM/GPT-OSS 当前每项主要只有一次 run，关键 comparisons 仍需独立 repeats；
- task-local router v2、output-aware post-selector v3 和 direction-calibrated v3.1 已完成 scaffold-valid
  development；matched-size curve 证明 BBB signal 随 data 增强，但 shared transfer gate 失败，router 主方法线
  已停止，formal test 按 gate 决定不启动（不是待补运行）；
- Skin Tier 1+2 是 test-triggered post-hoc diagnosis，不得升级成预注册 primary condition。

## 12. Git 发布里程碑

已直接推送到 `origin/main` 的 Starling migration：

```text
4595fde  Add Starling benchmark data splits
8c16ae6  Add Starling benchmark evaluation pipeline
```

`8c16ae6` 包含 random/scaffold matrix、held-out provenance、MiniMol train-all、Morgan KNN、bar chart
及相关入口。本轮 scoped publish 在其上补充 2026-07-27 的 Skin Tier 1+2 final-only
filtering/results、集中结果总账，以及 2026-07-28 的 MiniMol embedding cosine KNN、统一汇总和
canonical bar chart；同时存在的 viewer/coverage-selector 独立改动不属于该 publish scope。

2026-07-30 的 scoped publish 进一步加入 MiniMol/cosine operational agent retrieval 的 38-condition
零失败 gate、与已有结果同图比较的独立 SVG/PNG/TSV、绘图入口和本节结果记录；viewer 与
coverage-selector 独立改动仍不属于该 publish scope。

2026-08-11 的仓库整理已直接进入 `origin/main`：

```text
4ea18e4  Add one-pass RL training subsystem (#3)
fa8eec5  Centralize reasoning and prompt profiles
d359170  Finalize Starling benchmark lineages and diagnostics (#4)
```

`d359170` 同时冻结 BBB meaningful-CNS-access v2 数据/审计、Skin direct-scope parity、共享 paired
statistics、matched train-label/train-ratio diagnostics、当前结果总账和对应测试。发布后本地 `main`、
`origin/main` 与上述 commit 对齐，旧 dirty-worktree inventory 已关闭。
