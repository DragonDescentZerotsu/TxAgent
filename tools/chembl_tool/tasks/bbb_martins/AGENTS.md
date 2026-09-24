# BBB_Martins task notes

本文件只记录 BBB_Martins 的 task-specific 语义：label、当前数据版本、BBB evidence tier、过滤规则和 reasoning 边界。通用 ChEMBL workflow、wrapper 结构、batch/resume、viewer、cost 和目录规范统一记录在仓库根 `AGENTS.md`。

当前论文路径由 `experiment_config.py` 将细粒度 ChEMBL endpoint groups 合并为 4 个 mechanism families：
direct brain exposure、passive permeability、efflux transport 和 influx transport。`full_mechanism` 只为这 4
个 family 启动并行 reasoning；`full_flat` 使用同一 evidence union 并合成一个 branch。下文按
`Tier.endpoint_group` 分支的说明只描述旧 `run_reasoning_pipeline.py` native runner，不是新增任务的模板。
2026-07-22 已导入 Starling passive permeability、efflux 和 influx 三个 mechanism-family acquisition；它们与
既有 direct BBB evidence 合并到独立 full index，不改变旧 direct-only index。

## Task 定义

目标不是训练 BBB classifier，而是构建可审计的 BBB evidence library：给定 query molecule，先预取相似分子的 BBB / permeability / transporter evidence，再交给 reasoning LLM 判断 analog evidence 是否能 transfer 到 query molecule。

当前评估约定（Conditioned Benchmark）：

```text
Y=1 -> bbb_prediction=pass
Y=0 -> bbb_prediction=fail
final summary 必须在 pass/fail 中二选一；不再允许 uncertain prediction
```

当前 gold 是
系统给药后实验支持的 meaningful/adequate CNS access vs restricted/poor access，不是 passive permeability，
也不是任意 CNS trace detection：brain tissue、unbound brain、
brain/systemic ratio、CSF、PET/autoradiography 和明确体内 BBB outcome 可进入；PAMPA/细胞模型、计算预测、
mechanism-only proxy、非系统给药、altered barrier、间接疗效推断和明显方向冲突拒绝。低但非零 exposure
可以为 negative，CSF 保留为 proxy family。parent-level 冲突
继续按 accepted source records 计算 70% agreement，同 PMID 的多条 record 分别计票，精确 tie 拒绝。

唯一活跃 split 位于：

```text
data/conditioned_benchmark/BBB_Martins/scaffold/
```

train/valid/test 为 3,053/397/393 个 molecule-condition rows，identity/scaffold overlap 均为 0。旧
molecule-only、gold-vN 和 selected-vN 路径只是当前 cohort 的 source provenance，不是并列 benchmark；
精确迁移关系见 `data/conditioned_benchmark/migration_receipt.json`。

完整 progressive valid 使用 audited BBB record-level family overlay。Matched 397-row valid 已完成全部五层和
五个 baselines、零失败；当前结果和 freshness 只维护在 `paper_experiments/RESULTS.md` 与
`paper_experiments/current_conditioned_results.json`。五层固定为
direct measured CNS access、central functional/prediction/generic proxy、passive permeability、efflux、influx；
明确 efflux signal 优先于 uptake/influx，泛化 `ratio` 或 `transporter_mediated` 不构成 family assignment。
全量 `581,708` rows 的 ledger 和 0-violation gate 位于 source overlay 的 `purity_audit/`。完整构建、index、
逐层数据与分享入口见 `paper_experiments/CURRENT_STARLING_RETRIEVAL.md`；protocol 和 stable-identity
retrieval/reuse 规则见 `paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`。
涉及 5-FU/5-FC 的 external comparative row 因没有 influx assay 或可复核 experimental measurement，只在
near-direct 层可见，从未参与 gold vote；influx payload 不得复用该 card。

BBB reasoning contract 通过 `prompt_profiles.py` 独立版本化。旧 artifact 缺少 profile 字段时必须解释为
`meaningful_cns_access_v1`；跨 profile 的 single/group branch reuse 必须拒绝。2026-08-09 的两个 valid-only
候选均保留但未 promotion：`meaningful_cns_adjudication_v2` 只允许 direct negative outcome/measured efflux
支持 fail，导致 matched valid 322/366 预测 pass、macro-F1 降至 0.5652；v3 增加严格的
`convergent_intrinsic_barriers` fail basis，matched macro-F1 提高到 0.6696，但 full-Starling direct 为 0.6416，
未超过 v1 的 0.6452（paired CI 跨 0）。因此当前默认继续是 v1，v2/v3 只用于明确 opt-in 的历史复现；formal
test 未运行。详细 paired 结果见 `tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`。

Legacy native runner 边界：

```text
ChEMBL neighbor retrieval 不是 DeepSeek 可调用 tool。
ChEMBL neighbor retrieval 也不是当前 FastAPI service tool。
它是 run_reasoning_pipeline.py 内部的 evidence prefetch / context assembly 步骤。

DeepSeek group-level analysis 可调用的工具只有：
  mmp_structure_compare
  properties_compare

DeepSeek single-molecule analysis 可调用的工具只有：
  molecule_properties
```

## Task-specific 文件

```text
experiment_config.py
  Paper-facing direct、full_flat 和 4-family full_mechanism retrieval view；不包含 label policy。

endpoint_groups.py
  BBB Tier.endpoint_group、evidence_direction、evidence_strength 规则。

rules.py
  BBB assay screening 关键词、negative keywords、weak terms、transporter target genes。

scoring.py
  BBB assay 保留/剔除和打分统一入口。screen_assays.py 和 rescore_outputs.py 都调用 scored_row()。

run_reasoning_pipeline.py
  BBB_Martins retrieval/prompt assembly、reasoning stages 和 final-only rerun。

prompt_profiles.py
  versioned single/group/final schema、label scope、instructions 和 cross-field validation；旧 profile 不原地修改。

starling_benchmark.py
  Historical TDC-compatible mixed-permeability adapter；不得用于新 BBB 主结果。

experimental_meaningful_cns_access_benchmark.py
  当前 experimental meaningful-CNS-access gold adapter；隔离 prediction/in-vitro/altered-context、
  mechanism-only proxy 和 identity/provenance 不可靠 records。

build_starling_evidence_library.py
  从 `starling-labs/BBB` 构建 Starling BBB molecule-level evidence 和 neighbor index。
  支持 `--mode qualitative`（不向 LLM 暴露 quantitative metric/value）和 `--mode all`
  （保留 qualitative + quantitative 字段）两套 Tier 1 replacement source。

build_starling_full_evidence_library.py
  从既有 direct BBB evidence 加载 direct family，并通过公共 profile reader 接入
  `data/starling_data/bbb_martins/{passive_permeability,efflux_transport,influx_transport}`，构建论文
  `starling_full_flat` / `starling_full_mechanism` 共用的四-family index。
```

下面这些文件是 task-specific 配置 wrapper，公共实现见根 `AGENTS.md` 的 `tools/chembl_tool/common/task_workflows/` 说明：

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
run_reasoning_batch.py
```

不要在 wrapper 中新增业务规则；BBB assay 保留/剔除逻辑应只放在 `rules.py` 和 `scoring.py`，endpoint-group 语义应只放在 `endpoint_groups.py`。

## 当前数据和输出

当前推荐使用：

```text
outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/
```

其中：

```text
bbb_assay_candidates.csv
bbb_assay_candidates.jsonl
bbb_assay_report.md
bbb_health_check.md
bbb_activity_evidence.csv
```

当前 v6 统计：

```text
candidate assays: 20,369
activity evidence rows: 113,929
Tier 1: 4,306
Tier 2: 4,609
Tier 3: 10,752
Tier 4: 702
```

当前 evidence library 和 neighbor index：

```text
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_molecule_evidence.jsonl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl
outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.meta.json
```

Starling BBB Tier 1 replacement index 默认放在：

```text
outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/qualitative/
outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/all/
```

Starling BBB paper-facing full index 放在：

```text
outputs/paper/molecular_evidence_agent/evidence/bbb_starling_full/
```

构建命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_full_evidence_library \
  --workers 32 \
  --progress-every 10000
```

## E12 BBB ChEMBL distance expansion（C-family tree；v3 current）

该实验与旧 paper matrix 隔离，且从 base 到 extension 全程只使用 ChEMBL 36。旧 D/C index 和已有结果不改；
新产物位于 `outputs/chembl_tool/tasks/bbb_martins/distance_expansion/`。

v2 曾对全部 20,369 个 D/C assay 做 measured-state census。
除 direct/passive/efflux/influx 外，旧 D/C 已实际包含 tight-junction、efflux/influx-transporter abundance 和
functional PXR/CAR readouts；这些 node 及其新 assay 都不得重新进入 H1/H2。v1 prototype 因忽略该细粒度
base-state overlap 已被 v2 替代，不用于 LLM 实验。

新的冻结结构不是全局平行 H1/H2，而是：D 为 root，每个当前 C mechanism family 恰有一个聚合 H1 child，
每个 H1 至多有一个 optional H2 child。一个 child 可包含多个 targets/measurement families/assays，但它们共享
该 tree node 的 top-3 unique-neighbor budget 和相同 similarity threshold；不得按 target 各取 3 个。每条 assay
必须保存自己的 measured node 和 admissible path，并唯一归属一个 parent C family。

BBB v3 的可扩展 `c_family_ids` 仅为 `Mechanism.tier_2/3/4`。`Mechanism.tier_1` 组织 direct brain-exposure
evidence，语义上属于 D root；其中 `Tier 1.context_dependent` 为保持旧 `D+C` union 继续保留，但不视为一个需要
强行寻找 H1 的独立机制分支。

v2 historical prototype 曾使用：

```text
H1 mmp9_activity
  direct human MMP-9 activity/inhibition -> tight-junction integrity（已在 D/C）
  binding-only、mRNA-only、migration/invasion 等间接 phenotype 不纳入

H2 mmp3_activity（已判定无效）
  direct human MMP-3 activity/inhibition -> MMP-9 activity -> tight-junction integrity（已在 D/C）
  binding-only、非 MMP-3 target 和间接 cellular phenotype 不纳入
```

后续 shortcut audit 找到 `MMP-3 activity -> tight-junction integrity` 的直接支持，因此 MMP-3 到整个 B 的最短
路径为 1，不能继续作为 H2。v3 已按新 tree contract 实现：

```text
Mechanism.tier_2 passive/barrier
  Distance.h1.passive_permeability
    mmp9_activity -> tight_junction_integrity
    mmp3_activity -> tight_junction_integrity
  H2 unavailable（当前没有冻结可信 child）

Mechanism.tier_3 efflux
  Distance.h1.efflux_transport
    nrf2_activation -> efflux_transporter_abundance
  Distance.h2.efflux_transport
    keap1_nrf2_interaction -> nrf2_activation -> efflux_transporter_abundance

Mechanism.tier_4 influx
  Distance.h1.influx_transport
    hif1_activation -> influx_transporter_abundance（限 GLUT1 相关部分）
  Distance.h2.influx_transport
    phd2_activity -> hif1_activation -> influx_transporter_abundance
```

assay mapping 按实际 measured state 而不只按 target ID：KEAP1/NRF2 target 下的 cellular NRF2 translocation/ARE
reporter 属于 H1，只有 direct biochemical PPI inhibition 属于 H2；PHD2 target 下若读出本身是 cellular HIF/HRE
state 则属于 H1，只有 direct hydroxylase activity/inhibition 属于 H2。HIF 下游 VEGF/EPO-only、generic viability，
KEAP1 thermal-shift/SPR binding-only，以及所有缺少可用 compound endpoint 的记录均排除。

### v3 same-molecule causal continuity audit（2026-07-21）

后续审计发现：正确的 node-to-node shortest path 不保证能预测 **assay molecule 自己** 的 BBB label。完整记录见
`DISTANCE_SELF_RELEVANCE_AUDIT.md`，机器可读记录见 `distance_self_relevance.py`。

```text
pass_same_molecule:
  mmp9_activity
  mmp3_activity
  原因：compound 作为 barrier perturbagen 改变它自己也要面对的物理 junction/barrier；仍保留
        injury/inflammation、target exposure、时间尺度和 route sensitivity 等 scope 条件。

requires_query_role（正式 E12 不通过）:
  nrf2_activation
  keap1_nrf2_interaction
    缺失：同一 query molecule 是 induced ABCB1/ABCG2/ABCC2 substrate 的证据。

  hif1_activation
  phd2_activity
    缺失：同一 query molecule 是 GLUT1/SLC2A1 substrate 的证据。
```

NRF2 文献用 sulforaphane 诱导 transporter，但用 verapamil/其它 probe substrate 测 efflux；HIF-1 文献证明的是
GLUT1-dependent glucose uptake。两者都不能把 regulator perturbagen 自动等同于 downstream transporter substrate。
此前讨论的 AhR candidate 存在同一断点，已降为 `context/exposure modifier` candidate，不进入 H1。

因此 v3 manifest/index/retrieval replay 与 coverage 继续保留为 structural/retrieval engineering artifact，旧 D/C 与
旧 paper matrix 完全不改；但 v3 不得直接启动正式 E12 LLM performance run。下一版必须先通过：

```python
validate_self_relevance_audit(DISTANCE_CONFIG, SELF_RELEVANCE_AUDIT, require_publishable=True)
```

不能通过 prompt 提醒让 LLM 猜 substratehood 来修补缺失 evidence。若未来对同一 molecule 联合检索到经过验证的
transporter-substrate evidence，才可重新审计相应 family。

2026-07-22 已完成同-parent role-gated feasibility audit，入口为
`python -m tools.chembl_tool.tasks.bbb_martins.audit_role_gated_overlap`。严格只接受同一 molecular parent 的阳性
ABCB1/ABCG2/ABCC2 substrate/transport 或 GLUT1 substrate-uptake evidence；inhibitor、binding、ATPase、probe
accumulation、single-direction Papp、低 efflux ratio 和 inactive rows 不合格。实测：NRF2 为 27 overlap parents，
392-query >=1-neighbor coverage 5.87%、top-3 coverage 0.51%；KEAP1 为 6 overlap parents但 query coverage 0；
HIF-1/PHD2 均为 0 overlap。结论是 role-gated join 不足以形成稳定 Efflux/Influx H1/H2，它们继续
`requires_query_role`，不得启动正式 E12 LLM run。报告位于
`outputs/chembl_tool/tasks/bbb_martins/distance_expansion/analysis/v3/role_gated_overlap/`。

2026-07-22 的 bounded replacement search 已完成，详见 `PASSIVE_DISTANCE_CANDIDATE_AUDIT.md`。新增可接受 family
只有 MMP-2 H1 与 ROCK2 H1；二者聚合为一个 Passive H1 node 并共享 top-3，不增加独立 reasoning branch。ROCK1
缺 BBB isoform-specific support，MYLK/RhoA coverage 不足，MMP-14 因存在不经过 MMP-2 的 direct barrier shortcut
风险而不能声明 H2。Efflux/Influx 的同-parent与同-document补救仍不足，因此下一版候选结构为：

```text
Passive H1 = MMP-9 + MMP-3 + MMP-2 + ROCK2; Passive H2 unavailable
Efflux H1/H2 unavailable
Influx H1/H2 unavailable
```

正式构建 v4 前，公共 tree contract 必须先把 `H1 unavailable` 变成显式可审计状态；不得把不合格 family 塞入
节点来满足旧的“每个 C 恰有一个 H1”结构约束。v3 与旧 D/C/paper results 继续冻结不动。

ChEMBL 36 全库扫描 v2 historical artifact：

```text
scanned assays: 1,890,749
included extension assays: 696
base/extension assay-ID intersection: 0
H1 MMP-9: 443 assays, 2,268 indexed unique molecules
old H2 MMP-3: 253 assays, 1,206 indexed unique molecules（需要重分层）
extension evidence rows: 4,280
extension unique molecules: 2,747
superset index: 55,260 molecules, 21 source groups
```

ChEMBL 36 v3 current tree artifact：

```text
scanned assays: 1,890,749
included extension assays: 1,514
base/extension assay-ID intersection: 0
passive H1: 696 assays (MMP-9 443 + MMP-3 253)
efflux H1: 389 assays (functional NRF2)
efflux H2: 121 assays (KEAP1-NRF2 PPI)
influx H1: 146 assays (functional HIF-1)
influx H2: 162 assays (direct PHD2 hydroxylase)
extension activity rows: 34,838
extension indexed molecules: 21,127
superset index: 73,175 molecules, 24 source groups
```

`base_measured_state_census.tsv` 继续是下一版的强制输入。manifest builder 必须在 extension measured node 已出现在
D/C census 时直接失败；纳入集合与旧 D/C assay ID 的交集也必须为 0。不得为了补 coverage 把 tight-junction、
PXR/CAR 或 transporter-abundance assays 重新贴成 extension。

以下命令构建隔离的 v3 tree 产物；不会覆盖旧 D/C index 或旧 paper conditions：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_distance_assay_manifest \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --export-activities \
  --progress-every 100000

python -m tools.chembl_tool.tasks.bbb_martins.build_distance_extension_library \
  --workers 128 \
  --progress-every 50000

python -m tools.chembl_tool.paper_experiments.build_distance_index \
  --base-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl \
  --extension-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/extension/bbb_distance_extension_index.pkl \
  --output-index outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.pkl \
  --output-meta outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/bbb_distance_superset_index.meta.json \
  --index-version bbb_distance_expansion.v3 \
  --source-release "ChEMBL 36" \
  --workers 128

python -m tools.chembl_tool.paper_experiments.audit_distance_expansion

python -m tools.chembl_tool.paper_experiments.materialize_distance_retrieval
```

v3 audit 必须保持：D/D+C LLM-visible base parity 0 failures、tree-node source nestedness 0、
D+C->H1->H2 evidence retention 0、H1-after-H2 stability 0，并新增：每个 C 恰有一个 H1 spec、每个 H1 至多一个
H2 spec、每个 measurement family 唯一 parent、H2 全局 shortcut 0、每个 tree node top-3 与统一 similarity threshold。

2026-07-20 v2 实测 392-query retrieval audit 的旧四类严格 failure 均为 0。累计 coverage 为：D 0.673、
D+C 0.878、D+C+H1 0.888、D+C+H1+H2 0.888；H1 MMP-9 family coverage 为 0.199，H2 MMP-3
family coverage 为 0.120。由于 ontology 已失效，这些数字只保留作历史工程记录，既不是新 tree coverage，也不是
LLM performance；E12 macro-F1 仍未运行。

2026-07-20 v3 实测 392-query retrieval audit 的四类严格 failure 也均为 0。累计 coverage 为 D 0.673、
D+C 0.878、D+C+H1 0.923、D+C+H1+H2 0.923；mean unique neighbors 为 1.39、5.51、8.28、8.45。
tree-node coverage 为 passive H1 0.204、efflux H1 0.666、influx H1 0.528、efflux H2 0.043、influx H2
0.071。H2 coverage 很低且不增加总体 covered-query 数，正式报告必须明确这一点；macro-F1 尚未运行。

冻结 retrieval replay 位于：

```text
outputs/chembl_tool/tasks/bbb_martins/distance_expansion/retrieval_replay/v3/
  distance_d/
  distance_dc/
  distance_dc_h1/
  distance_dc_h1_h2/
  distance_mechanism_dc/
  distance_mechanism_dc_h1/
  distance_mechanism_dc_h1_h2/
```

每个 condition 均含 392 个 `runs/<condition>_idxNNNNN/retrieval.json`。reasoning batch 必须通过
`--retrieval-replay-source-batch` 使用这些冻结输入；不得让 task pipeline 按旧 experiment config 重新检索。

历史 TDC/E12 frozen query set（不是当前 Starling gold split）：

```text
data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl

fields:
  drug: query SMILES
  Y: BBB label
```

reasoning 产物统一放在：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/
```

## Legacy native BBB reasoning pipeline

```text
1. 读取 test_efflux.jsonl 的 query molecule。
2. 调用 retrieve_neighbors.py 预取每个 source-local Tier.endpoint_group 的 ChEMBL neighbor evidence。
3. 并发执行 single-molecule analysis；该历史 runner 中 DeepSeek 只可调用 molecule_properties。
4. 并发执行 endpoint-group analysis；该历史 runner 中 DeepSeek 只可调用 mmp_structure_compare 和 properties_compare。
5. final summary 读取 single + all group outputs，不暴露任何 tool。
6. 保存 retrieval/single/group/final/trace/manifest。
```

当传入 `--tier1-replacement-index` 时，pipeline 会从原 ChEMBL index 中排除所有
`Tier 1.*` groups，并用 replacement index 中的 `Tier 1.starling_direct_bbb_evidence`
替代；Tier 2/3/4 仍来自原 ChEMBL index。用于 Starling BBB 实验的典型命令：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode qualitative \
  --workers 128 \
  --progress-every 10000

python -m tools.chembl_tool.tasks.bbb_martins.build_starling_evidence_library \
  --mode all \
  --workers 128 \
  --progress-every 10000

python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --tier1-replacement-index outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling/qualitative/starling_bbb_neighbor_index.pkl \
  --batch-id <batch_id>
```

final-only rerun：

```bash
python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>
```

## Exact ChEMBL context

`chembl_exact_context.py` 是可选 evidence-rich 增强。它会用 query full InChIKey 查 ChEMBL exact molecule，并在 retrieved neighbor 涉及的 assay 中查 query activity，生成：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这会使用 query molecule 的已知 ChEMBL 实验记录，可能造成 prospective benchmark 的数据泄漏。因此默认关闭；只有显式传 `--enable-chembl-exact-context` 时才用于 retrospective / evidence-rich case study。默认批量评估不要开启。

默认 single-molecule prompt 不包含任何 ChEMBL 相关 payload 或 instruction。只有开启 exact context 且命中 query exact context 时，single-molecule payload 才包含 `exact_query_chembl_context`，并提示模型区分 direct same-molecule ChEMBL BBB evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

## Evidence 类型解释

### Tier 1

直接 BBB / brain exposure 证据，例如：

```text
brain/plasma ratio
brain to plasma ratio
logBB
Kp,uu,brain
brain concentration
brain level
brain perfusion
CSF/plasma
```

这是最接近“分子是否进入脑内”的证据。

### Tier 2

被动通透或体外屏障模型证据，例如：

```text
PAMPA-BBB
Caco-2 Papp
MDCK Papp
brain endothelial cell model
hCMEC/D3
bEnd.3
BMEC / BBMEC / RBEC
```

这类是 proxy evidence，不等价于体内脑暴露。

### Tier 3

外排转运体证据，例如：

```text
ABCB1 / P-gp / MDR1
ABCG2 / BCRP
ABCC / MRP
efflux ratio
bidirectional Papp
rhodamine 123 efflux
calcein AM
Hoechst 33342 accumulation
```

P-gp/BCRP substrate、efflux ratio、bidirectional Papp 比单纯 inhibitor IC50 更接近 BBB reasoning。

### Tier 4

摄取转运体证据，例如：

```text
LAT1 / SLC7A5
GLUT1 / SLC2A1
OATP1A2 / SLCO1A2
OAT3 / SLC22A8
MCT1 / SLC16A1
TFRC
```

Tier 4 必须是 functional uptake/transport/substrate 相关 assay。单纯 gene expression、Western blot、phosphorylation、binding affinity 等不保留。

---

## 重要过滤规则

### `brain plasma membrane` 不是 brain/plasma

`brain/plasma` 表示脑组织暴露量与血浆暴露量的比值，是 BBB evidence。

但：

```text
brain plasma membrane
brain plasma membranes
```

表示脑组织来源的细胞膜/膜制备物，经常出现在 receptor binding assay 中，不是 brain/plasma ratio。当前逻辑会过滤这类误命中。

### 不保留非功能性 influx assay

例如：

```text
GLUT1 expression
SLC2A1 RNA stability
Western blot
phosphorylation
Kinobead pull down
binding affinity
```

如果没有 uptake/transport/substrate 等功能性读数，不作为 BBB influx evidence。

### 非 transporter target 的 resistant-cell-line phenotype 噪声

例如 target 是 MAP3K5、proteasome 等，但 description 里出现：

```text
ABCB1-substrate-selected resistant cell line
ABCG2-substrate-selected resistant cell line
overexpressing ABCB1
overexpressing ABCG2
```

这类通常是 phenotypic/cytotoxicity setting，不作为高置信 BBB transporter assay。当前逻辑会过滤明显噪声。

但会保留功能性 transporter readout，例如：

```text
P-gp-mediated rhodamine 123 efflux
calcein AM assay
Hoechst 33342 accumulation
mitoxantrone accumulation
doxorubicin accumulation
transepithelial transport
```

---

## canonical SMILES 为空的情况

`bbb_activity_evidence.csv` 中少量行的 `canonical_smiles` 为空，这是 ChEMBL 原始结构数据缺失，不是 join 错误。

在 v6 中：

```text
activity evidence rows: 113,929
missing canonical_smiles: 279
affected molecule_chembl_id: 147
```

常见原因：

```text
structure_type = NONE
structure_type = SEQ
金属/无机物
放射性 technetium complex
蛋白/肽/序列类 molecule
ChEMBL 没有标准结构
```

后续构建相似性检索索引时，应过滤 `canonical_smiles` 为空的 molecule。原始 evidence 行可以保留，用于审计和报告。

---

## 长任务监控

一次性监控脚本：

```text
tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh
```

用途：每隔固定时间检查一个 tmux session 是否完成，完成后自动生成健康检查报告。

示例：

```bash
tmux new-session -d -s chembl_assay_monitor \
  'tools/chembl_tool/utils/BBB_Martins/monitor_chembl_assay.sh chembl_assay outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw 5400 /tmp/chembl_assay_monitor.log'
```

参数：

```text
chembl_assay: 被监控的 tmux session
outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw: 输出目录
5400: 检查间隔秒数，即 90 分钟
/tmp/chembl_assay_monitor.log: monitor 日志
```

这个脚本不是核心业务逻辑，可留作长任务运维工具。

---

## 测试

运行：

```bash
pytest -q tests/chembl_tool
```

当前覆盖：

```text
文本标准化
P-gp / Caco-2 / blood-brain barrier / Kp,uu brain 匹配
generic permeability / generic uptake 过滤
ABCB1 target + substrate 保留
brain plasma membrane 误命中过滤
Tier 4 非功能性 GLUT1/SLC2A1 assay 过滤
非 transporter target resistant-cell-line 噪声过滤
functional rhodamine efflux 保留
```

---

## 当前 retrieval / reasoning 状态

第二阶段已经基于 `bbb_activity_evidence.csv` 构建 molecule-level evidence library：

```text
1. 过滤 canonical_smiles 为空的 molecule
2. 标准化 molecule identity，排除 query exact same molecule，包括 full InChIKey、InChIKey connectivity layer 和 canonical SMILES 相同的记录
3. 用 Morgan fingerprint Tanimoto 检索相似 analog
4. 按 Tier、endpoint type、similarity bucket 聚合 evidence
5. 输出 analog evidence summary，而不是直接硬判定 BBB pass/fail
```

当前实现入口：

```text
build_evidence_library.py
retrieve_neighbors.py
run_reasoning_pipeline.py
```

ChEMBL neighbor retrieval 当前作为 pipeline 内部 evidence prefetch，不作为 LLM tool。
后续新增其他 ChEMBL task 时，继续复用 `tools/chembl_tool/common/` 的通用工具。

## TDC test-tuned indirect ablation (2026-09-24)

`indirect_ablation.py` and `indirect_applicability_v1.txt` own this opt-in experiment only. Do not replace the conditioned BBB target or the original baseline prompt. The prediction-only filter is lexical and incomplete; balanced50 is a budget with 24–50 actual records, not uniformly 50. The [reproduction guide](INDIRECT_ABLATION.md) is the single maintained instruction source.
