# KNN–Agent Router：train-only OOF 实施计划

> Archived/stopped research：本计划只用于复现已完成的 train-only diagnosis，不属于当前论文矩阵，
> 不得由正式 benchmark、baseline、summary 或 plotting 入口启动，也未运行 formal test。

## 目标与边界

本实验验证一个固定 KNN 与一个固定 direct-evidence agent 之间是否存在可学习的逐样本选择规律。
v1–v3.1 的可部署 router 始终按 task 分别训练，不共享参数。主线终止前只额外运行一次严格 train-only、
不可部署的 task-balanced shared-representation diagnosis，用来检验小 task 能否借用其它 task 的监督。

Router 不是新的 Starling benchmark matrix，也不改变当前 frozen valid/test 结果。所有 OOF 代码和产物使用
独立 namespace；正式 benchmark runner 仍只接受 `valid|test`。

第一版 candidate contract：

```text
KNN:
  train-label KNN；同时保存未加权和 similarity-weighted vote features。

Agent:
  model = gpt-oss-120b
  visibility = identity_blind
  neighbor identity = parent_disjoint
  source = Starling
  BBB_Martins = starling_direct
  Bioavailability_Ma = starling_direct_full（不是 numeric-only）
  Skin_Reaction = starling_direct
```

Agent 的 router label 只取上述固定 direct condition 的最终 `pred_label`。`none` 只用于生成 direct condition
必须复用的 single-molecule branch，不是 router candidate。第一版不运行 full-flat/full-mechanism，也不在多个
agent settings 之间做事后择优。

## 数据隔离合同

每个 task 独立构建 5 个 OOF folds。输入只来自当前 Starling scaffold `train`；valid/test 在任何 OOF
选择、训练或校准中均不可见。

Fold group key 使用 `scaffold_or_parent.v1`：

- 有环分子按 canonical Bemis–Murcko scaffold 分组；
- 无环分子没有可用 scaffold，按 `rdkit_fragment_parent.v1` parent identity 单独分组；
- 同一 group 不能跨 fold；fold assignment 固定 seed，并尽量平衡样本量与二分类标签数。

对每个 fold `i`：

1. query 是 train 中被分到 fold `i` 的分子；
2. fold-specific Starling evidence index 必须排除 `valid + test + fold i query parents`；
3. KNN 候选只来自其它 4 folds；
4. agent 运行 `none -> direct`，direct 复用同 fold 的 frozen single output；
5. query-time 仍启用 `parent_disjoint`，作为 fold-level exclusion 之外的第二层 identity gate；
6. 每个 train sample 恰好产生一条 OOF KNN prediction 和一条 OOF agent prediction。

任何 fold 都必须通过：query parent/index overlap 为 0、valid/test parent/index overlap 为 0、fold group overlap
为 0、输出行数与 query 行数一致、agent failure 为 0。失败样本通过同一 fold launcher 的 `--skip-existing`
恢复，不能静默删除。

## 模块边界

```text
tools/chembl_tool/paper_experiments/router_oof/
  contract.py
    TaskSpec、路径、direct condition、模型/visibility 默认值和 artifact schema version。

  folds.py
    读取 train provenance、生成 deterministic scaffold-aware folds、写 fold inputs/heldout union/manifest。

  indices.py
    从冻结 full index 按 parent identity 构建 fold-specific direct indices；复用原 fingerprint/evidence，
    只重映射 molecule/group index，并写 zero-overlap audit。旧 Skin index 的 exposure role 在 fold index
    内确定性修正为 `context_modifier`，源 builder 也已同步修正。

  knn.py
    生成严格 OOF Morgan KNN predictions/features；保存 unweighted/weighted p、margin、entropy 和 density。

  agent.py
    将 fold manifest 转成现有 batch/global-prompt-pool 命令；只调度 none + frozen direct condition。

  features.py
    从 query 通用 RDKit descriptors、KNN prediction、direct retrieval.json 和 direct final prediction构建
    固定数值 feature row；不读取 agent 最终 label、confidence 或 reasoning 文本作为输入特征。

  train.py
    每个 task 单独训练 regularized logistic regression 和 small GBDT；不跨 task 共享参数。

  valid.py
    从正式 scaffold-valid 的 full-train Morgan KNN、冻结 direct-agent retrieval/prediction 和 query descriptors
    构造同 schema features；只加载全 train deployment model/calibrator/threshold 做一次 held-out 评估，禁止
    在 valid 上重新选择 family 或 threshold。

  cli.py
    prepare / build-indices / run-knn / run-agent / progress / build-features / train / finalize /
    evaluate-valid / audit 子命令。

  postprocess_watch.py
    监控 detached full-agent PID；进程退出后调用 gated `finalize`。只有全部 sample 的 none/direct stage
    完整时才物化 features 和训练 routers，否则写 blocked 状态，不消费 partial outputs。

  post_v31_common.py
    v3.1 终止性诊断共享的 row-to-array、disagreement direction、direction-only comparator 和 provenance
    哈希语义；不包含拟合或实验 policy。

  post_v31_learning_curve.py
    固定 v3.1 family/profile 的 matched-size train-only curve；压缩保存逐 job predictions。

  post_v31_transfer.py
    固定 task-balanced Logistic 的 shared-representation train-only diagnosis；target-task calibration、threshold
    和 identity/scaffold exclusion 保持独立。
```

共享 identity、retrieval、batch、global prompt pool 和 JSON 原子写入逻辑继续由现有 common 模块负责；
`router_oof` 不复制 task prompt、LLM client、evidence index builder 或 result completeness 判断。

## Artifact contract

```text
outputs/paper/router_oof/gpt_oss_120b/scaffold/
  manifest.json
  <task>/
    folds.jsonl
    fold_00/ ... fold_04/
      agent_input/valid.jsonl
      query_molecule_labels.jsonl
      heldout_molecule_labels.jsonl
      evidence/<direct-index-name>/
      agent/
        runs_identity_blind_parent_disjoint/<task>/<none-or-direct-condition>/
      knn_predictions.jsonl
      fold_manifest.json
    oof_knn_predictions.jsonl
    router_features.jsonl
    router_features.tsv
    router/
      model.joblib
      model_manifest.json
      nested_oof_predictions.jsonl
      metrics.json
      report_zh.md
      valid/
        features.jsonl
        predictions.jsonl
        metrics.json
        manifest.json
        report_zh.md
```

Manifest 必须保存源文件 SHA-256、fold seed/version、identity normalizer、query/index parent counts、模型、endpoint、
visibility、identity policy、direct condition、feature schema、代码生成时间和所有子产物路径。不得保存 API key。

## Router target 与评估

每个 task 独立训练。训练目标先保留四种 paired outcome，而不是提前丢信息：

```text
agent_only_correct
knn_only_correct
both_correct
both_wrong
```

历史 v1 binary action policy：

- 默认 action 为 KNN；
- 只有模型预测 `agent_only_correct` 的校准概率超过 train-only 选择的阈值时才切换到 agent；
- `both_correct` 不奖励昂贵切换；`both_wrong` 单独报告，不伪造正确 action；
- threshold 和超参数只用 OOF train 内层 folds 选择。

当前 v2 action policy：

- 删除 `knn_reference_size_log1p`；它在 5-fold OOF train 表示约 80% reference pool、在 valid 表示 100%
  train pool，是 fold construction artifact，不是 query-specific confidence；
- 删除 k=3 下由 vote fraction 确定的 margin/entropy，以及 evidence neighbor/unique count 的重复输入；
- train-only nested folds 同时比较 `knn_compact`、`query_knn`、`query_knn_evidence` 三个通用 feature profile；
- 分别拟合并校准 `P(agent_only_correct)` 与 `P(knn_only_correct)`，routing score 为前者减后者；
- threshold 必须满足 accuracy 不低于 KNN 0.5 percentage point 以上，再按 macro-F1、accuracy、低 switch rate
  选择；
- nested selected candidate 还必须通过 2,000-repeat label-stratified paired bootstrap：macro-F1 delta lower
  95% bound > 0 且 accuracy delta lower bound >= -0.5 pp。失败时 deployment threshold 固定为 1.01，严格回退
  KNN；candidate 结果仍单独保存用于诊断。

v1 artifacts 保留在 `router/` 与 `router_features.*`；v2 独立写入 `router_v2/` 与
`router_features_v2.*`，两条 lineage 不覆盖、不混表。

### v3 output-aware post-selector（2026-08-05）

v3 是独立的 post-selector lineage，不覆盖 v1/v2。它只在 KNN 与 direct agent 已经给出不同 label 时判断
是否采用 agent；训练 target 是 disagreement rows 上的 `agent_only_correct` 对 `knn_only_correct`。三个 task
仍分别训练，所有 profile、模型族和方向阈值都只在 scaffold-train 的 nested OOF 中选择：

- `output_query_knn`（18 features）：输出方向、query descriptors 和 weighted/unweighted KNN confidence；
- `output_query_knn_evidence`（49 features）：保留上一项，并加入原始 evidence 汇总以及与 KNN/agent 决策相对的
  evidence direction、confidence、transferability 和 conflict 汇总；
- `output_query_knn_evidence_trace`（70 features）：再加入 final/single/group 的结构化 confidence、agreement、
  conflict/gap/caveat counts 和 reasoning length。它不读取 gold label，也不新增 LLM 调用。

每个 profile 比较 Logistic 与小型 HistGBDT，并分别选择 `KNN=0, agent=1` 和 `KNN=1, agent=0` 两个方向阈值。
train-only paired-bootstrap promotion gate 失败时，两个部署阈值均冻结为 `1.01`，严格返回 KNN。Neighbor
permutation、bounded evidence dropout 等 stability feature 需要额外 agent calls，未伪装成已有 trace feature，
留作后续独立 ablation。

产物使用 `post_selector_features_v3.jsonl` 与 `post_selector_v3/`；valid 汇总为
`post_selector_valid_result_v3.json`。阈值搜索通过两个 disagreement 方向的 confusion-matrix 增量做严格等价
加速；paired bootstrap 通过 label-stratified paired state 的 multinomial counts 向量化，不改变统计目标。

### v3.1 direction-calibrated selector（2026-08-05，冻结实现合同）

v3.1 只复用已经完整物化的 `post_selector_features_v3.jsonl` 和 valid feature/agent artifacts，不增加任何
LLM call，也不覆盖 v3。它针对 v3 diagnosis 暴露的方向不对称和 score-transfer 问题冻结如下合同：

- `KNN=0, agent=1` 与 `KNN=1, agent=0` 分别训练、选择 profile 和冻结 threshold；方向常量列从各自 head
  的 matrix 删除，其余 query/KNN/original evidence/decision-relative evidence/trace features 全部保留在相应
  profile；
- 每个 direction candidate 是 scaffold-fold-aware calibrated ensemble：base estimator 在一个 fold complement
  上训练，sigmoid calibrator 只读取该 component 的 held-out fold；external prediction 平均所有 calibrated
  components。Outer held-out score、deployment score 和 inner model-selection 都调用同一 ensemble constructor；
- nested outer fold 内只用 outer-train 选择 direction-specific family/profile/threshold，再预测 outer-heldout；
  full-train deployment 只用 cross-fitted train scores选择两个 heads 和 thresholds；
- threshold 必须有至少 20 个 routed train rows，并且 agent-win precision 的 one-sided 95% Wilson lower bound
  `> 0.5`；不满足时该方向 threshold=`1.01`。合格策略先最大化 net rescues/accuracy，再比较 macro-F1 与低
  switch rate；
- 同时报告 `route KNN=0/agent=1 only` 和 train-prior-safe direction baseline，避免复杂模型只是在学习输出方向；
- overall deployment 仍需 train-only paired-bootstrap accuracy promotion gate（observed accuracy delta > 0 且
  95% CI lower > 0）；macro-F1 的 -0.5 pp lower-CI guardrail 单独记录并通过 safe Pareto frontier 报告，
  不否决一个已证明 accuracy-safe 的策略。candidate、fallback 与方向诊断均保留，valid 不允许重新校准、
  重选 profile 或 threshold。

v3.1 独立写入 `post_selector_v31/`，consolidated valid receipt 为
`post_selector_valid_result_v31.json`。这是已经观察 v3 valid 后的方法开发实验；只能作为 development 证据，
formal test 在代码、gate 和报告冻结前不得运行。

### v3.1 matched-size learning curve（2026-08-05，终止性诊断合同）

该实验只回答“BBB 的差异是否主要来自更多 disagreement supervision”，不是新的 router 版本，也不允许继续
读取 valid/test。它复用 frozen `post_selector_features_v3.jsonl`、scaffold fold 和 v3.1 full-train deployment
manifest，不新增 LLM call，不覆盖 `post_selector_v31/`：

- 为隔离训练量效应，两个 direction 的 family/profile 固定为 v3.1 full-train manifest，不在每个 curve point
  重新搜索；threshold 仍只从相应 outer-train 的 nested calibrated scores 选择；
- 每个 outer fold 内按 `direction × scaffold fold × agent-win target` 确定性分层下采样，避免改变方向比例或
  正负 target mix；outer-heldout fold 始终完整评估；
- BBB 的 full-equivalent disagreement budgets 固定为 `800 / 1,600 / 3,200 / 7,130`，前三档运行 5 个 seed，
  full budget 为确定性单次；Bioavailability `772`、Skin `802` 只跑 full-size reference；
- primary comparator 是不读取 feature 的 direction-only OR rule：仅在 `KNN=0, agent=1` 时采用 agent。KNN
  继续报告但不再作为“learned feature 是否有增量”的充分 comparator；
- 每个 point 保存 candidate-vs-OR 的 paired accuracy/macro-F1 delta、bootstrap CI、switch/rescue/harm、direction
  threshold/coverage/precision 和 realized train counts；所有预测以压缩矩阵保存以便复核；
- 只有 BBB learned-vs-OR 增量随 budget 稳定上升，且 full-size 至少一个主指标 paired 95% CI lower `> 0`、
  另一指标不退化，才继续共享 representation + task-specific calibration。否则结束 router 主方法线，只保留
  reliability baseline/negative diagnosis。Formal test 在该判定完成前保持未触碰。

产物独立写入 `<task>/post_selector_v31_learning_curve/`；入口为
`router_oof.cli diagnose-post-v31-learning-curve`。

### Shared-representation transfer diagnosis（2026-08-05，条件触发合同）

Matched-size curve 若证明 BBB full-size learned-vs-OR 有 train-only 增量，只允许再做一个共享表示诊断，检验
小 task 是否能借监督；不再搜索 task-local profile/model：

- 两个 disagreement direction 分开拟合；base model 固定为 task-balanced Logistic，输入固定为完整通用
  `output_query_knn_evidence_trace` profile 加 task one-hot；不比较其它 family/profile；
- 每个 target task 仍使用独立 sigmoid calibration、threshold 和 Wilson risk gate，不共享 label prior 或最终
  decision boundary；每个 task 在自己的 scaffold outer folds 上完整 OOF 评估；
- shared base 的其它 task rows 若与 target heldout fold 具有相同 molecule identity 或 fold-group/scaffold key，
  必须排除；inner calibration fold 也执行同一 exclusion，不能通过跨 task 重叠绕过 scaffold holdout；
- 每个 direction 对各 source task 使用等总 sample weight，避免 BBB 的 7,130 disagreements 淹没 Oral/Skin；
- primary comparisons 为 shared-vs-direction-only OR 与 shared-vs-frozen-task-local curve prediction；valid/test
  继续不读取。只有至少一个小 task 相对 task-local 的 paired CI 明确为正、另一指标不退化，才值得把共享
  representation 发展为正式方法；否则结束 router 主线。这里的监督仍是最终 agent-win label，不能称为
  counterfactual evidence utility；后者需要新的 evidence add/drop intervention，是另一条方法线。

产物独立写入 `post_selector_v31_transfer/`；入口为
`router_oof.cli diagnose-post-v31-transfer`。

主 router receipt 至少报告 always-KNN、always-agent、learned router、retrospective oracle、disagreement subset
precision/recall、accuracy、macro-F1 和两类 recall；校准诊断按版本保存 Brier/ECE 或 direction-level AUC/Brier。
Valid 只在代码/feature/threshold 冻结后运行；test 是否运行由 continuation gate 决定。

## 执行阶段与 gate

1. **实现与单元测试**：fold、identity union、KNN weighted features、evidence aggregation、paired target、task-local training。
2. **离线 smoke**：每 task 准备 folds/index/KNN；用合成 rows 验证 feature/train/audit 全链。
3. **endpoint smoke**：每 task、每 fold `limit=1`，验证 none/direct dependency、trace、role 和 resume。
4. **小规模 pilot**：每 task 固定分层样本，确认失败率、tokens、吞吐和 feature missingness。
5. **全量 OOF**：22,065 个 train samples 各运行一次 none + direct；单一 launcher 共享 endpoint budget。
6. **task-local router**：分别训练 BBB、Bioavailability、Skin router；不共享参数。
7. **valid gate**：冻结设置后评估；不得根据 valid 改 evidence condition、feature 语义或 label policy。
8. **test gate**：仅当 continuation gate 通过时才允许一次正式 test；本轮 shared-transfer gate 已失败，
   因而 router 主方法线不启动 formal test。

历史启动门要求 endpoint smoke、产物完整性、fold/index overlap 和 failure recovery 全部通过后才可运行
22,065-sample OOF；这些 gate 已全部通过并完成全量运行。当前禁止项已经变为：不得根据 valid 继续搜索新的
router family/profile，也不得在 continuation gate 失败后消耗 formal test。

## 截至 2026-08-05 的执行状态

- [x] 三个 task 分别生成 5-fold scaffold-or-parent OOF assignment；22,065 个 train rows 恰好覆盖一次。
- [x] 15 个 fold-specific indices 构建完成；33 个必需 audit checks 全部通过，所有 residual parent overlap 为 0。
- [x] 三个 task 的 Morgan `k=3` OOF KNN 已完成；同时保存 weighted/unweighted router features。
- [x] endpoint smoke：3 tasks × fold 0 × 1 sample 的 none/direct 共 6 batches 全部成功。
- [x] 修复 global pool 在“相同 condition batch_id、不同 fold root”下的 state-key collision，并加入回归测试。
- [x] 8 samples/fold pilot 经 `--skip-existing` 修复后，15 个 task-fold 均为 8/8 successful、0 failed。
- [x] 全量 agent 已由单一 global pool（`parallelism=128`，复用 120 个 pilot samples）运行完成；detached
  postprocess watcher 已执行完整性 gate 和 `finalize`，最终三个 task 均达到 paired-complete/zero-failure。
- [x] Bioavailability 与 Skin 的全量 direct agent、feature materialization 和 task-local nested OOF router 已完成；
  两个 canonical SMILES 为单字符 `N` 的样本暴露了短结构 preflight 误报，已按通用 token-boundary 规则修复、
  补齐并通过 zero-failure gate。
- [x] 第一版冻结 valid gate 已执行，命令为：

  ```bash
  python -m tools.chembl_tool.paper_experiments.router_oof.cli evaluate-valid \
    --tasks bioavailability_ma skin_reaction
  ```

  该入口验证 full-train model、KNN `k=3`、agent model/visibility/identity/source/condition、样本行数和
  zero-failure 后才预测，并对九个输入 artifact 保存 SHA-256。结果如下；这些数值不得用于回调 v1 threshold：

  | task | valid n | KNN acc / macro-F1 | direct agent acc / macro-F1 | frozen router acc / macro-F1 | router - KNN |
  |---|---:|---:|---:|---:|---:|
  | Bioavailability | 209 | 0.7177 / 0.6199 | 0.6507 / 0.6434 | 0.6938 / 0.6587 | -0.0239 / +0.0388 |
  | Skin Reaction | 245 | 0.6857 / 0.6064 | 0.5633 / 0.5487 | 0.6408 / 0.5600 | -0.0449 / -0.0464 |

  Bioavailability 的 Macro-F1 提升在 held-out valid 上扩大，但 accuracy 仍下降；Skin 两项均下降。因此 v1
  task-local router 只证明部分 task 存在可学习 routing signal，不支持跨 task 稳定优于 KNN 的结论。
- [x] BBB 全量 agent 已补齐到 18,425/18,425 paired complete。最后一个样本 `C[Se]` 的 Starling neighbor
  evidence 包含同位素文本 `36Cl influx`；`Cl` 同时是单原子 canonical SMILES，旧 preflight 将正常 assay
  文本误报为身份泄漏。公共规则现统一把单原子 SMILES 视为不可从 assay prose 审计的 token，多原子
  `C[Se]` 仍严格审计；回归测试覆盖该边界。
- [x] v2 Bioavailability/Skin train-only nested OOF 已完成：Bio gate PASS，选择
  `hist_gradient_boosting + query_knn`；Skin candidate 虽有正 macro-F1 delta，但 accuracy lower CI 未通过
  guardrail，因此 gate FAIL 并冻结 KNN fallback。
- [x] v2 Bioavailability/Skin scaffold-valid 已执行：Bio router 保持 KNN accuracy 0.7177，同时 macro-F1 从
  0.6199 提高到 0.6339；Skin deployed router 因 gate fallback 与 KNN 完全相同（0.6857 / 0.6064），未再复现
  v1 的显著下降。Valid-informed v2 diagnostic 不能回写 feature/threshold；正式结论需冻结后在 test 一次评估。
- [x] BBB v2 train-only nested OOF 与 scaffold-valid 已完成；三 task 的 frozen v2 valid 汇总如下：

  | task | promotion gate | KNN acc / macro-F1 | direct agent acc / macro-F1 | deployed router acc / macro-F1 | router - KNN | switches / rescues / harms |
  |---|---|---:|---:|---:|---:|---:|
  | BBB | PASS | 0.7740 / 0.7127 | 0.7300 / 0.7138 | 0.7900 / 0.7330 | +0.0160 / +0.0203 | 42 / 25 / 17 |
  | Bioavailability | PASS | 0.7177 / 0.6199 | 0.6507 / 0.6434 | 0.7177 / 0.6339 | +0.0000 / +0.0139 | 24 / 12 / 12 |
  | Skin Reaction | FAIL, KNN fallback | 0.6857 / 0.6064 | 0.5633 / 0.5487 | 0.6857 / 0.6064 | +0.0000 / +0.0000 | 0 / 0 / 0 |

  BBB 的 valid point estimate 同时超过 KNN 与 direct agent，但 10,000-repeat paired bootstrap 的 accuracy
  delta 95% CI 为 `[-0.0100, 0.0420]`，macro-F1 delta 95% CI 为 `[-0.0096, 0.0506]`，均跨 0；当前只能称为
  promising signal，不能称为统计确认。Bioavailability 保持 KNN accuracy 并提高 macro-F1 1.39 pp，CI 同样
  跨 0。Skin 的 ungated candidate 会下降到 0.6531 / 0.5768，train-only promotion gate 正确触发 fallback，
  deployed 结果与 KNN 完全一致。三 task consolidated receipt 为
  `outputs/paper/router_oof/gpt_oss_120b/scaffold/valid_evaluation_result_v2.json`；当时未运行 formal test，后续
  termination gate 失败后已明确决定不启动。
- [x] v3 output-aware post-selector 已完成三个 task 的 train-only nested OOF。BBB gate PASS；Bioavailability
  和 Skin 的 candidate 虽提高 train macro-F1，但 accuracy delta 下界低于 -0.5 pp guardrail，因此均冻结 KNN
  fallback。全量 train 选择的部署 profile 分别为 BBB `HistGBDT + output_query_knn_evidence`、Bioavailability
  `HistGBDT + output_query_knn_evidence_trace`、Skin `HistGBDT + output_query_knn_evidence`，说明 evidence/trace
  feature 有信号，但不是稳定充分的 selector。
- [x] v3 scaffold-valid development evaluation 已完成：

  | task | train gate | KNN acc / macro-F1 | candidate acc / macro-F1 | deployed acc / macro-F1 | candidate rescue / harm |
  |---|---|---:|---:|---:|---:|
  | BBB | PASS | 0.7740 / 0.7127 | 0.7760 / 0.7382 | 0.7760 / 0.7382 | 36 / 35 |
  | Bioavailability | FAIL | 0.7177 / 0.6199 | 0.7177 / 0.6499 | 0.7177 / 0.6199 | 14 / 14 |
  | Skin Reaction | FAIL | 0.6857 / 0.6064 | 0.6694 / 0.6123 | 0.6857 / 0.6064 | 19 / 23 |

  BBB 的 macro-F1 point estimate 增加 2.55 pp，但 10,000-repeat valid paired-bootstrap CI
  `[-0.0114, 0.0653]` 仍跨 0；accuracy 仅增加 0.2 pp。Bio candidate 保持 accuracy 并增加 2.99 pp macro-F1，
  但 train-only gate 已预先冻结 fallback，不能在看到 valid 后解锁。Skin candidate accuracy 下降 1.63 pp，
  fallback 正确阻止退化。Canonical receipt 为
  `outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v3.json`；formal test 未运行，且后续
  termination gate 失败后不再启动。
- [x] v3.1 direction-calibrated selector 已完成三个 task 的 train-only nested fit 与 frozen scaffold-valid
  evaluation。三个 task 的 train accuracy promotion gate 均通过，但独立 held-out valid evidence gate 仅 BBB
  通过：

  | task | KNN acc / macro-F1 | v3.1 acc / macro-F1 | delta | switches / rescues / harms | accuracy delta 95% CI | valid evidence gate |
  |---|---:|---:|---:|---:|---:|---|
  | BBB | 0.7740 / 0.7127 | 0.8060 / 0.7473 | +0.0320 / +0.0346 | 26 / 21 / 5 | [+0.0140, +0.0520] | PASS |
  | Bioavailability | 0.7177 / 0.6199 | 0.7225 / 0.6020 | +0.0048 / -0.0180 | 9 / 5 / 4 | [-0.0239, +0.0335] | FAIL |
  | Skin Reaction | 0.6857 / 0.6064 | 0.6980 / 0.5962 | +0.0122 / -0.0102 | 13 / 8 / 5 | [-0.0163, +0.0408] | FAIL |

  Canonical receipt 为
  `outputs/paper/router_oof/gpt_oss_120b/scaffold/post_selector_valid_result_v31.json`。Valid receipt 明确拆分
  `train_gate_promoted` 与只作证据解释的 `valid_accuracy_evidence_gate`，防止把 train gate 误写成 held-out
  成功。不得依据上述 valid 重新选择 v3.1 family/profile/threshold；formal test 后续按 termination gate
  决定不启动。
- [x] v3.1 matched-size learning curve 已按冻结合同完成。BBB learned-vs-direction-only OR 的 mean delta 随
  full-equivalent disagreement budget 单调改善：`800` 为 accuracy/macro-F1 `-0.10/+0.50 pp`，`1,600`
  为 `+0.05/+0.91 pp`，`3,200` 为 `+0.49/+2.33 pp`，full `7,130` 为 `+0.74/+3.37 pp`；full paired
  95% CI 分别为 `[+0.45,+1.02]` 与 `[+2.90,+3.82] pp`。这确认 BBB task-local signal 是 data-limited，
  但不改变其 scaffold-valid learned-vs-OR 未显著的事实。
- [x] 条件触发的 shared transfer diagnosis 已完成：固定 task-balanced Logistic + 68-feature generic profile，
  task-specific calibration/threshold，且对 target outer/inner heldout 做 cross-task identity/scaffold exclusion。
  相对 frozen task-local，BBB accuracy/macro-F1 为 `-0.69/-2.84 pp`，Bioavailability `-0.42/-1.01 pp`，
  Skin `+0.05/+0.22 pp` 且 CI 跨 0。没有小 task 通过 paired improvement gate，最终 decision 为
  `stop_router_main_method`。Formal test 保持未运行；不得继续 valid-informed shared family/profile sweep。
- [x] 完成代码/产物布局复审：matched-size 与 shared-transfer 继续保持独立入口和 artifact namespace，只把两者
  必须一致的 row-to-array、direction-only comparator、direction masks 和 provenance hash 下沉到
  `post_v31_common.py`；transfer fold 汇总改为无副作用读取。逐样本 transfer JSONL（BBB 约 6 MB）保留用于
  审计，不把它误判为源码膨胀；源码目录中的 `__pycache__` 属于可再生垃圾，不进入版本控制。
- [x] 复审后的验证：相关 router/global-pool/identity-blind/Skin-profile 测试 `39 passed`；全量重跑两条
  termination diagnostics 后指标与 canonical receipts 等价；`pyflakes`、AST parse 和 `git diff --check` 通过。
