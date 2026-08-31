# TxAgent 当前系统：常驻分子工具服务与证据检索推理

## Env instruction
If you are on `node002` or `node001`, default to the `vllm` conda environment when you need RDKit or the local project dependencies. conda is at: /data1/tianang/anaconda3/condabin/conda

## 当前目标

本项目要构建一个可复用的分子证据检索与 reasoning 系统。BBB_Martins 是第一个概念验证任务；
当前同一套 workflow 已扩展到 Bioavailability_Ma、ClinTox 和 Skin_Reaction。整体流程是：给定一个 query molecule，
先通过常驻 FastAPI 工具服务计算分子属性、结构差异和属性差异，再从 task-specific ChEMBL evidence
library 中检索相似分子的实验读数，最后把工具输出和 assay evidence 交给 reasoning LLM，综合判断该
task 的目标 label。

当前已实现的 ChEMBL reasoning tasks：

```text
tools/chembl_tool/tasks/bbb_martins/
tools/chembl_tool/tasks/bioavailability_ma/
tools/chembl_tool/tasks/clintox/
tools/chembl_tool/tasks/skin_reaction/
```

## 当前 Conditioned Benchmark（2026-08-31）

四个任务只有一个活跃 benchmark 根，并提供 scaffold 与 random 两种 split：

```text
data/conditioned_benchmark/<Task>/{scaffold,random}/
```

不要在 runner、baseline 或结果图中直接引用历史的 molecule-only、`selected_vN`、BBB gold-vN 或
ClinTox source-build 路径。它们只用于 source provenance；原路径与当前数据的逐 split hash/行级等价关系统一
记录在 `data/conditioned_benchmark/migration_receipt.json`。公共路径常量、发布入口和完整合同为：

```text
tools/chembl_tool/common/starling/conditioned_benchmark.py
tools/chembl_tool/common/starling/publish_conditioned_benchmark.py
tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md
```

| task | scaffold train / valid / test | random train / valid / test | 当前 target |
|---|---:|---:|---|
| BBB_Martins | 3,053 / 397 / 393 | 3,075 / 384 / 384 | experimentally meaningful systemic CNS access |
| Bioavailability_Ma | 1,958 / 262 / 269 | 1,991 / 249 / 249 | oral bioavailability under the reported condition |
| ClinTox | 1,144 / 142 / 142 | 1,142 / 143 / 143 | clinical-trial toxicity failure versus approved comparator |
| Skin_Reaction | 1,997 / 246 / 248 | 1,993 / 249 / 249 | skin sensitization/contact allergy |

random split 使用相同 molecule-condition rows/labels，按 parent 整组做确定性、quality-stratified 80/10/10
分配；同一 parent 不会跨 split，每个 condition 在三路都出现，但 scaffold 允许跨 split。BBB、
Bioavailability 和 Skin 先最小化 valid+test 的 singleton-vote rows，再平衡 valid/test singleton，最后才平衡
label 和 condition；ClinTox 的 source count 不是 assay vote，不应用该质量目标。生成和审计入口为
`tools/chembl_tool/common/starling/build_conditioned_random_split.py`。所有 split 行使用统一的
molecule-condition schema。没有外部 condition 的行使用
`no_reported_external_condition`，prompt renderer 对它不输出 condition 句子；ClinTox 因没有合格的外部
condition，全部采用该值。两种 split 的 parent identity overlap 均为 0；只有 scaffold split 另外保证
Bemis-Murcko scaffold overlap 为 0。

BBB、Bioavailability 和 Skin 的当前 split 文件与已经完成评估的 conditioned cohort 字节级相同；ClinTox
只补 condition schema，ordered `(drug, Y)` 和 split 不变。已有 prediction 只能在 manifest input hash 与
migration receipt 匹配时复用，不能仅凭旧目录名复用。

Task-specific source voting 和 review 仍保留在各 task 模块中。BBB 的 direct gold 只接受系统给药后的实验性
meaningful CNS access；Bioavailability 的 L1 只包含实际 voter rows；Skin direct 只接受 sensitization/contact-
allergy final outcome；ClinTox 只由冻结 AACT toxicity-failure positives 与 SWEETLEAD/FDA-approved comparators
构造 label，broad Starling toxicity rows 不投票。版本化 source/retrieval contracts 属于 provenance，不是第二套 gold。

当前 Starling random/scaffold 的 frozen label 决策、formal GLM、MiniMol head、Morgan KNN、
MiniMol embedding cosine KNN、MiniMol/cosine agent retrieval、blind 进度、Skin retrieval degradation、
Tier 1+2 final-only 诊断和代码入口统一记录在：

```text
tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md
```

当前 paper-facing valid 默认 prompt profile 分别为 BBB `meaningful_cns_access_v1`、Bioavailability
`f20_evidence_calibrated_v2` 和 Skin `sensitization_aligned_v2`。BBB v2/v3、BBB property-compatible selector、
Skin negative-transfer v3、train-ratio prior 和 matched train-label agent 均为可复现但未 promotion 的 valid-only
历史实验，不得作为默认 pipeline 或据此运行 formal test。当前版本、最佳 valid 条件和 artifact 索引以
`tools/chembl_tool/paper_experiments/RESULTS.md` 顶部的 canonical snapshot 为准。

当前 Starling benchmark 的主要运行与汇总入口：

```text
tools/chembl_tool/common/starling/build_benchmark_datasets.py
tools/chembl_tool/common/starling/build_record_supported_benchmark.py
tools/chembl_tool/tasks/bioavailability_ma/build_canonical_starling_source.py
tools/chembl_tool/paper_experiments/analyze_starling_parent_provenance.py
tools/chembl_tool/paper_experiments/analyze_starling_majority_thresholds.py
tools/chembl_tool/paper_experiments/build_starling_benchmark_indices.py
tools/chembl_tool/paper_experiments/build_minimol_retrieval_features.py
tools/chembl_tool/paper_experiments/starling_benchmark_matrix.py
tools/chembl_tool/paper_experiments/seed_starling_matrix_reuse.py
tools/chembl_tool/paper_experiments/analyze_starling_direct_significance.py
tools/chembl_tool/paper_experiments/analyze_starling_best_agent_baselines.py
tools/chembl_tool/paper_experiments/router_oof/cli.py
tools/chembl_tool/paper_experiments/summarize_minimol_retrieval_agent.py
tools/chembl_tool/paper_experiments/summarize_starling_benchmark.py
tools/chembl_tool/paper_experiments/plot_starling_benchmark_overview.py
tools/chembl_tool/paper_experiments/plot_starling_model_comparison.py
tools/chembl_tool/paper_experiments/watch_glm_tunnel_and_matrix.py
tools/chembl_tool/paper_experiments/plot_starling_with_minimol_agent.py
tools/chembl_tool/paper_experiments/run_minimol_valid_matrix_gpt_oss_120b.py
tools/chembl_tool/paper_experiments/run_assay_retrieval_curve.py
tools/chembl_tool/paper_experiments/build_assay_family_catalog.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_family_curve.py
tools/chembl_tool/paper_experiments/run_conditioned_assay_progressive_curve.py
tools/chembl_tool/paper_experiments/audit_conditioned_assay_prompt_lengths.py
tools/chembl_tool/paper_experiments/plot_assay_retrieval_curve.py
tools/chembl_tool/paper_experiments/summarize_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/analyze_coverage_selector_retrieval_changes.py
tools/chembl_tool/paper_experiments/plot_coverage_selector_llm_matrix.py
tools/chembl_tool/paper_experiments/run_minimol_retrieval_agent_experiment.py
tools/chembl_tool/paper_experiments/run_train_ratio_prior_experiment.py
tools/chembl_tool/paper_experiments/train_ratio_prior_analysis.py
baselines/minimol/run_bioavailability_ma.py --train-all
baselines/minimol/run_train_cv.py
baselines/minimol/run_embedding_knn.py
baselines/conditioned_knn.py
baselines/structure_knn/run.py
```

Conditioned cumulative-family 的 Bioavailability nondirect context overlay 与 ClinTox source-native support
bridge 分别由 `tasks/bioavailability_ma/build_nondirect_assay_context.py` 和
`tasks/clintox/build_flat_assay_support_evidence.py` 构建；两者只生成版本化 source artifacts，不复制 reasoning
runner。完整合同、当前 valid 进度和复现命令统一见 `ASSAY_LEVEL_RETRIEVAL.md`。

`plot_starling_model_comparison.py` 是 GPT-OSS-20B、GPT-OSS-120B、train-label baselines 和后续
ablation 的唯一 Starling 总图入口。新增完整 model/visibility summary 通过可重复的
`--comparison-metrics` 追加；matched method experiment 继续通过可重复的 `--experiment-metrics` 追加。
不得为单个新实验新增独立 overview/bar-chart 模块或正式小图。完整 comparison summary 必须与 reference
使用相同 task/split/subset/sample count 和 baseline；experiment metrics 则必须按 task/split 提供一个与既有
candidate condition 对齐的 anchor row，绘图器会校验 `n` 和 macro-F1 后隐藏重复 anchor，只绘制新增实验行。

`router_oof/` 是与正式 valid/test matrix 隔离的 train-only KNN-vs-direct-agent 实验。它固定 scaffold-or-parent
5-fold OOF、fold-specific heldout-parent filtered index、Morgan `k=3`、gpt-oss-120b
`identity_blind + parent_disjoint` direct agent，并按 BBB/Bioavailability/Skin 分别训练 Logistic/HistGBDT；
v1–v3.1 可部署 router 不得共享 task 参数，也不得把 train fold 加入 `starling_benchmark_matrix.py` 的 evaluation
subset。唯一 shared 参数实验是已冻结、不可部署的 train-only transfer termination diagnosis。
当前 v2 router 不把 fold-dependent absolute reference size 或确定性重复的 k=3 margin/entropy 作为输入；它在
`knn_compact`、`query_knn`、`query_knn_evidence` 三个通用 profile 内做 nested train-only 选择，并分别校准
`P(agent-only-correct)` 与 `P(KNN-only-correct)`，用二者之差作为 routing score。只有 nested OOF paired-bootstrap
promotion gate 同时通过 macro-F1 improvement 和 accuracy guardrail 才部署，否则明确回退 KNN。v1 `router/`
artifact 保留为 historical diagnosis；v2 写入 `router_v2/` 和 `router_features_v2.*`，不得覆盖或混表。
v3 是独立 output-aware post-selector：只训练 KNN/agent disagreement rows，target 为 agent-only-correct 对
KNN-only-correct；保留原始 query/KNN/evidence features，并以 nested OOF 比较 18-feature base、49-feature
evidence 和 70-feature evidence+structured-trace profiles。两个 disagreement directions 使用独立 threshold，
promotion gate 失败时严格 KNN fallback。v3 写入 `post_selector_features_v3.jsonl` 与 `post_selector_v3/`，
不得覆盖 v1/v2；permutation/dropout stability 需要额外 calls，不能从既有 trace 伪造。
v3.1 复用同一 frozen feature rows，但为两个 disagreement direction 分别选择 family/profile/threshold，并使用
fold-heldout sigmoid-calibrated ensemble。每个 direction 必须至少 route 20 个 train rows 且 agent-win precision
的 one-sided 95% Wilson lower bound > 0.5；overall train promotion 以 accuracy paired-bootstrap lower CI > 0
为主 gate。它写入 `post_selector_v31/` 和 `post_selector_valid_result_v31.json`，不得覆盖 v3。Valid receipt 必须
分别标明 train-only promotion 与 held-out valid evidence gate；valid 不改变已冻结 policy。
v3.1 后续只允许两个已冻结的 train-only termination diagnostics：`post_selector_v31_learning_curve/` 固定
direction specs 做 matched-size curve；`post_selector_v31_transfer/` 固定 task-balanced Logistic shared
representation，并保留 task-specific calibration/threshold。当前 transfer gate 已失败，router 主方法线停止；
不得继续根据 valid 搜索新的 shared family/profile，formal test 仍未运行。
协议、命令与 gate 统一记录在 `tools/chembl_tool/paper_experiments/ROUTER_OOF_IMPLEMENTATION_PLAN.md`。

`watch_glm_tunnel_and_matrix.py` 是长 GLM matrix 的可恢复监控入口：检查 `/v1/models`、SSH tunnel 和唯一
launcher，断线时停止当前 process group、重连后依靠 `--skip-existing` 恢复。完成计数必须通过 task prediction、
single/final status、expected group count 和 group status 四层 gate；不能只数 final 文件。该入口不保存密码，
Duo approval 仍由用户完成。

数据构建入口：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.common.starling.build_benchmark_datasets
```

当前 split 使用统一 condition-aware schema。label provenance、source review、parent identity 和冲突记录
保存在同目录 audit artifacts。正式评估前必须针对 valid+test union 的 heldout detailed labels 重建
train-only retrieval index；现有从 full Starling source 构建的 evidence index 不能直接用于当前 benchmark。

旧 `data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl` 及
`data/processed/{Bioavailability_Ma,ClinTox,Skin_Reaction}` 是既有 TDC 实验的历史输入，不再代表上述
三个已迁移 task 的当前 benchmark；旧 ClinTox split 也不代表新的 parent-normalized reconstruction。
ClinTox 当前严格 split 位于 `data/conditioned_benchmark/ClinTox/scaffold/`。历史结果和复现命令可以保留，
但必须明确标注 historical lineage。

## 设计原则

1. 不把 BBB 逻辑写成一次性脚本。BBB 是第一个 task，但工具服务、检索协议、LLM 输入输出格式应能支持后续更多任务。
2. ChEMBL neighbor retrieval 是 pipeline 的 evidence prefetch / context assembly 步骤，不是当前暴露给 LLM 的 function tool，也不是当前 FastAPI service tool。后续 pKa、logD、solubility、toxicity、target affinity、PK property 等模型才按通用 tool contract 接入。
3. 长初始化模型要常驻。慢启动模型和大索引应在服务启动时加载，通过 FastAPI endpoint 调用，避免每个 query 反复初始化。
4. evidence retrieval 只提供证据，不直接替代 reasoning。retrieval payload 必须保留 assay 描述、activity 数值、endpoint 语义、similarity 和不确定性。
5. 默认 production/group-level retrieval 单元仍优先是 molecule-level evidence：先找相似 molecule，再展开
   assay/activity evidence。另有隔离的 Starling assay-level scaling experiment：先按冻结 biological relevance
   选择 cumulative assay prefix、先删除 valid+test parents 的 direct-outcome rows，再从保留的 source records
   检索 query-scaffold-disjoint molecules，并把相同 molecule 跨 assays 合并为一个 flat branch。该实验不得
   改写 production family mapping；协议见
   `tools/chembl_tool/paper_experiments/ASSAY_LEVEL_RETRIEVAL.md`。
   Conditioned cumulative-family assay experiments use `assay_compact.raw_v3`:
   at most three representative record cards per assay×molecule with complete
   raw card fields and support text. They do not apply field-level truncation
   and do not require or call a support-summary model.
   The isolated visible progressive experiment now uses
   `conditioned_assay_progressive_visible.v8`: candidate generation is global
   molecule-similarity retrieval within each cumulative record-family pool,
   without a per-assay neighbor cap. L1 selects at most 10 molecules; later
   families append bounded new-molecule and active-molecule card deltas; all
   prior cards remain visible. Assay identity remains card provenance and a
   diversity tie-break only. Family assignment is record-level: a physical
   assay may contribute cards to several levels, and its earliest level is only
   catalog ordering/coverage metadata, never a visibility or retrieval gate.
   BBB source-purity v5 restricts L1 to current accepted non-prediction voters
   plus records that replay the experimental CNS-access gold contract; predicted
   BBB outcomes and missing/generic proxies move to near-direct, while predicted
   passive-permeability or efflux readouts remain in their mechanism family.
   The v5 row ledger audits all 581,708 source records and rejects cross-family
   efflux/influx precedence violations before an index can be published.
   Its runner and artifacts must not replace the
   cumulative-family or geometric assay-prefix pipelines.
6. LLM reasoning 分为并发证据分支和 final 汇总：single-molecule 分支判断理化性质先验；paper-facing
   group-level 分支按少量、数据源无关的 mechanism family 判断 analog transferability；final-level 汇总所有
   证据。细粒度 `Tier.endpoint_group` 只用于 source-local normalization、检索审计和 legacy native runner，
   不得在新任务中一组对应一个并行 LLM branch。任务扩展规范见 `tools/chembl_tool/tasks/AGENTS.md`。

## 当前常驻工具服务

当前已经实现的通用工具服务入口：

```text
tools/service/app.py
  FastAPI app。注册工具并提供 /health、/tools、/tools/batch、/tools/{tool_name}/invoke、
  /tools/invoke、/tools/{tool_name}。

tools/service/config.py
  服务配置。读取 MolGpKa、bounded batch workers、persistent cache 和 native-thread budget；mmpdb 使用
  当前 Python 环境中已安装的 mmpdblib，不需要源码路径环境变量。

tools/service/registry.py
  ToolRegistry。负责初始化工具、复用共享实例、统一 invoke/batch invoke、persistent cache 和 single-flight。

tools/service/cache.py
  版本化 SQLite/WAL persistent cache、进程内 LRU 和 single-flight 的公共实现。

tools/service/runtime.py
  限制 PyTorch/OpenMP/MKL/OpenBLAS/NumExpr native threads，防止 request-level 并发再嵌套线程膨胀。

tools/service/molgpka_predictor.py
  ResidentMolGpKaPredictor；acid/base weights 每个 service process 只加载一次。

tools/service/schemas.py
  ToolRequest / ToolResponse / ToolError 等统一 schema。

tools/service/errors.py
  统一工具异常。

tools/service/tools/base.py
  BaseTool 抽象。

tools/service/tools/rdkit_properties.py
  molecule_properties v1。计算 RDKit descriptors、MolGpKa pKa/logD、AccFG 顶层 functional groups。

tools/service/tools/properties_compare.py
  properties_compare v1。比较两个分子的 molecule_properties 输出，functional groups 除外。

tools/service/tools/mmp_structure_compare.py
  mmp_structure_compare v1。只比较结构：Morgan Tanimoto、similarity bucket、mmpdb matched-pair transformation、MCS。
```

当前相关测试入口：

```text
tests/service/test_registry.py
tests/service/test_molgpka_predictor.py
tests/service/test_rdkit_properties.py
tests/service/test_properties_compare.py
tests/service/test_mmp_structure_compare.py
```

单进程开发启动命令：

```bash
uvicorn tools.service.app:app --host 127.0.0.1 --port 8765
```

node002 正式高吞吐启动、缓存、batch endpoint、线程预算和滚动切换规范统一维护在：

```text
tools/service/README.md
```

正式 benchmark 使用 32 个 Uvicorn process workers、每进程 8 个 bounded batch workers、每次 native
inference 1 thread，并将版本化 SQLite/WAL cache 放在 node-local `/local/tmp`。harness 将一个 sample 的固定
tool bundle 通过 `/tools/batch` 一次提交；服务按 tool/input/version 做 persistent cache 和 single-flight
去重。该层只消费统一 retrieval payload，不依赖 Morgan、MiniMol、coverage selector 或 future retriever
的内部实现。不得在新的 retrieval 方法里复制 tool-prefetch/cache 逻辑。

当前服务层暂时只冻结三个通用工具：

```text
molecule_properties
properties_compare
mmp_structure_compare
```

之前规划里的 `rdkit_properties`、`ml_pka` 不再作为独立 service tool 暴露；它们已经合并进 `molecule_properties`。`chembl_neighbors` 当前也不是常驻 service tool，也不是 DeepSeek 可调用 tool。ChEMBL neighbor retrieval 仍由各 task 的 `retrieve_neighbors.py` / `run_reasoning_pipeline.py` 在调用 LLM 前预取并注入为 group evidence context；后续需要时再包装成 service tool 或 task endpoint。

## LLM 可见输出约定

工具响应会保留结构化 JSON，供 workflow、debug、缓存和测试使用；最终展示给 LLM 的内容只使用：

```text
ToolResponse.output.text
```

不要把 `raw_features`、`comparisons`、`mcs`、`transformation`、`metadata` 等结构化字段整包塞进 LLM prompt。需要在 prompt 中区分工具时，可以加简短标题，例如：

```text
[molecule_properties]
<output.text>

[properties_compare]
<output.text>

[mmp_structure_compare]
<output.text>
```

所有工具面向 LLM 的文本中，数字最多保留两位小数。None / 缺失 / 不适用值应以自然语言说明，例如 `not applicable`，避免把 Python/JSON 内部表示直接暴露给 LLM。

Evidence library 内部可以保留 `evidence_direction`、`evidence_strength`、`endpoint_group_reason`、
`assay_reason` 等规则派生字段，方便 debug 和审计；这些字段不能发送给 reasoning LLM。所有 task 的
group prompt 必须通过 `tools/chembl_tool/common/evidence_contract.py` 将 ChEMBL、Starling 或其它 source row
转换为 `minimal_evidence.v1`。该 contract 只描述 source、molecule、group、endpoint/measurement、
evidence/context text、可选 role/scope、quality/uncertainty 和 provenance，不包含 label vote、threshold
policy 或 deterministic override。

## OpenAI-compatible / GLM-5.2 适配记录

2026-08-01 起，paper/Starling GLM runner 默认通过本机 SSH tunnel 直连 dgx008 vLLM：

```text
base_url: http://127.0.0.1:50000/v1
model: nvidia/GLM-5.2-NVFP4
api key env: GLM_LOCAL_API_KEY (loopback vLLM 无鉴权时 runner 自动注入非敏感占位值)
reasoning_effort: "" (omit the API parameter; matches the historical LiteLLM runs)
```

先建立 tunnel：

```bash
ssh -fNT parcc-glm
```

旧 Penn LiteLLM 仍可显式作为 fallback：

```bash
--api-key-env GLM_API_KEY \
--base-url https://litellm.parcc.upenn.edu/v1 \
--model zai-org/GLM-5.2-FP8 \
--reasoning-effort ""
```

`zai-org/GLM-5.2-FP8` 是旧 LiteLLM 请求别名；旧 response 和既有 trace 实际均报告
`hosted_vllm/nvidia/GLM-5.2-NVFP4`。不要把旧/新路径描述成 FP8 与 NVFP4 两种模型的比较。
正式 runner 继续沿用历史 `--disable-thinking --reasoning-effort ""`。这里的 `--disable-thinking` 只是不发送
DeepSeek-style `thinking` 参数，空 `reasoning_effort` 使 client 完全省略该 API 参数；它不会关闭 GLM 自己的
reasoning，provider 返回的 `reasoning_content` 或 `reasoning` 仍写入 trace。曾测得的 64/128/256/512
并发高吞吐数字使用了 `reasoning_effort=none`，属于关闭 reasoning 的 endpoint ceiling 诊断，未被采纳为
正式默认，也不能用于估算当前 reasoning-enabled agent pipeline 的加速比例。

OpenAI-compatible response 可能把思考文本放在 `reasoning_content` 或 `reasoning`；共享 client 两者都接受。

### 当前 Conditioned Benchmark 正式运行默认

当前正式 scaffold matrix 读取 `data/conditioned_benchmark/<Task>/scaffold/`，并冻结为：

```text
endpoint: http://127.0.0.1:50000/v1
model: nvidia/GLM-5.2-NVFP4
reasoning: --disable-thinking --reasoning-effort ""（与历史 GLM 设置一致，仍保存 reasoning）
visibility_mode: identity_blind
neighbor_identity_policy: parent_disjoint
run_operational_first: false
endpoint_concurrency_budget: 512
```

这里的 512 是单次正式 launcher 的全局 endpoint request budget，不是允许每一层并行各自再乘 512。
Reasoning-enabled valid 压力运行已证明 512/384 对长 group prompt 不稳定，因此当前默认执行形状为
全局 prompt pool 的 `--parallelism 128`，并且同一 endpoint 同时只启动一个 matrix launcher；任何多 split、
多 task 或多 condition 外层 fan-out 都必须共享这 512 个 slots。512 只表示硬上限，不是推荐并发。

matrix 只使用一个跨 task/condition 的 ready queue；`--parallelism` 是全局 outstanding prompt 上限。
single/group/final branch 共池，final 只在其依赖成功后入队；跨 condition 的 frozen single 依赖按 sample
动态解锁。不得重新引入 condition lane、整批 phase barrier，或通过多个 launcher 绕过全局预算。

新正式矩阵直接从 held-out-filtered index 做 fresh `parent_disjoint` retrieval，不再依赖 operational artifact、
same-parent diff 或 `reuse_plan.json`。`none` 仍必须运行，但其 identity policy 标记为不适用。新数据集先跑 valid
做 pipeline/completeness 检查，冻结设置后再跑 test；不得根据 test 调并发以外的模型、prompt、threshold 或
label policy。既有 operational、deployment-visible、matched-prefetch 和 reuse artifacts 都保留为 historical
lineage，不删除，也不混入新 v4 主结果。

当前代码已经完成以下实现 gate；GPT-OSS-120B v2 valid 已完成。2026-08-10 已首次完成 Bioavailability
full-flat/full-mechanism 及对应 Morgan KNN、MiniMol embedding KNN、MiniMol trained head 的 frozen
scaffold-test。2026-08-13 的 BBB DeepSeek residual-adjudication valid gate 已失败，BBB 方法开发停止且
formal test 不再列为待办；Skin formal test 与 GLM v2 valid/test 仍须等待独立 promotion/artifact gate：

1. runner 默认改为 `identity_blind + parent_disjoint`，并允许该组合 fresh-run；
2. parent-disjoint fresh-run 不要求 operational `reuse_plan.json`，且输出到独立
   `runs_identity_blind_parent_disjoint/`；
3. 默认 launcher 使用单一 128-slot global prompt pool，并阻止 `parallelism` 超过全局 512 budget；
4. manifest 显式保存 endpoint、served model、reasoning、visibility、identity policy、effective concurrency、
   evaluation subset 和 `operational_staging_used=false`；
5. valid/test 分区、manifest 和 held-out index 使用同一 valid+test union；完整矩阵仍必须通过
   `n_failed_runs=0`、identity leak=0、parent conflict=0 和 held-out overlap=0。

GLM 对 tool choice 和长 structured output 的遵循可能不稳定。所有 task 统一通过
`tools/chembl_tool/common/reasoning_validation.py` 检查必需 JSON 字段和允许值；当前默认最多 4 次总尝试，
每次 validation error 和 attempt count 都必须进入 trace。
该 validation 层不能修改有效 prediction，也不能实现 task-specific label policy。需要更长 final 输出时，
显式提高 `--max-tokens`；不要用 postprocess 修补 benchmark label。

## ChEMBL task workflow 目录

具体任务放在：

```text
tools/chembl_tool/tasks/<task_name>/
```

每个 task 目录只维护 task-specific 规则、scoring、endpoint assignment、默认路径、输出文件名和
reasoning prompt / final schema。跨 task 共享的 workflow 不放在 `tasks/` 目录下，而是放在：

```text
tools/chembl_tool/common/task_workflows/
  screen_assays.py
  rescore_outputs.py
  summarize_outputs.py
  assay_report.py
  evidence_library.py
  distance_assay_manifest.py
  retrieve_neighbors.py
  chembl_exact_context.py
  reasoning_batch.py
  global_prompt_pool.py
  reasoning_stage_runtime.py

tools/chembl_tool/common/evidence_contract.py
tools/chembl_tool/common/identity_blind.py
tools/chembl_tool/common/json_utils.py
tools/chembl_tool/common/openai_reasoning_client.py
tools/chembl_tool/common/reasoning_calls.py
tools/chembl_tool/common/reasoning_payload.py
tools/chembl_tool/common/reasoning_validation.py
tools/chembl_tool/common/final_decision_prior.py
tools/chembl_tool/common/prompt_profile.py
tools/chembl_tool/common/molecule_identity.py
tools/chembl_tool/common/retrieval_policy.py
tools/chembl_tool/common/neighbor_selection.py
tools/chembl_tool/common/coverage_reasoning.py
tools/chembl_tool/common/retrieval_ablation.py
tools/chembl_tool/common/retrieval_replay.py
tools/chembl_tool/common/experiment_retrieval.py
tools/chembl_tool/common/evidence_distance.py
tools/chembl_tool/common/distance_index.py
tools/chembl_tool/common/distance_retrieval.py
tools/chembl_tool/common/scalar_knn.py
tools/chembl_tool/common/starling/evidence_library.py
tools/chembl_tool/common/starling/assay_catalog.py
tools/chembl_tool/common/starling/benchmark_dataset.py
tools/chembl_tool/common/starling/build_benchmark_datasets.py
tools/chembl_tool/common/starling/heldout_index.py
tools/chembl_tool/common/assay_retrieval.py
```

这些公共 workflow 的职责：

```text
screen_assays.py
  扫描 ChEMBL assays，调用 task-specific scoring.scored_row，导出 assay candidates、
  activity evidence 和 report。

rescore_outputs.py
  对已有 candidate CSV 重打分，适合规则变严、重排 tier 或调整阈值；如果规则变宽，
  需要重新跑 screen_assays.py。支持 `--only-filter-activities`，用于 candidate 已经确定、
  只需要按现有 candidate 重新过滤 activity evidence 的场景。

summarize_outputs.py / assay_report.py
  生成 health check 和 Markdown report。

evidence_library.py
  从 assay candidates + activity evidence 构建 molecule-level evidence rows、RDKit fingerprint
  和 neighbor index。task 只配置输入路径、输出文件名、index version 和 assign_endpoint_group。
  支持 `--workers` 并行标准化 molecule / 构建 index，长任务进度会打印 elapsed、rate 和 ETA。

distance_assay_manifest.py
  E12 的通用 ChEMBL assay 扫描和冻结 manifest workflow。task-local classifier 只决定 family、scope、quality
  和 mapping reason；公共实现负责 source manifest、纳入/排除审计、activity export 和 graph/config provenance。

retrieve_neighbors.py
  source-local / legacy native retrieval：对细粒度 Tier.endpoint_group 做 analog retrieval，包含 molecule
  identity policy、Tanimoto ranking、similarity threshold、similarity bucket 和 JSONL batch retrieval CLI。
  Paper-facing direct/flat/mechanism 视图由 experiment_retrieval.py 在其上按 mechanism family 组装。

chembl_exact_context.py
  可选 exact-query ChEMBL context 和 shared-assay enrichment。默认 benchmark 不开启，
  避免 prospective evaluation 数据泄漏。

reasoning_batch.py
  多分子 batch 的参数、manifest、日志、结果采集和 predictions/metrics/report 公共实现。实际 prompt 调度
  统一委托给 global prompt pool。支持 `--groups` 透传给 task pipeline，用于 targeted
  group smoke test；支持 `--final-only-source-batch` 复用已有 single/group artifacts，
  并用 `--final-only-groups` 在重新汇总 final 前严格裁剪可见 group（不能用 `--groups`
  代替该过滤）；metrics 包含 positive-class precision/recall/F1、confusion matrix 和
  prediction distribution。

global_prompt_pool.py / reasoning_stage_runtime.py
  前者提供跨 task/condition 的唯一 ready queue 和全局 prompt 并发上限；后者提供 single/group/final stage
  checkpoint、依赖解锁、原子 artifact 写入和断点恢复。成功 prerequisite 改写会使旧 final/trace 失效；final
  只在 single 和精确 expected group set 全部成功后执行。五个现有 batch wrapper 都必须能接受公共
  `--prepare-only` seed 命令；尚未迁移到共享 retrieval contract 的 DILI 只允许 native/operational/standard
  默认组合，公共 parser 会拒绝伪装成 parent-disjoint 或 coverage ablation。
  Pool 内部 sample key 必须使用绝对 batch directory 加 query index；`batch_id` 只在单个 batch root 内唯一，
  不能作为跨 fold/跨 root 的全局 key。single dependency 的 source key 必须使用同一绝对目录合同。

evidence_contract.py
  `minimal_evidence.v1` 的唯一 schema/normalizer。旧 ChEMBL-like row 可以在 prompt-time 动态转换，
  新 source builder 应在建库时调用 `attach_minimal_evidence()`。该模块只描述 evidence，不预测 label。

identity_blind.py
  统一实现 identity redaction、harness-prefetched tool evidence 和 matched-prefetch tool replay。Paper task
  runner 只能通过该模块选择 visibility/tool-execution contract，不能在 task 内复制脱敏或 replay 逻辑。

reasoning_calls.py / json_utils.py
  共享 single/group branch 调用、冻结 single analysis 复用、group payload transport bound、JSON 提取以及
  JSON/JSONL/trace 同目录原子发布工具。
  Transport bound 只能确定性采样超大 evidence rows，不能改变 evidence source、label policy 或 inference setting。

reasoning_payload.py
  五个 task pipeline 共用的 LLM query identity surface、exact-match/shared-assay 清洗、单条 JSONL 读取和显式
  env-file 解析，以及 single/group/final trace 序列化。task 只绑定自身 prediction field，不得复制并逐 task
  漂移这些 frozen 字段合同。

molecule_identity.py / retrieval_policy.py
  数据源和任务无关的 whole-record、RDKit fragment/molecular-parent 和 mixture-component 标准化及
  neighbor exclusion policy。Operational 保留 same-parent evidence；parent_disjoint 额外排除并在既有
  similarity threshold 内回填。这里的 parent 不是药理学 active moiety，也不推断 prodrug/metabolite 关系。

neighbor_selection.py / coverage_reasoning.py
  两个正交的可插拔 contract：前者只从已通过 similarity、identity 和 evidence gate 的候选中选择 neighbor；
  后者只控制 LLM-visible analog-set context。`standard` context 是严格 no-op；`coverage_aware` 以匿名统计提供
  query Morgan feature/atom-environment coverage、逐 neighbor marginal coverage 和 region size，不改变
  retrieval.json、neighbor 集合、task JSON schema 或 tool-prefetch/cache。Coverage context 不暴露 SMILES、
  fingerprint bit ID、元素标签或分子身份，且当前只支持 Morgan retrieval feature。Visible-only
  `coverage_mmp_ledger` 是独立 opt-in profile：它复用常驻 `mmp_structure_compare` 的逐 neighbor MCS/MMP 文本，
  再以 Morgan marginal feature 统计组织 rank-by-rank 的互补、冗余和未覆盖区域 ledger；不重复实现 MCS/MMP，
  不修改 raw retrieval、neighbor set、task JSON schema 或默认 `standard` / `coverage_aware` 路径。Morgan feature
  coverage 不能解释为 atom coverage；没有 matched-pair transformation 时具体 fragment 对应必须标为 unresolved。

retrieval_ablation.py
  计算 LLM-visible retrieval/group input hash，支持完整 sample 和独立 mechanism branch 的确定性复用，
  并记录 provenance。该模块不能改变 evidence、阈值或 prediction。

retrieval_replay.py
  按冻结的 retrieval/tool artifacts 做 matched-prefetch replay，检查 sample coverage 和输入一致性；仅用于
  visibility/tool-execution 控制，不替代 agentic deployment-visible 主实验。

experiment_retrieval.py
  将 source-local endpoint groups 映射到 task 声明的 direct/mechanism families；确保 full_flat 与
  full_mechanism 使用同一 evidence union，并只改变 reasoning organization。

evidence_distance.py / distance_index.py / distance_retrieval.py
  E12 独立代码线，已实现 D-root/C-family tree contract：每个 C family 恰有一个聚合 H1 child，每个 H1 至多
  一个 optional H2 child。每个 C/H1/H2 tree node 独立最多取 3 个 neighbors，并共享同一 similarity threshold；
  child 内多个 target/measurement families 共享 node budget。公共 builder/retrieval/audit 已能物化
  D、D+C、D+C+H1、D+C+H1+H2 的 flat/mechanism views，并验证 base parity、nestedness、branch stability
  和 node budget；这仍是与旧 paper matrix 隔离的 engineering line，尚未注册为 paper LLM condition。
  graph hop validation 之外还必须执行 same-molecule causal continuity audit：若 assay molecule 只改变 system
  state，而 downstream endpoint 实际作用于另一个未观测 substrate，则标为 `requires_query_role` 或
  `context_only`，不得进入主 H1/H2。所有 future task 发布前必须完整声明 `FamilySelfRelevanceAudit` 并通过
  `validate_self_relevance_audit(..., require_publishable=True)`；prompt 不能替代缺失的 substrate/target role。
  这些模块不得注册进旧 `EXPERIMENT_MODES`，也不得改变旧 paper matrix 或旧 index。

scalar_knn.py
  共享标量 KNN baseline 实现；当前用于 Bioavailability numeric direct-F 对照，必须和 LLM agent 条件分开报告。

openai_reasoning_client.py
  所有 task 共享的 OpenAI-compatible JSON completion、bounded tool-call loop、常驻工具服务调用和 trace
  serialization。provider/model/base URL 由运行参数配置；task 文件不复制 client runtime。

reasoning_validation.py
  为 single/group/final branch 提供通用必需字段、允许值和必需工具结果验证；single/group 从发送给模型的
  `required_json_schema` 自动提取顶层 `a | b | c` 枚举并验证，非法近义值触发同设置 retry。当前默认最多
  4 次总尝试。
  不能在 response 有效时改写 prediction，也不能通过 retry 删除 evidence 或改变 inference setting。

final_decision_prior.py
  提供显式 opt-in 的 final-stage decision profile。默认 `standard` 严格 no-op；
  `train_ratio_tiebreak_v1` 只允许 BBB/Bio final-only valid 诊断在真正 evidence tie 时使用 frozen train majority，
  并要求 `evidence_state`、boolean `prior_used` 和 prediction 通过 cross-field validation。已失败的 BBB
  `direct_anchored_residual_v1` / `direct_override_recheck_v1` 只保留冻结 artifacts 和 no-go 结论；专用实现已删除，
  不得启动 formal test。任何 profile 都不得变成 batch quota。

prompt_profile.py
  只负责 task prompt profile 的 manifest provenance、历史缺省映射和 branch-reuse 一致性 gate。具体 task
  instructions/schema 继续放在各 task 的 `prompt_profiles.py`，不得把 task 语义塞进共享模块，也不得在 runner
  中复制 profile 解析逻辑。

common/starling/evidence_library.py
  profile-driven parquet ingestion。profile 只声明 SMILES、endpoint、value、unit、context、scope、role
  和 group 映射；公共实现负责 canonicalization、缺失 SMILES 统计、molecule-level 聚合、representative
  examples、provenance 和 neighbor-index 兼容 evidence row。

common/starling/benchmark_dataset.py
  Starling direct gold-label 构建公共引擎：source-row 决策、RDKit fragment-parent 聚合、70% record-majority、
  random/scaffold train/valid/test split、audit artifact 和 summary。task-specific threshold、
  population/scope/unit/free-text 规则只能由 task adapter 提供。

common/starling/build_benchmark_datasets.py
  Historical TDC-compatible 三 task CLI；读取冻结 source revision/local parquet，生成
  `data/processed_starling/<Task>/{random,scaffold}/` 及 task/root 汇总。BBB 新主线不使用该入口。

common/starling/build_record_supported_benchmark.py
  从冻结 binary parents 构造 scaffold-only quality split；默认 `record_supported_v2`，同时向 BBB 新 builder
  提供可配置 lineage/seed 的共享分配。lexicographic MILP 先最小化
  held-out singleton 和 valid/test imbalance，再优化 label balance，最后才最大化第一版 valid 复用。输出只含
  发生变化的 split/audit，根级 source rejection/conflict provenance 继续读取第一版目录，避免重复数据。

common/starling/publish_conditioned_benchmark.py / conditioned_benchmark.py
  当前四任务唯一 publication/path 入口；source-specific builders 先写入 `data/.build/conditioned_benchmark_sources/`，
  publisher 再统一 schema、paths、hash receipt。版本化 BBB build/migration scripts 只作 source QA provenance，
  不得成为 runner input。

common/starling/heldout_index.py
  从 full-source Starling evidence rows 中按 `rdkit_fragment_parent.v1` 删除 valid+test parents，重建
  train/evaluation 隔离的 retrieval index，并写 source/exclusion SHA-256、排除数量和 zero-overlap audit。
  构建时重算并校验 held-out parent key；无法解析 parent 的 source evidence row 保守排除。
```

典型 task wrapper 文件：

```text
screen_assays.py
rescore_outputs.py
summarize_outputs.py
report.py
build_evidence_library.py
retrieve_neighbors.py
chembl_exact_context.py
run_reasoning_pipeline.py
run_reasoning_batch.py
```

这些 wrapper 应该保持很薄，只配置 task-specific 参数；不要在多个 task 下复制公共实现。

通用命令模板：

```bash
# 全量扫描 ChEMBL assays
python -m tools.chembl_tool.tasks.<task_name>.screen_assays \
  --chembl-sqlite tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --min-score 40 \
  --progress-every 10000 \
  --export-activities

# 对已有候选重打分
python -m tools.chembl_tool.tasks.<task_name>.rescore_outputs \
  --in-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version> \
  --min-score 40 \
  --filter-activities

# candidate 已经确定时，只重新过滤 activity evidence
python -m tools.chembl_tool.tasks.<task_name>.rescore_outputs \
  --in-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/raw \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version> \
  --only-filter-activities

# 生成 health check
python -m tools.chembl_tool.tasks.<task_name>.summarize_outputs \
  --out-dir outputs/chembl_tool/tasks/<task_name>/assay_screening/<version>

# 构建 evidence library 和 neighbor index
python -m tools.chembl_tool.tasks.<task_name>.build_evidence_library \
  --workers 128 \
  --progress-every 50000

# 检索 analog neighbors
python -m tools.chembl_tool.tasks.<task_name>.retrieve_neighbors \
  --query-smiles '<SMILES>' \
  --top-k-per-group 3 \
  --min-similarity 0.3

# 批量 reasoning
python -m tools.chembl_tool.tasks.<task_name>.run_reasoning_batch \
  --input-jsonl <input.jsonl> \
  --parallelism 1 \
  --batch-id <batch_id>
```

需要只跑少数 endpoint groups 做 debug / smoke 时，给 pipeline 或 batch 加：

```bash
--groups "Tier 3.some_endpoint_group" "Tier 4.another_endpoint_group"
```

断点续跑统一使用：

```bash
python -m tools.chembl_tool.tasks.<task_name>.run_reasoning_batch \
  --input-jsonl <input.jsonl> \
  --batch-id <batch_id> \
  --skip-existing
```

`--skip-existing` 会跳过已经存在
`reasoning/batches/<batch_id>/runs/<batch_id>_idxNNNNN/final_reasoning_output.json`
的 molecule；已有 stdout/stderr log 不覆盖，没有 final 输出的 partial run 会重新运行。

输出目录统一为：

```text
outputs/chembl_tool/tasks/<task_name>/
  assay_screening/
  evidence_library/
  reasoning/
    single_runs/
    batches/
```

查看最终 paper trace：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

Viewer 默认注册 Starling random、Starling scaffold 和历史 TDC test。新 v4 dataset 默认扫描
`runs_identity_blind_parent_disjoint/`；历史 dataset 继续扫描 identity-blind、matched-prefetch、
deployment-visible 和 deployment-visible parent-disjoint 四个 legacy paper roots。页面按样本展示
single-molecule、mechanism-family/flat/direct 和 final stages，并递归展示
通用 JSON、工具调用、retrieval evidence 和 provenance。旧 task-specific reasoning output 不再支持；
需要其它 dataset 时可在端口后显式追加 trace root。

## Paper experiment split 与可视化入口

冻结论文矩阵的详细操作规范位于 `tools/chembl_tool/paper_experiments/AGENTS.md`。默认不加
`--split` 时使用 test 并写入 `outputs/paper/molecular_evidence_agent/`；validation 诊断重跑统一加
`--split valid`，产物隔离写入 `outputs/paper/molecular_evidence_agent_valid/`。可复用入口包括：

这里的既有 `test` / `valid` 和 2026-07-23 frozen results 来自旧 TDC lineage，应作为历史结果保留；
`--split test|valid` 目前不能解释为 Starling 的 `random|scaffold`。新 Starling 正式实验必须显式选择
`data/processed_starling/<Task>/random/test.jsonl` 或 `scaffold/test.jsonl`，使用相应 train-only
retrieval index，并写入与 TDC、另一种 Starling split 都隔离的新 output root/batch ID。完成输入接线、
test-parent exclusion 和 zero-overlap audit 前，不得把现有 paper 指标改称 Starling 结果。

Bioavailability `record_supported_v2`、Skin `record_supported_v2` 与 BBB
`experimental_meaningful_cns_access_v3` 后续 paper-facing
structural-analog 主结果默认使用
`identity_blind + parent_disjoint` fresh-run；不再先跑 operational，也不要求 operational diff/reuse plan。
旧 lineage 的 operational -> parent-disjoint 流程及 same-parent 暴露统计只作为 historical sensitivity
artifact 保留。若将来需要 operational 对照，必须作为显式 opt-in ablation 使用独立 root，不能成为主矩阵依赖。

新数据集的 Starling 主入口仍是
`tools.chembl_tool.paper_experiments.starling_benchmark_matrix`，正式默认必须解析为
`--visibility-mode identity_blind --neighbor-identity-policy parent_disjoint`，输出到各 lineage root 的
`runs_identity_blind_parent_disjoint/`。每个 split 必须覆盖完整 condition/sample 集并通过 failure、
query-SMILES leak、visibility-contract、parent identity 和 held-out overlap audit。旧 `identity_blind + operational`
结果仍是 historical supplemental control，不得冒充当前主结果。

2026-08-04 第一版 `record_agreement70_split811_v1` scaffold-valid 已完成 GLM、GPT-OSS-20B/120B 三套
22-condition blind matrix；GLM 为
6887/6887 严格成功，两个 GPT 各有一个不可修复 context-limit sample 并按预定 policy 计错。GPT 两套
deployment-visible+parent-disjoint 补充矩阵也已完成；GLM visible 同合同矩阵同样达到 6887/6887 严格成功，
并已加入 canonical blind+visible 总图。
Random-valid GLM 仍有两个 Bioavailability ChEMBL full sample-condition 未通过严格 gate。完整路径、指标、
failure policy 和 coverage context/MMP-ledger 结果见
`tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`。

```bash
python -m tools.chembl_tool.paper_experiments.molecular_evidence_agent --split valid ...
python -m tools.chembl_tool.paper_experiments.summarize_results --split valid
python -m tools.chembl_tool.paper_experiments.audit_prefetch_contract --split valid
python -m tools.chembl_tool.paper_experiments.parent_disjoint_ablation --split valid --materialize
python -m tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview.svg \
  --png-output outputs/paper/molecular_evidence_agent_valid/analysis/figures/retrieval_claims_overview_highres.png \
  --data-split valid

python -m tools.chembl_tool.paper_experiments.analyze_coverage_performance \
  --split valid \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis

python -m tools.chembl_tool.paper_experiments.plot_coverage_performance \
  --analysis-dir outputs/paper/molecular_evidence_agent_valid/analysis \
  --output outputs/paper/molecular_evidence_agent_valid/analysis/figures/coverage_performance_relationship.svg \
  --data-split valid
```

`summarize_parent_disjoint_results.py` 目前通过显式 `--operational-root`、`--parent-disjoint-root` 和
`--output-dir` 切换 split，详细 valid 命令见 paper-experiments 目录文档。

2026-07-23 的 valid 矩阵已扩展完成：三套 visibility/tool-execution 制度各 26 个条件、2,713 个
sample-condition 且 0 失败；prefetch audit 为 2,713/2,713；parent-disjoint 为 22 个条件、
2,275 个 sample-condition 且 0 失败。Test 的 identity-blind 和 deployment-visible 各 26 个条件，
matched-prefetch 仍为原 21 个条件。实测结果见 `tools/chembl_tool/paper_experiments/RESULTS.md`。

旧 TDC performance overview 仍以 `plot_retrieval_claims_overview.py` 为唯一模板；当前 v4 的 model、visibility、
baseline 和 matched ablation 总图统一由 `plot_starling_model_comparison.py` 生成。Coverage 与性能增幅的旧
诊断图使用 `plot_coverage_performance.py`。所有入口复用 `paper_figure_style.py`，每个正式 figures 目录只
保留 canonical SVG 和一份高分辨率 PNG，不保留 preview、QA 或一次性 overview。

## MiniMol baseline（历史结果与当前 v4 scaffold-valid 边界）

MiniMol baseline 代码放在：

```text
baselines/minimol/
  run_bioavailability_ma.py
  run_embedding_knn.py
  run_direct_gpu_sweep.sh
  run_hparam_sweep.py
```

`run_bioavailability_ma.py` 名字保留自第一次 Bioavailability_Ma 实验，但实际是通用 JSONL
二分类 runner。输入 split 约定：

```text
train.jsonl / valid.jsonl / test.jsonl
字段:
  drug: SMILES
  Y: 0/1 label
```

下面列出的旧命令、sweep 和指标使用历史 TDC 或 strict-conflict Starling split。当前 baseline 统一读取
`data/conditioned_benchmark/<Task>/scaffold/`，并按当前 scaffold condition-aware cohort 运行 MiniMol head、Morgan KNN
和 MiniMol embedding KNN。历史 molecule-only baseline 不得与当前 cohort 混表。该诊断不读取或调参于
test；正式 test 前仍须冻结所有设置。

上一版 strict-conflict formal Starling baseline 使用 `--train-all`：
不读取 `valid.jsonl`，每个
ensemble member 在全部 `train.jsonl` 上训练固定 epoch，并用冻结的 `threshold=0.5` 评估 test。
这种模式不得进行 test-selected early stopping 或 threshold tuning；输出中的 validation metrics 为 null。
这些历史 random/scaffold 结果和 output roots 见
`tools/chembl_tool/paper_experiments/STARLING_BENCHMARK_RESULTS.md`；下文单列的 sweep 数值仍全部是
旧 TDC lineage。

MiniMol embedding retrieval-only 对照复用上述正式 baseline 已保存、且与 split 逐行一致的
`embeddings/{train,test}.pt`，L2 normalize 后按 cosine similarity 取同 split train 中的 top-3 molecules。
它与 Morgan KNN 一样使用未加权多数票，score 为正类邻居比例；不训练新 head，也不需要重新占用 GPU。
入口与输出分别为：

```text
baselines/minimol/run_embedding_knn.py
outputs/baselines/minimol_embedding_knn_starling/<Task>/<random|scaffold>/
```

MiniMol feature agent ablation 与上述 label-vote KNN 不同：它只把 agent pipeline 的 neighbor
ranking 从 Morgan/Tanimoto 换成 L2-normalized MiniMol/cosine，保持 evidence source、top-k、GLM
和 inference settings 不变。当前 v4 完整 random/scaffold fresh parent-disjoint -> paired summary
-> figure 的可恢复入口为：

```bash
python -m tools.chembl_tool.paper_experiments.run_minimol_retrieval_agent_experiment
```

MiniMol agent retrieval 是另一条独立实验线：它不从 gold train label 做 KNN vote，而是把 ChEMBL /
Starling evidence index 的 Morgan/Tanimoto neighbor ranking 替换为 MiniMol v1 embedding cosine，
然后把检索到的 evidence 送入冻结 GLM agent pipeline。它保持 source/group/top-k/数值 min-similarity、
prompt/tool/model 和 fresh parent-disjoint contract 不变，并写入独立 output root。正式入口、feature
store contract、checkpoint/cache parity、test-parent audit 和完整命令见
`tools/chembl_tool/paper_experiments/AGENTS.md`；不得把这条 agent ablation 与
`baselines/minimol/run_embedding_knn.py` 的 label-vote baseline 混为一项。

运行环境和实现注意事项：

```text
conda env: intern
MiniMol 源码参考: /data1/tianang/Projects/minimol

实际运行优先使用 intern 环境已安装的 minimol 包。源码目录中的
minimol/ckpts/minimol_v1/state_dict.pth 当前是 Git LFS pointer，不是可直接 torch.load 的权重。

共享 `baselines/minimol/embedding_runtime.py` 做两个兼容 patch：
  1. Graphium CPU/fake-graph featurization 默认 float16 会触发 scipy.sparse dtype 错误，
     runner 在进程内强制用 float32 adjacency/pyg graph。
  2. MiniMol checkpoint 早于 PyTorch 2.6 weights_only=True 默认值，初始化 MiniMol 时临时
     以 weights_only=False 调用 torch.load。

MiniMol featurization 设置 featurization_n_jobs=1，避免 joblib 子进程丢失上述进程内 patch。
```

评估口径：

```text
MiniMol embeddings + leaderboard-style TaskHead。
每个 ensemble member 只用 train 训练，用 valid BCE loss 选 best epoch。
默认 ensemble_size=5, epochs=25, threshold=0.5。
accuracy / macro-F1 用 threshold=0.5；AUROC 用 probability score。
valid-tuned threshold 指标也会写入 metrics.json，但主报告使用 fixed 0.5。
```

单任务 baseline 命令模板：

```bash
/data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/processed/<TaskName> \
  --output-dir outputs/baselines/minimol/<task_name>
```

BBB_Martins 使用 MiniMol 原 `SWEEP_RESULTS['bbb_martins']` 超参：

```bash
/data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/processed/BBB_Martins \
  --output-dir outputs/baselines/minimol/bbb_martins \
  --hidden-dim 2048 \
  --depth 3 \
  --lr 0.0001
```

需要 GPU 状态或指定 GPU 时，必须在 sandbox 外运行；sandbox 内可能看不到 NVML / CUDA，
导致 runner 退回 CPU。可靠做法是直接用 shell 显式绑定 GPU：

```bash
env CUDA_VISIBLE_DEVICES=4 /data1/tianang/anaconda3/condabin/conda run -n intern python -m baselines.minimol.run_bioavailability_ma \
  --data-dir data/processed/ClinTox \
  --output-dir outputs/baselines/minimol/clintox
```

ClinTox / Skin_Reaction 不在 MiniMol 原 `SWEEP_RESULTS` 表中。当前对这两个 task 的超参搜索使用
MiniMol ADMET sweep 表里出现过的 11 个唯一 head 配置：

```text
(hidden_dim, depth, lr)
(512, 3, 0.0001)
(512, 3, 0.0003)
(512, 4, 0.0003)
(512, 4, 0.0005)
(1024, 3, 0.0003)
(1024, 3, 0.0005)
(1024, 4, 0.0001)
(1024, 4, 0.0005)
(2048, 3, 0.0001)
(2048, 4, 0.0003)
(2048, 4, 0.0005)
```

GPU sweep 的可靠入口是直接 shell 脚本；它会复用已有 embedding cache，并用 `env CUDA_VISIBLE_DEVICES=<gpu>`
直接启动每个训练 job：

```bash
baselines/minimol/run_direct_gpu_sweep.sh \
  clintox \
  data/processed/ClinTox \
  outputs/baselines/minimol/clintox/embeddings \
  outputs/baselines/minimol_sweeps_gpu \
  4,5,6,7

baselines/minimol/run_direct_gpu_sweep.sh \
  skin_reaction \
  data/processed/Skin_Reaction \
  outputs/baselines/minimol/skin_reaction/embeddings \
  outputs/baselines/minimol_sweeps_gpu \
  4,5,6,7
```

`run_hparam_sweep.py` 是 stdlib Python launcher，但在当前环境里嵌套 `conda run` 时曾出现 CUDA
不可见 / 退回 CPU 的情况；需要 GPU sweep 时优先用 `run_direct_gpu_sweep.sh`。

当前 MiniMol baseline / sweep 结果：

```text
Bioavailability_Ma, fixed h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/bioavailability_ma/
  test macro-F1 0.5674, accuracy 0.7813, AUROC 0.6721
  MiniMol paper/README reports Bioavailability Ma AUROC 0.689 +/- 0.020, so this is close.

BBB_Martins, MiniMol sweep config h=2048 d=3 lr=0.0001:
  output: outputs/baselines/minimol/bbb_martins/
  test macro-F1 0.8186, accuracy 0.8878, AUROC 0.9322

ClinTox, default h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/clintox/
  test macro-F1 0.5651, accuracy 0.9301, AUROC 0.6770

ClinTox, 11-config GPU sweep selected by valid AUROC:
  summary: outputs/baselines/minimol_sweeps_gpu/clintox/sweep_summary.json
  selected h=512 d=3 lr=0.0001
  valid AUROC 0.6969
  test macro-F1 0.5745, accuracy 0.9371, AUROC 0.6347
  Note: default h=512 d=3 lr=0.0003 has higher observed test AUROC 0.6770; do not use test to select config.

Skin_Reaction, default h=512 d=3 lr=0.0003:
  output: outputs/baselines/minimol/skin_reaction/
  test macro-F1 0.4864, accuracy 0.5854, AUROC 0.5872

Skin_Reaction, 11-config GPU sweep selected by valid AUROC:
  summary: outputs/baselines/minimol_sweeps_gpu/skin_reaction/sweep_summary.json
  selected h=1024 d=4 lr=0.0001
  valid AUROC 0.7475
  test macro-F1 0.5795, accuracy 0.6098, AUROC 0.6055
```

## ChEMBL assay activity transfer benchmark

这个独立 benchmark 用来研究：在同一个 ChEMBL assay endpoint 中，只根据两个分子的结构相似度，
能否判断 activity 是否可以从 neighbor transfer 到 query。第一版不调用 LLM，只建立
Tanimoto threshold baseline，作为后续 DeepSeek / 其他 LLM assay-transfer 推理的最低对照。

代码入口：

```text
tools/chembl_tool/activity_transfer_benchmark/
  __init__.py
  analyze_mcs_results.py
  benchmark_mcs_runtime.py
  build_llm_eval_set.py
  build_task_llm_eval_set.py
  plot_llm_run_comparison.py
  run_benchmark.py
  run_llm_benchmark.py
  run_task_assay_benchmark.py
```

`run_benchmark.py` 的功能：

```text
1. 从 tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db 读取 ChEMBL activities。
2. 连续值主分析只使用 pchembl_value，筛选 standard_relation='='、standard_flag=1，
   默认排除 data_validity_comment 非空和 potential_duplicate=1 的记录。
3. 在同一个 assay_id + standard_type 内聚合同一 molecule 的重复 pChEMBL 均值。
4. 从 tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz 读取 Morgan fingerprint。
5. 在每个 assay endpoint 内采样 molecule pairs，计算 Tanimoto 和 |delta pChEMBL|。
6. 标签规则：|delta pChEMBL| <= 0.5 为 similar；>= 1.0 为 different；中间为 ambiguous。
7. 扫描 Tanimoto threshold，输出 accuracy、macro-F1、balanced accuracy、precision/recall。
8. 计算 assay-specific enrichment：每个 assay endpoint 内先算随机 pair 背景率，
   再比较各 similarity bucket 的 similar-rate lift、fold lift 和 median-delta reduction。
9. 可选 dynamic range filter：按 molecule-level pChEMBL range 和 IQR 过滤低信息量 assay endpoint。
10. 辅助分析保守处理 binary activity_comment，只映射明确 active / inactive 类 comment。
11. 生成 TSV/GZ 数据、metrics、SVG 图表和中文 report。
```

`benchmark_mcs_runtime.py` 的功能：

```text
从 continuous_pairs.tsv.gz 按 similarity bucket 抽样 pair，用 RDKit FindMCS 计算
MCS atom coverage，并在多进程下估算全量 pair 的 MCS 计算耗时。worker 会把
OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / RDKIT_NUM_THREADS 等设为 1，
避免 RDKit 或底层库内部线程和外层进程并行互相争抢。长任务会向 stderr 输出进度：
completed、rate、elapsed、ETA 和 timeout 数。
全量 MCS 应使用 `--full-scan` 流式读取和写出结果，避免把全部 pair、task 和 result 都留在内存中。
如果 RDKit FindMCS 在最后少数 pair 上不返回，进程可能卡在 tail pending futures；
此时先终止卡住进程，保留 `.tmp`，再用 `--finalize-existing` 从已有结果生成 summary/report
和 `missing_result_indices.tsv`。
```

`analyze_mcs_results.py` 的功能：

```text
读取全量或 partial MCS TSV，排除 ambiguous label，扫描 mean MCS coverage threshold，
并在同一批 observed pair 上重新扫描 Tanimoto threshold，输出 threshold metrics、
MCS coverage bucket summary、Tanimoto x MCS heatmap、SVG 图表和中文报告。
```

`build_llm_eval_set.py` 的功能：

```text
从 dynamic_v1 continuous_pairs.tsv.gz 中抽取 LLM 小规模评估集。默认读取已有 MCS full-scan
partial 结果，只保留有 observed MCS 的 non-ambiguous pairs，并按 label x Tanimoto bucket
分层抽样。默认输出 3,000 pairs，similar/different 各 1,500，每个 similarity bucket 各 500。
输出 JSONL/TSV、summary.json 和中文 report，供 LLM benchmark 复用。
```

`run_llm_benchmark.py` 的功能：

```text
用 OpenAI-compatible endpoint 跑 assay activity transfer LLM benchmark。默认模型为本地 vLLM
host 的 gpt-oss-120b，也可跑 DeepSeek/OpenAI-compatible hosted endpoint；默认输入
dynamic_v1_llm_3k/eval_pairs.jsonl。prompt 隐藏 query pChEMBL，只暴露 reference molecule 的
pChEMBL、assay context、Tanimoto、bucket 和 MCS coverage。可选调用当前 tool server 中的
mmp_structure_compare / properties_compare；输出 per-sample run JSON、predictions.jsonl、
metrics.json、中文 report、model-vs-baseline SVG 图和 trace_viewer 可读的 trace_messages.jsonl。
当前也支持 HF prompt/completion/metadata 格式：completion A/B 映射为 similar/different，
原始 metadata 保留在 input_record.hf_metadata，并按 similarity_bucket、assay_type 输出分组指标。
默认 max-tool-rounds=3，断点续跑使用 --skip-existing。
```

`plot_llm_run_comparison.py` 的功能：

```text
汇总两个 LLM run 与 full-valid baseline，输出 overall、similarity_bucket、assay_type 三层
macro-F1 对比图、TSV 和 Markdown report。comparison 产物放在
outputs/chembl_tool/activity_transfer_benchmark/comparisons/，不要放进 llm_runs/。
```

MCS runtime 当前测试结果：

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/

dynamic_v1 continuous pairs: 1,989,152
timeout=1s, workers=128, chunksize=1:
  sampled 6,000 pairs, throughput ~490 pairs/s, full estimate ~1.1 h,
  timeout rate ~16%.

timeout=2s, workers=128, chunksize=1:
  sampled 3,000 pairs, throughput ~285 pairs/s, full estimate ~1.9 h,
  timeout rate ~13%.

workers=256 did not materially improve over 128 in the sampled test, likely due to process scheduling
and timeout-tail overhead. Prefer 128 workers first for full MCS runs.

推荐全量命令：

python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --full-scan \
  --workers 128 \
  --timeout-s 2 \
  --chunksize 1 \
  --progress-every 10000

卡住后收尾命令：

python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --finalize-existing \
  --workers 128 \
  --timeout-s 2

MCS threshold 分析命令：

python -m tools.chembl_tool.activity_transfer_benchmark.analyze_mcs_results \
  --run-id dynamic_v1_mcs_t2_analysis
```

当前 MCS full-scan partial 结果：

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/dynamic_v1_mcs_full_t2_w128_stream/
  mcs_sample_results.tsv.tmp
  missing_result_indices.tsv
  summary.json
  report_zh.md

完成 1,987,665 / 1,989,152 pairs，completion rate 99.9252%，missing 1,487。
timeout=2s 的 observed timeout count 为 218,664，约 11.0%。
```

当前 MCS threshold 分析结果：

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_analysis/dynamic_v1_mcs_t2_analysis/

non-ambiguous usable pairs: 1,500,676
best mean MCS coverage threshold: 0.70
best MCS macro-F1: 0.5538
best MCS balanced accuracy: 0.5542
best Tanimoto threshold on same subset: 0.48
best Tanimoto macro-F1 on same subset: 0.5688
best Tanimoto balanced accuracy on same subset: 0.5689

结论：mean MCS coverage 有 activity-transfer 信号，但单独做全局 threshold 时没有超过
Tanimoto。它更适合后续作为 LLM / learned classifier 的补充特征，而不是替代 Tanimoto。
```

当前 3K LLM eval set：

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/
  eval_pairs.jsonl
  eval_pairs.tsv
  summary.json
  report_zh.md

samples: 3,000
assay endpoints: 2,463
label counts: similar 1,500 / different 1,500
similarity buckets: 每个 bucket 500 pairs
baseline on this intentionally balanced set:
  Tanimoto>=0.50 macro-F1 0.4977, balanced accuracy 0.5020
  Tanimoto>=0.48 macro-F1 0.4903, balanced accuracy 0.4963
  MCS>=0.70 macro-F1 0.4941, balanced accuracy 0.4963
```

当前 gpt-oss-120b 3K LLM benchmark：

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/gpt_oss_120b_dynamic_v1_llm_3k_tools/
  manifest.json
  predictions.jsonl
  metrics.json
  report_zh.md
  figures/model_vs_baselines.svg
  runs/

model: gpt-oss-120b via local vLLM
tool service: http://127.0.0.1:8765
vLLM base URLs used: http://127.0.0.1:8001-8004/v1
api key used for this local vLLM server: EMPTY
n: 3,000, failed: 0, tool calls: 1,017
wall time: ~15.2 min with --parallelism 16 across 4 vLLM ports
usage: prompt_tokens 4,725,225; completion_tokens 1,651,970; total_tokens 6,377,195

LLM metrics:
  accuracy 0.5310
  balanced accuracy 0.5310
  macro-F1 0.5309

same-set baselines:
  Tanimoto>=0.50 macro-F1 0.4977
  Tanimoto>=0.48 macro-F1 0.4903
  MCS>=0.70 macro-F1 0.4941

gray zone subset, Tanimoto 0.40-0.70:
  n=818
  LLM macro-F1 0.5390
  Tanimoto>=0.50 macro-F1 0.4683

结论：在这个刻意按 label 和 similarity bucket 平衡的 3K stress-test 上，
gpt-oss-120b + tools 明显超过同集合里的简单 threshold baseline，但绝对性能仍然偏弱。
这个 3K set 不是 full dynamic_v1 分布，不应直接和 full-data best Tanimoto macro-F1 ~0.569
做一比一比较；它更适合作为 LLM 能否在困难样本上补充结构阈值的初版测试。
```

当前 DeepSeek-v4-pro 3K LLM benchmark：

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking/
  manifest.json
  predictions.jsonl
  metrics.json
  report_zh.md
  trace_messages.jsonl
  runs/

model: deepseek-v4-pro via https://api.deepseek.com
api key source: DEEPSEEK_API_KEY from .env
tool service: http://127.0.0.1:8765
n: 3,000, ok: 2,994, failed: 6, tool calls: 6,009
wall time: ~81.8 min with --parallelism 60
usage: prompt_tokens 13,141,595; completion_tokens 6,991,635; total_tokens 20,133,230
reasoning_content saved: 2,994 / 2,994 ok samples

LLM metrics:
  accuracy 0.5471
  balanced accuracy 0.5472
  macro-F1 0.5378
  similar recall 0.4052
  different recall 0.6892

same-success-subset baselines:
  Tanimoto>=0.50 macro-F1 0.4981
  MCS>=0.70 macro-F1 0.4941

gray zone subset, Tanimoto 0.40-0.70:
  n=818
  LLM macro-F1 0.5090
  Tanimoto>=0.50 macro-F1 0.4683

结论：DeepSeek-v4-pro 是当前 3K stress-test overall 指标最高的 LLM run，
macro-F1 0.5378 高于 gpt-oss-120b no-thinking 的 0.5309 和 thinking 的 0.5253。
但提升很小，并且主要来自更保守地预测 different；similar recall 偏低。
在更关键的 Tanimoto 0.40-0.70 灰区，DeepSeek 低于两个 gpt-oss run。
考虑 tool calls、token 和耗时，当前性价比不如本地 gpt-oss。
```

构建 3K eval set 命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_llm_eval_set \
  --run-id dynamic_v1_llm_3k
```

运行本地 gpt-oss-120b LLM benchmark 命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
  --run-id gpt_oss_120b_dynamic_v1_llm_3k_tools \
  --parallelism 16 \
  --base-urls http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
  --max-tool-rounds 1 \
  --max-tokens 1024 \
  --timeout-s 180 \
  --skip-existing \
  --api-key EMPTY \
  --progress-every 100
```

运行 DeepSeek-v4-pro thinking LLM benchmark / 断点续跑命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
  --run-id deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking \
  --model deepseek-v4-pro \
  --base-url https://api.deepseek.com \
  --env-file .env \
  --api-key-env DEEPSEEK_API_KEY \
  --parallelism 60 \
  --max-tool-rounds 3 \
  --max-tokens 20480 \
  --timeout-s 300 \
  --skip-existing \
  --reasoning-effort high \
  --enable-thinking \
  --progress-every 100
```

典型全量 baseline 命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_benchmark \
  --run-id chembl36_activity_transfer_v1 \
  --max-total-pairs 2000000 \
  --max-pairs-per-assay 5000 \
  --binary-max-total-pairs 500000 \
  --workers 32
```

推荐 dynamic-range filtered 命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_benchmark \
  --run-id chembl36_activity_transfer_dynamic_v1 \
  --min-pchembl-range 2.0 \
  --min-pchembl-iqr 0.75 \
  --max-total-pairs 2000000 \
  --max-pairs-per-assay 5000 \
  --binary-max-total-pairs 500000 \
  --workers 32
```

输出目录：

```text
outputs/chembl_tool/activity_transfer_benchmark/<run_id>/
  continuous_pairs.tsv.gz
  continuous_assay_endpoint_summary.tsv
  continuous_threshold_metrics.tsv
  continuous_similarity_bucket_summary.tsv
  continuous_assay_bucket_enrichment.tsv
  continuous_enrichment_summary.tsv
  binary_pairs.tsv.gz
  binary_assay_endpoint_summary.tsv
  binary_threshold_metrics.tsv
  binary_similarity_bucket_summary.tsv
  binary_assay_bucket_enrichment.tsv
  binary_enrichment_summary.tsv
  manifest.json
  report_zh.md
  figures/
    threshold_metrics.svg
    label_rates_by_bucket.svg
    median_delta_by_bucket.svg
    pair_counts_by_bucket.svg
    delta_lift_similar_rate_by_bucket.svg
    fold_lift_similar_rate_by_bucket.svg
    median_delta_reduction_by_bucket.svg
    binary_threshold_metrics.svg
    binary_label_rates_by_bucket.svg
    binary_delta_lift_similar_rate_by_bucket.svg
```

当前完整结果：

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_v1/
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/
```

未过滤 v1 的连续值主分析包含约 40k assay-endpoints、约 198 万 sampled pairs。最佳单一
Tanimoto threshold 约为 0.45，macro-F1 约 0.56。assay-specific enrichment 显示 close analog
相对各自 assay 背景的 macro similar-rate lift 约为 +0.13，distant / very_distant 为负；
结构相似度有弱到中等的 transfer 信号，但不应单独作为 activity transfer 判据。

dynamic_v1 使用 `pchembl_range >= 2.0` 且 `pchembl_iqr >= 0.75`，连续值主分析保留约 20k
assay-endpoints、约 199 万 sampled pairs。相比未过滤 v1，similar/different 标签更平衡，
median |delta pChEMBL| 更高，close analog 的 macro similar-rate lift 约为 +0.16；
这个版本更适合作为后续 LLM assay-transfer benchmark 的主数据。

## Task-specific native runner 记录

从这里开始的 BBB、ClinTox、Skin_Reaction 代码入口、旧 task prompt/schema、DeepSeek 运行参数和阶段性结果，
用于复现 `tools/chembl_tool/tasks/<task>/run_reasoning_pipeline.py` 的 native/legacy workflow。当前论文方法以
`tools/chembl_tool/paper_experiments/`、各 task 的 `experiment_config.py` 和
`tools/chembl_tool/tasks/AGENTS.md` 为准；两者冲突时，不得把旧 `Tier.endpoint_group` 分支、task-specific
prediction policy 或历史“下一步”恢复到 paper runner。

## BBB 代码入口

```text
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
  BBB endpoint_group、evidence_direction、evidence_strength 的规则。

tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
  BBB_Martins evidence library 构建入口。只保留 task 默认路径、输出文件名和 endpoint assignment
  配置；公共构建逻辑在 tools/chembl_tool/common/task_workflows/evidence_library.py。

tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
  BBB_Martins neighbor retrieval 入口。只保留默认 index 路径；公共检索逻辑在
  tools/chembl_tool/common/task_workflows/retrieve_neighbors.py。

tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
  BBB_Martins reasoning pipeline 入口。负责 neighbor retrieval、single-molecule analysis、
  group-level 并发 reasoning、final summary、trace 保存，以及 final-only rerun。

tools/chembl_tool/tasks/bbb_martins/run_reasoning_batch.py
  BBB_Martins 批量 reasoning 入口。按 query_index 调用单分子 pipeline，支持 molecule 级并行、
  可选 trace 保存/合并、prediction report、accuracy 和 macro-F1 评估。

tools/trace_viewer/viewer.html
  最终 paper trace 可视化页面。扫描 identity-blind/deployment-visible condition，查看单个样本的
  single/group/final messages、reasoning、tool calls、retrieval evidence 和通用 JSON response。
  不包含旧 task-specific structured field 适配。

tools/trace_viewer/start_viewer.sh
  在临时、受限的 serving root 中注册 Starling random/scaffold 与历史 TDC paper trace；第一个参数是端口，
  后续可选参数是要注册的 trace roots。

tools/chembl_tool/tasks/bbb_martins/
  其他 BBB evidence 清洗、打分、报告和输出汇总脚本。
```

## ClinTox 代码入口

当前 canonical lineage、source contract、split、retrieval hierarchy、prompt、实验结果和 no-promotion
结论统一记录在：

```text
tools/chembl_tool/tasks/clintox/AGENTS.md
tools/chembl_tool/tasks/clintox/CLINTOX_BENCHMARK.md
```

当前 gold 的 scientific source contract 是 source-reconstructed clinical-trial failure，活跃评估路径为
`data/conditioned_benchmark/ClinTox/scaffold/`；旧 `data/processed/ClinTox`、source-build 路径和早期
ChEMBL-native 结果只作 historical provenance。唯一 prompt profile 为
`tdc_source_aligned_v3`，旧 profile 已删除且旧/unversioned branch 不得复用。核心入口：

```text
tools/chembl_tool/tasks/clintox/build_clinical_trial_failure_benchmark.py
  从冻结 AACT positive 与 SWEETLEAD/FDA comparator 构造 parent labels 和 scaffold split。

tools/chembl_tool/tasks/clintox/starling_retrieval.py
  构造独立 direct/clinical/mechanistic retrieval library；Starling rows 不参与 gold label。

tools/chembl_tool/tasks/clintox/run_reasoning_pipeline.py
tools/chembl_tool/tasks/clintox/run_reasoning_batch.py
  复用公共 retrieval/reasoning workflow；默认 current split、heldout-filtered index、v3 prompt 和
  PARCC DeepSeek-V4-Flash tunnel。

tools/chembl_tool/tasks/clintox/audit_clinical_trial_failure_agent.py
  审计 direct provenance、coverage、label leak、structured retries 和 paired flips。
```

2026-08-16 scaffold-valid 的 best 是 `none` macro-F1 `0.6198`；direct/full-flat/full-mechanism 均未
通过 promotion gate。test 已经被查看，只保留 `post-test diagnostic`，不得作为新的 formal test，也不得继续在
同一 valid/test 调 prompt 或 selector。broad toxicity evidence 适合 risk explanation，但不能冒充
source-defined AACT association。

## Skin_Reaction 代码入口

Skin_Reaction 的 task-specific 细节记录在：

```text
tools/chembl_tool/tasks/skin_reaction/AGENTS.md
```

当前状态：

```text
已完成 task wrapper、assay scoring、endpoint grouping、evidence library、neighbor retrieval、
single/group/final reasoning pipeline 和 batch wrapper。

当前 label mapping:
  Y=1 -> risk
  Y=0 -> no_risk

当前 v1 benchmark batch:
  outputs/chembl_tool/tasks/skin_reaction/reasoning/batches/skin_reaction_calib_50_v1
  n=82, failed=0
  accuracy=0.682927, macro-F1=0.678140
  positive precision=0.733333, recall=0.702128, F1=0.717391
  confusion matrix: TN=23 FP=12 FN=14 TP=33
```

主要入口：

```text
tools/chembl_tool/tasks/skin_reaction/constants.py
  Skin_Reaction label 和 prediction mapping。

tools/chembl_tool/tasks/skin_reaction/rules.py
  skin sensitization、direct skin reaction、phototoxicity、irritation/corrosion、skin exposure
  和 weak/context evidence 的筛选关键词与排除规则。

tools/chembl_tool/tasks/skin_reaction/scoring.py
  assay 保留/剔除和打分入口。screen_assays.py 和 rescore_outputs.py 都调用 scored_row()。

tools/chembl_tool/tasks/skin_reaction/endpoint_groups.py
  Tier.endpoint_group、evidence_direction、evidence_strength 和 endpoint assignment 规则。

tools/chembl_tool/tasks/skin_reaction/build_evidence_library.py
  evidence library 构建入口。默认读取 assay_screening/v1，输出 molecule evidence、neighbor index 和 meta。

tools/chembl_tool/tasks/skin_reaction/retrieve_neighbors.py
  analog retrieval 入口。当前 benchmark 使用 top-k-per-group=3、min-similarity=0.35。

tools/chembl_tool/tasks/skin_reaction/run_reasoning_pipeline.py
  单分子 reasoning pipeline：retrieval prefetch、single-molecule branch、group-level 并发 reasoning、
  final summary、trace 保存，以及 final-only rerun。

tools/chembl_tool/tasks/skin_reaction/run_reasoning_batch.py
  批量 reasoning wrapper。复用 common reasoning_batch.py，输出 predictions、metrics、report、logs、
  runs 和 combined trace。
```

## BBB evidence 分组标准

BBB 第二阶段不按每个 assay 单独检索。应按：

```text
Tier -> endpoint_group
```

组合生成 retrieval groups。

### Tier 1: direct BBB / brain exposure

建议 endpoint groups：

```text
direct_brain_plasma
  bpr
  brain/plasma
  b/p
  bbr
  ratio
  ratio auc

direct_unbound_brain
  k(p,uu,brain)
  k(p,uu,csf)
  kp
  fu

direct_logbb_or_brain_level
  logbb
  log bb
  brain level
  brain concentration
  brain penetration index
  bpi

direct_brain_uptake_or_perfusion
  drug uptake
  drug uptake(free)
  uptake
  brain uptake
```

### Tier 2: passive permeability / barrier model

建议 endpoint groups：

```text
passive_papp
  papp
  logpapp
  logp app
  papp e-6

passive_caco2
  caco-2 papp
  caco-2 permeability
  pcaco2

passive_generic_permeability
  permeability
  permeability coefficient
  peff
  logpeff
  log pe
  pc
  pm
  pbbb

passive_transport_or_recovery
  drug transport
  drug recovery
```

### Tier 3: efflux transporter

Tier 3 必须拆分强弱证据，不能把 transporter inhibition 直接解释成 efflux substrate。

建议 endpoint groups：

```text
efflux_functional_ratio_or_bidirectional
  efflux ratio
  ratio
  ratio_papp
  papp a to b (mean)
  papp b to a (mean)
  papp

efflux_transport_or_accumulation
  drug transport
  drug uptake
  activity
  flu intensity
  rfu
  fluorescence

efflux_atpase_or_probe
  ratio_atpase activity
  atpase
  relative jc-1 accumulation

efflux_inhibition_or_binding
  inhibition
  ic50
  ki
  ec50
  kd
  km
  kon
  k_off
  ratio ic50
  ratio ec50
  fc
```

### Tier 4: influx transporter

Tier 4 也必须区分 functional uptake 和普通 binding/inhibition。

建议 endpoint groups：

```text
influx_functional_uptake_or_transport
  drug uptake
  uptake
  drug transport
  transport

influx_kinetic_or_substrate
  km
  vmax
  jmax
  kin

influx_inhibition_or_binding
  inhibition
  ic50
  ki
  kd
  kon
  k_off
  ec50
  ratio ic50
```

### Unknown / weak context

下面 endpoint 只能作为 context-dependent evidence，不能单独强解释：

```text
activity
ratio
inhibition
survival
cc50
gi50
ec90
mic
flu intensity
rfu
fluorescence
```

如果这些 endpoint 出现在明确 assay context 中，可以被 endpoint group 规则提升；否则应标记为：

```text
endpoint_group: context_dependent
evidence_strength: weak
```

## BBB evidence library

应从当前 assay candidates 和 activity evidence 构建 molecule-level library。每一条 evidence 至少包含：

```text
molecule_chembl_id
canonical_smiles
standard_inchi_key
assay_chembl_id
assay_id
assay_tier
endpoint_group
endpoint_group_reason
assay_description
target_chembl_id
target_pref_name
target_genes
standard_type
standard_relation
standard_value
standard_units
pchembl_value
data_validity_comment
activity_comment
evidence_direction
evidence_strength
evidence_reason
```

其中：

```text
evidence_direction:
  supports_bbb_crossing
  argues_against_bbb_crossing
  efflux_risk
  influx_support
  permeability_support
  context_dependent
  unknown_direction

evidence_strength:
  strong
  moderate
  weak
  context_dependent
```

`endpoint_group` 由 `assay_tier + standard_type + assay_description + target_genes` 共同决定。不能只看 `standard_type`，因为 `ratio`、`activity`、`inhibition` 等 endpoint 在不同 assay context 下含义不同。

## Neighbor retrieval 设计

输入：

```text
query_smiles
top_k_per_group: default 3
min_similarity: default 0.3
groups: optional list[Tier.endpoint_group]
exclude_exact: default true
```

流程：

```text
1. 标准化 query molecule，生成 canonical SMILES、InChIKey、Morgan fingerprint。
2. 按 evidence library 中的 Tier.endpoint_group 建立 group membership。
3. 对每个 group 独立计算 query 与该 group molecule fingerprints 的 Tanimoto。
4. 每个 group 返回 top 3 non-identical neighbors，默认过滤 Tanimoto < 0.3 的 very distant analog。
5. 对同一 neighbor molecule 聚合其在该 group 下的所有 assay/activity evidence。
6. 返回 group-level retrieval payload。
```

同一分子排除标准：

```text
same molecule_chembl_id
same full standard_inchi_key
same InChIKey connectivity layer, i.e. the first block before "-"
same canonical_smiles
```

相似度 bucket：

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

默认使用 `min_similarity=0.3` 过滤 very distant analog，减少 sparse group 中几乎不可迁移的 evidence。保留的低相似度 analog 仍应交给 group-level LLM 判断 transferability。LLM prompt 必须明确：`distant_analog` 和 `very_distant_analog` 不能作为正负证据，除非共享 scaffold 和 assay mechanism 有很强的药化理由。

当前实现会用预计算 ChEMBL fingerprints：

```text
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

构建 BBB evidence molecule subset 的 fingerprint/group index。这个 subset 当前约 5 万个
molecule，query-time 对每个 group 做 `BulkTanimotoSimilarity` 足够快。后续如需扩展为
全 ChEMBL 背景邻居检索，可以在不改变 BBB MVP payload contract 的前提下新增背景索引。

### Query ChEMBL exact context / shared-assay enrichment

当前实现保留一个可选的 evidence-rich 增强：

```text
tools/chembl_tool/tasks/bbb_martins/chembl_exact_context.py
```

它可以用 query full InChIKey 查 ChEMBL exact molecule，读取 ChEMBL compound properties、
query molecule 的 BBB-relevant evidence rows，并在 retrieved neighbor assay 中查 query activity，
生成两类 shared-assay context：

```text
same_endpoint_activity:
  same assay_chembl_id + same normalized standard_type

same_assay_different_endpoint_activity:
  same assay_chembl_id + different standard_type
```

这类信息使用了 query molecule 的已知 ChEMBL 实验记录。为避免 prospective evaluation 中的数据泄漏，
默认必须关闭，因此不会添加 `query_chembl_context` / `exact_query_chembl_context`，也不会把 query
的 ChEMBL exact evidence 注入 single-molecule prompt、group prompt 或 batch 评估。只有显式传：

```bash
--enable-chembl-exact-context
```

才允许作为 retrospective / evidence-rich case study 使用。默认 benchmark、accuracy 和 macro-F1 报告
都应保持该开关关闭。

## LLM reasoning 流程

BBB_Martins 的 legacy native runner 分为并发 evidence branches 和 final summary。当前 paper runner 会先把
下面的 source-local groups 合并为 `experiment_config.py` 声明的 mechanism families。

### Group-level reasoning

Legacy native runner 中每个 `Tier.endpoint_group` 独立执行：

```text
input:
  query molecule
  top 3 neighbors for this group
  cleaned raw ChEMBL assay/activity evidence rows
  tier and endpoint_group

available tools:
  mmp_structure_compare
  properties_compare

not sent:
  query standard_inchi_key
  query fingerprint
  group n_candidate_molecules
  evidence_direction
  evidence_strength
  endpoint_group_reason
  assay_reason

output:
  group_id
  useful_for_bbb_reasoning: true/false
  transferability:
    high
    moderate
    low
    not_applicable
  evidence_direction:
    supports_bbb_crossing
    argues_against_bbb_crossing
    efflux_risk
    influx_support
    neutral_or_unclear
  confidence:
    high
    moderate
    low
  reasoning_summary
  key_evidence:
    molecule_chembl_id
    similarity
    similarity_bucket
    assay_signal
    activity_values
    tool_summary
    transferability
    effect_on_bbb_reasoning
  caveats
```

`key_evidence` 是当前格式。旧的 `key_neighbors` 不再使用。

每个 group 的 DeepSeek 对话、reasoning、tool calls 和 tool messages 都保存到 trace。
这些 group 没有严格依赖关系，可以并发执行。

### Single-molecule reasoning

每个 query 还会并发执行一个单分子分析分支：

```text
input:
  query molecule
  exact_query_chembl_context only when --enable-chembl-exact-context is enabled and exact context is found

available tools:
  molecule_properties

not sent by default:
  ChEMBL neighbor evidence
  exact_query_chembl_context
  mmp_structure_compare
  properties_compare

output:
  passive_bbb_plausibility
  efflux_or_transporter_prior
  confidence
  reasoning_summary
  property_drivers
  caveats
```

默认情况下这个分支只看到 query molecule 和 `molecule_properties`，不会出现 ChEMBL neighbor
evidence，也不会出现任何 `exact_query_chembl_context` 相关 payload 或 prompt instruction。
只有显式开启 exact ChEMBL context 且命中 query exact context 时，才会把
`exact_query_chembl_context` 放入 single-molecule payload，并提示模型区分 direct same-molecule
ChEMBL evidence 和 physicochemical prior。ChEMBL neighbor evidence 仍只进入 group-level context。

### Final reasoning

final LLM 读取：

```text
query molecule
single-molecule analysis output
all group-level reasoning outputs
coverage summary
```

final 阶段不暴露工具，只综合前面分支的 structured outputs。当前 final prompt 规则：

```text
Return compact complete JSON.
Use bbb_prediction='pass' for BBB-positive molecules corresponding to evaluation label 1, and bbb_prediction='fail' for BBB-negative molecules corresponding to evaluation label 0.
Use the single-molecule analysis as the physicochemical prior.
Use group analyses as analog evidence; downweight groups marked low confidence or low transferability.
Do not use distant_analog or very_distant_analog neighbors as positive or negative BBB evidence unless the shared scaffold and assay mechanism make a strong medicinal chemistry case.
Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.
```

输出：

```text
bbb_prediction:
  pass
  fail
  uncertain

confidence:
  high
  moderate
  low

main_reasons
efflux_risk_assessment
influx_support_assessment
passive_permeability_assessment
direct_brain_exposure_analog_assessment
evidence_gaps
final_summary
```

测试集里 `Y` 只能用于评估，不应进入 retrieval 或 LLM prompt。
当前评估约定：`Y=1` 对应 `bbb_prediction=pass`，`Y=0` 对应 `bbb_prediction=fail`。
`uncertain` 在 overall accuracy 和 macro-F1 中按未命中计入；报告中也会给出 decided-only accuracy。

### Trace 保存和可视化

每次 reasoning run 输出到：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

当前文件：

```text
retrieval.json
single_molecule_reasoning_output.json
group_reasoning_outputs.jsonl
final_reasoning_output.json
trace_messages.jsonl
manifest.json
```

`trace_messages.jsonl` 中每条记录对应一个 trace item：

```text
single_molecule
Tier <n>.<endpoint_group>
final_summary
```

每条 trace record 包含：

```text
index
sample_id
molecule_key
smiles
label
status
prediction
response_text
messages
tool_count
usage
raw_output
```

`label` 只用于本地评估和 trace 审计，不进入 LLM prompt。`sample_id` 当前等于
`query_index`，`molecule_key` 当前形如 `index:9`。viewer 会按 molecule package 分组，
方便在一个 run 或上传的 JSONL 中选择不同分子的 trace 包，再查看该分子内部的所有阶段。

旧 task reasoning trace 已不再由 viewer 支持。最终论文 trace 统一启动：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

然后打开：

```text
http://localhost:8776/.trace_viewer.html?v=paper-v2
```

Viewer 默认分别注册 Starling random、Starling scaffold 和历史 TDC test；每个 dataset 只扫描 `runs`、
`runs_deployment_visible_prefetched`、`runs_deployment_visible` 和
`runs_deployment_visible_parent_disjoint` 中由正式 `predictions.jsonl` 引用的样本级 trace，不跨 dataset
合并指标。对于 parent-disjoint 样本，viewer 还会读取 manifest 和 `reuse.json`，显示 identity policy，
并区分 retrieval 变化后的重跑与 LLM-visible input 未变化时的 artifact reuse。旧 task reasoning 目录的
保留和清理规则见 `tools/chembl_tool/paper_experiments/TRACE_RETENTION.md`。

常用 pipeline 命令：

```bash
# 历史 TDC native runner：完整运行一个 test_efflux 分子
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --query-index 0 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-workers 4 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --run-id <run_id>

# 只重跑已有 run 的 final summary
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline \
  --resume-final-from-run-dir outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id> \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro

# 历史 TDC batch 复现；新 Starling benchmark 不得沿用这个 input path 或 full-source index
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --input-jsonl data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl \
  --parallelism 1 \
  --top-k-per-group 3 \
  --min-similarity 0.3 \
  --max-tool-rounds 10 \
  --timeout-s 300 \
  --max-tokens 8192 \
  --model deepseek-v4-pro \
  --batch-id <batch_id>
```

默认会把每个单分子 run 的 stderr 进度实时打印到控制台，例如
`[idx00003 stderr] [bbb_reasoning_pipeline] group done: ...`，同时完整保存到
`outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/`。如果只想写日志文件、不想在控制台显示进度，可加
`--no-stream-logs`。

batch 断点续跑使用 `--skip-existing`。用同一个 `--batch-id` 重新运行时，脚本会检查
`reasoning/batches/<batch_id>/runs/<batch_id>_idxNNNNN/final_reasoning_output.json`：
存在则认为该 molecule 已完成并跳过，不覆盖已有 stdout/stderr log；不存在则重新运行该 molecule。
因此中断后的 partial run 会自动补跑，已完成结果会进入新的 predictions、metrics、report 和
batch `trace_messages.jsonl` 汇总。这个逻辑由 `tools/chembl_tool/common/task_workflows/reasoning_batch.py`
统一实现，BBB_Martins、Bioavailability_Ma、ClinTox 和 Skin_Reaction 共用。

批量输出：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/manifest.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/predictions.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/metrics.json
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/report.md
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/trace_messages.jsonl
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/logs/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/
```

batch 中每个 molecule 的独立 run 保留在 batch 目录内部：

```text
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00000/
outputs/chembl_tool/tasks/bbb_martins/reasoning/batches/<batch_id>/runs/<batch_id>_idx00001/
...
```

旧 task batch 可以生成合并 trace 供离线审计，但最终 paper runner 必须使用
`--no-combine-traces`，只保留每个 molecule 自己的 trace。最终 viewer 固定启动为：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

Viewer 从 condition 的 `predictions.jsonl` 构建样本列表，再按需加载 per-run trace 和 retrieval。
新增 task 时必须沿用通用 stage/message/tool/JSON contract；不要向 viewer 添加 task prediction、
Tier、expert-policy 或 `key_evidence` effect 字段的专用适配。

### LLM usage 与成本估算

DeepSeek API response 会返回 token usage，但不会在每次 response 中直接返回美元费用。
估算 batch 费用时用 trace 中保存的 usage 汇总 token，再乘以 DeepSeek 官方 pricing。
价格会变，生产估算前必须先查官方页面：

```text
https://api-docs.deepseek.com/quick_start/pricing/
```

最终 paper viewer 只负责 trace 审计与可视化，不内置 provider-specific 价格或成本估算。
需要估算旧 run 成本时，从 trace 的 usage 字段离线汇总 token，并使用运行当日的官方价格。

截至 2026-05-12，官方页面显示 `deepseek-v4-pro` 当前折扣价为：

```text
input cache hit:  $0.003625 / 1M tokens
input cache miss: $0.435    / 1M tokens
output:           $0.87     / 1M tokens
```

折扣截至 2026-05-31 15:59 UTC；原价为：

```text
input cache hit:  $0.0145 / 1M tokens
input cache miss: $1.74   / 1M tokens
output:           $3.48   / 1M tokens
```

从一个 run 或 batch 的 `trace_messages.jsonl` 汇总 usage：

```bash
jq -s 'reduce .[] as $r (
  {calls:0,prompt:0,completion:0,total:0,cache_hit:0,cache_miss:0,reasoning:0};
  .calls += (if ($r.usage//null) then 1 else 0 end) |
  .prompt += (($r.usage.prompt_tokens // 0) | tonumber) |
  .completion += (($r.usage.completion_tokens // 0) | tonumber) |
  .total += (($r.usage.total_tokens // 0) | tonumber) |
  .cache_hit += (($r.usage.prompt_cache_hit_tokens // $r.usage.prompt_tokens_details.cached_tokens // 0) | tonumber) |
  .cache_miss += (($r.usage.prompt_cache_miss_tokens // 0) | tonumber) |
  .reasoning += (($r.usage.completion_tokens_details.reasoning_tokens // 0) | tonumber)
)' outputs/chembl_tool/tasks/<task_name>/reasoning/batches/<batch_id>/trace_messages.jsonl
```

如果估算 standalone 单分子 run，把路径替换为：

```text
outputs/chembl_tool/tasks/<task_name>/reasoning/single_runs/<run_id>/trace_messages.jsonl
```

其中 `output_tokens` 使用 usage 中的 `completion_tokens`。

费用公式：

```text
cost =
  cache_hit_tokens  / 1,000,000 * cache_hit_price
+ cache_miss_tokens / 1,000,000 * cache_miss_price
+ output_tokens     / 1,000,000 * output_price
```

当前 BBB_Martins smoke 估算基线：

```text
single molecule example:
  14 LLM calls
  prompt 186,050 tokens
  completion 54,562 tokens
  cache_hit 121,088
  cache_miss 64,962
  estimated discounted cost: ~$0.076 / molecule

3 molecule parallelism=3 smoke:
  40 LLM calls
  prompt 459,509 tokens
  completion 152,893 tokens
  cache_hit 340,352
  cache_miss 119,157
  estimated discounted cost: ~$0.186 total, ~$0.062 / molecule
```

这些只用于粗估。不同 molecule 的 endpoint group 覆盖、tool rounds、final prompt 长度会变化；
全量预算应先抽样 3-10 个 molecule，按平均成本乘以 molecule 数量，并留出余量。

## FastAPI 常驻服务标准

服务启动时初始化慢资源，后续每次 tool invoke 复用已加载对象：

```text
RDKit standardization config
MolGpKa import/model state
AccFG import/state
mmpdb Python package import
后续其他慢启动 ML models 或大索引
```

当前 endpoint：

```text
GET /health
GET /tools
POST /tools/{tool_name}/invoke
POST /tools/invoke
POST /tools/{tool_name}
```

其中 `/tools/{tool_name}/invoke` 是标准通用工具接口。`/tools/invoke` 和 `/tools/{tool_name}` 是兼容入口。`/tasks/bbb_martins/retrieve`、`/tasks/bbb_martins/reason`、`/tasks/bbb_martins/predict` 后续如果需要 task-level orchestration 再新增。

### 通用 ToolRequest

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "input": {},
  "options": {
    "timeout_s": 120,
    "return_debug": false
  }
}
```

### 通用 ToolResponse

```json
{
  "request_id": "string",
  "tool_name": "molecule_properties",
  "version": "v1",
  "status": "ok",
  "output": {
    "text": "LLM-readable natural language output",
    "..."
  },
  "warnings": [],
  "errors": [],
  "metadata": {
    "started_at": "ISO-8601",
    "finished_at": "ISO-8601",
    "latency_ms": 0,
    "model_or_index_version": "string"
  }
}
```

失败时：

```json
{
  "status": "error",
  "output": null,
  "warnings": [],
  "errors": [
    {
      "code": "INVALID_SMILES",
      "message": "Could not parse query SMILES.",
      "recoverable": true
    }
  ]
}
```

## Tool: molecule_properties v1

输入：

```json
{
  "query_smiles": "CCO"
}
```

职责：

```text
1. 标准化 query SMILES，返回 canonical_smiles 和 standard_inchi_key。
2. 计算易解释 RDKit descriptors。
3. 通过 MolGpKa 计算 acidic/basic pKa 和 logD。
4. 通过 AccFG 识别最顶层 functional groups。
5. 生成自然语言 output.text，作为 LLM 唯一直接消费的工具文本。
```

主要输出字段：

```text
output.text
output.query
output.properties
output.functional_groups
output.raw_features
```

`properties` 至少覆盖：

```text
MolGpKa pKa/logD features
RDKit molecular weight
logP
TPSA
HBD/HBA
rotatable bonds
formal charge
heavy atom count
aromatic rings
fraction Csp3
QED
rule flags
```

## Tool: properties_compare v1

输入：

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

职责：

```text
1. 复用 registry 中同一个 molecule_properties 工具实例，避免重复初始化 MolGpKa/AccFG。
2. 比较 molecule_properties 中所有非 functional-group properties。
3. delta 定义为 query_value - reference_value。
4. 生成自然语言 output.text，说明哪些 properties 增加、降低或不适用。
```

主要输出字段：

```text
output.text
output.query
output.reference
output.comparisons
```

functional groups 不在此工具中比较。FG 信息只由 `molecule_properties` 单独返回。

## Tool: mmp_structure_compare v1

输入：

```json
{
  "query_smiles": "CCN(CC)CC",
  "reference_smiles": "CCO",
  "query_label": "query",
  "reference_label": "reference"
}
```

职责：

```text
1. 计算 Morgan fingerprint Tanimoto similarity。
2. 按 similarity bucket 标记结构相似度。
3. 调用 mmpdb 的 fragmentation / matched-pair 逻辑，描述可解释的 matched-pair transformation。
4. 计算 RDKit MCS coverage，辅助判断共同骨架比例。
5. 生成自然语言 output.text，作为 LLM 可读结构差异说明。
```

主要输出字段：

```text
output.text
output.query
output.reference
output.similarity
output.transformation
output.mcs
```

`mmp_structure_compare` 不返回 descriptor/property deltas。所有属性差异比较必须使用 `properties_compare`。

结构 similarity bucket：

```text
very_close_analog: Tanimoto >= 0.95
close_analog: 0.80 <= Tanimoto < 0.95
moderate_analog: 0.60 <= Tanimoto < 0.80
weak_analog: 0.40 <= Tanimoto < 0.60
distant_analog: 0.20 <= Tanimoto < 0.40
very_distant_analog: Tanimoto < 0.20
```

## ChEMBL neighbor retrieval 状态

`chembl_neighbors` 仍是 BBB_Martins reasoning 的重要 evidence retrieval 概念，但当前实现不是
DeepSeek function tool，也没有作为 `tools/service/` 的已注册常驻工具暴露。当前代码入口是：

```text
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

Legacy native runner 的调用方式：

```text
run_reasoning_pipeline.py 在 LLM 调用前读取 BBB neighbor index，
按 Tier.endpoint_group 预取每组 top 3 neighbors，
清理内部派生字段后把 neighbors/evidence_rows 放进 group-level user prompt。
```

因此 group-level DeepSeek 看到 neighbor evidence，但不能主动调用 `chembl_neighbors`。它能调用的工具只有：

```text
mmp_structure_compare
properties_compare
```

后续如果需要把 BBB evidence retrieval 也放进常驻服务，应复用现有 ToolRequest / ToolResponse contract，并把服务端启动时加载 BBB evidence library 和 fingerprint group index。不要复制服务框架。

## 并发策略

服务内部可并发执行：

```text
molecule_properties
properties_compare
mmp_structure_compare
ChEMBL neighbor retrieval / evidence prefetch for each group（pipeline 内部；后续可 service 化）
group-level LLM reasoning for each group
```

Legacy native runner 与当前 paper runner 的并发边界：

```text
1. source-local retrieval/normalization 可以保持 endpoint-group 粒度。
2. legacy native runner 的 group-level reasoning 粒度是 endpoint group；paper runner 必须按 mechanism family。
3. final reasoning 必须等待所有启用的 mechanism-family/group reasoning 完成。
4. 每个 query 要有 request_id/run_id，所有中间产物可追踪。
```

## 实施计划

### Phase 0: 文档与接口冻结

状态：

```text
已完成，并在本文件中记录当前工具入口和 LLM 可见输出约定。
```

当前冻结的 service tool 名称：

```text
molecule_properties
properties_compare
mmp_structure_compare
```

### Phase 1: BBB evidence library

状态：

```text
已实现核心入口：
tools/chembl_tool/tasks/bbb_martins/endpoint_groups.py
tools/chembl_tool/tasks/bbb_martins/build_evidence_library.py
```

目标保持不变：

```text
能从 v6 assay/activity 输出生成 molecule-level evidence。
每条 evidence 有 endpoint_group、evidence_direction、evidence_strength。
测试覆盖 Tier.endpoint_group 映射规则。
```

### Phase 2: BBB neighbor retrieval

状态：

```text
已实现核心入口：
tools/chembl_tool/tasks/bbb_martins/retrieve_neighbors.py
```

当前行为：

```text
给定一个 SMILES，每个 Tier.endpoint_group 返回 top 3 non-identical neighbors。
返回完整 assay description 和 activity rows。
能批量处理 test_efflux.jsonl。
默认使用 min_similarity=0.3 过滤 very distant analog；保留的低相似度 analog 继续标记 bucket。
```

### Phase 3: 常驻 FastAPI service

状态：

```text
已实现：
tools/service/app.py
tools/service/config.py
tools/service/registry.py
tools/service/schemas.py
tools/service/errors.py
tools/service/tools/base.py
tools/service/tools/rdkit_properties.py
tools/service/tools/properties_compare.py
tools/service/tools/mmp_structure_compare.py
```

当前验收：

```text
服务启动时初始化 MolGpKa、AccFG、mmpdb 等慢资源。
POST /tools/{tool_name}/invoke 可调用 molecule_properties、properties_compare、mmp_structure_compare。
GET /tools 可列出工具 schema 和版本。
所有工具 output.text 为 LLM 可见文本，数字最多保留两位小数。
```

注意：当前 `chembl_neighbors` 没有注册为 service tool，也不是 LLM 可调用 tool；它是 BBB pipeline 内部的 evidence prefetch/context assembly。DeepSeek 只接收预取后的 group evidence。后续如果需要 service 化，应新增 `tools/service/tools/chembl_neighbors.py`，但不要替换或复制现有通用工具框架。

### Phase 4: LLM reasoning payloads

状态：

```text
已实现核心 orchestration：
tools/chembl_tool/tasks/bbb_martins/run_reasoning_pipeline.py
```

当前行为：

```text
1. 从 test_efflux.jsonl 读取 query。
2. 用 retrieve_neighbors.py 获取每个 Tier.endpoint_group 的 top 3 neighbors。
3. 并发执行 single-molecule analysis，只暴露 molecule_properties。
4. 并发执行每个 group-level analysis，只暴露 mmp_structure_compare 和 properties_compare。
5. 等待所有 group 和 single 分支完成后执行 final summary，不暴露工具。
6. 保存 retrieval、single、group、final、trace_messages 和 manifest。
7. 支持 --resume-final-from-run-dir 复用已有 retrieval/single/group 输出，只重跑 final summary。
```

当前 DeepSeek 调用约定：

```text
OpenAI SDK
base_url=https://api.deepseek.com
model=deepseek-v4-pro
thinking enabled
reasoning_effort=high
```

API key 默认从 `--env-file .env` 中读取 `DEEPSEEK_API_KEY`。`run_reasoning_pipeline.py`
会让 `.env` 中的值覆盖当前 shell 已存在的同名环境变量；这是为了保证直接从 shell 跑 batch
时仍以项目 `.env` 为准。若要临时切换 key，应显式传 `--api-key-env <ENV_NAME>`，并在 `.env`
中配置对应变量。

当前 trace 验收：

```text
完整保存 system/user/assistant/tool messages。
保存 assistant reasoning_content。
保存 assistant tool_calls。
保存 ToolResponse.output.text 作为 tool message content。
保存 molecule_key，方便 viewer 按分子 trace package 分组。
```

### Phase 5: BBB_Martins evaluation

状态：

```text
已完成单分子端到端 smoke runs，输出目录：
outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs/<run_id>/
```

已验证的示例：

```text
query_index=0:
  run_id=first_efflux_full_key_evidence_20260505_182029
  final prediction=fail
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items

query_index=9:
  run_id=last_efflux_full_key_evidence_20260505_185201
  final prediction=fail after final-only prompt rerun
  confidence=moderate
  19 group outputs, 21 trace records, 57 key_evidence items
```

后续 evaluation 工作：

```text
对 test_efflux.jsonl 批量运行 retrieval + reasoning。
评估时只在最后对照 Y，不把 Y 传给工具或 LLM。
报告 coverage、prediction accuracy、uncertain rate、典型成功/失败案例。
batch trace 支持多个 molecule package，viewer 按 molecule_key 分组浏览。
```

### Phase 6: 扩展到其他任务

新任务只新增：

```text
tools/<domain_tool>/tasks/<task_name>/
tools/service/tasks/<task_name>.py
```

对应输出统一放在：

```text
outputs/<domain_tool>/tasks/<task_name>/
  assay_screening/
  evidence_library/
  reasoning/
    single_runs/
    batches/
```

如果需要新模型或新索引，则新增：

```text
tools/service/tools/<tool_name>.py
```

不要复制已有服务框架、tool schema、request/response 标准。下面这些 task workflow helper 也不要复制：

```text
tools/chembl_tool/common/task_workflows/screen_assays.py
tools/chembl_tool/common/task_workflows/rescore_outputs.py
tools/chembl_tool/common/task_workflows/summarize_outputs.py
tools/chembl_tool/common/task_workflows/assay_report.py
tools/chembl_tool/common/task_workflows/evidence_library.py
tools/chembl_tool/common/task_workflows/retrieve_neighbors.py
tools/chembl_tool/common/task_workflows/chembl_exact_context.py
tools/chembl_tool/common/task_workflows/reasoning_batch.py
```

新 task 的 `run_reasoning_batch.py` 应作为薄 wrapper 调用
`tools/chembl_tool/common/task_workflows/reasoning_batch.py`，只配置：

```text
default_input
default_batch_root
default_index
pipeline_module
prediction_field
positive/negative label mapping
```

task-specific pipeline、retrieval、prompt 和 final schema 可以继续放在各自 task 目录中；
当这些部分也稳定到足够通用时，再抽取公共模块。

新增 task 必须把 task-specific 输出保留在通用 JSON response 内，并沿用统一
stage/message/tool contract。Viewer 会递归展示这些 JSON，不应新增 task-specific 渲染分支。

## 当前不做的事情

```text
不训练 BBB classifier。
不把每个 assay 作为独立 retrieval group。
不把 IC50/inhibition 直接解释成 substrate 或 transport。
不把 test_efflux.jsonl 的 Y 暴露给 retrieval 或 LLM prompt。
不在每次 query 时重新 import 或初始化 MolGpKa、AccFG、mmpdb、ML model 或大索引。
不把 descriptor/property deltas 放进 mmp_structure_compare；属性差异统一走 properties_compare。
不把工具结构化 JSON 整包展示给 LLM；LLM 默认只看 output.text。
不为已安装的 mmpdb 增加源码路径环境变量。
```

## 历史 BBB 阶段性下一步

下面是早期 BBB 单分子 MVP 阶段留下的计划，已经被当前 paper experiment plan 取代，仅用于解释历史实现：

```text
1. 批量评估 test_efflux.jsonl，形成 prediction/label 对照表和错误分析。
2. 继续审计 final summary 的证据加权，必要时增加 explicit adjudication fields。
3. 如果需要让其他系统复用 retrieval，再把 chembl_neighbors 包装为 service tool 或 task endpoint；当前 BBB pipeline 继续把它作为内部 evidence prefetch。
4. 后续接入更多常驻 ML tools，例如更慢的 pKa/logD、solubility、PK 或 toxicity 模型。
```
