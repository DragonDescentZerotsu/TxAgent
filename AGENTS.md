# TxAgent 当前系统：常驻分子工具服务与证据检索推理

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

当前 BBB 数据基础：

```text
assay candidates:
  outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_assay_candidates.csv

activity evidence:
  outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_activity_evidence.csv

ChEMBL fingerprints:
  tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz

test molecules:
  data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl
```

`test_efflux.jsonl` 当前字段：

```text
drug: query SMILES
Y: BBB label
```

## 设计原则

1. 不把 BBB 逻辑写成一次性脚本。BBB 是第一个 task，但工具服务、检索协议、LLM 输入输出格式应能支持后续更多任务。
2. ChEMBL neighbor retrieval 是 pipeline 的 evidence prefetch / context assembly 步骤，不是当前暴露给 LLM 的 function tool，也不是当前 FastAPI service tool。后续 pKa、logD、solubility、toxicity、target affinity、PK property 等模型才按通用 tool contract 接入。
3. 长初始化模型要常驻。慢启动模型和大索引应在服务启动时加载，通过 FastAPI endpoint 调用，避免每个 query 反复初始化。
4. evidence retrieval 只提供证据，不直接替代 reasoning。retrieval payload 必须保留 assay 描述、activity 数值、endpoint 语义、similarity 和不确定性。
5. retrieval 单元优先是 molecule-level evidence，不是 assay-level evidence。assay 信息要保留，但 query-time ranking 应先找相似 molecule，再展开其 assay/activity evidence。
6. LLM reasoning 分为并发证据分支和 final 汇总：single-molecule 分支判断理化性质先验，
   group-level 分支判断每个 Tier.endpoint_group 的 analog transferability，final-level 汇总所有证据。

## 当前常驻工具服务

当前已经实现的通用工具服务入口：

```text
tools/service/app.py
  FastAPI app。注册工具并提供 /health、/tools、/tools/{tool_name}/invoke、/tools/invoke、/tools/{tool_name}。

tools/service/config.py
  服务配置。MolGpKa 相关开关在这里读取；mmpdb 使用当前 Python 环境中已安装的 mmpdblib，不需要源码路径环境变量。

tools/service/registry.py
  ToolRegistry。负责初始化工具、复用共享实例、统一 invoke。

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
tests/service/test_rdkit_properties.py
tests/service/test_properties_compare.py
tests/service/test_mmp_structure_compare.py
```

服务启动命令：

```bash
uvicorn tools.service.app:app --host 127.0.0.1 --port 8765
```

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

BBB evidence library 内部可以保留 `evidence_direction`、`evidence_strength`、`endpoint_group_reason`、`assay_reason` 等规则派生字段，方便 debug 和审计；但这些字段不要发送给 reasoning LLM。LLM payload 中的 activity evidence row 应只包含原始 ChEMBL assay/activity 字段和必要 metadata，例如 assay id、tier、description、target、standard_type/value/units、activity_comment、confidence_score、relationship_type。分组层可以保留 `tier` / `endpoint_group`，因为并行分析本身按这个分组运行。

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
  retrieve_neighbors.py
  chembl_exact_context.py
  reasoning_batch.py
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

retrieve_neighbors.py
  对每个 Tier.endpoint_group 做 analog retrieval，包含 exact-molecule 排除、Tanimoto ranking、
  similarity bucket 和 JSONL batch retrieval CLI。

chembl_exact_context.py
  可选 exact-query ChEMBL context 和 shared-assay enrichment。默认 benchmark 不开启，
  避免 prospective evaluation 数据泄漏。

reasoning_batch.py
  多分子 batch orchestration，包括 molecule 级并行、日志、trace 合并、断点续跑、
  predictions/metrics/report 输出。支持 `--groups` 透传给 task pipeline，用于 targeted
  group smoke test；metrics 包含 positive-class precision/recall/F1、confusion matrix 和
  prediction distribution。
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
  --group-workers 4 \
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

查看 trace：

```bash
# standalone single runs
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/<task_name>/reasoning/single_runs \
  8776

# batch runs
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/<task_name>/reasoning/batches \
  8776
```

新增 task 时必须检查 `tools/trace_viewer/viewer.html` 是否已经适配该 task 的 structured output。
尤其要确认 final prediction 字段、`key_evidence` 中 task-specific effect 字段
（例如 `effect_on_bbb_reasoning`、`effect_on_bioavailability_reasoning`、
`effect_on_clintox_reasoning`）和新增 summary 字段会被正确渲染；否则 trace 原始 JSON 有值，
viewer 页面也可能显示为空。

## MiniMol baseline

MiniMol baseline 代码放在：

```text
baselines/minimol/
  run_bioavailability_ma.py
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

运行环境和实现注意事项：

```text
conda env: intern
MiniMol 源码参考: /data1/tianang/Projects/minimol

实际运行优先使用 intern 环境已安装的 minimol 包。源码目录中的
minimol/ckpts/minimol_v1/state_dict.pth 当前是 Git LFS pointer，不是可直接 torch.load 的权重。

runner 内部做了两个兼容 patch：
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
  run_benchmark.py
  run_llm_benchmark.py
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
  本地 trace 可视化页面。支持选择 run、选择 molecule trace package、查看单个分子的
  single/group/final messages、reasoning、tool calls 和 parsed JSON response。新增 task
  或新增 task-specific structured field 时，需要同步检查 viewer 渲染逻辑。

tools/trace_viewer/start_viewer.sh
  启动通用 trace viewer 的静态 HTTP server。查看 standalone 单分子 run 时指向
  reasoning/single_runs；查看 batch run 时指向 reasoning/batches。

tools/chembl_tool/tasks/bbb_martins/
  其他 BBB evidence 清洗、打分、报告和输出汇总脚本。
```

## ClinTox 代码入口

ClinTox 的 task-specific 细节记录在：

```text
tools/chembl_tool/tasks/clintox/AGENTS.md
```

当前 ClinTox 状态是归档 / stress-test，而不是继续优化的主线任务。结论：

```text
ClinTox 可以复用当前 ChEMBL evidence retrieval + reasoning workflow，但不适合作为该系统的
主要 benchmark 分类任务。

核心原因是 label ontology 和 ChEMBL evidence ontology 不完全匹配：
  ClinTox 的 Y=1/Y=0 是高层 clinical toxicity / clinical failure 类二分类；
  ChEMBL 检索到的 evidence 更多是 heterogeneous toxicity liability，包括 hERG、5-HT2B、
  CYP/transporter/DDI、cell viability、DILI、LD50、MTD、organ stress 等。

这些 evidence 对 toxicity risk explanation 有价值，但很多并不等价于 ClinTox-positive。
因此系统容易把机制性 liability 或 broad medicinal-chemistry risk 解释成 toxic，导致 FP 偏多。
同时一些 ClinTox label 本身有边界噪声，例如 test set 中存在同 InChIKey connectivity
但 label 相反的分子对。
```

已归档的主要结果：

```text
v7 full final-only, missing rerun 合并估计：
  batch: outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_full_prompt_v7_final_only_from_v2
  fill:  outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_full_prompt_v7_missing_rerun_from_v2
  estimate: TN=220 FP=48 FN=12 TP=6, macro-F1 ~0.523, positive F1 ~0.167

v8 keygroups smoke:
  batch: outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_group_prompt_v8_keygroups_smoke
  targeted 9 examples: TN=1 FP=3 FN=1 TP=4, macro-F1=0.50, positive recall=0.80

keygroups 的含义：
  手动只选择更接近 ClinTox label 的 high-value endpoint groups 进入 targeted smoke，
  例如 clinical toxicity/MTD、in vivo toxicity/LD50/NOAEL、DILI、hepatic injury、
  mitochondrial stress、DNA damage、general cytotoxicity，以及少量 off-target/CYP/transporter
  作为背景。

实验结论：
  keygroups 能救回部分 positive examples（例如 idx73、idx250）并保住部分 TP
  （例如 idx84、idx170），说明 final context selection/compression 是有效方向；
  但 FP 仍然顽固（例如 idx40、idx56、idx65），idx124 仍不稳定。
```

后续维护原则：

```text
1. 保留 ClinTox 代码、AGENTS.md、audit 脚本和已产出的 batch 结果用于复现和案例分析。
2. 不再继续围绕 ClinTox macro-F1 做 prompt 迭代，除非明确把目标改成 dataset-specific calibration。
3. 如果未来重启 ClinTox，应优先做 final-context compression/filter，而不是继续堆 final prompt：
   把 evidence 分成 direct severe clinical anchor、in vivo dose-limiting anchor、
   mechanistic liability、weak/background context；机制性 liability 不能单独决定 toxic。
4. ClinTox 更适合作为 toxicity evidence retrieval / mechanistic risk explanation 的 stress test，
   不适合作为证明通用 workflow 有效性的主任务。主线任务应优先选择 label 与 ChEMBL evidence
   语义更一致的 endpoint。
```

主要入口：

```text
tools/chembl_tool/tasks/clintox/constants.py
  ClinTox label 和 prediction mapping。当前约定：Y=1 -> toxic，Y=0 -> non_toxic。

tools/chembl_tool/tasks/clintox/rules.py
  ClinTox assay screening 关键词、negative keywords、weak/context-dependent terms、
  toxicology target genes 和 assay family 配置。

tools/chembl_tool/tasks/clintox/scoring.py
  ClinTox assay 保留/剔除和打分入口。screen_assays.py 和 rescore_outputs.py 都调用 scored_row()。

tools/chembl_tool/tasks/clintox/endpoint_groups.py
  ClinTox Tier.endpoint_group、evidence_direction、evidence_strength 和 endpoint assignment 规则。

tools/chembl_tool/tasks/clintox/build_evidence_library.py
  ClinTox evidence library 构建入口。默认读取 assay_screening/v6，输出
  clintox_molecule_evidence.jsonl、clintox_neighbor_index.pkl 和 meta。

tools/chembl_tool/tasks/clintox/retrieve_neighbors.py
  ClinTox analog retrieval 入口。默认 top-k-per-group=3、min-similarity=0.3。

tools/chembl_tool/tasks/clintox/run_reasoning_pipeline.py
  ClinTox 单分子 reasoning pipeline：retrieval prefetch、single-molecule branch、group-level
  并发 reasoning、final summary、trace 保存，以及 final-only rerun。支持 `--groups` 做 targeted
  endpoint-group smoke test。

tools/chembl_tool/tasks/clintox/run_reasoning_batch.py
  ClinTox 批量 reasoning wrapper。复用 common reasoning_batch.py，输出 predictions、metrics、
  report、logs、runs 和 combined trace。

tools/chembl_tool/tasks/clintox/audit_reasoning_batch.py
  ClinTox batch 诊断入口。读取已有 predictions/final/group 输出，不重跑 LLM；汇总 FP/FN/TP/TN、
  evidence category、group evidence direction/confidence/transferability、service/group errors，以及
  test set 中同 InChIKey connectivity 但 label 相反的分子对。默认输出到目标 batch 的 `audit/`。
```

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
默认必须关闭，不进入 retrieval、single-molecule prompt、group prompt 或 batch 评估。只有显式传：

```bash
--enable-chembl-exact-context
```

才允许作为 retrospective / evidence-rich case study 使用。默认 benchmark、accuracy 和 macro-F1 报告
都应保持该开关关闭。

## LLM reasoning 流程

BBB_Martins reasoning 当前分为并发 evidence branches 和 final summary。

### Group-level reasoning

每个 `Tier.endpoint_group` 独立执行：

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

available tools:
  molecule_properties

not used:
  ChEMBL neighbor evidence
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

这个分支只能看到 `molecule_properties`，不能看到 ChEMBL neighbor evidence 或分子比较工具。

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

viewer 启动：

```bash
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/bbb_martins/reasoning/single_runs \
  8776
```

然后打开：

```text
http://localhost:8776/.trace_viewer.html
```

常用 pipeline 命令：

```bash
# 完整运行一个 test_efflux 分子
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

# 批量运行一个 JSONL 中的分子，并生成评估报告
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch \
  --input-jsonl data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl \
  --parallelism 1 \
  --group-workers 4 \
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

`trace_messages.jsonl` 会合并每个 molecule 的 trace。查看 batch trace 时启动 viewer 指向
`reasoning/batches`，然后选择 `<batch_id>`：

```bash
bash tools/trace_viewer/start_viewer.sh \
  outputs/chembl_tool/tasks/bbb_martins/reasoning/batches \
  8776
```

viewer 可以选择 `<batch_id>`，再通过 `Molecule package` 下拉框切换分子。
若传 `--no-save-trace`，不会生成 batch combined trace。

注意：viewer 对常见字段有专门渲染逻辑。新增 task 时要同步适配
`tools/trace_viewer/viewer.html`，至少检查 prediction 字段、final summary 字段和
`key_evidence` 里的 task-specific effect 字段；否则后台 trace 正常保存，页面仍可能把对应列显示为空。

### LLM usage 与成本估算

DeepSeek API response 会返回 token usage，但不会在每次 response 中直接返回美元费用。
估算 batch 费用时用 trace 中保存的 usage 汇总 token，再乘以 DeepSeek 官方 pricing。
价格会变，生产估算前必须先查官方页面：

```text
https://api-docs.deepseek.com/quick_start/pricing/
```

`tools/trace_viewer` 会在页面中按当前填写的 token 单价估算费用：左侧显示 loaded trace
总费用和当前筛选结果费用，每条 trace item 显示单条估算费用，详情页显示 cache hit、
cache miss 和 output 三部分拆分。viewer 默认价格使用下面的 `deepseek-v4-pro` 当前折扣价；
价格变化时直接在 viewer 页面改三个 USD / 1M tokens 输入框。

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

当前调用方式：

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

建议边界：

```text
1. retrieval / evidence prefetch 的逻辑粒度是 endpoint group。
2. group-level reasoning 并发粒度也是 endpoint group。
3. final reasoning 必须等待所有 group reasoning 完成。
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

新增 task 的 trace schema 如果引入 task-specific 字段，也要同步检查
`tools/trace_viewer/viewer.html`，确保 viewer 能显示新的 prediction、summary 和 key evidence
effect 字段。

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

## 下一步

当前通用工具层暂时冻结，BBB_Martins 单分子端到端 MVP 已能运行。下一步优先级：

```text
1. 批量评估 test_efflux.jsonl，形成 prediction/label 对照表和错误分析。
2. 继续审计 final summary 的证据加权，必要时增加 explicit adjudication fields。
3. 如果需要让其他系统复用 retrieval，再把 chembl_neighbors 包装为 service tool 或 task endpoint；当前 BBB pipeline 继续把它作为内部 evidence prefetch。
4. 后续接入更多常驻 ML tools，例如更慢的 pKa/logD、solubility、PK 或 toxicity 模型。
```
