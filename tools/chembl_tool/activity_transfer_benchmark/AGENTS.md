# ChEMBL Activity Transfer Benchmark

## 目标

这个 benchmark 研究：在同一个 ChEMBL assay endpoint 内，已知 reference molecule 的
pChEMBL activity 时，能否只根据 query/reference 的结构和 assay context 判断 activity 是否可迁移。

当前标签规则：

```text
similar:   |delta pChEMBL| <= 0.5
different: |delta pChEMBL| >= 1.0
ambiguous: 中间区间，只用于数据分析，不进入二分类评估
```

主数据源：

```text
tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

主输出根目录：

```text
outputs/chembl_tool/activity_transfer_benchmark/
```

## 代码入口

```text
tools/chembl_tool/activity_transfer_benchmark/
  run_benchmark.py
  run_task_assay_benchmark.py
  benchmark_mcs_runtime.py
  analyze_mcs_results.py
  build_llm_eval_set.py
  build_task_llm_eval_set.py
  run_llm_benchmark.py
  plot_llm_run_comparison.py
```

脚本职责：

```text
run_benchmark.py
  从 ChEMBL 读取 assay activity，构建同 assay endpoint 内的 molecule pairs。
  连续值主分析使用 pchembl_value，计算 Tanimoto、|delta pChEMBL|、标签、threshold metrics、
  assay-specific enrichment、binary comment 辅助分析，并生成 TSV/GZ、SVG 和中文 report。

run_task_assay_benchmark.py
  从 data/processed 四个任务 pipeline 的 assay evidence 出发，构建 task-scoped transfer benchmark。
  支持 raw_robust_z、log_raw_robust_z、pchembl_delta 三套标签；raw/log raw 标签会做单位归一化和 assay 内 robust sigma。

benchmark_mcs_runtime.py
  对已有 continuous_pairs 计算 RDKit FindMCS mean atom coverage。
  支持 sampled runtime benchmark、full-scan 流式计算、timeout、进度输出和 finalize-existing 收尾。

analyze_mcs_results.py
  读取 MCS 结果，扫描 mean MCS coverage threshold，并在同一 observed subset 上重扫 Tanimoto。
  输出 MCS/Tanimoto 对比指标、bucket summary、heatmap、SVG 和中文 report。

build_llm_eval_set.py
  从 dynamic_v1 pairs 中构建 LLM 小评估集。默认 3,000 pairs，按 label x Tanimoto bucket 分层平衡，
  并只保留有 observed MCS 的 non-ambiguous pairs。

build_task_llm_eval_set.py
  从 task-scoped pairs 中构建小规模 LLM 评估集。默认 3,000 pairs，先按 task 平衡，再按 label 和
  Tanimoto bucket 分层，并限制每个 assay endpoint 的样本数。

run_llm_benchmark.py
  用 OpenAI-compatible endpoint 跑 LLM activity-transfer 判断。
  默认支持本地 vLLM gpt-oss-120b，也可跑 DeepSeek/OpenAI-compatible hosted endpoint。
  可选调用 tool server 的 mmp_structure_compare 和 properties_compare。
  支持原 dynamic_v1/task-assay JSONL，也支持 HF prompt/completion/metadata 格式：
  completion A/B 映射为 similar/different，metadata 原样保留到 input_record.hf_metadata。
  HF metadata 中的 similarity_bucket、assay_type 会进入 metrics/report 的分组指标。
  默认 max-tool-rounds=3；使用 --skip-existing 断点续跑。
  输出 per-sample JSON、predictions、metrics、report、SVG 和 trace_messages.jsonl；当输入含 task_name 时，
  metrics/report 会额外输出 per-task performance；当输入含 HF metadata 时，会额外输出 per-assay_type
  和 per-similarity_bucket performance。
  trace_messages.jsonl 使用 tools/trace_viewer/viewer.html 可识别的格式，
  每个 sample 一行，并把 reasoning_content 放进 assistant message 的 reasoning 字段供 viewer 展示。

plot_llm_run_comparison.py
  对两个 LLM run 和 full-valid baseline 做汇总可视化。
  默认比较 HF assay-mol-disjoint no-tanimoto valid10k 上的 gpt-oss-120b 与 DeepSeek-v4-pro，
  输出 overall、similarity_bucket、assay_type 三层 macro-F1 对比图和 TSV/report。
  还输出 true-label subset recall：true similar recall 用于看 positive transfer / scaffold-hop，
  true different recall 用于看 negative transfer / activity-cliff；label-specific 图的 x-axis
  用 S=<true similar count>、D=<true different count> 标出每组 full-valid 样本量。
```

## 输出组织约定

```text
outputs/chembl_tool/activity_transfer_benchmark/
  hf_jiosephlee_valid10k/<hf-dataset-name>/
    validation.jsonl
    summary.json
  llm_runs/<run_id>/
    manifest.json
    predictions.jsonl
    metrics.json
    report_zh.md
    trace_messages.jsonl
    runs/
  comparisons/hf_jiosephlee_valid10k/<hf-dataset-name>/
    comparison_report.md
    comparison_metrics.tsv
    figures/
      comparison_dashboard.*
      label_recall_by_similarity_bucket.*
      label_recall_by_assay_type.*
```

说明：

```text
smoke/debug 结果不作为长期产物保留；正式 run、输入数据和 comparison 分开存放。
HF valid10k 的原始 metadata 必须保留，后续分析会用到 similarity_bucket、assay_type、
weighted_tanimoto 等字段。
```

## 版本索引

### chembl36_activity_transfer_v1

第一版完整 Tanimoto baseline，未做 dynamic-range filter。

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_v1/
```

核心设置：

```text
max-total-pairs: 2,000,000
max-pairs-per-assay: 5,000
binary-max-total-pairs: 500,000
```

当前结论：

```text
连续值主分析约 40k assay endpoints、约 198 万 sampled pairs。
最佳单一 Tanimoto threshold 约 0.45，macro-F1 约 0.56。
assay-specific enrichment 显示 close analog 相对 assay 背景有正 lift，
但结构相似度本身不足以作为可靠 transfer 判据。
```

### chembl36_activity_transfer_dynamic_v1

当前主 baseline 版本。过滤低动态范围 assay endpoint，降低“所有 pair 都 similar”的假信号。

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/
```

核心设置：

```text
min-pchembl-range: 2.0
min-pchembl-iqr: 0.75
max-total-pairs: 2,000,000
max-pairs-per-assay: 5,000
```

当前结论：

```text
连续值主分析约 20k assay endpoints、约 199 万 sampled pairs。
similar/different 标签比未过滤 v1 更平衡，median |delta pChEMBL| 更高。
best Tanimoto threshold 约 0.50，macro-F1 / balanced accuracy 约 0.57。
这是后续 MCS 和 LLM benchmark 的主数据版本。
```

### dynamic_v1_mcs_full_t2_w128_stream

dynamic_v1 的全量 MCS 计算结果，timeout=2s，workers=128。

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/dynamic_v1_mcs_full_t2_w128_stream/
```

当前状态：

```text
total pairs: 1,989,152
completed: 1,987,665
completion rate: 99.9252%
missing: 1,487
observed timeout: 218,664, about 11.0%
```

说明：

```text
full-scan 在 tail pending futures 阶段可能卡住。
遇到卡住时终止进程，保留 mcs_sample_results.tsv.tmp，
再用 --finalize-existing 生成 summary/report/missing_result_indices.tsv。
```

### dynamic_v1_mcs_t2_analysis

MCS threshold 分析版本。

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_analysis/dynamic_v1_mcs_t2_analysis/
```

当前结论：

```text
non-ambiguous usable pairs: 1,500,676
best mean MCS coverage threshold: 0.70
best MCS macro-F1: 0.5538
best MCS balanced accuracy: 0.5542
same subset best Tanimoto threshold: 0.48
same subset best Tanimoto macro-F1: 0.5688
```

解释：

```text
MCS coverage 有 activity-transfer 信号，但单独全局 threshold 没有超过 Tanimoto。
它更适合作为 LLM / learned classifier 的补充特征。
```

### dynamic_v1_llm_3k

LLM 小评估集。它是分层平衡 stress-test，不是 full dynamic_v1 的自然分布。

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/
```

构成：

```text
samples: 3,000
assay endpoints: 2,463
labels: similar 1,500 / different 1,500
similarity buckets: 6 buckets，每个 500 pairs
```

同集合 baseline：

```text
Tanimoto>=0.50 macro-F1 0.4977, balanced accuracy 0.5020
Tanimoto>=0.48 macro-F1 0.4903, balanced accuracy 0.4963
MCS>=0.70      macro-F1 0.4941, balanced accuracy 0.4963
```

### task_assay_raw_robust_z_llm_3k

当前推荐的 task-scoped LLM 小评估集，用于比较四个 data/processed task 的 pipeline 表现和
同 task assay 上的 transfer performance。

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/task_assay_raw_robust_z_llm_3k/
  eval_pairs.jsonl
  eval_pairs.tsv
  summary.json
  report_zh.md
```

构建命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_task_llm_eval_set \
  --run-id task_assay_raw_robust_z_llm_3k \
  --n-total 3000 \
  --label-mode raw_robust_z \
  --max-per-endpoint 6
```

构成：

```text
samples: 3,000
assay endpoint groups: 1,889
labels: similar 1,500 / different 1,500
tasks: bbb_martins 750 / bioavailability_ma 750 / clintox 750 / skin_reaction 750
each task label split: similar 375 / different 375
```

同集合 baseline：

```text
Tanimoto>=0.50 macro-F1 0.4991, balanced accuracy 0.5043
Tanimoto>=0.48 macro-F1 0.4962, balanced accuracy 0.5030

Per-task Tanimoto>=0.50 macro-F1:
bbb_martins 0.4962
bioavailability_ma 0.5013
clintox 0.5106
skin_reaction 0.4881
```

### HF jiosephlee valid10k

当前只 materialize 四个 HF dataset 的 validation 10k split；不要默认下载全量 split。

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid10k/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/
  chembl-mol12-stdsep-mol-disjoint-no-props/
  chembl-mol12-stdsep-mol-disjoint-no-props-no-tanimoto/
```

已完成 full LLM run 的 setting：

```text
input:
  hf_jiosephlee_valid10k/chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/validation.jsonl
runs:
  llm_runs/gpt_oss_120b_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools/
  llm_runs/deepseek_v4_pro_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools/
comparison:
  comparisons/hf_jiosephlee_valid10k/chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/
```

当前 overall macro-F1：

```text
gpt-oss-120b:     0.5365
DeepSeek-v4-pro:  0.5068
Tanimoto >= 0.5:  0.5364
Bucket majority:  0.3606
```

LLM 运行示例：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_llm_benchmark \
  --input-jsonl outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid10k/chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/validation.jsonl \
  --run-id gpt_oss_120b_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools \
  --base-urls http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1 \
  --model gpt-oss-120b \
  --api-key EMPTY \
  --tool-service-url http://127.0.0.1:8765 \
  --parallelism 16 \
  --max-tool-rounds 3 \
  --skip-existing
```

### gpt_oss_120b_dynamic_v1_llm_3k_tools

本地 vLLM gpt-oss-120b + tool server 的第一版 3K LLM benchmark。

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/gpt_oss_120b_dynamic_v1_llm_3k_tools/
  benchmark_explanation.html
  report_zh.md
  metrics.json
  predictions.jsonl
  figures/model_vs_baselines.svg
  runs/
```

运行设置：

```text
model: gpt-oss-120b
vLLM base URLs: http://127.0.0.1:8001-8004/v1
local vLLM api key: EMPTY
tool service: http://127.0.0.1:8765
parallelism: 16
max tool rounds: 1
```

结果：

```text
n: 3,000
failed: 0
tool calls: 1,017
sample-average tool call rate: about 33.9% by call count
wall time: about 15.2 min
total tokens: 6,377,195

LLM accuracy: 0.5310
LLM balanced accuracy: 0.5310
LLM macro-F1: 0.5309

same-set Tanimoto>=0.50 macro-F1: 0.4977
same-set MCS>=0.70 macro-F1: 0.4941
```

灰区结果：

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5390
Tanimoto>=0.50 macro-F1: 0.4683
```

解释：

```text
gpt-oss-120b + tools 在 3K balanced stress-test 上超过简单 threshold baseline，
但绝对性能仍弱。该结果不能直接和 full dynamic_v1 自然分布上的 best Tanimoto macro-F1 ~0.57
一比一比较。
benchmark_explanation.html 是面向阅读的 HTML 报告，记录数据构建、方法对比、prompt、工具调用率和一个带 tool call 的可见 trace 示例。
```

### gpt_oss_120b_dynamic_v1_llm_3k_tools_thinking

本地 vLLM gpt-oss-120b + tool server + `--enable-thinking` 的 3K LLM benchmark。

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/gpt_oss_120b_dynamic_v1_llm_3k_tools_thinking/
  benchmark_explanation.html
  error_analysis_zh.md
  report_zh.md
  metrics.json
  predictions.jsonl
  figures/model_vs_baselines.svg
  runs/
```

运行设置：

```text
model: gpt-oss-120b
vLLM base URLs: http://127.0.0.1:8001-8004/v1
local vLLM api key: EMPTY
tool service: http://127.0.0.1:8765
parallelism: 16
max tool rounds: 1
max tokens: 4096
thinking: enabled
```

结果：

```text
n: 3,000
failed: 0
tool calls: 1,536
sample-level tool call rate: 50.53%
reasoning_content saved: 2,996 / 3,000
reasoning_summary saved: 3,000 / 3,000
wall time: about 14.7 min
total tokens: 8,133,936

LLM accuracy: 0.5253
LLM balanced accuracy: 0.5253
LLM macro-F1: 0.5253

same-set Tanimoto>=0.50 macro-F1: 0.4977
same-set MCS>=0.70 macro-F1: 0.4941
```

灰区结果：

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5301
Tanimoto>=0.50 macro-F1: 0.4683
```

解释：

```text
thinking 版本保存了 reasoning_content 和 reasoning_summary，适合 trace 审计；
但分类性能低于非 thinking 版本：macro-F1 0.5253 vs 0.5309。
benchmark_explanation.html 已更新为 thinking run 的结果，并包含 sample_00107 的 reasoning/tool trace 示例；
该样本是 very_close analog 但真实 different，LLM 判断正确。
error_analysis_zh.md 记录 thinking run 的错误分层、典型 FP/FN、失败原因和后续改进建议。
```

### deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking

DeepSeek-v4-pro + tool server + thinking 的 3K LLM benchmark。

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking/
  report_zh.md
  metrics.json
  predictions.jsonl
  trace_messages.jsonl
  runs/
```

运行设置：

```text
model: deepseek-v4-pro
base URL: https://api.deepseek.com
api key env: DEEPSEEK_API_KEY from .env
tool service: http://127.0.0.1:8765
parallelism: 60
max tool rounds: 3
max tokens: 20,480
thinking: enabled
```

结果：

```text
n: 3,000
ok: 2,994
failed: 6
tool calls: 6,009
sample-level tool call rate: 99.93%
reasoning_content saved: 2,994 / 2,994 ok samples
reasoning_summary saved: 2,994 / 2,994 ok samples
wall time: about 81.8 min
total tokens: 20,133,230

LLM accuracy: 0.5471
LLM balanced accuracy: 0.5472
LLM macro-F1: 0.5378
similar recall: 0.4052
different recall: 0.6892

same-success-subset Tanimoto>=0.50 macro-F1: 0.4981
same-success-subset MCS>=0.70 macro-F1: 0.4941
```

灰区结果：

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5090
Tanimoto>=0.50 macro-F1: 0.4683
```

解释：

```text
DeepSeek-v4-pro 在整体 3K stress-test 上是当前最高的 LLM run：
macro-F1 0.5378，高于 gpt-oss-120b no-thinking 的 0.5309 和 thinking 的 0.5253。
但提升幅度小，且主要来自更保守地预测 different；similar recall 明显偏低。
在 Tanimoto 0.40-0.70 的灰区，DeepSeek macro-F1 0.5090，低于两个 gpt-oss run。
因此它是 overall 最好，但不是 gray-zone 最好；考虑 tool calls、token 和 wall time 后，
当前性价比不如本地 gpt-oss。
```

## 常用命令

全量 baseline：

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

MCS full-scan：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --full-scan \
  --workers 128 \
  --timeout-s 2 \
  --chunksize 1 \
  --progress-every 10000
```

MCS 卡住后收尾：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --finalize-existing \
  --workers 128 \
  --timeout-s 2
```

MCS threshold 分析：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.analyze_mcs_results \
  --run-id dynamic_v1_mcs_t2_analysis
```

构建 3K LLM eval set：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_llm_eval_set \
  --run-id dynamic_v1_llm_3k
```

构建 task-scoped assay transfer benchmark：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_task_assay_benchmark \
  --run-id task_assay_transfer_v2 \
  --workers 256 \
  --max-pairs-per-endpoint 5000 \
  --max-total-pairs-per-mode 2000000 \
  --progress-every-endpoints 250
```

说明：

```text
run_task_assay_benchmark.py
  只使用四个 task pipeline 已筛出的 assay/activity evidence：
  BBB_Martins v6、Bioavailability_Ma v4、ClinTox v6、Skin_Reaction v1。
  默认输出三套 label mode：
    raw_robust_z:     在同 task + assay + endpoint + normalized units 内，用 raw standard_value 的 robust sigma 标注。
    log_raw_robust_z: 同上，但用 log10(raw standard_value)，更适合 IC50/EC50/Ki 等数量级型 endpoint。
    pchembl_delta:    兼容旧规则，|delta pChEMBL| <= 0.5 为 similar，>= 1.0 为 different。
  单位归一化会把 nM/uM/mM/M 统一到 nM，把常见通透率单位统一到 cm/s，
  并把常见 clearance 单位统一到 mL/min 系列；无法安全跨分子量换算的单位
  （如 ug/mL）保留为独立 normalized unit，避免混合 endpoint。
  raw/log robust sigma 使用 IQR/1.349，IQR 为 0 时 fallback 到 MAD*1.4826；
  sigma 仍为 0 的 endpoint 不进入 raw/log label mode。
```

运行本地 gpt-oss-120b LLM benchmark：

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

运行 DeepSeek-v4-pro thinking LLM benchmark / 断点续跑：

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

## 后续优先实验

```text
1. no-tools ablation：同一个 3K set，不允许工具调用。
2. no-MCS-in-prompt ablation：保留 Tanimoto 和 assay context，移除 MCS coverage。
3. full-distribution eval：从 dynamic_v1 自然分布抽样，和 full-data Tanimoto threshold 更公平比较。
4. learned classifier：用 Tanimoto、MCS、assay dynamic range、endpoint metadata 做轻量模型 baseline。
```
