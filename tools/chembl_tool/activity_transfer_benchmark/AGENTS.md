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
  benchmark_mcs_runtime.py
  analyze_mcs_results.py
  build_llm_eval_set.py
  run_llm_benchmark.py
```

脚本职责：

```text
run_benchmark.py
  从 ChEMBL 读取 assay activity，构建同 assay endpoint 内的 molecule pairs。
  连续值主分析使用 pchembl_value，计算 Tanimoto、|delta pChEMBL|、标签、threshold metrics、
  assay-specific enrichment、binary comment 辅助分析，并生成 TSV/GZ、SVG 和中文 report。

benchmark_mcs_runtime.py
  对已有 continuous_pairs 计算 RDKit FindMCS mean atom coverage。
  支持 sampled runtime benchmark、full-scan 流式计算、timeout、进度输出和 finalize-existing 收尾。

analyze_mcs_results.py
  读取 MCS 结果，扫描 mean MCS coverage threshold，并在同一 observed subset 上重扫 Tanimoto。
  输出 MCS/Tanimoto 对比指标、bucket summary、heatmap、SVG 和中文 report。

build_llm_eval_set.py
  从 dynamic_v1 pairs 中构建 LLM 小评估集。默认 3,000 pairs，按 label x Tanimoto bucket 分层平衡，
  并只保留有 observed MCS 的 non-ambiguous pairs。

run_llm_benchmark.py
  用 OpenAI-compatible endpoint 跑 LLM activity-transfer 判断。
  默认支持本地 vLLM gpt-oss-120b，也可跑 DeepSeek/OpenAI-compatible hosted endpoint。
  可选调用 tool server 的 mmp_structure_compare 和 properties_compare。
  输出 per-sample JSON、predictions、metrics、report、SVG 和 trace_messages.jsonl。
  trace_messages.jsonl 使用 tools/trace_viewer/viewer.html 可识别的格式，
  每个 sample 一行，并把 reasoning_content 放进 assistant message 的 reasoning 字段供 viewer 展示。
```

## 版本索引

### smoke runs

历史 smoke / debug 输出，用于检查流程和图表，不作为正式结论：

```text
outputs/chembl_tool/activity_transfer_benchmark/smoke/
outputs/chembl_tool/activity_transfer_benchmark/smoke_binary/
outputs/chembl_tool/activity_transfer_benchmark/smoke_dynamic_filter/
outputs/chembl_tool/activity_transfer_benchmark/smoke_enrichment/
```

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
