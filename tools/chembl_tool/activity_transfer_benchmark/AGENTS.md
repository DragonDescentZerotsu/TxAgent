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
  build_dynamic_v1_mlp_splits.py
  build_oral_bioavailability_transfer_dataset.py
  build_oral_bioavailability_pair_splits.py
  export_oral_bioavailability_hf_upload.py
  export_oral_bioavailability_clean_hf_upload.py
  build_single_endpoint_raw_transfer_splits.py
  build_single_endpoint_exhaustive_raw_pairs.py
  sample_exhaustive_compact_pairs_to_jsonl.py
  materialize_hf_valid_split.py
  prepare_hf_mlp_features.py
  train_hf_mlp_baseline.py
  run_llm_benchmark.py
  run_qwen3_4b_valid20k_four_settings.sh
  plot_llm_run_comparison.py
  plot_llm_multi_run_comparison.py
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

build_dynamic_v1_mlp_splits.py
  将 chembl36_activity_transfer_dynamic_v1 的 continuous_pairs.tsv.gz 转换成 HF prompt/completion/
  metadata JSONL split，供 prepare_hf_mlp_features.py 和 train_hf_mlp_baseline.py 复用。
  默认 endpoint-disjoint split；也支持更严格的 pChEMBL delta 标签阈值。这个脚本仍是 pChEMBL
  endpoint benchmark，不用于 raw `% inhibition` 单 endpoint 版本。

build_oral_bioavailability_transfer_dataset.py
  从 HuggingFace `starling-labs/Oral_Bioavailability` 构建 clean oral bioavailability evidence。
  默认保留 absolute；当前主版本保留 absolute、unspecified、systemic_availability，只要
  oral_bioavailability_value 能安全解析成 numeric F%。输出 line-level clean rows、dropped rows、
  aggregate molecule records、pair candidates、eval pairs、HF prompt/completion JSONL 和 value 分布图。

build_oral_bioavailability_pair_splits.py
  从 oral bioavailability aggregate_molecules.jsonl 构建大规模 HF prompt/completion transfer split。
  支持 unordered_pair_random 和 molecule_disjoint；molecule_disjoint 会按 canonical SMILES 全局分配
  split，并只写 split 内部 pair，保证同一个 molecule 不跨 train/validation/test。

export_oral_bioavailability_hf_upload.py
  将 Oral_Bioavailability transfer/direction molecule-disjoint split 包装成 Hugging Face dataset repo
  目录。只读取现有 split，不修改本地训练/评估用原始 JSONL。输出 data/{train,validation,test}.jsonl.gz、
  README.md、export_summary.json 和 source_summary.json。导出时统一 metadata 字段，例如
  completion_a_label、completion_b_label、label_text、benchmark_version、source_dataset 和 split_mode。

export_oral_bioavailability_clean_hf_upload.py
  将 Oral_Bioavailability clean numeric data 包装成 Hugging Face dataset repo 目录。只读取
  absolute_unspecified_systemic_broad_condition_full_text_v1，不修改本地 clean/pair/prompt 数据。
  输出 data/aggregate_molecules.jsonl.gz、data/molecule_records.jsonl.gz、README.md、export_summary.json、
  source_summary.json、source_report_zh.md 和 value_distribution figure。README 中明确 upstream
  starling-labs/Oral_Bioavailability 未在 HF metadata 中检测到显式 license 字段。

build_single_endpoint_raw_transfer_splits.py
  面向没有 pChEMBL 的单个 assay endpoint 构建 raw-value transfer JSONL。当前默认 endpoint 是
  CHEMBL4513218 / inhibition；标签按 raw standard_value 的 endpoint sample std 定义：
  similar <= 0.5 std，different >= 1.5 std，中间 ambiguous 排除。支持 full_range 和
  no_lt_minus_100 两个 value version，支持 molecule_disjoint split，并在 sampled 200k 版本中
  计算 Tanimoto / similarity_bucket metadata。

build_single_endpoint_exhaustive_raw_pairs.py
  为 CHEMBL4513218 / inhibition 生成 split 内部 exhaustive non-ambiguous raw-value pairs。
  输出是 compact TSV.GZ shards，不是 prompt/completion JSONL；保留 molecule-disjoint split，
  只写同一 split 内部 pair，避免 train/validation/test molecule 混用。为节省时间和空间，
  compact shards 不包含 Tanimoto。

sample_exhaustive_compact_pairs_to_jsonl.py
  从 build_single_endpoint_exhaustive_raw_pairs.py 的 compact shards 流式采样 HF prompt/completion
  JSONL。默认采样 20,000,000 pairs，按 7:1:2 输出 train/validation/test。采样是 split 内部、
  无放回、按源 shard 顺序 exact sequential sampling；metadata 保留 activity_a/activity_b/delta，
  但 prompt 不暴露 raw activity，当前 MLP feature pipeline 也不使用 metadata activity。

materialize_hf_valid_split.py
  将 HF prompt/completion/metadata 数据集的 validation split 落盘到
  outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/<dataset>/。
  默认用于 proper_assay_transfer 两个 dataset，limit=20000；实际 full validation 为 19,986 行。
  不默认下载 train/test 全量 split；summary.json 记录 label、bucket、
  assay_type 和 weighted_tanimoto 分布。

prepare_hf_mlp_features.py
  为 trained MLP baseline 准备 HF assay-transfer 特征缓存。输入 HF prompt/completion/metadata
  JSONL，解析 endpoint 完整描述和 Molecule A/B SMILES；endpoint 用 Qwen3-Embedding-8B
  计算 semantic embedding；molecule 默认用 RDKit 计算 Morgan fingerprint 和全量 RDKit descriptors，
  也支持 `--molecule-feature-backend molformer` 用 HuggingFace
  `ibm-research/MoLFormer-XL-both-10pct` 的 `AutoModel(...).pooler_output` 替换 RDKit feature；
  输出 clean_splits、endpoints/molecules metadata、endpoint_embeddings.npy、molecule_features.npz
  和 row_indices/*.npz。实现使用 streaming 清洗 train JSONL，避免 1200 万行 full train 一次性驻留内存；
  RDKit worker 会将 OMP/MKL/OPENBLAS/RDKIT 线程设为 1，防止外层多进程和内部线程互相争抢。
  MolFormer 使用 `--molformer-devices auto` 时会把所有可见 CUDA GPU 都作为 worker 使用，即使单 GPU
  也通过 spawn 子进程隔离 RDKit 和 MolFormer runtime；当前 vllm env 的 transformers 需要脚本内
  compatibility shim、rotary cache rebuild 和 grad-enabled forward 后立即 detach，manifest 会记录
  molformer model、dtype、max_length、random_seed 和 resolved_devices。当前环境下 MolFormer
  remote code 的 rotary `inv_freq` 是 non-persistent buffer，加载后可能是未初始化/非有限值；
  脚本会先重建 `inv_freq`，再重建 cos/sin cache，并强制 feature_map eval deterministic。这个修复后
  多 seed 小样本和 256 molecule 抽样的官方 `pooler_output` 均为 finite。每个 MolFormer worker 会把分配到的 physical GPU
  remap 成进程内 `cuda:0`；这是为了避免直接使用 `cuda:6` 等非零 device id 时的 NaN，同时仍然
  并行使用所有 visible GPUs。如果 `pooler_output` 非有限，脚本会使用最深的 finite hidden_state
  做 masked mean pooling，并记录 `molformer_hidden_state_fallback` / `molformer_hidden_fallback_done`。
  支持 --test-jsonl 从头构建 train/validation/test 统一 cache；尚未实现对已有 cache 的
  incremental append mode。GPU embedding 必须在 sandbox 外运行；sandbox 内可能 CUDA 不可见。

train_hf_mlp_baseline.py
  读取 prepare_hf_mlp_features.py 的 preprocessed cache，训练 Qwen endpoint embedding +
  molecule feature 的 Lightning MLP baseline。模型使用 endpoint tower、molecule/pair tower
  和 fusion head；RDKit backend 输入包括 endpoint embedding、Mol A/B Morgan fingerprint、
  fingerprint XOR、Mol A/B standardized descriptors 和 descriptor absolute difference；
  MolFormer backend 输入包括 Mol A embedding、Mol B embedding 和 absolute embedding difference。
  训练支持 A100 bf16-mixed、
  DDP 多 GPU、W&B 记录、Lightning 进度条、定期 full validation、按 similarity_bucket/assay_type/
  eval_subset 的分组指标、predictions.jsonl/metrics.json/report_zh.md 输出，以及 best/final/last
  checkpoint 保存。full validation callback 在 DDP 下会用 barrier 同步所有 rank；best/final checkpoint
  是 rank0-only 的 torch state_dict checkpoint，last.ckpt 由 Lightning ModelCheckpoint 保存。

run_llm_benchmark.py
  用 OpenAI-compatible endpoint 跑 LLM activity-transfer 判断。
  默认支持本地 vLLM gpt-oss-120b，也可跑 DeepSeek/OpenAI-compatible hosted endpoint。
  可选调用 tool server 的 mmp_structure_compare 和 properties_compare。
  支持原 dynamic_v1/task-assay JSONL，也支持 HF prompt/completion/metadata 格式：
  completion A/B 映射为 similar/different，metadata 原样保留到 input_record.hf_metadata。
  HF metadata 中的 similarity_bucket、assay_type 会进入 metrics/report 的分组指标。
  --model 可直接传本地 vLLM 暴露的模型名，例如 gpt-oss-120b、qwen3-4b、qwen3-8b。
  默认 --output-mode json；小模型可用 --output-mode choice，让模型只输出 A/B，
  choice 模式未显式传 --max-tokens 时默认 max_tokens=1，且不会追加 JSON 输出指令。
  reasoning 默认不强制开启；--enable-thinking 会按模型名对 Qwen 发送 chat_template_kwargs enable_thinking=true，
  对非 Qwen 发送 thinking enabled；--disable-thinking 会对 Qwen/vLLM 发送 enable_thinking=false。
  默认 max-tool-rounds=3；使用 --skip-existing 断点续跑。
  progress 输出包含 completed/total、elapsed、ETA、rate、macro-F1 和 failed。
  输出 per-sample JSON、predictions、metrics、report、SVG 和 trace_messages.jsonl；当输入含 task_name 时，
  metrics/report 会额外输出 per-task performance；当输入含 HF metadata 时，会额外输出 per-assay_type
  per-similarity_bucket 和 per-eval_subset performance。
  trace_messages.jsonl 使用 tools/trace_viewer/viewer.html 可识别的格式，
  每个 sample 一行，并把 reasoning_content 放进 assistant message 的 reasoning 字段供 viewer 展示。

run_qwen3_4b_valid20k_four_settings.sh
  顺序跑 qwen3-4b proper valid20k 四个标准 setting：no-props choice/no-thinking、
  no-props json/thinking、props choice/no-thinking、props json/thinking；每步都用
  --skip-existing 支持断点续跑，最后调用 plot_llm_multi_run_comparison.py 生成 qwen3-4b comparison。
  可用环境变量覆盖 PYTHON_BIN、MODEL、BASE_URL、API_KEY、PARALLELISM、TIMEOUT_S、
  PROGRESS_EVERY、MAX_TOKENS_THINK、OUT_ROOT、COMPARISON_DIR。

plot_llm_run_comparison.py
  对两个 LLM run 和 full-valid baseline 做汇总可视化。
  默认比较 HF assay-mol-disjoint no-tanimoto valid10k 上的 gpt-oss-120b 与 DeepSeek-v4-pro，
  输出 overall、similarity_bucket、assay_type 三层 macro-F1 对比图和 TSV/report。
  还输出 true-label subset recall：true similar recall 用于看 positive transfer / scaffold-hop，
  true different recall 用于看 negative transfer / activity-cliff；label-specific 图的 x-axis
  用 S=<true similar count>、D=<true different count> 标出每组 full-valid 样本量。

plot_llm_multi_run_comparison.py
  plot_llm_run_comparison.py 的通用多 run 版本。用重复的 --run label=path 传入任意多个
  llm_runs，和 full-valid baseline 一起输出同样格式的 dashboard、overall、similarity_bucket、
  assay_type、eval_subset、label-specific recall 图、TSV 和 report。读取旧 run 时如果 metrics.json
  缺 per_eval_subset，会从 predictions.jsonl 重新计算。后续新模型/新 prompt setting 优先用这个入口；
  旧的 two-run 脚本暂时保留用于一键复现 valid10k gpt-oss/DeepSeek 图。
```

## 输出组织约定

```text
outputs/chembl_tool/activity_transfer_benchmark/
  hf_jiosephlee_valid10k/<hf-dataset-name>/
  hf_jiosephlee_valid20k/<hf-dataset-name>/
    validation.jsonl
    summary.json
  hf_jiosephlee_train/<hf-dataset-name>/
    train.jsonl
    summary.json
  dynamic_v1_mlp_splits/<split-id>/
    train.jsonl
    validation.jsonl
    test.jsonl
    summary.json
  single_endpoint_raw_transfer/
    <single-endpoint-sampled-run-id>/
      train.jsonl
      validation.jsonl
      test.jsonl
      summary.json
    exhaustive_compact/<single-endpoint-compact-run-id>/
      summary.json
      shards/*.tsv.gz
  mlp_baselines/qwen3_embedding_rdkit_v1/
    preprocessed/
      clean_splits/
      endpoints.jsonl
      endpoint_embeddings.npy
      molecules.jsonl
      molecule_features.npz
      row_indices/
      manifest.json
  llm_runs/<run_id>/
    manifest.json
    predictions.jsonl
    metrics.json
    report_zh.md
    trace_messages.jsonl
    runs/
  comparisons/hf_jiosephlee_valid10k/<hf-dataset-name>/
  comparisons/hf_jiosephlee_valid20k/<comparison-name>/
    comparison_report.md
    comparison_metrics.tsv
    figures/
      comparison_dashboard.*
      label_recall_by_similarity_bucket.*
      label_recall_by_assay_type.*
      label_recall_by_eval_subset.*
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

旧的四个 chembl-mol12 dataset 只 materialize validation 10k split；不要默认下载全量 split。

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid10k/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/
  chembl-mol12-stdsep-mol-disjoint-no-props/
  chembl-mol12-stdsep-mol-disjoint-no-props-no-tanimoto/
```

### HF jiosephlee valid20k proper assay transfer

两个 proper_assay_transfer dataset 已 materialize 完整 validation split；实际每个 split 为 19,986 行。

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/
  proper_assay_transfer_no_prop_no_tanimoto/
  proper_assay_transfer_no_tanimoto/
```

两者 schema 仍是 prompt/completion/metadata，可直接输入 run_llm_benchmark.py。
`proper_assay_transfer_no_prop_no_tanimoto` 不含 properties / Tanimoto；
`proper_assay_transfer_no_tanimoto` 含 molecule properties，但不含 Tanimoto。

当前 full-validation baseline：

```text
rows: 19,986
labels: similar 10,893 / different 9,093
Tanimoto>=0.50 macro-F1: 0.5335
Bucket-majority macro-F1: 0.4799
```

当前 qwen3-8b full-validation runs（排除 1-sample smoke；两个旧的误导性 properties/json run 已删除）：

```text
llm_runs/qwen3_8b_proper_no_prop_no_tanimoto_valid20k_choice_no_thinking/
  no properties, choice/no-thinking, macro-F1 0.5120

llm_runs/qwen3_8b_proper_no_prop_no_tanimoto_valid20k_choice_with_thinking/
  no properties, json/thinking trace, macro-F1 0.5069

llm_runs/qwen3_8b_proper_no_tanimoto_valid20k_choice_no_thinking_fixed/
  properties, choice/no-thinking, macro-F1 0.4603

llm_runs/qwen3_8b_proper_no_tanimoto_valid20k_json_with_thinking_trace/
  properties, fixed json/thinking trace, macro-F1 0.5349
```

当前 qwen3-8b valid20k comparison：

```text
outputs/chembl_tool/activity_transfer_benchmark/comparisons/hf_jiosephlee_valid20k/qwen3_8b_proper_valid20k/
  comparison_report.md
  comparison_metrics.tsv
  figures/
    comparison_dashboard.*
    overall_metrics.*
    label_recall_by_similarity_bucket.*
    label_recall_by_assay_type.*
    label_recall_by_eval_subset.*
```

当前 qwen3-4b full-validation runs：

```text
llm_runs/qwen3_4b_proper_no_prop_no_tanimoto_valid20k_choice_no_thinking/
  no properties, choice/no-thinking, macro-F1 0.5273

llm_runs/qwen3_4b_proper_no_prop_no_tanimoto_valid20k_json_with_thinking_trace/
  no properties, json/thinking trace, macro-F1 0.4708

llm_runs/qwen3_4b_proper_no_tanimoto_valid20k_choice_no_thinking/
  properties, choice/no-thinking, macro-F1 0.4888

llm_runs/qwen3_4b_proper_no_tanimoto_valid20k_json_with_thinking_trace/
  properties, json/thinking trace, macro-F1 0.5236
```

当前 qwen3-4b valid20k comparison：

```text
outputs/chembl_tool/activity_transfer_benchmark/comparisons/hf_jiosephlee_valid20k/qwen3_4b_proper_valid20k/
  comparison_report.md
  comparison_metrics.tsv
  figures/
    comparison_dashboard.*
    overall_metrics.*
    label_recall_by_similarity_bucket.*
    label_recall_by_assay_type.*
    label_recall_by_eval_subset.*
```

当前 gpt-oss-120b valid20k run：

```text
llm_runs/gpt_oss_120b_proper_no_tanimoto_valid20k_json_with_thinking_trace/
  properties, json/thinking trace, macro-F1 0.5434
  reasoning_content present in sampled run JSON and trace_messages.jsonl.
```

当前 gpt-oss-120b valid20k comparison：

```text
outputs/chembl_tool/activity_transfer_benchmark/comparisons/hf_jiosephlee_valid20k/gpt_oss_120b_proper_valid20k/
  comparison_report.md
  comparison_metrics.tsv
  figures/
    comparison_dashboard.*
    overall_metrics.*
    label_recall_by_similarity_bucket.*
    label_recall_by_assay_type.*
```

### HF proper assay-transfer MLP baseline preprocessing

当前已为 `jiosephlee/proper_assay_transfer_no_prop_no_tanimoto` 建立 trained MLP baseline 的
train/validation 特征缓存。这个缓存只包含 train 和 validation；test split 尚未 append。

本地 full train 已 materialize：

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_train/proper_assay_transfer_no_prop_no_tanimoto/
  train.jsonl
  summary.json

rows: 12,327,157
labels: A/similar 6,957,330; B/different 5,369,827
```

Qwen3-Embedding-8B 已下载到：

```text
/data1/tianang/cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
```

正式 preprocessed cache：

```text
outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_v1/preprocessed/
  clean_splits/train.jsonl
  clean_splits/validation.jsonl
  clean_splits/train_invalid_rows.jsonl
  clean_splits/validation_invalid_rows.jsonl
  endpoints.jsonl
  endpoint_embeddings.npy
  molecules.jsonl
  molecule_features.npz
  row_indices/train.npz
  row_indices/validation.npz
  manifest.json
```

当前 full cache 统计：

```text
train rows: 12,327,157
validation rows: 19,986
invalid rows: 0
unique endpoints: 154,370
unique molecules: 1,337,289
invalid molecules: 0

endpoint_embeddings.npy: (154370, 4096) float16
molecule_features.npz:
  fingerprints: (1337289, 2048) uint8
  descriptors:  (1337289, 217) float32
row_indices/train.npz labels:
  different 5,369,827; similar 6,957,330
row_indices/validation.npz labels:
  different 9,093; similar 10,893
```

正式构建命令模板：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  RDKIT_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  /data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.activity_transfer_benchmark.prepare_hf_mlp_features \
    --train-jsonl outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_train/proper_assay_transfer_no_prop_no_tanimoto/train.jsonl \
    --validation-jsonl outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/proper_assay_transfer_no_prop_no_tanimoto/validation.jsonl \
    --out-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_v1/preprocessed \
    --embedding-model /data1/tianang/cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
    --embedding-batch-size 32 \
    --devices auto \
    --rdkit-workers 256 \
    --progress-every 50000
```

运行说明：

```text
1. `clean_records` 阶段只做 JSONL streaming 解析和 clean split 写出，不使用 GPU。
2. `endpoint_embedding_start` 后才会加载 Qwen3-Embedding-8B 并使用 GPU。
3. `rdkit_features` 阶段使用 CPU 多进程计算 Morgan fingerprint 和 RDKit descriptors。
4. `TOKENIZERS_PARALLELISM=false` 是为了避免 tokenizer 内部线程和多进程/GPU worker 抢 CPU；
   不影响 GPU 并行。
5. sentence-transformers 的 `encode_multi_process` deprecation warning 和 multiprocessing
   resource_tracker semaphore warning 不影响已写出的 cache；以 manifest、array shape 和 row count
   为准。
```

后续需要给 test split 加特征时，优先实现 incremental append mode：

```text
输入现有 preprocessed/ + test JSONL；
只解析 test；
只补算 missing endpoint embeddings 和 missing molecule RDKit features；
重写 endpoints/molecules/feature arrays；
新增 row_indices/test.npz；
不重算已有 154,370 endpoints 和 1,337,289 molecules。
```

MLP 训练入口：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  /data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.activity_transfer_benchmark.train_hf_mlp_baseline \
    --preprocessed-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_v1/preprocessed \
    --run-id qwen3_embedding_rdkit_mlp_v1 \
    --batch-size 4096 \
    --eval-batch-size 8192 \
    --num-workers 8 \
    --eval-num-workers 4 \
    --devices auto \
    --strategy ddp \
    --precision bf16-mixed \
    --max-steps 5000 \
    --val-every-steps 250 \
    --checkpoint-every-steps 1000 \
    --wandb-mode online
```

训练输出目录：

```text
outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_v1/runs/<run_id>/
  config.json
  descriptor_stats.npz
  predictions.jsonl
  metrics.json
  best_predictions.jsonl
  best_metrics.json
  report_zh.md
  manifest.json
  checkpoints/
    best.ckpt
    final.ckpt
    last.ckpt
```

训练实现注意事项：

```text
1. endpoint embedding 已经 L2-normalized，训练时用 LayerNorm，不额外 z-score。
2. Morgan fingerprint 是 uint8 0/1，训练 collate 时转 float；不输入 Tanimoto scalar。
3. RDKit descriptors 用 train molecule set 计算 mean/std，NaN/inf 填 0，z-score 后 clip 到 [-10, 10]。
4. validation full metrics 复用 run_llm_benchmark.compute_metrics 的字段形状，`llm` key 表示 MLP prediction。
5. W&B 会记录 overall macro-F1/accuracy/recall，以及 similarity_bucket、assay_type、eval_subset
   下的 macro-F1 和 label-specific recall。
6. 当前推荐命令的 `--max-steps 5000` 在 8 GPU、per-GPU batch size 4096 下约等于 13.3 epochs；
   Lightning 进度条中的 `Epoch N/-2` 是 max_steps 模式下的显示占位，训练停止条件看 global_step。
7. full validation 在 rank0 上生成 metrics/report/predictions 并更新 best checkpoint；其他 DDP rank
   会在 barrier 等待，避免 validation 后继续训练时出现 NCCL allreduce timeout。
```

### dynamic_v1 endpoint-disjoint MLP train-eval v1

用途：

```text
把 chembl36_activity_transfer_dynamic_v1 的 pChEMBL-delta non-ambiguous pairs
转换成 HF prompt/completion/metadata 兼容格式，复用 Qwen endpoint embedding +
RDKit molecule feature 的 MLP baseline，评估 trained classifier 是否超过 Tanimoto threshold。
```

数据与 split：

```text
source pairs:
  outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/continuous_pairs.tsv.gz
endpoint context:
  outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/continuous_assay_endpoint_summary.tsv
split output:
  outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2/

label rule:
  similar:   |delta pChEMBL| <= 0.5
  different: |delta pChEMBL| >= 1.0
  ambiguous: excluded
split mode:
  endpoint_disjoint by assay_id + standard_type
counts:
  train 1,051,257 rows; validation 150,180 rows; test 300,359 rows
  indexed train rows after RDKit filtering: 1,051,253
  validation/test indexed rows: all rows
```

构建 split：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2 \
  --progress-every 500000
```

特征 cache：

```text
output:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_dynamic_v1/preprocessed/
endpoint_embeddings.npy:
  (19987, 4096) float16
molecule_features.npz:
  fingerprints (569058, 2048) uint8
  descriptors  (569058, 217) float32
invalid molecule:
  1 invalid SMILES with [Ar], causing 4 train pairs skipped
descriptor handling:
  abs(descriptor) > 1e12 is stored as NaN; prevents RDKit Ipc outliers from overflowing train mean/std.
```

特征构建命令：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RDKIT_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  /data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.activity_transfer_benchmark.prepare_hf_mlp_features \
    --train-jsonl outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2/train.jsonl \
    --validation-jsonl outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2/validation.jsonl \
    --test-jsonl outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2/test.jsonl \
    --out-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_dynamic_v1/preprocessed \
    --embedding-model /data1/tianang/cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
    --embedding-batch-size 64 \
    --devices auto \
    --rdkit-workers 200
```

训练与结果：

```text
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/qwen3_embedding_rdkit_dynamic_v1/runs/dynamic_v1_endpoint_disjoint_7_1_2_mlp_v1/
recommended setting:
  max_steps 100; val_every_steps 25; batch_size 4096; precision bf16-mixed; strategy ddp
best checkpoint:
  step 50 selected by validation macro-F1
validation:
  MLP macro-F1 0.5856
  Tanimoto>=0.50 macro-F1 0.5715
test:
  MLP macro-F1 0.5989; accuracy 0.5992; balanced accuracy 0.5992
  Tanimoto>=0.50 macro-F1 0.5798
notes:
  final checkpoint at step 100 overfits and is worse; report best checkpoint.
  valid-tuned threshold around 0.524 does not improve test macro-F1 over fixed 0.5.
  W&B keys include best/validation/*, test/best/*, and test/final/*.
```

### dynamic_v1 strict pChEMBL split candidate

用途：

```text
更干净标签版 activity-transfer data，用于判断原 0.5/1.0 pChEMBL delta 标签是否过噪。
只重定义 label/split data，尚未训练 MLP。
```

配置与输出：

```text
source:
  chembl36_activity_transfer_dynamic_v1/continuous_pairs.tsv.gz
label rule:
  similar:   |delta pChEMBL| <= 0.3
  different: |delta pChEMBL| >= 1.2
  middle: excluded
split mode:
  endpoint_disjoint by assay_id + standard_type, 7:1:2
output:
  outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2_strict_pchembl_0p3_1p2/
counts:
  total 1,083,804
  train 758,663; validation 108,381; test 216,760
  validation labels: different 60,605; similar 47,776
  test labels: different 123,707; similar 93,053
same-set Tanimoto baseline:
  validation best threshold ~0.57, macro-F1 0.5940
  test best threshold ~0.55, macro-F1 0.5958
  test Tanimoto>=0.50 macro-F1 0.5906
```

构建命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits \
  --similar-delta 0.3 \
  --different-delta 1.2 \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2_strict_pchembl_0p3_1p2 \
  --progress-every 500000
```

### CHEMBL4513218 / inhibition raw-value single endpoint

用途：

```text
研究没有 pChEMBL 的大规模 publication assay endpoint 是否能构造 raw-value activity-transfer benchmark。
该 endpoint 是 CHEMBL4513218 / inhibition，standard_units 为 %，来自 DOI 10.1021/acsinfecdis.9b00482，
assay 描述为 P. berghei liver stage luciferase screen at 10uM。
```

重要数据事实：

```text
pChEMBL rows: 0
molecules with fingerprints used by benchmark: 68,570
raw standard_value range:
  min -336.0
  median 18.6
  max 100.0
  sample std 33.705262
label rule:
  similar   abs(delta raw %) <= 0.5 std = 16.852631
  different abs(delta raw %) >= 1.5 std = 50.557893
  ambiguous middle region excluded
negative inhibition values are valid noisy assay readouts; do not clip to 0-100 by default.
```

200k sampled HF JSONL versions:

```text
outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer/
  CHEMBL4513218_inhibition_full_range_raw_std_sim_le_0p5_diff_ge_1p5_molecule_disjoint_7_1_2_n200000/
    train.jsonl      140,000
    validation.jsonl  20,000
    test.jsonl        40,000
    summary.json

  CHEMBL4513218_inhibition_no_lt_minus_100_raw_std_sim_le_0p5_diff_ge_1p5_molecule_disjoint_7_1_2_n200000/
    train.jsonl      140,000
    validation.jsonl  20,000
    test.jsonl        40,000
    summary.json
```

200k 构建命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_single_endpoint_raw_transfer_splits \
  --n-pairs 200000 \
  --split-mode molecule_disjoint \
  --versions full_range,no_lt_minus_100
```

200k full_range 结果：

```text
MLP best test:
  macro-F1 0.5292
  accuracy 0.5483
  balanced accuracy 0.5333
Tanimoto>=0.50 baseline on same test:
  macro-F1 0.3101
  accuracy 0.4489
  balanced accuracy 0.5000
reason:
  molecule-disjoint random pairs are overwhelmingly low-Tanimoto, so Tanimoto>=0.5 predicts almost all pairs
  as different. MLP is better than this baseline, but absolute performance is still weak.
```

Exhaustive compact full_range pairs:

```text
outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer/exhaustive_compact/
  CHEMBL4513218_inhibition_full_range_raw_std_sim_le_0p5_diff_ge_1p5_molecule_disjoint_7_1_2_exhaustive_nonambig_compact/
    summary.json
    shards/*.tsv.gz
```

Planned non-ambiguous split-internal pair counts:

```text
train:
  molecules 47,999
  non-ambiguous pairs 644,570,479
  similar 351,118,893
  different 293,451,586
validation:
  molecules 6,857
  non-ambiguous pairs 13,190,178
  similar 7,144,196
  different 6,045,982
test:
  molecules 13,714
  non-ambiguous pairs 52,544,847
  similar 28,957,144
  different 23,587,703
total non-ambiguous pairs: 710,305,504
```

Exhaustive compact 命令：

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_single_endpoint_exhaustive_raw_pairs \
  --value-version full_range \
  --progress-every 5000000
```

20M sampled full_range JSONL:

```text
outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer/
  CHEMBL4513218_inhibition_full_range_raw_std_sim_le_0p5_diff_ge_1p5_molecule_disjoint_7_1_2_n20000000/
    train.jsonl       14,000,000
    validation.jsonl   2,000,000
    test.jsonl         4,000,000
    summary.json

directory size: about 44G
labels:
  train      similar 7,626,387; different 6,373,613
  validation similar 1,082,960; different   917,040
  test       similar 2,203,617; different 1,796,383
```

20M sampling command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.sample_exhaustive_compact_pairs_to_jsonl \
  --n-pairs 20000000 \
  --out-root outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer \
  --progress-every 1000000
```

20M preprocessing and MLP run:

```text
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/CHEMBL4513218_full_range_20m/preprocessed/
  rows train/validation/test: 14,000,000 / 2,000,000 / 4,000,000
  unique endpoints: 1
  unique molecules: 68,570

run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/CHEMBL4513218_full_range_20m/runs/molecule_disjoint_20m_mlp_v2/
setting:
  max_steps 10,000
  batch_size 4,096
  val_every_steps 2,000
  precision bf16-mixed
  strategy ddp
best checkpoint:
  step 6,000 selected by validation macro-F1
validation:
  macro-F1 0.5375
  accuracy 0.5910
  balanced accuracy 0.5667
  similar recall 0.8598
  different recall 0.2736
test:
  macro-F1 0.5439
  accuracy 0.5995
  balanced accuracy 0.5699
  similar recall 0.8611
  different recall 0.2787
valid-tuned threshold:
  default threshold 0.5 is poorly calibrated; score distribution is saturated near 1.
  valid best threshold is about 0.99, giving valid macro-F1 about 0.565.
  applying threshold 0.99 to test gives macro-F1 0.5717 and balanced accuracy 0.5811.
AUROC:
  validation about 0.602
  test about 0.607
```

20M caveats:

```text
1. Compact/exhaustive 20M data does not contain Tanimoto. Do not report a 20M Tanimoto>=0.50 baseline
   unless Tanimoto is recomputed for those pairs. The 200k sampled full_range set is the same-endpoint set
   where Tanimoto metadata exists.
2. The prompt and current MLP features do not expose Molecule A/reference raw activity. Metadata contains
   activity_a/activity_b for audit, but prepare_hf_mlp_features.py only parses endpoint text and Mol A/B SMILES.
   Therefore this 20M task is closer to pairwise raw-activity prediction than to strict transferability
   ("known reference activity transfers to query").
3. 20M pairs are not 20M independent SAR observations. They are dense combinations of 68,570 molecules from
   one endpoint; endpoint embedding is constant, and pair labels are highly correlated through molecule activities.
4. A 300k validation sample showed Tanimoto has almost no same-set signal for this construction:
   median Tanimoto about 0.13, Tanimoto AUROC for similar about 0.506, and mean Tanimoto is nearly identical
   for similar and different pairs.
5. Interpretation: MLP learns weak structure/activity signal and beats the trivial Tanimoto>=0.5 rule on
   the 200k set, but CHEMBL4513218 raw `% inhibition` is a noisy phenotypic HTS endpoint; current formulation
   should not be used as evidence that a reference-activity-aware transfer model cannot work.
```

Recommended next experiment:

```text
Build a reference-activity-aware version: expose Molecule A raw activity in the prompt and add it as a scalar
feature to the MLP. That matches the intended transfer question: given reference activity, decide whether it
transfers to Molecule B. If that version still performs near macro-F1 0.55, then the assay's SAR transfer
signal is likely genuinely weak.
```

旧 valid10k 已完成 full LLM run 的 setting：

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

### starling-labs Oral_Bioavailability transfer dataset

非 ChEMBL/Joseph Lee 来源的 oral bioavailability transfer benchmark。

源数据：

```text
HuggingFace: starling-labs/Oral_Bioavailability
split: train
rows: 163,815
columns:
  pmid, support_text, molecule_name, oral_bioavailability_value,
  bioavailability_report_type, species_or_population, dose,
  oral_exposure_mode, qualifying_conditions, comparator, extra_details, smiles
```

清洗规则：

```text
主版本保留 report types:
  absolute, unspecified, systemic_availability
丢弃:
  relative_comparison、不能安全解析成 numeric explicit value 的行、RDKit invalid SMILES、
  默认 0-1000% 范围外值。
value parser:
  about/~approximately: 取主 numeric value
  per cent/percent: 统一成 %
  mean/average/median: 优先取对应值
  x ± y: 取 x
  x to y / x-y range: 取 midpoint
  无 % 且 0<=value<=1.5: 按 fraction 转成 percent
  AUC-only、fold/higher/lower/comparable 等 relative 描述: 丢弃
condition_text:
  写入所有实验条件字段和值；空值写 not specified。
  字段包括 species_or_population, dose, oral_exposure_mode,
  qualifying_conditions, comparator, extra_details。
metadata:
  原始 source row 完整保存，方便后续重清洗。
```

clean evidence 主输出：

```text
outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf/
  absolute_unspecified_systemic_broad_condition_full_text_v1/
    molecule_records.jsonl
    dropped_rows.jsonl
    aggregate_molecules.jsonl
    eval_pairs.jsonl
    hf_prompt_completion.jsonl
    summary.json
    report_zh.md
    figures/value_distribution.svg

clean numeric rows: 82,496
aggregate molecule-condition records: 66,154
unique condition groups: 37,883
value median/mean/max: 42.0 / 46.1 / 942.0 %
```

构建 clean evidence：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_transfer_dataset \
  --run-id absolute_unspecified_systemic_broad_condition_full_text_v1 \
  --allowed-report-types absolute,unspecified,systemic_availability
```

pair label：

```text
similar:   |delta oral bioavailability percentage points| <= 10
different: |delta oral bioavailability percentage points| >= 30
ambiguous: excluded
Pairs are generated within the same condition_key.
```

condition_key 注意事项：

```text
主版本使用 broad_condition:
  species_or_population | oral_exposure_mode | qualifying_conditions | comparator
dose 和 extra_details 不进 key，但一定保留在 condition_text/prompt/metadata。
```

#### Oral Bioavailability MLP: non-molecule-disjoint diagnostic

先做过 directed max / unordered-pair split 版本，结果很高但不代表 molecule 泛化。

```text
split:
  outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/
    absolute_unspecified_systemic_directed_max_7_1_2_v2_same_unordered_split/
rows:
  train 3,391,750; validation 484,534; test 969,068
rule:
  同一个 unordered pair 的 A->B/B->A 保持同 split，但 molecule 可跨 split。
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_directed_max_qwen3_embedding_rdkit_v2_same_unordered_split/preprocessed/
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_directed_max_qwen3_embedding_rdkit_v2_same_unordered_split/runs/
    oral_bioavailability_directed_max_mlp_v2_same_unordered_split/
best validation:
  step 5000 macro-F1 0.9830, accuracy 0.9861
best test:
  macro-F1 0.9821, accuracy 0.9854
interpretation:
  这是 leakage-prone diagnostic，不作为 prospective 泛化结果。
```

#### Oral Bioavailability MLP: molecule-disjoint main result

严格 molecule-disjoint 版本：按 canonical SMILES 全局分配 split，同一个 SMILES 不跨 split；
只写 split 内部 pair。为了让 pair 数接近 7:1:2，用 molecule split ratio 约
0.522774/0.197629/0.279597，因为同 split pair 数近似随 molecule fraction 平方缩放。

```text
split:
  outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/
    absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1/
molecule split counts:
  train 7,089; validation 2,680; test 3,791
SMILES overlap check:
  train∩validation 0; train∩test 0; validation∩test 0
max directed non-ambiguous pairs after molecule-disjoint filtering:
  1,890,486
rows:
  train 1,312,272; validation 177,854; test 400,360
label counts:
  train similar 382,160 / different 930,112
  validation similar 51,266 / different 126,588
  test similar 112,510 / different 287,850
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed/
cache:
  unique endpoints 2,563; unique molecules 9,859; invalid rows/molecules 0
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_pairratio_7_1_2_qwen3_embedding_rdkit_v1/runs/
    oral_bioavailability_molecule_disjoint_mlp_v1/
best validation:
  step 4000 macro-F1 0.5404, accuracy 0.6429, balanced accuracy 0.5397
  similar recall 0.2961, different recall 0.7833
best test:
  macro-F1 0.5528, accuracy 0.6700, balanced accuracy 0.5516
  similar recall 0.2812, different recall 0.8220
conclusion:
  Strict molecule-disjoint 泛化很弱，明显低于 non-molecule-disjoint 的约 0.98 macro-F1。
  该结果说明前者的高分主要来自 molecule/pair overlap；后续报告应使用 molecule-disjoint 版本。
```

构建 molecule-disjoint split：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_pair_splits \
  --run-id absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1 \
  --split-mode molecule_disjoint \
  --ratios 0.522774,0.197629,0.279597 \
  --target-pairs 30000000
```

预处理：

```bash
env CUDA_VISIBLE_DEVICES=4 \
  /data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.prepare_hf_mlp_features \
  --train-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1/train.jsonl \
  --validation-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1/validation.jsonl \
  --test-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1/test.jsonl \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-batch-size 64 \
  --devices auto \
  --rdkit-workers 128
```

训练：

```bash
env CUDA_VISIBLE_DEVICES=4 \
  /data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.train_hf_mlp_baseline \
  --preprocessed-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed \
  --out-root outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_pairratio_7_1_2_qwen3_embedding_rdkit_v1/runs \
  --run-id oral_bioavailability_molecule_disjoint_mlp_v1 \
  --accelerator gpu \
  --devices 1 \
  --strategy auto \
  --precision bf16-mixed \
  --batch-size 4096 \
  --eval-batch-size 8192 \
  --num-workers 8 \
  --eval-num-workers 4 \
  --max-steps 5000 \
  --val-every-steps 500 \
  --full-eval-every-n-validation 1 \
  --checkpoint-every-steps 1000 \
  --wandb-mode disabled
```

#### Oral Bioavailability MLP: molecule-disjoint higher/lower ordered result

有向 ordered 版本不再判断 transfer similar/different，而是判断 Molecule B/query 的 F% 是否高于
Molecule A/reference。每个可用 unordered pair 写两个方向：A->B 和 B->A；因此只要两者 F%
不完全相等，就会成对贡献 `query_higher` 和 `query_lower`。`completion A=query_higher`，
`completion B=query_lower`，训练中正类概率表示 `query_higher`。

```text
split:
  outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/
    absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1/
label rule:
  query_higher: Molecule B F% > Molecule A F%
  query_lower:  Molecule B F% < Molecule A F%
  ties: exact equal values excluded; min_ordered_delta default 0.0
molecule split counts:
  train 7,089; validation 2,680; test 3,791
SMILES overlap check:
  train∩validation 0; train∩test 0; validation∩test 0
max directed pairs after molecule-disjoint filtering:
  2,607,910
rows:
  train 1,813,174; validation 247,026; test 547,710
label balance:
  train query_higher 906,587 / query_lower 906,587
  validation query_higher 123,513 / query_lower 123,513
  test query_higher 273,855 / query_lower 273,855
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed/
cache:
  unique endpoints 2,908; unique molecules 10,208; invalid rows/molecules 0
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/runs/
    oral_bioavailability_molecule_disjoint_higher_lower_mlp_v1/
best validation:
  step 500 macro-F1 0.6614, accuracy 0.6617, balanced accuracy 0.6617
  query_higher recall 0.6311, query_lower recall 0.6924
test with best checkpoint:
  macro-F1 0.6721, accuracy 0.6724, balanced accuracy 0.6724
  query_higher recall 0.6446, query_lower recall 0.7002
test with final checkpoint:
  macro-F1 0.6757, accuracy 0.6757, balanced accuracy 0.6757
  query_higher recall 0.6791, query_lower recall 0.6723
note:
  manifest selects best checkpoint by validation macro-F1, so canonical reported test metric is best-checkpoint
  macro-F1 0.6721. Final checkpoint has slightly higher observed test macro-F1 but should not be selected by test.
```

MolFormer pooler-fix molecule embedding run:

```text
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_endpoint_molformer_poolerfix_v1/preprocessed/
cache:
  endpoint embeddings 2,908 x 4,096
  MolFormer molecule embeddings 10,208 x 768, float16, finite, L2-normalized
  invalid rows/molecules 0
  MolFormer batch size 64, random_seed 2, all 8 visible A100 GPUs used
  no hidden-state fallback logged after rebuilding rotary inv_freq + cos/sin cache
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_endpoint_molformer_poolerfix_v1/runs/
    oral_bioavailability_molecule_disjoint_higher_lower_molformer_poolerfix_mlp_v1/
wandb:
  https://wandb.ai/reasonv/txagent-assay-transfer-mlp/runs/twv8678e
best validation:
  step 1000 macro-F1 0.6427, accuracy 0.6427, balanced accuracy 0.6427
  query_higher recall 0.6562, query_lower recall 0.6292
test with best checkpoint:
  macro-F1 0.6671, accuracy 0.6672, balanced accuracy 0.6672
  query_higher recall 0.6818, query_lower recall 0.6526
test with final checkpoint:
  macro-F1 0.6669, accuracy 0.6669, balanced accuracy 0.6669
  query_higher recall 0.6720, query_lower recall 0.6618
note:
  Fixing MolFormer pooler extraction removes the large gap to RDKit: canonical best-checkpoint
  test macro-F1 is 0.6671 vs RDKit 0.6721. RDKit final checkpoint remains highest observed
  test macro-F1 at 0.6757, but final checkpoint is not selected by validation.
```

MolFormer molecule embedding run below was generated before the rotary `inv_freq` rebuild fix; it is finite because
the script fell back to hidden-state pooling for many molecules, but should be regenerated before treating
`pooler_output` as the molecule feature source.

MolFormer molecule embedding run:

```text
preprocessed:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_endpoint_molformer_v1/preprocessed/
cache:
  endpoint embeddings 2,908 x 4,096
  MolFormer molecule embeddings 10,208 x 768, float16, finite, L2-normalized
  invalid rows/molecules 0
run:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_endpoint_molformer_v1/runs/
    oral_bioavailability_molecule_disjoint_higher_lower_molformer_mlp_v1/
wandb:
  https://wandb.ai/reasonv/txagent-assay-transfer-mlp/runs/l2jsdbma
best validation:
  step 1500 macro-F1 0.5868, accuracy 0.5874, balanced accuracy 0.5874
  query_higher recall 0.5503, query_lower recall 0.6245
test with best checkpoint:
  macro-F1 0.6061, accuracy 0.6066, balanced accuracy 0.6066
  query_higher recall 0.5733, query_lower recall 0.6398
test with final checkpoint:
  macro-F1 0.6058, accuracy 0.6058, balanced accuracy 0.6058
  query_higher recall 0.6038, query_lower recall 0.6078
note:
  MolFormer embedding underperforms the RDKit fingerprint+descriptor baseline on this ordered
  molecule-disjoint benchmark. Canonical best-checkpoint test macro-F1 is 0.6061 vs RDKit 0.6721.
```

LLM benchmark support:

```text
run_llm_benchmark.py 支持该 ordered HF prompt/completion 格式。它会从 metadata 读取：
  completion_a_label=query_higher
  completion_b_label=query_lower
然后自动切换到 ordered prompt/schema：
  predicted_direction: query_higher or query_lower
而不是旧 transfer benchmark 的 predicted_transferability=similar/different。

Tool calling 仍复用同一个 OpenAI-compatible runner，允许 LLM 调用：
  mmp_structure_compare
  properties_compare
工具服务入口仍是 http://127.0.0.1:8765。

注意：
  对 higher/lower ordered task，不再使用 Tanimoto>=0.50 作为 meaningful baseline；
  report 中主要看 LLM 和 bucket-majority 等可用 baseline。
```

构建 ordered molecule-disjoint split：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_pair_splits \
  --run-id absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1 \
  --split-mode molecule_disjoint \
  --label-mode higher_lower \
  --ratios 0.522774,0.197629,0.279597 \
  --target-pairs 30000000
```

预处理时必须显式传入 ordered label mapping：

```bash
env CUDA_VISIBLE_DEVICES=4 \
  /data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.prepare_hf_mlp_features \
  --train-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1/train.jsonl \
  --validation-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1/validation.jsonl \
  --test-jsonl outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1/test.jsonl \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-batch-size 64 \
  --devices auto \
  --rdkit-workers 128 \
  --completion-a-label query_higher \
  --completion-b-label query_lower
```

ordered MLP 训练 / W&B 入口：

```bash
env CUDA_VISIBLE_DEVICES=4 \
  /data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.train_hf_mlp_baseline \
  --preprocessed-dir outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/preprocessed \
  --out-root outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/runs \
  --run-id oral_bioavailability_molecule_disjoint_higher_lower_mlp_wandb_v1 \
  --accelerator gpu \
  --devices 1 \
  --strategy auto \
  --precision bf16-mixed \
  --batch-size 4096 \
  --eval-batch-size 8192 \
  --num-workers 8 \
  --eval-num-workers 4 \
  --max-steps 5000 \
  --val-every-steps 500 \
  --full-eval-every-n-validation 1 \
  --checkpoint-every-steps 1000 \
  --wandb-mode online \
  --wandb-project txagent-assay-transfer-mlp
```

W&B rerun:

```text
output:
  outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/
    oral_bioavailability_hf_molecule_disjoint_higher_lower_pairratio_7_1_2_qwen3_embedding_rdkit_v1/runs/
    oral_bioavailability_molecule_disjoint_higher_lower_mlp_wandb_v1/
wandb:
  https://wandb.ai/reasonv/txagent-assay-transfer-mlp/runs/482gdneb
logged:
  train/loss, val/loss, val_macro_f1, validation class recalls, best/test metrics.
observation:
  val macro-F1 was already near plateau at step 500; later steps mainly changed class-bias/calibration.
  Train loss kept dropping while val loss increased, so report macro-F1 from full validation rather than val loss.
```

MLP feature contract:

```text
prepare_hf_mlp_features.py embeds only the endpoint/context block, not Molecule A and Molecule B
condition text separately. The endpoint text is the prompt section between `## Endpoint` and
`## Molecule A`; for oral higher/lower this section contains both reference and query experimental
contexts. Qwen3-Embedding-8B produces one normalized endpoint embedding per endpoint_key.

Default RDKit molecule pair features:
  fp_A
  fp_B
  fp_A XOR fp_B
  descriptor_A
  descriptor_B
  abs(descriptor_A - descriptor_B)

Optional MolFormer molecule pair features (`--molecule-feature-backend molformer`):
  molformer_emb_A
  molformer_emb_B
  abs(molformer_emb_A - molformer_emb_B)

The MLP has an endpoint tower and a pair tower, then concatenates both representations before
the final fusion MLP. It does not separately encode Molecule A text or Molecule B text.
```

Ordered full-validation LLM run:

```text
run:
  outputs/chembl_tool/activity_transfer_benchmark/llm_runs/
    oral_bioavailability_higher_lower_gpt_oss_120b_9001_validation_full_tools/
input:
  validation.jsonl from absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1
model/base_url:
  gpt-oss-120b via http://127.0.0.1:9001/v1, api key EMPTY
tool service:
  http://127.0.0.1:8765
command shape:
  --parallelism 64 --max-tool-rounds 3 --max-tokens 10240 --output-mode json --skip-existing
state as of 2026-06-14:
  still running; manifest exists but final metrics.json is not written yet.
  User-observed progress around 108,800/247,026 reported macro-F1 about 0.6565.
  A local same-index snapshot over 107,496 ok saved samples gave LLM macro-F1 0.6564.
same-index MLP comparison on those 107,496 samples:
  MLP final macro-F1 0.6706
  MLP best-valid macro-F1 0.6729
interpretation:
  LLM is close to the MLP baseline but still slightly lower in the fair same-index comparison.
  Current LLM predictions are biased toward query_lower, with query_higher recall about 0.50 and
  query_lower recall about 0.83 in that snapshot.
metric caveat:
  progress macro-F1 is ok-only. Failed samples are counted in progress logs but do not necessarily
  enter the saved per-sample JSON metric until final merge/reporting, so final strict metrics should
  check how failures are handled.
trace viewer:
  run_llm_benchmark.py writes trace_messages.jsonl at finalization. View with:
  bash tools/trace_viewer/start_viewer.sh outputs/chembl_tool/activity_transfer_benchmark/llm_runs 8776
```

Hugging Face exports:

```text
export root:
  outputs/chembl_tool/activity_transfer_benchmark/hf_upload_exports/

clean numeric repo:
  https://huggingface.co/datasets/Kiria-Nozan/Starling-bioavailability-clean
  local export: hf_upload_exports/clean/
  tables:
    aggregate_molecules: 66,154 rows
    molecule_records: 82,496 rows
  files:
    README.md
    source_summary.json
    source_report_zh.md
    export_summary.json
    data/aggregate_molecules.jsonl.gz
    data/molecule_records.jsonl.gz
    figures/value_distribution.svg
    figures/value_distribution.png

transfer repo:
  https://huggingface.co/datasets/Kiria-Nozan/Starling-bioavailability-transfer
  local export: hf_upload_exports/transfer/
  rows: train 1,312,272; validation 177,854; test 400,360
  labels: A=similar, B=different

direction repo:
  https://huggingface.co/datasets/Kiria-Nozan/Starling-bioavailability-direction
  local export: hf_upload_exports/direction/
  rows: train 1,813,174; validation 247,026; test 547,710
  labels: A=query_higher, B=query_lower

HF upload status:
  transfer and direction repos were created and uploaded on 2026-06-14.
  clean numeric repo was created and uploaded on 2026-06-15.
```

Export command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.export_oral_bioavailability_clean_hf_upload \
  --progress-every 25000

python -m tools.chembl_tool.activity_transfer_benchmark.export_oral_bioavailability_hf_upload \
  --datasets transfer direction \
  --progress-every 250000
```

Upload commands:

```bash
hf repo create Kiria-Nozan/Starling-bioavailability-clean --repo-type dataset --exist-ok
hf repo create Kiria-Nozan/Starling-bioavailability-transfer --repo-type dataset --exist-ok
hf repo create Kiria-Nozan/Starling-bioavailability-direction --repo-type dataset --exist-ok

hf upload Kiria-Nozan/Starling-bioavailability-clean \
  outputs/chembl_tool/activity_transfer_benchmark/hf_upload_exports/clean . \
  --repo-type dataset \
  --commit-message "Add clean numeric oral bioavailability tables"

hf upload Kiria-Nozan/Starling-bioavailability-transfer \
  outputs/chembl_tool/activity_transfer_benchmark/hf_upload_exports/transfer . \
  --repo-type dataset \
  --commit-message "Add molecule-disjoint oral bioavailability transfer benchmark"

hf upload Kiria-Nozan/Starling-bioavailability-direction \
  outputs/chembl_tool/activity_transfer_benchmark/hf_upload_exports/direction . \
  --repo-type dataset \
  --commit-message "Add molecule-disjoint oral bioavailability direction benchmark"
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

## 历史 3K benchmark 后续计划

下面是最初完成 `dynamic_v1_llm_3k` 时记录的消融清单，不再代表当前项目优先级。第 4 项 learned
classifier 已由本文件前面的 endpoint-disjoint、HF proper-assay-transfer 和 Oral Bioavailability MLP
实验实质完成；不得因为这份旧清单再次把它登记为“尚未开始”。其余三项只有在重新进入该 3K
stress-test 研究问题时才执行，当前优先级以论文执行计划为准。

```text
1. no-tools ablation：同一个 3K set，不允许工具调用。
2. no-MCS-in-prompt ablation：保留 Tanimoto 和 assay context，移除 MCS coverage。
3. full-distribution eval：从 dynamic_v1 自然分布抽样，和 full-data Tanimoto threshold 更公平比较。
4. learned classifier：已由后续 MLP 系列实验取代并完成，不再是待办项。
```
