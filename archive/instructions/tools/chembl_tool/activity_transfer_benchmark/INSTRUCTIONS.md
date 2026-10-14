# ChEMBL Activity Transfer Benchmark

## Active data ownership

Active data lives with its semantic owner: gold-bound data under
`data/gold_labels/<Task>/<version>/`, evidence data under its task/release, and
shared reusable caches under `data/caches/`. `data/artifacts/` is audit-only and
must not be a required build or runtime input; complete retired products belong
under `data/legacy/`. Do not add compatibility symlinks.

The evidence-library pipeline owns scientific level assignment. Preserved
gold-version mappings live under `data/gold_labels/<Task>/level_mappings/<version>/`.
BBB and Bioavailability runtime consumers use the active release-owned
`data/evidence_libraries/<task>/<release>/level_mapping/`; Ames, DILI,
Carcinogens, and Skin keep their gold-owned mappings until reviewed replacements.
Voter membership may validate L1 coverage but must never derive or rewrite levels.
Corrections and publication belong to the evidence-library pipeline and must use
reviewed UID decisions with pinned input hashes.

## Testing discipline

Do not add circular tests that merely assert newly written prompt prose or copy
implementation literals into the test. Prompt wording is validated with reviewed
input/output fixtures or a real pilot/evaluation. Automated tests should cover
executable behavior, failure modes, schemas, rendering validity, and provenance.

## Goals

This benchmark studies: within the same ChEMBL assay endpoint, given the pChEMBL activity of a known reference molecule, can the activity transferability be determined solely from the query/reference structures and assay context.

Current label rules:

```text
similar:   |delta pChEMBL| <= 0.5
different: |delta pChEMBL| >= 1.0
ambiguous: intermediate range, used only for data analysis, not in binary classification evaluation
```

Main data sources:

```text
tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db
tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz
```

Main output root directory:

```text
outputs/chembl_tool/activity_transfer_benchmark/
```

## Code entry points

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

Script responsibilities:

```text
run_benchmark.py
  Read assay activity from ChEMBL, build molecule pairs within the same assay endpoint.
  Continuous-value main analysis uses pchembl_value, computing Tanimoto, |delta pChEMBL|, labels, threshold metrics,
  assay-specific enrichment, binary-comment auxiliary analysis, and TSV/GZ, SVG, and English report outputs.

run_task_assay_benchmark.py
  Build task-scoped transfer benchmark from assay evidence of the four task pipelines in data/gold_labels/legacy/processed.
  Supports three label sets: raw_robust_z, log_raw_robust_z, pchembl_delta; raw/log raw labels undergo unit normalization and within-assay robust sigma.

benchmark_mcs_runtime.py
  Compute RDKit FindMCS mean atom coverage for existing continuous_pairs.
  Supports sampled runtime benchmark, full-scan streaming computation, timeout, progress output, and finalize-existing completion.

analyze_mcs_results.py
  Read MCS results, scan mean MCS coverage thresholds, and rescan Tanimoto on the same observed subset.
  Output MCS/Tanimoto comparison metrics, bucket summaries, heatmaps, SVG figures, and an English report.

build_llm_eval_set.py
  Build a small LLM evaluation set from dynamic_v1 pairs. Default 3,000 pairs, stratified balanced by label x Tanimoto bucket,
  and keep only non-ambiguous pairs with observed MCS.

build_task_llm_eval_set.py
  Build a small-scale LLM evaluation set from task-scoped pairs. Default 3,000 pairs, first balanced by task, then stratified by label and
  Tanimoto bucket, and limit the number of samples per assay endpoint.

build_dynamic_v1_mlp_splits.py
  Convert continuous_pairs.tsv.gz of chembl36_activity_transfer_dynamic_v1 into HF prompt/completion/
  metadata JSONL splits for reuse by prepare_hf_mlp_features.py and train_hf_mlp_baseline.py.
  Default endpoint-disjoint split; also supports stricter pChEMBL delta label thresholds. This script remains a pChEMBL
  endpoint benchmark and is not used for the raw `% inhibition` single-endpoint version.

build_oral_bioavailability_transfer_dataset.py
  Build clean oral bioavailability evidence from HuggingFace `starling-labs/Oral_Bioavailability`.
  Default keeps absolute; current main version keeps absolute, unspecified, systemic_availability, as long as
  oral_bioavailability_value can be safely parsed into numeric F%. Outputs line-level clean rows, dropped rows,
  aggregate molecule records, pair candidates, eval pairs, HF prompt/completion JSONL, and value distribution plots.

build_oral_bioavailability_pair_splits.py
  Build large-scale HF prompt/completion transfer splits from oral bioavailability aggregate_molecules.jsonl.
  Supports unordered_pair_random and molecule_disjoint; molecule_disjoint globally assigns splits by canonical SMILES
  and only writes pairs within the same split, ensuring the same molecule does not cross train/validation/test.

export_oral_bioavailability_hf_upload.py
  Wrap the Oral_Bioavailability transfer/direction molecule-disjoint split into a Hugging Face dataset repo
  directory. Only reads existing splits, does not modify local training/evaluation raw JSONL. Outputs data/{train,validation,test}.jsonl.gz,
  README.md, export_summary.json, and source_summary.json. Unifies metadata fields during export, e.g.,
  completion_a_label, completion_b_label, label_text, benchmark_version, source_dataset, and split_mode.

export_oral_bioavailability_clean_hf_upload.py
  Wrap Oral_Bioavailability clean numeric data into a Hugging Face dataset repo directory. Only reads
  absolute_unspecified_systemic_broad_condition_full_text_v1, does not modify local clean/pair/prompt data.
  Outputs data/aggregate_molecules.jsonl.gz, data/molecule_records.jsonl.gz, README.md, export_summary.json,
  source_summary.json, source_report_zh.md, and value_distribution figure. README explicitly states that upstream
  starling-labs/Oral_Bioavailability has no explicit license field detected in HF metadata.

build_single_endpoint_raw_transfer_splits.py
  Build raw-value transfer JSONL for a single assay endpoint without pChEMBL. Current default endpoint is
  CHEMBL4513218 / inhibition; labels are defined by endpoint sample std of raw standard_value:
  similar <= 0.5 std, different >= 1.5 std, intermediate ambiguous excluded. Supports full_range and
  no_lt_minus_100 value versions, supports molecule_disjoint split, and computes Tanimoto / similarity_bucket metadata in the sampled 200k version.

build_single_endpoint_exhaustive_raw_pairs.py
  Generate split-internal exhaustive non-ambiguous raw-value pairs for CHEMBL4513218 / inhibition.
  Output is compact TSV.GZ shards, not prompt/completion JSONL; preserves molecule-disjoint split,
  only writes pairs within the same split to avoid train/validation/test molecule mixing. To save time and space,
  compact shards do not include Tanimoto.

sample_exhaustive_compact_pairs_to_jsonl.py
  Stream-sample HF prompt/completion JSONL from the compact shards of build_single_endpoint_exhaustive_raw_pairs.py.
  Default samples 20,000,000 pairs, outputting train/validation/test at 7:1:2. Sampling is split-internal,
  without replacement, exact sequential sampling by source shard order; metadata retains activity_a/activity_b/delta,
  but the prompt does not expose raw activity, and the current MLP feature pipeline does not use metadata activity.

materialize_hf_valid_split.py
  Materialize the validation split of the HF prompt/completion/metadata dataset to
  outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/<dataset>/.
  Default for proper_assay_transfer two datasets, limit=20000; actual full validation is 19,986 rows.
  Does not download full train/test splits by default; summary.json records label, bucket,
  assay_type, and weighted_tanimoto distributions.

prepare_hf_mlp_features.py
  Prepare HF assay-transfer feature cache for trained MLP baseline. Input HF prompt/completion/metadata
  JSONL, parse endpoint full description and Molecule A/B SMILES; endpoint uses Qwen3-Embedding-8B
  to compute semantic embedding; molecule defaults to RDKit Morgan fingerprint and full RDKit descriptors,
  also supports `--molecule-feature-backend molformer` using HuggingFace
  `ibm-research/MoLFormer-XL-both-10pct`'s `AutoModel(...).pooler_output` to replace RDKit features;
  output clean_splits, endpoints/molecules metadata, endpoint_embeddings.npy, molecule_features.npz
  and row_indices/*.npz. Implementation uses streaming to clean train JSONL, avoiding 12 million rows of full train residing in memory at once;
  RDKit worker sets OMP/MKL/OPENBLAS/RDKIT threads to 1, preventing contention between outer multiprocessing and internal threads.
  MolFormer uses `--molformer-devices auto` and treats all visible CUDA GPUs as workers, even single GPU
  uses spawn subprocess to isolate RDKit and MolFormer runtime; current vllm env's transformers requires script-internal
  compatibility shim, rotary cache rebuild, and detach immediately after grad-enabled forward; manifest records
  molformer model, dtype, max_length, random_seed, and resolved_devices. In current environment, MolFormer
  remote code's rotary `inv_freq` is non-persistent buffer, may be uninitialized/non-finite after loading;
  script first rebuilds `inv_freq`, then rebuilds cos/sin cache, and forces feature_map eval deterministic. After this fix,
  multi-seed small samples and 256-molecule sampling official `pooler_output` are all finite. Each MolFormer worker remaps assigned physical GPU
  to in-process `cuda:0`; this avoids NaN when directly using non-zero device IDs like `cuda:6`, while still
  using all visible GPUs in parallel. If `pooler_output` is non-finite, script uses deepest finite hidden_state
  for masked mean pooling, and records `molformer_hidden_state_fallback` / `molformer_hidden_fallback_done`.
  Supports --test-jsonl building train/validation/test unified cache from scratch; incremental append mode for existing cache
  not yet implemented. GPU embedding must run outside sandbox; CUDA may not be visible inside sandbox.

train_hf_mlp_baseline.py
  Reads preprocessed cache from prepare_hf_mlp_features.py, trains Lightning MLP baseline with Qwen endpoint embedding +
  molecule features. Model uses endpoint tower, molecule/pair tower, and fusion head; RDKit backend inputs include endpoint embedding, Mol A/B Morgan fingerprint,
  fingerprint XOR, Mol A/B standardized descriptors, and descriptor absolute difference;
  MolFormer backend inputs include Mol A embedding, Mol B embedding, and absolute embedding difference.
  Training supports A100 bf16-mixed,
  DDP multi-GPU, W&B logging, Lightning progress bar, periodic full validation, grouped metrics by similarity_bucket/assay_type/
  eval_subset, predictions.jsonl/metrics.json/report_zh.md outputs, and best/final/last
  checkpoint saving. Full validation callback uses barrier to sync all ranks under DDP; best/final checkpoint
  are rank0-only torch state_dict checkpoints, last.ckpt saved by Lightning ModelCheckpoint.

run_llm_benchmark.py
  Runs LLM activity-transfer judgment with OpenAI-compatible endpoint.
  Default supports local vLLM gpt-oss-120b, also can run DeepSeek/OpenAI-compatible hosted endpoint.
  Optionally calls tool server's mmp_structure_compare and properties_compare.
  Supports original dynamic_v1/task-assay JSONL, also HF prompt/completion/metadata format:
  completion A/B maps to similar/different, metadata preserved as-is in input_record.hf_metadata.
  similarity_bucket, assay_type in HF metadata enter grouped metrics/report.
  --model can directly pass local vLLM exposed model name, e.g., gpt-oss-120b, qwen3-4b, qwen3-8b.
  Default --output-mode json; small models can use --output-mode choice, letting model output only A/B,
  choice mode defaults max_tokens=1 when --max-tokens not explicitly passed, and does not append JSON output instruction.
  reasoning not forced by default; --enable-thinking sends chat_template_kwargs enable_thinking=true to Qwen by model name,
  sends thinking enabled to non-Qwen; --disable-thinking sends enable_thinking=false to Qwen/vLLM.
  Default max-tool-rounds=3; use --skip-existing for resume from breakpoint.
  progress output includes completed/total, elapsed, ETA, rate, macro-F1, and failed.
  Outputs per-sample JSON, predictions, metrics, report, SVG, and trace_messages.jsonl; when input contains task_name,
  metrics/report additionally output per-task performance; when input contains HF metadata, additionally output per-assay_type
  per-similarity_bucket and per-eval_subset performance.
  trace_messages.jsonl uses format recognizable by tools/trace_viewer/viewer.html,
  one line per sample, and puts reasoning_content into assistant message's reasoning field for viewer display.

run_qwen3_4b_valid20k_four_settings.sh
  Sequentially runs qwen3-4b proper valid20k four standard settings: no-props choice/no-thinking,
  no-props json/thinking, props choice/no-thinking, props json/thinking; each step uses
  --skip-existing for resume, finally calls plot_llm_multi_run_comparison.py to generate qwen3-4b comparison.
  Environment variables can override PYTHON_BIN, MODEL, BASE_URL, API_KEY, PARALLELISM, TIMEOUT_S,
  PROGRESS_EVERY, MAX_TOKENS_THINK, OUT_ROOT, COMPARISON_DIR.

plot_llm_run_comparison.py
  Creates summary visualization for two LLM runs and full-valid baseline.
  Default compares gpt-oss-120b with DeepSeek-v4-pro on HF assay-mol-disjoint no-tanimoto valid10k,
  outputs overall, similarity_bucket, assay_type three-layer macro-F1 comparison plots and TSV/report.
  Also outputs true-label subset recall: true similar recall for positive transfer / scaffold-hop,
  true different recall for negative transfer / activity-cliff; label-specific plots' x-axis
  uses S=<true similar count>, D=<true different count> to mark each group's full-valid sample size.

plot_llm_multi_run_comparison.py
  General multi-run version of plot_llm_run_comparison.py. Uses repeated --run label=path to pass any number of
  llm_runs, and with full-valid baseline outputs same format dashboard, overall, similarity_bucket,
  assay_type, eval_subset, label-specific recall plots, TSV, and report. When reading old runs, if metrics.json
  lacks per_eval_subset, recomputes from predictions.jsonl. Prefer this entry for new models/new prompt settings;
  old two-run script kept temporarily for one-click reproduction of valid10k gpt-oss/DeepSeek plots.
```

## Output organization conventions

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

Notes:

```text
smoke/debug results are not kept as long-term artifacts; formal runs, input data, and comparisons are stored separately.
HF valid10k raw metadata must be preserved; later analysis uses similarity_bucket, assay_type,
weighted_tanimoto, and other fields.
```

## Version index

### chembl36_activity_transfer_v1

First full Tanimoto baseline, no dynamic-range filter.

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_v1/
```

Core settings:

```text
max-total-pairs: 2,000,000
max-pairs-per-assay: 5,000
binary-max-total-pairs: 500,000
```

Current conclusions:

```text
Continuous-value main analysis about 40k assay endpoints, about 1.98 million sampled pairs.
Best single Tanimoto threshold about 0.45, macro-F1 about 0.56.
Assay-specific enrichment shows close analogs have positive lift over assay background,
but structural similarity alone is insufficient as a reliable transfer criterion.
```

### chembl36_activity_transfer_dynamic_v1

Current main baseline version. Filters low dynamic-range assay endpoints to reduce false signal where all pairs are similar.

```text
outputs/chembl_tool/activity_transfer_benchmark/chembl36_activity_transfer_dynamic_v1/
```

Core settings:

```text
min-pchembl-range: 2.0
min-pchembl-iqr: 0.75
max-total-pairs: 2,000,000
max-pairs-per-assay: 5,000
```

Current conclusions:

```text
Continuous-value main analysis about 20k assay endpoints, about 1.99 million sampled pairs.
similar/different labels more balanced than unfiltered v1, median |delta pChEMBL| higher.
Best Tanimoto threshold about 0.50, macro-F1 / balanced accuracy about 0.57.
This is the main data version for subsequent MCS and LLM benchmarks.
```

### dynamic_v1_mcs_full_t2_w128_stream

Full MCS computation results for dynamic_v1, timeout=2s, workers=128.

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/dynamic_v1_mcs_full_t2_w128_stream/
```

Current status:

```text
total pairs: 1,989,152
completed: 1,987,665
completion rate: 99.9252%
missing: 1,487
observed timeout: 218,664, about 11.0%
```

Notes:

```text
full-scan may hang during tail pending futures phase.
If stuck, terminate process, keep mcs_sample_results.tsv.tmp,
then use --finalize-existing to generate summary/report/missing_result_indices.tsv.
```

### dynamic_v1_mcs_t2_analysis

MCS threshold analysis version.

```text
outputs/chembl_tool/activity_transfer_benchmark/mcs_analysis/dynamic_v1_mcs_t2_analysis/
```

Current conclusions:

```text
non-ambiguous usable pairs: 1,500,676
best mean MCS coverage threshold: 0.70
best MCS macro-F1: 0.5538
best MCS balanced accuracy: 0.5542
same subset best Tanimoto threshold: 0.48
same subset best Tanimoto macro-F1: 0.5688
```

Interpretation:

```text
MCS coverage has activity-transfer signal, but a single global threshold does not exceed Tanimoto.
It is better suited as a supplementary feature for LLM / learned classifiers.
```

### dynamic_v1_llm_3k

Small LLM evaluation set. It is a stratified balanced stress-test, not the natural distribution of full dynamic_v1.

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/dynamic_v1_llm_3k/
```

Composition:

```text
samples: 3,000
assay endpoints: 2,463
labels: similar 1,500 / different 1,500
similarity buckets: 6 buckets, each 500 pairs
```

Same-set baselines:

```text
Tanimoto>=0.50 macro-F1 0.4977, balanced accuracy 0.5020
Tanimoto>=0.48 macro-F1 0.4903, balanced accuracy 0.4963
MCS>=0.70      macro-F1 0.4941, balanced accuracy 0.4963
```

### task_assay_raw_robust_z_llm_3k

Currently recommended task-scoped LLM small evaluation set, used to compare pipeline performance across four data/gold_labels/legacy/processed tasks and
transfer performance on same-task assays.

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets/task_assay_raw_robust_z_llm_3k/
  eval_pairs.jsonl
  eval_pairs.tsv
  summary.json
  report_zh.md
```

Build command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_task_llm_eval_set \
  --run-id task_assay_raw_robust_z_llm_3k \
  --n-total 3000 \
  --label-mode raw_robust_z \
  --max-per-endpoint 6
```

Composition:

```text
samples: 3,000
assay endpoint groups: 1,889
labels: similar 1,500 / different 1,500
tasks: bbb_martins 750 / bioavailability_ma 750 / clintox 750 / skin_reaction 750
each task label split: similar 375 / different 375
```

Same-set baselines:

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

Old four chembl-mol12 datasets only materialize validation 10k split; do not download full split by default.

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid10k/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props/
  chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/
  chembl-mol12-stdsep-mol-disjoint-no-props/
  chembl-mol12-stdsep-mol-disjoint-no-props-no-tanimoto/
```

### HF jiosephlee valid20k proper assay transfer

Two proper_assay_transfer datasets have materialized full validation split; each split is actually 19,986 rows.

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/
  proper_assay_transfer_no_prop_no_tanimoto/
  proper_assay_transfer_no_tanimoto/
```

Both schemas are still prompt/completion/metadata, directly usable as input to run_llm_benchmark.py.
`proper_assay_transfer_no_prop_no_tanimoto` does not contain properties / Tanimoto;
`proper_assay_transfer_no_tanimoto` contains molecule properties, but no Tanimoto.

Current full-validation baselines:

```text
rows: 19,986
labels: similar 10,893 / different 9,093
Tanimoto>=0.50 macro-F1: 0.5335
Bucket-majority macro-F1: 0.4799
```

Current qwen3-8b full-validation runs (excluding 1-sample smoke; two old misleading properties/json runs deleted):

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

Current qwen3-8b valid20k comparison:

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

Current qwen3-4b full-validation runs:

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

Current qwen3-4b valid20k comparison:

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

Current gpt-oss-120b valid20k run:

```text
llm_runs/gpt_oss_120b_proper_no_tanimoto_valid20k_json_with_thinking_trace/
  properties, json/thinking trace, macro-F1 0.5434
  reasoning_content present in sampled run JSON and trace_messages.jsonl.
```

Current gpt-oss-120b valid20k comparison:

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

Currently, the train/validation feature cache for the trained MLP baseline has been built for `jiosephlee/proper_assay_transfer_no_prop_no_tanimoto`. This cache contains only train and validation; the test split has not been appended yet.

The local full train has been materialized:

```text
outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_train/proper_assay_transfer_no_prop_no_tanimoto/
  train.jsonl
  summary.json

rows: 12,327,157
labels: A/similar 6,957,330; B/different 5,369,827
```

Qwen3-Embedding-8B has been downloaded to:

```text
/data1/tianang/cache/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
```

The official preprocessed cache:

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

Current full cache statistics:

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

Official build command template:

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

Run instructions:

```text
1. The `clean_records` stage only performs JSONL streaming parsing and clean split writing, without using GPU.
2. Only after `endpoint_embedding_start` will Qwen3-Embedding-8B be loaded and GPU used.
3. The `rdkit_features` stage uses CPU multi-process to compute Morgan fingerprints and RDKit descriptors.
4. `TOKENIZERS_PARALLELISM=false` is to prevent tokenizer internal threads and multi-process/GPU workers from competing for CPU; it does not affect GPU parallelism.
5. The sentence-transformers `encode_multi_process` deprecation warning and multiprocessing resource_tracker semaphore warning do not affect the written cache; rely on manifest, array shapes, and row counts.
```

When adding features for the test split later, prioritize implementing incremental append mode:

```text
Input existing preprocessed/ + test JSONL;
Only parse test;
Only compute missing endpoint embeddings and missing molecule RDKit features;
Rewrite endpoints/molecules/feature arrays;
Add row_indices/test.npz;
Do not recompute existing 154,370 endpoints and 1,337,289 molecules.
```

MLP training entry point:

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

Training output directory:

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

Training implementation notes:

```text
1. Endpoint embeddings are already L2-normalized; use LayerNorm during training, no additional z-score.
2. Morgan fingerprints are uint8 0/1; convert to float in the training collate; do not input Tanimoto scalar.
3. RDKit descriptors use train molecule set to compute mean/std; NaN/inf filled with 0; z-score then clip to [-10, 10].
4. Validation full metrics reuse the field shape of run_llm_benchmark.compute_metrics; the `llm` key represents MLP prediction.
5. W&B will log overall macro-F1/accuracy/recall, as well as macro-F1 and label-specific recall under similarity_bucket, assay_type, and eval_subset.
6. The current recommended command's `--max-steps 5000` is approximately 13.3 epochs with 8 GPUs and per-GPU batch size 4096; the `Epoch N/-2` in the Lightning progress bar is a display placeholder in max_steps mode; the training stop condition is based on global_step.
7. Full validation generates metrics/report/predictions on rank0 and updates the best checkpoint; other DDP ranks wait at the barrier to avoid NCCL allreduce timeout when continuing training after validation.
```

### dynamic_v1 endpoint-disjoint MLP train-eval v1

Purpose:

```text
Convert the pChEMBL-delta non-ambiguous pairs of chembl36_activity_transfer_dynamic_v1 into HF prompt/completion/metadata compatible format, reuse the Qwen endpoint embedding + RDKit molecule feature MLP baseline, and evaluate whether the trained classifier exceeds the Tanimoto threshold.
```

Data and split:

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

Build split:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2 \
  --progress-every 500000
```

Feature cache:

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

Feature build command:

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

Training and results:

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

Purpose:

```text
A cleaner label version of activity-transfer data, used to determine whether the original 0.5/1.0 pChEMBL delta labels are too noisy.
Only redefines label/split data; MLP has not been trained yet.
```

Configuration and output:

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

Build command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits \
  --similar-delta 0.3 \
  --different-delta 1.2 \
  --out-dir outputs/chembl_tool/activity_transfer_benchmark/dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2_strict_pchembl_0p3_1p2 \
  --progress-every 500000
```

### CHEMBL4513218 / inhibition raw-value single endpoint

Purpose:

```text
Investigate whether a large-scale publication assay endpoint without pChEMBL can construct a raw-value activity-transfer benchmark.
This endpoint is CHEMBL4513218 / inhibition, standard_units is %, from DOI 10.1021/acsinfecdis.9b00482, assay description is P. berghei liver stage luciferase screen at 10uM.
```

Important data facts:

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

200k build command:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_single_endpoint_raw_transfer_splits \
  --n-pairs 200000 \
  --split-mode molecule_disjoint \
  --versions full_range,no_lt_minus_100
```

200k full_range results:

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

Exhaustive compact command:

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

Old valid10k completed full LLM run settings:

```text
input:
  hf_jiosephlee_valid10k/chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/validation.jsonl
runs:
  llm_runs/gpt_oss_120b_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools/
  llm_runs/deepseek_v4_pro_hf_assay_mol_disjoint_no_tanimoto_valid10k_tools/
comparison:
  comparisons/hf_jiosephlee_valid10k/chembl-mol12-stdsep-assay-mol-disjoint-no-props-no-tanimoto/
```

Current overall macro-F1:

```text
gpt-oss-120b:     0.5365
DeepSeek-v4-pro:  0.5068
Tanimoto >= 0.5:  0.5364
Bucket majority:  0.3606
```

LLM run example:

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

First version of the 3K LLM benchmark with local vLLM gpt-oss-120b + tool server.

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/gpt_oss_120b_dynamic_v1_llm_3k_tools/
  benchmark_explanation.html
  report_zh.md
  metrics.json
  predictions.jsonl
  figures/model_vs_baselines.svg
  runs/
```

Run settings:

```text
model: gpt-oss-120b
vLLM base URLs: http://127.0.0.1:8001-8004/v1
local vLLM api key: EMPTY
tool service: http://127.0.0.1:8765
parallelism: 16
max tool rounds: 1
```

Results:

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

Gray-zone results:

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5390
Tanimoto>=0.50 macro-F1: 0.4683
```

Interpretation:

```text
gpt-oss-120b + tools exceeds the simple threshold baseline on the 3K balanced stress test,
but absolute performance is still weak. This result cannot be directly compared one-to-one
with the best Tanimoto macro-F1 ~0.57 on the full dynamic_v1 natural distribution.
benchmark_explanation.html is a reader-oriented HTML report documenting data construction, method comparison,
prompt, tool call rate, and a visible trace example with tool calls.
```

### gpt_oss_120b_dynamic_v1_llm_3k_tools_thinking

3K LLM benchmark with local vLLM gpt-oss-120b + tool server + `--enable-thinking`.

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

Run settings:

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

Results:

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

Gray-zone results:

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5301
Tanimoto>=0.50 macro-F1: 0.4683
```

Interpretation:

```text
The thinking version saves reasoning_content and reasoning_summary, suitable for trace auditing;
but classification performance is lower than the non-thinking version: macro-F1 0.5253 vs 0.5309.
benchmark_explanation.html has been updated to the thinking run results and includes a reasoning/tool trace example
for sample_00107; that sample is a very_close analog but truly different, and the LLM judged correctly.
error_analysis_zh.md records the thinking run's error stratification, typical FP/FN, failure reasons, and future improvement suggestions.
```

### deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking

DeepSeek-v4-pro + tool server + thinking 3K LLM benchmark.

```text
outputs/chembl_tool/activity_transfer_benchmark/llm_runs/deepseek_v4_pro_dynamic_v1_llm_3k_tools_thinking/
  report_zh.md
  metrics.json
  predictions.jsonl
  trace_messages.jsonl
  runs/
```

Run settings:

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

Results:

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

Gray-zone results:

```text
Tanimoto 0.40-0.70 subset n=818
LLM macro-F1: 0.5090
Tanimoto>=0.50 macro-F1: 0.4683
```

Interpretation:

```text
DeepSeek-v4-pro is currently the highest LLM run on the overall 3K stress test:
macro-F1 0.5378, higher than gpt-oss-120b no-thinking 0.5309 and thinking 0.5253.
But the improvement is small and mainly comes from more conservative prediction of different; similar recall is notably low.
In the Tanimoto 0.40-0.70 gray zone, DeepSeek macro-F1 0.5090 is lower than both gpt-oss runs.
Therefore it is best overall but not best in the gray zone; considering tool calls, tokens, and wall time,
current cost-effectiveness is lower than local gpt-oss.
```

### starling-labs Oral_Bioavailability transfer dataset

Oral bioavailability transfer benchmark not from ChEMBL/Joseph Lee sources.

Source data:

```text
HuggingFace: starling-labs/Oral_Bioavailability
split: train
rows: 163,815
columns:
  pmid, support_text, molecule_name, oral_bioavailability_value,
  bioavailability_report_type, species_or_population, dose,
  oral_exposure_mode, qualifying_conditions, comparator, extra_details, smiles
```

Cleaning rules:

```text
Main version retains report types:
  absolute, unspecified, systemic_availability
Discard:
  relative_comparison, rows that cannot be safely parsed into numeric explicit values, RDKit invalid SMILES,
  values outside the default 0-1000% range.
value parser:
  about/~approximately: take the main numeric value
  per cent/percent: normalize to %
  mean/average/median: prefer the corresponding value
  x ± y: take x
  x to y / x-y range: take midpoint
  no % and 0<=value<=1.5: convert as fraction to percent
  AUC-only, fold/higher/lower/comparable and other relative descriptions: discard
condition_text:
  write all experimental condition fields and values; empty values write not specified.
  Fields include species_or_population, dose, oral_exposure_mode,
  qualifying_conditions, comparator, extra_details.
metadata:
  original source row fully saved for later re-cleaning.
```

Clean evidence main output:

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

Build clean evidence:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_transfer_dataset \
  --run-id absolute_unspecified_systemic_broad_condition_full_text_v1 \
  --allowed-report-types absolute,unspecified,systemic_availability
```

Pair label:

```text
similar:   |delta oral bioavailability percentage points| <= 10
different: |delta oral bioavailability percentage points| >= 30
ambiguous: excluded
Pairs are generated within the same condition_key.
```

condition_key notes:

```text
Main version uses broad_condition:
  species_or_population | oral_exposure_mode | qualifying_conditions | comparator
dose and extra_details are not in the key, but are always retained in condition_text/prompt/metadata.
```

#### Oral Bioavailability MLP: non-molecule-disjoint diagnostic

Earlier versions with directed max / unordered-pair split were done, results were high but do not represent molecule generalization.

```text
split:
  outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf_pair_splits/
    absolute_unspecified_systemic_directed_max_7_1_2_v2_same_unordered_split/
rows:
  train 3,391,750; validation 484,534; test 969,068
rule:
  A->B/B->A of the same unordered pair stay in the same split, but molecules can cross splits.
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
  This is a leakage-prone diagnostic, not a prospective generalization result.
```

#### Oral Bioavailability MLP: molecule-disjoint main result

Strict molecule-disjoint version: assign splits globally by canonical SMILES, the same SMILES does not cross splits;
only write pairs within a split. To make pair counts close to 7:1:2, use molecule split ratio approximately
0.522774/0.197629/0.279597, because within-split pair counts scale roughly with the square of molecule fraction.

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
  Strict molecule-disjoint generalization is weak, significantly lower than the non-molecule-disjoint macro-F1 of about 0.98.
  This result indicates that the high score of the former mainly comes from molecule/pair overlap; subsequent reports should use the molecule-disjoint version.
```

Build molecule-disjoint split:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_pair_splits \
  --run-id absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1 \
  --split-mode molecule_disjoint \
  --ratios 0.522774,0.197629,0.279597 \
  --target-pairs 30000000
```

Preprocessing:

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

Training:

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

The directed ordered version no longer judges transfer similar/different, but instead judges whether Molecule B/query's F% is higher than
Molecule A/reference. Each available unordered pair writes two directions: A->B and B->A; therefore, as long as the two F%
values are not exactly equal, they contribute a pair of `query_higher` and `query_lower`. `completion A=query_higher`,
`completion B=query_lower`, during training the positive class probability represents `query_higher`.

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
run_llm_benchmark.py supports this ordered HF prompt/completion format. It reads from metadata:
  completion_a_label=query_higher
  completion_b_label=query_lower
then automatically switches to the ordered prompt/schema:
  predicted_direction: query_higher or query_lower
instead of the old transfer benchmark's predicted_transferability=similar/different.

Tool calling still reuses the same OpenAI-compatible runner, allowing the LLM to call:
  mmp_structure_compare
  properties_compare
The tool service entry point remains http://127.0.0.1:8765.

Note:
  For the higher/lower ordered task, Tanimoto>=0.50 is no longer used as a meaningful baseline;
  the report mainly looks at the LLM and bucket-majority and other available baselines.
```

Build ordered molecule-disjoint split:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.activity_transfer_benchmark.build_oral_bioavailability_pair_splits \
  --run-id absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1 \
  --split-mode molecule_disjoint \
  --label-mode higher_lower \
  --ratios 0.522774,0.197629,0.279597 \
  --target-pairs 30000000
```

During preprocessing, the ordered label mapping must be explicitly passed:

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

Ordered MLP training / W&B entry point:

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

## Common Commands

Full baseline:

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

MCS full-scan:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --full-scan \
  --workers 128 \
  --timeout-s 2 \
  --chunksize 1 \
  --progress-every 10000
```

MCS stuck cleanup:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.benchmark_mcs_runtime \
  --run-id dynamic_v1_mcs_full_t2_w128_stream \
  --finalize-existing \
  --workers 128 \
  --timeout-s 2
```

MCS threshold analysis:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.analyze_mcs_results \
  --run-id dynamic_v1_mcs_t2_analysis
```

Build 3K LLM eval set:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.build_llm_eval_set \
  --run-id dynamic_v1_llm_3k
```

Build task-scoped assay transfer benchmark:

```bash
python -m tools.chembl_tool.activity_transfer_benchmark.run_task_assay_benchmark \
  --run-id task_assay_transfer_v2 \
  --workers 256 \
  --max-pairs-per-endpoint 5000 \
  --max-total-pairs-per-mode 2000000 \
  --progress-every-endpoints 250
```

Notes:

```text
run_task_assay_benchmark.py
  Only uses assay/activity evidence already filtered by the four task pipelines:
  BBB_Martins v6, Bioavailability_Ma v4, ClinTox v6, Skin_Reaction v1.
  Default output includes three label modes:
    raw_robust_z:     Within the same task + assay + endpoint + normalized units, label using robust sigma of raw standard_value.
    log_raw_robust_z: Same as above, but using log10(raw standard_value), more suitable for magnitude-type endpoints like IC50/EC50/Ki.
    pchembl_delta:    Compatible with old rules, |delta pChEMBL| <= 0.5 is similar, >= 1.0 is different.
  Unit normalization unifies nM/uM/mM/M to nM, common permeability units to cm/s,
  and common clearance units to mL/min series; units that cannot be safely converted across molecular weights
  (e.g., ug/mL) are kept as separate normalized units to avoid mixing endpoints.
  raw/log robust sigma uses IQR/1.349, with fallback to MAD*1.4826 when IQR is 0;
  endpoints with sigma still 0 are excluded from raw/log label modes.
```

Run local gpt-oss-120b LLM benchmark:

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

Run DeepSeek-v4-pro thinking LLM benchmark / resume from checkpoint:

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

## Historical 3K Benchmark Follow-up Plan

The following is the ablation checklist recorded when `dynamic_v1_llm_3k` was initially completed, and no longer represents current project priorities. Item 4, learned classifier, has been substantially completed by the endpoint-disjoint, HF proper-assay-transfer, and Oral Bioavailability MLP experiments described earlier in this file; it must not be re-registered as "not yet started" because of this old list. The remaining three items are only to be executed if re-entering that 3K stress-test research question; current priorities follow the paper execution plan.

```text
1. no-tools ablation: same 3K set, no tool calls allowed.
2. no-MCS-in-prompt ablation: keep Tanimoto and assay context, remove MCS coverage.
3. full-distribution eval: sample from dynamic_v1 natural distribution, fairer comparison with full-data Tanimoto threshold.
4. learned classifier: already superseded and completed by subsequent MLP series experiments, no longer a to-do item.
```
