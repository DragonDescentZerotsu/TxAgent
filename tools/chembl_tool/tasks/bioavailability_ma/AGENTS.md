# Bioavailability_Ma paper-path notes

本目录只保留可进入论文主方法的简洁 pipeline：通用 evidence contract、molecule-level retrieval、
single/group/final LLM reasoning 和 task ontology。历史 Fa/Fg/Fh full expert policy、deterministic
force/block/rescue、fallback calibration、postprocess 和 test-error-driven evolution 已从 `main` 删除。

旧实现的完整快照保存在：

```text
branch: archive/bioavailability-full-expert-policy-20260710
commit: 14803c2
```

不要从该 archive 向 paper path 恢复 class-changing policy。

## Task contract

```text
input:
  drug: query SMILES
  Y: 0/1 evaluation label

label:
  Y=1 -> high, oral bioavailability F >= 20%
  Y=0 -> low, oral bioavailability F < 20%
```

Exact-query evidence 默认关闭。Neighbor retrieval 是 evidence prefetch，不是 LLM function tool。

## Paper pipeline

```text
source rows
  -> minimal_evidence.v1
  -> molecule-level aggregation and fingerprint index
  -> operational or parent-disjoint identity filtering
  -> source-local groups mapped into direct/mechanism families
  -> parallel single-molecule and mechanism-family reasoning
  -> final LLM synthesis
  -> structured-output validation only
```

允许的工程防护：

- canonical SMILES 和 molecule-level aggregation；
- exact-query exclusion；
- source/provenance 保留；
- single branch 的 `molecule_properties`；
- group branch 的 `mmp_structure_compare` / `properties_compare`；
- JSON schema validation 和有 trace 的 bounded retry（当前默认最多 4 次总尝试）；
- trace、batch resume 和 metrics。

禁止添加：

- `force_high` / `force_low`；
- final prediction override；
- 针对某个 test molecule 或 failure pattern 的 blocker/rescue；
- Starling-specific label policy；
- valid/test-selected postprocess；
- 将内部 `evidence_direction` / `evidence_strength` 直接发送给 LLM。

## Minimal evidence contract

所有 ChEMBL、Starling 和未来 source row 在 LLM prompt 中统一使用：

```text
tools/chembl_tool/common/evidence_contract.py
contract_version: minimal_evidence.v1
```

Contract 包含：

```text
source
molecule
group
endpoint + measurement
evidence/context text
annotations: evidence_role, scope, transferability, uncertainty
quality
provenance
representative examples
```

其中 `transferability=not_assessed` 是 retrieval-time 默认值；query-specific transferability 必须由
group LLM 根据结构比较和 evidence context 判断。Contract 不包含 threshold vote 或 label recommendation。

## Data sources

Paper-facing source/group mapping 的唯一配置入口：

```text
tools/chembl_tool/tasks/bioavailability_ma/experiment_config.py
```

它声明 ChEMBL/Starling 的 direct groups 和 5 个 mechanism families；不得在 runner 或 source adapter 中
复制该 mapping。

ChEMBL evidence library：

```text
outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/
  bioavailability_molecule_evidence.jsonl
  bioavailability_neighbor_index.pkl
```

Starling task data：

```text
data/starling_data/bioavailability_ma/
  Oral_AUC-Cmax_Exposure/extractions.parquet
  Fa/extractions.parquet
  Fg/extractions.parquet
  Fh/extractions.parquet
```

Starling factor builder：

```text
tools/chembl_tool/tasks/bioavailability_ma/build_starling_factor_evidence_library.py
```

它使用 shared profile ingestion：

```text
tools/chembl_tool/common/starling/evidence_library.py
```

Task wrapper 只声明 column mapping 和 group/role：

| Source | Group | Evidence role |
|---|---|---|
| HF Oral Bioavailability | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| Oral_AUC-Cmax `bioavailability` rows | `Observed.direct_oral_bioavailability` | `direct_outcome` |
| Other Oral_AUC-Cmax rows | `Observed.oral_auc_cmax_exposure` | `surrogate_proxy` |
| Fa parquet | `Fa.absorption_solubility_permeability` | `mechanistic_factor` |
| Fg parquet | `Fg.gut_wall_efflux_intestinal_metabolism` | `mechanistic_factor` |
| Fh parquet | `Fh.hepatic_clearance_metabolic_stability` | `mechanistic_factor` |

Fa/Fg/Fh 是 task ontology，不是 deterministic classifier。Final prediction 仍由 LLM 根据 group outputs
综合得出。

## Source ingestion rules

- Parquet column 差异通过 `StarlingSourceProfile` 配置解决，不为每个 parquet 写独立 parser。
- 缺少 SMILES 或 RDKit 无法解析的 row 不进入结构 retrieval，并计入 source stats。
- SMILES 在聚合前 canonicalize；同一 canonical molecule 跨 profile 使用相同 source molecule id。
- 多条 source row 聚合成一个 molecule/group evidence row；代表性 examples 保留 endpoint、value、unit、
  context 和 support text 的绑定。
- PMID/DOI 可保留在 raw internal row 中，但不能进入 LLM-visible minimal evidence。
- Direct numeric outcome 只有在 endpoint 单一、unit 一致时才生成 aggregate measurement；proxy/mechanism
  evidence 保留 examples，不把异构数值混成一个 synthetic value。

## Commands

构建 Starling index：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.build_starling_factor_evidence_library \
  --workers 32
```

用通用 pipeline 跑 Starling source：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
  python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --index outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_factor/starling_factor_neighbor_index.pkl \
  --api-key-env GLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --batch-id bioavailability_ma_paper_starling_<date>
```

API key 只能通过环境变量或未提交的本地 env file 提供，不能写入代码、manifest、命令示例或 git。

## Tests

```bash
/data1/tianang/anaconda3/envs/vllm/bin/python -m pytest \
  tests/chembl_tool/common \
  tests/chembl_tool/tasks/bioavailability_ma -q
```

至少覆盖：

- legacy ChEMBL row -> `minimal_evidence.v1`；
- profile-driven parquet column mapping；
- missing/invalid SMILES stats；
- molecule-level aggregation和 exact-query exclusion；
- direct/proxy role split；
- JSON required-field/value/tool validation retry；
- API key 不进入 tracked files。

## Evaluation boundary

历史 Bioavailability test set 已在旧 expert-policy 迭代中被反复检查，不能作为论文的 untouched final
test。Paper result 应使用新 holdout、重新冻结的 split 或外部 evaluation。Archive branch 的历史 metrics
不能作为当前 simplified paper pipeline 的结果。

## Cached assay-transfer reranking

The optional `assay_transfer` retrieval reranker is restricted to the five Starling families declared in
`experiment_config.STARLING`. It is disabled by default. Candidate selection is frozen as Tanimoto top 100,
then the shared identity exclusion policy, then every survivor, then cached model reranking, then the existing
final `top_k_per_group`. It never scans or backfills below the raw top 100.

The current imported cache is the retrieval-agnostic `cache_v2` store under
`evidence_library/assay_transfer_rerank/`. It contains an append-only SQLite score store and its provenance
artifacts. Score identity includes only the exact prompt hash, immutable model revision, scoring contract,
and template hash. Catalog, split, source composition, and identity policy are audit/join metadata and never
change score identity.

A separate soft-checkpoint cache is preserved at
`evidence_library/assay_transfer_rerank/cache_soft_5fc06af6/`. It contains all 45,057 unique prompts in the
frozen prepared-HF validation catalog and was built on GPUs 0--5 with
`jiosephlee/assay-transfer-tool-soft@5fc06af66b490575e8eb32231d96aeada26794c6`. Its immutable
`VERSION.json` records the exact catalog/manifest hashes, scoring contract, model runtime, devices, and row
count. The original `cache_v2` remains unchanged. Soft-checkpoint validation uses these scores only to rank
neighbors; it does not pass the scores or their semantics to reasoning prompts.

The soft checkpoint's custom tokenizer requires `transformers==4.57.6` and
`huggingface-hub==0.36.0`. The local ignored `.runtime/transformers_4_57_6/` overlay supplies those versions
for cache inference without changing the project environment. Precompute commands for this checkpoint must
prepend that directory to `PYTHONPATH`; reasoning runs only read SQLite and do not need the overlay.

Both direct-bioavailability source schemas are normalized: HF Oral Bioavailability records use
`oral_bioavailability_value_percent`, while direct rows from Oral_AUC-Cmax use
`reported_value`/`reported_units`. This normalization is only for assay-transfer prompts; it does not change
the Starling evidence indexes or ordinary Morgan retrieval.

The versioned `v6_5_query_context_copy` prompt profile vendors the exact
`assay_transfer_v6_5_intern/default.jinja` contract (SHA-256
`e30f995988db7214cae4b170a2c36f3a5fd61b6bee6188b39ce54377fa67f5bd`). It renders the retrieval
record with its canonical SMILES, scalar measurement, unit, canonical endpoint, assay concept, and complete
assay context. The query uses canonical SMILES and copies the retrieval endpoint/concept/context while never
receiving a scalar or known-value field. `legacy_v3` remains the default so existing catalogs, cache identities,
and runs are unchanged. Cache and reasoning CLIs select the new contract explicitly with
`--assay-transfer-template-profile v6_5_query_context_copy`.

For the frozen prepared-HF validation manifest at similarity floor 0.30, there are 9,863 record-level prompt
references. Exact v6.5 rendering collapses 39 duplicate references into 9,824 unique retrieval-agnostic prompt
hashes; the older v6 template produced 9,844 unique hashes. The v6.5 SQLite cache therefore stores 9,824 scores
while covering every record reference. Its `VERSION.json` records both counts and the reason for the difference.

## Hosted GLM validation concurrency

These instructions apply directly to `run_reasoning_batch.py`, `run_reasoning_pipeline.py`, and all
Bioavailability hosted-GLM launch configurations.

### Concurrency target

- Treat 256 as the maximum number of outstanding GLM completion requests, not as a target for molecule
  subprocesses or a multi-prompt request payload.
- For a fresh `full_mechanism` Bioavailability batch with the current five mechanism families, default to
  `--parallelism 48 --group-workers 5`. This permits approximately 240 simultaneous group requests and leaves
  headroom below the 256-request API limit.
- Calculate effective fan-out as
  `min(parallelism, unfinished_molecules) * min(group_workers, active_group_count)`. Do not report
  `parallelism * group_workers` when fewer groups exist. For example, `32 * 8` has an effective group-stage
  ceiling of 160 when only five mechanism families are active.
- For a different experiment view, choose the largest safe molecule parallelism that targets 224--240
  effective requests without exceeding 256. Set `group_workers` to the number of concurrently active group
  branches unless there is a measured reason to use less.
- Do not default new hosted-GLM runs to serial execution. If the observed endpoint or node cannot sustain the
  target, reduce concurrency in measured 20--25% steps and record the errors and final setting.

### Launch and resume requirements

- Use the local `txagent-glm` environment and inject the ignored local API key through `LITELLM_API_KEY`;
  never print or persist the key.
- Use `--skip-existing` whenever resuming so completed molecule outputs are preserved. A partial molecule
  without `final_reasoning_output.json` may be rerun.
- Preserve separate retrieval, group, final, and batch output directories for each k value or condition.
  Never reuse final predictions across k values.
- Archive a genuinely failed attempt before a clean retry. An intentional concurrency restart is a resume,
  not a failed-attempt archive.
- Keep score-hidden runs score-hidden: assay-transfer scores may rank neighbors but must not appear in any
  GLM-visible prompt.

### Preflight and monitoring

Before issuing GLM requests, verify:

- the tool service health endpoint is healthy;
- SQLite `quick_check` is `ok` and the required score count and immutable provenance match;
- the frozen catalog, candidate manifest, model revision, template profile, and template hash match;
- the node has enough available memory for the requested molecule-process count, because each molecule
  process loads retrieval state;
- the configured effective GLM fan-out is no greater than 256.

During the run, monitor completed outputs, active molecule processes, HTTP 429/5xx responses, timeouts,
validation retries, and memory. Concurrency should remain near the target when healthy. Endpoint throttling,
repeated transport failures, or memory pressure are valid reasons to reduce it; document the reason rather
than silently reverting to serial execution.

After completion, require 64/64 outputs and zero failures, then run the task's retrieval, identity,
similarity-floor, group-size, below-k attribution, prompt-visibility, and provenance audits.

Prompt-audit tooling lives in the `prompt_audit/` subpackage: `view_run.py` (trace/prompt viewer),
`dump_prompt_examples.py` (example generator), and the committed `prompt_audit/prompt_examples/`. The examples
are one per stage (single, group, final), with the group stage in all three formats (`group.legacy`,
`group.morganfingerprint`, `group.assay_transfer_tool`). They are **compiled prompts only** (no model output),
generated from one frozen real datapoint (`tests/.../fixtures/prompt_examples/fixture.json`) by
`python -m tools.chembl_tool.tasks.bioavailability_ma.prompt_audit.dump_prompt_examples --write`, and pinned by
the golden test `tests/.../test_prompt_examples.py` so they cannot drift from the prompt code — regenerate after
any prompt change. Never add hidden labels, API credentials, or non-visible provenance to an example, and never
add assay-transfer scores to a **score-hidden** example; the `assay_transfer_tool` format legitimately shows its
by-design visible transfer score.

The group text-format instruction blocks are editable `.txt` files under `prompt_instructions/`
(`morganfingerprint.txt`, `assay_transfer_tool.txt`; one instruction per line, `#` comments ignored), loaded by
`group_prompt_render.load_instructions`. Which metadata fields appear is likewise editable in
`group_prompt_field_policy.py`.

Cached assay-transfer reranking code lives in the `reranking/` subpackage
(`tasks/bioavailability_ma/reranking/`: rerank, prompt policy, catalog/precompute builders, and the scoring
`assay_transfer_templates/`).

Assay-transfer scoreability filtering and neighbor selection are independent of prompt visibility. A score-hidden
ablation may use the cached scores to select the same ordered neighbors while omitting all transfer scores and
score-policy semantics from LLM-visible group and final prompts.

Attached numeric source-record examples become candidate-scoped prompt records; qualitative-only candidates
remain in the retained pool but do not receive fabricated prompts. Scoring mirrors training evaluation:
append `(A)` and `(B)` with no leading space, find their first divergent token, and softmax the two next-token
logits.

### Assay-transfer inference memory and batch size

The pinned `jiosephlee/assay-transfer-tool@9515603b...` checkpoint is a Qwen3 causal LM with 36 layers,
hidden size 4096, and about 8.2B BF16 parameters. Its four weight shards occupy 16.40 GB decimal
(15.28 GiB), which is the main fixed per-worker GPU cost because precompute loads one independent model
replica per GPU.

Measured cache inference with BF16 and batch size 32 peaked at 29.76--31.00 GiB per full B200 worker. The
293,021-prompt four-worker MIG90 run peaked at 22.11--23.54 GiB per worker and completed inference in 2,041
seconds. GPU memory is not determined by batch size alone:
approximately 15.3 GiB is fixed weights, while the remaining allocator/runtime, activations, padded tokens,
KV state, and output logits grow with both batch size and the longest prompt in that batch. Consequently,
do not extrapolate VRAM as purely linear in batch size.

For future 90 GB MIG or 180 GB full-B200 runs, use batch size 64 as the next default and retain the 64 GiB
free-memory preflight. First verify batch 64 on a bounded query subset and record the reported per-device peak.
Do not raise beyond 64 without a new measurement because prompt-length padding can change the peak. If a worker
OOMs, the append-only SQLite writer preserves already committed batches; rerun with batch size 32 to resume only
the missing scores.

### Runtime profiles: node002 and VAST/SLURM

Machine paths are centralized in `reranking/runtime_profile.sh`. Select one profile instead of editing Python
or launch scripts:

```bash
# node002: /data1 checkout + txagent-glm; direct execution
TXAGENT_RUNTIME_PROFILE=node002 \
  bash tools/chembl_tool/tasks/bioavailability_ma/reranking/build_in_distribution_library.sh
TXAGENT_RUNTIME_PROFILE=node002 \
  bash tools/chembl_tool/tasks/bioavailability_ma/reranking/precompute_in_distribution_validation.sh

# VAST: shared checkout/cache; scheduler wrappers select vast_slurm automatically
sbatch tools/chembl_tool/tasks/bioavailability_ma/reranking/slurm/build_in_distribution_library.sbatch
sbatch tools/chembl_tool/tasks/bioavailability_ma/reranking/slurm/precompute_in_distribution_validation_mig90.sbatch
```

The profiles define only non-secret runtime locations: project root, Python, Starling checkout, Hugging Face
cache/offline behavior, devices, workers, and batch size. Override any `TXAGENT_*` value in the environment for
a one-off machine layout. API credentials remain outside profiles. The evidence/index/catalog/cache paths stay
repository-relative, so copying or synchronizing the corresponding `outputs/.../starling_in_distribution/`
tree preserves the condition manifest and cache provenance across machines; otherwise rebuild the artifacts
from the same eligible-record checksum.

Build the candidate-scoped catalog and manifest, then populate only missing scores with independent BF16
model replicas. The validated v2 model provenance is
`jiosephlee/assay-transfer-tool@9515603b1a5c4586e41c221dcdbc5e7487c0c3f5` and scoring contract
`assay_transfer_chat_first_divergent_token_logits.v1`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
  -m tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank \
  --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_five_source_prepared_hf/starling_factor_neighbor_index.pkl \
  --experiment-mode full_mechanism \
  --neighbor-identity-policy parent_disjoint \
  --top-k-per-group 10 \
  --min-similarity 0.0 \
  --rerank-raw-pool-size 100 \
  --rerank-candidate-size 100 \
  --rerank-catalog <prepared_hf_validation_r100_c100_min0/catalog.jsonl> \
  --candidate-manifest <manifest.jsonl> \
  --rerank-cache <cache_v2/scores.sqlite3> \
  --rerank-devices 0,1,2,3,4,5,6,7 \
  --rerank-batch-size 64
```

Reasoning batches only read the cache and run a complete coverage preflight before starting subprocesses:

```bash
python -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --retrieval-source starling \
  --experiment-mode full_mechanism \
  --index outputs/paper/molecular_evidence_agent/evidence/bioavailability_starling_five_source_prepared_hf/starling_factor_neighbor_index.pkl \
  --retrieval-reranker assay_transfer \
  --rerank-raw-pool-size 100 \
  --rerank-candidate-size 100 \
  --rerank-catalog <prepared_hf_validation_r100_c100_min0/catalog.jsonl> \
  --rerank-candidate-manifest <prepared_hf_validation_r100_c100_min0/manifest.jsonl> \
  --rerank-cache <cache_v2/scores.sqlite3> \
  --rerank-cache-mode read_only \
  --enable-assay-transfer-scores \
  --top-k-per-group 7
```

Candidate-scoped catalogs require the exact condition manifest at runtime. The requested k is preserved;
there is no legacy forced-k5 behavior. With `--enable-assay-transfer-scores`, each selected score is rounded
to two decimals and exposed only to its group reasoning prompt. Full-precision scores, winning record IDs,
selection ranks, and structural ranks remain audit-only retrieval metadata and do not enter group or final
reasoning prompts.

Use `--assay-transfer-min-score <probability>` to apply an optional inclusive cached-score floor before the
final `--top-k-per-group` truncation. The default is unset so historical retrieval remains unchanged. The
threshold and rejected-record counts are recorded in retrieval, preflight, and batch audit metadata. For the
frozen in-distribution validation condition, `k=3` with a `0.5` floor retains all 960 selected records across
all 320 query-family groups; the minimum retained top-three score is exactly `0.5`.

The portable GLM validation launcher is `reranking/run_in_distribution_glm_validation.sh`; its VAST scheduler
wrapper is `reranking/slurm/run_in_distribution_glm_validation.sbatch`. Runtime profiles expose a separate
`TXAGENT_REASONING_PYTHON` because the VAST cache-inference and resident-tool environments have different
dependency sets. The launcher imports the ignored `keys.py` only when `LITELLM_API_KEY` is absent and never
prints or persists the credential.
