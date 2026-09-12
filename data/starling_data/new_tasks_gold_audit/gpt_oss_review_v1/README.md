# Exhaustive DILI / Carcinogens source review

**Historical stopped plan:** the user subsequently requested relaxed rules and
fewer reviews. This two-pass launcher was stopped, with completed ledgers retained.
The active replacement is `../targeted_review_v2/README.md`; do not restart this
exhaustive launcher alongside it. `scope_change_stop.json` records the transition.

Authorized 2026-09-08: use all eight idle node001 GPUs to serve GPT-OSS-120B,
and re-review discarded positive and negative candidates before rebuilding data
and considering fresh prediction experiments. No new prediction experiment is launched.

The review covers **every base source row**: DILI 189,172 and Carcinogens 354,386,
including uncertain categories, accepted records, rule rejections and identity holds.
Two independent passes mean 1,087,116 planned record reviews. Review priority is
the full explicit-negative census followed by all remaining records; priority never
changes inclusion. Mechanism v1-v5 sources are outside this base-source census.

`manifest.json` pins source Parquet bytes, previous vote/decision artifacts and prompts.
`dili_prompt.txt` and `carcinogens_prompt.txt` contain the exact model instructions.
The complete raw record is supplied without truncation; previous rule decisions,
current gold labels, split memberships and class-balance targets are not supplied.
The model may recover a literal claim despite incorrect extracted direction, and
must also withdraw unsupported positive claims. It may preserve multiple distinct
arms, but must not invent missing study identity, specimen attribution or conditions.

Semantic output is **not automatically gold**. Explicit source quotes are validated;
unresolved studies and identities remain distinguishable from out-of-scope claims.
Two-pass disagreements require adjudication, not automatic conversion to negative or
silent rejection. Errors remain errors and are retried on resumption. Pending reviews
must never be counted as semantic exclusions. Source identity checks, independent-study
deduplication, parent-condition agreement and split feasibility are separate downstream
stages. Sparse-condition accepted gold must remain visible even if excluded from a
three-way benchmark. Preserve the existing benchmark until the staged replacement
passes these checks; do not reuse old predictions on changed labels or retrieval.

Implementation: `tools/chembl_tool/common/starling/review_source_records.py`.
`round1.jsonl` / `round2.jsonl` are append-only model verdict/attempt/trace ledgers;
source hashes, execution hash and prompt hashes bind every success. Interrupted final
writes are repaired on resume; complete corrupt lines fail rather than being skipped.
Only one launcher may hold `run.lock`. `round*.progress.json` reports actual progress.
Model-derived labels must be described as model-assisted semantic review, not human
expert annotation or full-paper validation.

Deployment helper `serve_node001.py` reuses the four-TP2 HAProxy topology, verifies
host/GPU/port availability, uses the frozen local model snapshot and existing vLLM
environment, and writes logs here. It does not stop existing processes. Backends bind
node001 loopback 8001-8004; HAProxy binds loopback 9001 with least-connection routing.
`option http-server-close` prevents idle reused backend sockets from distorting the
least-connection counts; client keep-alive remains available. The change used a soft
proxy reload with in-flight requests drained, recorded in `proxy_connection_rebalance.json`.
All 22 model files are SHA256-verified against the frozen shared snapshot and staged
at `/local/tmp/tianang-gptoss-source-review/model` to avoid shared-filesystem loading
contention. Each backend uses two A100-80GB GPUs, max model length 32768 and GPU memory
utilization 0.90, `max_num_seqs=1024` and `max_num_batched_tokens=8192`.
The initial vLLM defaults limited each backend to 256 active sequences; the initial
execution/deployment receipts, logs and scheduler-restart coverage are preserved.
The resumed runner uses the identical source/prompt/model contract and skips its
3,695 already successful reviews. No environment upgrade or model download was needed.

The full reviewer runs directly on node001 in the persistent tmux session
`gptoss-source-review-run`, independently of the node002 SSH connection. The four model
servers and proxy are in `gptoss-source-review`. The verified node002 port 19001 SSH
forward is used only for the isolated preflight.

```sh
/data1/tianang/anaconda3/condabin/conda run -n vllm python -m tools.chembl_tool.common.starling.review_source_records prepare
# On node001, after the separate preflight succeeds:
tmux new-session -d -s gptoss-source-review-run 'sh /data1/tianang/Projects/TxAgent/data/starling_data/new_tasks_gold_audit/gpt_oss_review_v1/run_review.sh'
```

Run a separate smoke root before the full census; do not mix changed model or prompt
contracts into existing ledgers. Retain measured throughput and ETA from the real
review workload, not historical short synthetic benchmark throughput.

The frozen full-run contract is `exhaustive_source_semantic_review.v3`, GPT-OSS-120B,
medium reasoning, temperature 0, and 8192 maximum completion tokens. The launcher has
2048 worker slots; `concurrency.json` sets the live global request limit, tested at 512,
1024 and now 2048,
and is checked every 30 seconds. Increase only while measuring successful review
throughput, latency, backend KV-cache pressure and preemptions; in-flight requests are
never cancelled when the limit is reduced. The HTTP connection pool matches the worker
budget. Three per-record attempts and up to three error-recovery passes are allowed.
Unresolved errors in the first pass do not prevent the second independent pass; final
incomplete coverage is explicitly recorded as `incomplete_review_errors` and exits
unsuccessfully, never as a completed census. The current runner/client checks pass
24 tests, including interrupted resume, retries and this two-pass coverage behavior.

`smoke`, `smoke_v2` and `smoke_v3` are isolated 48-record stratified preflight ledgers,
not full-census results. Earlier preflights exposed endpoint and study-attribution
mistakes. The frozen contract explicitly distinguishes target absence from narrower
null findings, requires the model to state its negative-evidence category, and separates
single original studies from pooled or generic assertions. These checks improve
self-consistency; a schema-valid model answer is not proof of scientific correctness.
`preflight_findings.json` preserves known residual semantic/study-attribution concerns
for downstream adjudication, even if the two passes agree. `throughput_observations.jsonl`
records measured backend activity, GPU utilization and ledger progress.
