> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

Completed source-direction review: all queued reviews finalized, including 371 Codex residual decisions. See [SOURCE_LABEL_REPORT.md](SOURCE_LABEL_REPORT.md) and `finalization_receipt.json`. This stage only produced direction ledgers; subsequent task-specific builders published the current Starling-only gold_v4.

# Targeted source-direction review

This replaces the exhaustive two-pass plan after the user's 2026-09-08 request
to relax overly strict filters and reduce review volume. The old reviewer is
stopped; its complete ledger is preserved and SHA256-pinned in this manifest.
Execution used node001 four-TP2 GPT-OSS-120B servers and HAProxy; their availability is not a current restoration requirement.

This stage retains **source-label candidates**, not newly certified gold votes.
Explicit source positive/negative categories can be retained without requiring
universal hazard certainty, an external condition, a primary study label, or a
particular negative-result regex. Single-patient causal exclusions, null
comparisons, severity-specific and site-specific negatives keep their actual
scope. A context flag alone does not force rejection or model review.
These rules intentionally change source-candidate retention; the existing frozen
gold policy, benchmark rows, retrieval indices and experiment results are untouched.

| Action | Raw records |
|---|---:|
| Retain explicit source label | 381,573 |
| Reuse prior supported source direction for nonbinary extraction | 3,601 |
| Review nonbinary source conclusion | 145,626 |
| Review source/prior-model direction conflict | 1,253 |
| Hold non-single specimen/material for identity handling | 11,505 |
| Total traced raw records | 543,558 |

Add 300 deterministic diagnostic samples for each task and each source direction:
1,200 QA rows already included in the first table's retained-label population.
The review queue therefore has **148,079 unique records**: DILI 64,572 and
Carcinogens 83,507. One planned pass reduces model calls by 86.4% relative to the
former 1,087,116-call plan. Exact semantic-payload deduplication keeps PMID, molecule,
structure, all conditions, labels and uncertainty; it never combines distinct
studies, exposure arms or opposite directions. In this selected queue it removed
no additional records. Every raw source row remains in a selection Parquet ledger.

The original negative census is not silently dropped: 15,159 DILI and 31,893
Carcinogens negative source records are retained directly; remaining negatives
are routed to direction conflict or specimen handling as recorded per row.
These counts precede identity validation, independent-study deduplication,
conflict aggregation and benchmark selection and are **not final gold counts**.
Secondary assertions retain their meaning but never create an invented original
study. A citing PMID cannot automatically create another independent vote.

The focused model returns direction, focal-entity attribution, scope, one literal
quote and a short reason. It does not reconstruct long condition arrays or invent
study identities. A source that remains uncertain or mixed may remain so; a binary
label is not forced. Out-of-scope verdicts preserve the literal source meaning.
No old stricter `out_of_scope`/`needs_context` verdict is reused as a final rejection.
Only a prior supported focal claim may supply a reused direction, with exact prior
ledger line and payload hashes recorded. This is reuse of the source direction,
not endorsement of the old model's study attribution or benchmark eligibility.

Execution is GPT-OSS-120B, medium reasoning, temperature 0, max completion 2048,
one shared 2048-worker queue across both tasks, per-request attempts 3, and up to
two recovery sweeps after the initial pass. No second complete pass is scheduled.
Errors remain errors. A new gold build requires explicit coverage, identity,
study, conflict, condition and heldout-index checks; this runner publishes no gold.

Historical launch command (the per-round script has been removed):

```sh
sh data/starling_data/new_tasks_gold_audit/targeted_review_v2/run_review.sh
```

`manifest.json` pins original sources, selection ledgers, queues, prior reviews
and the prompt. `progress.json` recorded progress during execution; `reviews.jsonl`
contains complete answers, reasoning and output-validation errors. The successful
64-record, two-task preflight is in `smoke.jsonl`; the three discussed negative
scope examples are separately retained in `scope_examples_smoke.jsonl`.
Nineteen reviewer/client tests passed, including relaxed negative retention, exact
deduplication, both-task coverage/resume, quotation/attribution validation and
uneven client-pool allocation within the global concurrency budget.

The reviewer uses eight HTTP client pools of 256 connections, with fixed workers
per pool and 2,048 workers total (`--client-pools 8`). This bounds HTTPcore's
connection-scanning overhead; it does not multiply endpoint concurrency. Live
diagnostics found about 12 seconds with all backends empty and all GPUs idle while
the previous single-pool client saturated one CPU core. The runtime change reused
54,535 successful reviews and preserved the execution/protocol hash. Prior code
and execution are in `runtime_before_pool_sharding/`; the change receipt and
before/after samples are `pool_sharding_receipt.json` and
`idle_diagnostic_{before,after}.jsonl`. The completed review is not restarted during restoration.

Earlier GPT-OSS task reviews were selected condition-extension queues rather than
raw-source censuses: BBB 2,466, Bioavailability 2,452, Skin 14,059 and ClinTox 191.
Their queue manifests are under each task's `context_conditioned*_review*` source
directory. They started after endpoint/proposed-condition/identity selection;
their small size does not demonstrate that every upstream exclusion was reviewed.
