> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# DILI retrieved-source review v1 (2026-09-10)

This is a retrieval-only release authorized after individual source/trace review.
112 unique records were read: 44 move, 10 exclude, 46 keep, 12 uncertain/retained.
The 13 earlier move suggestions were reread and included in the 44 accepted moves.
The source review made no new DeepSeek requests, source-direction review, gold changes, or reasoning-setting changes.
The 46,871 earlier batch exclusion recommendations are **not** applied as a batch.

`individual_reviews.jsonl` contains all decisions and exact source quotes.
`record_review.json` contains only the 54 actionable decisions, bound to source UID,
raw payload SHA-256, original level, and the original source-file SHA-256.
It contains no benchmark label, prediction, query, or desired benchmark effect.
Candidate selection used inspected error traces; this is not an independent blinded
evaluation or an external-paper verification. Decisions concern the record's own
subject, endpoint and level, independent of whether its direction helps a query.

The implementation preserves every raw row, scientific field and voter bit.
Exclusion means `retrieval_eligible=false` and level 0, with the raw record retained.
No L1 record changes. All remaining L2 records stay eligible before query-time
disjoint filtering. Source counts and hashes are in `source/record_review_receipt.json`.

Recreate the source into a **new** output directory:

```sh
python -m tools.chembl_tool.common.starling.stage_new_task_retrieval \
  --task dili --source data/starling_data/dili/gold_v4/retrieval/source \
  --output /path/to/new/dili-reviewed-source \
  --record-review data/starling_data/dili/retrieval_review_v1/record_review.json
```

The current snapshot is registered in
`artifacts/chembl_tool/starling/current_records/manifest.json`; current catalog/index
hashes are in `tools/chembl_tool/paper_experiments/current_starling_retrieval.json`.
Use the existing `rebuild_current_starling_retrieval.py` DILI-only restore/build/verify
entrypoints. Never replay the old gold builder to reconstruct these placement edits.
`publication.json` records final validation. `tables/manifest.json` records the
refreshed source-to-index card export. Older model outputs are retained as pre-review
results and require replay or a selected-surface equivalence audit before reuse.
Matched baselines remain valid because their train/evaluation data are unchanged.

The subsequent no_prior_grounded_sim0 scaffold valid/test progressive/full-flat replay
is complete: all 804 rows were retrieved again, 215 rows were affected, 1,748 level
outputs were newly inferred and 9,508 unchanged-input outputs were verified and reused.
All four cells passed completion validation with zero remaining failures. This hybrid
result is not an independent replicate. The standard-prior suite and random split
remain unevaluated on this source. Per-level metrics and paired error changes are in
`outputs/paper/starling_conditioned_dili_retrieval_review_v1_no_prior_grounded_sim0/REPORT.md`;
the corresponding registry suite is `dili_retrieval_review_v1_no_prior_grounded_sim0`.

Review and trace evidence:
`outputs/paper/analysis/dili_retrieved_source_audit_20260910/implementation/REPORT.md`.
The storage receipt records a hash-verified relocation of historical scaffold index
files to node002 local storage to accommodate the nearly full shared filesystem.
Current source, packaged snapshot and new indices remain on shared storage.
