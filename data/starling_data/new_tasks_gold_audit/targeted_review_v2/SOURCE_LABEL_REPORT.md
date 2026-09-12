> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# Final source-record classification

All 148,079 queued records now have a validated decision: 147,708 GPT-OSS-120B successes plus 371 records individually read by the Codex assistant. This is not human expert review or external-paper verification. The residual review changed 111 directions; 260 retained their direction, with literal quotes and scope corrections recorded.

Counts cover every original base record once. Reviewed results override rule labels for the 1,200 QA samples as well as the ambiguous/conflicting queue; the QA samples are not counted twice. Unreviewed explicit source labels and supported prior directions remain rule/prior decisions, not newly reviewed records.

| Direction/status | DILI | Carcinogens | Total |
|---|---:|---:|---:|
| positive | 156,476 | 296,079 | 452,555 |
| negative | 20,573 | 35,031 | 55,604 |
| mixed | 3,396 | 7,340 | 10,736 |
| uncertain | 3,781 | 7,406 | 11,187 |
| out_of_scope | 346 | 1,625 | 1,971 |
| specimen_or_material_hold | 4,600 | 6,905 | 11,505 |
| Total | 189,172 | 354,386 | 543,558 |

Mixed means different directions by endpoint, exposure, comparison or study; it is not automatically an extraction failure. Uncertain means direction or attribution cannot be settled from the supplied record. Out-of-scope records include pure prediction/mechanistic or otherwise non-outcome material. The 11,505 specimen/material holds predate this residual review; their original labels are retained in the output but they are not assigned a final molecule-specific binary candidate.

Record-scoped negative findings include causal exclusions, null comparisons and specified-severity/site negatives. Defined combination-exposure findings can be retained with the combination explicitly recorded. An unresolved differential among alternative culprits is not treated as settled individual attribution. Source counts are not molecule counts or independent original-study votes. Gold, benchmark splits and retrieval indices were not rebuilt or published.

Artifacts: `codex_residual_review.jsonl` holds each of the 371 decisions, original proposal, source hash and reason; `source_label_summary.json` provides counts and input/output hashes; `{task}_source_labels.parquet` preserves one row per original source with selection and final decision provenance. `finalization_receipt.json` is the combined completion receipt. Historical `completion.json` remains the GPT-OSS-only receipt with its original 371 failures; it does not describe the combined finalized result.

Reproduce with the vllm Python environment from the repository root:

```sh
PYTHONPATH=. /data1/tianang/anaconda3/envs/vllm/bin/python data/starling_data/new_tasks_gold_audit/targeted_review_v2/finalize_source_labels.py
```

The report retains the balanced 300-per-task-per-direction QA transitions in the JSON summary. These are diagnostics against source labels, not an independent estimate of model accuracy.
