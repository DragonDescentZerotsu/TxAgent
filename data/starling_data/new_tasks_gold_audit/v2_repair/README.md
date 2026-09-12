> 历史阶段记录（2026-09-12 归档说明）：正文中的计数、状态、路径和命令描述该阶段，不代表当前运行配置。当前两任务均使用 Starling-only gold_v4 和 `retrieval_final`；旧暂存 source/index 及一次性脚本已清理，保留的账本仍用于来源追溯。恢复与维护请见[当前合同](../../../../tools/chembl_tool/common/starling/NEW_TASK_SOURCE_DATA.md)。

# v2.1 repair evidence

See [current release report](../REBUILD_V2.md).

`before_receipt.json` and `*_before_votes.jsonl` pin the old release.
`*_affected_sample.jsonl` and `*_affected_votes.jsonl` document the initial rule-change screening, not final dispositions.
`retained_candidate_sample.jsonl` documents the second candidate screen.
`reviewed_records.jsonl` contains91 individual decisions applied to current ledgers.
`*_negative_recovery_search.jsonl` retain the bounded source search; only specifically reviewed claims were accepted.
`*_vote_changes.jsonl` and `validation.json` describe final membership changes, exact known withdrawals and two deduplicated Ketamine studies.
`tests.xml` contains the68-test result.
