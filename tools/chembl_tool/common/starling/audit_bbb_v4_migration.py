"""Audit BBB v4 parent labels and splits against the frozen v3 lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.audit_bbb_v3_migration import (
    _all_vote_rows_by_identity,
    _gold_by_identity,
    _source_record_ids,
    _split_by_identity,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_metric_direction_review import (
    MANUAL_SOURCE_EXCLUSIONS,
    REVIEWED_PROPOSED_SOURCE_INDICES,
)


V3_GOLD = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v3/BBB_Martins"
)
V4_GOLD = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v4/BBB_Martins"
)


def audit(output_path: Path) -> dict[str, Any]:
    old = _gold_by_identity(V3_GOLD)
    new = _gold_by_identity(V4_GOLD)
    old_vote_rows = _all_vote_rows_by_identity(V3_GOLD)
    new_vote_rows = _all_vote_rows_by_identity(V4_GOLD)
    old_split = _split_by_identity(V3_GOLD)
    new_split = _split_by_identity(V4_GOLD)
    shared = set(old) & set(new)
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    label_changes = sorted(
        key for key in shared if int(old[key]["Y"]) != int(new[key]["Y"])
    )
    split_changes = sorted(key for key in shared if old_split[key] != new_split[key])

    added_vote_ids = sorted(
        _source_record_ids(new_vote_rows) - _source_record_ids(old_vote_rows)
    )
    expected_vote_ids = {
        f"row:{index}"
        for index in REVIEWED_PROPOSED_SOURCE_INDICES - set(MANUAL_SOURCE_EXCLUSIONS)
    }
    if set(added_vote_ids) != expected_vote_ids:
        raise RuntimeError("v4 accepted-vote delta does not match the frozen review")
    added_vote_id_set = set(added_vote_ids)
    affected_keys = sorted(
        key
        for key, row in new_vote_rows.items()
        if added_vote_id_set & set(map(str, row.get("source_record_ids") or ()))
    )
    affected_rows = [
        {
            "molecule_identity_key": key,
            "drug": new_vote_rows[key].get("drug"),
            "added_source_record_ids": sorted(
                added_vote_id_set
                & set(map(str, new_vote_rows[key].get("source_record_ids") or ()))
            ),
            "old_label_counts": (old_vote_rows.get(key) or {}).get("label_counts"),
            "new_label_counts": new_vote_rows[key].get("label_counts"),
            "old_label": (old.get(key) or {}).get("Y"),
            "new_label": (new.get(key) or {}).get("Y"),
            "old_split": old_split.get(key),
            "new_split": new_split.get(key),
            "old_parent_status": "binary" if key in old else "not_binary",
            "new_parent_status": "binary" if key in new else "not_binary",
        }
        for key in affected_keys
    ]
    changed_rows = [
        {
            "molecule_identity_key": key,
            "drug": old[key].get("drug"),
            "change": "removed_after_revote",
            "old_label": old[key].get("Y"),
            "old_split": old_split.get(key),
            "new_vote_counts": (new_vote_rows.get(key) or {}).get("label_counts"),
        }
        for key in removed
    ] + [
        {
            "molecule_identity_key": key,
            "drug": new[key].get("drug"),
            "change": "added_after_revote",
            "new_label": new[key].get("Y"),
            "new_split": new_split.get(key),
            "new_vote_counts": new[key].get("label_counts"),
        }
        for key in added
    ] + [
        {
            "molecule_identity_key": key,
            "drug": new[key].get("drug"),
            "change": "label_changed",
            "old_label": old[key].get("Y"),
            "new_label": new[key].get("Y"),
            "old_split": old_split.get(key),
            "new_split": new_split.get(key),
            "old_vote_counts": old[key].get("label_counts"),
            "new_vote_counts": new[key].get("label_counts"),
        }
        for key in label_changes
    ]

    changed_path = output_path.with_name("changed_gold_parents.jsonl")
    affected_path = output_path.with_name("reviewed_vote_affected_parents.jsonl")
    write_jsonl_atomic(changed_path, changed_rows)
    write_jsonl_atomic(affected_path, affected_rows)
    summary = {
        "old_gold_lineage": "experimental_meaningful_cns_access_v3",
        "new_gold_lineage": "experimental_meaningful_cns_access_v4",
        "old_binary_parents": len(old),
        "new_binary_parents": len(new),
        "shared_binary_parents": len(shared),
        "removed_binary_parents": len(removed),
        "added_binary_parents": len(added),
        "shared_parent_label_changes": len(label_changes),
        "shared_parent_split_changes": len(split_changes),
        "reviewed_source_votes_added": len(added_vote_ids),
        "reviewed_vote_affected_parent_groups": len(affected_rows),
        "split_overlap": {
            split: sum(
                old_split.get(key) == split and new_split.get(key) == split
                for key in shared
            )
            for split in ("train", "valid", "test")
        },
        "changed_gold_parents": str(changed_path),
        "reviewed_vote_affected_parents": str(affected_path),
    }
    write_json_atomic(output_path, summary)
    report_path = output_path.with_name("migration_from_v3_zh.md")
    with atomic_output_path(report_path) as temporary:
        temporary.write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BBB experimental metric-direction review：v3 → v4",
            "",
            "## 审查合同",
            "",
            "- 只审查 v3 中仅因缺少 `bbb_permeability_label` 而被拒绝的 7,226 条记录。",
            "- 不对 heterogeneous numeric endpoints 使用统一阈值。",
            "- 只有原记录自身给出无歧义 qualitative direction 才进入人工复核。",
            "- 81 条规则候选全部做 source-index 级复核；16 条拒绝，65 条获准投票。",
            "- 无法可靠确定方向的 7,145 条继续作为 non-voting retrieval evidence。",
            "",
            "## Parent 重投票结果",
            "",
            f"- binary parents：{summary['old_binary_parents']:,} → "
            f"{summary['new_binary_parents']:,}",
            f"- 新增/移除 parents：{summary['added_binary_parents']:,}/"
            f"{summary['removed_binary_parents']:,}",
            f"- shared-parent label flips：{summary['shared_parent_label_changes']:,}",
            f"- shared-parent split changes：{summary['shared_parent_split_changes']:,}",
            f"- 新增 voting records：{summary['reviewed_source_votes_added']:,}",
            f"- 受影响 parent vote groups："
            f"{summary['reviewed_vote_affected_parent_groups']:,}",
            "",
            "## Split 合同",
            "",
            "所有 surviving v3 parents 保留原 split。新增 parent 若 scaffold 已存在则继承该 "
            "scaffold 的 split；全新 scaffold 只进入 train。最终 identity/scaffold overlap 均为 0。",
            "",
            "详细 parent 变化见 `changed_gold_parents.jsonl`；所有新增 vote 对 parent 的影响见 "
            "`reviewed_vote_affected_parents.jsonl`。",
            "",
        ]
    )


def main() -> None:
    output = V4_GOLD / "migration_from_v3.json"
    print(json.dumps(audit(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
