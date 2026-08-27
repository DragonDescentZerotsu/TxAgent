"""Rebuild the selected BBB conditioned benchmark on corrected gold v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.condition_review import merge_terminal_verdicts
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.starling.reviewed_conditioned_benchmark import (
    ConditionedBenchmarkConfig,
    build_reviewed_conditioned_benchmark,
)
from tools.chembl_tool.tasks.bbb_martins.context_conditioned_benchmark import (
    FROZEN_CANDIDATE_SOURCE,
    REVIEW_ROOT,
    SELECTED_EXTERNAL_CONDITION_GROUPS,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark_v3 import (
    label_record,
)


LINEAGE = "bbb_meaningful_cns_access_context_conditioned_selected_v2"
PROTOCOL_VERSION = "bbb_context_conditioned_selected.v2"
FROZEN_LINEAGE = "experimental_meaningful_cns_access_v3"
FROZEN_ROOT = Path(
    "data/processed_starling_experimental_meaningful_cns_access_v3/"
    "BBB_Martins/scaffold"
)
OUTPUT_ROOT = Path(
    "data/processed_starling_context_conditioned_selected_v2/"
    "BBB_Martins/scaffold"
)


def _source_rows_by_index() -> dict[int, dict[str, Any]]:
    frame = pd.read_parquet(FROZEN_CANDIDATE_SOURCE)
    return {
        int(row["source_index"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def _apply_v3_endpoint_contract(
    review_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    source_rows = _source_rows_by_index()
    output: list[dict[str, Any]] = []
    newly_rejected = 0
    for row in review_rows:
        updated = dict(row)
        if str(row.get("review_status") or "").lower() == "accepted":
            index = int(row["source_index"])
            label, method = label_record(
                source_rows[index],
                source_index=index,
                allow_conditioned_context=True,
            )
            if label is None:
                updated.update(
                    {
                        "review_status": "rejected",
                        "review_reason": f"v3_endpoint_contract_rejected:{method}",
                    }
                )
                newly_rejected += 1
            else:
                if int(row["Y"]) != int(label):
                    raise RuntimeError(
                        f"v3 changed accepted condition label at source_index={index}"
                    )
                updated["label_method"] = method
        output.append(updated)
    return output, newly_rejected


def build() -> dict[str, Any]:
    reviewed = merge_terminal_verdicts(
        queue_path=REVIEW_ROOT / "review_queue.jsonl",
        verdict_path=REVIEW_ROOT / "review_verdicts.jsonl",
    )
    reviewed, newly_rejected = _apply_v3_endpoint_contract(reviewed)
    summary = build_reviewed_conditioned_benchmark(
        config=ConditionedBenchmarkConfig(
            task_name="BBB_Martins",
            lineage=LINEAGE,
            protocol_version=PROTOCOL_VERSION,
            frozen_root=FROZEN_ROOT,
            output_root=OUTPUT_ROOT,
            source_artifacts=(
                FROZEN_CANDIDATE_SOURCE,
                REVIEW_ROOT / "review_queue.jsonl",
                REVIEW_ROOT / "review_queue_manifest.json",
                REVIEW_ROOT / "model_prereview.jsonl",
                REVIEW_ROOT / "model_prereview_round2.jsonl",
                REVIEW_ROOT / "review_comparison_summary.json",
                REVIEW_ROOT / "review_verdicts.jsonl",
            ),
            row_id_prefix="BBBCTX",
            frozen_lineage=FROZEN_LINEAGE,
            reference_conditioned_root=Path(
                "data/processed_starling_context_conditioned_selected_v1/"
                "BBB_Martins/scaffold"
            ),
            allowed_condition_groups=SELECTED_EXTERNAL_CONDITION_GROUPS,
            publish_allowed_group_audit_only=True,
            review_policy=(
                "reuse frozen dual semantic condition review; independently "
                "revalidate accepted endpoint rows against BBB gold v3"
            ),
        ),
        review_rows=reviewed,
    )
    summary["n_previously_accepted_condition_rows_rejected_by_v3"] = newly_rejected
    write_json_atomic(OUTPUT_ROOT / "summary.json", summary)
    return summary


def main() -> int:
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
