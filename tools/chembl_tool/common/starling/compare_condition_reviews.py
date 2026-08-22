"""Compare two complete semantic pre-review ledgers without promoting gold."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(queue_path: Path, first_path: Path, second_path: Path, output_root: Path) -> dict[str, Any]:
    queue = _read(queue_path)
    ledgers = []
    for path in (first_path, second_path):
        rows = _read(path)
        by_id = {str(row["source_record_id"]): row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != {str(row["source_record_id"]) for row in queue}:
            raise ValueError(f"Review ledger does not exactly cover queue: {path}")
        if any(row.get("review_status") not in {"accepted", "rejected"} for row in rows):
            raise ValueError(f"Review ledger has non-terminal rows: {path}")
        ledgers.append(by_id)

    comparisons = []
    conservative_verdicts = []
    for candidate in queue:
        record_id = str(candidate["source_record_id"])
        first, second = ledgers[0][record_id], ledgers[1][record_id]
        if first.get("source_payload_sha256") != candidate.get("source_payload_sha256"):
            raise ValueError(f"Stale first review for {record_id}")
        if second.get("source_payload_sha256") != candidate.get("source_payload_sha256"):
            raise ValueError(f"Stale second review for {record_id}")
        first_contract = str(first.get("review_contract_sha256") or "")
        second_contract = str(second.get("review_contract_sha256") or "")
        if not first_contract or first_contract != second_contract:
            raise ValueError(f"Review contract mismatch for {record_id}")
        status_agreement = first["review_status"] == second["review_status"]
        comparisons.append(
            {
                "source_record_id": record_id,
                "source_payload_sha256": candidate["source_payload_sha256"],
                "review_contract_sha256": first_contract,
                "molecule_identity_key": candidate.get("molecule_identity_key", ""),
                "proposed_condition_group": candidate.get("proposed_condition_group", ""),
                "round1_status": first["review_status"],
                "round1_reason": first.get("review_reason", ""),
                "round1_notes": first.get("review_notes", ""),
                "round2_status": second["review_status"],
                "round2_reason": second.get("review_reason", ""),
                "round2_notes": second.get("review_notes", ""),
                "status_agreement": status_agreement,
                "consensus_status": first["review_status"] if status_agreement else "needs_adjudication",
            }
        )
        accepted = first["review_status"] == second["review_status"] == "accepted"
        conservative_verdicts.append(
            {
                "source_record_id": record_id,
                "source_payload_sha256": candidate["source_payload_sha256"],
                "review_contract_sha256": first_contract,
                "review_protocol_version": first.get("review_protocol_version", ""),
                "review_status": "accepted" if accepted else "rejected",
                "reviewer": "dual_semantic_prereview_unanimity_v2",
                "review_reason": (
                    "dual_semantic_review_unanimous_accept"
                    if accepted
                    else "dual_semantic_review_not_unanimously_accepted"
                ),
                "review_notes": (
                    f"round1={first['review_status']}:{first.get('review_reason', '')}; "
                    f"round2={second['review_status']}:{second.get('review_reason', '')}"
                ),
                "condition_group": candidate.get("proposed_condition_group", ""),
                "condition_atoms": candidate.get("proposed_condition_atoms", []),
                "Y": int(candidate["Y"]),
                "label_method": candidate.get("label_method", ""),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "review_comparison.jsonl", comparisons)
    disagreements = [row for row in comparisons if not row["status_agreement"]]
    write_jsonl_atomic(output_root / "review_disagreements.jsonl", disagreements)
    write_jsonl_atomic(output_root / "conservative_prereview_verdicts.jsonl", conservative_verdicts)
    summary = {
        "n_queue": len(queue),
        "n_status_agreement": len(queue) - len(disagreements),
        "n_status_disagreement": len(disagreements),
        "agreement_rate": (len(queue) - len(disagreements)) / len(queue) if queue else 1.0,
        "consensus_status_counts": dict(Counter(row["consensus_status"] for row in comparisons)),
        "disagreement_directions": dict(
            Counter(f"{row['round1_status']}->{row['round2_status']}" for row in disagreements)
        ),
        "notice": (
            "Agreement is a QA signal; conservative verdicts remain model-derived "
            "even if copied into the terminal benchmark ledger."
        ),
        "conservative_prereview_policy": (
            "accept only when both independent semantic reviews accept; reject "
            "every disagreement and every unanimous rejection"
        ),
    }
    write_json_atomic(output_root / "review_comparison_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.queue, args.first, args.second, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
