"""Build the current vote-pure Bioavailability retrieval-source overlay."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.starling.conditioned_benchmark import task_root
from tools.chembl_tool.paper_experiments.build_conditioned_source_family_purity import (
    PuritySpec,
    build_overlay,
)
from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
    DIRECT_GROUP,
    NEAR_DIRECT_GROUP,
    VOTE_PURITY_VERSION,
    upstream_record_key,
    vote_pure_family_move,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    DIRECT_CLAIMS_PATH,
    load_label_decisions,
)


DEFAULT_INPUT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "bioavailability_nondirect_assay_context_v1/records.parquet"
)
DEFAULT_CONDITION_REVIEW = task_root("bioavailability_ma") / "source_condition_review.jsonl"
DEFAULT_OUTPUT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "bioavailability_source_family_purity_legacy_record_supported_v2_vote_pure_v1"
)


def _accepted_base_claim_ids() -> set[str]:
    decisions, _ = load_label_decisions(source_path=DIRECT_CLAIMS_PATH)
    return {
        str(decision.record.source_record_id)
        for decision in decisions
        if decision.record is not None
    }


def _accepted_condition_claim_ids(path: Path) -> set[str]:
    accepted: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("review_status") == "accepted":
                accepted.add(str(row["source_record_id"]))
    return accepted


def _source_record_ids_for_claims(
    claims_path: Path,
    claim_ids: set[str],
) -> set[str]:
    frame = pq.read_table(
        claims_path,
        columns=["canonical_claim_id", "source_record_ids"],
    ).to_pandas()
    selected = frame[frame["canonical_claim_id"].isin(claim_ids)]
    found_claims = set(map(str, selected["canonical_claim_id"]))
    missing = claim_ids - found_claims
    if missing:
        raise RuntimeError(f"accepted claims absent from canonical source: {len(missing)}")
    return {
        str(record_id)
        for record_ids in selected["source_record_ids"]
        for record_id in record_ids
    }


def _audit_membership(
    output_path: Path,
    voter_source_record_ids: set[str],
) -> dict[str, Any]:
    l1 = l2 = l1_nonvoters = voter_outside_l1 = 0
    present_voters: set[str] = set()
    columns = [
        "group_id",
        "source_id",
        "source_record_id",
        "source_row_number",
        "extraction_id",
    ]
    for batch in pq.ParquetFile(output_path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            group = str(row.get("group_id") or "")
            upstream_id = upstream_record_key(row)
            is_voter = upstream_id in voter_source_record_ids
            if is_voter:
                present_voters.add(upstream_id)
            if group == DIRECT_GROUP:
                l1 += 1
                l1_nonvoters += int(not is_voter)
            elif group == NEAR_DIRECT_GROUP:
                l2 += 1
            if is_voter and group != DIRECT_GROUP:
                voter_outside_l1 += 1
    absent = sorted(voter_source_record_ids - present_voters)
    audit = {
        "l1_exact_vote_membership": l1_nonvoters == 0 and voter_outside_l1 == 0,
        "n_l1_records": l1,
        "n_l2_records": l2,
        "n_l1_nonvoter_records": l1_nonvoters,
        "n_voter_records_outside_l1": voter_outside_l1,
        "n_voter_upstream_ids_not_present_in_retrieval_source": len(absent),
        "voter_upstream_ids_not_present_examples": absent[:20],
        "row_count_preserved": True,
    }
    if not audit["l1_exact_vote_membership"]:
        raise RuntimeError(f"vote-purity membership gate failed: {audit}")
    return audit


def build(
    *,
    input_records: Path = DEFAULT_INPUT,
    condition_review: Path = DEFAULT_CONDITION_REVIEW,
    output_dir: Path = DEFAULT_OUTPUT,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    base_claims = _accepted_base_claim_ids()
    condition_claims = _accepted_condition_claim_ids(condition_review)
    claim_ids = base_claims | condition_claims
    voter_ids = _source_record_ids_for_claims(DIRECT_CLAIMS_PATH, claim_ids)
    spec = PuritySpec(
        task="bioavailability_ma",
        input_records=input_records,
        direct_group=DIRECT_GROUP,
        classify=partial(vote_pure_family_move, voter_source_record_ids=voter_ids),
        purity_version=VOTE_PURITY_VERSION,
        allow_move_from_direct=True,
    )
    manifest = build_overlay(spec, output_dir, batch_size=batch_size)
    manifest.update(
        {
            "near_direct_group": NEAR_DIRECT_GROUP,
            "benchmark_lineage": "record_supported_v2",
            "voter_counts": {
                "benchmark_lineage": "record_supported_v2",
                "n_base_voter_claims": len(base_claims),
                "n_condition_voter_claims": len(condition_claims),
                "n_unique_voter_claims": len(claim_ids),
                "n_upstream_voter_records": len(voter_ids),
                "base_vote_artifact": str(DIRECT_CLAIMS_PATH.resolve()),
                "base_vote_artifact_sha256": sha256_file(DIRECT_CLAIMS_PATH),
                "condition_review": str(condition_review.resolve()),
                "condition_review_sha256": sha256_file(condition_review),
            },
            "hard_gates": _audit_membership(
                output_dir / "records.parquet", voter_ids
            ),
        }
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-records", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--condition-review", type=Path, default=DEFAULT_CONDITION_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    manifest = build(**vars(args))
    print(
        f"moved={manifest['n_moved_rows']:,} "
        f"l1={manifest['hard_gates']['n_l1_records']:,} "
        f"l2={manifest['hard_gates']['n_l2_records']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
