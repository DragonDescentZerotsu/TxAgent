"""Legacy audit for the retired non-absolute pair-bucket review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.final_endpoint_pruning import (
    semantic_review_columns,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
)


POLICY_VERSION = "bioavailability_pair_bucket_semantic_eligibility.v1"
MINIMUM_UNIQUE_MOLECULES = 20
TASK_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = (
    TASK_ROOT
    / "data_processing/pair_bucket_semantic_eligibility_v1/policy.json"
)
DEFAULT_LIBRARY_ROOT = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7"
)


def evidence_hash(record_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in record_ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_reviewed_entry(entry: Mapping[str, Any]) -> None:
    key = str(entry.get("pair_bucket_key") or "")
    if entry.get("decision") not in {"eligible", "ineligible"}:
        raise ValueError(f"invalid pair-bucket semantic decision: {key}")
    if not str(entry.get("reason_code") or "") or not str(
        entry.get("reason") or ""
    ):
        raise ValueError(f"pair-bucket semantic decision lacks a reason: {key}")
    reviews = entry.get("reviews") or []
    reviewers = {
        str(review.get("reviewer") or "")
        for review in reviews
        if review.get("review_role") != "adjudicator"
    }
    reviewers.discard("")
    if len(reviewers) < 2:
        raise ValueError(f"pair-bucket semantic decision lacks two reviewers: {key}")
    decisions = {
        str(review.get("decision") or "")
        for review in reviews
        if review.get("review_role") != "adjudicator"
    }
    if len(decisions) > 1:
        adjudications = [
            review for review in reviews if review.get("review_role") == "adjudicator"
        ]
        if len(adjudications) != 1 or adjudications[0].get("decision") != entry.get(
            "decision"
        ):
            raise ValueError(f"pair-bucket semantic disagreement is unresolved: {key}")
    elif decisions != {entry.get("decision")}:
        raise ValueError(f"pair-bucket semantic consensus drift: {key}")


def build_candidate_inventory(
    stage05_records_path: str | Path,
    stage06_records_path: str | Path,
) -> list[dict[str, Any]]:
    """Return every reviewable non-absolute bucket with 20 collapsed molecules."""
    stage05 = pd.read_parquet(
        stage05_records_path,
        columns=["canonical_record_id", "pair_bucket_key"],
    )
    stage06 = pd.read_parquet(
        stage06_records_path,
        columns=[
            "aggregation_method",
            "assay_transfer_eligible",
            "assay_transfer_ineligibility_reason",
            "canonical_reference_scope",
            "canonical_smiles",
            "finite_scalar_value",
            "measurement_kind",
            "pair_bucket_key",
        ],
    )
    reasons = stage06["assay_transfer_ineligibility_reason"].fillna("").astype(str)
    reviewable = stage06[
        stage06["pair_bucket_key"].notna()
        & stage06["measurement_kind"].eq("continuous")
        & pd.to_numeric(stage06["finite_scalar_value"], errors="coerce").map(
            math.isfinite
        )
        & stage06["aggregation_method"].eq("continuous_median")
        & ~stage06["canonical_reference_scope"].isin(
            {"absolute", "not_applicable"}
        )
        & (
            stage06["assay_transfer_eligible"].astype(bool)
            | reasons.str.startswith("reference_scope_")
            | reasons.str.startswith("semantic_policy_")
        )
    ].copy()
    counts = reviewable.groupby("pair_bucket_key")["canonical_smiles"].nunique()
    keys = set(counts[counts >= MINIMUM_UNIQUE_MOLECULES].index.astype(str))
    stage05 = stage05[stage05["pair_bucket_key"].astype(str).isin(keys)].copy()
    ids_by_key = stage05.groupby("pair_bucket_key")["canonical_record_id"].apply(
        lambda values: [str(value) for value in values]
    )
    output = []
    for key in sorted(keys):
        identity = json.loads(key)
        if not isinstance(identity, list) or len(identity) < 3:
            raise ValueError(f"invalid pair bucket key: {key}")
        rows = reviewable[reviewable["pair_bucket_key"].astype(str) == key]
        scopes = sorted(set(rows["canonical_reference_scope"].astype(str)))
        if len(scopes) != 1:
            raise ValueError(f"pair bucket spans reference scopes: {key}")
        ids = ids_by_key.get(key, [])
        output.append(
            {
                "pair_bucket_key": key,
                "source_id": str(identity[0]),
                "endpoint": str(identity[1]),
                "canonical_unit_text": str(identity[2]),
                "canonical_reference_scope": scopes[0],
                "stage05_record_count": len(ids),
                "stage06_unique_molecule_count": int(counts[key]),
                "evidence_sha256": evidence_hash(ids),
            }
        )
    return output


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "task_id",
        "pair_bucket_version",
        "minimum_unique_molecules",
        "stage05_records_sha256",
        "entries",
    }
    if set(payload) != required:
        raise ValueError("pair-bucket semantic policy fields differ from contract")
    if payload["version"] != POLICY_VERSION or payload["task_id"] != "bioavailability_ma":
        raise ValueError("pair-bucket semantic policy version/task mismatch")
    if payload["pair_bucket_version"] != BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION:
        raise ValueError("pair-bucket semantic policy bucket-version mismatch")
    if payload["minimum_unique_molecules"] != MINIMUM_UNIQUE_MOLECULES:
        raise ValueError("pair-bucket semantic policy support threshold mismatch")
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise ValueError("pair-bucket semantic policy entries must be a list")
    keys: set[str] = set()
    for entry in entries:
        key = str(entry.get("pair_bucket_key") or "")
        if not key or key in keys:
            raise ValueError("pair-bucket semantic policy keys must be unique")
        keys.add(key)
        _validate_reviewed_entry(entry)
    return payload


def write_review_packets(
    *,
    stage05_records_path: str | Path,
    stage06_records_path: str | Path,
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    candidates = build_candidate_inventory(stage05_records_path, stage06_records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    keys = {entry["pair_bucket_key"] for entry in candidates}
    all_columns = pq.read_schema(stage05_records_path).names
    columns = list(semantic_review_columns(all_columns))
    if "pair_bucket_key" not in columns:
        columns.append("pair_bucket_key")
    rows = pd.read_parquet(stage05_records_path, columns=columns)
    rows = rows[rows["pair_bucket_key"].astype(str).isin(keys)].sort_values(
        ["pair_bucket_key", "canonical_record_id"]
    )
    with (target / "review_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows.to_dict("records"):
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )
    return candidates


def compile_policy(
    *,
    candidates: Sequence[Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    stage05_records_path: str | Path,
) -> dict[str, Any]:
    by_key = {str(row["pair_bucket_key"]): dict(row) for row in reviews}
    candidate_keys = {str(row["pair_bucket_key"]) for row in candidates}
    if set(by_key) != candidate_keys:
        raise ValueError("review keys differ from candidate pair buckets")
    entries = []
    for candidate in sorted(candidates, key=lambda row: str(row["pair_bucket_key"])):
        review = by_key[str(candidate["pair_bucket_key"])]
        entry = {
            **dict(candidate),
            **{
                field: review.get(field)
                for field in (
                    "decision",
                    "confidence",
                    "reason_code",
                    "reason",
                    "suggested_split_dimensions",
                    "reviews",
                )
            },
        }
        _validate_reviewed_entry(entry)
        entries.append(entry)
    payload = {
        "version": POLICY_VERSION,
        "task_id": "bioavailability_ma",
        "pair_bucket_version": BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
        "minimum_unique_molecules": MINIMUM_UNIQUE_MOLECULES,
        "stage05_records_sha256": file_sha256(stage05_records_path),
        "entries": entries,
    }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage05-records",
        default=str(DEFAULT_LIBRARY_ROOT / "05_deduplicated_records/records.parquet"),
    )
    parser.add_argument(
        "--stage06-records",
        default=str(DEFAULT_LIBRARY_ROOT / "06_collapsed_records/records.parquet"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reviews", default="")
    parser.add_argument("--policy-out", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    candidates = write_review_packets(
        stage05_records_path=args.stage05_records,
        stage06_records_path=args.stage06_records,
        out_dir=args.out_dir,
    )
    if args.reviews:
        reviews = [
            json.loads(line)
            for line in Path(args.reviews).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy = compile_policy(
            candidates=candidates,
            reviews=reviews,
            stage05_records_path=args.stage05_records,
        )
        output = Path(args.policy_out or DEFAULT_POLICY_PATH)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"pair_bucket_candidates={len(candidates)} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_POLICY_PATH",
    "MINIMUM_UNIQUE_MOLECULES",
    "POLICY_VERSION",
    "build_candidate_inventory",
    "compile_policy",
    "evidence_hash",
    "load_policy",
    "write_review_packets",
]
