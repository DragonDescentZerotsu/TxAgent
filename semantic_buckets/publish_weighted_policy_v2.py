"""Publish expert-weighted eligibility without mutating reviewed generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from semantic_buckets.publication import sha256_file, tree_inventory
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "policies/semantic_weighted_top10_v2.json"
GENERATION = "gold_v1_protected_incremental_v1"
ELIGIBILITY = f"{GENERATION}_weighted_v2"
TASKS = ("bbb_martins", "bioavailability_ma")


def _weights(task: str) -> dict[str, float]:
    document = json.loads(POLICY.read_text(encoding="utf-8"))
    levels = document["tasks"][task]["levels"]
    return {
        bucket: float(weight)
        for spec in levels.values()
        for bucket, weight in zip(spec["bucket_ids"], spec["weights"], strict=True)
    }


def _request_receipts(task: str) -> pd.DataFrame:
    source = ROOT / "provenance/incremental_review_20260916" / task
    schedule = pd.read_parquet(source / "final_reviews/schedule.parquet")
    schedule = schedule[schedule.kind.eq("eligibility")][["semantic_bucket_id", "request_id"]]
    connection = sqlite3.connect(source / "final_reviews/requests.sqlite3")
    requests = pd.read_sql_query(
        "SELECT request_id, served_model, provider_name, attempts, input_tokens, "
        "output_tokens, prompt_sha256 FROM requests WHERE kind='retrieval_eligibility'",
        connection,
    )
    connection.close()
    return schedule.merge(requests, on="request_id", how="left", validate="one_to_one")


def _decisions(task: str, generation: Path) -> pd.DataFrame:
    source = generation / "incremental_review_provenance/final_reviews/bucket_decisions.parquet"
    decisions = pd.read_parquet(source)
    receipts = _request_receipts(task)
    metadata = receipts.set_index("semantic_bucket_id").to_dict("index")
    for column in receipts.columns.drop("semantic_bucket_id"):
        decisions[column] = decisions.semantic_bucket_id.map(
            lambda bucket: metadata.get(str(bucket), {}).get(column)
        )
    weights = _weights(task)
    decisions["expert_weight"] = decisions.semantic_bucket_id.map(weights)
    reviewed = decisions.decision.eq("include")
    decisions["retrieval_eligible"] = reviewed.where(
        decisions.expert_weight.isna(), decisions.expert_weight.gt(0)
    )
    decisions["prior_zero_weight"] = decisions.expert_weight.eq(0)
    return decisions


def _records(release: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    source = release / f"eligibility/{GENERATION}/record_eligibility.parquet"
    records = pd.read_parquet(source)
    policy = decisions.set_index("semantic_bucket_id")
    for column in ("decision", "reason_code", "retrieval_eligible", "expert_weight", "prior_zero_weight"):
        records[column] = records.semantic_bucket_id.map(policy[column])
    if records.retrieval_eligible.isna().any():
        raise ValueError("eligibility policy does not cover every L2-L4 record")
    return records


def publish(task: str) -> dict:
    release = ROOT / "releases" / task / "v10"
    generation = release / "generations" / GENERATION
    output = release / "eligibility" / ELIGIBILITY
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    decisions = _decisions(task, generation)
    records = _records(release, decisions)
    decisions.to_parquet(output / "bucket_decisions.parquet", index=False)
    records.to_parquet(output / "record_eligibility.parquet", index=False)
    write_json_atomic(output / "manifest.json", {
        "version": "semantic_weighted_top10.v2",
        "status": "complete_reviewed",
        "task": task,
        "source_generation": GENERATION,
        "policy": str(POLICY.relative_to(ROOT)),
        "policy_sha256": sha256_file(POLICY),
        "bucket_decisions": len(decisions),
        "records": len(records),
        "eligible_records": int(records.retrieval_eligible.sum()),
    })
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected"]["retrieval_eligibility"] = f"eligibility/{ELIGIBILITY}/record_eligibility.parquet"
    manifest["coverage"]["eligible_records"] = int(records.retrieval_eligible.sum())
    manifest["eligibility_generations"][ELIGIBILITY] = tree_inventory(output)
    write_json_atomic(manifest_path, manifest)
    return {"task": task, "records": len(records), "eligible": int(records.retrieval_eligible.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        print(json.dumps(publish(task), sort_keys=True))


if __name__ == "__main__":
    main()
