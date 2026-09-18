"""Build compact TSV audits for the active semantic-bucket policy."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pandas as pd
from scipy.stats import spearmanr

from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
POLICY = ROOT / "policies/semantic_weighted_top10_v2.json"
OUTPUT = ROOT / "audits/ranking_weight_eligibility_v2"
GENERATION = "gold_v1_protected_incremental_v1"
ELIGIBILITY = f"{GENERATION}_weighted_v2"
TASKS = ("bbb_martins", "bioavailability_ma")
EXAMPLE_BUCKETS = {
    "bbb_martins": "sb_1f5dba8db75a3bd2fc9b",
    "bioavailability_ma": "sb_44b6c9e5770cd98c32c8",
}


def _paths(task: str) -> tuple[Path, Path, Path]:
    release = ROOT / "releases" / task / "v10"
    generation = release / "generations" / GENERATION
    eligibility = release / "eligibility" / ELIGIBILITY
    provenance = ROOT / "provenance/incremental_review_20260916" / task
    return generation, eligibility, provenance


def _top_ten(task: str, policy: dict) -> pd.DataFrame:
    generation, eligibility, _ = _paths(task)
    ranks = pd.read_parquet(generation / "degree25_rankings/semantic_bucket_rankings.parquet")
    decisions = pd.read_parquet(eligibility / "bucket_decisions.parquet")
    records = pd.read_parquet(generation / "degree25_rankings/record_relevance_rankings.parquet")
    counts = records.groupby("semantic_bucket_id").size().rename("record_count")
    rows = []
    for level, spec in policy["tasks"][task]["levels"].items():
        for rank, (bucket, weight, rationale) in enumerate(zip(
            spec["bucket_ids"], spec["weights"], spec["rationales"], strict=True
        ), 1):
            rows.append({"task": task, "level": level, "policy_rank": rank,
                         "semantic_bucket_id": bucket, "expert_weight": weight,
                         "weight_rationale": rationale})
    frame = pd.DataFrame(rows).merge(
        ranks.drop(columns="task_id"),
        on=["level", "semantic_bucket_id"],
        how="left",
        validate="one_to_one",
    )
    columns = ["semantic_bucket_id", "decision", "reason_code", "retrieval_eligible"]
    return frame.merge(
        decisions[columns], on="semantic_bucket_id", how="left", validate="one_to_one"
    ).join(counts, on="semantic_bucket_id")


def _new_decisions(task: str) -> pd.DataFrame:
    _, eligibility, provenance = _paths(task)
    alignment = pd.read_parquet(provenance / "alignment/decisions.parquet")
    new_ids = set(alignment.loc[~alignment.attached, "semantic_bucket_id"].astype(str))
    decisions = pd.read_parquet(eligibility / "bucket_decisions.parquet")
    output = decisions[decisions.semantic_bucket_id.astype(str).isin(new_ids)].copy()
    output.insert(0, "task", task)
    return output.sort_values(["level", "semantic_bucket_id"])


def _correlations(top: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", "all", top)]
    groups.extend((task, "all", frame) for task, frame in top.groupby("task"))
    groups.extend((task, level, frame) for (task, level), frame in top.groupby(["task", "level"]))
    for task, level, frame in groups:
        result = spearmanr(frame.level_rank, frame.expert_weight)
        rows.append({"task": task, "level": level, "n": len(frame),
                     "rank_number_rho": result.statistic,
                     "ranking_priority_rho": -result.statistic,
                     "p_value": result.pvalue})
    return pd.DataFrame(rows)


def _l5_additions() -> pd.DataFrame:
    task = "bbb_martins"
    generation, _, _ = _paths(task)
    current = pd.read_parquet(generation / "semantic_bucket_map.parquet")
    old_root = ROOT / "history/pre_gold_v2" / task / "v10/semantic_buckets"
    old_manifest = json.loads((old_root / "manifest.json").read_text())
    old = pd.read_parquet(old_root / old_manifest["selected"]["semantic_map"])
    old_ids = set(old.loc[old.level.eq("L5"), "semantic_bucket_id"])
    added = current[current.level.eq("L5") & ~current.semantic_bucket_id.isin(old_ids)].copy()
    added["assay_transfer_eligible"] = False
    return added.sort_values(["semantic_bucket_id", "atom_id"])


def _rendered_requests(task: str) -> None:
    _, _, provenance = _paths(task)
    schedule = pd.read_parquet(provenance / "final_reviews/schedule.parquet")
    rows = schedule[schedule.semantic_bucket_id.eq(EXAMPLE_BUCKETS[task])]
    rows = rows.sort_values(["kind", "request_id"]).groupby("kind").head(1)
    connection = sqlite3.connect(provenance / "final_reviews/requests.sqlite3")
    connection.row_factory = sqlite3.Row
    destination = OUTPUT / "rendered_requests"
    destination.mkdir(exist_ok=True)
    for item in rows.itertuples(index=False):
        request = connection.execute(
            "SELECT request_id, kind, phase, prompt_sha256, prompt, response_json, "
            "served_model, provider_name FROM requests WHERE request_id=?",
            (item.request_id,),
        ).fetchone()
        write_json_atomic(destination / f"{task}_{item.kind}.json", dict(request))
    connection.close()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    top = pd.concat([_top_ten(task, policy) for task in TASKS], ignore_index=True)
    new = pd.concat([_new_decisions(task) for task in TASKS], ignore_index=True)
    top.to_csv(OUTPUT / "top10_policy.tsv", sep="\t", index=False)
    new.to_csv(OUTPUT / "incremental_bucket_decisions.tsv", sep="\t", index=False)
    _correlations(top).to_csv(OUTPUT / "rank_weight_correlations.tsv", sep="\t", index=False)
    _l5_additions().to_csv(OUTPUT / "bbb_l5_non_assay_transfer.tsv", sep="\t", index=False)
    for task in TASKS:
        _rendered_requests(task)
    outputs = {
        str(path.relative_to(OUTPUT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json_atomic(OUTPUT / "manifest.json", {
        "version": "ranking_weight_eligibility_audit.v2", "status": "complete",
        "policy": str(POLICY.relative_to(ROOT)), "policy_sha256": sha256_file(POLICY),
        "top10_rows": len(top), "incremental_bucket_decisions": len(new),
        "bbb_l5_non_assay_transfer_atoms": len(_l5_additions()), "outputs": outputs,
    })


if __name__ == "__main__":
    main()
