import json
from pathlib import Path

import pandas as pd

from semantic_buckets.artifacts import resolve_semantic_bucket_artifacts
from semantic_buckets.publication import sha256_file


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies/semantic_weighted_top10_v2.json"


def _entry(task: str, level: str, bucket: str) -> tuple[int, float]:
    document = json.loads(POLICY.read_text(encoding="utf-8"))
    spec = document["tasks"][task]["levels"][level]
    index = spec["bucket_ids"].index(bucket)
    return index + 1, float(spec["weights"][index])


def test_policy_is_bound_to_active_rankings() -> None:
    document = json.loads(POLICY.read_text(encoding="utf-8"))
    assert document["version"] == "semantic_weighted_top10.v2"
    for task, spec in document["tasks"].items():
        assert sha256_file(ROOT.parent / spec["ranking_path"]) == spec["ranking_sha256"]
        assert sha256_file(ROOT.parent / spec["record_ranking_path"]) == spec["record_ranking_sha256"]
        rankings = pd.read_parquet(ROOT.parent / spec["ranking_path"])
        for level, level_spec in spec["levels"].items():
            observed = rankings[rankings.level.eq(level)].nsmallest(10, "level_rank")
            assert observed.semantic_bucket_id.tolist() == level_spec["bucket_ids"]


def test_new_top_ten_decisions_are_effective() -> None:
    assert _entry("bbb_martins", "L4", "sb_1f5dba8db75a3bd2fc9b") == (5, 0.7)
    assert _entry("bioavailability_ma", "L2", "sb_44b6c9e5770cd98c32c8") == (3, 0.0)
    expected = {
        "bbb_martins": ("sb_1f5dba8db75a3bd2fc9b", True, 0.7),
        "bioavailability_ma": ("sb_44b6c9e5770cd98c32c8", False, 0.0),
    }
    for task, (bucket, eligible, weight) in expected.items():
        artifact = resolve_semantic_bucket_artifacts(task)
        frame = pd.read_parquet(artifact.retrieval_eligibility)
        rows = frame[frame.semantic_bucket_id.eq(bucket)]
        assert set(rows.retrieval_eligible) == {eligible}
        assert set(rows.expert_weight) == {weight}


def test_incremental_reviews_retain_request_provenance() -> None:
    generation = "gold_v1_protected_incremental_v1_weighted_v2"
    for task in ("bbb_martins", "bioavailability_ma"):
        path = ROOT / f"releases/{task}/v10/eligibility/{generation}/bucket_decisions.parquet"
        decisions = pd.read_parquet(path)
        reviewed = decisions[decisions.request_id.notna()]
        assert not reviewed.empty
        assert reviewed.served_model.notna().all()
        assert reviewed.prompt_sha256.notna().all()
