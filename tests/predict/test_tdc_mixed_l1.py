from __future__ import annotations

import json
from pathlib import Path

import pytest

from predict.retrieval.assay_reranking.runtime import (
    DATA_ACTIVE_CACHE_ROOT,
    cache_profile_root,
)
from predict.utils.json import read_jsonl, sha256_file


@pytest.mark.parametrize("task,subset,queries", [
    ("bbb_martins", "valid", 197),
    ("bbb_martins", "test", 530),
    ("bioavailability_ma", "valid", 64),
    ("bioavailability_ma", "test", 128),
])
def test_mixed_l1_cache_is_complete_and_disjoint(
    task: str, subset: str, queries: int
) -> None:
    root = cache_profile_root("tdc_mixed_l1_v1") / task
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["subsets"][subset]
    rows = read_jsonl(root / entry["cache"])

    assert cache_profile_root("tdc_mixed_l1_v1").parent == DATA_ACTIVE_CACHE_ROOT
    assert manifest["schema_version"] == "tdc_mixed_l1_morgan.v1"
    assert manifest["status"] == "complete"
    assert len(rows) == entry["queries"] == queries
    assert sha256_file(root / entry["cache"]) == entry["cache_sha256"]
    assert all(len(row["cards"]) == 10 for row in rows)
    assert all(
        card["molecule_identity_key"] != row["query_molecule_identity_key"]
        and (
            not row["query_scaffold"]
            or card["bemis_murcko_scaffold"] != row["query_scaffold"]
        )
        for row in rows
        for card in row["cards"]
    )
    assert {card["source_kind"] for row in rows for card in row["cards"]} == {
        "gold_v1",
        "tdc_v1",
    }
