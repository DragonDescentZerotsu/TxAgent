from predict.retrieval.assay_reranking import build_indirect_morgan_semantic_cache_v3 as v3
from predict.retrieval.assay_reranking import indirect_cache


def test_v3_payload_keeps_bbb_semantic_display_metadata():
    assert {
        "canonical_measurement_scale_id",
        "canonical_category_id",
        "canonical_transporter_identifier",
    } <= v3.CORE_PAYLOAD_FIELDS


def test_top_quartile_rounds_up_by_whole_semantic_bucket():
    binding = {}
    for rank, (bucket, count) in enumerate((("a", 2), ("b", 4), ("c", 14)), 1):
        for index in range(count):
            binding[f"{bucket}-{index}"] = {
                "level": "L2", "semantic_bucket_id": bucket,
                "level_rank": rank, "retrieval_eligible": True,
            }
    for level in ("L3", "L4"):
        binding[level] = {"level": level, "semantic_bucket_id": level,
                          "level_rank": 1, "retrieval_eligible": True}

    selected, audit = v3._top_quartile(binding)

    assert {"a", "b"} <= selected
    assert "c" not in selected
    assert audit["L2"] == {"eligible_records": 20, "target_records": 5,
                           "selected_records": 6, "selected_buckets": 2,
                           "last_bucket_rank": 2}


def test_semantic_sample_uses_morgan_parent_order_and_cap_ten():
    query = {"benchmark_row_id": "q"}
    grouped = {}
    parents = [f"p{index}" for index in range(1, 11)]
    for parent in parents:
        grouped["L2", parent] = [
            {"record_id": f"{parent}-{index}", "parent_id": parent, "level": "L2",
             "semantic_bucket_id": "kept", "retrieval_eligible": True}
            for index in range(15)
        ]

    selected = v3._sample(
        query, "bbb_martins", "L2",
        [(parent, 1 - index / 100) for index, parent in enumerate(parents)],
        grouped, {"kept"}, True, v3.LLM_SEMANTIC_LIMIT,
    )

    assert len(selected) == 100
    assert [row["parent_id"] for row in selected[:10]] == ["p1"] * 10
    assert [row["parent_id"] for row in selected[10:20]] == ["p2"] * 10


def test_reader_accepts_v3_capacity_contract():
    assert indirect_cache._expected_contract(indirect_cache.SCHEMA_V3) == {
        "control_records_per_level": 25,
        "semantic_records_per_level": 25,
        "llm_semantic_candidates_per_level": 100,
        "records_per_parent": 10,
    }


def test_direct_semantic_is_prefix_of_llm_semantic_candidates():
    query = {"benchmark_row_id": "q"}
    ranked = [(f"p{index}", 1 - index / 100) for index in range(12)]
    grouped = {
        ("L2", parent): [
            {"record_id": f"{parent}-{record}", "parent_id": parent, "level": "L2",
             "semantic_bucket_id": "kept", "retrieval_eligible": True}
            for record in range(10)
        ]
        for parent, _ in ranked
    }
    direct = v3._sample(query, "bbb_martins", "L2", ranked, grouped,
                        {"kept"}, True, v3.SEMANTIC_LIMIT)
    candidates = v3._sample(query, "bbb_martins", "L2", ranked, grouped,
                            {"kept"}, True, v3.LLM_SEMANTIC_LIMIT)

    assert len(direct) == 25
    assert len(candidates) == 100
    assert [row["record_id"] for row in direct] == [
        row["record_id"] for row in candidates[:25]
    ]
