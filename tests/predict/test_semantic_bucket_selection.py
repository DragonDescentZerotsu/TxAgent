from predict.retrieval.assay_reranking.semantic_bucket_selection import (
    select_global_molecule_records,
    select_molecule_records,
    select_records,
    select_semantic_lap,
    select_semantic_weighted,
)


def test_global_molecule_selection_pairs_all_control_with_filtered_semantic():
    rows = [
        {"record_id": "bad", "parent_id": "p1", "parent_rank": 1,
         "semantic_bucket_id": "bad", "semantic_rank": 1,
         "retrieval_eligible": False, "tie_key": "0"},
        {"record_id": "p2-good", "parent_id": "p2", "parent_rank": 2,
         "semantic_bucket_id": "good", "semantic_rank": 1,
         "retrieval_eligible": True, "tie_key": "1"},
        {"record_id": "p1-good", "parent_id": "p1", "parent_rank": 1,
         "semantic_bucket_id": "later", "semantic_rank": 2,
         "retrieval_eligible": True, "tie_key": "2"},
        {"record_id": "unmapped", "parent_id": "p2", "parent_rank": 2,
         "semantic_bucket_id": None, "semantic_rank": None,
         "retrieval_eligible": None, "tie_key": "3"},
    ]
    control = select_global_molecule_records(rows, method="control", limit=3)
    semantic = select_global_molecule_records(rows, method="semantic", limit=3)
    assert [row["record_id"] for row in control] == ["bad", "p2-good", "p1-good"]
    assert [row["record_id"] for row in semantic] == ["p2-good", "p1-good"]


def test_global_molecule_selection_applies_per_parent_cap():
    rows = [
        {"record_id": f"{parent}-{index}", "parent_id": parent,
         "parent_rank": rank, "semantic_bucket_id": "good",
         "semantic_rank": 1, "retrieval_eligible": True,
         "tie_key": f"{index}{rank}"}
        for rank, parent in enumerate(("p1", "p2"), 1)
        for index in range(3)
    ]
    selected = select_global_molecule_records(
        rows, method="semantic", limit=4, per_parent_limit=2,
    )
    assert len(selected) == 4
    assert [row["parent_id"] for row in selected].count("p1") == 2
    assert [row["parent_id"] for row in selected].count("p2") == 2


def test_global_semantic_selection_caps_each_parent_bucket_cell():
    rows = [
        {"record_id": f"{parent}-{bucket}-{index}", "parent_id": parent,
         "parent_rank": rank, "semantic_bucket_id": bucket,
         "semantic_rank": bucket_rank, "retrieval_eligible": True,
         "tie_key": f"{index:02d}"}
        for rank, parent in enumerate(("p1", "p2"), 1)
        for bucket_rank, bucket in enumerate(("a", "b"), 1)
        for index in range(7)
    ]
    selected = select_global_molecule_records(
        rows, method="semantic", limit=20, per_parent_limit=10,
        per_parent_bucket_limit=5,
    )
    cells = [(row["parent_id"], row["semantic_bucket_id"]) for row in selected]
    assert len(selected) == 20
    assert max(cells.count(cell) for cell in set(cells)) == 5


def test_molecule_selection_filters_exclusions_and_drains_best_bucket():
    rows = [
        {"record_id": "b2", "semantic_bucket_id": "b", "semantic_rank": 2,
         "retrieval_eligible": True, "tie_key": "1"},
        {"record_id": "a2", "semantic_bucket_id": "a", "semantic_rank": 1,
         "retrieval_eligible": True, "tie_key": "2"},
        {"record_id": "a1", "semantic_bucket_id": "a", "semantic_rank": 1,
         "retrieval_eligible": True, "tie_key": "1"},
        {"record_id": "x", "semantic_bucket_id": "a", "semantic_rank": 1,
         "retrieval_eligible": False, "tie_key": "0"},
    ]
    semantic = select_molecule_records(rows, method="semantic", limit=3)
    control = select_molecule_records(rows, method="control", limit=3)
    assert [row["record_id"] for row in semantic] == ["a1", "a2", "b2"]
    assert [row["record_id"] for row in control] == ["b2", "a1", "a2"]


def test_select_records_filters_and_round_robins_pair_buckets():
    rows = [
        {"record_id": "a1", "pair_bucket_key": "a", "semantic_percentile": 90,
         "morgan_similarity": .8, "assay_transfer_score": .2, "tie_key": "1"},
        {"record_id": "a2", "pair_bucket_key": "a", "semantic_percentile": 90,
         "morgan_similarity": .7, "assay_transfer_score": .9, "tie_key": "2"},
        {"record_id": "b1", "pair_bucket_key": "b", "semantic_percentile": 80,
         "morgan_similarity": .9, "assay_transfer_score": .8, "tie_key": "3"},
        {"record_id": "x", "pair_bucket_key": "x", "semantic_percentile": 19.9,
         "morgan_similarity": 1.0, "assay_transfer_score": 1.0, "tie_key": "4"},
    ]
    assert [row["record_id"] for row in select_records(rows, method="morgan", limit=3)] == [
        "a1", "b1", "a2"
    ]
    assert [row["record_id"] for row in select_records(rows, method="assay_transfer", limit=3)] == [
        "a2", "b1", "a1"
    ]


def test_semantic_lap_takes_three_per_bucket_and_revisits_sparse_inventory():
    rows = [
        {
            "record_id": f"{bucket}{index}",
            "semantic_bucket_id": bucket,
            "semantic_rank": rank,
            "morgan_similarity": 1 - index / 100,
            "tie_key": f"{bucket}{index}",
        }
        for rank, bucket in enumerate(("a", "b", "c"), 1)
        for index in range(1, 7)
    ]
    selected = select_semantic_lap(rows, limit=12)
    assert [row["record_id"] for row in selected] == [
        "a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3",
        "a4", "a5", "a6",
    ]
    assert [row["semantic_lap"] for row in selected] == [1] * 9 + [2] * 3


def test_semantic_weighted_uses_weight_times_similarity_and_caps_buckets():
    rows = [
        {
            "record_id": f"{bucket}{index}",
            "semantic_bucket_id": bucket,
            "expert_weight": weight,
            "morgan_similarity": similarity,
            "tie_key": f"{bucket}{index}",
        }
        for bucket, weight, similarities in (
            ("direct", 1.0, [.9, .8, .7]),
            ("proxy", .5, [1.0, .9]),
            ("excluded", 0.0, [1.0]),
        )
        for index, similarity in enumerate(similarities)
    ]
    selected = select_semantic_weighted(rows, limit=4, per_bucket_limit=2)
    assert [row["record_id"] for row in selected] == [
        "direct0", "direct1", "proxy0", "proxy1",
    ]
    assert [row["semantic_utility"] for row in selected] == [.9, .8, .5, .45]
