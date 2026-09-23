import json

import pandas as pd

from semantic_buckets.score_ames_endpoint_children import _schedule_pass2
from semantic_buckets.publish_ames_endpoint_candidate_weights import _weights


def test_child_only_chains_start_with_old_anchors_then_slide() -> None:
    children = [f"child_{index:02d}" for index in range(31)]
    pass1 = pd.DataFrame({
        "level": ["L2"] * len(children),
        "semantic_bucket_id": children,
        "weight": [1 - index / 100 for index in range(len(children))],
    })
    lineage = pd.DataFrame({
        "candidate_semantic_bucket_id": children,
        "parent_semantic_bucket_id": ["parent"] * len(children),
    })
    old = pd.DataFrame({
        "level": ["L2"] * 4,
        "semantic_bucket_id": ["parent", "old_1", "old_2", "old_3"],
        "weight": [0.9, 0.88, 0.86, 0.84],
    })

    schedule = _schedule_pass2(pass1, lineage, old).sort_values("wave")
    candidates = [json.loads(value) for value in schedule.candidate_bucket_ids_json]
    anchors = [json.loads(value) for value in schedule.anchor_bucket_ids_json]

    assert [len(batch) for batch in candidates] == [12, 7, 7, 5]
    assert [item for batch in candidates for item in batch] == children
    assert anchors[0][0] == "parent" and set(anchors[0]) == set(old.semantic_bucket_id)
    assert anchors[1] == candidates[0][-5:]
    assert anchors[2] == candidates[1][-2:]
    assert anchors[3] == candidates[2][-2:]


def test_publish_weights_drops_retired_parents_and_recomputes_ranks() -> None:
    original = pd.DataFrame({
        "task": ["ames"] * 3, "level": ["L2"] * 3,
        "semantic_bucket_id": ["retired", "kept_a", "kept_b"],
        "weight": [0.99, 0.7, 0.4], "rationale": ["old"] * 3,
    })
    children = pd.DataFrame({
        "task": ["ames", "ames"], "level": ["L2", "L2"],
        "semantic_bucket_id": ["child_a", "child_b"],
        "weight": [0.8, 0.3], "rationale": ["new", "new"],
    })
    semantic = pd.DataFrame({
        "semantic_bucket_id": ["kept_a", "kept_b", "child_a", "child_b"],
    })

    actual = _weights(original, children, semantic)

    assert actual.semantic_bucket_id.tolist() == ["child_a", "kept_a", "kept_b", "child_b"]
    assert actual.final_rank.tolist() == [1, 2, 3, 4]
    assert actual.weight_origin.tolist() == [
        "endpoint_child_two_pass", "original_two_pass", "original_two_pass",
        "endpoint_child_two_pass",
    ]
