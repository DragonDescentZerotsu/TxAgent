import pandas as pd

from semantic_buckets import publish_frozen_weights as publication


def test_weight_rank_is_per_level_and_deterministic_for_ties() -> None:
    frame = pd.DataFrame([
        {"task": "bbb_martins", "level": "L2", "semantic_bucket_id": "a", "weight": 0.5, "level_rank": 2},
        {"task": "bbb_martins", "level": "L2", "semantic_bucket_id": "b", "weight": 0.5, "level_rank": 1},
        {"task": "bbb_martins", "level": "L3", "semantic_bucket_id": "c", "weight": 0.2, "level_rank": 1},
    ])

    ranked = publication._rank_weights(frame)

    assert ranked.semantic_bucket_id.tolist() == ["b", "a", "c"]
    assert ranked.weight_rank.tolist() == [1, 2, 1]
    assert ranked.weight_percentile.tolist() == [100.0, 0.0, 0.0]
