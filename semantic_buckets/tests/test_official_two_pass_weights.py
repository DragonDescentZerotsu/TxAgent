import json

import pandas as pd

from semantic_buckets import official_two_pass_weights as weights


def test_pass2_schedule_uses_first_five_then_trailing_two_anchors() -> None:
    rows = [
        {"level": "L2", "semantic_bucket_id": f"b{index:02d}", "weight": 1 - index / 100}
        for index in range(27)
    ]

    schedule = weights._pass2_schedule(pd.DataFrame(rows))

    assert schedule.candidate_bucket_ids_json.map(json.loads).map(len).tolist() == [12, 7, 7, 1]
    assert json.loads(schedule.iloc[1].anchor_bucket_ids_json) == [
        "b07", "b08", "b09", "b10", "b11",
    ]
    assert json.loads(schedule.iloc[2].anchor_bucket_ids_json) == ["b17", "b18"]
    assert json.loads(schedule.iloc[3].anchor_bucket_ids_json) == ["b24", "b25"]
