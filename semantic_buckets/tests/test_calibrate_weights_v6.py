import json

import pandas as pd
import pytest

from semantic_buckets import calibrate_weights_v6 as calibration


def _payload() -> dict:
    return {
        "identity": {"source_components": [{
            "source_id": "fg",
            "source_label": "Intestinal availability evidence",
            "canonical_dimensions": {
                "canonical_endpoint_concept": ["fraction escaping intestinal metabolism"],
                "canonical_species_context": ["__unknown__"],
            },
        }]},
        "sample_records": [{
            "source_id": "fg",
            "source_name": "Normalized literature extraction",
            "transporter_or_enzyme": "OATP2B1",
            "substrate_status": "substrate",
            "qualifying_conditions": "X" * 300,
        }],
    }


def test_selection_uses_v5_weight_then_original_rank_and_half_floor() -> None:
    frame = pd.DataFrame({
        "semantic_bucket_id": [f"b{i:02d}" for i in range(55)],
        "weight": [0.5] * 55,
        "level_rank": list(range(55, 0, -1)),
    })

    selected = calibration._selected(frame)

    assert len(selected) == 32
    assert selected.level_rank.tolist() == list(range(1, 33))


def test_selection_fraction_can_cover_top_three_quarters() -> None:
    frame = pd.DataFrame({
        "semantic_bucket_id": [f"b{i:02d}" for i in range(100)],
        "weight": [1 - i / 100 for i in range(100)],
        "level_rank": list(range(1, 101)),
    })

    selected = calibration._selected(frame, 0.75)

    assert len(selected) == 75
    assert selected.level_rank.tolist() == list(range(1, 76))


def test_task_level_shard_requires_complete_v5_level() -> None:
    schedule = pd.DataFrame([
        {"task": "bbb_martins", "level": "L3",
         "candidate_bucket_ids_json": '["a", "b"]'},
        {"task": "bbb_martins", "level": "L5",
         "candidate_bucket_ids_json": '["c", "d"]'},
    ])
    partial = pd.DataFrame([
        {"task": "bbb_martins", "level": "L3", "semantic_bucket_id": "a"},
        {"task": "bbb_martins", "level": "L3", "semantic_bucket_id": "b"},
        {"task": "bbb_martins", "level": "L5", "semantic_bucket_id": "c"},
    ])

    selected = calibration._select_task_levels(
        partial, schedule, [("bbb_martins", "L3")]
    )

    assert selected.semantic_bucket_id.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="V5 task-level is incomplete"):
        calibration._select_task_levels(
            partial, schedule, [("bbb_martins", "L5")]
        )


def test_full_schedule_has_planned_coverage_and_request_count() -> None:
    sizes = {
        ("bbb_martins", "L2"): 8532, ("bbb_martins", "L3"): 525,
        ("bbb_martins", "L4"): 4073, ("bbb_martins", "L5"): 290,
        ("bioavailability_ma", "L2"): 133, ("bioavailability_ma", "L3"): 4010,
        ("bioavailability_ma", "L4"): 1652, ("bioavailability_ma", "L5"): 714,
        ("bioavailability_ma", "L6"): 110,
    }
    rows = []
    for (task, level), count in sizes.items():
        rows.extend({"task": task, "level": level, "semantic_bucket_id": f"{task}-{level}-{i}",
                     "level_rank": i + 1, "weight": 1 - i / count, "rationale": "old"}
                    for i in range(count))

    schedule = calibration.build_schedule(pd.DataFrame(rows))

    assert len(schedule) == 1429
    assert sum(len(json.loads(value)) for value in schedule.candidate_bucket_ids_json) == 10021


def test_v6_uses_two_chains_and_four_rolling_anchors() -> None:
    buckets = [f"bucket-{index:02d}" for index in range(55)]

    rows = calibration._schedule_level("bbb_martins", "L2", buckets)

    assert all(json.loads(row["anchor_bucket_ids_json"]) == buckets[7:12]
               for row in rows[1:3])
    next_anchors = buckets[17:19] + buckets[24:26]
    assert all(json.loads(row["anchor_bucket_ids_json"]) == next_anchors
               for row in rows[3:5])
    assert max(row["batch"] for row in rows) == 1


def test_local_speculative_queue_never_requires_openrouter_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame([{
        "batch_id": "bbb_martins-L3-r0001-b00", "task": "bbb_martins",
        "level": "L3", "round": 1,
        "candidate_bucket_ids_json": '["candidate"]',
        "anchor_bucket_ids_json": '["anchor"]',
    }])
    monkeypatch.setattr(calibration, "_render", lambda *args: "prompt")
    connection = calibration.core._request_database(":memory:")
    try:
        request_ids = calibration._queue(
            connection, rows, {}, {}, {"anchor": {}}, local_speculative=True,
            local_base_url="http://dgx005:50002/v1",
        )
        validation = json.loads(connection.execute(
            "select validation_json from requests where request_id=?", request_ids
        ).fetchone()[0])
    finally:
        connection.close()

    assert validation["requested_model"] == calibration.speculative.DEFAULT_MODEL
    assert validation["selected_provider_route"] == "dgx005_50002_speculative_first4"


def test_enriched_prompt_shows_prior_context_and_hides_bucket_id() -> None:
    payloads = {"secret-bucket": _payload()}
    priors = {"secret-bucket": {"weight": 0.44, "rationale": "old rationale"}}

    prompt = calibration._render(
        "bioavailability_ma", "L3", ["secret-bucket"], [], payloads, priors, {}
    )

    assert "Previous weight: 0.44" in prompt
    assert "OATP2B1" in prompt and "substrate" in prompt
    assert "not reported" in prompt
    assert "secret-bucket" not in prompt and "old rationale" not in prompt
    assert "X" * 240 not in prompt


def test_final_frame_preserves_v5_audit_and_carries_unselected_weight() -> None:
    old = pd.DataFrame([
        {"semantic_bucket_id": "a", "task": "bbb_martins", "level": "L2",
         "level_rank": 1, "weight": 0.7, "rationale": "old a", "request_id": "v5-a"},
        {"semantic_bucket_id": "b", "task": "bbb_martins", "level": "L2",
         "level_rank": 2, "weight": 0.3, "rationale": "old b", "request_id": "v5-b"},
    ])
    direct = pd.DataFrame([{
        "semantic_bucket_id": "a", "task": "bbb_martins", "level": "L2",
        "weight": 0.8, "rationale": "increased", "request_id": "v6-a",
    }])

    frame = calibration._final_frame(old, direct).set_index("semantic_bucket_id")

    assert frame.loc["a", "v5_weight"] == 0.7
    assert frame.loc["a", "v5_rationale"] == "old a"
    assert frame.loc["a", "final_weight"] == 0.8
    assert frame.loc["a", "calibration_status"] == "direct_v6"
    assert frame.loc["b", "final_weight"] == 0.3
    assert frame.loc["b", "calibration_status"] == "carried_forward_v5"


def test_weight_bands_keep_zero_separate_from_negligible() -> None:
    assert calibration._band(0) == "unusable"
    assert calibration._band(0.01) == "negligible"
    assert calibration._band(0.95) == "exceptional"
