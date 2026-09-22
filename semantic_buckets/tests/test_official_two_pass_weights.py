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
    assert schedule.execution_backend.tolist() == [
        "local_dgx", "openrouter", "local_dgx", "openrouter"
    ]


def test_pass2_schedule_splits_logical_requests_as_evenly_as_possible() -> None:
    rows = [
        {"level": "L4", "semantic_bucket_id": f"b{index:05d}", "weight": 1 - index / 100_000}
        for index in range(10_563)
    ]

    counts = weights._pass2_schedule(pd.DataFrame(rows)).execution_backend.value_counts()

    assert counts["local_dgx"] == 755
    assert counts["openrouter"] == 754


def test_ames_configuration_is_unreviewed_and_uses_all_later_levels() -> None:
    try:
        weights.configure_task("ames")

        assert weights.LEVELS == ("L2", "L3", "L4", "L5")
        assert weights.EXPECTED_BUCKET_COUNT == 11_391
        assert weights.FINAL_STATUS == "complete_unreviewed_candidate"
        assert weights.SEMANTIC_REVIEW_STATUS == "unreviewed_candidate"
    finally:
        weights.configure_task("skin_reaction")


def test_component_bounds_large_dimensions_and_reports_omissions() -> None:
    try:
        weights.configure_task("ames")
        rows = pd.DataFrame({
            "values_json": [json.dumps({"assay": f"assay-{index}"}) for index in range(12)]
        })

        component = weights._component("fixed_mutation", rows)
        values = component["canonical_dimensions"]["assay"]

        assert len(values) == weights.DIMENSION_VALUE_LIMIT + 1
        assert values[-1] == "[4 additional values omitted]"
    finally:
        weights.configure_task("skin_reaction")


def test_v6_card_renders_enriched_experimental_metadata() -> None:
    try:
        weights.configure_task("ames")
        payload = {
            "identity": {"source_components": [{
                "source_id": "mutagenicity_outcomes",
                "source_label": "Mutagenicity outcome evidence",
                "canonical_dimensions": {"canonical_endpoint_concept": ["micronucleus"]},
            }]},
            "sample_records": [{
                "source_id": "mutagenicity_outcomes",
                "source_name": "reviewed source",
                "test_system": "human lymphocytes",
                "metabolic_activation": "with S9",
                "result_call": "positive",
            }],
        }

        card = weights._card("Candidate 1", payload)

        assert "Evidence family: Mutagenicity outcome evidence" in card
        assert "test system: human lymphocytes" in card
        assert "metabolic activation: with S9" in card
        assert "result call: positive" in card
    finally:
        weights.configure_task("skin_reaction")
