import json

import pandas as pd

from semantic_buckets import official_two_pass_weights as weights


def test_pass2_schedule_uses_twelve_candidates_and_last_prior_score() -> None:
    rows = [
        {"level": "L2", "semantic_bucket_id": f"b{index:02d}", "weight": 1 - index / 100}
        for index in range(27)
    ]

    schedule = weights._pass2_schedule(pd.DataFrame(rows))

    assert schedule.candidate_bucket_ids_json.map(json.loads).map(len).tolist() == [12, 12, 3]
    assert schedule.wave.tolist() == [0, 1, 2]
    assert schedule.chain.tolist() == [0, 0, 0]
    assert json.loads(schedule.iloc[0].anchor_bucket_ids_json) == []
    assert json.loads(schedule.iloc[1].anchor_bucket_ids_json) == ["b11"]
    assert json.loads(schedule.iloc[2].anchor_bucket_ids_json) == ["b23"]
    assert schedule.execution_backend.unique().tolist() == ["fixed_mixed"]


def test_l4_schedule_uses_four_seed_chains_and_shared_tail_anchors() -> None:
    rows = [
        {"level": "L4", "semantic_bucket_id": f"b{index:05d}", "weight": 1 - index / 100_000}
        for index in range(100)
    ]

    schedule = weights._pass2_schedule(pd.DataFrame(rows))

    assert schedule.groupby("wave").size().tolist() == [4, 4, 1]
    assert schedule[schedule.wave.eq(0)].anchor_bucket_ids_json.map(json.loads).tolist() == [
        [], [], [], [],
    ]
    expected = ["b00011", "b00023", "b00035", "b00047"]
    assert schedule[schedule.wave.eq(1)].anchor_bucket_ids_json.map(json.loads).tolist() == [
        expected, expected, expected, expected,
    ]
    assert schedule.candidate_bucket_ids_json.map(json.loads).explode().is_unique


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
