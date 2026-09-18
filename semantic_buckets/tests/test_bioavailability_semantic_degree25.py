import json

import pytest

from semantic_buckets import bioavailability_semantic_degree25 as ranking


def test_bradley_terry_orders_consistent_winners() -> None:
    scores = ranking._fit_bradley_terry(
        [("a", "b", "a"), ("a", "c", "a"), ("b", "c", "b")]
    )
    assert scores["a"] > scores["b"] > scores["c"]
    assert abs(sum(scores.values())) < 1e-8


def test_bradley_terry_tie_gives_equal_scores() -> None:
    scores = ranking._fit_bradley_terry([("a", "b", "TIE")])
    assert abs(scores["a"] - scores["b"]) < 1e-8


def test_ranking_payload_uses_at_most_five_distinct_atoms() -> None:
    atoms = [f"atom-{index}" for index in range(7)]
    lookup = {
        atom: {
            "source_id": "fa",
            "values": {
                "canonical_endpoint_concept": f"endpoint-{index}",
                "canonical_unit_text": "unit",
                "canonical_assay_context": "context",
                "canonical_species_context": "human",
            },
        }
        for index, atom in enumerate(atoms)
    }
    cards = {
        atom: {
            "canonical_endpoint_concept": f"endpoint-{index}",
            "canonical_unit_text": "unit",
        }
        for index, atom in enumerate(atoms)
    }
    payload = ranking._bucket_payload("bucket", atoms, lookup, cards)
    assert len(payload["sample_records"]) == 5
    assert len({row["canonical_endpoint_concept"] for row in payload["sample_records"]}) == 5


def test_ranking_payload_uses_only_approved_prompt_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        ranking.semantic,
        "PROMPT_DIMENSION_COLUMNS",
        {"fa": ("canonical_endpoint_concept",)},
    )
    lookup = {
        "atom": {
            "source_id": "fa",
            "values": {
                "canonical_endpoint_concept": "solubility",
                "canonical_direct_condition_group": "must-not-leak",
            },
        }
    }
    cards = {
        "atom": {
            "canonical_endpoint_concept": "solubility",
            "canonical_direct_condition_group": "must-not-leak",
            "canonical_record_id": "must-not-leak",
        }
    }

    payload = ranking._bucket_payload("bucket", ["atom"], lookup, cards)

    dimensions = payload["identity"]["source_components"][0][
        "canonical_dimensions"
    ]
    assert dimensions == {"canonical_endpoint_concept": ["solubility"]}
    assert payload["sample_records"] == [
        {"source_id": "fa", "canonical_endpoint_concept": "solubility"}
    ]


def test_unique_id_copy_error_is_adjudicated(tmp_path) -> None:
    connection = ranking.semantic._request_database(tmp_path / "requests.sqlite3")
    ranking.semantic._queue_request(
        connection,
        request_id="request",
        kind="ranking",
        phase="ranking",
        prompt="prompt",
        reasoning_effort="low",
        max_tokens=100,
        validation={
            "candidate_bucket_ids": [
                "sb_266568b4546be18199f3",
                "sb_266a19f56915ef8fd9f4",
            ]
        },
    )
    connection.execute(
        "UPDATE requests SET status='failed',attempts=2 WHERE request_id='request'"
    )
    connection.executemany(
        "INSERT INTO request_attempt_receipts VALUES (?,?,?)",
        [
            (
                "request",
                1,
                json.dumps(
                    {
                        "model": ranking.semantic.MODEL,
                        "provider_name": "test-provider",
                        "provider_base_url": "http://test-provider/v1",
                        "raw_response": json.dumps(
                            {"winner_bucket_id": "sb_266b4546be18199f3"}
                        ),
                    }
                ),
            ),
            ("request", 2, json.dumps({"raw_response": "{}"})),
        ],
    )
    connection.commit()

    assert ranking._adjudicate_unique_prefix_failures(connection, ["request"]) == 1
    row = connection.execute(
        "SELECT status,response_json,provider_name,provider_base_url,error "
        "FROM requests WHERE request_id='request'"
    ).fetchone()
    connection.close()
    assert row["status"] == "complete"
    assert (
        json.loads(row["response_json"])["winner_bucket_id"]
        == "sb_266568b4546be18199f3"
    )
    assert row["error"].startswith("adjudication_fallback:")
    assert row["provider_name"] == "test-provider"
    assert row["provider_base_url"] == "http://test-provider/v1"


def test_completed_request_lookup_fails_closed(tmp_path) -> None:
    connection = ranking.semantic._request_database(tmp_path / "requests.sqlite3")
    for request_id in ("complete", "pending"):
        ranking.semantic._queue_request(
            connection,
            request_id=request_id,
            kind="ranking",
            phase="ranking",
            prompt=request_id,
            reasoning_effort="low",
            max_tokens=100,
            validation={"candidate_bucket_ids": ["a", "b"]},
        )
    connection.execute(
        "UPDATE requests SET status='complete',response_json=? WHERE request_id='complete'",
        (json.dumps({"winner_bucket_id": "a"}),),
    )
    connection.commit()

    assert set(ranking._completed_request_lookup(connection, ["complete"])) == {
        "complete"
    }
    with pytest.raises(ValueError, match="pending"):
        ranking._completed_request_lookup(connection, ["complete", "pending"])
    connection.close()
