from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from data.processing.evidence_library.record_informativeness import (
    _budget_summary,
    _complete_budget,
    _connect_database,
    _initialize_budget,
    _reserve_budget,
    iter_groups,
    render_group,
    validate_response,
    visible_record,
)


def test_grouping_prompt_and_response_contract() -> None:
    frame = pd.DataFrame(
        [
            {
                "source_id": "direct_bbb",
                "pmid": "123",
                "canonical_smiles": "CCO",
                "source_smiles": "C(C)O",
                "cleaned_record_id": "b",
                "support_text": "Second observation.",
                "measurement_text": "low",
                "confidence": 0.9,
                "needs_more_context": True,
                "measurement_resolution_route": "categorical",
            },
            {
                "source_id": "direct_bbb",
                "pmid": "123",
                "canonical_smiles": "CCO",
                "source_smiles": "CCO",
                "cleaned_record_id": "a",
                "support_text": "First observation.",
                "measurement_text": "high",
                "confidence": 0.8,
                "needs_more_context": False,
                "measurement_resolution_route": "categorical",
            },
        ]
    ).sort_values(["source_id", "pmid", "canonical_smiles", "cleaned_record_id"])

    groups = list(iter_groups(frame, "bbb_martins"))
    assert len(groups) == 1
    assert [row["cleaned_record_id"] for row in groups[0]["rows"]] == ["a", "b"]
    assert [row["id"] for row in groups[0]["rows"]] == ["0", "1"]
    visible = visible_record(frame.iloc[0].to_dict())
    assert visible["support_text"] == "First observation."
    assert not {
        "source_smiles",
        "cleaned_record_id",
        "confidence",
        "needs_more_context",
        "measurement_resolution_route",
    } & set(visible)
    prompt = render_group(groups[0], "Experimentally meaningful systemic CNS access.")
    assert "canonical SMILES: CCO" in prompt
    assert "First observation." in prompt
    assert '"confidence"' not in prompt

    response = {
        "items": [
            {
                "id": "1",
                "relevance_score": 0.8,
                "completeness_score": 0.6,
                "informativeness_score": 0.7,
                "context_requirement": "helpful",
            },
            {
                "id": "0",
                "relevance_score": 0.9,
                "completeness_score": 0.75,
                "informativeness_score": 0.85,
                "context_requirement": "none",
            },
        ]
    }
    assert [row["id"] for row in validate_response(response, ["0", "1"])] == [
        "0",
        "1",
    ]
    response["items"][1]["id"] = "1"
    with pytest.raises(ValueError, match="duplicate"):
        validate_response(response, ["0", "1"])


def test_budget_counts_only_reported_usage(tmp_path) -> None:
    metadata = {"version": "test"}
    connection = _connect_database(
        tmp_path / "requests.sqlite3", metadata=metadata, resume=False
    )
    _initialize_budget(connection, 1_000)
    lock = asyncio.Lock()

    async def exercise() -> None:
        assert await _reserve_budget(
            connection, lock, reservation_id="one"
        ) == "reserved"
        assert await _reserve_budget(
            connection, lock, reservation_id="two"
        ) == "reserved"
        await _complete_budget(
            connection,
            lock,
            reservation_id="one",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert await _reserve_budget(
            connection, lock, reservation_id="released"
        ) == "reserved"
        await _complete_budget(
            connection,
            lock,
            reservation_id="released",
            usage=None,
            release_reason="http_429",
        )
        await _complete_budget(
            connection,
            lock,
            reservation_id="two",
            usage={"input_tokens": 850, "output_tokens": 0},
        )
        assert await _reserve_budget(
            connection, lock, reservation_id="three"
        ) == "exhausted"

    asyncio.run(exercise())
    assert _budget_summary(connection) == {
        "max_tokens": 1_000,
        "input_tokens": 950,
        "output_tokens": 50,
        "conservative_unreported_tokens": 0,
        "requests_completed": 2,
        "spent_tokens": 1_000,
        "released_requests": 1,
    }
    connection.close()
    with pytest.raises(ValueError, match="metadata drift"):
        _connect_database(
            tmp_path / "requests.sqlite3",
            metadata={"version": "different"},
            resume=True,
        )
