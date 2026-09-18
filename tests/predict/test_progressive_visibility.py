"""Test executable score-scope projection without asserting prompt prose."""

from copy import deepcopy
import json

import pytest

from predict.harnesses.progressive._visibility import (
    similarity_view,
    validate_visible_messages,
)


def _active():
    return {
        "analog": {
            "morgan_similarity": 0.123456,
            "query_analog_tool_summaries": [
                {
                    "content": (
                        "Morgan fingerprint Tanimoto similarity: 0.12.\n"
                        "MCS query coverage: 0.75."
                    )
                }
            ],
            "cards": {"record": {"morgan_similarity": 0.12}},
        }
    }


def test_similarity_projection_is_detached_and_molecule_scoped():
    source = _active()
    before = deepcopy(source)

    projected = similarity_view(source, precision=4)

    assert source == before
    assert projected["analog"]["morgan_similarity"] == 0.1235
    assert "morgan_similarity" not in projected["analog"]["cards"]["record"]
    assert projected["analog"]["query_analog_tool_summaries"][0]["content"] == (
        "MCS query coverage: 0.75."
    )


def test_hidden_similarity_does_not_require_a_record_score():
    source = {
        "analog": {
            "group_kind": "parent_molecule",
            "morgan_similarity": 0.4,
            "cards": {"record": {"transfer_likelihood": 0.8}},
        }
    }

    projected = similarity_view(source, hidden=True)

    assert "morgan_similarity" not in projected["analog"]
    assert projected["analog"]["cards"]["record"] == {"transfer_likelihood": 0.8}


def test_visibility_validation_rejects_record_level_similarity():
    payload = {
        "active_evidence": [
            {
                "morgan_similarity": 0.2,
                "evidence_cards": [{"card_id": "R1"}],
                "query_analog_tool_summaries": [],
            }
        ]
    }
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": json.dumps(payload)},
    ]
    validate_visible_messages(messages)
    payload["active_evidence"][0]["evidence_cards"][0]["morgan_similarity"] = 0.2
    messages[1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="into a record"):
        validate_visible_messages(messages)
