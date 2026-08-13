import json

import pytest

from tools.chembl_tool.common.final_evidence_surface import (
    CARDS_ONLY,
    FINAL_EVIDENCE_CARD_CONTRACT_VERSION,
    MAX_CARDS,
    SUMMARY_ONLY,
    SUMMARY_PLUS_CARDS,
    build_final_evidence_cards,
    build_final_evidence_fields,
    compact_group_reasoning_outputs,
    final_evidence_instructions,
)


def _evidence_row(index: int) -> dict:
    return {
        "minimal_evidence": {
            "contract_version": "minimal_evidence.v1",
            "source": {"name": "Starling/test", "record_id": ""},
            "molecule": {"id": f"neighbor_{index}", "canonical_smiles": "", "names": []},
            "group": {"id": "Mechanism.test", "tier": "Tier 1", "endpoint_group": "test"},
            "endpoint": {
                "name": "test endpoint",
                "measurement": {"relation": "=", "value": index, "unit": "%"},
            },
            "text": {"evidence": "evidence " + "x" * 900, "context": "rat oral"},
            "annotations": {
                "evidence_role": "direct_outcome",
                "scope": {"species": "rat"},
                "transferability": "not_assessed",
                "uncertainty": [],
            },
            "quality": {"confidence": 0.9},
            "provenance": {"assay_id": "", "source_record_count": 2},
            "examples": [
                {
                    "molecule_name": "hidden name",
                    "smiles": "CCO",
                    "support_text": "support",
                }
            ],
        }
    }


def _retrieval(n_neighbors: int = 2, n_rows: int = 5) -> dict:
    return {
        "groups": [
            {
                "group_id": "Flat.all_evidence",
                "neighbors": [
                    {
                        "rank": index + 1,
                        "similarity": 0.8 - index * 0.01,
                        "similarity_bucket": "close_analog",
                        "similarity_metric": "tanimoto",
                        "molecule_relation": "structural_analog",
                        "source_group_ids": ["Direct.test"],
                        "evidence_rows": [_evidence_row(row) for row in range(n_rows)],
                        "prefetched_comparisons": [
                            {
                                "tool_name": "properties_compare",
                                "status": "ok",
                                "content": "comparison " + "y" * 1000,
                            }
                        ],
                    }
                    for index in range(n_neighbors)
                ],
            }
        ]
    }


def test_summary_only_is_a_strict_no_op_for_group_outputs():
    outputs = [{"group_id": "g", "content": {"x": 1}}]
    fields, audit = build_final_evidence_fields(
        _retrieval(),
        outputs,
        surface=SUMMARY_ONLY,
    )
    assert fields == {"group_reasoning_outputs": outputs}
    assert fields["group_reasoning_outputs"] is outputs
    assert audit is None
    assert final_evidence_instructions(SUMMARY_ONLY) == []


def test_group_branch_compaction_keeps_only_validated_content():
    valid = {
        "group_id": "valid",
        "status": "ok",
        "llm": {
            "content": {"assessment": "usable"},
            "structured_output_validation": {"valid": True},
        },
    }
    invalid = {
        "group_id": "invalid",
        "status": "error",
        "llm": {"content": {"assessment": "ignore"}},
    }
    assert compact_group_reasoning_outputs([valid, invalid]) == [
        {"group_id": "valid", "status": "ok", "content": {"assessment": "usable"}},
        {"group_id": "invalid", "status": "error", "content": None},
    ]


def test_summary_plus_cards_is_deterministic_bounded_and_identity_safe():
    retrieval = _retrieval(n_neighbors=MAX_CARDS + 4, n_rows=7)
    first = build_final_evidence_cards(retrieval, surface=SUMMARY_PLUS_CARDS)
    second = build_final_evidence_cards(retrieval, surface=SUMMARY_PLUS_CARDS)

    assert first == second
    assert first["contract_version"] == FINAL_EVIDENCE_CARD_CONTRACT_VERSION
    assert len(first["cards"]) == MAX_CARDS
    assert first["audit"]["n_candidate_cards"] == MAX_CARDS + 4
    assert all(len(card["evidence_rows"]) == 3 for card in first["cards"])
    serialized = json.dumps(first, ensure_ascii=False)
    assert "hidden name" not in serialized
    assert "CCO" not in serialized
    assert "…[truncated]" in serialized


def test_cards_only_omits_group_summaries_but_plus_cards_keeps_them():
    outputs = [{"group_id": "g", "content": {"x": 1}}]
    cards_only, cards_only_audit = build_final_evidence_fields(
        _retrieval(),
        outputs,
        surface=CARDS_ONLY,
    )
    plus, plus_audit = build_final_evidence_fields(
        _retrieval(),
        outputs,
        surface=SUMMARY_PLUS_CARDS,
    )
    assert "group_reasoning_outputs" not in cards_only
    assert "group_reasoning_outputs" in plus
    assert cards_only_audit is not None
    assert plus_audit is not None


def test_unknown_surface_fails_closed():
    with pytest.raises(ValueError, match="Unknown final evidence surface"):
        build_final_evidence_fields(_retrieval(), [], surface="unknown")
