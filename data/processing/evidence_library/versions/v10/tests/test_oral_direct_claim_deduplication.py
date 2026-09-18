from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma import (
    direct_record_mapping,
)


def test_gold_vote_grouping_does_not_join_physical_evidence_rows(monkeypatch):
    uids = [
        "sr_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "sr_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ]
    voters = {
        uid: {
            "source_row_uid": uid,
            "vote_label": 1,
            "condition_group": "prandial_state=fasted",
        }
        for uid in uids
    }
    monkeypatch.setattr(
        direct_record_mapping,
        "_mapping_inputs",
        lambda: ({"hf:10", "local:4:ext_1"}, voters),
    )
    records = [
        {
            "canonical_record_id": "record-hf",
            "source_row_uid": uids[0],
            "source_id": "hf_bioavailability",
            "source_record_id": "10",
            "source_row_number": 11,
        },
        {
            "canonical_record_id": "record-local",
            "source_row_uid": uids[1],
            "source_id": "oral_exposure",
            "source_record_id": "ext_1",
            "source_row_number": 5,
        },
    ]

    mappings = direct_record_mapping.build_direct_record_mapping(records)

    assert len(mappings) == 2
    assert {row["direct_vote_label"] for row in mappings} == {1}
    assert {row["condition_group"] for row in mappings} == {
        "prandial_state=fasted"
    }
    assert len({row["direct_vote_unit_id"] for row in mappings}) == 2
    assert all("canonical_claim_id" not in row for row in mappings)


def test_nonvoter_direct_record_has_no_gold_label(monkeypatch):
    monkeypatch.setattr(
        direct_record_mapping,
        "_mapping_inputs",
        lambda: ({"local:4:ext_1"}, {}),
    )
    [mapping] = direct_record_mapping.build_direct_record_mapping(
        [
            {
                "canonical_record_id": "record-local",
                "source_row_uid": "sr_cccccccccccccccccccccccccccccccc",
                "source_id": "oral_exposure",
                "source_record_id": "ext_1",
                "source_row_number": 5,
                "qualifying_conditions": "special diet",
            }
        ]
    )

    assert mapping["retrieval_source_id"] == "direct_residual"
    assert mapping["direct_vote_label"] is None
    assert mapping["condition_group"] == "unresolved:record-local"
    assert "canonical_claim_id" not in mapping
