from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from tools.chembl_tool.tasks.skin_reaction.data_processing import (
    species_context_reconciliation as review,
)


SPECIES_COLUMNS = ("assay_type", "experimental_conditions", "support_text")


def _tuple(assay, conditions, support):
    return review.encode_species_tuple((assay, conditions, support))


def _cache_and_candidate():
    human = _tuple(
        "t-cell proliferation",
        "human donor lymphocytes",
        "stimulation index was measured in donor cells",
    )
    unsupported = _tuple(
        "THP-1 activation",
        "cells incubated for 24 hours",
        "CD86 expression was measured",
    )
    cattle = _tuple(
        "protein binding",
        "primary bovine keratinocytes",
        "binding was measured in the keratinocytes",
    )
    mouse = _tuple(
        "llna",
        "female CBA mice",
        "lymph-node proliferation was measured in treated mice",
    )
    cache = [
        {
            "identity": "cache-a",
            "members": [human, unsupported],
            "mapping": {"v0000": "human", "v0001": "human"},
        },
        {
            "identity": "cache-b",
            "members": [cattle, mouse],
            "mapping": {"v0000": "bovine", "v0001": "mouse"},
        },
    ]
    candidate = {
        "source_columns": list(SPECIES_COLUMNS),
        "mapping": {
            human: "human",
            unsupported: None,
            cattle: "cattle",
            mouse: "mouse",
        },
    }
    return cache, candidate, {
        "human": human,
        "unsupported": unsupported,
        "cattle": cattle,
        "mouse": mouse,
    }


def _assignments():
    cache, candidate, keys = _cache_and_candidate()
    return review.build_candidate_assignments(cache, candidate), keys


def _review_record(packet, *, role, reviewer_id, exceptions=()):
    return {
        "review_version": review.REVIEW_VERSION,
        "packet_id": packet["packet_id"],
        "packet_sha256": review.hashlib.sha256(review._json_bytes(packet)).hexdigest(),
        "reviewer_id": reviewer_id,
        "role": role,
        "reviewed_assignment_count": packet["assignment_count"],
        "complete": True,
        "exceptions": list(exceptions),
    }


def _packet_manifest(packets):
    return {
        "packets": [
            {
                "packet_id": packet["packet_id"],
                "sha256": review.hashlib.sha256(
                    review._json_bytes(packet)
                ).hexdigest(),
                "assignments": packet["assignment_count"],
                "reviewer_roles": packet["reviewer_roles"],
            }
            for packet in packets
        ]
    }


def _retain_reviews(packets):
    primary = []
    checker = []
    for packet in packets:
        primary.append(
            _review_record(
                packet,
                role="primary",
                reviewer_id=packet["reviewer_roles"]["primary"],
            )
        )
        checker.append(
            _review_record(
                packet,
                role="checker",
                reviewer_id=packet["reviewer_roles"]["checker"],
            )
        )
    return primary, checker


def _base_mapping(candidate):
    return {
        "mapping_version": "starling_auxiliary.skin_reaction.globally_reconciled.v3",
        "sources": {
            "direct_skin_reaction": {
                "global_context": {
                    "source_columns": ["assay_or_test"],
                    "mapping": {'["patch test"]': "patch test"},
                }
            },
            "sensitization_aop": {
                "global_context": {
                    "source_columns": ["assay_type"],
                    "mapping": {'["llna"]': "llna"},
                },
                "global_species_context": copy.deepcopy(candidate),
                "global_endpoint_context": {
                    "source_columns": ["endpoint_or_target"],
                    "mapping": {'["stimulation index"]': "stimulation index"},
                },
            },
            "phototoxicity_irritation_local_damage": {
                "global_context": {
                    "source_columns": ["assay_method"],
                    "mapping": {'["3T3 NRU"]': "3t3 nru"},
                }
            },
            "skin_exposure": {
                "global_context": {
                    "source_columns": ["study_design"],
                    "mapping": {'["in vitro"]': "in vitro"},
                }
            },
        },
    }


def test_species_tuple_is_readable_and_round_trips_partial_nulls():
    fields = {
        "assay_type": "LLNA",
        "experimental_conditions": None,
        "support_text": "Measured in female CBA mice",
    }
    key = review.encode_species_tuple(tuple(fields[column] for column in SPECIES_COLUMNS))

    assert key == '["LLNA",null,"Measured in female CBA mice"]'
    assert review.decode_species_tuple(key) == (
        fields["assay_type"],
        fields["experimental_conditions"],
        fields["support_text"],
    )
    assert json.loads(key) == [
        "LLNA",
        None,
        "Measured in female CBA mice",
    ]


def test_candidate_assignments_keep_raw_effective_and_final_labels_separate():
    assignments, keys = _assignments()
    by_tuple = assignments.set_index("tuple_key")

    assert by_tuple.at[keys["unsupported"], "raw_gpt_label"] == "human"
    assert pd.isna(by_tuple.at[keys["unsupported"], "candidate_label"])
    assert not bool(by_tuple.at[keys["unsupported"], "review_eligible"])
    assert by_tuple.at[keys["cattle"], "raw_gpt_label"] == "bovine"
    assert by_tuple.at[keys["cattle"], "candidate_label"] == "cattle"

    packets = review.packetize_assignments(assignments, max_packet_bytes=100_000)
    primary, checker = _retain_reviews(packets)
    final, audit = review.reconcile_assignment_reviews(
        assignments, packets, primary, checker, adjudications=[]
    )
    final_by_tuple = final.set_index("tuple_key")
    assert pd.isna(final_by_tuple.at[keys["unsupported"], "final_label"])
    assert final_by_tuple.at[keys["cattle"], "final_label"] == "cattle"
    assert audit["null_promotions"] == 0
    assert assignments["assignment_id"].nunique() == len(assignments)
    assert assignments["cluster_id"].str.startswith("cluster_").all()


def test_packets_are_cluster_atomic_bounded_and_own_every_assignment_once():
    assignments, _ = _assignments()
    # Make each of the two local clusters large enough that they cannot share a
    # 10 KB packet while either cluster still fits by itself.
    assignments.loc[:, "support_text"] = assignments["support_text"].map(
        lambda value: value + " evidence" * 400
    )
    packets = review.packetize_assignments(
        assignments, max_packet_bytes=10_000, max_assignments=200
    )

    assert len(packets) == 2
    assert all(len(review._json_bytes(packet)) <= 10_000 for packet in packets)
    owned = [
        item["assignment_id"]
        for packet in packets
        for cluster in packet["clusters"]
        for item in cluster["assignments"]
    ]
    eligible = set(assignments.loc[assignments.review_eligible, "assignment_id"])
    assert len(owned) == len(set(owned)) == len(eligible)
    assert set(owned) == eligible
    assignment_cluster = {
        row.assignment_id: row.cluster_id
        for row in assignments.itertuples(index=False)
    }
    cluster_packets = {}
    for packet in packets:
        for cluster in packet["clusters"]:
            for item in cluster["assignments"]:
                assignment_id = item["assignment_id"]
                cluster_packets.setdefault(
                    assignment_cluster[assignment_id], set()
                ).add(packet["packet_id"])
    assert all(len(packet_ids) == 1 for packet_ids in cluster_packets.values())


def test_two_review_coverage_and_three_agent_role_rotation():
    assignments, _ = _assignments()
    packets = review.packetize_assignments(assignments, max_packet_bytes=100_000)
    agents = review.REVIEWER_IDS

    assert review.reviewer_roles(0) == {
        "primary": agents[0],
        "checker": agents[1],
        "adjudicator": agents[2],
    }
    assert review.reviewer_roles(1) == {
        "primary": agents[1],
        "checker": agents[2],
        "adjudicator": agents[0],
    }
    assert review.reviewer_roles(2) == {
        "primary": agents[2],
        "checker": agents[0],
        "adjudicator": agents[1],
    }

    primary, checker = _retain_reviews(packets)
    manifest = _packet_manifest(packets)
    audit = review.validate_two_review_coverage(
        primary, checker, manifest
    )
    assert audit["assignments_reviewed_twice"] == int(
        assignments.review_eligible.sum()
    )
    assert audit["primary_complete"] is True
    assert audit["checker_complete"] is True
    assert audit["independent_reviewers"] is True

    broken = copy.deepcopy(checker)
    broken[0]["reviewer_id"] = primary[0]["reviewer_id"]
    with pytest.raises(ValueError, match="independent|role|reviewer"):
        review.validate_two_review_coverage(
            primary, broken, manifest
        )


def test_null_candidate_cannot_be_promoted_by_review():
    assignments, keys = _assignments()
    packets = review.packetize_assignments(assignments, max_packet_bytes=100_000)
    packet = packets[0]
    unsupported = next(
        row
        for row in assignments.to_dict("records")
        if row["tuple_key"] == keys["unsupported"]
    )
    primary = [
        _review_record(
            packet,
            role="primary",
            reviewer_id=packet["reviewer_roles"]["primary"],
            exceptions=[
                {
                    "assignment_id": unsupported["assignment_id"],
                    "action": "correct",
                    "proposed_label": "human",
                    "evidence_fields": ["support_text"],
                    "rationale": "attempted promotion",
                }
            ],
        )
    ]
    checker = [
        _review_record(
            packet,
            role="checker",
            reviewer_id=packet["reviewer_roles"]["checker"],
        )
    ]

    with pytest.raises(ValueError, match="unknown assignment"):
        review.reconcile_assignment_reviews(
            assignments,
            packets,
            primary,
            checker,
            adjudications=[],
        )

    final = {
        row.assignment_id: (
            None if pd.isna(row.candidate_label) else row.candidate_label
        )
        for row in assignments.itertuples(index=False)
    }
    final[unsupported["assignment_id"]] = "human"
    _, candidate, _ = _cache_and_candidate()
    with pytest.raises(ValueError, match="promote"):
        review.build_proposed_mapping(
            _base_mapping(candidate), final, assignments
        )


@pytest.mark.parametrize(
    ("label", "fields", "supported"),
    [
        (
            "human",
            {
                "assay_type": "activation assay",
                "experimental_conditions": "primary human keratinocytes",
                "support_text": "rabbit antibody used for detection",
            },
            True,
        ),
        (
            "cattle",
            {
                "assay_type": "binding assay",
                "experimental_conditions": "bovine keratinocytes",
                "support_text": None,
            },
            True,
        ),
        (
            "human",
            {
                "assay_type": "THP-1 activation",
                "experimental_conditions": "cells incubated for 24 hours",
                "support_text": "CD86 was measured",
            },
            False,
        ),
    ],
)
def test_final_non_null_label_requires_literal_packet_support(label, fields, supported):
    assert review.validate_literal_support(
        label,
        tuple(fields[column] for column in SPECIES_COLUMNS),
    ) is supported


def test_proposal_changes_only_sensitization_species_and_preserves_null_mask():
    cache, candidate, keys = _cache_and_candidate()
    assignments = review.build_candidate_assignments(cache, candidate)
    final = {
        row.assignment_id: (
            None if pd.isna(row.candidate_label) else row.candidate_label
        )
        for row in assignments.itertuples(index=False)
    }
    human = assignments.loc[assignments.tuple_key == keys["human"]].iloc[0]
    final[human.assignment_id] = None
    base = _base_mapping(candidate)
    before = copy.deepcopy(base)

    proposal = review.build_proposed_mapping(base, final, assignments)

    assert base == before, "proposal construction must not mutate its input"
    assert (
        proposal["sources"]["sensitization_aop"]["global_species_context"]
        ["mapping"][keys["human"]]
        is None
    )
    assert (
        proposal["sources"]["sensitization_aop"]["global_species_context"]
        ["mapping"][keys["unsupported"]]
        is None
    )
    for source_id, outputs in before["sources"].items():
        for output_name, section in outputs.items():
            if (source_id, output_name) != (
                "sensitization_aop",
                "global_species_context",
            ):
                assert proposal["sources"][source_id][output_name] == section


def test_exact_proposal_replay_accepts_exact_bytes_and_rejects_tampering():
    cache, candidate, _ = _cache_and_candidate()
    assignments = review.build_candidate_assignments(cache, candidate)
    base = _base_mapping(candidate)
    packets = review.packetize_assignments(assignments)
    primary, checker = _retain_reviews(packets)
    final, _ = review.reconcile_assignment_reviews(
        assignments, packets, primary, checker, adjudications=[]
    )
    proposal = review.build_proposed_mapping(base, None, final)

    assert review.replay_proposal(
        base,
        assignments,
        (primary, checker),
        [],
        proposal,
        packets,
    ) is True

    tampered = copy.deepcopy(proposal)
    species = tampered["sources"]["sensitization_aop"][
        "global_species_context"
    ]["mapping"]
    first_non_null = next(key for key, value in species.items() if value is not None)
    species[first_non_null] = "rat"
    assert review.replay_proposal(
        base,
        assignments,
        (primary, checker),
        [],
        tampered,
        packets,
    ) is False


def test_snapshot_default_is_the_frozen_v3_candidate_after_v4_publication():
    runtime_mapping = (
        review.DATA_PROCESSING_DIR
        / "globally_reconciled_auxiliary_value_mapping.json"
    )
    assert review.DEFAULT_CANDIDATE_MAPPING == (
        review.DEFAULT_RECONCILIATION_ROOT
        / "provenance"
        / "candidate_v3_mapping.json"
    )
    assert json.loads(
        review.DEFAULT_CANDIDATE_MAPPING.read_text(encoding="utf-8")
    )["mapping_version"] == review.CANDIDATE_VERSION
    assert json.loads(runtime_mapping.read_text(encoding="utf-8"))[
        "mapping_version"
    ] == review.PROPOSAL_VERSION


def test_snapshot_accepts_an_explicit_v3_candidate_and_rejects_v4(tmp_path):
    cache, candidate, _ = _cache_and_candidate()
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        "".join(json.dumps(row) + "\n" for row in cache),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(_base_mapping(candidate)), encoding="utf-8"
    )
    mapping_manifest = tmp_path / "mapping_manifest.json"
    mapping_manifest.write_text("{}\n", encoding="utf-8")

    result = review.write_snapshot(
        output_dir=tmp_path / "snapshot",
        cache_path=cache_path,
        candidate_mapping_path=candidate_path,
        mapping_manifest_path=mapping_manifest,
    )
    assert result["candidate_version"] == review.CANDIDATE_VERSION
    assert result["counts"]["assignments"] == 4

    published = _base_mapping(candidate)
    published["mapping_version"] = review.PROPOSAL_VERSION
    candidate_path.write_text(json.dumps(published), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen v3 mapping"):
        review.write_snapshot(
            output_dir=tmp_path / "rejected",
            cache_path=cache_path,
            candidate_mapping_path=candidate_path,
            mapping_manifest_path=mapping_manifest,
        )


def test_snapshot_cli_passes_the_candidate_override(monkeypatch, tmp_path):
    captured = {}

    def fake_write_snapshot(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    candidate = tmp_path / "candidate.json"
    monkeypatch.setattr(review, "write_snapshot", fake_write_snapshot)

    assert review.main(
        [
            "snapshot",
            "--work-root",
            str(tmp_path / "work"),
            "--candidate-mapping",
            str(candidate),
        ]
    ) == 0
    assert captured == {
        "output_dir": tmp_path / "work" / "provenance",
        "candidate_mapping_path": str(candidate),
    }
