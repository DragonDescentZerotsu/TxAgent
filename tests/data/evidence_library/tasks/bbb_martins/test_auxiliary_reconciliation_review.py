import json

import pandas as pd
import pytest

from data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.auxiliary_reconciliation_review import (
    apply_accepted_decisions,
    build_label_catalog,
    build_review_packets,
    consolidate_review,
    validate_primary_decisions,
    validate_reviewer_packet_coverage,
    write_proposal,
)


def _assignments() -> pd.DataFrame:
    rows = [
        ("direct_bbb", "global_context", "d1", "v0000", "MRI", "mri"),
        ("direct_bbb", "global_context", "d2", "v0000", "MR imaging", "mr imaging"),
        ("direct_bbb", "global_species_context", "ds", "v0000", "rat", "rat"),
        ("passive_permeability", "global_context", "p", "v0000", "PAMPA-BBB", "pampa-bbb"),
        ("passive_permeability", "global_species_context", "ps", "v0000", "human", "human"),
        ("efflux_transport", "global_context", "e", "v0000", "A to B", "transport"),
        ("efflux_transport", "global_context", "e", "v0001", "B to A", "transport"),
        ("efflux_transport", "global_species_context", "es", "v0000", "mice", "mice"),
    ]
    records = []
    for source, output, cluster, item, raw, label in rows:
        records.append(
            {
                "source_id": source,
                "output_field": output,
                "input_column": "input",
                "cluster_id": cluster,
                "item_id": item,
                "raw_value": raw,
                "tuple_key": json.dumps([raw], separators=(",", ":")),
                "provisional_label": label,
                "cache_identity": f"cache-{cluster}",
                "served_model": "model",
                "response_sha256": f"{sum(cluster.encode()):064x}",
            }
        )
    return pd.DataFrame(records)


def _decision(decision_id, namespace, action, inputs, finals, examples, **extra):
    return {
        "decision_id": decision_id,
        "namespace": namespace,
        "action": action,
        "input_labels": inputs,
        "final_labels": finals,
        "affected_raw_value_count": len(examples),
        "supporting_raw_examples": examples,
        "evidence_summary": "The raw values support this disposition.",
        "rationale": "Preserve explicit assay meaning while removing a lexical duplicate.",
        "preserved_dimensions": ["assay_or_model_type"],
        "removed_details": [],
        "reviewer_id": "primary",
        "decision_status": "proposed",
        **extra,
    }


def _primary():
    return [
        _decision(
            "merge-mri",
            "direct_bbb/global_context",
            "merge",
            ["mri", "mr imaging"],
            ["mri"],
            ["MRI", "MR imaging"],
        ),
        _decision("retain-rat", "direct_bbb/global_species_context", "retain", ["rat"], ["rat"], ["rat"]),
        _decision("retain-pampa", "passive_permeability/global_context", "retain", ["pampa-bbb"], ["pampa-bbb"], ["PAMPA-BBB"]),
        _decision("retain-human", "passive_permeability/global_species_context", "retain", ["human"], ["human"], ["human"]),
        _decision(
            "split-direction",
            "efflux_transport/global_context",
            "split",
            ["transport"],
            ["a-to-b transport", "b-to-a transport"],
            ["A to B", "B to A"],
            raw_value_assignments=[
                {"raw_value": "A to B", "final_label": "a-to-b transport"},
                {"raw_value": "B to A", "final_label": "b-to-a transport"},
            ],
        ),
        _decision("rename-mice", "efflux_transport/global_species_context", "rename", ["mice"], ["mouse"], ["mice"]),
    ]


def _mapping(assignments):
    sources = {}
    for (source, output), group in assignments.groupby(["source_id", "output_field"]):
        mapping = {row.tuple_key: row.provisional_label for row in group.itertuples()}
        mapping["[null]"] = None
        sources.setdefault(source, {})[output] = {
            "source_columns": ["input"],
            "mapping": mapping,
        }
    return {"mapping_version": "provisional", "sources": sources}


def test_review_packets_are_atomic_complete_and_byte_bounded(tmp_path):
    assignments = _assignments()
    catalog = build_label_catalog(assignments)
    manifest = build_review_packets(
        assignments, output_dir=tmp_path, max_packet_bytes=10_000
    )
    assert len(catalog) == 7
    assert manifest["coverage"] == {
        "namespaces": 6,
        "section_clusters": 7,
        "assignments": 8,
        "packets": 6,
    }
    assert all(record["bytes"] <= 10_000 for record in manifest["packets"])
    assert all(manifest["validations"].values())
    coverage = validate_reviewer_packet_coverage(
        [
            {
                "reviewer_id": "primary",
                "reviewed_packet_ids": [row["packet_id"] for row in manifest["packets"]],
                "reviewed_cluster_count": 7,
                "reviewed_assignment_count": 8,
            }
        ],
        packet_manifest=manifest,
    )
    assert coverage["every_packet_owned_exactly_once"] is True


def test_three_stage_review_replays_merge_split_and_rename(tmp_path):
    assignments = _assignments()
    primary = _primary()
    audit = validate_primary_decisions(primary, assignments=assignments)
    assert audit["labels_missing"] == 0
    change_ids = [item["decision_id"] for item in primary if item["action"] != "retain"]
    checks = [
        {"decision_id": value, "verdict": "agree", "checker_id": "checker", "rationale": "Source support and coverage verified."}
        for value in change_ids
    ]
    adjudications = [
        {"decision_id": value, "verdict": "accept", "adjudicator_id": "adjudicator", "rationale": "The checked proposal follows policy."}
        for value in change_ids
    ]
    accepted, review_audit = consolidate_review(
        primary, checks, adjudications, assignments=assignments
    )
    proposal, changes, proposal_audit = apply_accepted_decisions(
        _mapping(assignments), assignments, accepted
    )
    assert review_audit["accepted_changes"] == 3
    assert len(changes) == 4
    assert set(changes["final_label"]) == {
        "mri",
        "a-to-b transport",
        "b-to-a transport",
        "mouse",
    }
    assert proposal["sources"]["efflux_transport"]["global_context"]["mapping"][
        '["A to B"]'
    ] == "a-to-b transport"
    assert proposal_audit["validations"]["human_approved"] is False
    manifest = write_proposal(
        output_dir=tmp_path,
        provisional_mapping=_mapping(assignments),
        assignments=assignments,
        primary=primary,
        checks=checks,
        adjudications=adjudications,
    )
    assert manifest["publication_status"] == "awaiting_human_approval"
    assert manifest["publication_blockers"] == ["human_approval_not_recorded"]
    assert (tmp_path / "RECONCILIATION_PROPOSAL.md").exists()


def test_rejects_incomplete_split_and_cross_namespace_decision():
    assignments = _assignments()
    incomplete = _primary()
    incomplete[4]["raw_value_assignments"] = incomplete[4]["raw_value_assignments"][:1]
    with pytest.raises(ValueError, match="raw coverage mismatch"):
        validate_primary_decisions(incomplete, assignments=assignments)
    cross_namespace = _primary()
    cross_namespace[0]["namespace"] = "not/a_namespace"
    with pytest.raises(ValueError, match="forbidden namespace"):
        validate_primary_decisions(cross_namespace, assignments=assignments)
