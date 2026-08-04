import json
import hashlib

import pandas as pd
import pytest

from tools.chembl_tool.tasks.bbb_martins.data_processing.build_direct_endpoint_mapping import (
    MAPPING_VERSION as LOCAL_MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bbb_martins.data_processing.reconcile_direct_endpoints import (
    _cleaned_final_mapping,
    _near_duplicate_pairs,
    _validate_change_reviews,
    _validate_catalog_primary,
    build_global_catalog,
    build_review_packets,
    consolidate,
)
from tools.chembl_tool.tasks.bbb_martins.starling_endpoint_normalization import (
    EXPECTED_DIRECT_INVENTORY_COUNT,
    EXPECTED_DIRECT_INVENTORY_SHA256,
    EXPECTED_DIRECT_SOURCE_SHA256,
)


def _assignments(path, rows):
    frame = pd.DataFrame(
        [
            {
                "source_id": "direct_bbb",
                "input_column": "quant_metric",
                "output_field": "canonical_endpoint",
                "cluster_id": f"cluster_{index % 2}",
                "raw_value": raw,
                "provisional_endpoint": provisional,
            }
            for index, (raw, provisional) in enumerate(rows)
        ]
    )
    frame.to_parquet(path, index=False)
    return frame


def _jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _local_mapping(path, rows):
    mapping = {
        json.dumps([raw], separators=(",", ":")): provisional
        for raw, provisional in rows
    }
    mapping[json.dumps([None], separators=(",", ":"))] = None
    path.write_text(
        json.dumps(
            {
                "mapping_version": LOCAL_MAPPING_VERSION,
                "sources": {
                    "direct_bbb": {
                        "canonical_endpoint": {
                            "source_columns": ["quant_metric"],
                            "mapping": mapping,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_provenance(root, local_mapping, assignments, primary, checker, adjudication):
    generation = root / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "mapping_version": LOCAL_MAPPING_VERSION,
                "mapping_sha256": _sha(local_mapping),
                "model": "gpt-5.4-mini",
                "reasoning_effort": "medium",
                "max_cluster_size": 500,
                "global_model_reconciliation_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    provisional = root / "provisional_manifest.json"
    provisional.write_text(
        json.dumps(
            {
                "human_approved": False,
                "generation": {"mapping_sha256": _sha(local_mapping)},
                "artifacts": {"cluster_assignments": {"sha256": _sha(assignments)}},
                "source": {
                    "sha256": EXPECTED_DIRECT_SOURCE_SHA256,
                    "normalized_inventory_count_including_missing": EXPECTED_DIRECT_INVENTORY_COUNT,
                    "normalized_inventory_sha256": EXPECTED_DIRECT_INVENTORY_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    packet_manifest = root / "packet_manifest.json"
    packet_manifest.write_text(
        json.dumps({"input": {"sha256": _sha(assignments)}}), encoding="utf-8"
    )
    catalog_manifest = root / "catalog_manifest.json"
    global_catalog = root / "global_candidate_catalog.json"
    global_catalog.write_text("{}\n", encoding="utf-8")
    catalog_manifest.write_text(
        json.dumps(
            {
                "inputs": {
                    "assignments_sha256": _sha(assignments),
                    "primary_review_sha256": _sha(primary),
                    "checker_review_sha256": _sha(checker),
                    "adjudication_review_sha256": _sha(adjudication),
                },
                "artifacts": {
                    "global_catalog": {
                        "path": global_catalog.name,
                        "sha256": _sha(global_catalog),
                    },
                    "ownership_packets": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return generation, provisional, packet_manifest, catalog_manifest


def _local_reviews(root):
    primary = root / "primary.jsonl"
    checker = root / "checker.jsonl"
    adjudication = root / "adjudication.jsonl"
    _jsonl(
        primary,
        [
            {
                "decision_id": "local_auc_brain",
                "reviewer_id": "primary_a",
                "provisional_endpoint": "auc_brain",
                "action": "retain",
                "final_endpoint": "auc_brain",
                "rationale": "already precise",
            },
            {
                "decision_id": "local_brain_auc",
                "reviewer_id": "primary_a",
                "provisional_endpoint": "brain_auc",
                "action": "retain",
                "final_endpoint": "brain_auc",
                "rationale": "defer namespace merge to global catalog",
            },
            {
                "decision_id": "local_csf",
                "reviewer_id": "primary_a",
                "provisional_endpoint": "csf_ratio",
                "action": "rename",
                "final_endpoint": "csf_to_plasma_ratio",
                "rationale": "raw text states the denominator",
            },
            {
                "decision_id": "local_rejected",
                "reviewer_id": "primary_a",
                "provisional_endpoint": "literal_measure",
                "action": "rename",
                "final_endpoint": "invented_measure",
                "rationale": "proposal intentionally rejected in the fixture",
            },
        ],
    )
    _jsonl(
        checker,
        [
            {"decision_id": "local_csf", "checker_id": "checker_b", "verdict": "agree", "rationale": "supported"},
            {"decision_id": "local_rejected", "checker_id": "checker_b", "verdict": "disagree", "rationale": "unsupported"},
        ],
    )
    _jsonl(
        adjudication,
        [
            {"decision_id": "local_csf", "adjudicator_id": "adjudicator_c", "verdict": "accept", "rationale": "supported"},
            {"decision_id": "local_rejected", "adjudicator_id": "adjudicator_c", "verdict": "reject", "rationale": "retain literal label"},
        ],
    )
    return primary, checker, adjudication


def test_review_packets_cover_every_cluster_label_and_raw_value(tmp_path):
    assignments = tmp_path / "assignments.parquet"
    _assignments(
        assignments,
        [("brain AUC", "brain_auc"), ("AUC brain", "auc_brain"), ("CSF/plasma", "csf_ratio")],
    )
    manifest = build_review_packets(
        assignments_path=assignments,
        output_dir=tmp_path / "packets",
        max_packet_bytes=10_000,
        enforce_source_pins=False,
    )
    assert manifest["coverage"] == {
        "local_clusters": 2,
        "provisional_labels": 3,
        "raw_assignments": 3,
        "packets": 1,
    }
    assert all(
        value
        for key, value in manifest["validations"].items()
        if key != "source_and_inventory_pins_verified"
    )
    assert manifest["validations"]["source_and_inventory_pins_verified"] is False


def test_global_catalog_exposes_one_namespace_to_every_owner(tmp_path):
    rows = [
        ("brain AUC", "brain_auc"),
        ("AUC brain", "auc_brain"),
        ("CSF/plasma", "csf_ratio"),
        ("literal", "literal_measure"),
    ]
    assignments = tmp_path / "assignments.parquet"
    _assignments(assignments, rows)
    primary, checker, adjudication = _local_reviews(tmp_path)
    manifest = build_global_catalog(
        assignments_path=assignments,
        primary_path=primary,
        checker_path=checker,
        adjudication_path=adjudication,
        output_dir=tmp_path / "catalog",
        owned_labels_per_packet=2,
        enforce_source_pins=False,
    )
    assert manifest["counts"]["candidate_labels"] == 4
    assert manifest["counts"]["catalog_packets"] == 2
    assert manifest["validations"]["cross_local_cluster_namespace_review_enabled"]
    for packet in manifest["artifacts"]["ownership_packets"]:
        payload = json.loads((tmp_path / "catalog" / packet["path"]).read_text())
        assert payload["global_catalog_path"] == "global_candidate_catalog.json"
        assert payload["require_full_catalog_comparison"] is True


def test_catalog_merge_targets_must_be_retained_representatives():
    rows = [
        {"decision_id": "a", "reviewer_id": "p", "candidate_endpoint": "a", "action": "merge", "final_endpoint": "b", "rationale": "merge"},
        {"decision_id": "b", "reviewer_id": "p", "candidate_endpoint": "b", "action": "merge", "final_endpoint": "c", "rationale": "chain"},
        {"decision_id": "c", "reviewer_id": "p", "candidate_endpoint": "c", "action": "retain", "final_endpoint": "c", "rationale": "representative"},
    ]
    with pytest.raises(ValueError, match="retained representative"):
        _validate_catalog_primary(rows, {"a", "b", "c"})
    rows[1].update(action="retain", final_endpoint="b")
    validated = _validate_catalog_primary(rows, {"a", "b", "c"})
    assert validated["a"]["final_endpoint"] == "b"


def test_near_duplicate_audit_preserves_direction_and_matrix_order():
    pairs = _near_duplicate_pairs(
        [
            "auc_brain",
            "brain_auc",
            "brain_to_plasma_ratio",
            "plasma_to_brain_ratio",
            "brain_blood_ratio",
            "blood_brain_ratio",
        ]
    )
    names = {(row["left"], row["right"]) for row in pairs}
    assert ("auc_brain", "brain_auc") in names
    assert ("brain_to_plasma_ratio", "plasma_to_brain_ratio") not in names
    assert ("blood_brain_ratio", "brain_blood_ratio") not in names


def test_cleaned_raw_aliases_must_converge():
    mapping, aliases = _cleaned_final_mapping({"A": "endpoint", "Ａ": "endpoint"})
    assert mapping == {"A": "endpoint"}
    assert aliases == {"A": ["A", "Ａ"]}
    with pytest.raises(ValueError, match="conflicting global labels"):
        _cleaned_final_mapping({"A": "endpoint_a", "Ａ": "endpoint_b"})


def test_changed_decision_requires_three_independent_reviewers(tmp_path):
    primary = {
        "change": {
            "decision_id": "change",
            "reviewer_id": "same_person",
            "action": "rename",
        }
    }
    checker = tmp_path / "checker.jsonl"
    adjudication = tmp_path / "adjudication.jsonl"
    _jsonl(checker, [{"decision_id": "change", "checker_id": "same_person", "verdict": "agree", "rationale": "not independent"}])
    _jsonl(adjudication, [{"decision_id": "change", "adjudicator_id": "third_person", "verdict": "accept", "rationale": "fixture"}])
    with pytest.raises(ValueError, match="not independent"):
        _validate_change_reviews(
            primary=primary,
            checker_path=checker,
            adjudication_path=adjudication,
            change_actions={"rename"},
            primary_identity_field="reviewer_id",
        )


def test_consolidation_binds_local_provenance_and_applies_checked_catalog_merge(tmp_path):
    rows = [
        ("brain AUC", "brain_auc"),
        ("AUC brain", "auc_brain"),
        ("CSF/plasma", "csf_ratio"),
        ("literal", "literal_measure"),
    ]
    assignments = tmp_path / "assignments.parquet"
    _assignments(assignments, rows)
    local_mapping = tmp_path / "local.json"
    _local_mapping(local_mapping, rows)
    primary, checker, adjudication = _local_reviews(tmp_path)
    catalog_primary = tmp_path / "catalog_primary.jsonl"
    catalog_checker = tmp_path / "catalog_checker.jsonl"
    catalog_adjudication = tmp_path / "catalog_adjudication.jsonl"
    _jsonl(
        catalog_primary,
        [
            {"decision_id": "cat_auc", "reviewer_id": "catalog_p", "candidate_endpoint": "auc_brain", "action": "retain", "final_endpoint": "auc_brain", "rationale": "shared representative"},
            {"decision_id": "cat_brain_auc", "reviewer_id": "catalog_p", "candidate_endpoint": "brain_auc", "action": "merge", "final_endpoint": "auc_brain", "rationale": "same matrix and AUC endpoint"},
            {"decision_id": "cat_csf", "reviewer_id": "catalog_p", "candidate_endpoint": "csf_to_plasma_ratio", "action": "merge", "final_endpoint": "auc_brain", "rationale": "intentionally rejected fixture merge"},
            {"decision_id": "cat_literal", "reviewer_id": "catalog_p", "candidate_endpoint": "literal_measure", "action": "retain", "final_endpoint": "literal_measure", "rationale": "rejected local inference falls back"},
        ],
    )
    _jsonl(catalog_checker, [
        {"decision_id": "cat_brain_auc", "checker_id": "catalog_c", "verdict": "agree", "rationale": "synonyms"},
        {"decision_id": "cat_csf", "checker_id": "catalog_c", "verdict": "disagree", "rationale": "different endpoint"},
    ])
    _jsonl(catalog_adjudication, [
        {"decision_id": "cat_brain_auc", "adjudicator_id": "catalog_a", "verdict": "accept", "rationale": "merge"},
        {"decision_id": "cat_csf", "adjudicator_id": "catalog_a", "verdict": "reject", "rationale": "retain candidate"},
    ])
    generation, provisional, packet_manifest, catalog_manifest = _generation_provenance(
        tmp_path, local_mapping, assignments, primary, checker, adjudication
    )
    manifest = consolidate(
        assignments_path=assignments,
        local_mapping_path=local_mapping,
        primary_path=primary,
        checker_path=checker,
        adjudication_path=adjudication,
        catalog_primary_path=catalog_primary,
        catalog_checker_path=catalog_checker,
        catalog_adjudication_path=catalog_adjudication,
        output_dir=tmp_path / "proposal",
        enforce_source_pins=False,
        generation_audit_path=generation,
        provisional_manifest_path=provisional,
        review_packet_manifest_path=packet_manifest,
        catalog_manifest_path=catalog_manifest,
    )
    proposal = json.loads((tmp_path / "proposal/proposed_endpoint_mapping.json").read_text())
    assert proposal["approval"]["human_approved"] is False
    assert proposal["mapping"]["brain AUC"] == "auc_brain"
    assert proposal["mapping"]["AUC brain"] == "auc_brain"
    assert proposal["mapping"]["CSF/plasma"] == "csf_to_plasma_ratio"
    assert proposal["mapping"]["literal"] == "literal_measure"
    assert manifest["publication_status"] == "awaiting_human_approval"
    assert manifest["counts"]["final_labels_spanning_provisional_labels"] == 1
    assert manifest["validations"]["local_mapping_exactly_matches_cluster_assignments"]
    assert manifest["validations"]["generation_provenance_verified"]

    broken = json.loads(local_mapping.read_text())
    broken["sources"]["direct_bbb"]["canonical_endpoint"]["mapping"][json.dumps(["brain AUC"], separators=(",", ":"))] = "wrong_label"
    local_mapping.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        consolidate(
            assignments_path=assignments,
            local_mapping_path=local_mapping,
            primary_path=primary,
            checker_path=checker,
            adjudication_path=adjudication,
            catalog_primary_path=catalog_primary,
            catalog_checker_path=catalog_checker,
            catalog_adjudication_path=catalog_adjudication,
            output_dir=tmp_path / "broken",
            enforce_source_pins=False,
            verify_generation_provenance=False,
        )
