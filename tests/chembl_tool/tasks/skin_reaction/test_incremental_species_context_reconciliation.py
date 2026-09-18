from __future__ import annotations

import json

import pytest

from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
    species_context_reconciliation as review,
)


def _key(assay: str, conditions: str, support: str) -> str:
    return review.encode_species_tuple((assay, conditions, support))


def _base_mapping() -> dict:
    return {
        "mapping_version": review.PROPOSAL_VERSION,
        "sources": {
            review.SOURCE_ID: {
                review.OUTPUT_FIELD: {
                    "source_columns": list(review.SPECIES_COLUMNS),
                    "mapping": {
                        _key("llna", "female CBA mice", "measured in mice"): "mouse",
                        _key("patch test", "human volunteers", "human reactions"): "human",
                    },
                }
            }
        },
    }


def _candidate_and_cache():
    human = _key("activation", "human keratinocytes", "measured in human cells")
    unknown = _key("activation", "cells", "species not reported")
    candidate = _base_mapping()
    candidate["mapping_version"] = "main_universe_local_candidate.v1"
    candidate["sources"][review.SOURCE_ID][review.OUTPUT_FIELD]["mapping"].update(
        {human: "human", unknown: None}
    )
    cache = [
        {
            "identity": "delta-cluster-1",
            "members": [human, unknown],
            "mapping": {"v0000": "human", "v0001": None},
        }
    ]
    return candidate, cache, human, unknown


def _write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _retain_reviews(packets):
    primary, checker = [], []
    for packet in packets:
        packet_sha = review.hashlib.sha256(review._json_bytes(packet)).hexdigest()
        common = {
            "review_version": review.REVIEW_VERSION,
            "packet_id": packet["packet_id"],
            "packet_sha256": packet_sha,
            "reviewed_assignment_count": packet["assignment_count"],
            "complete": True,
            "exceptions": [],
        }
        primary.append(
            {
                **common,
                "role": "primary",
                "reviewer_id": packet["reviewer_roles"]["primary"],
            }
        )
        checker.append(
            {
                **common,
                "role": "checker",
                "reviewer_id": packet["reviewer_roles"]["checker"],
            }
        )
    return primary, checker


def test_incremental_snapshot_keeps_only_new_tuples(tmp_path):
    base = _base_mapping()
    candidate, cache, human, unknown = _candidate_and_cache()
    candidate = {
        "artifact_version": "skin_auxiliary.incremental_delta_candidate.v1",
        "sources": {
            review.SOURCE_ID: {
                review.OUTPUT_FIELD: {
                    "source_columns": list(review.SPECIES_COLUMNS),
                    "mapping": {human: "human", unknown: None},
                }
            }
        },
    }
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    cache_path = tmp_path / "cache.jsonl"
    _write_json(base_path, base)
    _write_json(candidate_path, candidate)
    _write_jsonl(cache_path, cache)

    manifest = review.write_incremental_snapshot(
        output_dir=tmp_path / "work/provenance",
        cache_path=cache_path,
        base_mapping_path=base_path,
        candidate_mapping_path=candidate_path,
    )

    assert manifest["counts"]["new_tuples"] == 2
    delta = json.loads(
        (tmp_path / "work/provenance/candidate_delta_mapping.json").read_text()
    )["mapping"]
    assert delta == {human: "human", unknown: None}
    assert manifest["candidate_mapping"]["shape"] == "delta_only"
    assert manifest["validations"]["reviewed_base_assignments_unchanged"] is True


def test_incremental_packets_anchor_existing_labels_and_merge_after_two_reviews(tmp_path):
    base = _base_mapping()
    candidate, cache, _, _ = _candidate_and_cache()
    paths = {
        "base": tmp_path / "base.json",
        "candidate": tmp_path / "candidate.json",
        "cache": tmp_path / "cache.jsonl",
    }
    _write_json(paths["base"], base)
    _write_json(paths["candidate"], candidate)
    _write_jsonl(paths["cache"], cache)
    root = tmp_path / "work"
    review.write_incremental_snapshot(
        output_dir=root / "provenance",
        cache_path=paths["cache"],
        base_mapping_path=paths["base"],
        candidate_mapping_path=paths["candidate"],
    )
    review.write_incremental_review_packets(work_root=root)
    packets = review._load_packet_payloads(root / "review_packets")
    anchors = packets[0]["review_policy"]["existing_label_anchors"]
    assert {row["label"] for row in anchors} == {"human", "mouse"}
    primary, checker = _retain_reviews(packets)
    _write_jsonl(root / "primary.jsonl", primary)
    _write_jsonl(root / "checker.jsonl", checker)
    _write_jsonl(root / "adjudication.jsonl", [])

    manifest = review.write_incremental_proposal(
        work_root=root,
        primary_paths=[root / "primary.jsonl"],
        checker_paths=[root / "checker.jsonl"],
        adjudication_paths=[root / "adjudication.jsonl"],
    )

    assert all(manifest["validations"].values())
    assert manifest["counts"] == {
        "reviewed_base_tuples": 2,
        "new_tuples": 2,
        "merged_tuples": 4,
    }
    proposal = json.loads(
        (root / "proposal/proposed_globally_reconciled_mapping.json").read_text()
    )
    assert proposal["mapping_version"] == review.INCREMENTAL_PROPOSAL_VERSION


def test_incremental_proposal_requires_complete_checker_and_adjudication_coverage(tmp_path):
    candidate, cache, _, _ = _candidate_and_cache()
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    cache_path = tmp_path / "cache.jsonl"
    _write_json(base_path, _base_mapping())
    _write_json(candidate_path, candidate)
    _write_jsonl(cache_path, cache)
    root = tmp_path / "work"
    review.write_incremental_snapshot(
        output_dir=root / "provenance",
        cache_path=cache_path,
        base_mapping_path=base_path,
        candidate_mapping_path=candidate_path,
    )
    review.write_incremental_review_packets(work_root=root)
    packets = review._load_packet_payloads(root / "review_packets")
    primary, checker = _retain_reviews(packets)
    _write_jsonl(root / "primary.jsonl", primary)
    _write_jsonl(root / "checker.jsonl", [])
    _write_jsonl(root / "adjudication.jsonl", [])

    with pytest.raises(ValueError, match="checker review coverage mismatch"):
        review.write_incremental_proposal(
            work_root=root,
            primary_paths=[root / "primary.jsonl"],
            checker_paths=[root / "checker.jsonl"],
            adjudication_paths=[root / "adjudication.jsonl"],
        )

    assignment_id = packets[0]["clusters"][0]["assignments"][0]["assignment_id"]
    primary[0]["exceptions"] = [
        {
            "assignment_id": assignment_id,
            "action": "abstain",
            "proposed_label": None,
            "rationale": "The packet does not establish a unique measurement subject.",
        }
    ]
    _write_jsonl(root / "primary.jsonl", primary)
    _write_jsonl(root / "checker.jsonl", checker)
    with pytest.raises(ValueError, match="adjudication coverage mismatch"):
        review.write_incremental_proposal(
            work_root=root,
            primary_paths=[root / "primary.jsonl"],
            checker_paths=[root / "checker.jsonl"],
            adjudication_paths=[root / "adjudication.jsonl"],
        )
