from __future__ import annotations

import hashlib
import json

import pandas as pd

from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
    build_embedding_bucket_mapping as builder,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
    generic_incremental_reconciliation as reconciliation,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_auxiliary_metadata import (
    AuxiliaryMetadataAttacher,
)


def _base_and_candidate():
    sources = {}
    candidate_sources = {}
    for source_id, source in builder.SOURCE_SPECS.items():
        sources[source_id] = {}
        candidate_sources[source_id] = {}
        for output in source.outputs:
            old_key = json.dumps(
                ["old"] * len(output.source_columns), separators=(",", ":")
            )
            section = {
                "source_columns": list(output.source_columns),
                "mapping": {old_key: "old label"},
            }
            sources[source_id][output.output_name] = section
            if f"{source_id}/{output.output_name}" != reconciliation.SPECIALIZED_NAMESPACE:
                candidate_sources[source_id][output.output_name] = {
                    "source_columns": list(output.source_columns),
                    "mapping": {},
                }
    candidate_sources["direct_skin_reaction"]["global_context"]["mapping"] = {
        '["new"]': "new label"
    }
    candidate_sources["direct_skin_reaction"]["global_severity_grade"]["mapping"] = {
        '["grade"]': "3",
    }
    return (
        {"mapping_version": reconciliation.BASE_VERSION, "sources": sources},
        {
            "artifact_version": builder.INCREMENTAL_ARTIFACT_VERSION,
            "publication_status": "unpublished_requires_agent_reconciliation",
            "sources": candidate_sources,
        },
    )


def test_proposal_overlays_delta_without_changing_v4_or_specialized_species():
    base, candidate = _base_and_candidate()
    original_species = base["sources"]["sensitization_aop"][
        "global_species_context"
    ].copy()
    reviewed = pd.DataFrame(
        [
            {
                "namespace": "direct_skin_reaction/global_context",
                "tuple_key": '["new"]',
                "final_label": "existing canonical label",
            }
        ]
    )

    proposal = reconciliation.build_proposal(base, candidate, reviewed)

    assert proposal["mapping_version"] == reconciliation.PROPOSAL_VERSION
    assert proposal["sources"]["direct_skin_reaction"]["global_context"][
        "mapping"
    ] == {
        '["new"]': "existing canonical label",
        '["old"]': "old label",
    }
    assert proposal["sources"]["direct_skin_reaction"]["global_severity_grade"][
        "mapping"
    ]['["grade"]'] == "3"
    assert proposal["sources"]["sensitization_aop"][
        "global_species_context"
    ] == original_species
    assert base["sources"]["direct_skin_reaction"]["global_context"][
        "mapping"
    ] == {'["old"]': "old label"}


def test_cleaned_alias_inherits_reviewed_base_label():
    base, _ = _base_and_candidate()
    section = base["sources"]["direct_skin_reaction"]["global_context"]
    section["mapping"] = {'["21\u2011day patch test"]': "reviewed label"}
    additions = {
        ("direct_skin_reaction", "global_context"): {
            '["21\u2010day patch test"]': "new conflicting label"
        }
    }

    inherited = reconciliation._inherit_cleaned_base_labels(base, additions)

    assert inherited == 1
    assert additions[("direct_skin_reaction", "global_context")] == {
        '["21\u2010day patch test"]': "reviewed label"
    }
    merged = json.loads(json.dumps(base))
    merged["sources"]["direct_skin_reaction"]["global_context"]["mapping"].update(
        additions[("direct_skin_reaction", "global_context")]
    )
    reconciliation._validate_cleaned_mapping(merged)


def test_exact_cache_cover_ignores_overlapping_superseded_partitions():
    records = [
        {"cluster_order": 0, "keys": frozenset({"a", "b"}), "rows": ["old"]},
        {"cluster_order": 1, "keys": frozenset({"a"}), "rows": ["a"]},
        {"cluster_order": 2, "keys": frozenset({"b", "c"}), "rows": ["bc"]},
    ]

    selected = reconciliation._select_exact_cache_cover(
        {"a", "b", "c"}, records, namespace="test/output"
    )

    assert sorted(record["rows"] for record in selected) == [["a"], ["bc"]]


def test_snapshot_proves_cache_coverage_and_freezes_severity(tmp_path, monkeypatch):
    base, candidate = _base_and_candidate()
    candidate["sources"]["direct_skin_reaction"]["global_severity_grade"][
        "mapping"
    ]['["grade-null"]'] = None
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "incremental_delta_candidate.json"
    cache_dir = tmp_path / "cluster_cache"
    cache_dir.mkdir()
    base_path.write_text(json.dumps(base), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    manifest = {
        "candidate": {"sha256": reconciliation.file_sha256(candidate_path)},
        "reviewed_base_mapping": {"sha256": reconciliation.file_sha256(base_path)},
        "model": "test-model",
        "reasoning_effort": "high",
        "max_tokens": 1234,
    }
    (tmp_path / "incremental_delta_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for output_field, member, label in (
        ("global_context", "new", "new label"),
        ("global_severity_grade", "grade", "3"),
        ("global_severity_grade", "grade-null", None),
    ):
        extraction = next(
            item
            for item in builder._source_extractions("direct_skin_reaction")
            if item.output_name == output_field
        )
        cluster_id = reconciliation._cluster_id([member])
        cluster = builder.Cluster(cluster_id, (member,))
        identity = builder._cache_identity(
            extraction,
            cluster,
            model="test-model",
            reasoning_effort="high",
            max_tokens=1234,
        )
        record = {
            "identity": identity,
            "members": [member],
            "mapping": {"v0000": label},
            "generation": {
                "attempt_count": 1,
                "attempts": [
                    {"status": "valid", "response_sha256": "a" * 64}
                ],
                "usage": {"total_tokens": 10},
            },
        }
        with (cache_dir / f"direct_skin_reaction__{output_field}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record) + "\n")
    monkeypatch.setattr(
        reconciliation,
        "EXPECTED_NONEMPTY_OPEN_NAMESPACES",
        frozenset({"direct_skin_reaction/global_context"}),
    )

    snapshot = reconciliation.write_snapshot(
        work_root=tmp_path / "work",
        candidate_path=candidate_path,
        cache_dir=cache_dir,
        base_mapping_path=base_path,
    )

    assert snapshot["counts"] == {
        "assignments": 3,
        "review_eligible": 1,
        "frozen": 2,
    }
    assignments = pd.read_parquet(
        tmp_path / "work/provenance/cluster_assignments.parquet"
    )
    assert set(assignments.namespace) == {
        "direct_skin_reaction/global_context",
        "direct_skin_reaction/global_severity_grade",
    }


def test_two_complete_retains_need_no_adjudication(monkeypatch):
    assignment_id = "a" * 64
    assignments = pd.DataFrame(
        [
            {
                "assignment_id": assignment_id,
                "candidate_label": "patch test",
                "review_eligible": True,
            }
        ]
    )
    packet = {
        "packet_version": reconciliation.PACKET_VERSION,
        "packet_id": "packet_1",
        "namespace": "direct_skin_reaction/global_context",
        "reviewer_roles": {
            "primary": "reviewer_a",
            "checker": "reviewer_b",
            "adjudicator": "reviewer_c",
        },
        "assignment_count": 1,
        "clusters": [
            {
                "cluster_id": "cluster_1",
                "assignments": [{"assignment_id": assignment_id}],
            }
        ],
    }
    packet_sha = hashlib.sha256(reconciliation._json_bytes(packet)).hexdigest()
    common = {
        "review_version": reconciliation.REVIEW_VERSION,
        "packet_id": "packet_1",
        "packet_sha256": packet_sha,
        "complete": True,
        "reviewed_assignment_count": 1,
        "exceptions": [],
    }
    primary = {**common, "role": "primary", "reviewer_id": "reviewer_a"}
    checker = {**common, "role": "checker", "reviewer_id": "reviewer_b"}

    monkeypatch.setattr(
        reconciliation,
        "_load_jsonl",
        lambda path: (
            [primary] if path == "primary" else [checker] if path == "checker" else []
        ),
    )
    final, audit = reconciliation.reconcile_reviews(
        assignments,
        [packet],
        ["primary"],
        ["checker"],
        ["adjudication"],
    )

    assert final.loc[0, "final_label"] == "patch test"
    assert audit["assignments_reviewed_twice"] == 1
    assert audit["routed_to_adjudication"] == 0


def test_merge_species_composes_disjoint_reviewed_v5_deltas(tmp_path):
    base, candidate = _base_and_candidate()
    generic = reconciliation.build_proposal(
        base,
        candidate,
        pd.DataFrame(
            [
                {
                    "namespace": "direct_skin_reaction/global_context",
                    "tuple_key": '["new"]',
                    "final_label": "patch test",
                }
            ]
        ),
    )
    species = json.loads(json.dumps(base))
    species["mapping_version"] = reconciliation.PROPOSAL_VERSION
    species["sources"]["sensitization_aop"]["global_species_context"][
        "mapping"
    ]['["new species","new conditions","new support"]'] = "mouse"
    base_path = tmp_path / "base.json"
    generic_path = tmp_path / "generic.json"
    species_path = tmp_path / "species.json"
    for path, payload in (
        (base_path, base),
        (generic_path, generic),
        (species_path, species),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    generic_manifest_path = tmp_path / "generic_manifest.json"
    species_manifest_path = tmp_path / "species_manifest.json"
    common = {
        "publication_status": "awaiting_human_approval",
        "base_version": reconciliation.BASE_VERSION,
        "proposal_version": reconciliation.PROPOSAL_VERSION,
        "validations": {"reviewed": True},
    }
    generic_manifest_path.write_text(
        json.dumps(
            {
                **common,
                "manifest_version": reconciliation.PROPOSAL_MANIFEST_VERSION,
                "files": {
                    generic_path.name: reconciliation.file_sha256(generic_path)
                },
            }
        ),
        encoding="utf-8",
    )
    species_manifest_path.write_text(
        json.dumps(
            {
                **common,
                "manifest_version": "skin_species_context_incremental_proposal.v1",
                "files": {
                    species_path.name: reconciliation.file_sha256(species_path)
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = reconciliation.merge_species_proposal(
        base_mapping_path=base_path,
        generic_proposal_path=generic_path,
        generic_manifest_path=generic_manifest_path,
        species_proposal_path=species_path,
        species_manifest_path=species_manifest_path,
        output_dir=tmp_path / "merged",
    )

    merged = json.loads(
        (tmp_path / "merged/globally_reconciled_auxiliary_value_mapping.v5.json").read_text()
    )
    assert manifest["counts"] == {
        "generic_delta_tuples": 2,
        "species_delta_tuples": 1,
        "merged_delta_tuples": 3,
        "cleaned_aliases_inheriting_reviewed_v4_label": 0,
    }
    assert merged["sources"]["direct_skin_reaction"]["global_context"][
        "mapping"
    ]['["new"]'] == "patch test"
    assert merged["sources"]["sensitization_aop"]["global_species_context"][
        "mapping"
    ]['["new species","new conditions","new support"]'] == "mouse"
    assert merged["sources"]["direct_skin_reaction"]["global_context"][
        "mapping"
    ]['["old"]'] == "old label"
    assert AuxiliaryMetadataAttacher(
        tmp_path / "merged/globally_reconciled_auxiliary_value_mapping.v5.json"
    ).manifest()["mapping_version"] == reconciliation.PROPOSAL_VERSION
