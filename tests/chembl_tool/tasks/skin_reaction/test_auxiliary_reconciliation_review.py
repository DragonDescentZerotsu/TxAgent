import json
from pathlib import Path

import pandas as pd
import pytest

from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing.auxiliary_reconciliation import (
    FROZEN_NAMESPACES,
    REVIEW_NAMESPACES,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing.auxiliary_reconciliation_review import (
    apply_accepted_decisions,
    build_label_catalog,
    build_review_packets,
    consolidate_review,
    load_policy,
    validate_primary_decisions,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.data_processing.reconcile_auxiliary_mapping import (
    _build_parser,
)


def _rows() -> pd.DataFrame:
    values = [
        ("direct_skin_reaction", "global_context", "d1", "MRI", "patch testing"),
        ("direct_skin_reaction", "global_context", "d2", "Patch test", "patch test"),
        ("direct_skin_reaction", "global_species_context", "ds", "rats", "rat"),
        ("sensitization_aop", "global_context", "sc", "DPRA", "dpra"),
        ("sensitization_aop", "global_endpoint_context", "se", "Cys depletion", "cysteine depletion"),
        ("sensitization_aop", "global_endpoint_context", "se", "Lys depletion", "depletion"),
        ("sensitization_aop", "global_species_context", "ss", "human volunteers", "human"),
        ("phototoxicity_irritation_local_damage", "global_context", "pc", "3T3 NRU", "3t3 nru"),
        ("phototoxicity_irritation_local_damage", "global_species_context", "ps", "in vitro cells", "in vitro cells"),
        ("skin_exposure", "global_species_context", "xs", "porcine skin", "pig"),
    ]
    records = []
    for index, (source, output, cluster, raw, label) in enumerate(values):
        records.append(
            {
                "source_id": source,
                "output_field": output,
                "input_column": "input",
                "cluster_id": cluster,
                "item_id": f"v{index:04d}",
                "raw_value": raw,
                "tuple_key": json.dumps([raw], separators=(",", ":")),
                "cluster_local_label": label,
                "provisional_label": label,
                "cache_identity": f"cache-{cluster}",
                "requested_model": "gpt-5.4-mini",
                "response_sha256": f"{sum(cluster.encode()):064x}",
                "review_eligible": True,
            }
        )
    return pd.DataFrame(records)


def _mapping(frame: pd.DataFrame) -> dict:
    sources = {}
    for (source, output), group in frame.groupby(["source_id", "output_field"]):
        sources.setdefault(source, {})[output] = {
            "source_columns": ["input"],
            "mapping": {
                **{row.tuple_key: row.provisional_label for row in group.itertuples()},
                "[null]": None,
            },
        }
    sources["direct_skin_reaction"]["global_severity_grade"] = {
        "source_columns": ["effect_metric"],
        "mapping": {'["++"]': "2", "[null]": None},
    }
    sources.setdefault("skin_exposure", {})["global_context"] = {
        "source_columns": ["study_design"],
        "mapping": {'["in vitro"]': "in vitro", "[null]": None},
    }
    return {
        "mapping_version": "starling_auxiliary.skin_reaction.globally_reconciled.v2",
        "sources": sources,
    }


def _decision(decision_id, namespace, action, inputs, finals, raw_values, **extra):
    return {
        "decision_id": decision_id,
        "namespace": namespace,
        "action": action,
        "input_labels": inputs,
        "final_labels": finals,
        "affected_raw_value_count": len(raw_values),
        "supporting_raw_examples": raw_values,
        "evidence_summary": "Exact raw values were inspected.",
        "rationale": "The disposition preserves every explicit protected distinction.",
        "preserved_dimensions": ["core method or biological concept"],
        "removed_details": [],
        "reviewer_id": "primary",
        "decision_status": "proposed",
        **extra,
    }


def _primary() -> list[dict]:
    return [
        _decision(
            "direct-context-merge",
            "direct_skin_reaction/global_context",
            "merge",
            ["patch test", "patch testing"],
            ["patch test"],
            ["MRI", "Patch test"],
        ),
        _decision("direct-species", "direct_skin_reaction/global_species_context", "retain", ["rat"], ["rat"], ["rats"]),
        _decision("sens-context", "sensitization_aop/global_context", "retain", ["dpra"], ["dpra"], ["DPRA"]),
        _decision(
            "sens-endpoint-split",
            "sensitization_aop/global_endpoint_context",
            "split",
            ["depletion"],
            ["lysine depletion", "other depletion"],
            ["Lys depletion"],
            raw_value_assignments=[
                {"raw_value": "Lys depletion", "final_label": "lysine depletion"}
            ],
        ),
        _decision("sens-endpoint-retain", "sensitization_aop/global_endpoint_context", "retain", ["cysteine depletion"], ["cysteine depletion"], ["Cys depletion"]),
        _decision("sens-species", "sensitization_aop/global_species_context", "retain", ["human"], ["human"], ["human volunteers"]),
        _decision("photo-context", "phototoxicity_irritation_local_damage/global_context", "retain", ["3t3 nru"], ["3t3 nru"], ["3T3 NRU"]),
        _decision("photo-species", "phototoxicity_irritation_local_damage/global_species_context", "retain", ["in vitro cells"], ["in vitro cells"], ["in vitro cells"]),
        _decision("exposure-species", "skin_exposure/global_species_context", "retain", ["pig"], ["pig"], ["porcine skin"]),
    ]


def test_policy_matches_eight_review_and_two_frozen_namespaces():
    policy = load_policy()
    assert set(policy["namespace_policy"]["independent_namespaces"]) == REVIEW_NAMESPACES
    assert set(policy["namespace_policy"]["frozen_passthrough_namespaces"]) == FROZEN_NAMESPACES
    assert "publish" not in _build_parser()._subparsers._group_actions[0].choices


def test_frozen_real_snapshot_records_the_complete_local_pass():
    root = Path(
        "tools/chembl_tool/tasks/skin_reaction/data_processing/auxiliary_reconciliation_v2"
    )
    manifest_path = root / "provenance/provisional_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Skin reconciliation provenance has not been materialized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "frozen_namespaces": 2,
        "llm_assignments": 228_639,
        "local_to_aggregated_deltas": 2_395,
        "output_namespaces": 10,
        "review_assignments": 218_198,
        "review_namespaces": 8,
        "section_clusters": 1_649,
    }
    assert manifest["aggregated_mapping"]["sha256"] == (
        "29ca01c3ae54644d0bf6a8bbdc2079222a69200b911bf8db8ca709a13eb24e4a"
    )
    assert manifest["runtime_mapping"]["exists"] is False
    assert manifest["downstream_artifacts_before"]


def test_packets_cover_each_cluster_and_label(tmp_path):
    assignments = _rows()
    catalog = build_label_catalog(assignments)
    manifest = build_review_packets(
        assignments, output_dir=tmp_path, max_packet_bytes=10_000
    )
    assert len(catalog) == 10
    assert manifest["coverage"] == {
        "namespaces": 8,
        "section_clusters": 9,
        "assignments": 10,
        "packets": 8,
    }
    assert all(manifest["validations"].values())


def test_review_replays_changes_and_preserves_frozen_sections():
    assignments = _rows()
    primary = _primary()
    # The synthetic one-row split is invalid because every declared child must
    # receive evidence; use a rename for the successful replay case.
    primary[3] = _decision(
        "sens-endpoint-rename",
        "sensitization_aop/global_endpoint_context",
        "rename",
        ["depletion"],
        ["lysine depletion"],
        ["Lys depletion"],
    )
    audit = validate_primary_decisions(primary, assignments=assignments)
    assert audit["labels_missing"] == 0
    change_ids = [row["decision_id"] for row in primary if row["action"] != "retain"]
    checks = [
        {
            "decision_id": decision_id,
            "verdict": "agree",
            "checker_id": "checker",
            "rationale": "Exact source support and protected distinctions were checked.",
        }
        for decision_id in change_ids
    ]
    adjudications = [
        {
            "decision_id": decision_id,
            "verdict": "accept",
            "adjudicator_id": "adjudicator",
            "rationale": "The independently checked change follows the policy.",
        }
        for decision_id in change_ids
    ]
    accepted, review_audit = consolidate_review(
        primary, checks, adjudications, assignments=assignments
    )
    baseline = _mapping(assignments)
    proposal, changes, _ = apply_accepted_decisions(
        baseline, assignments, accepted
    )
    assert review_audit["accepted_changes"] == 2
    assert len(changes) == 2
    assert (
        proposal["sources"]["direct_skin_reaction"]["global_severity_grade"]
        == baseline["sources"]["direct_skin_reaction"]["global_severity_grade"]
    )
    assert (
        proposal["sources"]["skin_exposure"]["global_context"]
        == baseline["sources"]["skin_exposure"]["global_context"]
    )


def test_rejects_incomplete_split_invalid_label_and_nonindependent_roles():
    assignments = _rows()
    primary = _primary()
    with pytest.raises(ValueError, match="does not use every final label"):
        validate_primary_decisions(primary, assignments=assignments)

    primary = _primary()
    primary[3] = _decision(
        "bad-label",
        "sensitization_aop/global_endpoint_context",
        "rename",
        ["depletion"],
        ["Unknown"],
        ["Lys depletion"],
    )
    with pytest.raises(ValueError, match="invalid final label"):
        validate_primary_decisions(primary, assignments=assignments)

    primary = _primary()
    primary[3] = _decision(
        "rename",
        "sensitization_aop/global_endpoint_context",
        "rename",
        ["depletion"],
        ["lysine depletion"],
        ["Lys depletion"],
    )
    changes = [row["decision_id"] for row in primary if row["action"] != "retain"]
    checks = [
        {"decision_id": value, "verdict": "agree", "checker_id": "primary", "rationale": "Checked."}
        for value in changes
    ]
    adjudications = [
        {"decision_id": value, "verdict": "accept", "adjudicator_id": "adjudicator", "rationale": "Adjudicated."}
        for value in changes
    ]
    with pytest.raises(ValueError, match="must be distinct"):
        consolidate_review(primary, checks, adjudications, assignments=assignments)
