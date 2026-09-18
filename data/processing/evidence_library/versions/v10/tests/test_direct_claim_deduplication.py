import argparse
import hashlib
import json

import pytest

from data.processing.evidence_library.versions.v10.build_normalized_evidence_library import (
    _attach_stage1_canonical_measurements,
)
from data.processing.evidence_library.versions.v10.stage1_exact_deduplication import (
    POLICY_PATH,
    POLICY_SHA256,
    deduplicate_stage1_exact_records,
    load_policy,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.mapping_registry import (
    mapping_path,
    mapping_registry,
    validate_mapping_hashes,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.starling_source_column_contracts import (
    SOURCE_COLUMNS as BBB_SOURCE_COLUMNS,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.mapping_registry import (
    mapping_path as bbb_mapping_path,
    mapping_registry as bbb_mapping_registry,
    validate_mapping_hashes as validate_bbb_mapping_hashes,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS as ORAL_SOURCE_COLUMNS,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_policy import (
    POLICY as SKIN_POLICY,
    STAGE1_DEDUPLICATION_POLICY as SKIN_DEDUPLICATION_POLICY,
    STAGE1_DEDUPLICATION_POLICY_SHA256 as SKIN_DEDUPLICATION_POLICY_SHA256,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_source_column_contracts import (
    SOURCE_COLUMNS as SKIN_SOURCE_COLUMNS,
)


def _row(
    uid: str,
    *,
    source_id: str,
    value: str,
    unit: str = "%",
    pmid: str = "123",
    **fields,
) -> dict:
    return {
        "cleaned_record_id": f"clean:{uid}",
        "source_row_uid": uid,
        "source_id": source_id,
        "source_row_number": 1,
        "source_record_id": uid,
        "source_sha256": "a" * 64,
        "canonical_smiles": "CCO",
        "pmid": pmid,
        "canonical_measurement_text": value,
        "canonical_unit_text": unit,
        **fields,
    }


def test_stage1_exact_dedup_is_all_source_and_literal():
    records = [
        _row(
            "uid_c", source_id="oral_exposure", value="50",
            exposure_measure="F", statistic_type="mean", oral_dose="10 mg",
            study_context="healthy", comparator_exposure="IV",
            qualifying_conditions=None,
        ),
        _row(
            "uid_a", source_id="hf_bioavailability", value="50",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="20 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        ),
        _row(
            "uid_b", source_id="hf_bioavailability", value="50.0",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="20 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        ),
        _row(
            "uid_d", source_id="fa", value="7", unit="uM",
            endpoint_category="solubility", assay_system="buffer",
            condition_medium="water", biological_context=None,
            formulation_or_solid_form=None, qualifying_conditions="pH 7",
        ),
        _row(
            "uid_e", source_id="fa", value="7", unit="uM",
            endpoint_category="solubility", assay_system="buffer",
            condition_medium="water", biological_context=None,
            formulation_or_solid_form=None, qualifying_conditions="pH 7",
        ),
    ]

    result = deduplicate_stage1_exact_records(
        records, task_id="bioavailability_ma", source_columns=ORAL_SOURCE_COLUMNS
    )

    assert [row["source_row_uid"] for row in result.records] == [
        "uid_a",
        "uid_b",
        "uid_d",
    ]
    assert result.manifest["duplicates_removed"] == 2
    assert result.manifest["duplicate_scope_counts"] == {
        "cross_source": 1,
        "within_source": 1,
    }
    assert {row["retained_source_row_uid"] for row in result.audit_rows} == {
        "uid_a",
        "uid_d",
    }
    assert result.manifest["authoritative_stage1_deduplication"] is True
    assert result.manifest["deduplication_complete"] is True
    assert result.manifest["retained_source_row_uid_count"] == 3
    assert result.manifest["retained_source_row_uid_sha256"] == hashlib.sha256(
        b"uid_a\nuid_b\nuid_d\n"
    ).hexdigest()
    assert result.manifest["duplicate_audit"]["count"] == 2


def test_exact_scientific_fields_preserve_distinct_arms_and_missingness():
    records = [
        _row(
            "uid_a", source_id="hf_bioavailability", value="high",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="10 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        ),
        _row(
            "uid_b", source_id="hf_bioavailability", value="high",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="10 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions="",
        ),
        _row(
            "uid_c", source_id="oral_exposure", value="high",
            exposure_measure="F", statistic_type="mean", oral_dose="10 mg",
            study_context="healthy", comparator_exposure="IV",
            qualifying_conditions=None,
        ),
        _row(
            "uid_d", source_id="oral_exposure", value="high",
            exposure_measure="F", statistic_type="mean", oral_dose="20 mg",
            study_context="healthy", comparator_exposure="IV",
            qualifying_conditions=None,
        ),
    ]

    result = deduplicate_stage1_exact_records(
        records, task_id="bioavailability_ma", source_columns=ORAL_SOURCE_COLUMNS
    )

    assert [row["source_row_uid"] for row in result.records] == [
        "uid_a", "uid_b", "uid_d",
    ]
    assert result.audit_rows[0]["source_row_uid"] == "uid_c"
    assert result.audit_rows[0]["comparison_fields"] == ["qualifying_conditions"]


def test_exact_dedup_prefers_frozen_gold_voter_then_uid(tmp_path):
    records = [
        _row(
            uid, source_id="hf_bioavailability", value="50",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="10 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        )
        for uid in ("uid_a", "uid_z")
    ]
    preferred = tmp_path / "preferred.json"
    preferred.write_text(
        json.dumps(
            {
                "version": "gold_v1_physical_voter_lineage.v1",
                "task": "bioavailability_ma",
                "source_row_uids": ["uid_z"],
                "source_row_uid_sha256": hashlib.sha256(b"uid_z\n").hexdigest(),
            }
        )
    )

    result = deduplicate_stage1_exact_records(
        records,
        argparse.Namespace(stage1_preferred_voter_uids=str(preferred)),
        task_id="bioavailability_ma",
        source_columns=ORAL_SOURCE_COLUMNS,
    )

    assert [row["source_row_uid"] for row in result.records] == ["uid_z"]
    assert result.audit_rows[0]["source_row_uid"] == "uid_a"
    assert result.manifest["validations"][
        "nonprotected_representatives_follow_uid_order"
    ] is True


def test_exact_dedup_retains_all_frozen_gold_voters(tmp_path):
    records = [
        _row(
            uid, source_id="hf_bioavailability", value="50",
            bioavailability_report_type="absolute", species_or_population="human",
            dose="10 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        )
        for uid in ("uid_a", "uid_b", "uid_c")
    ]
    preferred = tmp_path / "preferred.json"
    preferred.write_text(
        json.dumps(
            {
                "version": "gold_v1_physical_voter_lineage.v1",
                "task": "bioavailability_ma",
                "source_row_uids": ["uid_b", "uid_c"],
                "source_row_uid_sha256": hashlib.sha256(
                    b"uid_b\nuid_c\n"
                ).hexdigest(),
            }
        )
    )

    result = deduplicate_stage1_exact_records(
        records,
        argparse.Namespace(stage1_preferred_voter_uids=str(preferred)),
        task_id="bioavailability_ma",
        source_columns=ORAL_SOURCE_COLUMNS,
    )

    assert [row["source_row_uid"] for row in result.records] == ["uid_b", "uid_c"]
    assert result.audit_rows[0]["source_row_uid"] == "uid_a"
    assert result.audit_rows[0]["retained_source_row_uid"] == "uid_b"
    assert result.manifest["validations"]["all_protected_voters_retained"] is True


def test_incomplete_canonical_base_never_deduplicates():
    records = [
        _row(
            uid, source_id="hf_bioavailability", value=None, unit=None,
            bioavailability_report_type="absolute", species_or_population="human",
            dose="10 mg", oral_exposure_mode="oral", comparator="IV",
            qualifying_conditions=None,
        )
        for uid in ("uid_a", "uid_b")
    ]

    result = deduplicate_stage1_exact_records(
        records,
        task_id="bioavailability_ma",
        source_columns=ORAL_SOURCE_COLUMNS,
    )

    assert [row["source_row_uid"] for row in result.records] == ["uid_a", "uid_b"]
    assert result.manifest["duplicates_removed"] == 0


def test_bbb_policy_covers_mechanism_sources_and_is_hash_pinned():
    records = [
        _row(
            "uid_z", source_id="efflux_transport", value="2", unit="ratio",
            transporter_identifier="ABCB1", evidence_type="transport",
            interaction_conclusion="substrate", assay_system="MDCK",
            quantitative_metric="efflux ratio", perturbation=None,
            qualifying_conditions="37 C",
        ),
        _row(
            "uid_a", source_id="efflux_transport", value="2", unit="ratio",
            transporter_identifier="ABCB1", evidence_type="transport",
            interaction_conclusion="substrate", assay_system="MDCK",
            quantitative_metric="efflux ratio", perturbation=None,
            qualifying_conditions="37 C",
        ),
    ]

    result = deduplicate_stage1_exact_records(
        records, task_id="bbb_martins", source_columns=BBB_SOURCE_COLUMNS
    )

    assert [row["source_row_uid"] for row in result.records] == ["uid_a"]
    assert result.audit_rows[0]["retained_source_row_uid"] == "uid_a"
    assert load_policy()["authoritative_stage1_deduplication"] is True
    assert result.manifest["policy_sha256"] == POLICY_SHA256
    assert set(load_policy()["tasks"]["bbb_martins"]["sources"]) == {
        "direct_bbb", "passive_permeability", "efflux_transport", "influx_transport",
    }
    assert set(load_policy()["tasks"]["bioavailability_ma"]["sources"]) == {
        "oral_exposure", "fa", "fg", "fh", "hf_bioavailability",
    }


def test_bbb_and_oral_stage1_policies_use_the_shared_deduplicator():
    from data.processing.evidence_library.versions.v10.tasks.bbb_martins._v7_base_policy import (
        POLICY as bbb_policy,
    )
    from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_policy import (
        POLICY as oral_policy,
    )

    for policy in (bbb_policy, oral_policy):
        assert policy.stage1_canonical_deduplicator.func is (
            deduplicate_stage1_exact_records
        )
        assert POLICY_PATH in policy.scientific_assets


def test_skin_stage1_policy_uses_shared_exact_deduplication():
    assert SKIN_POLICY.stage1_canonical_deduplicator.func is (
        deduplicate_stage1_exact_records
    )
    assert SKIN_DEDUPLICATION_POLICY in SKIN_POLICY.scientific_assets


def test_skin_stage1_dedup_ignores_prose_but_preserves_scientific_arms():
    common = {
        "source_id": "direct_skin_reaction",
        "value": "25",
        "outcome_label": "positive",
        "reaction_type": "sensitization",
        "assay_or_test": "LLNA",
        "species_or_population": "mouse",
        "dose_or_concentration": "10%",
        "positive_count": 1,
        "total_tested": 4,
    }
    records = [
        _row("uid_b", support_text="first wording", extraction_id="x", **common),
        _row("uid_a", support_text="second wording", extraction_id="y", **common),
        _row(
            "uid_c",
            support_text="different arm",
            extraction_id="z",
            **{**common, "dose_or_concentration": "20%"},
        ),
    ]

    result = deduplicate_stage1_exact_records(
        records,
        task_id="skin_reaction",
        source_columns=SKIN_SOURCE_COLUMNS,
        task_policy_path=SKIN_DEDUPLICATION_POLICY,
        task_policy_sha256=SKIN_DEDUPLICATION_POLICY_SHA256,
    )

    assert [row["source_row_uid"] for row in result.records] == ["uid_a", "uid_c"]
    assert result.audit_rows[0]["source_row_uid"] == "uid_b"
    assert result.audit_rows[0]["retained_source_row_uid"] == "uid_a"
    assert result.manifest["version"] == "skin_reaction_stage1_exact_deduplication.v1"


def test_skin_stage1_dedup_never_collapses_an_incomplete_base():
    rows = [
        _row(
            uid,
            source_id="direct_skin_reaction",
            value="25",
            unit="",
            outcome_label="positive",
            reaction_type="sensitization",
            assay_or_test="LLNA",
            species_or_population="mouse",
            dose_or_concentration="10%",
            positive_count=1,
            total_tested=4,
        )
        for uid in ("uid_b", "uid_a")
    ]

    result = deduplicate_stage1_exact_records(
        rows,
        task_id="skin_reaction",
        source_columns=SKIN_SOURCE_COLUMNS,
        task_policy_path=SKIN_DEDUPLICATION_POLICY,
        task_policy_sha256=SKIN_DEDUPLICATION_POLICY_SHA256,
    )

    assert len(result.records) == 2
    assert not result.audit_rows


def test_skin_stage1_dedup_requires_a_shared_cross_source_scientific_field():
    rows = [
        _row(
            "uid_a",
            source_id="direct_skin_reaction",
            value="5",
            outcome_label="positive",
            reaction_type="sensitization",
            assay_or_test="patch test",
            species_or_population="population A",
            dose_or_concentration="5%",
            positive_count=None,
            total_tested=200,
        ),
        _row(
            "uid_b",
            source_id="sensitization_aop",
            value="5",
            assay_type="patch test",
            aop_event="adverse outcome",
            endpoint_or_target="sensitization",
            result_label="positive",
            experimental_conditions="population B; 1% concentration",
            qualifying_conditions=None,
        ),
    ]

    result = deduplicate_stage1_exact_records(
        rows,
        task_id="skin_reaction",
        source_columns=SKIN_SOURCE_COLUMNS,
        task_policy_path=SKIN_DEDUPLICATION_POLICY,
        task_policy_sha256=SKIN_DEDUPLICATION_POLICY_SHA256,
    )

    assert len(result.records) == 2
    assert not result.audit_rows


def test_stage1_dedup_fails_on_an_unclassified_source_column():
    changed = dict(ORAL_SOURCE_COLUMNS)
    changed["hf_bioavailability"] = (
        *changed["hf_bioavailability"],
        "new_scientific_field",
    )
    with pytest.raises(ValueError, match="Unclassified Stage-1 source fields"):
        deduplicate_stage1_exact_records(
            [], task_id="bioavailability_ma", source_columns=changed
        )


def test_stage1_publishes_only_exact_mapped_canonical_measurements():
    records = [
        {
            "measurement_unit_mapping_status": "mapped",
            "resolved_measurement_text": "50",
            "resolved_unit_text": "%",
        },
        {
            "measurement_unit_mapping_status": "excluded",
            "resolved_measurement_text": None,
            "resolved_unit_text": None,
        },
    ]

    _attach_stage1_canonical_measurements(records)

    assert records[0]["canonical_measurement_text"] == "50"
    assert records[0]["canonical_unit_text"] == "%"
    assert records[1]["canonical_measurement_text"] is None
    assert records[1]["canonical_unit_text"] is None


def test_mapping_registry_is_complete_and_hash_pinned():
    registry = mapping_registry()

    assert set(registry["mappings"]) == {
        "source_smiles",
        "reviewed_source_repairs",
        "reviewed_source_drops",
        "reviewed_name_smiles_conflicts",
        "measurement_resolution",
        "exact_measurement_units",
        "endpoint_concepts",
        "auxiliary_context",
        "reference_semantics",
    }
    assert mapping_path("measurement_resolution").is_file()
    assert mapping_path("endpoint_concepts", "fg").is_file()
    assert registry["mappings"]["measurement_resolution"]["depends_on_fields"] == [
        "canonical_endpoint_name"
    ]
    assert registry["mappings"]["endpoint_concepts"]["depends_on_fields"] == [
        "canonical_endpoint_name"
    ]
    assert registry["mappings"]["endpoint_concepts"]["generation_method"] == (
        "legacy_exhaustive_hand_review"
    )
    assert registry["mappings"]["endpoint_concepts"]["provenance_status"] == (
        "grandfathered_without_two_stage_receipts"
    )
    assert registry["mappings"]["auxiliary_context"]["outputs_by_source"][
        "oral_exposure"
    ] == ["canonical_species_context", "canonical_biological_matrix"]
    validate_mapping_hashes()


def test_bbb_mapping_registry_selects_complete_v10_measurement_map():
    registry = bbb_mapping_registry()

    assert set(registry["mappings"]) == {
        "reviewed_source_repairs",
        "reviewed_source_drops",
        "reviewed_name_smiles_conflicts",
        "direct_endpoint_names",
        "measurement_resolution",
        "exact_measurement_units",
        "endpoint_concepts",
        "auxiliary_context",
        "reference_semantics",
        "missing_endpoint_recovery",
    }
    assert registry["mappings"]["measurement_resolution"]["sha256"] == (
        "220ec428161dc0594635d94cc4118c296e49244b7455caf6f4e515a96f37e2de"
    )
    assert "v10_percentage_delta/normalization" in str(
        bbb_mapping_path("measurement_resolution")
    )
    validate_bbb_mapping_hashes()
