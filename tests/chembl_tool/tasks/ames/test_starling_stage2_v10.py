from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_exact_unit_mapping as exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_measurement_resolution as measurement_config,
)
from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_policy as ames_policy,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    compile_candidate_resolution as candidate_compiler,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    POLICY as CATEGORICAL_POLICY,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_exact_unit_mapping import (
    DECISION_VERSION,
    compile_exact_unit_mapping,
    write_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_family_assignment import (
    FAMILY_ASSIGNMENTS,
    family_assignment,
    family_assignment_manifest,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_policy import (
    _enrich_record,
    build_hooks,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_record_canonicalization import (
    normalization_validity_status,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_schema import (
    ENDPOINT_PRODUCER_FIELD,
    PAIR_PRODUCER_FIELD,
    RECORD_CONTRACT,
    SOURCE_ENDPOINT_PRODUCER_IDS,
    SOURCE_EXTRACTION_PAIR_PRODUCER_IDS,
    SOURCE_PAIR_PRODUCER_IDS,
    SOURCE_PAIR_VERSION,
    SOURCE_RULE_PAIR_PRODUCER_IDS,
    SOURCES,
)


def _base_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        "canonical_endpoint": "micronucleus_assay",
        "canonical_measurement": "12",
        "canonical_unit": "%",
        "finite_scalar_value": 12.0,
        "measurement_resolution_status": "ok",
        "measurement_unit_mapping_status": "mapped",
        "measurement_numeric_domain": "any",
    }
    record.update(updates)
    return record


def _write_decisions(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    cleaned = tmp_path / "cleaned.parquet"
    resolved = tmp_path / "measurement_resolution.parquet"
    cleaned.write_bytes(b"cleaned fixture")
    resolved.write_bytes(b"resolution fixture")
    payload: dict[str, object] = {
        "version": DECISION_VERSION,
        "task": "ames",
        "inputs": {
            "cleaned_records": {"path": str(cleaned), "sha256": file_sha256(cleaned)},
            "measurement_resolution": {
                "path": str(resolved),
                "sha256": file_sha256(resolved),
            },
        },
        "observed_pairs": [
            {
                "canonical_endpoint": "comet_assay",
                "input_unit": "% tail DNA",
                "origins": ["llm"],
                "rows": 3,
            },
            {
                "canonical_endpoint": "micronucleus_assay",
                "input_unit": "%",
                "origins": ["source_exact"],
                "rows": 8,
            },
        ],
        "decisions": [
            {
                "canonical_endpoints": ["comet_assay"],
                "input_unit": "% tail DNA",
                "action": "exclude",
                "review_basis": "not a comparable physical quantity",
            },
            {
                "canonical_endpoints": ["micronucleus_assay"],
                "input_unit": "%",
                "action": "map",
                "canonical_unit": "%",
                "review_basis": "source notation preserved",
            },
        ],
    }
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_stage2_provenance_targets_only_compiled_v2_mapping(tmp_path, monkeypatch):
    mapping = tmp_path / "measurement_resolution.parquet"
    called = []
    monkeypatch.setattr(
        candidate_compiler,
        "validate_compiled_mapping",
        lambda path: called.append(Path(path)),
    )
    measurement_config.validate_mapping_provenance(mapping)
    assert measurement_config.MAPPING_VERSION == "ames_measurement_resolution.v2"
    assert measurement_config.DEFAULT_MAPPING_PATH.parent.name == (
        "measurement_resolution_v2"
    )
    assert called == [mapping]


def test_stage2_policy_stays_disabled_and_blocks_normalize_without_assets():
    assert not ames_policy.POLICY.measurement_resolution_enabled
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit) as error:
        ames_policy.validate_arguments(
            parser, argparse.Namespace(through_stage="normalize")
        )
    assert error.value.code == 2
    ames_policy.validate_arguments(parser, argparse.Namespace(through_stage="clean"))


def test_enabled_stage2_rejects_an_explicitly_empty_mapping(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(
        ames_policy,
        "POLICY",
        replace(ames_policy.POLICY, measurement_resolution_enabled=True),
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        ames_policy, "_validate_stage2_assets", lambda path: validated.append(path)
    )
    args = argparse.Namespace(
        through_stage="normalize", measurement_resolution_mapping=""
    )
    with pytest.raises(SystemExit) as error:
        ames_policy.validate_arguments(parser, args)
    assert error.value.code == 2
    assert validated == []


def test_enabled_stage2_rejects_partial_measurement_resolution(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(
        ames_policy,
        "POLICY",
        replace(ames_policy.POLICY, measurement_resolution_enabled=True),
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        ames_policy, "_validate_stage2_assets", lambda path: validated.append(path)
    )
    args = argparse.Namespace(
        through_stage="normalize",
        measurement_resolution_mapping="mapping.parquet",
        allow_partial_measurement_resolution=True,
    )
    with pytest.raises(SystemExit) as error:
        ames_policy.validate_arguments(parser, args)
    assert error.value.code == 2
    assert validated == []


def test_enabled_stage2_requires_full_validation(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(
        ames_policy,
        "POLICY",
        replace(ames_policy.POLICY, measurement_resolution_enabled=True),
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        ames_policy, "_validate_stage2_assets", lambda path: validated.append(path)
    )
    mapping = "mapping.parquet"
    strict = argparse.Namespace(
        through_stage="normalize",
        measurement_resolution_mapping=mapping,
        validation_level="strict",
    )
    with pytest.raises(SystemExit) as error:
        ames_policy.validate_arguments(parser, strict)
    assert error.value.code == 2
    assert validated == []

    full = argparse.Namespace(
        through_stage="normalize",
        measurement_resolution_mapping=mapping,
        validation_level="full",
    )
    ames_policy.validate_arguments(parser, full)
    assert validated == [Path(mapping)]


def test_exact_unit_inputs_must_bind_to_selected_mapping(tmp_path, monkeypatch):
    out_dir = tmp_path / "v10"
    cleaned = out_dir / "01_cleaned/records.parquet"
    selected = tmp_path / "selected.parquet"
    other = tmp_path / "other.parquet"
    cleaned.parent.mkdir(parents=True)
    cleaned.write_bytes(b"cleaned")
    selected.write_bytes(b"selected")
    other.write_bytes(b"other")
    monkeypatch.setattr(ames_policy, "DEFAULT_OUT_DIR", str(out_dir))
    decisions = {
        "inputs": {
            "cleaned_records": {
                "path": str(cleaned),
                "sha256": file_sha256(cleaned),
            },
            "measurement_resolution": {
                "path": str(other),
                "sha256": file_sha256(other),
            },
        }
    }
    with pytest.raises(ValueError, match="measurement_resolution lineage mismatch"):
        ames_policy._validate_exact_unit_input_lineage(decisions, selected)


def test_exact_unit_review_lineage_must_be_durable(tmp_path):
    decisions = {
        "review": {
            "packet_manifest_path": str(tmp_path / "packets/manifest.json"),
            "review_path": str(tmp_path / "reviews.jsonl"),
        }
    }
    with pytest.raises(ValueError, match="outside the durable asset root"):
        ames_policy._validate_durable_review_lineage(decisions)


def test_exact_unit_cache_lineage_must_be_durable(tmp_path, monkeypatch):
    root = tmp_path / "durable"
    root.mkdir()
    receipt = root / "reviews.manifest.json"
    receipt.write_text(json.dumps({"cache": {"path": str(tmp_path / "outside")}}))
    monkeypatch.setattr(
        ames_policy, "REVIEWED_UNIT_DECISIONS_PATH", root / "decisions.json"
    )
    decisions = {
        "review": {
            "packet_manifest_path": str(root / "packets/manifest.json"),
            "review_path": str(root / "reviews.jsonl"),
            "review_run_receipt_path": str(receipt),
        }
    }
    with pytest.raises(ValueError, match="cache is outside the durable asset root"):
        ames_policy._validate_durable_review_lineage(decisions)


def test_family_assignments_are_non_ranked_source_partitions():
    expected = {
        "mutagenicity_outcomes": ("Observed", "direct_outcome"),
        "fixed_mutation": ("Observed", "direct_outcome"),
        "premutagenic_damage": ("Mechanism", "mechanistic_factor"),
        "mutagenicity_mechanism": ("Mechanism", "mechanistic_factor"),
    }
    for source_id, (layer, role) in expected.items():
        assignment = family_assignment(source_id, "ignored")
        assert assignment == FAMILY_ASSIGNMENTS[source_id]
        assert (assignment.assay_tier, assignment.evidence_role) == (layer, role)
        assert not assignment.assay_tier.startswith("Tier ")
    assert family_assignment("unknown", "ignored") is None
    assert (
        family_assignment_manifest()["semantics"] == "non_ranked_source_layer_partition"
    )


def test_schema_declares_endpoint_and_atomic_pair_provenance():
    for source_id, profile in SOURCES.items():
        dimensions = {item.output_field: item for item in profile.canonical_dimensions}
        endpoint = dimensions["canonical_endpoint_name"]
        measurement = dimensions["canonical_measurement_text"]
        unit = dimensions["canonical_unit_text"]
        producers = {item.producer_id: item for item in measurement.producers}
        assert endpoint.producer_id_field == ENDPOINT_PRODUCER_FIELD
        assert endpoint.producer_id == SOURCE_ENDPOINT_PRODUCER_IDS[source_id]
        assert measurement.producer_id_field == PAIR_PRODUCER_FIELD
        assert unit.producer_id_field == PAIR_PRODUCER_FIELD
        assert measurement.atomic_group == unit.atomic_group
        source = producers[SOURCE_PAIR_PRODUCER_IDS[source_id]]
        rule = producers[SOURCE_RULE_PAIR_PRODUCER_IDS[source_id]]
        extraction = producers[SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id]]
        assert source.version == SOURCE_PAIR_VERSION
        assert rule.version == (
            f"{MEASUREMENT_ROUTING_VERSION}+{EXACT_UNIT_MAPPING_VERSION}"
        )
        assert extraction.version == (
            f"{measurement_config.MAPPING_VERSION}+{EXACT_UNIT_MAPPING_VERSION}"
        )
        assert (
            not {"canonical_reference_scope", "canonical_reference_basis"}
            & dimensions.keys()
        )


def test_stage2_hooks_select_declared_pair_producers():
    categorical = _enrich_record(
        _base_record(
            source_id="fixed_mutation",
            result_call="positive",
            finite_scalar_value=None,
            canonical_measurement=None,
            canonical_unit=None,
            measurement_resolution_route="reject",
            measurement_resolution_status="not_extracted",
        )
    )
    extracted = _enrich_record(
        _base_record(
            source_id="fixed_mutation",
            result_call=None,
            measurement_resolution_route="extract",
        )
    )
    rejected = _enrich_record(
        _base_record(
            source_id="fixed_mutation",
            result_call=None,
            measurement_resolution_route="reject",
        )
    )
    assert categorical[PAIR_PRODUCER_FIELD] == "ames_fixed_mutation_ordinal.v1"
    assert (
        extracted[PAIR_PRODUCER_FIELD]
        == SOURCE_EXTRACTION_PAIR_PRODUCER_IDS["fixed_mutation"]
    )
    assert rejected[PAIR_PRODUCER_FIELD] == SOURCE_PAIR_PRODUCER_IDS["fixed_mutation"]
    assert (
        categorical[ENDPOINT_PRODUCER_FIELD]
        == SOURCE_ENDPOINT_PRODUCER_IDS["fixed_mutation"]
    )
    assert build_hooks(type("Args", (), {})()).run_state == {}


def test_reject_route_projects_with_source_pair_provenance():
    record = _base_record(
        source_id="mutagenicity_outcomes",
        endpoint_name="bacterial_reverse_mutation",
        measurement_text="equivocal",
        unit_text=None,
        smiles="CCO",
        canonical_endpoint="bacterial_reverse_mutation",
        canonical_measurement="equivocal",
        canonical_unit=None,
        finite_scalar_value=None,
        measurement_resolution_route="reject",
        measurement_resolution_status="not_extracted",
        normalized_record_id="record",
        cleaned_record_id="record",
        source_row_uid="source-row",
        source_payload_json=json.dumps({"category": None}),
    )
    record.update(_enrich_record(record))
    projected = RECORD_CONTRACT.canonical_projection(record)
    assert (
        projected[PAIR_PRODUCER_FIELD] == SOURCE_PAIR_PRODUCER_IDS[record["source_id"]]
    )
    assert projected["canonical_measurement_text"] == "equivocal"
    assert projected["canonical_unit_text"] == "free-text"
    assert projected["canonical_measurement_scale_id"] is None


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"structure_status": "unresolved"}, "unresolved_structure"),
        ({"canonical_endpoint": None}, "missing_canonical_endpoint"),
        ({"measurement_unit_mapping_status": "excluded"}, "exact_measurement_excluded"),
        ({"measurement_resolution_status": "relative"}, "relative_measurement"),
        ({"measurement_resolution_status": "unsure"}, "measurement_resolution_unsure"),
        ({"measurement_resolution_status": "unavailable"}, "non_scalar_measurement"),
        (
            {"measurement_numeric_domain": "positive", "finite_scalar_value": 0},
            "outside_reviewed_numeric_domain",
        ),
        ({"variation_value": -0.1}, "negative_variation"),
        ({}, "valid"),
    ],
)
def test_scalar_validity_statuses_are_fail_closed(updates, expected):
    assert normalization_validity_status(_base_record(**updates)) == expected


def test_existing_categorical_encoder_remains_valid():
    record = _base_record(
        source_id="mutagenicity_outcomes",
        measurement_text="strong_positive",
        canonical_measurement=None,
        canonical_unit=None,
        finite_scalar_value=None,
    )
    record.update(CATEGORICAL_POLICY.apply(record))
    assert normalization_validity_status(record) == "valid"


def test_stage2_validations_use_the_persisted_validity_field():
    row = {
        "source_id": "fixed_mutation",
        "measurement_resolution_route": "accept",
        ENDPOINT_PRODUCER_FIELD: SOURCE_ENDPOINT_PRODUCER_IDS["fixed_mutation"],
        PAIR_PRODUCER_FIELD: SOURCE_RULE_PAIR_PRODUCER_IDS["fixed_mutation"],
        "canonicalization_status": "valid",
    }
    validations = ames_policy._stage2_validations(
        [row], {"policy_version": "test-policy"}
    )
    assert validations["policy_independent_validity_present"]
    row["canonicalization_status"] = None
    validations = ames_policy._stage2_validations(
        [row], {"policy_version": "test-policy"}
    )
    assert not validations["policy_independent_validity_present"]


def test_stage2_validations_require_the_route_specific_pair_producer():
    row = {
        "source_id": "fixed_mutation",
        "measurement_resolution_route": "reject",
        ENDPOINT_PRODUCER_FIELD: SOURCE_ENDPOINT_PRODUCER_IDS["fixed_mutation"],
        PAIR_PRODUCER_FIELD: SOURCE_RULE_PAIR_PRODUCER_IDS["fixed_mutation"],
        "canonicalization_status": "valid",
    }
    validations = ames_policy._stage2_validations(
        [row], {"policy_version": "test-policy"}
    )
    assert not validations["canonical_pair_producer_declared"]

    row[PAIR_PRODUCER_FIELD] = SOURCE_PAIR_PRODUCER_IDS["fixed_mutation"]
    validations = ames_policy._stage2_validations(
        [row], {"policy_version": "test-policy"}
    )
    assert validations["canonical_pair_producer_declared"]


def test_stage_documents_reject_invalid_persisted_rows():
    row = {
        "source_id": "fixed_mutation",
        "measurement_resolution_route": "reject",
        "canonical_endpoint_name": "bacterial_reverse_mutation",
        ENDPOINT_PRODUCER_FIELD: SOURCE_ENDPOINT_PRODUCER_IDS["fixed_mutation"],
        PAIR_PRODUCER_FIELD: SOURCE_PAIR_PRODUCER_IDS["fixed_mutation"],
        "canonicalization_status": None,
    }
    with pytest.raises(ValueError, match="policy_independent_validity_present"):
        ames_policy.stage_documents(
            args=argparse.Namespace(),
            hooks=build_hooks(argparse.Namespace()),
            normalized=[],
            persisted=[row],
            unit_policy_manifest={"policy_version": "test-policy"},
        )


def test_compiler_covers_source_and_llm_units_with_identity_default(tmp_path):
    decisions, _ = _write_decisions(tmp_path)
    compiled = compile_exact_unit_mapping(decisions)
    output = write_exact_unit_mapping(decisions, tmp_path / "mapping.json")
    mapping = load_exact_unit_mapping(output)
    assert not output.with_suffix(".json.tmp").exists()
    assert output.read_text() == json.dumps(compiled, indent=2, sort_keys=True) + "\n"
    assert compiled["ames_v10_contract"]["observed_pair_count"] == 2
    assert mapping[("ames", "micronucleus_assay", "%")]["scale"] == "1"
    assert mapping[("ames", "comet_assay", "% tail DNA")]["action"] == "exclude"


@pytest.mark.parametrize("existing_name", ("mapping.json", "mapping.json.tmp"))
def test_exact_unit_writer_refuses_existing_outputs(tmp_path, existing_name):
    decisions, _ = _write_decisions(tmp_path)
    output = tmp_path / "mapping.json"
    existing = tmp_path / existing_name
    existing.write_bytes(b"immutable\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_exact_unit_mapping(decisions, output)
    assert existing.read_bytes() == b"immutable\n"
    assert output.exists() is (existing == output)


def test_exact_unit_writer_does_not_replace_a_racing_final(tmp_path, monkeypatch):
    decisions, _ = _write_decisions(tmp_path)
    output = tmp_path / "mapping.json"
    original_link = os.link

    def create_final_then_link(source, destination):
        output.write_bytes(b"racing writer\n")
        return original_link(source, destination)

    monkeypatch.setattr(exact_unit_mapping.os, "link", create_final_then_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_exact_unit_mapping(decisions, output)
    assert output.read_bytes() == b"racing writer\n"
    assert not output.with_suffix(".json.tmp").exists()


def test_exact_unit_compiler_normalizes_decision_path(tmp_path, monkeypatch):
    decisions, _ = _write_decisions(tmp_path)
    absolute = compile_exact_unit_mapping(decisions)
    monkeypatch.chdir(tmp_path)
    relative = compile_exact_unit_mapping(Path(decisions.name))
    assert relative == absolute


def test_compiler_rejects_incomplete_or_unreviewed_transform(tmp_path):
    path, payload = _write_decisions(tmp_path)
    payload["decisions"] = payload["decisions"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed observed exact-unit pairs"):
        compile_exact_unit_mapping(path)

    _, payload = _write_decisions(tmp_path)
    payload["decisions"][1]["scale"] = "0.01"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nonidentity exact-unit scale is unreviewed"):
        compile_exact_unit_mapping(path)


def test_loader_requires_a_real_reviewed_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="reviewed Ames exact-unit decisions"):
        compile_exact_unit_mapping(tmp_path / "missing.json")
