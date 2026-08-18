from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
    INDEX_SPECS,
    _select_specs,
)
from tools.chembl_tool.tasks.clintox.build_normalized_starling_evidence_library import (
    ARTIFACT_STAGES,
)
from tools.chembl_tool.tasks.clintox.build_starling_downstream_artifacts import (
    get_spec,
)
from tools.chembl_tool.tasks.clintox.experiment_config import get_source_config
from tools.chembl_tool.tasks.clintox.starling_categorical_response import (
    POLICY as CATEGORICAL_POLICY,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import (
    controlled_vocabulary_violations,
)
from tools.chembl_tool.tasks.clintox.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    SOURCE_COLUMNS,
    source_profiles,
)
from tools.chembl_tool.tasks.clintox.starling_measurement_semantics import (
    load_measurement_semantics_policy,
    resolve_measurement_pair,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
)
from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    load_measurement_policy,
)
from tools.chembl_tool.tasks.clintox.starling_policy import (
    POLICY,
    _enrich_record,
    family_assignment,
)
from tools.chembl_tool.tasks.clintox.starling_v7_artifact_store import PROFILE
from tools.chembl_tool.tasks.clintox.starling_source import (
    DIRECT_SOURCE_ID,
    SOURCE_RELEASE,
)


def test_seven_source_contract_is_pinned_and_excludes_global_identifier():
    assert sum(EXPECTED_SOURCE_ROWS.values()) == 4_846_914
    assert set(POLICY.record_contract.sources) == set(EXPECTED_SOURCE_ROWS)
    assert set(SOURCE_COLUMNS) == set(EXPECTED_SOURCE_ROWS)
    assert all("global_identifier" not in columns for columns in SOURCE_COLUMNS.values())
    assert all(profile.structure_mode == "direct" for profile in source_profiles(Path("root")))


def test_controlled_encoders_are_fail_closed():
    positive = CATEGORICAL_POLICY.apply(
        {
            "source_id": DIRECT_SOURCE_ID,
            "measurement_text": "hepatotoxicity",
            "finite_scalar_value": None,
        }
    )
    negative = CATEGORICAL_POLICY.apply(
        {
            "source_id": DIRECT_SOURCE_ID,
            "measurement_text": "toxicity_absent",
            "finite_scalar_value": None,
        }
    )
    assert positive["finite_scalar_value"] == 1.0
    assert negative["finite_scalar_value"] == 0.0
    assert positive["categorical_encoder_id"] == "clintox_human_toxicity_binary.v1"
    assert not CATEGORICAL_POLICY.apply(
        {
            "source_id": DIRECT_SOURCE_ID,
            "measurement_text": "toxicity suspected",
            "finite_scalar_value": None,
        }
    )
    assert controlled_vocabulary_violations(
        {
            "source_id": DIRECT_SOURCE_ID,
            "measurement_text": "toxicity suspected",
        }
    ) == ("measurement_text",)


def test_direct_family_reuses_the_frozen_gold_adapter_scope():
    accepted = {
        "measurement_text": "cardiotoxicity",
        "toxicity_outcome": "grade 3 cardiac toxicity",
        "needs_more_context": False,
        "source_smiles": "CCO",
        "support_text": "The treated patients developed cardiac toxicity.",
        "pmid": "123",
        "source_record_id": "row-1",
    }
    assert family_assignment(
        DIRECT_SOURCE_ID, "human_clinical_toxicity", accepted
    ).group_id == "Direct.human_clinical_toxicity"
    assert family_assignment(
        DIRECT_SOURCE_ID,
        "human_clinical_toxicity",
        {**accepted, "toxicity_outcome": None},
    ) is None
    assert family_assignment(
        "organ_specific_toxicity", "liver injury", {}
    ).group_id == "Mechanism.organ_specific_toxicity"


def test_enrichment_assigns_controlled_and_percent_reference_semantics():
    categorical = _enrich_record(
        {
            "source_id": "cellular_stress",
            "measurement_text": "reduces_or_protects",
            "endpoint_name": "oxidative_stress",
            "canonical_endpoint": "oxidative_stress",
            "canonical_measurement": "reduces_or_protects",
            "canonical_unit": None,
            "finite_scalar_value": None,
            "source_payload_json": "{}",
        }
    )
    assert categorical["finite_scalar_value"] == -1.0
    assert categorical["canonical_reference_scope"] == "not_applicable"
    assert categorical["normalization_validity_status"] == "valid"
    assert categorical["is_absolute_and_continuous"] is False

    percent = _enrich_record(
        {
            "source_id": "general_cytotoxicity",
            "measurement_text": "50",
            "endpoint_name": "viability_percent",
            "canonical_endpoint": "viability_percent",
            "canonical_measurement": "50",
            "canonical_unit": "%",
            "finite_scalar_value": 50.0,
            "source_payload_json": "{}",
        }
    )
    assert percent["canonical_reference_scope"] == "standardized_control_ratio"
    assert percent["canonical_reference_basis"] == "assay_control"
    assert percent["normalization_validity_status"] == "valid"


def test_reviewed_measurement_registry_standardizes_atomic_pairs():
    policy = load_measurement_semantics_policy()
    assert policy.policy_version == "clintox_measurement_semantics.v2"
    assert len(policy.rules) == 32
    assert len(policy.unit_aliases) == 7
    assert policy.source_sha256 == EXPECTED_SOURCE_SHA256

    potency = resolve_measurement_pair(
        {
            "source_id": "general_cytotoxicity",
            "endpoint_name": "IC50",
            "measurement_text": "5",
            "unit_text": "nM",
        },
        "ic50",
        normalize_measurement_and_unit("5", "nM", task="clintox"),
    )
    assert potency.canonical_measurement == "0.000000005"
    assert potency.canonical_unit == "M"

    dose = resolve_measurement_pair(
        {
            "source_id": "nonclinical_in_vivo_toxicity",
            "endpoint_name": "NOAEL",
            "measurement_text": "10",
            "unit_text": "mg/kg bw/day",
        },
        "noael",
        normalize_measurement_and_unit("10", "mg/kg bw/day", task="clintox"),
    )
    assert dose.canonical_measurement == "10"
    assert dose.canonical_unit == "mg/kg/d"

    for unit, endpoint, measurement, expected_unit in (
        ("mg/kg of body weight", "ld50", "143 (81-251)", "mg/kg"),
        ("% (50 h)", "mortality_or_survival", "75", "%"),
        ("% of cells", "cell_death_percent", "45", "%"),
    ):
        source_id = (
            "general_cytotoxicity"
            if endpoint == "cell_death_percent"
            else "nonclinical_in_vivo_toxicity"
        )
        pair = resolve_measurement_pair(
            {
                "source_id": source_id,
                "endpoint_name": endpoint,
                "measurement_text": measurement,
                "unit_text": unit,
            },
            endpoint,
            normalize_measurement_and_unit(measurement, unit, task="clintox"),
        )
        assert pair.canonical_unit == expected_unit


def test_censored_support_points_fail_closed_without_changing_source_fields():
    source = {
        "source_id": "general_cytotoxicity",
        "endpoint_name": "IC50",
        "canonical_endpoint": "ic50",
        "measurement_text": "54",
        "unit_text": "µg/mL",
        "canonical_measurement": "54",
        "canonical_unit": "mg/L",
        "finite_scalar_value": 54.0,
        "support_text": "The IC50 was much greater than 54 µg/mL.",
        "source_payload_json": "{}",
    }
    enriched = _enrich_record(source)
    assert enriched["normalization_validity_status"] == (
        "support_reports_censored_value"
    )
    assert source["measurement_text"] == "54"
    assert source["unit_text"] == "µg/mL"

    exact = _enrich_record(
        {
            **source,
            "support_text": (
                "The IC50 was 54 µg/mL and activity was greater than control."
            ),
        }
    )
    assert exact["normalization_validity_status"] == "valid"


def test_percentage_reference_semantics_do_not_mix_denominators():
    protein_binding = _enrich_record(
        {
            "source_id": "off_target_ddi_exposure",
            "endpoint_name": "plasma protein binding",
            "canonical_endpoint": "plasma_protein_binding",
            "measurement_text": "95",
            "unit_text": "%",
            "canonical_measurement": "95",
            "canonical_unit": "%",
            "finite_scalar_value": 95.0,
            "evidence_type": "protein_binding_or_partitioning",
            "result_metric": "percent bound",
            "source_payload_json": "{}",
        }
    )
    assert protein_binding["canonical_reference_scope"] == "endpoint_defined_ratio"
    assert protein_binding["canonical_reference_basis"] == "other_explicit"

    bioavailability = _enrich_record(
        {
            "source_id": "off_target_ddi_exposure",
            "endpoint_name": "bioavailability",
            "canonical_endpoint": "bioavailability",
            "measurement_text": "40",
            "unit_text": "%",
            "canonical_measurement": "40",
            "canonical_unit": "%",
            "finite_scalar_value": 40.0,
            "evidence_type": "exposure_or_toxicokinetics",
            "result_metric": "bioavailability",
            "source_payload_json": "{}",
        }
    )
    assert bioavailability["canonical_reference_scope"] == "endpoint_defined_ratio"
    assert bioavailability["canonical_reference_basis"] == "oral_dose"

    generic = _enrich_record(
        {
            "source_id": "nonclinical_in_vivo_toxicity",
            "endpoint_name": "animal_toxicity_finding",
            "canonical_endpoint": "animal_toxicity_finding",
            "measurement_text": "50",
            "unit_text": "% of control",
            "canonical_measurement": "50",
            "canonical_unit": "%·control·of",
            "finite_scalar_value": 50.0,
            "source_payload_json": "{}",
        }
    )
    assert generic["canonical_reference_scope"] == "unknown"
    assert generic["normalization_validity_status"] == (
        "unreviewed_measurement_semantics"
    )


def test_assay_transfer_tail_policy_is_frozen_and_fail_closed():
    policy = load_measurement_policy(POLICY.assay_transfer_measurement_policy)
    assert policy["policy_version"] == "clintox_assay_transfer_measurements.v2"
    assert len(policy["bucket_decisions"]) == 1_437
    assert Counter(
        decision["transform"] for decision in policy["bucket_decisions"].values()
    ) == {"log10": 1_410, "raw": 27}
    assert len(policy["record_ineligibility"]) == 54
    assert Counter(
        item["reason"] for item in policy["record_ineligibility"].values()
    ) == {
        "source_point_outside_reported_interval": 53,
        "implausible_source_literal_concentration_scale": 1,
    }


def test_measurement_unit_manual_audits_are_complete():
    root = (
        Path(__file__).resolve().parents[4]
        / "tools/chembl_tool/tasks/clintox/data_processing/measurement_unit_audit_v2"
    )
    census = json.loads((root / "census.json").read_text(encoding="utf-8"))
    manual_path = root / "manual_audit.tsv"
    censored_path = root / "censored_support_manual_audit.tsv"
    interval_path = root / "source_interval_consistency_manual_audit.tsv"
    with manual_path.open(encoding="utf-8", newline="") as handle:
        manual = list(csv.DictReader(handle, delimiter="\t"))
    with censored_path.open(encoding="utf-8", newline="") as handle:
        censored = list(csv.DictReader(handle, delimiter="\t"))
    with interval_path.open(encoding="utf-8", newline="") as handle:
        intervals = list(csv.DictReader(handle, delimiter="\t"))

    assert len(manual) == 360
    assert {row["manual_review_status"] for row in manual} == {"pass"}
    assert census["manual_review_status"] == "complete"
    assert census["manual_review_passes"] == 360
    assert census["manual_review_failures"] == 0
    assert census["manual_review_tsv_sha256"] == hashlib.sha256(
        manual_path.read_bytes()
    ).hexdigest()

    supplemental = census["supplemental_censored_support_review"]
    assert len(censored) == 72
    assert {row["manual_review_status"] for row in censored} == {"pass"}
    assert supplemental["status"] == "complete"
    assert supplemental["passes"] == 72
    assert supplemental["failures"] == 0
    assert supplemental["tsv_sha256"] == hashlib.sha256(
        censored_path.read_bytes()
    ).hexdigest()

    interval_review = census["supplemental_source_interval_consistency_review"]
    assert len(intervals) == 112
    assert {row["manual_review_status"] for row in intervals} == {"pass"}
    assert interval_review["status"] == "complete"
    assert interval_review["passes"] == 112
    assert interval_review["failures"] == 0
    assert interval_review["new_assay_transfer_exclusions"] == 53
    assert interval_review["already_fail_closed"] == 59
    assert interval_review["tsv_sha256"] == hashlib.sha256(
        interval_path.read_bytes()
    ).hexdigest()


def test_scaffold_only_paper_view_is_registered_without_changing_defaults():
    spec = get_spec()
    assert spec.filter_source_id == DIRECT_SOURCE_ID
    assert spec.benchmark_splits == ("scaffold",)
    assert [item["name"] for item in INDEX_SPECS if item["task_id"] == "clintox"] == [
        "clintox_starling_v7"
    ]
    assert "clintox_starling_v7" not in {
        item["name"] for item in _select_specs([])
    }
    selected = _select_specs(["clintox_starling_v7"])[0]
    assert selected["benchmark_splits"] == ("scaffold",)


def test_public_source_and_artifact_contracts():
    source = get_source_config("starling_v7")
    assert len(source.direct_groups) == 1
    assert len(source.mechanism_groups) == 7
    assert ARTIFACT_STAGES == (
        "01_cleaned",
        "02_canonicalized",
        "03_records",
        "04_pair_buckets",
        "05_distance_calibration",
    )
    assert PROFILE.stages == ARTIFACT_STAGES
    assert POLICY.manifest_versions()["source_release"] == SOURCE_RELEASE
