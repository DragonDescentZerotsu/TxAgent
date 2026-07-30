import json

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_buckets import (
    UNKNOWN_TOKEN,
    materialize_pair_buckets,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments import (
    ASSIGNMENTS_FILENAME,
    REGISTRY_FILENAME,
    build_endpoint_policy_assignments,
)
from tools.chembl_tool.tasks.bioavailability_ma.audit_starling_pair_bucket_distributions import (
    AUDIT_FILENAME,
    audit_distributions,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


def _record(
    number,
    *,
    source="fa",
    endpoint="absorption",
    unit="%",
    status="valid",
    smiles="CCO",
    **canonical_fields,
):
    return {
        "normalized_record_id": f"record-{number}",
        "source_id": source,
        "canonical_endpoint": endpoint,
        "canonical_unit": unit,
        "normalization_validity_status": status,
        "canonical_smiles": smiles,
        "molecule_id": f"molecule-{smiles}",
        **canonical_fields,
    }


def _materialize(records):
    return materialize_pair_buckets(
        records,
        source_required_fields=SOURCE_PAIR_FIELDS,
        contract_version=BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    )


def test_bioavailability_source_field_mapping_uses_only_persisted_canonical_fields():
    assert SOURCE_PAIR_FIELDS == {
        "direct_hf": ("canonical_bioavailability_report_type",),
        "oral_exposure": (
            "canonical_dose_quantity_kind",
            "canonical_dose_basis",
            "canonical_dose_bin",
            "canonical_dose_regimen",
        ),
        "fa": ("canonical_assay_system",),
        "fg": ("canonical_assay_system",),
        "fh": ("canonical_species", "canonical_assay_system"),
    }


def test_materializer_maps_persisted_values_without_recanonicalizing_them():
    records = [
        _record(1, canonical_assay_system="Caco-2|EXACT"),
        _record(2, smiles="CCN", canonical_assay_system="caco_2|exact"),
    ]
    rows, metadata = _materialize(records)
    assert rows[0]["pair_bucket_key"] != rows[1]["pair_bucket_key"]
    assert metadata["stats"]["buckets"] == 2


def test_unknown_fields_match_unknown_and_sources_remain_distinct():
    records = [
        _record(
            1,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            canonical_species=None,
            canonical_assay_system="liver_microsomes|clearance",
        ),
        _record(
            2,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            smiles="CCN",
            canonical_species=None,
            canonical_assay_system="liver_microsomes|clearance",
        ),
        _record(
            3,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            smiles="CCC",
            canonical_species="rat",
            canonical_assay_system="liver_microsomes|clearance",
        ),
    ]
    rows, metadata = _materialize(records)
    fields = json.loads(rows[0]["canonical_pair_fields_json"])
    assert fields["canonical_species"] == UNKNOWN_TOKEN
    assert rows[0]["pair_bucket_key"] == rows[1]["pair_bucket_key"]
    assert rows[0]["pair_bucket_key"] != rows[2]["pair_bucket_key"]
    assert metadata["unknown_field_rates"]["canonical_species"] == pytest.approx(2 / 3)
    assert all(metadata["validations"].values())


def test_policy_key_is_absent_from_readable_bucket_key():
    record = _record(
        1,
        source="oral_exposure",
        endpoint="auc",
        unit="ng·h/mL",
        canonical_dose_quantity_kind="mass",
        canonical_dose_basis="per_kg",
        canonical_dose_bin="log2:3",
        canonical_dose_regimen="single",
    )
    rows, _ = _materialize([record])
    assert json.loads(rows[0]["pair_bucket_key"]) == [
        "oral_exposure",
        "auc",
        "ng·h/mL",
        "mass",
        "per_kg",
        "log2:3",
        "single",
    ]


@pytest.mark.parametrize(
    "status",
    [
        "unresolved_structure",
        "non_scalar_measurement",
        "outside_bounded_percentage_domain",
        "incompatible_canonical_unit",
    ],
)
def test_normalization_validity_excludes_without_sidecar_recomputation(status):
    rows, metadata = _materialize(
        [_record(1, status=status, canonical_assay_system="caco_2|permeability")]
    )
    assert rows[0]["bucket_eligible"] is False
    assert rows[0]["pair_bucket_key"] is None
    assert rows[0]["bucket_exclusion_reason"] == status
    assert metadata["validations"]["ineligible_records_have_no_bucket"] is True


def test_materializer_rejects_unmapped_sources():
    with pytest.raises(ValueError, match="no pair-bucket field mapping"):
        _materialize(
            [_record(1, source="new_source", canonical_assay_system="aqueous")]
        )


def test_standalone_builder_writes_exactly_two_files_without_rewriting_v5(tmp_path):
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            _record(1, canonical_assay_system="caco_2|permeability"),
            _record(
                2,
                smiles="CCN",
                canonical_assay_system="caco_2|permeability",
            ),
        ]
    ).to_parquet(records_path, index=False)
    original_hash = file_sha256(records_path)
    output = tmp_path / "pair_buckets"
    output.mkdir()
    for legacy in (
        "endpoint_pair_registry.json",
        "pair_bucket_audit.json",
        "manifest.json",
    ):
        (output / legacy).write_text("legacy", encoding="utf-8")
    metadata = build_sidecar(records_path=records_path, out_dir=output)
    assert file_sha256(records_path) == original_hash
    assert metadata["stats"] == {
        "input_records": 2,
        "sidecar_records": 2,
        "eligible_records": 2,
        "excluded_records": 0,
        "buckets": 1,
        "pairable_buckets": 1,
        "singleton_bucket_rate": 0.0,
    }
    assert {
        "pair_bucket_records.parquet",
        "pair_bucket_metadata.json",
    } == {path.name for path in output.iterdir()}
    sidecar = pd.read_parquet(output / "pair_bucket_records.parquet")
    assert list(sidecar.columns) == [
        "normalized_record_id",
        "source_id",
        "canonical_endpoint",
        "canonical_unit",
        "canonical_pair_fields_json",
        "pair_bucket_key",
        "bucket_eligible",
        "bucket_exclusion_reason",
    ]


def test_distribution_audit_uses_policy_comparison_space(tmp_path):
    records_path = tmp_path / "records.parquet"
    records = [
        {
            **_record(
                1,
                canonical_assay_system="caco_2|permeability",
            ),
            "finite_scalar_value": 10.0,
        },
        {
            **_record(
                2,
                smiles="CCN",
                canonical_assay_system="caco_2|permeability",
            ),
            "finite_scalar_value": 50.0,
        },
    ]
    pd.DataFrame(records).to_parquet(records_path, index=False)
    sidecar_dir = tmp_path / "sidecar"
    build_sidecar(records_path=records_path, out_dir=sidecar_dir)
    policy_dir = tmp_path / "policies"
    build_endpoint_policy_assignments(
        records_path=records_path,
        out_dir=policy_dir,
    )
    analysis_dir = tmp_path / "analysis"
    summary = audit_distributions(
        records_path=records_path,
        sidecar_path=sidecar_dir / "pair_bucket_records.parquet",
        assignments_path=policy_dir / ASSIGNMENTS_FILENAME,
        registry_path=policy_dir / REGISTRY_FILENAME,
        out_dir=analysis_dir,
    )
    assert summary["stats"]["buckets"] == 1
    assert summary["stats"]["pairable_buckets"] == 1
    assert summary["stats"]["wide_buckets"] == 1
    audit = pd.read_parquet(analysis_dir / AUDIT_FILENAME)
    assert audit.loc[0, "comparison_space_robust_span"] == pytest.approx(36.0)
    assert audit.loc[0, "robust_span_over_far_threshold"] == pytest.approx(1.2)
