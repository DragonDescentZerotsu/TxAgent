import json

import pandas as pd

from tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments import (
    ASSIGNMENTS_FILENAME,
    METADATA_FILENAME,
    REGISTRY_FILENAME,
    build_endpoint_policy_assignments,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_endpoint_policies import (
    POLICY_TEMPLATES,
    assign_endpoint_policies,
    inherited_template_key,
)


def _record(number, *, endpoint="auc", unit="ng·h/mL", validity="valid", **extra):
    return {
        "normalized_record_id": f"record-{number}",
        "source_id": "oral_exposure",
        "canonical_endpoint": endpoint,
        "canonical_unit": unit,
        "canonical_bioavailability_report_type": "__unknown__",
        "normalization_validity_status": validity,
        **extra,
    }


def test_endpoint_specific_keys_inherit_existing_cutoffs():
    assignments, registry, audit = assign_endpoint_policies(
        [_record(1), _record(2, endpoint="cmax", unit="ng/mL")]
    )
    assert assignments[0]["endpoint_policy_key"] == (
        "auc.positive_scalar.log10_fold.v1"
    )
    assert assignments[1]["endpoint_policy_key"] == (
        "cmax.positive_scalar.log10_fold.v1"
    )
    assert assignments[0]["endpoint_policy_key"] != assignments[1]["endpoint_policy_key"]
    for row in assignments:
        policy = registry["endpoint_policies"][row["endpoint_policy_key"]]
        assert policy["transfer_max"] == POLICY_TEMPLATES[
            "positive_scalar.log10_fold.v1"
        ]["transfer_max"]
        assert policy["not_transfer_min"] == POLICY_TEMPLATES[
            "positive_scalar.log10_fold.v1"
        ]["not_transfer_min"]
    assert audit["stats"]["assigned_records"] == 2


def test_legacy_missing_policy_semantics_remain_unassigned():
    records = [
        _record(1, endpoint="bioavailability", unit="fold"),
        _record(2, endpoint="metabolic_half_life", unit="%"),
        _record(3, endpoint="tmax", unit="fold"),
    ]
    assignments, registry, audit = assign_endpoint_policies(records)
    assert {row["policy_assignment_status"] for row in assignments} == {
        "unsupported_assay_transfer_semantics"
    }
    assert all(row["endpoint_policy_key"] is None for row in assignments)
    assert registry["endpoint_policies"] == {}
    assert audit["stats"]["unassigned_records"] == 3


def test_normalization_invalid_record_is_not_assigned():
    assignments, _, _ = assign_endpoint_policies(
        [_record(1, validity="outside_permeability_domain")]
    )
    assert assignments == [
        {
            "normalized_record_id": "record-1",
            "endpoint_policy_key": None,
            "policy_assignment_status": (
                "normalization_invalid:outside_permeability_domain"
            ),
        }
    ]


def test_inherited_selector_preserves_report_aware_direct_behavior():
    absolute = _record(
        1,
        endpoint="oral_bioavailability",
        unit="%",
        source_id="direct_hf",
        canonical_bioavailability_report_type="absolute",
    )
    relative = {
        **absolute,
        "canonical_bioavailability_report_type": "relative_comparison",
    }
    assert inherited_template_key(absolute) == "bounded_percentage.absolute_pp.v1"
    assert inherited_template_key(relative) == "positive_scalar.log10_fold.v1"


def test_standalone_policy_builder_writes_three_hashed_files(tmp_path):
    records_path = tmp_path / "records.parquet"
    pd.DataFrame([_record(1), _record(2, validity="non_scalar_measurement")]).to_parquet(
        records_path, index=False
    )
    out_dir = tmp_path / "endpoint_policies"
    metadata = build_endpoint_policy_assignments(
        records_path=records_path, out_dir=out_dir
    )
    assert {path.name for path in out_dir.iterdir()} == {
        ASSIGNMENTS_FILENAME,
        REGISTRY_FILENAME,
        METADATA_FILENAME,
    }
    assert metadata["stats"]["assigned_records"] == 1
    assert all(metadata["validations"].values())
    registry = json.loads((out_dir / REGISTRY_FILENAME).read_text())
    assert registry["scope"]["pair_labels"] is False
