from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    CANONICAL_TUPLE_CONTRACT_VERSION,
    assay_transfer_axis_key,
    canonicalize_assay_transfer_base,
    display_measurement_tuple,
    display_scalar_value,
    finalize_assay_transfer_measurement,
    load_measurement_policy,
    validate_final_assay_transfer_measurements,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
)
from tools.chembl_tool.common.starling.normalization.organization import (
    deduplicate_within_source,
)
from tools.chembl_tool.common.starling.pair_buckets import materialize_pair_buckets


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        pair_buckets={"source": SimpleNamespace(additional_dimensions=("context",))}
    )


def test_display_tuple_reverses_only_explicit_canonical_log10() -> None:
    display = display_measurement_tuple(
        {
            "canonical_measurement_text": "-5.3",
            "finite_scalar_value": -5.3,
            "canonical_unit_text": "log10(cm/s)",
            "measurement_kind": "continuous",
        }
    )
    assert display["display_scalar_value"] == pytest.approx(5.011872336e-6)
    assert display["display_unit_text"] == "cm/s"
    assert display["display_transform_id"] == "inverse_log10.v1"
    assert display_scalar_value(-6.0, "inverse_log10.v1") == pytest.approx(1e-6)


def test_continuous_logit_response_has_fraction_display():
    display = display_measurement_tuple(
        {
            "canonical_measurement_text": "0",
            "canonical_unit_text": "logit_response",
            "measurement_kind": "continuous",
            "finite_scalar_value": 0.0,
        }
    )

    assert display == {
        "display_measurement_text": "0.5",
        "display_scalar_value": 0.5,
        "display_unit_text": "fraction",
        "display_transform_id": "inverse_logit.v1",
    }
    assert display_scalar_value(-1.0986122886681098, "inverse_logit.v1") == pytest.approx(
        0.25
    )

    pka = display_measurement_tuple(
        {
            "canonical_measurement_text": "7.4",
            "finite_scalar_value": 7.4,
            "canonical_unit_text": "pKa",
            "measurement_kind": "continuous",
        }
    )
    assert pka["display_scalar_value"] == 7.4
    assert pka["display_unit_text"] == "pKa"


def _rows(*, value: float, unit: str) -> tuple[dict, dict]:
    source = {
        "canonical_record_id": "record-1",
        "source_id": "source",
        "measurement_text": str(value),
        "unit_text": unit,
        "support_text": "source support is unchanged",
        "retrieval_eligible": True,
        "canonical_measurement": str(value),
        "canonical_unit": unit,
        "finite_scalar_value": value,
        "variation_value": None,
    }
    projected = {
        **source,
        "canonical_endpoint_name": "endpoint",
        "canonical_measurement_text": str(value),
        "canonical_unit_text": unit,
        "measurement_kind": "continuous",
        "context": "context",
    }
    return source, projected


def _policy(key: str, transform: str = "raw") -> dict:
    return {
        "policy_version": "test.assay_transfer.v1",
        "record_corrections": {},
        "bucket_decisions": {key: {"transform": transform}},
    }


def _axis_policy(projected: dict, transform: str) -> dict:
    return {
        "policy_version": "test.assay_transfer.v2",
        "record_corrections": {},
        "record_ineligibility": {},
        "bucket_decisions": {},
        "axis_decisions": {
            assay_transfer_axis_key(projected): {"transform": transform}
        },
        "axis_decision_required_sources": ["source"],
    }


def test_u2010_scientific_minus_is_10_to_negative_six():
    pair = normalize_measurement_and_unit("7.73 ± 0.20", "10‐6 cm/sec")

    assert pair.canonical_measurement == "0.00000773 ± 0.0000002"
    assert pair.canonical_unit == "cm/s"
    assert pair.unit_notation_factor == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("unit", "factor", "target"),
    [
        ("%", 0.01, "ratio"),
        ("%/d", 0.01, "ratio/d"),
        ("%/wk", 0.01, "ratio/wk"),
        ("% applied dose", 0.01, "fraction of applied dose"),
        ("%/cm^2·h", 0.01, "fraction/cm^2·h"),
        ("mL/%·m^2·min", 100.0, "mL/fraction·m^2·min"),
    ],
)
def test_percent_base_conversion_is_atomic(unit, factor, target):
    working, projected = _rows(value=12.2, unit=unit)
    key_parts = ["source", "endpoint", target, "context"]
    working, changed = canonicalize_assay_transfer_base(
        working, _policy(json.dumps(key_parts, separators=(",", ":")))
    )
    projected.update(
        {
            "canonical_measurement_text": working["canonical_measurement"],
            "canonical_unit_text": working["canonical_unit"],
            "finite_scalar_value": working["finite_scalar_value"],
            "variation_value": working["variation_value"],
            "assay_transfer_scale_factor": working["assay_transfer_scale_factor"],
            "assay_transfer_scale_status": working["assay_transfer_scale_status"],
        }
    )
    updated, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=_policy(json.dumps(key_parts, separators=(",", ":"))),
    )

    assert changed is True
    assert persisted["finite_scalar_value"] == pytest.approx(12.2 * factor)
    assert persisted["canonical_unit_text"] == target
    assert updated["canonical_unit"] == target
    assert persisted["measurement_text"] == "12.2"
    assert persisted["support_text"] == "source support is unchanged"
    assert persisted["retrieval_eligible"] is True
    assert validate_final_assay_transfer_measurements([persisted]) == []


def test_reviewed_log_transform_is_the_final_canonical_tuple():
    working, projected = _rows(value=1e-6, unit="cm/s")
    working.update(
        {
            "is_absolute_and_continuous": True,
            "absolute_and_continuous_value": 1e-6,
        }
    )
    projected.update(
        {
            "is_absolute_and_continuous": True,
            "absolute_and_continuous_value": 1e-6,
        }
    )
    key = json.dumps(
        ["source", "endpoint", "cm/s", "context"], separators=(",", ":")
    )
    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=_policy(key, "log10"),
    )

    assert persisted["finite_scalar_value"] == -6
    assert persisted["canonical_measurement_text"] == "-6"
    assert persisted["canonical_unit_text"] == "log10(cm/s)"
    assert persisted["assay_transfer_measurement_contract_version"] == (
        CANONICAL_TUPLE_CONTRACT_VERSION
    )
    assert persisted["assay_transfer_pretransform_scalar_value"] == 1e-6
    assert persisted["absolute_and_continuous_value"] == -6
    assert persisted["retrieval_eligible"] is True
    assert validate_final_assay_transfer_measurements([persisted]) == []
    stale = {**persisted, "absolute_and_continuous_value": 1e-6}
    assert validate_final_assay_transfer_measurements([stale]) == [
        "record-1: absolute continuous scalar mismatch"
    ]


def test_v2_axis_policy_applies_across_pair_bucket_contexts():
    working, projected = _rows(value=100.0, unit="ng/mL")
    projected["canonical_reference_scope"] = "absolute"
    policy = _axis_policy(projected, "log10")
    projected["context"] = "a different pair-bucket dimension"

    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=policy,
    )

    assert persisted["finite_scalar_value"] == 2.0
    assert persisted["canonical_unit_text"] == "log10(ng/mL)"
    assert persisted["assay_transfer_policy_key"] == assay_transfer_axis_key(
        {**projected, "context": "irrelevant"}
    )


def test_v2_axis_policy_fails_closed_when_axis_is_unreviewed():
    working, projected = _rows(value=1.0, unit="ng/mL")
    policy = _axis_policy(projected, "log10")
    projected["canonical_endpoint_name"] = "unreviewed"

    with pytest.raises(ValueError, match="lacks an exact policy decision"):
        finalize_assay_transfer_measurement(
            working,
            projected,
            record_contract=_contract(),
            policy=policy,
        )


def test_v2_policy_loader_requires_exact_axis_decisions(tmp_path: Path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "contract_version": "assay_transfer_canonical_measurement.v2",
                "policy_version": "test.v2",
                "bucket_decisions": {},
                "axis_decisions": {},
                "axis_decision_required_sources": ["source"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks axis_decisions"):
        load_measurement_policy(path)


def test_nonpositive_log_axis_is_retained_but_marked_for_stage04_exclusion():
    working, projected = _rows(value=0.0, unit="ng/mL")
    policy = _axis_policy(projected, "log10")
    policy["record_ineligibility"] = {
        "record-1": {"reason": "outside_log10_domain"}
    }

    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=policy,
    )

    assert persisted["finite_scalar_value"] == 0.0
    assert persisted["canonical_unit_text"] == "ng/mL"
    assert persisted["assay_transfer_transform_id"] == "raw.v1"


def test_reviewed_log_transform_does_not_touch_invalid_rows():
    working, projected = _rows(value=0.0, unit="M")
    working["normalization_validity_status"] = "outside_positive_domain"
    projected["canonicalization_status"] = "outside_positive_domain"
    key = json.dumps(
        ["source", "endpoint", "M", "context"], separators=(",", ":")
    )
    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=_policy(key, "log10"),
    )

    assert persisted["finite_scalar_value"] == 0.0
    assert persisted["canonical_unit_text"] == "M"
    assert persisted["assay_transfer_transform_id"] == "raw.v1"


def test_raw_non_scalar_numeric_text_does_not_become_a_scalar():
    working = {
        "canonical_record_id": "record-1",
        "source_id": "source",
        "canonical_measurement": "1.5 times higher",
        "canonical_unit": "ratio",
        "finite_scalar_value": None,
        "variation_value": None,
    }
    projected = {
        **working,
        "canonical_endpoint_name": "endpoint",
        "canonical_measurement_text": "1.5 times higher",
        "canonical_unit_text": "ratio",
        "measurement_kind": "non_scalar",
        "context": "context",
    }
    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=_policy("unused"),
    )

    assert persisted["finite_scalar_value"] is None
    assert validate_final_assay_transfer_measurements([persisted]) == []


def test_raw_continuous_text_is_derived_from_the_final_scalar():
    working, projected = _rows(value=-3.6109179126442243, unit="logit_response")
    projected["canonical_measurement_text"] = "-3.61092"
    working["canonical_measurement"] = "-3.61092"

    _, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=_contract(),
        policy=_policy("unused"),
    )

    assert persisted["canonical_measurement_text"] == "-3.61091791264"
    assert validate_final_assay_transfer_measurements([persisted]) == []


def test_stage04_retains_unit_defect_as_assay_transfer_ineligible():
    records = [
        {
            "canonical_record_id": "record-1",
            "source_id": "source",
            "canonical_endpoint_name": "endpoint",
            "canonical_unit_text": "cm/s",
            "canonicalization_status": "valid",
            "molecule_id": "molecule-1",
            "context": "context",
            "measurement_kind": "continuous",
        }
    ]
    rows, audit = materialize_pair_buckets(
        records,
        source_required_fields={"source": ("context",)},
        assay_transfer_record_ineligibility={
            "record-1": {"reason": "probable_unit_scale_defect"}
        },
    )

    assert len(rows) == 1
    assert rows[0]["assay_transfer_eligible"] is False
    assert rows[0]["assay_transfer_ineligibility_reason"] == "probable_unit_scale_defect"
    assert rows[0]["pair_bucket_key"] is None
    assert audit["stats"]["ineligible_records"] == 1


def test_retrieval_dedup_uses_the_frozen_prebase_pair():
    common = {
        "source_id": "source",
        "source_smiles": "CC",
        "canonical_smiles": "CC",
        "group_id": "group",
        "endpoint_name": "endpoint",
        "canonical_measurement": "0.1",
        "canonical_unit": "ratio",
        "evidence_context_json": "{}",
        "support_text": "same support",
    }
    rows = [
        {
            **common,
            "normalized_record_id": "record-1",
            "source_record_id": "source-1",
            "source_row_number": 1,
            "assay_transfer_prebase_measurement_text": "10",
            "assay_transfer_prebase_unit_text": "%",
        },
        {
            **common,
            "normalized_record_id": "record-2",
            "source_record_id": "source-2",
            "source_row_number": 2,
            "assay_transfer_prebase_measurement_text": "0.1",
            "assay_transfer_prebase_unit_text": "ratio",
        },
    ]

    kept, duplicates = deduplicate_within_source(rows)

    assert len(kept) == 2
    assert duplicates == []
    assert all(row["retrieval_eligible"] for row in kept)


def test_endpoint_identity_gate_applies_only_to_configured_sources() -> None:
    common = {
        "canonical_smiles": "CCO",
        "group_id": "group",
        "endpoint_name": "",
        "canonical_endpoint": "missing_endpoint",
        "canonical_measurement": "reported outcome",
        "canonical_unit": None,
        "evidence_context_json": "{}",
    }
    kept, _ = deduplicate_within_source(
        [
            {
                **common,
                "source_id": "semantic",
                "source_record_id": "semantic-1",
                "source_row_number": 1,
                "normalized_record_id": "semantic-1",
            },
            {
                **common,
                "source_id": "direct",
                "source_record_id": "direct-1",
                "source_row_number": 1,
                "normalized_record_id": "direct-1",
            },
        ],
        endpoint_identity_required_sources=("semantic",),
    )
    by_source = {row["source_id"]: row for row in kept}
    assert by_source["semantic"]["retrieval_eligible"] is False
    assert by_source["semantic"]["organization_status"] == (
        "missing_endpoint_identity"
    )
    assert by_source["direct"]["retrieval_eligible"] is True
