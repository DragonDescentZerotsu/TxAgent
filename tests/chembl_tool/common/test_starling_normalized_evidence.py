import json

import pytest

from tools.chembl_tool.common.starling.normalized_evidence import (
    EndpointOrthography,
    FamilyAssignment,
    MeasurementPair,
    NormalizedSourceProfile,
    aggregate_molecule_family_records,
    canonicalize_endpoint,
    normalize_measurement_and_unit,
    normalize_source_rows,
    validate_cleaned_normalized_identity,
    validate_measurement_pairs,
)
from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_measurement_text,
    clean_text,
)


def _orthography(source_id: str, endpoint_name: str):
    return EndpointOrthography(
        endpoint_name,
        endpoint_name,
        "unchanged",
        "no_reviewed_correction",
        "test.v1",
    )


def _family(source_id: str, endpoint_name: str):
    return FamilyAssignment(
        "Observed.test",
        "Observed",
        "test",
        "surrogate_proxy",
        "test endpoint",
    )


def test_measurement_and_unit_are_folded_as_one_pair():
    pair = normalize_measurement_and_unit(2.5, "×10^-6 cm/s")
    assert pair.canonical_measurement == "0.0000025"
    assert pair.canonical_unit == "cm/s"


@pytest.mark.parametrize(
    ("measurement", "unit", "expected_measurement", "expected_unit"),
    [
        ("59 hr", "hr", "59", "h"),
        ("176 L/day", "L/day", "176", "L/d"),
        ("18.0 ×10⁻⁶ cm/sec", "cm/sec", "0.000018", "cm/s"),
        ("2.5×10⁻⁶ cm/s", "cm/s", "0.0000025", "cm/s"),
        (
            "175 ± 19 ×10⁻⁶",
            "cm/s",
            "0.000175 ± 0.000019",
            "cm/s",
        ),
        (
            "(7.1 ± 0.7) ×10^-6",
            "cm/s",
            "0.0000071 ± 0.0000007",
            "cm/s",
        ),
        (
            "3.8×10^-5 ± 0.1×10^-5",
            "cm/s",
            "0.000038 ± 0.000001",
            "cm/s",
        ),
    ],
)
def test_source_pair_is_normalized_once(
    measurement, unit, expected_measurement, expected_unit
):
    pair = normalize_measurement_and_unit(measurement, unit)
    assert pair.canonical_measurement == expected_measurement
    assert pair.canonical_unit == expected_unit


def test_clean_and_normalize_stages_use_one_authoritative_source_pair():
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_constant="half_life",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
    )
    result = normalize_source_rows(
        [{"smiles": "CCO", "value": "59 hr", "unit": "hr"}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
    )
    cleaned = result.cleaned_records[0]
    normalized = result.normalized_records[0]
    assert cleaned["measurement_text"] == "59 hr"
    assert cleaned["unit_text"] == "hr"
    assert normalized["canonical_measurement"] == "59"
    assert normalized["canonical_unit"] == "h"
    assert normalized["finite_scalar_value"] == 59.0
    assert validate_measurement_pairs(result.normalized_records) == []


@pytest.mark.parametrize(
    "measurement",
    [
        "2.5×106",
        "2.5×10⁇",
        "3.8×10^-5 ± 0.1×10^-6",
        "2.5×10^-6",
    ],
)
def test_ambiguous_measurement_side_notation_is_not_promoted(measurement):
    unit = "×10^-6 cm/s" if measurement == "2.5×10^-6" else "cm/s"
    pair = normalize_measurement_and_unit(measurement, unit)
    assert pair.status == "ambiguous_scientific_notation"
    assert pair.unit_notation_status == "ambiguous_scientific_notation"


@pytest.mark.parametrize(
    "measurement",
    ["<2 ×10^-6", "5–12 ×10^-5", "Papp 2.5 ×10^-6; ratio 3.1"],
)
def test_non_atomic_measurement_side_notation_remains_non_scalar(measurement):
    pair = normalize_measurement_and_unit(measurement, "cm/s")
    assert pair.status == "cleaned_only_non_scalar"


@pytest.mark.parametrize(
    "value",
    ["AUC0-t", "AUC0_t", "AUC0 t", "AUC0–t", "AUC0—t"],
)
def test_endpoint_separator_canonicalization_is_mechanical(value):
    assert canonicalize_endpoint(value) == "auc0_t"


def test_endpoint_canonicalization_preserves_meaningful_symbols():
    assert canonicalize_endpoint("AUCINF/Dose") == "aucinf/dose"
    assert canonicalize_endpoint("AUC0_∞") == "auc0_∞"
    assert canonicalize_endpoint("AUC0_τ") == "auc0_τ"

    pair = normalize_measurement_and_unit("5 ± 1", "×10^-6 cm/s")
    assert pair.canonical_measurement == "0.000005 ± 0.000001"
    assert pair.canonical_unit == "cm/s"


def test_measurement_context_and_qualitative_text_remain_lossless():
    pair = normalize_measurement_and_unit("57% (gmean) [90% CI 49, 68]", "%")
    assert pair.canonical_measurement == "57 (gmean) [90% CI 49, 68]"
    assert pair.canonical_unit == "%"

    pair = normalize_measurement_and_unit("5 ± 1% of dose", "%")
    assert pair.canonical_measurement == "5 ± 1 of dose"
    assert pair.canonical_unit == "%"
    assert pair.status == "cleaned_pair_with_context"

    pair = normalize_measurement_and_unit("high", "wibbles/mL")
    assert pair.canonical_measurement == "high"
    assert pair.canonical_unit == "wibbles/mL"


@pytest.mark.parametrize(
    "measurement",
    [
        "Papp A-B 4.89 ×10⁻⁶ cm/s; efflux ratio = 3.1",
        "AUC ratio 2.1; Cmax ratio 1.4",
        "Vmax = 12 pmol/min; Km = 4 µM",
        "mean 5.2; median 4.8; SD 1.1",
        "[1.2, 2.4, 3.8] µM",
        "pH 6.5: 2.1; pH 7.4: 4.9",
    ],
)
def test_compound_measurements_remain_one_intact_non_scalar_record(measurement):
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_constant="compound_endpoint",
        measurement_field="value",
        embedded_unit=True,
        structure_mode="direct",
    )
    result = normalize_source_rows(
        [{"smiles": "CCO", "value": measurement}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
    )

    assert len(result.cleaned_records) == 1
    assert len(result.normalized_records) == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert record["measurement_text"] == clean_measurement_text(measurement)
    assert record["canonical_measurement"] == clean_measurement_text(measurement)
    assert record["finite_scalar_value"] is None
    assert record["absolute_and_continuous_value"] is None
    assert record["retrieval_eligible"] is True
    assert validate_cleaned_normalized_identity(
        result.cleaned_records, result.normalized_records
    ) == []


def test_normalization_accepts_qualitative_records_and_deduplicates_within_source():
    rows = [
        {
            "smiles": "CCO",
            "kind": "Cmax",
            "value": "high",
            "unit": " ng / ml ",
            "support": "Qualitative evidence.",
            "context": "human",
            "record": "a",
        },
        {
            "smiles": "CCO",
            "kind": "Cmax",
            "value": "high",
            "unit": "ng/mL",
            "support": "Qualitative evidence.",
            "context": "human",
            "record": "b",
        },
        {
            "smiles": "CCN",
            "kind": "Cmax",
            "value": "5",
            "unit": "ng/mL",
            "support": "Numeric evidence.",
            "context": "human",
            "record": "c",
        },
    ]
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_field="kind",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="record",
        support_text_field="support",
        context_fields=("context",),
    )

    result = normalize_source_rows(
        rows,
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
        source_sha256="source",
    )

    assert result.stats["n_input_rows"] == 3
    assert result.stats["n_records"] == 2
    assert result.stats["n_duplicate_records_removed"] == 1
    qualitative = next(row for row in result.records if row["canonical_smiles"] == "CCO")
    assert qualitative["canonical_measurement"] == "high"
    assert qualitative["canonical_unit"] == "ng/mL"
    assert qualitative["is_absolute_and_continuous"] is False
    assert qualitative["duplicate_group_size"] == 2
    assert "kind" not in qualitative
    assert "value" not in qualitative
    assert "unit" not in qualitative
    payload = json.loads(qualitative["source_payload_json"])
    assert payload["_source_endpoint"] == "Cmax"
    assert payload["_source_measurement"] == "high"
    assert qualitative["normalized_record_id"] != qualitative["cleaned_record_id"]
    assert not {
        "parent_record_id",
        "expansion_version",
        "expanded_record_id",
        "child_index",
        "expansion_status",
        "component_parse_status",
        "source_span_start",
        "source_span_end",
    } & set(qualitative)
    assert qualitative["endpoint_name"] == "Cmax"
    assert qualitative["spacing_and_spelling_endpoint"] == "Cmax"
    assert qualitative["canonical_endpoint"] == "cmax"
    assert validate_measurement_pairs(result.records) == []


def test_deduplication_preserves_rows_with_different_source_facing_values():
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_field="kind",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="record",
    )
    result = normalize_source_rows(
        [
            {
                "smiles": "CCO",
                "kind": "AUC0-t",
                "value": "5",
                "unit": "ng·h/mL",
                "record": "a",
            },
            {
                "smiles": "OCC",
                "kind": "AUC0_t",
                "value": "5",
                "unit": "h·ng/mL",
                "record": "b",
            },
        ],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
    )
    assert len(result.normalized_records) == 2
    assert len(result.records) == 2
    assert {row["canonical_endpoint"] for row in result.records} == {"auc0_t"}
    assert {row["canonical_unit"] for row in result.records} == {"h·ng/mL"}
    assert {row["endpoint_name"] for row in result.records} == {"AUC0-t", "AUC0_t"}
    assert all(row["duplicate_group_size"] == 1 for row in result.records)


def test_aggregate_retains_all_records_but_hides_internal_scalar_metadata_from_examples():
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_field="kind",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
        record_id_field="record",
    )
    result = normalize_source_rows(
        [
            {"smiles": "CCO", "kind": "AUC", "value": "10", "unit": "ng·h/mL", "record": "1"},
            {"smiles": "CCO", "kind": "Cmax", "value": "2", "unit": "ng/mL", "record": "2"},
        ],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
    )

    rows = aggregate_molecule_family_records(result.records)

    assert len(rows) == 1
    assert len(rows[0]["normalized_records"]) == 2
    examples = rows[0]["minimal_evidence"]["examples"]
    assert {item["endpoint_type"] for item in examples} == {"AUC", "Cmax"}
    assert {item["reported_value"] for item in examples} == {"10", "2"}
    assert {item["reported_units"] for item in examples} == {"ng·h/mL", "ng/mL"}
    assert all("canonical_measurement" not in item for item in examples)
    assert all("is_absolute_and_continuous" not in item for item in examples)


def test_missing_structure_is_retained_but_excluded_from_retrieval():
    profile = NormalizedSourceProfile(
        source_id="test",
        source_name="test/source",
        endpoint_field="kind",
        measurement_field="value",
        unit_field="unit",
        structure_mode="direct",
    )
    result = normalize_source_rows(
        [{"smiles": "", "kind": "Cmax", "value": "high", "unit": "ng/mL"}],
        profile,
        smiles_mapping=None,
        endpoint_normalizer=_orthography,
        family_resolver=_family,
    )
    assert len(result.records) == 1
    assert result.records[0]["retrieval_eligible"] is False
    assert result.records[0]["structure_status"] == "missing_structure"
    assert len(result.rejections) == 1
    assert result.rejections[0]["organization_status"] == "missing_structure"
