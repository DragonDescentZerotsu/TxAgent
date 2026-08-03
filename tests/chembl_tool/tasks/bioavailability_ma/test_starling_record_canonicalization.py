import pytest

from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    enrich_bioavailability_validity,
    normalization_validity_status,
    normalize_bioavailability_report_type,
)


def _record(
    *,
    source="fa",
    endpoint="absorption",
    measurement="50",
    unit="%",
    value=50.0,
    report=None,
):
    return {
        "source_id": source,
        "canonical_endpoint": endpoint,
        "canonical_measurement": measurement,
        "canonical_unit": unit,
        "finite_scalar_value": value,
        "variation_value": None,
        "measurement_unit_status": "cleaned_pair",
        "canonical_smiles": "CCO",
        "structure_status": "resolved",
        "bioavailability_report_type": report,
    }


def test_only_direct_report_type_is_canonicalized():
    assert normalize_bioavailability_report_type("absolute bioavailability") == "absolute"
    assert normalize_bioavailability_report_type("relative") == "relative_comparison"
    result = enrich_bioavailability_validity(_record(report="absolute"))
    assert "canonical_dose_key" not in result
    assert "canonical_assay_system" not in result
    assert "canonical_species" not in result


@pytest.mark.parametrize(
    ("report", "value", "expected_status"),
    [
        ("absolute", 0.0, "valid"),
        ("absolute", 100.0, "valid"),
        ("absolute", 100.1, "outside_bounded_percentage_domain"),
        ("relative_comparison", 140.0, "valid"),
        ("relative_comparison", 0.0, "nonpositive_positive_scalar"),
    ],
)
def test_direct_percentage_policy_is_report_type_aware(report, value, expected_status):
    result = enrich_bioavailability_validity(
        _record(
            source="direct_hf",
            endpoint="oral_bioavailability",
            measurement=str(value),
            unit="%",
            value=value,
            report=report,
        )
    )
    assert result["normalization_validity_status"] == expected_status


def test_duration_validity_uses_existing_canonical_hours():
    assert normalization_validity_status(
        _record(
            source="fh",
            endpoint="metabolic_half_life",
            measurement="2",
            unit="h",
            value=2.0,
        )
    ) == "valid"


def test_invalid_metric_rows_are_retained_with_a_status_not_a_comparison_value():
    result = enrich_bioavailability_validity(
        _record(endpoint="solubility", measurement="-1", unit="µM", value=-1.0)
    )
    assert result["normalization_validity_status"] == "nonpositive_positive_scalar"
    assert "comparison_value" not in result


@pytest.mark.parametrize("unit", ["log(cm/s)", "log cm/s", "log10(cm/s)"])
def test_negative_transformed_permeability_is_factually_valid(unit):
    result = enrich_bioavailability_validity(
        _record(
            endpoint="intestinal_effective_permeability",
            measurement="-3.35",
            unit=unit,
            value=-3.35,
        )
    )
    assert result["normalization_validity_status"] == "valid"


@pytest.mark.parametrize(
    ("value", "expected_status"),
    [
        (0.000005, "valid"),
        (1.0, "valid"),
        (1.000001, "outside_permeability_domain"),
    ],
)
def test_permeability_has_a_reviewed_physical_domain_gate(value, expected_status):
    result = enrich_bioavailability_validity(
        _record(
            endpoint="intestinal_effective_permeability",
            measurement=str(value),
            unit="cm/s",
            value=value,
        )
    )
    assert result["normalization_validity_status"] == expected_status
