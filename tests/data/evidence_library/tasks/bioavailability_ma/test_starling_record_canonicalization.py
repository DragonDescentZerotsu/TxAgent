import pytest

from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_record_canonicalization import (
    canonical_oral_dose,
    enrich_bioavailability_validity,
    normalization_validity_status,
    normalize_bioavailability_report_type,
)


@pytest.mark.parametrize(
    ("raw", "key", "value"),
    [
        ("10 mg/kg oral", "mass|per_kg|mg|10", 10.0),
        ("200 µg oral misoprostol", "mass|absolute|mg|0.2", 0.2),
        ("26.6 µmol kg−1", "molar|per_kg|µmol|26.6", 26.6),
    ],
)
def test_oral_dose_normalization_uses_exact_quantity_unit_and_basis(raw, key, value):
    result = canonical_oral_dose(raw)
    assert result["canonical_oral_dose_mapping_status"] == "resolved"
    assert result["canonical_oral_dose_key"] == key
    assert result["canonical_oral_dose_value"] == value


@pytest.mark.parametrize("raw", [None, "10 mg/mL", "800 mg PO and 400 mg IV"])
def test_oral_dose_normalization_fails_closed(raw):
    result = canonical_oral_dose(raw)
    assert result["canonical_oral_dose_key"] == "__unknown__"
    assert result["canonical_oral_dose_mapping_status"] != "resolved"


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


def test_hf_evidence_scope_is_row_level_and_does_not_relabel_other_sources():
    direct = enrich_bioavailability_validity(
        _record(source="hf_bioavailability", report="unspecified")
    )
    nondirect = enrich_bioavailability_validity(
        _record(source="hf_bioavailability", report="relative_comparison")
    )
    other = enrich_bioavailability_validity(_record(source="fa", report=None))
    assert direct["canonical_bioavailability_report_type"] == "unspecified"
    assert direct["canonical_bioavailability_evidence_scope"] == "direct"
    assert nondirect["canonical_bioavailability_evidence_scope"] == "nondirect"
    assert other["canonical_bioavailability_evidence_scope"] is None


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
            source="hf_bioavailability",
            endpoint="oral_bioavailability",
            measurement=str(value),
            unit="%",
            value=value,
            report=report,
        )
    )
    assert result["normalization_validity_status"] == expected_status
    assert result["canonical_bioavailability_evidence_scope"] == (
        "direct" if report in {"absolute", "unspecified"} else "nondirect"
    )


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


def test_declared_categorical_anchor_uses_encoded_validity_contract():
    record = _record(
        source="fg",
        endpoint="intestinal_efflux",
        measurement="1",
        unit="binary_outcome_class",
        value=1.0,
    )
    record["categorical_encoder_id"] = "fg_substrate_status_binary.v1"
    assert normalization_validity_status(record) == "valid"


def test_encoded_unit_without_encoder_fails_closed():
    record = _record(
        source="fg",
        endpoint="intestinal_efflux",
        measurement="1",
        unit="binary_outcome_class",
        value=1.0,
    )
    assert normalization_validity_status(record) == "encoded_unit_without_encoder"


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
