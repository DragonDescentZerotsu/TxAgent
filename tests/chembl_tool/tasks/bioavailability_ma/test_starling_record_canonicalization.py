import pytest

from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    canonicalize_assay_system,
    canonicalize_bioavailability_record,
    canonicalize_dose,
    canonicalize_species,
    normalization_validity_status,
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


def test_dose_uses_factor_two_bins_basis_and_regimen():
    left = canonicalize_dose("10 mg/kg single oral dose")
    same_bin = canonicalize_dose("15000 µg/kg single-dose")
    next_bin = canonicalize_dose("20 mg/kg single oral dose")
    absolute = canonicalize_dose("10 mg single oral dose")
    repeated = canonicalize_dose("10 mg/kg once daily")
    assert left["canonical_dose_bin"] == "log2:3"
    assert same_bin["canonical_dose_bin"] == "log2:3"
    assert next_bin["canonical_dose_bin"] == "log2:4"
    assert left["canonical_dose_basis"] == "per_kg"
    assert absolute["canonical_dose_basis"] == "absolute"
    assert repeated["canonical_dose_regimen"] == "repeated"
    assert left["canonical_dose_key"] != repeated["canonical_dose_key"]


@pytest.mark.parametrize(
    "dose",
    [
        "30 mg/kg/day",
        "200 mg/day",
        "30 mg QD",
        "30 mg q.d.",
        "30 mg per day",
        "30 mg/d",
    ],
)
def test_daily_dose_notation_maps_to_repeated_regimen(dose):
    assert canonicalize_dose(dose)["canonical_dose_regimen"] == "repeated"


def test_daily_regimen_is_distinct_from_unspecified_and_day_number_is_not_frequency():
    daily = canonicalize_dose("30 mg/kg/day")
    unspecified = canonicalize_dose("30 mg/kg")
    day_number = canonicalize_dose("30 mg on day 1")
    assert daily["canonical_dose_key"] != unspecified["canonical_dose_key"]
    assert unspecified["canonical_dose_regimen"] == "unspecified"
    assert day_number["canonical_dose_regimen"] == "unspecified"


def test_conflicting_single_and_daily_signals_receive_a_distinct_regimen():
    result = canonicalize_dose("30 mg/day, single oral dose")
    assert result["canonical_dose_regimen"] == "conflicting_single_and_repeated"
    assert result["canonical_dose_key"].endswith(
        "|conflicting_single_and_repeated"
    )


def test_unparsed_dose_is_an_exact_fallback_not_a_shared_unknown():
    result = canonicalize_dose("approximately one tablet")
    assert result["canonical_dose_key"] == "unmapped:approximately_one_tablet"
    assert result["dose_mapping_status"] == "unmapped_exact_fallback"


def test_species_strips_sex_strain_and_model_but_preserves_qualifiers():
    result = canonicalize_species("male Sprague-Dawley rats (humanized)")
    assert result["canonical_species"] == "rat"
    assert result["species_sex"] == "male"
    assert result["species_strain"] == "sprague-dawley"
    assert result["species_model"] == "humanized"


def test_species_composites_are_sorted_and_taxonomically_distinct():
    assert canonicalize_species("rat and human")["canonical_species"] == "human+rat"
    assert canonicalize_species("cynomolgus monkey")["canonical_species"] == (
        "cynomolgus_monkey"
    )
    assert canonicalize_species("rhesus monkey")["canonical_species"] != (
        "cynomolgus_monkey"
    )


def test_assay_system_retains_platform_protocol_and_protected_identifier():
    result = canonicalize_assay_system(
        "MDCK-MDR1 bidirectional transport assay", source_id="fg"
    )
    assert result["canonical_assay_platform"] == "mdck"
    assert result["canonical_assay_protocol"] == "bidirectional_transport"
    assert "mdr1" in result["canonical_assay_modifiers"]
    assert result["assay_system_mapping_status"] == "reviewed_rule"


def test_scientifically_distinct_assays_never_converge():
    cyp2c9 = canonicalize_assay_system(
        "recombinant CYP2C9 inhibition assay", source_id="fh"
    )
    cyp2c19 = canonicalize_assay_system(
        "recombinant CYP2C19 inhibition assay", source_id="fh"
    )
    assert cyp2c9["canonical_assay_system"] != cyp2c19["canonical_assay_system"]


def test_unmapped_assay_uses_exact_fallback():
    result = canonicalize_assay_system("bespoke rotating vial method", source_id="fa")
    assert result["canonical_assay_system"] == (
        "unmapped:bespoke_rotating_vial_method"
    )
    assert result["assay_system_mapping_status"] == "unmapped_exact_fallback"


@pytest.mark.parametrize(
    ("report", "value", "expected_status"),
    [
        ("absolute", 0.0, "valid"),
        ("absolute", 100.0, "valid"),
        (
            "absolute",
            100.1,
            "outside_bounded_percentage_domain",
        ),
        (
            "relative_comparison",
            140.0,
            "valid",
        ),
        (
            "relative_comparison",
            0.0,
            "nonpositive_positive_scalar",
        ),
    ],
)
def test_direct_percentage_policy_is_report_type_aware(
    report, value, expected_status
):
    result = canonicalize_bioavailability_record(
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
    assert "comparison_policy_key" not in result
    assert "assay_transfer_status" not in result


def test_duration_validity_uses_existing_canonical_hours():
    record = _record(
        source="fh",
        endpoint="metabolic_half_life",
        measurement="2",
        unit="h",
        value=2.0,
    )
    assert normalization_validity_status(record) == "valid"


def test_invalid_metric_rows_are_retained_with_a_status_not_a_comparison_value():
    result = canonicalize_bioavailability_record(
        _record(endpoint="solubility", measurement="-1", unit="µM", value=-1.0)
    )
    assert result["normalization_validity_status"] == "nonpositive_positive_scalar"
    assert "comparison_value" not in result
    assert "comparison_unit" not in result
    assert "comparison_domain_valid" not in result
    assert "comparison_domain_reason" not in result


@pytest.mark.parametrize("unit", ["log(cm/s)", "log cm/s", "log10(cm/s)"])
def test_negative_transformed_permeability_is_factually_valid(unit):
    result = canonicalize_bioavailability_record(
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
    result = canonicalize_bioavailability_record(
        _record(
            endpoint="intestinal_effective_permeability",
            measurement=str(value),
            unit="cm/s",
            value=value,
        )
    )
    assert result["normalization_validity_status"] == expected_status
