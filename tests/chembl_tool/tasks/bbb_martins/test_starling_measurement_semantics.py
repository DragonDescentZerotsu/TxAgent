from __future__ import annotations

from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
)
from tools.chembl_tool.tasks.bbb_martins.starling_measurement_semantics import (
    numeric_domain_status,
    resolve_measurement_semantics,
    resolve_qualified_unit_alias,
    unit_is_compatible,
)
from tools.chembl_tool.tasks.bbb_martins.starling_record_canonicalization import (
    normalization_validity_status,
)


def _record(endpoint: str, value: float, unit: str, **extra):
    return {
        "source_id": "direct_bbb",
        "endpoint_name": endpoint,
        "canonical_endpoint": endpoint,
        "canonical_measurement": str(value),
        "canonical_unit": unit,
        "finite_scalar_value": value,
        "variation_value": None,
        "measurement_unit_status": "cleaned_pair",
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        **extra,
    }


def test_numeric_domains_are_endpoint_specific_not_global_percent_rules():
    bounded = resolve_measurement_semantics(_record("percent_transport", 120, "%"))
    index = resolve_measurement_semantics(_record("brain_uptake_index", 141, "%"))
    signed = resolve_measurement_semantics(
        _record("percent_influx_change", -77, "%")
    )
    assert numeric_domain_status(bounded, 120) == "outside_bounded_0_100_domain"
    assert numeric_domain_status(index, 141) is None
    assert numeric_domain_status(signed, -77) is None


def test_log_and_difference_values_allow_negative_scalars():
    assert normalization_validity_status(_record("logbb", -1.63, "dimensionless")) == "valid"
    assert (
        normalization_validity_status(
            _record("brain_concentration_difference", -2.0, "ng/mL")
        )
        == "valid"
    )


def test_ratio_and_permeability_units_are_not_interchangeable():
    ratio = resolve_measurement_semantics(_record("brain_to_plasma_ratio", 1.2, "ratio"))
    permeability = resolve_measurement_semantics(
        _record("papp", 1e-6, "cm/s")
    )
    assert unit_is_compatible(ratio, "ratio") is True
    assert unit_is_compatible(ratio, "nm/s") is False
    assert unit_is_compatible(permeability, "cm/s") is True
    assert unit_is_compatible(permeability, "ratio") is False


def test_ratio_suffix_gets_a_conservative_dimensionless_default():
    semantics = resolve_measurement_semantics(
        _record("csf_to_blood_auc_ratio", 0.3, "ratio")
    )
    assert semantics.rule_id == "bbb.ratio_suffix.v1"
    assert semantics.default_unit == "ratio"


def test_influx_clearance_and_tissue_auc_dimensions_are_supported():
    influx = resolve_measurement_semantics(_record("k1", 0.2, "mL/g/min"))
    auc = resolve_measurement_semantics(_record("brain_auc", 5.0, "h·ng/g"))
    assert unit_is_compatible(influx, "mL/g/min") is True
    assert unit_is_compatible(influx, "nmol/g/min") is False
    assert unit_is_compatible(auc, "h·ng/g") is True
    assert unit_is_compatible(auc, "h·µM") is True


def test_tissue_concentration_accepts_mass_and_whole_tissue_bases():
    semantics = resolve_measurement_semantics(
        _record("brain_concentration", 3.0, "ng/mg")
    )
    assert unit_is_compatible(semantics, "ng/mg") is True
    assert unit_is_compatible(semantics, "µg/brain") is True
    assert unit_is_compatible(semantics, "%") is False


def test_raw_bbb_endpoint_is_conservatively_evidence_only():
    record = _record("brain_to_blood_ratio", -1.63, "ratio", endpoint_name="BBB")
    semantics = resolve_measurement_semantics(record)
    assert semantics.status == "evidence_only"
    assert normalization_validity_status(record) == "unreviewed_endpoint_semantics"


def test_parquet_nan_scale_id_does_not_turn_scalar_rows_categorical():
    record = _record(
        "unknown_quantitative_endpoint",
        1.0,
        "ratio",
        canonical_measurement_scale_id=float("nan"),
    )
    semantics = resolve_measurement_semantics(record)
    assert semantics.status == "evidence_only"
    assert semantics.quantity_kind == "unreviewed"


def test_reviewed_qualified_unit_aliases_preserve_the_qualifier():
    record = {
        "source_id": "direct_bbb",
        "unit_text": "% ID/g",
    }
    assert resolve_qualified_unit_alias(record) == (
        "%ID/g",
        "bbb.unit.percent_injected_dose_per_g.v1",
    )
    pair = normalize_measurement_and_unit("4", "%ID/g")
    assert pair.canonical_unit == "%ID/g"


def test_a_bare_log_unit_lands_on_the_same_axis_as_the_rest_of_logbb():
    """The extraction spells logBB's unit `log10`; logBB rows are dimensionless.

    Pair buckets key on the canonical unit string, so admitting `log10` as a
    second compatible unit would split one endpoint across two axes. The alias
    normalizes the spelling instead.
    """
    for spelling in ("log10", "logBB", "log BB", "log10(Brain/Blood)"):
        alias = resolve_qualified_unit_alias(
            {"source_id": "direct_bbb", "unit_text": spelling}
        )
        assert alias == ("dimensionless", "bbb.unit.dimensionless_log_ratio.v1")
    semantics = resolve_measurement_semantics(
        {"source_id": "direct_bbb", "endpoint_name": "logbb", "canonical_endpoint": "logbb"}
    )
    assert unit_is_compatible(semantics, "dimensionless") is True
    # A log concentration is not dimensionless and must stay out.
    assert unit_is_compatible(semantics, "log10(mol/L)") is not True
