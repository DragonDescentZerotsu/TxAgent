from __future__ import annotations

from data.processing.evidence_library.shared.v1.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_auxiliary_metadata import (
    AuxiliaryMetadataAttacher,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_measurement_semantics import (
    MEASUREMENT_SEMANTICS_VERSION,
    default_policy,
)


def _record(
    *,
    source_id: str,
    endpoint_name: str,
    frozen_endpoint: str,
    measurement: str,
    unit: str,
    support_text: str = "",
) -> dict:
    pair = normalize_measurement_and_unit(measurement, unit)
    parsed = parse_point_measurement(pair.canonical_measurement)
    return {
        "source_id": source_id,
        "endpoint_name": endpoint_name,
        "global_endpoint_context": frozen_endpoint,
        "canonical_endpoint": endpoint_name,
        "measurement_text": measurement,
        "unit_text": unit,
        "canonical_measurement": pair.canonical_measurement,
        "canonical_unit": pair.canonical_unit,
        "measurement_unit_status": pair.status,
        "unit_notation_status": pair.unit_notation_status,
        "unit_notation_factor": pair.unit_notation_factor,
        "finite_scalar_value": parsed.value,
        "support_text": support_text,
    }


def test_registry_is_pinned_and_fail_closed():
    manifest = default_policy().manifest()
    assert manifest["policy_version"] == MEASUREMENT_SEMANTICS_VERSION
    assert manifest["n_rules"] == 87
    assert manifest["fail_closed"] is True
    assert manifest["counts_are_human_reviewed"] is False


def test_sensitization_endpoint_mapping_uses_cleaned_role_alias():
    attached = AuxiliaryMetadataAttacher().attach(
        {
            "source_id": "sensitization_aop",
            "endpoint_name": "EC3",
            "assay_type": None,
        }
    )
    assert attached["global_endpoint_context"] == "ec3"


def test_ec3_percentage_is_a_threshold_not_an_incidence_percentage():
    result = default_policy().apply(
        _record(
            source_id="sensitization_aop",
            endpoint_name="EC3",
            frozen_endpoint="ec3",
            measurement="2.4",
            unit="%",
        )
    )
    assert result["measurement_semantics_status"] == "approved"
    assert result["canonical_endpoint"] == "llna_ec3"
    assert result["measurement_quantity_kind"] == "threshold_concentration"
    assert result["measurement_numeric_domain"] == "positive"


def test_ec2_7_percentage_remains_distinct_from_ec3():
    result = default_policy().apply(
        _record(
            source_id="sensitization_aop",
            endpoint_name="skin sensitization (EC2.7)",
            frozen_endpoint="ec3",
            measurement="32.02",
            unit="%",
        )
    )
    assert result["measurement_semantics_status"] == "approved"
    assert result["canonical_endpoint"] == "llna_ec2_7"
    assert result["measurement_quantity_kind"] == "threshold_concentration"


def test_kinetic_dpra_log_k_is_not_cysteine_depletion_percentage():
    result = default_policy().apply(
        _record(
            source_id="sensitization_aop",
            endpoint_name="cysteine depletion",
            frozen_endpoint="cysteine depletion",
            measurement="-2.22",
            unit="log(s^-1 M^-1)",
            support_text="Kinetic DPRA log k highest reactivity was -2.22.",
        )
    )
    assert result["measurement_semantics_status"] == "approved"
    assert result["canonical_endpoint"] == "kinetic_dpra_log_k"
    assert result["measurement_quantity_kind"] == "kinetic_dpra_log_rate_constant"


def test_ambiguous_clinical_percentage_is_evidence_only():
    result = default_policy().apply(
        _record(
            source_id="sensitization_aop",
            endpoint_name="clinical sensitization",
            frozen_endpoint="clinical sensitization",
            measurement="0.32",
            unit="%",
            support_text="A 0.32% patch dilution was tested.",
        )
    )
    assert result["measurement_semantics_status"] == "evidence_only"
    assert result["canonical_unit"] is None


def test_explicit_applied_dose_basis_gets_its_own_unit():
    result = default_policy().apply(
        _record(
            source_id="skin_exposure",
            endpoint_name="dermal_absorption",
            frozen_endpoint="dermal_absorption",
            measurement="42.6",
            unit="% of applied dose",
        )
    )
    assert result["measurement_semantics_status"] == "approved"
    assert result["canonical_unit"] == "% applied dose"
    assert result["measurement_semantic_unit_basis"] == "percent_of_applied_dose"


def test_control_relative_percentage_can_exceed_one_hundred():
    result = default_policy().apply(
        _record(
            source_id="skin_exposure",
            endpoint_name="relative_penetration",
            frozen_endpoint="relative_penetration",
            measurement="145",
            unit="% of control",
        )
    )
    assert result["measurement_semantics_status"] == "approved"
    assert result["canonical_unit"] == "% control"
    assert result["measurement_numeric_domain"] == "nonnegative_unbounded"


def test_incompatible_permeability_area_time_is_excluded():
    result = default_policy().apply(
        _record(
            source_id="skin_exposure",
            endpoint_name="permeability_coefficient",
            frozen_endpoint="permeability_coefficient",
            measurement="1.2",
            unit="cm^2/h",
            support_text="The source reports Kp without defining diffusivity.",
        )
    )
    assert result["measurement_semantics_status"] == "evidence_only"
    assert result["measurement_semantics_exclusion_reason"] == (
        "coefficient_area_time_conflict"
    )



def test_micro_sign_unit_policies_resolve_their_canonical_unit():
    """`casefold()` maps µ (U+00B5) onto Greek mu, so µ-bearing policies used to miss.

    `unit_by_policy` is keyed on the micro sign, so `convert_to_µg/cm²` silently derived no
    `canonical_unit_text` at all and the rule degraded to `measurement_class="preserve"`.
    Non-µ policies were unaffected, which is why it went unnoticed.
    """
    derived = {
        rule.rule_id: rule.action.get("canonical_unit_text")
        for rule in default_policy().rules
        if "µ" in str(rule.action.get("canonical_unit_policy") or "")
        and str(rule.action.get("canonical_unit_policy") or "").startswith("convert_to_")
    }
    assert derived, "expected at least one µ-bearing convert_to_ policy in the registry"
    assert all(derived.values()), f"policies still deriving no unit: {derived}"
    assert derived["sensitization.noel_nesil.compound_areic_mass.v1"] == "µg/cm^2"
