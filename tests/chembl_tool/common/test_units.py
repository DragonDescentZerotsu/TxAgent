import json

import pytest

from tools.chembl_tool.common.units import (
    load_qualifier_vocabulary,
    qualifier_vocabulary_manifest,
    canonical_unit,
    canonicalize_measurement,
    canonicalize_unit,
    canonicalized_unit,
    canonicalized_value,
    clean_unit,
    load_contextual_unit_policy,
    unit_dimension,
    units_compatible,
)


# --------------------------------------------------------------------------- #
# Level 1: clean_unit -- meaning-preserving display cleaning
# --------------------------------------------------------------------------- #

def test_clean_unit_matches_specification_table():
    assert clean_unit("  µg / mL ") == "µg/mL"
    assert clean_unit("μg/ml") == "µg/mL"  # Greek mu -> micro sign, ml -> mL
    assert clean_unit("ng · h / mL") == "ng·h/mL"
    assert clean_unit("×10⁻⁶  cm / s") == "×10^-6 cm/s"
    assert clean_unit(" mL / min / kg ") == "mL/min/kg"
    assert clean_unit("％") == "%"


def test_clean_unit_preserves_molar_case_unlike_lowercasing():
    # The v6.5 cleaner turned µM into "um" (colliding with micrometres); we keep µM.
    assert clean_unit("µM") == "µM"
    assert clean_unit("uM") == "µM"
    assert clean_unit("nM") == "nM"
    # ... while lowercase-m metre forms stay distinct from molar.
    assert clean_unit("µm") == "µm"
    assert clean_unit("um") == "µm"


def test_clean_unit_canonicalizes_time_and_permeability_spellings():
    assert clean_unit("hr") == "h"
    assert clean_unit("hrs") == "h"
    assert clean_unit("cm/sec") == "cm/s"
    assert clean_unit("ng/ml") == "ng/mL"
    assert clean_unit("µg/ml") == "µg/mL"
    assert clean_unit("msec") == "ms"
    assert clean_unit("milliseconds") == "ms"
    assert clean_unit("ng•h/mL") == "ng·h/mL"

    # Unit casing is meaningful: these are not safe millisecond aliases.
    assert canonicalize_unit("mS").unknown_tokens == ("mS",)
    assert canonicalize_unit("MS").unknown_tokens == ("MS",)


def test_long_signed_identifier_suffix_fails_closed_without_overflow():
    for unit in (
        "% of DA-8159 dose",
        "% of dose of DA-8159",
        "% of oral dose of DA-8159",
    ):
        parsed = canonicalize_unit(unit, task="clintox")
        assert parsed.scale == 1.0
        assert parsed.dimension == ()
        assert "Da-8159" in parsed.unknown_tokens

    malformed_power = canonicalize_unit("Da^-8159", task="clintox")
    assert malformed_power.unknown_tokens == ("Da^-8159",)


def test_clintox_qualifiers_do_not_leak_to_other_tasks():
    clintox = canonicalize_unit("µL/min/pmol CYP3A4", task="clintox")
    assert not clintox.unknown_tokens
    assert canonicalize_unit(
        "µL/min/pmol CYP3A4", task="bbb_martins"
    ).unknown_tokens == ("CYP3A4",)


def test_clean_unit_parses_common_composite_notations():
    assert clean_unit("ng per mL") == "ng/mL"
    assert clean_unit("mL per min per kg") == "mL/min/kg"
    assert clean_unit("ng.h/mL") == "ng·h/mL"
    assert clean_unit("mcg/mL") == "µg/mL"
    assert clean_unit("mL-1") == "mL^-1"


def test_negative_exponent_forms_are_dimensionally_equivalent_to_slash_forms():
    assert unit_dimension("ng·mL-1") == unit_dimension("ng/mL")
    assert unit_dimension("cm·s-1") == unit_dimension("cm/s")
    assert not canonicalize_unit("µL·min-1·mg-1").unknown_tokens


def test_clean_unit_treats_nan_and_null_as_empty():
    assert clean_unit("nan") is None
    assert clean_unit("") is None
    assert clean_unit(None) is None
    assert clean_unit("unspecified") is None


# --------------------------------------------------------------------------- #
# Level 2: dimensional canonicalization
# --------------------------------------------------------------------------- #

def test_scientific_notation_variants_share_one_canonical_and_scale():
    forms = [
        "×10⁻⁶ cm/s",
        "x10^-6 cm/s",
        "10^-6 cm/s",
        "10-6 cm/s",
        "10‐6 cm/s",
        "10‑6 cm/s",
        "×10^-6 cm/s",
    ]
    results = [canonicalize_unit(f) for f in forms]
    canonicals = {r.canonical for r in results}
    scales = {r.scale for r in results}
    dims = {r.dimension for r in results}
    assert canonicals == {"cm/s"}
    # Total scale to base SI folds the 1e-6 notation factor with cm's centi prefix.
    assert scales == {1e-8}
    assert dims == {(("length", 1), ("time", -1))}


def test_suffix_scale_notation_folds_without_double_applying_exponent():
    # "cm×10^-6" (scale as a suffix) must fold to a single 1e-6 factor, not 1e-6**-6.
    a = canonicalize_unit("cm×10^-6")
    b = canonicalize_unit("×10^-6 cm")
    assert a.scale == b.scale == 1e-8
    assert a.dimension == b.dimension == (("length", 1),)


def test_ocr_compressed_scientific_notation_is_quarantined_not_folded():
    for form in ["×106 cm/s", "x106 cm/s", "106 cm/s"]:
        result = canonicalize_unit(form)
        assert result.notation_status == "ambiguous_scientific_notation"
        assert result.notation_factor is None
        assert result.unknown_tokens
        assert result.canonical != "cm/s"

    explicit = canonicalize_unit("×10^6 cm/s")
    assert explicit.notation_status == "unambiguous_scientific_notation"
    assert explicit.notation_factor == 1e6
    assert explicit.canonical == "cm/s"


def test_dimensionally_equivalent_factor_orderings_collapse():
    a = canonicalize_unit("ng·h/mL")
    b = canonicalize_unit("h·ng/mL")
    assert a.canonical == b.canonical
    assert a.dimension == b.dimension
    assert a.dimension == (("mass", 1), ("time", 1), ("volume", -1))


def test_composite_clearance_and_concentration_dimensions():
    assert unit_dimension("mL/min") == (("time", -1), ("volume", 1))
    assert unit_dimension("µM") == (("amount", 1), ("volume", -1))
    assert unit_dimension("ng/mL") == (("mass", 1), ("volume", -1))


def test_canonical_unit_convenience_returns_the_display_key():
    assert canonical_unit("ng/ml") == "ng/mL"
    assert canonical_unit("×10⁻⁶ cm/s") == "cm/s"
    assert canonical_unit("h·ng/mL") == "h·ng/mL"
    assert canonical_unit("hr·ng/ml") == "h·ng/mL"  # same key as the form above
    assert canonical_unit("log(cm/s)") == "log(cm/s)"
    assert canonical_unit("nan") is None
    assert canonical_unit(None) is None
    # matches the field on the full result
    assert canonical_unit("µg/cm²/h") == canonicalize_unit("µg/cm²/h").canonical


def test_measurement_class_does_not_override_explicit_reciprocal_exponents():
    value, unit = canonicalize_measurement(
        14.5,
        "h nmol^-1 mL^-1",
        measurement_class="auc_molar",
    )
    assert value is None
    assert unit is None


def test_negative_exponent_moves_to_denominator_in_canonical():
    # cm/s, cm·s^-1 and cm s-1 are the same unit -> one canonical key.
    forms = ["cm/s", "cm·s^-1", "cm s-1", "cm s⁻¹"]
    assert {canonicalize_unit(f).canonical for f in forms} == {"cm/s"}


def test_latex_brace_and_spaced_scientific_notation():
    forms = ["10^-6 cm/s", "10^{-6} cm/s", "10^-6 cm s^-1", "10^{-6} cm s^{-1}", "×10⁻⁶ cm/s"]
    results = [canonicalize_unit(f) for f in forms]
    assert {r.canonical for r in results} == {"cm/s"}
    assert {r.scale for r in results} == {1e-8}


def test_areic_dose_units_resolve_and_collapse():
    # Skin-exposure areic dose: area handled via the cm^2 exponent path.
    assert unit_dimension("µg/cm²") == (("length", -2), ("mass", 1))
    forms = ["µg/cm²/h", "µg cm-2 h-1", "µg cm⁻² h⁻¹", "µg/cm2/h"]
    assert {canonicalize_unit(f).canonical for f in forms} == {"µg/cm^2·h"}


def test_dimensionless_domain_tokens_resolve():
    assert canonicalize_unit("ppm").is_dimensionless
    assert canonicalize_unit("dimensionless").is_dimensionless
    assert clean_unit("unitless") == "dimensionless"
    assert not canonicalize_unit("ppm").unknown_tokens


def test_parenthesised_denominator_groups_match_flat_forms():
    assert canonicalize_unit("µg/(cm²·h)").canonical == canonicalize_unit("µg/cm²/h").canonical
    assert canonicalize_unit("mL/(min·kg)").canonical == canonicalize_unit("mL/min/kg").canonical
    assert canonicalize_unit("µL/(min·mg)").canonical == canonicalize_unit("µL/min/mg").canonical


def test_log_transforms_are_marked_transforms_not_plain_dimensionless():
    for form in ["log(cm/s)", "log10(cm/s)", "ln(ng m⁻²)", "-log(M)", "log(µmol cm⁻² h⁻¹)"]:
        result = canonicalize_unit(form)
        # The magnitude carries no base dimension (can't take the log of a dimensioned value)
        assert result.dimension == (), form
        # ...but it is NOT a plain unit-free ratio: it is a transformed quantity.
        assert not result.is_dimensionless, form
        assert result.transform, form
        assert result.canonical.startswith(("log", "ln", "-log"))


def test_log_transform_is_not_conflated_with_a_plain_ratio_or_other_logs():
    # A log(cm/s) value must not look like a percentage, a fold-change, or a log of a
    # different unit -- distinct canonical keys, and not compatible with a raw endpoint.
    assert canonicalize_unit("log(cm/s)").canonical != canonicalize_unit("%").canonical
    assert canonicalize_unit("log(cm/s)").canonical != canonicalize_unit("log(cm/h)").canonical
    assert units_compatible("log(cm/s)", "fraction_percent") is None
    assert units_compatible("log(cm/s)", "permeability") is None


def test_space_delimited_log_unit_is_the_same_transformed_coordinate():
    parenthesized = canonicalize_unit("log(cm/s)")
    spaced = canonicalize_unit("log cm/s")
    assert spaced.canonical == parenthesized.canonical == "log(cm/s)"
    assert spaced.transform == parenthesized.transform == "log"


def test_bare_log_and_p_prefixed_readouts_are_recognized_as_transforms():
    for form in ["logP", "logD", "logBB", "logPapp"]:
        result = canonicalize_unit(form)
        assert result.transform == "log", form
        assert result.dimension == () and not result.is_dimensionless and not result.unknown_tokens
    assert canonicalize_unit("log10P").transform == "log10"
    for form in ["pIC50", "pEC50", "pEC3", "pKa", "pKi", "pA2", "pD2", "pGI50"]:
        result = canonicalize_unit(form)
        assert result.transform == "-log10", form  # p = -log10 of a molar quantity
        assert result.dimension == () and not result.is_dimensionless and not result.unknown_tokens


def test_si_pico_units_are_not_mistaken_for_p_log_readouts():
    # pM / pmol / pg / pL / pm stay pico-prefixed SI units, not transforms.
    assert canonicalize_unit("pM").dimension == (("amount", 1), ("volume", -1))
    assert canonicalize_unit("pmol").dimension == (("amount", 1),)
    assert canonicalize_unit("pg").dimension == (("mass", 1),)
    for u in ["pM", "pmol", "pg", "pL", "pm", "ppm", "ppb"]:
        assert canonicalize_unit(u).transform == "", u


def test_nested_ratio_precedence_is_respected():
    # ng/mL/(mg/kg) == (ng/mL)/(mg/kg): dividing by (mg/kg) keeps kg in the numerator,
    # which a naive paren-strip to ng/mL/mg/kg would get wrong.
    nested = canonicalize_unit("ng/mL/(mg/kg)")
    assert nested.canonical == canonicalize_unit("(ng/mL)/(mg/kg)").canonical
    assert nested.dimension == (("mass", 1), ("volume", -1))
    # Distinct from the flattened (incorrect) reading.
    assert nested.canonical != canonicalize_unit("ng/mL/mg/kg").canonical


def test_slash_then_dot_defaults_conservatively_to_denominator():
    # /X·Y is endpoint-ambiguous (AUC vs flux vs clearance); the safe default puts the trailing
    # factor in the denominator. ng/mL·h therefore parses like ng/(mL·h), not AUC.
    conservative = canonicalize_unit("ng/mL·h")
    assert conservative.dimension == (("mass", 1), ("time", -1), ("volume", -1))
    # Explicit AUC notation (h in the numerator) is unambiguous and stays AUC.
    assert canonicalize_unit("ng·h/mL").dimension == (("mass", 1), ("time", 1), ("volume", -1))
    # Skin flux and intrinsic clearance are read correctly under the same conservative rule.
    assert canonicalize_unit("µg/cm²·h").dimension == (("length", -2), ("mass", 1), ("time", -1))
    assert canonicalize_unit("µl/min·mg").dimension == (("mass", -1), ("time", -1), ("volume", 1))


def test_measurement_class_override_resolves_the_ambiguous_auc_form():
    # An endpoint-aware caller forces AUC: ng/mL·h is then treated as ng·h/mL.
    value, unit = canonicalize_measurement(5, "ng/mL·h", measurement_class="auc")
    assert unit == "ng·h/mL"
    assert value == pytest.approx(5.0)
    # unit conversion under the forced class still applies prefixes (µg·h/mL -> ng·h/mL).
    value, unit = canonicalize_measurement(5, "µg/mL·h", measurement_class="auc")
    assert unit == "ng·h/mL"
    assert value == pytest.approx(5000.0)


def test_measurement_class_guard_flags_units_incompatible_with_the_class():
    # A unit that cannot belong to the declared class is flagged (None), not coerced.
    assert canonicalize_measurement(5, "%", measurement_class="auc") == (None, None)
    assert canonicalize_measurement(5, "fold", measurement_class="auc") == (None, None)
    assert canonicalize_measurement(5, "ng/mL", measurement_class="auc") == (None, None)  # no time
    assert canonicalize_measurement(5, "wibbles/mL", measurement_class="auc") == (None, None)
    # A compatible unit still resolves and converts.
    assert canonicalize_measurement(5, "µg·h/mL", measurement_class="auc") == (
        pytest.approx(5000.0),
        "ng·h/mL",
    )


def test_intrinsic_clearance_class_round_trips():
    value, unit = canonicalize_measurement(8.3, "µl/min·mg", measurement_class="intrinsic_clearance")
    assert unit == "µL/min/mg protein"
    assert value == pytest.approx(8.3)
    # in-vivo clearance in mL/min/kg is a different class -> stays distinct, not merged
    assert canonicalized_unit("mL/min/kg", measurement_class="weight_normalized_clearance") == "mL/min/kg"


def test_qualifier_after_slash_stays_in_denominator():
    # "mg protein" must keep protein with mg in the denominator despite left-to-right rules.
    result = canonicalize_unit("µL/min/mg protein")
    assert result.dimension == (("mass", -1), ("time", -1), ("volume", 1))
    # protein is preserved on the denominator side of the canonical key
    assert "protein" in result.canonical.split("/", 1)[1]


def test_time_units_carry_real_scales_relative_to_seconds():
    assert canonicalize_unit("h").scale == 3600.0
    assert canonicalize_unit("min").scale == 60.0
    assert canonicalize_unit("s").scale == 1.0
    assert canonicalize_unit("d").scale == 86400.0


def test_canonicalized_unit_is_fold_only_by_default_no_standardization():
    # Default keeps the source's own unit/prefix (no magnitude-shifting standardization).
    assert canonicalized_unit("µM") == "µM"
    assert canonicalized_unit("µg/mL") == "µg/mL"
    assert canonicalized_unit("nm/s") == "nm/s"
    assert canonicalized_unit("cm/h") == "cm/h"
    assert canonicalized_unit("×10⁻⁶ cm/s") == "cm/s"      # ×10^N notation still folded out
    assert canonicalized_unit("%") == "%"
    assert canonicalized_unit("log(cm/s)") == "log(cm/s)"  # transform kept as-is
    assert canonicalized_unit("nan") is None


def test_canonicalized_value_is_fold_only_by_default():
    # No prefix/time conversion by default -> no magnitude shifts.
    assert canonicalized_value(2, "µM") == pytest.approx(2.0)              # µM stays µM
    assert canonicalized_value(60, "min") == pytest.approx(60.0)          # min stays min
    assert canonicalized_value(2.5, "×10⁻⁶ cm/s") == pytest.approx(2.5e-6)  # only fold ×10^N
    assert canonicalized_value(2.0, "log(cm/s)") == 2.0
    assert canonicalized_value("nan", "ng/mL") is None
    assert canonicalized_value("not a number", "ng/mL") is None


def test_opt_in_standardization_via_measurement_class_and_targets():
    # measurement_class (from the endpoint) is the opt-in standardization path.
    assert canonicalize_measurement(2, "µM", measurement_class="potency") == (
        pytest.approx(2000.0),  # µM -> nM
        "nM",
    )
    assert canonicalize_measurement(2, "µM", measurement_class="molar_concentration") == (
        pytest.approx(2.0),  # µM -> µM (kept; sensible target)
        "µM",
    )
    # rate constants standardize to h^-1 (not tiny s^-1 decimals)
    assert canonicalized_unit("min^-1", measurement_class="rate_constant") == "h^-1"
    # explicit per-call target override
    value, unit = canonicalize_measurement(3.0, "×10⁻⁶ cm/s", targets=["cm/h"])
    assert unit == "cm/h"
    assert value == pytest.approx(3e-6 * 3600)


def test_unknown_tokens_are_flagged_not_silently_dropped():
    result = canonicalize_unit("wibbles/mL")
    assert "wibbles" in result.unknown_tokens
    assert not result.is_dimensionless


def test_explicit_decimal_scientific_unit_variants_fold_safely():
    expected = canonicalize_unit("×10^-6 cm/s")
    for value in (
        "1e-6 cm/s",
        "1 × 10^-6 cm/s",
        "10^-6cm/s",
        "10^(-6) cm/s",
    ):
        actual = canonicalize_unit(value)
        assert actual.canonical == "cm/s"
        assert actual.scale == pytest.approx(expected.scale)
        assert actual.unknown_tokens == ()
        assert actual.notation_status == "unambiguous_scientific_notation"

    leading_times_e = canonicalize_unit("x 1e-6 cm/s")
    assert leading_times_e.canonical == "cm/s"
    assert leading_times_e.scale == pytest.approx(1e-8)
    ten_e_minus_six = canonicalize_unit("10E-6 cm/s")
    assert ten_e_minus_six.canonical == "cm/s"
    assert ten_e_minus_six.scale == pytest.approx(expected.scale * 10)


@pytest.mark.parametrize(
    ("source_unit", "expected_value"),
    [
        ("1e-6 cm/s", 3e-6),
        ("2E-6 cm/s", 6e-6),
        ("x 1e-6 cm/s", 3e-6),
        ("×10^-6 cm/s", 3e-6),
        ("cm/s ×10^-6", 3e-6),
    ],
)
def test_measurement_class_conversion_preserves_scientific_unit_factor(
    source_unit, expected_value
):
    value, unit = canonicalize_measurement(
        3,
        source_unit,
        measurement_class="permeability",
    )
    assert value == pytest.approx(expected_value)
    assert unit == "cm/s"


def test_compact_inverse_second_permeability_is_exactly_normalized():
    assert canonicalize_unit("cms-1").canonical == "cm/s"
    assert canonicalize_unit("cmsec^-1").canonical == "cm/s"


def test_conventional_unit_words_and_abbreviation_punctuation_are_cleaned():
    assert canonicalize_unit("µg./mL.").canonical == "µg/mL"
    assert canonicalize_unit("µg/milliliter").canonical == "µg/mL"
    assert canonicalize_unit("µg/gm").canonical == "µg/g"


def test_compact_areic_time_units_preserve_every_factor():
    expected = canonicalize_unit("µg/cm^2/h")
    for value in (
        "µg/cm²h",
        "µg/cm2.h",
        "µg.cm^-2.h^-1",
        "µg·cm^-2·h^-1",
    ):
        actual = canonicalize_unit(value)
        assert actual.canonical == expected.canonical
        assert actual.dimension == expected.dimension
        assert actual.unknown_tokens == ()

    diffusivity = canonicalize_unit("cm^2/s")
    compact = canonicalize_unit("cm2s^-1")
    assert compact.canonical == diffusivity.canonical
    assert compact.dimension == diffusivity.dimension


def test_malformed_exponent_suffix_fails_closed():
    result = canonicalize_unit("cm^2oops")
    assert result.unknown_tokens
    assert result.canonical != "cm"


# --------------------------------------------------------------------------- #
# Endpoint compatibility
# --------------------------------------------------------------------------- #

def test_units_compatible_matches_expected_quantity_kind():
    assert units_compatible("cm/s", "permeability") is True
    assert units_compatible("×10^-6 cm/s", "permeability") is True
    assert units_compatible("ng/mL", "permeability") is False
    assert units_compatible("ng/mL", "mass_concentration") is True
    assert units_compatible("µM", "concentration") is True


def test_units_compatible_returns_none_for_unfamiliar_units():
    assert units_compatible("wibbles/mL", "permeability") is None
    assert units_compatible("", "permeability") is None


def test_contextual_policy_always_converts_an_exact_dynamic_assay_match():
    assay = {
        "source_id": "fa",
        "canonical_endpoint": "caco2_mdck_pampa_permeability",
        "global_context": "caco_2",
        "global_species_context": None,
        "extra_field_is_ignored": "yes",
    }
    parsed = canonicalize_unit(
        "ng/cm^2·min", task="bioavailability_ma", assay=assay
    )
    assert parsed.canonical == "pg/cm^2·min"
    assert parsed.contextual_policy_status == "converted"
    assert parsed.contextual_rule_id == "bioavailability_ma.fa.caco2_flux.pg_ng.v1"
    assert parsed.contextual_conversion_factor == pytest.approx(1000.0)
    assert canonicalize_measurement(
        8.0,
        "ng/cm^2·min",
        task="bioavailability_ma",
        assay=assay,
    ) == (pytest.approx(8000.0), "pg/cm^2·min")


def test_contextual_policy_distinguishes_missing_null_and_wrong_context():
    base = {
        "source_id": "fa",
        "canonical_endpoint": "caco2_mdck_pampa_permeability",
        "global_context": "caco_2",
    }
    for assay in (
        base,
        {**base, "global_species_context": "human"},
        {**base, "global_context": "pampa", "global_species_context": None},
    ):
        result = canonicalize_unit(
            "ng/cm^2·min", task="bioavailability_ma", assay=assay
        )
        assert result.canonical == "ng/cm^2·min"
        assert result.contextual_policy_status == "no_matching_rule"


def test_contextual_policy_requires_task_and_assay_together():
    # An assay without a task can never match a rule, so it is still an error.
    with pytest.raises(ValueError, match="task and assay together"):
        canonicalize_unit("ng/mL", assay={"source_id": "fa"})


def test_task_alone_selects_the_qualifier_vocabulary_without_an_assay():
    """``task`` alone is meaningful: it picks the vocabulary, not a contextual rule."""
    result = canonicalize_unit("µg/g of skin", task="skin_reaction")
    assert result.unknown_tokens == ()
    assert result.contextual_policy_status == "not_requested"
    # The same string keeps ``skin`` unresolved for a task that never uses the token.
    assert canonicalize_unit("µg/g of skin", task="bioavailability_ma").unknown_tokens == (
        "skin",
    )
    assert canonicalize_unit("µg/g of skin").unknown_tokens == ("skin",)


def test_contextual_policy_loader_rejects_overlapping_dynamic_rules(tmp_path):
    policy = {
        "schema_version": "contextual_canonical_unit_policy.schema.v1",
        "policy_version": "test.v1",
        "rules": [
            {
                "rule_id": rule_id,
                "task": "test",
                "match": match,
                "accepted_units": ["ng/mL", "µg/mL"],
                "canonical_unit": "ng/mL",
                "review": {},
            }
            for rule_id, match in (
                ("broad", {"platform": "caco_2"}),
                ("specific", {"platform": "caco_2", "species": "human"}),
            )
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous contextual unit rules"):
        load_contextual_unit_policy(path)


def test_contextual_policy_loader_rejects_dimension_mismatch(tmp_path):
    policy = {
        "schema_version": "contextual_canonical_unit_policy.schema.v1",
        "policy_version": "test.v1",
        "rules": [
            {
                "rule_id": "bad",
                "task": "test",
                "match": {"platform": "caco_2"},
                "accepted_units": ["ng/mL", "ng"],
                "canonical_unit": "ng/mL",
                "review": {},
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_contextual_unit_policy(path)


def _vocabulary(**overrides):
    payload = {
        "schema_version": "qualifier_vocabulary.schema.v1",
        "policy_version": "test.v1",
        "vocabularies": [
            {"scope": "shared", "task": None, "tokens": ["protein"], "review": {}},
            {
                "scope": "task",
                "task": "skin_reaction",
                "tokens": ["skin"],
                "review": {},
            },
        ],
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload):
    path = tmp_path / "qualifiers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qualifier_vocabulary_rejects_a_token_that_is_a_real_unit(tmp_path):
    """Declaring a unit dimensionless would silently erase its dimension everywhere."""
    payload = _vocabulary()
    payload["vocabularies"][0]["tokens"] = ["mg"]
    with pytest.raises(ValueError, match="resolves as a unit"):
        load_qualifier_vocabulary(_write(tmp_path, payload))


def test_qualifier_vocabulary_rejects_a_task_redeclaring_a_shared_token(tmp_path):
    payload = _vocabulary()
    payload["vocabularies"][1]["tokens"] = ["protein"]
    with pytest.raises(ValueError, match="redeclares shared qualifier tokens"):
        load_qualifier_vocabulary(_write(tmp_path, payload))


def test_qualifier_vocabulary_rejects_unsorted_or_duplicated_tokens(tmp_path):
    payload = _vocabulary()
    payload["vocabularies"][1]["tokens"] = ["skin", "applied"]
    with pytest.raises(ValueError, match="must be sorted"):
        load_qualifier_vocabulary(_write(tmp_path, payload))
    payload["vocabularies"][1]["tokens"] = ["skin", "skin"]
    with pytest.raises(ValueError, match="duplicate qualifier tokens"):
        load_qualifier_vocabulary(_write(tmp_path, payload))


def test_qualifier_vocabulary_requires_a_shared_entry(tmp_path):
    payload = _vocabulary()
    payload["vocabularies"] = [payload["vocabularies"][1]]
    with pytest.raises(ValueError, match="requires a shared entry"):
        load_qualifier_vocabulary(_write(tmp_path, payload))


def test_qualifier_vocabulary_manifest_records_provenance():
    manifest = qualifier_vocabulary_manifest()
    assert manifest["schema_version"] == "qualifier_vocabulary.schema.v1"
    assert len(manifest["sha256"]) == 64
    assert manifest["scoping"] == "shared_plus_declared_task_fail_closed"
    # Skin-only vocabulary must not reach the other tasks, and a token shared by two tasks
    # but absent from the third must stay scoped rather than drifting into `shared`.
    assert "skin" in manifest["tasks"]["skin_reaction"]
    assert "skin" not in manifest["shared_tokens"]
    assert "skin" not in manifest["tasks"]["bioavailability_ma"]
    assert "administered" not in manifest["tasks"]["bioavailability_ma"]
    assert "absorbed" not in manifest["tasks"]["bbb_martins"]
    assert "absorbed" not in manifest["shared_tokens"]
