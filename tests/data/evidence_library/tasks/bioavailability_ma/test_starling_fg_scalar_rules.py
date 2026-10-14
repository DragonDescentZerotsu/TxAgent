import pytest

from data.processing.evidence_library.shared.v1.normalization.audit import (
    validate_measurement_pairs,
)
from data.processing.evidence_library.shared.v1.normalization.contracts import MeasurementPair
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
    normalize_cleaned_records,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AuxiliaryMetadataAttacher,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma import (
    starling_policy,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_fg_scalar_rules import (
    FG_SCALAR_RULE_VERSION,
    fg_scalar_rule_provenance,
    propose_fg_scalar,
    resolve_fg_measurement_pair,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_normalization_policy import (
    endpoint_specific_standardization_of_unit,
)
from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    family_assignment,
    spacing_and_spelling_decision,
)


class _NoReference:
    def attach(self, record):
        return {"canonical_reference_scope": "not_applicable"}


@pytest.mark.parametrize(
    ("text", "value", "variation", "semantic_label"),
    [
        ("efflux ratio 2.10", 2.10, None, "directional_efflux_ratio"),
        ("ER = 2.8", 2.8, None, "directional_efflux_ratio"),
        ("efflux ratio 2.8 ± 0.3", 2.8, 0.3, "directional_efflux_ratio"),
        ("uptake ratio 8", 8.0, None, "uptake_ratio"),
        ("AUC ratio 1.25 ± 0.32", 1.25, 0.32, "auc_ratio"),
    ],
)
def test_explicit_atomic_ratios_are_proposed(
    text, value, variation, semantic_label
):
    decision = propose_fg_scalar(text)

    assert decision.accepted is True
    assert decision.rule_id == "explicit_ratio_label"
    assert decision.semantic_label == semantic_label
    assert decision.canonical_unit == "ratio"
    assert decision.finite_scalar_value == pytest.approx(value)
    assert decision.variation_value == variation
    assert decision.to_dict()["rule_version"] == FG_SCALAR_RULE_VERSION


@pytest.mark.parametrize(
    ("text", "value", "approximate"),
    [
        ("1.5-fold", 1.5, False),
        ("14-fold", 14.0, False),
        ("approximately sixfold", 6.0, True),
        ("four-fold", 4.0, False),
        ("≈6-fold", 6.0, True),
    ],
)
def test_explicit_atomic_fold_changes(
    text, value, approximate
):
    decision = propose_fg_scalar(text)

    assert decision.accepted is True
    assert decision.rule_id == "explicit_fold_label"
    assert decision.canonical_unit == "fold"
    assert decision.finite_scalar_value == pytest.approx(value)
    assert decision.approximate is approximate


@pytest.mark.parametrize(
    ("text", "unit", "value", "semantic_label"),
    [
        ("Fg = 0.14", "fraction", 0.14, "fg_fraction"),
        ("FG 0.51", "fraction", 0.51, "fg_fraction"),
        ("F_G = 70.9% ± 8.1%", "%", 70.9, "fg_fraction"),
        ("Fa·Fg = 0.11 ± 0.03", "fraction", 0.11, "fa_times_fg_fraction"),
        ("F_a × F_g = 0.26", "fraction", 0.26, "fa_times_fg_fraction"),
        ("FaFg = 0.3", "fraction", 0.3, "fa_times_fg_fraction"),
        ("Fabs × Fg = 78%", "%", 78.0, "fabs_times_fg_fraction"),
    ],
)
def test_explicit_atomic_fg_fractions(
    text, unit, value, semantic_label
):
    decision = propose_fg_scalar(text)

    assert decision.accepted is True
    assert decision.rule_id == "explicit_fg_fraction_label"
    assert decision.canonical_unit == unit
    assert decision.finite_scalar_value == pytest.approx(value)
    assert decision.semantic_label == semantic_label


def test_relative_fg_percentage_is_not_mislabeled_as_absolute_fg():
    decision = propose_fg_scalar(
        "fraction escaping gut (Fg) reduced by ~57%",
        canonical_endpoint="gut_wall_extraction_or_first_pass",
    )

    assert decision.accepted is False
    assert decision.reason == "directional_or_comparative_measurement"
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    ("text", "value", "variation"),
    [
        ("46%", 46.0, None),
        ("149.99 ± 0.27 %", 149.99, 0.27),
        ("29.1%", 29.1, None),
        ("≈50%", 50.0, None),
    ],
)
def test_single_percentage_outcomes(text, value, variation):
    decision = propose_fg_scalar(text)

    if value > 100:
        assert decision.accepted is False
        assert decision.reason == "outside_bounded_percentage_domain"
        return
    assert decision.accepted is True
    assert decision.rule_id == "explicit_single_percentage"
    assert decision.canonical_unit == "%"
    assert decision.finite_scalar_value == pytest.approx(value)
    assert decision.variation_value == variation


@pytest.mark.parametrize(
    "text",
    [
        "AUC ratio 1.25 ± 0.32 (NS)",
        "absorption increased 1.5-fold",
        "14-fold increase in AUC (oral) with GF120918",
        "four-fold higher permeability",
        "Relative absorption = 3.7 (fold vs free suspension)",
        "in vivo FG 0.51",
        "46% inhibition of serosal-to-mucosal transport",
        "slight increase in absorptive transport at Labrasol 0.1% (v/v)",
    ],
)
def test_contextual_scalar_fragments_are_not_extracted(text):
    decision = propose_fg_scalar(text)
    assert decision.accepted is False
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    ("text", "endpoint", "unit", "value"),
    [
        (
            "Papp A-B = 27.09 ± 0.89 ×10^-6 cm/s",
            "bidirectional_permeability",
            "cm/s",
            27.09e-6,
        ),
        (
            "CLint: 0.0132 ± 0.00284 ml/min/mg protein",
            "intestinal_metabolism",
            "mL/mg·min·protein",
            0.0132,
        ),
        ("clearance k_B = 40 s^-1", "uptake_or_absorptive_transport", "/s", 40.0),
        ("23.80 µm", "efflux_or_secretory_transport", "µm", 23.8),
    ],
)
def test_atomic_embedded_physical_units(text, endpoint, unit, value):
    decision = propose_fg_scalar(text, canonical_endpoint=endpoint)

    assert decision.accepted is True
    assert decision.rule_id == "explicit_embedded_physical_unit"
    assert decision.canonical_unit == unit
    assert decision.finite_scalar_value == pytest.approx(value)


def test_scientific_notation_provenance_survives_fg_resolution():
    decision = propose_fg_scalar(
        "Papp = 2.5 ×10^-6 cm/s",
        canonical_endpoint="bidirectional_permeability",
    )
    pair = resolve_fg_measurement_pair(
        {"source_id": "fg", "measurement_text": "Papp = 2.5 ×10^-6 cm/s"},
        "bidirectional_permeability",
        MeasurementPair(None, None, "cleaned_only_non_scalar"),
    )

    assert decision.unit_notation_status == "unambiguous_scientific_notation"
    assert decision.unit_notation_factor == pytest.approx(1e-6)
    assert pair.unit_notation_status == decision.unit_notation_status
    assert pair.unit_notation_factor == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "text",
    [
        "Papp A-B 0.85 ± 0.10 x10^-6 cm/s; Papp B-A 5.96 ± 0.31 x10^-6 cm/s; ER 7.0",
        "Jmax 697 ± 151 pmol/min/mg protein; Km 46.0 ± 22.5 µM",
        "AUC0-24h reduced 25%; Cmax reduced 26%",
        "AUC0^-∞ and Cmax increased ~2.5-fold",
        "~60% reduction in peak plasma concentration and bioavailability",
        "FG 0.58 ± 0.27 vs FA 0.56 ± 0.30 (200 mg); FG 0.58 ± 0.32 vs FA 0.64 ± 0.39 (800 mg)",
        "Fa·Fg increased from 0.22 ± 0.18 to 0.84 ± 0.10",
    ],
)
def test_compound_measurements_are_never_split(text):
    decision = propose_fg_scalar(text)

    assert decision.accepted is False
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    "text",
    [
        "ER decreased by 75% (both apical and basolateral)",
        "FaFg ≈ 0.3 and unchanged by elacridar",
        "efflux ratio 6.4-fold higher than jejunum and lower ileum",
        "apparent oral bioavailability +35% (TTT and CGC/CGT homozygotes)",
        "∼40% recovery in both directions",
        "~40% inhibition (single‐ and double‐strength mGFJ extracts)",
        "1.4-fold increase in K_a and P_app with inhibitor co‐perfusion",
        "unidirectional and net fluxes reduced by ~97%",
    ],
)
def test_reviewed_coordinated_fg_outcomes_remain_non_scalar(text):
    decision = propose_fg_scalar(text)

    assert decision.accepted is False
    assert decision.reason == "compound_or_multi_outcome"
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (">7-fold increase in AUC", "bounded_measurement"),
        ("up to nine-fold increase in Papp", "bounded_measurement"),
        ("1.4-7.9% extraction", "range_measurement"),
        ("FG 0.49–0.70", "range_measurement"),
    ],
)
def test_bounds_and_ranges_remain_non_scalar(text, reason):
    decision = propose_fg_scalar(text)

    assert decision.accepted is False
    assert decision.reason == reason


def test_condition_number_is_not_selected_as_the_measurement():
    decision = propose_fg_scalar(
        "no change in absorption at 23 µM delafloxacin"
    )

    assert decision.accepted is False
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    "text",
    [
        "no difference in oxalate transport with or without 1.5 % oxalate supplementation",
        "AUC increased ~50%; absorption unchanged in intestinal loop",
        "bioavailability increased 49% in male rats; no change in females",
        "approximately 50% reduction in plasma AUC; unchanged half-life",
        "A→B transport 4%; B→A transport not observed",
        "absorption inhibited at 3 % ethanol",
        "significant depression of absorption in the presence of 5.4 % ethanol",
    ],
)
def test_numeric_conditions_and_additional_qualitative_outcomes_are_rejected(text):
    decision = propose_fg_scalar(text)

    assert decision.accepted is False
    assert decision.finite_scalar_value is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Fg = 1.2", "outside_bounded_fraction_domain"),
        ("efflux ratio 0", "nonpositive_dimensionless_ratio"),
        ("-3 µm", "nonpositive_physical_measurement"),
    ],
)
def test_semantic_domain_validation(text, reason):
    decision = propose_fg_scalar(text)

    assert decision.accepted is False
    assert decision.reason == reason


def test_source_text_is_not_rewritten_or_returned_as_canonical_text():
    source = "efflux ratio 2.10"
    decision = propose_fg_scalar(source)

    assert source == "efflux ratio 2.10"
    assert decision.canonical_measurement == "2.1"
    assert decision.canonical_unit == "ratio"


def test_variation_does_not_imply_approximation():
    decision = propose_fg_scalar(
        "Papp = 2.5 ± 0.2 ×10^-6 cm/s",
        canonical_endpoint="bidirectional_permeability",
    )

    assert decision.accepted is True
    assert decision.approximate is False
    assert decision.canonical_measurement == "0.0000025 ± 0.0000002"


def test_permeability_domain_is_applied_after_unit_folding():
    decision = propose_fg_scalar(
        "0.26 ×10^4 cm/s",
        canonical_endpoint="uptake_or_absorptive_transport_permeability",
    )

    assert decision.accepted is False
    assert decision.reason == "outside_permeability_domain"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "13.4 ± 1.6 µmol/100 cm·h",
            "unsupported_numeric_unit_denominator",
        ),
        (
            "2.0 ± 0.04 nmol/min per 100 mg tissue",
            "unsupported_numeric_unit_denominator",
        ),
        (
            "0.28 ± 0.18 nmol min^-1 mg^-1 protein",
            "ambiguous_protein_normalization_notation",
        ),
        (
            "4.18 µmol min-1 mg protein-1",
            "ambiguous_protein_normalization_notation",
        ),
    ],
)
def test_ambiguous_or_unscaled_physical_units_are_rejected(text, reason):
    decision = propose_fg_scalar(text, canonical_endpoint="intestinal_metabolism")

    assert decision.accepted is False
    assert decision.reason == reason


def test_fg_measurement_pair_resolver_is_strictly_source_scoped():
    baseline = MeasurementPair(
        "efflux ratio 2.1",
        None,
        "cleaned_only_non_scalar",
    )

    resolved = resolve_fg_measurement_pair(
        {"source_id": "fg", "measurement_text": "efflux ratio 2.1"},
        "efflux_or_secretory_transport",
        baseline,
    )
    untouched = resolve_fg_measurement_pair(
        {"source_id": "fa", "measurement_text": "efflux ratio 2.1"},
        "efflux_or_secretory_transport",
        baseline,
    )

    assert resolved == MeasurementPair(
        "2.1",
        "ratio",
        SOURCE_SPECIFIC_ATOMIC_SCALAR_STATUS,
    )
    assert untouched == baseline


def test_fg_scalar_rules_integrate_with_normalization_and_pair_validation():
    cleaned = [
        {
            "cleaned_record_id": "cleaned-ratio",
            "source_id": "fg",
            "source_row_number": 1,
            "endpoint_name": "efflux_or_secretory_transport",
            "measurement_text": "efflux ratio 2.10",
            "unit_text": None,
            "source_payload_json": "{}",
            "structure_status": "resolved",
            "canonical_smiles": "CCO",
        },
        {
            "cleaned_record_id": "cleaned-zero",
            "source_id": "fg",
            "source_row_number": 2,
            "endpoint_name": "gut_wall_extraction_or_first_pass",
            "measurement_text": "0% intestinal extraction (entire dose absorbed)",
            "unit_text": None,
            "source_payload_json": "{}",
            "structure_status": "resolved",
            "canonical_smiles": "CCN",
        },
        {
            "cleaned_record_id": "cleaned-compound",
            "source_id": "fg",
            "source_row_number": 3,
            "endpoint_name": "oral_exposure_change_due_to_gut_wall",
            "measurement_text": "AUC0^-∞ and Cmax increased ~2.5-fold",
            "unit_text": None,
            "source_payload_json": "{}",
            "structure_status": "resolved",
            "canonical_smiles": "CCC",
        },
        {
            "cleaned_record_id": "cleaned-unresolved",
            "source_id": "fg",
            "source_row_number": 4,
            "endpoint_name": "gut_wall_extraction_or_first_pass",
            "measurement_text": "Fg = 0.5",
            "unit_text": None,
            "source_payload_json": "{}",
            "structure_status": "unresolved",
            "canonical_smiles": None,
        },
    ]
    auxiliary_attacher = AuxiliaryMetadataAttacher()
    records = normalize_cleaned_records(
        cleaned,
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        source_measurement_resolver=resolve_fg_measurement_pair,
        family_resolver=family_assignment,
        record_enricher=lambda record: starling_policy._enrich_record(
            record, auxiliary_attacher, _NoReference()
        ),
    )

    ratio, zero, compound, unresolved = records
    assert ratio["measurement_text"] == "efflux ratio 2.10"
    assert ratio["canonical_measurement"] == "2.1"
    assert ratio["canonical_unit"] == "ratio"
    assert ratio["finite_scalar_value"] == pytest.approx(2.1)
    assert ratio["source_scalar_rule_version"] == FG_SCALAR_RULE_VERSION
    assert ratio["source_scalar_rule_id"] == "explicit_ratio_label"
    assert ratio["normalization_validity_status"] == "valid"

    assert zero["finite_scalar_value"] is None
    assert zero["canonical_unit"] is None
    assert zero["normalization_validity_status"] == "missing_canonical_unit"

    assert compound["finite_scalar_value"] is None
    assert compound["source_scalar_rule_reason"] == "compound_or_multi_outcome"
    assert compound["normalization_validity_status"] == "missing_canonical_unit"

    assert unresolved["finite_scalar_value"] == pytest.approx(0.5)
    assert unresolved["normalization_validity_status"] == "unresolved_structure"

    assert (
        validate_measurement_pairs(
            records,
            endpoint_specific_standardization_of_unit,
            resolve_fg_measurement_pair,
        )
        == []
    )


def test_fg_rule_provenance_is_null_for_other_sources():
    provenance = fg_scalar_rule_provenance(
        {
            "source_id": "fa",
            "measurement_text": "efflux ratio 2.1",
            "canonical_endpoint": "efflux_or_secretory_transport",
        }
    )

    assert provenance == {
        "source_scalar_rule_version": None,
        "source_scalar_rule_id": None,
        "source_scalar_rule_reason": None,
        "scalar_semantic_label": None,
    }
