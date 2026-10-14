"""Correctness invariants for deterministic measurement routing.

Routing is the only component that discards evidence without an extraction pass
behind it, so these tests are written around what it must *refuse* to settle.  Real
Starling strings are used throughout; the cases that must reach extraction are the
ones that motivated the design.
"""

import pytest

from data.processing.evidence_library.versions.v7.measurement_routing import (
    attach_stage1_routes,
    DeclarativeNonScalarRule,
    NO_DIGIT_RULE_ID,
    PURE_NUMBER_RULE_ID,
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    is_obvious_unit,
    is_pure_number,
    route,
)


TASK = "bbb_martins"


def _rules(**overrides) -> SourceRoutingRules:
    base = {"source_id": "direct_bbb"}
    base.update(overrides)
    return SourceRoutingRules(**base)


def _route(measurement, unit=None, **kw):
    return route(
        {"measurement_text": measurement, "unit_text": unit},
        _rules(**kw),
        task=TASK,
    )


# --------------------------------------------------------------------------- #
# reject: one airtight test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "measurement",
    [
        None,
        "",
        "   ",
        "high",
        "low",
        "PSt > PSf",
        "Glucose is transported across the barrier by facilitative diffusion.",
        "one-tenth to about one-third",
    ],
)
def test_a_measurement_column_without_a_digit_is_rejected(measurement) -> None:
    """No digit means nothing for this stage to extract.

    Stage 01 calls the controlled encoder before this scalar router. Quantities
    written in words are rejected here too: they are 3.9% of digit-free text, not
    worth a special case.
    """
    assert _route(measurement, "cm/s") == RouteDecision("reject", NO_DIGIT_RULE_ID)


# --------------------------------------------------------------------------- #
# accept: a positive decimal with a nonempty unit; the exact map decides the unit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "measurement,expected",
    [("12", "12"), ("2.5", "2.5"), ("0.004", "0.004"),
     (4.0, "4"), (2.5, "2.5"), (17, "17")],
)
def test_a_positive_decimal_with_a_unit_is_accepted_exactly(
    measurement, expected
) -> None:
    assert _route(measurement, "cm/s") == RouteDecision(
        "accept", PURE_NUMBER_RULE_ID, expected, "cm/s"
    )


def test_scientific_notation_is_not_rewritten_by_the_router() -> None:
    assert _route("6.0902e-06", "cm/s") == RouteDecision(
        "accept", PURE_NUMBER_RULE_ID, "6.0902e-06", "cm/s"
    )


def test_float_overflow_reaches_extraction() -> None:
    assert _route("1e309", "cm/s").bucket == "extract"


@pytest.mark.parametrize("unit", ["%", "fold", "10^-6 cm/s", "SUV"])
def test_the_exact_map_not_the_router_decides_unit_semantics(unit) -> None:
    assert _route("12", unit) == RouteDecision(
        "accept", PURE_NUMBER_RULE_ID, "12", unit
    )


@pytest.mark.parametrize(
    "measurement",
    [
        "45%",                      # a percentage is not a bare number
        "8 ± 1.5",                  # dispersion: the pair is not one value
        "33.64±3.42",
        "40-70",                    # no single value exists
        "1.00-1.50",
        "<0.1",                     # censored: bounded, not known
        ">100",
        "~45",                      # the qualifier is not carried by the unit
        "0.35 (0.30-0.46)",         # CI, range or n?
        "90-fold higher",
        "3.7% (1/27)",
        "Papp = 15.1 × 10^-6 cm/s",
        "396 nmol/cm2",             # unit embedded in the value
        "1.77 + 0.08",              # is + a dispersion marker or an addition?
        "12,000",
    ],
)
def test_anything_other_than_a_bare_number_reaches_extraction(measurement) -> None:
    assert _route(measurement, "cm/s").bucket == "extract"


@pytest.mark.parametrize("measurement", ["-6.36", "-3.54", "-0.0", "0", "0.0"])
def test_non_positive_values_reach_extraction(measurement) -> None:
    """An obvious unit names a strictly positive quantity.

    A negative value against a plain unit means the *unit label* is wrong: 238 of
    240 negative BBB candidates are log values whose endpoint reads ``log Pe`` or
    ``Log Kp`` while the unit column still says ``cm/s``.  Accepting -6.36 as cm/s
    would place a logarithm on the same axis as linear permeabilities.  Zero goes
    too: a reported permeability of 0 is usually a not-detected placeholder.
    """
    assert _route(measurement, "cm/s").bucket == "extract"


def test_a_source_without_a_unit_column_can_never_accept() -> None:
    """A bare number with no unit cannot be given one mechanically."""
    rules = _rules(source_id="efflux_transport", unit_field="")
    record = {"measurement_text": "2.4", "unit_text": "cm/s"}
    assert route(record, rules, task=TASK).bucket == "extract"


# --------------------------------------------------------------------------- #
# what makes a unit "obvious"
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "unit",
    ["cm/s", "µM", "nM", "mM", "ng/mL", "mg/L", "nm/s", "min", "hours", "ml/min",
     "µmol/ml", "mL/100 g/min"],
)
def test_plain_physical_units_are_obvious(unit) -> None:
    assert is_obvious_unit(unit, task=TASK)


@pytest.mark.parametrize("unit", ["mg/kg", "ng/g", "ng/g tissue", "µg/g", "%ID/g"])
def test_mass_over_mass_concentrations_are_obvious(unit) -> None:
    """Not gated on ``is_dimensionless``: mass/mass cancels dimensionally.

    ``mg/kg`` and ``ng/g tissue`` are absolute concentrations even though their
    dimensions cancel.  Gating on dimensionlessness refused every one of them and
    cost 3,238 direct_bbb rows for nothing.
    """
    assert is_obvious_unit(unit, task=TASK)


@pytest.mark.parametrize("unit", ["cm s^-1", "min-1", "ml g^-1 min^-1", "µl·min^-1·g^-1"])
def test_dimension_exponents_are_obvious(unit) -> None:
    """A digit in a unit is not automatically a scale factor.

    ``cm s^-1`` *is* ``cm/s``; the digit is a dimension exponent and unambiguous.
    Refusing every unit containing a digit would lose these for no gain.
    """
    assert is_obvious_unit(unit, task=TASK)


@pytest.mark.parametrize("unit", ["%", "percent", "per cent", "fold", "ratio", "ppm"])
def test_bare_proportions_are_not_obvious(unit) -> None:
    """These name no denominator, so a bare number against them is not absolute."""
    assert not is_obvious_unit(unit, task=TASK)


@pytest.mark.parametrize(
    "unit",
    ["10^-6 cm/s", "×10^-6 cm/s", "10^{-6} cm s^{-1}", "× 10^6 cm/s", "x 10^6 cm s^-1"],
)
def test_embedded_scale_factors_are_not_obvious(unit) -> None:
    """A scale factor inside a unit label is a presentation convention.

    A header ``Papp x 10^6 (cm/s)`` means the printed number *is* Papp x 10^6, so
    the true value is smaller -- while the literal reading multiplies, turning
    carbamazepine's PAMPA permeability of 1.15e-05 cm/s into 1.15e+07.  Rather than
    admit the safe half of an ambiguous convention, all scale-factor spellings go to
    extraction where the support text can be read.
    """
    assert not is_obvious_unit(unit, task=TASK)


def test_implausible_dimension_exponents_are_not_obvious() -> None:
    """``10-6 cm s-6`` is OCR damage for ``s-1`` and resolves to ``cm/s^6``."""
    assert not is_obvious_unit("10-6 cm s-6", task=TASK)


@pytest.mark.parametrize("unit", [None, "", "   ", "SUV", "mg%", "of 20 guinea pigs"])
def test_unresolved_or_absent_units_are_not_obvious(unit) -> None:
    assert not is_obvious_unit(unit, task=TASK)


# --------------------------------------------------------------------------- #
# declarative rules: retained mechanism, no BBB instance
# --------------------------------------------------------------------------- #


def test_bbb_declares_no_declarative_rule_outs() -> None:
    """Audited and rejected: those columns record a conclusion, not a quantity.

    ``efflux.evidence_type == qualitative_transporter_claim`` would rule out 2,823
    rows, 794 of which carry a real IC50 or Km -- a paper can report IC50 = 0.51 uM
    and still conclude only "this is an inhibitor".
    """
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        source_routing_rules,
    )

    for rules in source_routing_rules().values():
        assert rules.declarative_non_scalar == ()


def test_a_declarative_rule_only_fires_on_rows_that_carry_a_number() -> None:
    """The no-digit test runs first, keeping declarative IDs auditable.

    The two rule-outs overlap heavily.  Letting ``no_digit`` absorb the overlap
    means a declarative rule ID appears only where the source *did* report a
    number -- exactly the population needing manual review.
    """
    rule = DeclarativeNonScalarRule(
        "report_type_relative.v1", "report_type", frozenset({"relative_comparison"})
    )
    rules = _rules(declarative_non_scalar=(rule,))
    assert route(
        {"measurement_text": "2.4", "report_type": "relative_comparison"},
        rules,
        task=TASK,
    ).rule_id == "report_type_relative.v1"
    assert route(
        {"measurement_text": "substrate", "report_type": "relative_comparison"},
        rules,
        task=TASK,
    ).rule_id == NO_DIGIT_RULE_ID


def test_a_missing_column_is_never_a_declaration() -> None:
    """An empty declared value would silently rule out every unlabelled row."""
    with pytest.raises(ValueError, match="empty value"):
        DeclarativeNonScalarRule("bad.v1", "f", frozenset({""}))


# --------------------------------------------------------------------------- #
# structural guarantees
# --------------------------------------------------------------------------- #


def test_booleans_are_not_measurements() -> None:
    """``isinstance(True, int)`` is True, so this needs its own guard."""
    assert not has_digit(True)
    assert not is_pure_number(False)
    assert _route(True, "cm/s").bucket == "reject"


def test_non_finite_floats_are_not_measurements() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        assert not has_digit(value)


def test_integral_floats_do_not_gain_a_spelling_the_source_lacked() -> None:
    """A float column yields ``4``, not ``4.0``: routing must not invent digits."""
    assert _route(4.0, "cm/s").measurement_text == "4"


def test_every_row_lands_in_exactly_one_bucket() -> None:
    cases = [
        (None, None, "reject"),
        ("prose with no number", "cm/s", "reject"),
        ("12", "cm/s", "accept"),
        ("12", "fold", "accept"),
        ("12", "10^-6 cm/s", "accept"),
        ("8 ± 1", "cm/s", "extract"),
        ("-6.36", "cm/s", "extract"),
    ]
    assert [_route(m, u).bucket for m, u, _ in cases] == [e for _, _, e in cases]


def test_stage1_routes_numeric_measurements_before_categorical_fallback() -> None:
    rows = [
        {
            "source_id": "efflux_transport",
            "endpoint_name": "Efflux Ratio",
            "measurement_text": "2.4",
            "interaction_conclusion": "substrate",
        },
        {
            "source_id": "influx_transport",
            "endpoint_name": "Influx transport",
            "measurement_text": "2.4",
        },
        {
            "source_id": "passive_permeability",
            "endpoint_name": "Papp",
            "measurement_text": "2.4",
            "unit_text": "cm/s",
        },
        {
            "source_id": "efflux_transport",
            "endpoint_name": "Efflux transport",
            "measurement_text": None,
            "interaction_conclusion": "substrate",
        },
    ]

    routed = attach_stage1_routes(rows, task=TASK)

    assert [row["measurement_resolution_route"] for row in routed] == [
        "extract",
        "extract",
        "accept",
        "categorical",
    ]
    assert routed[3]["measurement_resolution_rule_id"] == (
        "efflux_substrate_binary.v1"
    )
    assert routed[0]["canonical_endpoint_name"] == "efflux_ratio"
    assert routed[1]["canonical_endpoint_name"] == "influx_transport"
    assert routed[2]["canonical_endpoint_name"] == "apparent_permeability"
    assert routed[2]["measurement_resolution_exact_measurement"] == "2.4"
    assert routed[2]["measurement_resolution_exact_unit"] == "cm/s"


def test_stage1_can_use_the_launchers_endpoint_mapping() -> None:
    [routed] = attach_stage1_routes(
        [
            {
                "source_id": "passive_permeability",
                "endpoint_name": "Papp",
                "measurement_text": "2.4",
                "unit_text": "cm/s",
            }
        ],
        task=TASK,
        endpoint_resolver=lambda record: f"mapped_{record['endpoint_name']}",
    )
    assert routed["canonical_endpoint_name"] == "mapped_Papp"


def test_skin_structured_counts_are_a_source_exact_fraction() -> None:
    [routed] = attach_stage1_routes(
        [
            {
                "source_id": "direct_skin_reaction",
                "endpoint_name": "sensitization",
                "canonical_endpoint_name": "sensitization",
                "measurement_text": None,
                "positive_count": 4,
                "total_tested": 10,
            }
        ],
        task="skin_reaction",
    )

    assert routed["measurement_resolution_route"] == "accept"
    assert routed["measurement_resolution_rule_id"] == (
        "structured_positive_count_fraction.v1"
    )
    assert routed["measurement_resolution_exact_measurement"] == "0.4"
    assert routed["measurement_resolution_exact_unit"] == "fraction"
    assert routed["measurement_resolution_exact_unit_is_canonical"] is True


def test_routing_is_deterministic_across_repeated_calls() -> None:
    """Offline candidate selection and runtime resolution must agree exactly."""
    assert len({_route("12", "cm/s") for _ in range(25)}) == 1


def test_unresolved_rows_carry_no_rule_output() -> None:
    with pytest.raises(ValueError, match="carry no rule output"):
        RouteDecision("extract", PURE_NUMBER_RULE_ID)


def test_settled_rows_must_name_their_rule_and_carry_a_pair() -> None:
    with pytest.raises(ValueError, match="must name the rule"):
        RouteDecision("reject")
    with pytest.raises(ValueError, match="must carry a measurement"):
        RouteDecision("accept", PURE_NUMBER_RULE_ID)
    with pytest.raises(ValueError, match="must carry a unit"):
        RouteDecision("accept", PURE_NUMBER_RULE_ID, "12")


def test_reserved_rule_ids_cannot_be_reused() -> None:
    with pytest.raises(ValueError, match="reserved rule ID"):
        SourceRoutingRules(
            source_id="x",
            declarative_non_scalar=(
                DeclarativeNonScalarRule(NO_DIGIT_RULE_ID, "f", frozenset({"v"})),
            ),
        )


# --------------------------------------------------------------------------- #
# the prompt template
# --------------------------------------------------------------------------- #


def test_the_prompt_omits_the_unit_column_for_sources_that_have_none() -> None:
    """efflux and influx have no unit column, so they must not be told about one.

    Naming a field that is always null invites the model to invent a value for it.
    Those sources get the branch that reads the unit out of the measurement or
    support text instead.
    """
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        prompt_row_fields,
        render_prompt,
    )

    assert "unit_text" in prompt_row_fields("direct_bbb")
    assert "unit_text" not in prompt_row_fields("efflux_transport")

    with_unit = render_prompt("direct_bbb")
    without = render_prompt("efflux_transport")
    assert "A unit label may carry a power of ten" in with_unit
    assert "This source has no unit column" in without
    # The doubled-exponent warning has to survive in both branches.
    assert "applied twice" in with_unit and "applied twice" in without


def test_the_prompt_states_the_actual_batch_size() -> None:
    """A row count in prose that drifts from the real batch is a silent lie."""
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        BATCH_SIZE,
        render_prompt,
    )

    assert f"at most {BATCH_SIZE} rows" in render_prompt("direct_bbb")


def test_the_prompt_covers_every_status_and_the_multi_quantity_rule() -> None:
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        SOURCE_IDS,
        render_prompt,
    )

    for source_id in SOURCE_IDS:
        text = render_prompt(source_id)
        for status in ("ok", "unsure", "relative", "unavailable"):
            assert f'"{status}"' in text
        # same unit -> unsure; different units -> one entry each
        assert "share a unit" in text and "DIFFERENT units" in text
        assert "not automatically relative" in text


def test_an_unknown_source_cannot_render_a_prompt() -> None:
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        render_prompt,
    )

    with pytest.raises(ValueError, match="unknown source_id"):
        render_prompt("not_a_source")


def test_the_prompt_manifest_records_template_and_render_digests() -> None:
    """The template digest moves when the file is edited; renders are what was sent."""
    from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
        SOURCE_IDS,
        prompt_manifest,
    )

    manifest = prompt_manifest()
    assert len(manifest["template_sha256"]) == 64
    assert set(manifest["rendered_sha256"]) == set(SOURCE_IDS)
    # Two distinct renders: with and without a unit column.
    assert len(set(manifest["rendered_sha256"].values())) == 2
