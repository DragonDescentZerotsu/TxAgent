"""Structural guards on the hand-labelled measurement-resolution gold set.

The corpus is the accuracy oracle for the extraction pass. Its manifest records the
pre-model seed labels, post-replay corrections, and the independently reviewed
    extensions. Expectations are never machine-regenerated; these tests only assert that
the file stays complete, internally consistent, and consistent with its routing rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v7.measurement_routing import (
    route,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_measurement_resolution import (
    source_routing_rules,
)


GOLD_PATH = (
    Path(__file__).resolve().parent / "gold" / "bbb_martins.v7.jsonl"
)
V8_GOLD_PATH = GOLD_PATH.with_name("bbb_martins.v8.jsonl")
STATUSES = ("ok", "unsure", "relative", "unavailable")


def _load() -> tuple[dict, list[dict]]:
    lines = [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[0], lines[1:]


def test_corpus_is_fixed_complete_and_manually_reviewed() -> None:
    manifest, cases = _load()
    assert manifest["corpus_version"] == "measurement_resolution_gold.v7"
    assert manifest["task_id"] == "bbb_martins"
    assert manifest["expectation_update_policy"] == "manual_review_only"
    assert manifest["labelled_before_any_model_run"] is False
    assert manifest["routing_version"] == "starling_measurement_routing.v2"
    assert manifest["cases"] == len(cases) == 500
    assert len({case["audit_case_id"] for case in cases}) == 500


def test_v8_corpus_has_500_active_reviewed_cases() -> None:
    new_lines = [json.loads(line) for line in V8_GOLD_PATH.read_text().splitlines()]
    manifest, cases = new_lines[0], new_lines[1:]
    refresh = manifest["active_stage1_refresh"]
    active_ids = set(
        pq.read_table(
            refresh["cleaned_records_path"], columns=["cleaned_record_id"]
        ).column("cleaned_record_id").to_pylist()
    )

    assert manifest["corpus_version"] == "measurement_resolution_gold.v8.1"
    assert manifest["cases"] == len(cases) == 500
    assert len({case["audit_case_id"] for case in cases}) == 500
    assert {
        case["audit_case_id"].split(":", 1)[1] for case in cases
    } <= active_ids
    assert refresh["old_cases_retained"] == 371
    assert refresh["structure_rejected_cases_removed"] == 129
    assert refresh["active_replacement_cases"] == 129
    assert refresh["manually_reviewed_extraction_replacements"] == 36
    assert refresh["model_outputs_used_as_label_evidence"] is False
    assert all(
        len(answer["measurements"]) <= 1
        for case in cases
        for answer in [case["expected"], *case["expected"].get("alternatives", [])]
    )


def test_every_stratum_is_represented() -> None:
    """A stratum silently losing coverage would hide a whole failure mode."""
    manifest, cases = _load()
    first_pass = {
        "accept", "reject", "pos_exponent", "neg_exponent", "relative_language",
        "multi_quantity", "dispersion", "range_or_censored", "percent", "prose",
        "other_extract",
    }
    second_pass = {
        "influx_prose", "influx_numeric", "efflux", "multi_semicolon",
        "different_units", "ratio_endpoint", "censored_or_range", "percent_of",
        "plain_extract",
    }
    extension = {
        "direct_absolute", "direct_absolute_with_relative",
        "direct_endpoint_change", "direct_missing_unit",
        "direct_multi", "direct_multi_context", "direct_percent", "direct_range",
        "direct_ratio", "direct_relative", "efflux_absolute",
        "efflux_endpoint_change", "efflux_percent", "efflux_potency",
        "efflux_ratio", "efflux_relative", "efflux_unavailable", "influx_absolute",
        "influx_absolute_with_relative", "influx_kinetic", "influx_multi",
        "influx_multi_absolute", "influx_percent", "influx_relative",
        "influx_unavailable", "passive_dimensionless", "passive_endpoint_change",
        "passive_log", "passive_missing_unit", "passive_notation",
        "passive_percent", "passive_ratio", "passive_unit_conflict",
    }
    # The second pass shares ``pos_exponent`` with the first, deliberately: it is
    # the stratum carrying the 10^12 correction, so it gets sampled twice.
    original = first_pass | second_pass
    assert set(manifest["stratum_counts"]) == original | extension
    for stratum in original:
        assert manifest["stratum_counts"][stratum] >= 8, stratum
    for stratum in extension:
        assert manifest["stratum_counts"][stratum] >= 1, stratum
    assert manifest["stratum_counts"] == {
        stratum: sum(1 for case in cases if case["stratum"] == stratum)
        for stratum in manifest["stratum_counts"]
    }


def test_only_ok_carries_measurements() -> None:
    """The schema's core invariant: a non-ok status publishes no value."""
    _, cases = _load()
    for case in cases:
        expected = case["expected"]
        assert expected["status"] in STATUSES, case["audit_case_id"]
        if expected["status"] == "ok":
            assert expected["measurements"], case["audit_case_id"]
        else:
            assert expected["measurements"] == [], case["audit_case_id"]


def test_every_measurement_carries_its_own_complete_unit() -> None:
    """No entry may inherit a unit, so an entry maps 1:1 onto a future exploded row."""
    _, cases = _load()
    for case in cases:
        for entry in case["expected"]["measurements"]:
            assert entry["measurement"], case["audit_case_id"]
            assert entry["unit"], case["audit_case_id"]


def test_every_measurement_is_a_plain_decimal_coefficient() -> None:
    import re

    plain_decimal = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    _, cases = _load()
    for case in cases:
        for entry in case["expected"]["measurements"]:
            assert plain_decimal.fullmatch(entry["measurement"]), case[
                "audit_case_id"
            ]


def test_no_expected_unit_carries_a_positive_power_of_ten() -> None:
    """A positive ``x10^n`` in a unit is the convention that is wrong by 10^2n.

    Fifteen sampled rows carry one (``PS ... ml/gm/min x 10^4`` for a sucrose PS of
    order 1e-4), and every one is labelled with the corrected negative exponent.  A
    positive exponent surviving into an expectation would enshrine the bug.
    """
    import re

    positive = re.compile(r"10\s*\^?\s*\{?\s*[1-9]")
    negative = re.compile(r"10\s*\^?\s*\{?\s*[-−–]")
    _, cases = _load()
    for case in cases:
        for entry in case["expected"]["measurements"]:
            unit = entry["unit"] or ""
            if positive.search(unit) and not negative.search(unit):
                pytest.fail(f"{case['audit_case_id']}: positive exponent {unit!r}")


def test_current_router_only_moves_plain_decimals_out_of_extraction() -> None:
    rules = source_routing_rules()
    _, cases = _load()
    transitions = {}
    for case in cases:
        record = dict(case["input"])
        decision = route(record, rules[case["source_id"]], task="bbb_martins")
        key = (case["route_bucket"], decision.bucket)
        transitions[key] = transitions.get(key, 0) + 1
    assert transitions == {
        ("accept", "accept"): 25,
        ("extract", "accept"): 63,
        ("extract", "extract"): 392,
        ("reject", "reject"): 20,
    }


def test_historical_accepted_rows_remain_source_exact() -> None:
    rules = source_routing_rules()
    _, cases = _load()
    checked = 0
    for case in cases:
        if case["route_bucket"] != "accept":
            continue
        decision = route(dict(case["input"]), rules[case["source_id"]], task="bbb_martins")
        expected = case["expected"]["measurements"]
        assert case["expected"]["status"] == "ok", case["audit_case_id"]
        assert len(expected) == 1, case["audit_case_id"]
        assert decision.bucket == "accept"
        assert decision.measurement_text is not None
        assert decision.unit_text is not None
        checked += 1
    assert checked == 25  # only the first pass sampled the accept bucket


def test_rejected_rows_expect_no_measurement() -> None:
    """A rejected row is never sent, so its label records what the answer would be."""
    _, cases = _load()
    rejected = [case for case in cases if case["route_bucket"] == "reject"]
    assert len(rejected) == 20  # only the first pass sampled the reject bucket
    for case in rejected:
        assert case["expected"]["status"] == "unavailable", case["audit_case_id"]


def test_status_mix_is_broad_enough_to_score_each_branch() -> None:
    """All four statuses need real support, or the corpus cannot detect a model that
    never uses one of them -- the specific worry being a model that never says
    ``unsure`` and guesses instead."""
    manifest, cases = _load()
    counts = manifest["status_counts"]
    assert counts == {
        status: sum(1 for case in cases if case["expected"]["status"] == status)
        for status in STATUSES
    }
    for status in STATUSES:
        assert counts[status] >= 35, (status, counts[status])


def test_final_endpoint_aware_review_is_recorded() -> None:
    manifest, cases = _load()
    review = manifest["review_lineage"]["final_endpoint_aware_review"]
    assert review["cases_reviewed"] == len(cases) == 500
    assert review["cases_relabelled"] == len(review["case_ids"]) == 3
    assert review["measurement_pairs_reformatted"] == 3
    assert len(review["measurement_pair_case_ids"]) == 3
    by_id = {case["audit_case_id"]: case for case in cases}
    assert by_id[review["case_ids"][0]]["expected"]["status"] == "unavailable"
    assert by_id[review["case_ids"][1]]["expected"]["status"] == "unsure"
    assert by_id[review["case_ids"][2]]["expected"]["status"] == "unsure"


# --------------------------------------------------------------------------- #
# the folded scalar: what the pipeline must produce from a label
# --------------------------------------------------------------------------- #

_EXPONENT = __import__("re").compile(r"10\s*\^?\s*\{?\s*(-|−|–)?\s*(\d+)\s*\}?")


def _naive_fold(measurement: str, unit: str) -> float:
    """A deliberately independent fold: printed number times the unit's power of ten.

    Not a reimplementation for its own sake -- the point is that the production
    folder and this agree.  A single implementation checking itself would not catch
    the failure mode that matters, which is an exponent applied zero times or twice.
    """
    value = float(measurement)
    match = _EXPONENT.search(unit or "")
    if match:
        sign = -1 if match.group(1) else 1
        value *= 10 ** (sign * int(match.group(2)))
    if __import__("re").search(r"(?:/|per\s*)100\s*g", unit, flags=__import__("re").I):
        value /= 100
    return value


def test_every_ok_entry_records_the_scalar_the_pipeline_must_produce() -> None:
    manifest, cases = _load()
    assert manifest["records_expected_folded_scalar"] is True
    for case in cases:
        if case["expected"]["status"] != "ok":
            continue
        for entry in case["expected"]["measurements"]:
            assert entry["expected_scalar"] is not None, case["audit_case_id"]
            assert entry["expected_canonical_unit"], case["audit_case_id"]


def test_the_production_folder_reproduces_every_expected_scalar() -> None:
    """End-to-end: label -> folded number, checked against the recorded expectation.

    This is the gate on folding happening exactly once.  ``finite_scalar`` is read
    straight off the resolver's text (``measurements.py:580-598``) and
    ``unit_notation_factor`` is never re-applied, so a resolver returning an unfolded
    pair silently drops the factor -- ``11.5`` with ``10^-6 cm/s`` publishing ``11.5``
    rather than ``1.15e-05``, wrong by 10^6 and looking entirely healthy.
    """
    from data.processing.evidence_library.shared.v1.normalization.measurements import (
        normalize_measurement_and_unit,
        parse_point_measurement,
    )

    _, cases = _load()
    checked = 0
    for case in cases:
        for entry in case["expected"]["measurements"]:
            pair = normalize_measurement_and_unit(
                entry["measurement"], entry["unit"], task="bbb_martins"
            )
            folded = parse_point_measurement(pair.canonical_measurement).value
            assert folded is not None, case["audit_case_id"]
            assert folded == pytest.approx(entry["expected_scalar"], rel=1e-9)
            assert pair.canonical_unit == entry["expected_canonical_unit"]
            checked += 1
    assert checked == 290


def test_expected_scalars_agree_with_an_independent_fold() -> None:
    """Two implementations of the same arithmetic must land on the same number."""
    _, cases = _load()
    for case in cases:
        for entry in case["expected"]["measurements"]:
            naive = _naive_fold(entry["measurement"], entry["unit"])
            assert naive == pytest.approx(entry["expected_scalar"], rel=1e-9), (
                case["audit_case_id"]
            )


def test_corrected_exponents_land_in_a_plausible_physical_range() -> None:
    """A sign correction must move the value toward physical reality, not away.

    Every positive-exponent row was relabelled with a negative exponent, so the
    resulting scalars should look like the quantities they claim to be.  Permeability
    in cm/s spans roughly 1e-9..1e-2; the literal reading of these rows produced
    values above 1e+3, which is what makes the correction necessary.
    """
    _, cases = _load()
    checked = 0
    for case in cases:
        if case["stratum"] != "pos_exponent":
            continue
        for entry in case["expected"]["measurements"]:
            scalar = entry["expected_scalar"]
            assert 1e-9 < abs(scalar) < 1.0, (case["audit_case_id"], scalar)
            checked += 1
    # 23 positive-exponent rows sampled across both passes; two are multi-region or
    # ranged and labelled unsure, so 21 carry a corrected scalar.
    assert checked == 21


def test_the_positive_exponent_correction_records_why_it_is_the_right_label() -> None:
    """The contested labels must carry their justification, not just their value.

    ``x 10^n`` in a unit label is genuinely ambiguous in the literature -- the literal
    "multiply" reading is also used -- so a bare corrected label would be an
    unexplained assertion.  What disambiguates each row is that the literal reading
    is physically impossible for the quantity named, corroborated across three
    populations in the corpus and, for two papers, against the source itself.
    """
    manifest, cases = _load()
    correction = manifest["positive_exponent_correction"]
    assert "not_a_general_rule" in correction
    evidence = correction["corpus_evidence"]
    # The two unambiguous populations agree; the literal reading does not.
    plain = evidence["plain_cm_s_no_exponent"]["median_cm_s"]
    negative = evidence["negative_exponent"]["median_cm_s"]
    literal = evidence["positive_exponent_literal"]["median_cm_s"]
    corrected = evidence["positive_exponent_corrected"]["median_cm_s"]
    assert plain == pytest.approx(negative, rel=0.5)
    assert corrected == pytest.approx(plain, rel=0.5)
    assert literal / corrected == pytest.approx(1e12, rel=0.01)

    positive = [case for case in cases if case["stratum"] == "pos_exponent"]
    assert len(positive) == 23
    assert "corpus_evidence_caveat" in correction
    assert "flow_ceiling_argument" in correction
    assert all(case.get("pmid") for case in positive), "every row must name its paper"
    with_paper = [case for case in positive if case.get("label_provenance")]
    assert len(with_paper) == correction["rows_with_paper_provenance"] >= 3
    for case in with_paper:
        provenance = case["label_provenance"]
        assert provenance["paper"] and provenance["evidence"]
        assert "confirmed" in provenance["verdict"]


def test_the_unsure_policy_is_recorded_and_applied() -> None:
    """A label that resolves what the source left open is the reviewer's, not the data's.

    Three rows were relabelled on this basis: two ``SUV`` and one ``BUI`` where the row
    states no unit and the reviewer supplied the endpoint's own name as one.  A unit the
    source *did* state is kept even when the normalizer cannot resolve every token --
    ``ug EB/g tissue`` names Evans Blue inside the unit, which is a normalizer gap and
    not an ambiguity.
    """
    manifest, cases = _load()
    policy = manifest["unsure_policy"]
    for key in ("ambiguous_unit_reading", "undetermined_unit", "applied"):
        assert policy[key]

    # Nothing may be published under a unit the reviewer invented for a unitless row.
    for case in cases:
        if case["input"].get("unit_text") is not None:
            continue
        for entry in case["expected"]["measurements"]:
            assert entry["unit"] not in ("SUV", "BUI"), case["audit_case_id"]


def test_skin_indirect_gold_is_complete_under_current_routing() -> None:
    import re

    from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_measurement_resolution import (
        source_routing_rules as skin_routing_rules,
    )

    path = GOLD_PATH.with_name("skin_reaction.v1.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    manifest, cases = lines[0], lines[1:]
    assert manifest["cases"] == len(cases) == 300
    assert manifest["source_counts"] == {
        "sensitization_aop": 150,
        "skin_exposure": 150,
    }
    assert manifest["status_counts"] == {
        "ok": 257,
        "unavailable": 20,
        "unsure": 17,
        "relative": 6,
    }
    residual_review = manifest["review"]["post_v2_residual_independent_review"]
    assert residual_review["cases_reviewed"] == len(residual_review["case_ids"]) == 8
    assert residual_review["model_outputs_used_as_label_evidence"] is False
    assert residual_review["cases_relabelled"] == 1
    assert residual_review["relabelled_case_ids"] == [
        "c96ed388e2d7cb389a3bfd1538f2fcf909814e7f7fe9746f866c3f3dba0da7f0"
    ]
    assert len({case["audit_case_id"] for case in cases}) == 300
    rules = skin_routing_rules()
    decimal = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    route_counts = {"accept": 0, "extract": 0}
    for case in cases:
        bucket = route(
            case["input"], rules[case["source_id"]], task="skin_reaction"
        ).bucket
        route_counts[bucket] += 1
        assert case["canonical_endpoint_name"]
        assert case["request_endpoint_profile_blocks"]
        expected = case["expected"]
        assert bool(expected["measurements"]) == (expected["status"] == "ok")
        assert all(decimal.fullmatch(item["measurement"]) for item in expected["measurements"])
        assert all(item["unit"] for item in expected["measurements"])
    assert route_counts == {"accept": 137, "extract": 163}
