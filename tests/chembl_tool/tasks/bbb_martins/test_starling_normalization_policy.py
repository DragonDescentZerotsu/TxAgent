import json
import hashlib

import pandas as pd

from data.processing.evidence_library.shared.v1.normalization.measurements import (
    normalize_measurement_and_unit,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_endpoint_normalization import (
    DIRECT_ENDPOINT_MAPPING_VERSION,
    EndpointNormalizer,
    canonical_efflux_endpoint,
    canonical_influx_endpoint,
    canonical_passive_endpoint,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins import (
    starling_endpoint_normalization,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_normalization_policy import (
    source_measurement_resolver,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_PATHS,
    endpoint_concept,
    family_assignment,
    spacing_and_spelling_decision,
)


def test_endpoint_concept_maps_cover_every_frozen_source_pair():
    for path in ENDPOINT_CONCEPT_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["mappings"]:
            assert endpoint_concept(
                payload["source_id"],
                row["endpoint_name"],
                row["canonical_endpoint_name"],
            ) == row["canonical_endpoint_concept"]

    assert endpoint_concept(
        "direct_bbb", "LogPS(Blood Brain Barrier Permeability)", "logps"
    ) == "log_ps"
    assert endpoint_concept(
        "passive_permeability", "Papp AP", "apparent_permeability"
    ) == "papp_a_to_b"
    assert endpoint_concept(
        "passive_permeability", "Papp BL", "apparent_permeability"
    ) == "papp_b_to_a"
    assert endpoint_concept("direct_bbb", "log PS predicted", "log_ps") == (
        "log_ps_predicted"
    )


def test_endpoint_concept_lookup_fails_closed():
    try:
        endpoint_concept("direct_bbb", "new endpoint", "new_endpoint")
    except ValueError as error:
        assert "unreviewed endpoint concept" in str(error)
    else:
        raise AssertionError("unreviewed endpoint concept was accepted")


def test_four_source_families_are_stable_and_exact():
    expected = {
        "direct_bbb": "Tier 1.starling_direct_bbb_evidence",
        "passive_permeability": "Mechanism.passive_permeability",
        "efflux_transport": "Mechanism.efflux_transport",
        "influx_transport": "Mechanism.influx_transport",
    }
    assert {
        source: family_assignment(source, "anything").group_id
        for source in expected
    } == expected


def test_direct_endpoint_remains_unactivated_before_human_approval():
    decision = spacing_and_spelling_decision("direct_bbb", "  Log BB  ")
    assert decision.endpoint_name == "Log BB"
    assert decision.spacing_and_spelling_endpoint == "Log BB"
    assert decision.reason == "direct_mapping_awaiting_human_approval"


def test_deterministic_sources_use_measurement_endpoints_not_policy_context():
    assert canonical_passive_endpoint("Papp (A→B)")[0] == "papp_a_to_b"
    assert canonical_efflux_endpoint("Km")[0] == "michaelis_menten_km"
    assert canonical_influx_endpoint("blood‑to‑brain transport")[0] == "blood_to_brain_transport"
    assert canonical_passive_endpoint(float("nan"))[0] == "missing_endpoint"
    assert canonical_efflux_endpoint(None)[0] == "missing_endpoint"
    assert canonical_influx_endpoint("missing_endpoint")[0] == "missing_endpoint"


def test_compact_direction_notation_does_not_match_letters_inside_words():
    assert canonical_passive_endpoint("apparent permeability")[0] == "apparent_permeability"
    assert canonical_passive_endpoint("Papp cassette incubation")[0] == "apparent_permeability"
    assert canonical_passive_endpoint("PappAB")[0] == "papp_a_to_b"
    assert canonical_passive_endpoint("PappBA")[0] == "papp_b_to_a"
    assert canonical_passive_endpoint("Papp (apical-to-basolateral)")[0] == "papp_a_to_b"
    for value in ("Papp,AP-BL", "Papp AP→BL", "PappAP-BL"):
        assert canonical_passive_endpoint(value)[0] == "papp_a_to_b"
    for value in ("Papp, BL-AP", "Papp BL→AP"):
        assert canonical_passive_endpoint(value)[0] == "papp_b_to_a"


def test_concentration_is_not_mistaken_for_a_ratio_and_log_pe_stays_permeability():
    for value in ("concentration", "Acceptor concentration", "final nominal concentration"):
        assert canonical_passive_endpoint(value)[0] == "concentration"
    for value in ("-logPe", "-logPₑ", "-log Pe (cm s⁻¹)"):
        assert canonical_passive_endpoint(value)[0] == "negative_log_effective_permeability"
    assert canonical_passive_endpoint("-logP")[0] == "negative_log_partition_coefficient"
    assert canonical_passive_endpoint("logP0PAMPA-BBB")[0] == "log_partition_coefficient"
    assert canonical_passive_endpoint("Permeability Log[10-6 cm/s]")[0] == "log_passive_permeability"
    assert canonical_passive_endpoint(
        "fold increase in Cu concentration relative to vehicle control"
    )[0] == "concentration_fold_change"


def test_direct_mapping_loader_requires_explicit_human_approval(tmp_path, monkeypatch):
    source = tmp_path / "records.parquet"
    pd.DataFrame({"quant_metric": ["Log BB"]}).to_parquet(source, index=False)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(starling_endpoint_normalization, "DIRECT_SOURCE_PATH", source)
    monkeypatch.setattr(starling_endpoint_normalization, "EXPECTED_DIRECT_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(starling_endpoint_normalization, "EXPECTED_DIRECT_INVENTORY_COUNT", 1)
    monkeypatch.setattr(starling_endpoint_normalization, "EXPECTED_DIRECT_INVENTORY_SHA256", "test-inventory")
    monkeypatch.setattr(starling_endpoint_normalization, "_expected_direct_endpoint_keys", lambda: {"Log BB"})
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "mapping_version": DIRECT_ENDPOINT_MAPPING_VERSION,
                "approval": {"human_approved": False},
                "source": {
                    "sha256": source_sha,
                    "inventory_count_including_missing": 1,
                    "inventory_sha256": "test-inventory",
                },
                "mapping": {"Log BB": "logbb"},
            }
        ),
        encoding="utf-8",
    )
    try:
        EndpointNormalizer(path)
    except ValueError as exc:
        assert "not human approved" in str(exc)
    else:
        raise AssertionError("unapproved Direct mapping was accepted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["approval"] = {
        "human_approved": True,
        "approved_by": "reviewer",
        "approved_at": "2026-08-03T00:00:00Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    decision = EndpointNormalizer(path).decision("direct_bbb", "Log BB")
    assert decision.spacing_and_spelling_endpoint == "logbb"
    assert decision.reason == "direct_approved_global_mapping"


def _resolve(measurement, unit, endpoint, *, endpoint_text=None):
    record = {
        "source_id": "passive_permeability",
        "measurement_text": measurement,
        "unit_text": unit,
        "endpoint_name": endpoint_text or endpoint,
    }
    pair = normalize_measurement_and_unit(measurement, unit)
    return source_measurement_resolver(record, endpoint, pair)


def test_unit_resolution_precedence_explicit_embedded_and_reviewed_default():
    explicit = _resolve("2.5", "×10^-6 cm/s", "apparent_permeability")
    embedded = _resolve("2.5 ×10^-6 cm/s", None, "apparent_permeability")
    endpoint_embedded = _resolve(
        "2.5", None, "apparent_permeability", endpoint_text="Papp (×10^-6 cm/s)"
    )
    defaulted = _resolve("2.5", None, "apparent_permeability")
    assert explicit.canonical_measurement == embedded.canonical_measurement == "0.0000025"
    assert endpoint_embedded.canonical_measurement == "0.0000025"
    assert defaulted.canonical_measurement == "2.5"
    assert {explicit.canonical_unit, embedded.canonical_unit, endpoint_embedded.canonical_unit} == {"cm/s"}
    assert defaulted.canonical_unit is None
    assert embedded.status.startswith("embedded_measurement_unit")
    assert endpoint_embedded.status.startswith("embedded_endpoint_unit")
    assert defaulted.status != "reviewed_endpoint_default_unit"


def test_missing_unit_physical_permeability_never_guesses_a_scale():
    for endpoint, value in (
        ("apparent_permeability", "177"),
        ("effective_permeability", "160"),
        ("passive_permeability", "2422"),
    ):
        resolved = _resolve(value, None, endpoint)
        assert resolved.canonical_measurement == value
        assert resolved.canonical_unit is None
    assert canonical_passive_endpoint("Kintr")[0] == "intrinsic_partition_coefficient"


def test_percent_default_and_fraction_standardization_do_not_raise():
    defaulted = _resolve("20", None, "percent_transport")
    fractional = _resolve("0.2", "fraction", "percent_transport")
    assert (defaulted.canonical_measurement, defaulted.canonical_unit) == ("20", "%")
    assert (fractional.canonical_measurement, fractional.canonical_unit) == ("20", "%")


def test_explicit_log_permeability_unit_is_treated_as_the_log_basis():
    positive_log = _resolve("-5.2", "cm/s", "log_effective_permeability")
    negative_log = _resolve("5.2", "cm/s", "negative_log_effective_permeability")
    assert (positive_log.canonical_measurement, positive_log.canonical_unit) == (
        "-5.2",
        "log10(cm/s)",
    )
    assert (negative_log.canonical_measurement, negative_log.canonical_unit) == (
        "5.2",
        "-log10(cm/s)",
    )
    embedded_positive = _resolve(
        "-5.2 ×10^-6 cm/s", None, "log_effective_permeability"
    )
    embedded_negative = _resolve(
        "5.2 ×10^-6 cm/s", None, "negative_log_effective_permeability"
    )
    assert (embedded_positive.canonical_measurement, embedded_positive.canonical_unit) == (
        "-5.2",
        "log10(cm/s)",
    )
    assert (embedded_negative.canonical_measurement, embedded_negative.canonical_unit) == (
        "5.2",
        "-log10(cm/s)",
    )
    negative_explicit_log = _resolve(
        "4.6", "log(cm/s)", "negative_log_effective_permeability"
    )
    scaled_explicit_log = _resolve(
        "-4.7", "[10^-6 cm/s]", "log_apparent_permeability"
    )
    endpoint_basis = _resolve(
        "-4.9",
        None,
        "log_passive_permeability",
        endpoint_text="Permeability Log[10-6 cm/s]",
    )
    assert (negative_explicit_log.canonical_measurement, negative_explicit_log.canonical_unit) == (
        "4.6",
        "-log10(cm/s)",
    )
    assert (scaled_explicit_log.canonical_measurement, scaled_explicit_log.canonical_unit) == (
        "-4.7",
        "log10(cm/s)",
    )
    assert (endpoint_basis.canonical_measurement, endpoint_basis.canonical_unit) == (
        "-4.9",
        "log10(cm/s)",
    )
    incompatible_basis = _resolve(
        "-5.2", "nm/s", "log_effective_permeability"
    )
    assert incompatible_basis.canonical_unit != "log10(cm/s)"
    missing_log_value = _resolve(
        None, "cm/s", "negative_log_effective_permeability"
    )
    assert missing_log_value.canonical_measurement is None


def test_unreviewed_direct_concentration_does_not_infer_ratio():
    record = {
        "source_id": "direct_bbb",
        "measurement_text": "0.2",
        "unit_text": None,
        "endpoint_name": "brain concentration",
    }
    baseline = normalize_measurement_and_unit("0.2", None)
    resolved = source_measurement_resolver(record, "brain_concentration", baseline)
    assert resolved.canonical_unit is None


def test_explicit_per_100_basis_is_scaled_without_touching_numerator_times_100():
    """A per-100 basis divides, wherever the 100 sits relative to the inverted unit.

    This used to be a BBB-local rewrite (`_fold_explicit_denominator_100_basis`). It is now
    a shared-parser rule, because a basis count means the same thing in every task -- bio and
    skin were silently reporting these 100x high while BBB was correct.
    """
    for unit in ("mL/100 g/min", "mL/100g/min", "mL·100 g^-1·min^-1"):
        resolved = _resolve("25 ± 5", unit, "influx_rate_constant")
        assert resolved.canonical_measurement == "0.25 ± 0.05"
        assert resolved.canonical_unit == "mL/g·min"

    # A trailing multiplier with no basis to attach to is genuinely ambiguous -- a per-100
    # basis or a reported-value multiplier -- so it must fail closed rather than be guessed.
    # What matters is that the value is never rescaled, whichever gate rejects it.
    untouched = _resolve("25", "mL/g/min ×100", "influx_rate_constant")
    assert untouched.canonical_measurement == "25"
    assert untouched.status in {
        "cleaned_only_unrecognized_unit",
        "incompatible_endpoint_unit",
    }
