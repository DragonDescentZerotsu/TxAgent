import math
from copy import deepcopy

import pandas as pd
import pytest

from tools.chembl_tool.tasks.bioavailability_ma.build_starling_endpoint_policy_assignments_v2 import (
    ASSIGNMENTS_FILENAME,
    METADATA_FILENAME,
    REGISTRY_FILENAME,
    build_endpoint_policy_assignments_v2,
)
from tools.chembl_tool.tasks.bioavailability_ma.audit_starling_endpoint_policy_v2_calibration import (
    aggregate_pair_yields,
)
from tools.chembl_tool.tasks.bioavailability_ma.audit_starling_pair_bucket_distributions import (
    AUDIT_FILENAME,
    audit_distributions,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_endpoint_policies_v2 import (
    POLICY_FAMILIES,
    THRESHOLD_PROFILES,
    THRESHOLD_PROFILE_VERSION,
    assign_endpoint_policies_v2,
    classify_measurement_subtype,
    format_normalized_policy_distance,
    label_distance,
    normalize_policy_distance,
    policy_distance,
    policy_distance_result,
    resolve_threshold_profile,
)


def _record(
    number,
    *,
    endpoint="auc0_inf",
    unit="ng·h/mL",
    validity="valid",
    **extra,
):
    return {
        "normalized_record_id": f"record-{number}",
        "source_id": "oral_exposure",
        "canonical_endpoint": endpoint,
        "canonical_unit": unit,
        "canonical_bioavailability_report_type": "__unknown__",
        "normalization_validity_status": validity,
        **extra,
    }


@pytest.mark.parametrize(
    ("record", "subtype", "family"),
    [
        (_record(1), "auc_exposure", "standard_log_metric"),
        (_record(2, endpoint="tmax", unit="h"), "tmax", "ratio_or_tmax"),
        (
            _record(3, endpoint="metabolic_half_life", unit="h"),
            "explicit_half_life",
            "standard_log_metric",
        ),
        (
            _record(4, endpoint="hepatic_clearance", unit="mL/min/kg"),
            "hepatic_clearance",
            "standard_log_metric",
        ),
        (
            _record(5, endpoint="intrinsic_clearance", unit="µL/min/mg protein"),
            "intrinsic_or_metabolic_clearance",
            "wide_log_metric",
        ),
        (
            _record(6, endpoint="fraction_absorbed", unit="%"),
            "absorption_percentage",
            "bounded_percentage",
        ),
        (
            _record(7, endpoint="fraction_absorbed", unit="fraction"),
            "absorption_fraction",
            "bounded_fraction",
        ),
    ],
)
def test_explicit_subtype_precedence(record, subtype, family):
    assert classify_measurement_subtype(record) == (subtype, family, "assigned")


def test_tmax_and_half_life_use_log_ratio_not_absolute_hours():
    assignments, registry, _ = assign_endpoint_policies_v2(
        [
            _record(1, endpoint="tmax", unit="h"),
            _record(2, endpoint="metabolic_half_life", unit="h"),
        ]
    )
    policies = registry["endpoint_policies"]
    assert all(
        policies[row["endpoint_policy_key"]]["distance"] == "absolute_log10_ratio"
        for row in assignments
    )
    assert policy_distance(policies[assignments[0]["endpoint_policy_key"]], 2, 3) == (
        pytest.approx(math.log10(1.5))
    )


def test_reviewed_log_permeability_uses_pretransformed_coordinate_policy():
    record = _record(
        1,
        endpoint="intestinal_effective_permeability",
        unit="log(cm/s)",
        extra_details="human intestinal Peff",
        unit_notation_status="none",
    )
    assert classify_measurement_subtype(record) == (
        "peff_unspecified__log10_coordinate",
        "pretransformed_standard_log10_metric",
        "assigned",
    )
    assignments, registry, _ = assign_endpoint_policies_v2([record])
    policy = registry["endpoint_policies"][assignments[0]["endpoint_policy_key"]]
    assert policy_distance(policy, -3.35, -4.89) == pytest.approx(1.54)
    result = policy_distance_result(policy, -3.35, -4.89)
    assert result["normalized_distance_0_100"] == 100.0
    assert result["normalization_saturated"] is True


def test_scaled_log_reference_is_valid_evidence_but_not_policy_assigned():
    record = _record(
        1,
        endpoint="intestinal_effective_permeability",
        unit="log(cm/s)",
        unit_notation_status="unambiguous_scientific_notation",
    )
    assert classify_measurement_subtype(record)[2] == (
        "unresolved_measurement_subtype"
    )


def test_unreviewed_transformed_metric_does_not_receive_raw_positive_policy():
    record = _record(
        1,
        endpoint="solubility",
        unit="log(µM)",
        assay_system="aqueous solubility measurement",
    )
    assert classify_measurement_subtype(record) == (
        None,
        None,
        "unresolved_measurement_subtype",
    )


def test_required_context_gates_and_directional_permeability():
    missing_time = _record(1, endpoint="dissolution", unit="%")
    resolved_time = {
        **missing_time,
        "extra_details": "Measured after 30 minutes.",
    }
    directional = _record(
        2,
        endpoint="caco2_mdck_pampa_permeability",
        unit="cm/s",
        extra_details="Direction: apical to basolateral (A-B).",
    )
    assert classify_measurement_subtype(missing_time)[2] == (
        "missing_required_comparison_context"
    )
    assert classify_measurement_subtype(resolved_time) == (
        "matched_time_dissolution_percentage__time_1800s",
        "bounded_percentage",
        "assigned",
    )
    assert classify_measurement_subtype(directional) == (
        "papp_apical_to_basolateral",
        "standard_log_metric",
        "assigned",
    )


@pytest.mark.parametrize(
    ("notation", "expected_direction"),
    [
        ("AP-BL", "apical_to_basolateral"),
        ("AP‑BL", "apical_to_basolateral"),
        ("AP→BL", "apical_to_basolateral"),
        ("AP -> BL", "apical_to_basolateral"),
        ("AP to BL", "apical_to_basolateral"),
        ("AP-to-BL", "apical_to_basolateral"),
        ("BL-AP", "basolateral_to_apical"),
        ("BL‑AP", "basolateral_to_apical"),
        ("BL→AP", "basolateral_to_apical"),
        ("BL -> AP", "basolateral_to_apical"),
        ("BL to AP", "basolateral_to_apical"),
        ("BL-to-AP", "basolateral_to_apical"),
    ],
)
def test_ap_bl_direction_abbreviations_are_canonicalized(
    notation, expected_direction
):
    record = _record(
        1,
        endpoint="caco2_mdck_pampa_permeability",
        unit="cm/s",
        extra_details=f"Direction: {notation}.",
    )
    assert classify_measurement_subtype(record) == (
        f"papp_{expected_direction}",
        "standard_log_metric",
        "assigned",
    )


def test_ap_bl_direction_is_applied_to_pretransformed_permeability():
    record = _record(
        1,
        endpoint="caco2_mdck_pampa_permeability",
        unit="log(cm/s)",
        unit_notation_status="none",
        extra_details="Direction: AP-BL.",
    )
    assert classify_measurement_subtype(record) == (
        "papp_apical_to_basolateral__log10_coordinate",
        "pretransformed_standard_log10_metric",
        "assigned",
    )


def test_mixed_ap_bl_directions_are_not_assigned_a_scalar_transfer_policy():
    record = _record(
        1,
        endpoint="caco2_mdck_pampa_permeability",
        unit="cm/s",
        extra_details="AP-BL versus BL-AP permeability.",
    )
    assert classify_measurement_subtype(record) == (
        None,
        None,
        "mixed_measurement_semantics",
    )


def test_opposing_permeability_directions_receive_distinct_policy_keys():
    records = [
        _record(
            1,
            endpoint="caco2_mdck_pampa_permeability",
            unit="cm/s",
            extra_details="Direction: AP-BL.",
        ),
        _record(
            2,
            endpoint="caco2_mdck_pampa_permeability",
            unit="cm/s",
            extra_details="Direction: BL-AP.",
        ),
    ]
    assignments, _, _ = assign_endpoint_policies_v2(records)
    assert all(row["policy_assignment_status"] == "assigned" for row in assignments)
    assert assignments[0]["endpoint_policy_key"] != assignments[1]["endpoint_policy_key"]


def test_endpoint_and_subtype_profiles_resolve_with_one_precedence_path():
    profile_key, _, resolution = resolve_threshold_profile(
        "cmax", "plasma_concentration_exposure", "standard_log_metric"
    )
    assert (profile_key, resolution) == (
        "cmax_exposure",
        "endpoint_subtype_override",
    )
    profile_key, _, resolution = resolve_threshold_profile(
        "caco2_mdck_pampa_permeability",
        "papp_apical_to_basolateral",
        "standard_log_metric",
    )
    assert (profile_key, resolution) == (
        "permeability",
        "subtype_prefix_override",
    )


def test_permeability_resolves_to_wide_thresholds_and_own_anchor():
    record = _record(
        1,
        endpoint="caco2_mdck_pampa_permeability",
        unit="cm/s",
        extra_details="Direction: AP-BL.",
    )
    assignments, registry, _ = assign_endpoint_policies_v2([record])
    policy = registry["endpoint_policies"][assignments[0]["endpoint_policy_key"]]
    assert policy["classified_policy_family"] == "standard_log_metric"
    assert policy["policy_family"] == "wide_log_metric"
    assert policy["threshold_profile"] == "permeability"
    assert policy["threshold_profile_version"] == THRESHOLD_PROFILE_VERSION
    assert policy["thresholds"]["primary"] == (
        POLICY_FAMILIES["wide_log_metric"]["thresholds"]["primary"]
    )
    assert policy["normalization"]["raw_distance_anchor"] == pytest.approx(
        math.log10(20.0)
    )
    result = policy_distance_result(policy, 1.0, 3.0)
    assert result["normalized_distance_0_100"] == pytest.approx(
        100.0 * math.log10(3.0) / math.log10(20.0)
    )
    assert result["raw_distance_label"] == "transfer"


def test_pretransformed_permeability_uses_wide_thresholds_without_retransform():
    record = _record(
        1,
        endpoint="caco2_mdck_pampa_permeability",
        unit="log(cm/s)",
        unit_notation_status="none",
        extra_details="Direction: AP-BL.",
    )
    assignments, registry, _ = assign_endpoint_policies_v2([record])
    policy = registry["endpoint_policies"][assignments[0]["endpoint_policy_key"]]
    assert policy["threshold_profile"] == "pretransformed_permeability"
    assert policy["transform"] == "identity"
    assert policy["input_domain"] == "finite"
    assert policy["thresholds"]["permissive"]["not_transfer_min"] == (
        pytest.approx(math.log10(20.0))
    )
    assert policy_distance(policy, -3.0, -4.0) == pytest.approx(1.0)


def test_matched_time_measurements_receive_distinct_subtypes():
    fifteen = _record(
        1,
        endpoint="dissolution",
        unit="%",
        extra_details="Measurement at 15 min time point.",
    )
    thirty = {
        **fifteen,
        "normalized_record_id": "record-2",
        "extra_details": "Measurement at 30 min time point.",
    }
    assignments, _, _ = assign_endpoint_policies_v2([fifteen, thirty])
    assert assignments[0]["measurement_subtype"].endswith("time_900s")
    assert assignments[1]["measurement_subtype"].endswith("time_1800s")
    assert assignments[0]["endpoint_policy_key"] != assignments[1]["endpoint_policy_key"]


def test_solubility_modes_and_generic_scalar_rejection():
    equilibrium = _record(
        1,
        endpoint="solubility",
        unit="mg/mL",
        assay_system="aqueous solubility measurement",
    )
    kinetic = _record(
        2,
        endpoint="solubility",
        unit="µM",
        extra_details="Kinetic solubility from a supersaturation assay.",
    )
    generic = _record(3, endpoint="unknown_endpoint", unit="mg/mL")
    assert classify_measurement_subtype(equilibrium)[:2] == (
        "equilibrium_solubility",
        "wide_log_metric",
    )
    assert classify_measurement_subtype(kinetic)[:2] == (
        "kinetic_solubility",
        "wide_log_metric",
    )
    assert classify_measurement_subtype(generic) == (
        None,
        None,
        "unresolved_measurement_subtype",
    )


def test_mixed_kinetic_parameter_is_rejected_and_vmax_requires_basis():
    mixed = _record(
        1,
        endpoint="cyp_metabolism",
        unit="µL/min/mg",
        endpoint_name="Vmax/Km",
    )
    vmax_missing_basis = _record(
        2,
        endpoint="cyp_metabolism",
        unit="µM",
        endpoint_name="Vmax",
    )
    assert classify_measurement_subtype(mixed)[2] == "mixed_measurement_semantics"
    assert classify_measurement_subtype(vmax_missing_basis)[2] == (
        "missing_required_comparison_context"
    )


@pytest.mark.parametrize(
    "record",
    [
        _record(1, endpoint="bioavailability", unit="fold"),
        _record(2, endpoint="metabolic_half_life", unit="%"),
        _record(3, endpoint="tmax", unit="fold"),
    ],
)
def test_v1_unsupported_semantic_boundary_remains_rejected(record):
    assert classify_measurement_subtype(record) == (
        None,
        None,
        "unresolved_measurement_subtype",
    )


@pytest.mark.parametrize("family", POLICY_FAMILIES.values())
def test_exact_boundaries_and_deadbands_are_frozen(family):
    primary = family["thresholds"]["primary"]
    assert label_distance(family, primary["transfer_max"]) == "transfer"
    assert label_distance(family, primary["not_transfer_min"]) == "not_transfer"
    midpoint = (primary["transfer_max"] + primary["not_transfer_min"]) / 2
    assert label_distance(family, midpoint) is None


def test_log_transform_rejects_nonpositive_values():
    policy = POLICY_FAMILIES["standard_log_metric"]
    assert policy_distance(policy, 2.0, 10.0) == pytest.approx(math.log10(5))
    with pytest.raises(ValueError, match="positive"):
        policy_distance(policy, 0.0, 10.0)


@pytest.mark.parametrize(
    ("family_key", "raw_distances", "expected"),
    [
        ("bounded_percentage", [10.0, 30.0, 40.0], [25.0, 75.0, 100.0]),
        ("bounded_fraction", [0.10, 0.30, 0.40], [25.0, 75.0, 100.0]),
        (
            "standard_log_metric",
            [math.log10(2.0), math.log10(5.0), math.log10(10.0)],
            [100.0 * math.log10(2.0), 100.0 * math.log10(5.0), 100.0],
        ),
    ],
)
def test_exact_family_distance_normalization_examples(
    family_key, raw_distances, expected
):
    policy = POLICY_FAMILIES[family_key]
    assert [
        normalize_policy_distance(policy, distance) for distance in raw_distances
    ] == pytest.approx(expected)


@pytest.mark.parametrize("family", POLICY_FAMILIES.values())
def test_normalization_bounds_monotonicity_anchor_and_saturation(family):
    anchor = family["normalization"]["raw_distance_anchor"]
    distances = [0.0, anchor / 4.0, anchor, anchor * 2.0]
    normalized = [
        normalize_policy_distance(family, distance) for distance in distances
    ]
    assert normalized[0] == 0.0
    assert normalized[-2:] == [100.0, 100.0]
    assert normalized == sorted(normalized)
    assert all(0.0 <= value <= 100.0 for value in normalized)

    exact = policy_distance_result(family, 1.0, 1.0)
    assert exact["normalized_distance_0_100"] == 0.0
    assert exact["normalization_saturated"] is False


@pytest.mark.parametrize("family", POLICY_FAMILIES.values())
def test_all_raw_thresholds_have_registered_normalized_cutoffs(family):
    for variant, thresholds in family["thresholds"].items():
        normalized_thresholds = family["normalization"]["normalized_thresholds"][
            variant
        ]
        assert normalize_policy_distance(
            family, thresholds["transfer_max"]
        ) == pytest.approx(normalized_thresholds["transfer_max"])
        assert normalize_policy_distance(
            family, thresholds["not_transfer_min"]
        ) == pytest.approx(normalized_thresholds["not_transfer_min"])


def test_published_normalized_threshold_cutoffs_before_display_rounding():
    expected = {
        "bounded_percentage": [(12.50, 50.00), (25.00, 75.00), (37.50, 100.00)],
        "bounded_fraction": [(12.50, 50.00), (25.00, 75.00), (37.50, 100.00)],
        "ratio_or_tmax": [(13.86, 43.07), (25.19, 68.26), (43.07, 100.00)],
        "standard_log_metric": [
            (17.61, 47.71),
            (30.10, 69.90),
            (47.71, 100.00),
        ],
        "wide_log_metric": [(23.14, 53.72), (36.67, 76.86), (53.72, 100.00)],
    }
    for family_key, cutoff_pairs in expected.items():
        policy = POLICY_FAMILIES[family_key]
        for variant, pair in zip(
            ("strict", "primary", "permissive"), cutoff_pairs
        ):
            thresholds = policy["normalization"]["normalized_thresholds"][variant]
            assert round(thresholds["transfer_max"], 2) == pair[0]
            assert round(thresholds["not_transfer_min"], 2) == pair[1]


def test_distance_result_is_symmetric_variant_invariant_and_labels_raw_distance():
    policy = POLICY_FAMILIES["standard_log_metric"]
    results = [
        policy_distance_result(policy, 2.0, 10.0, variant=variant)
        for variant in ("strict", "primary", "permissive")
    ]
    reverse = policy_distance_result(policy, 10.0, 2.0)
    assert {row["normalized_distance_0_100"] for row in results} == {
        results[0]["normalized_distance_0_100"]
    }
    assert reverse["raw_distance"] == pytest.approx(results[0]["raw_distance"])
    assert reverse["normalized_distance_0_100"] == pytest.approx(
        results[0]["normalized_distance_0_100"]
    )
    assert results[0]["raw_distance_label"] == label_distance(
        policy, results[0]["raw_distance"], variant="strict"
    )
    assert results[0]["normalization_version"]
    assert results[0]["normalization_provenance"]["raw_distance_anchor"] == 1.0


@pytest.mark.parametrize("bad_distance", [-1.0, math.inf, -math.inf, math.nan])
def test_normalization_and_labeling_reject_invalid_raw_distances(bad_distance):
    policy = POLICY_FAMILIES["standard_log_metric"]
    with pytest.raises(ValueError):
        normalize_policy_distance(policy, bad_distance)
    with pytest.raises(ValueError):
        label_distance(policy, bad_distance)


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan])
def test_policy_distance_rejects_nonfinite_measurements(bad_value):
    with pytest.raises(ValueError, match="finite"):
        policy_distance(POLICY_FAMILIES["standard_log_metric"], bad_value, 1.0)


@pytest.mark.parametrize(
    "family_key", ["bounded_percentage", "bounded_fraction", "standard_log_metric"]
)
def test_policy_distance_result_rejects_negative_measurements(family_key):
    with pytest.raises(ValueError, match="nonnegative|positive"):
        policy_distance_result(POLICY_FAMILIES[family_key], -1.0, 1.0)


@pytest.mark.parametrize("bad_anchor", [0.0, -1.0, math.inf, math.nan])
def test_normalization_rejects_invalid_anchors(bad_anchor):
    policy = deepcopy(POLICY_FAMILIES["standard_log_metric"])
    policy["normalization"]["raw_distance_anchor"] = bad_anchor
    with pytest.raises(ValueError, match="anchor"):
        normalize_policy_distance(policy, 0.5)


def test_full_precision_storage_is_separate_from_two_decimal_presentation():
    policy = POLICY_FAMILIES["standard_log_metric"]
    result = policy_distance_result(policy, 1.0, 2.0)
    assert result["normalized_distance_0_100"] == pytest.approx(
        100.0 * math.log10(2.0)
    )
    assert result["normalized_distance_0_100"] != 30.10
    assert (
        format_normalized_policy_distance(
            result["normalized_distance_0_100"], policy
        )
        == "30.10"
    )


@pytest.mark.parametrize("family", POLICY_FAMILIES.values())
def test_label_parity_at_boundaries_and_deadbands(family):
    for variant, thresholds in family["thresholds"].items():
        points = [
            0.0,
            thresholds["transfer_max"],
            (thresholds["transfer_max"] + thresholds["not_transfer_min"]) / 2.0,
            thresholds["not_transfer_min"],
            family["normalization"]["raw_distance_anchor"] * 2.0,
        ]
        before = [label_distance(family, point, variant=variant) for point in points]
        _ = [normalize_policy_distance(family, point) for point in points]
        after = [label_distance(family, point, variant=variant) for point in points]
        assert before == after


def test_v2_assignment_contract_and_standalone_builder(tmp_path):
    records = [
        _record(1),
        _record(2, endpoint="unknown", unit="mg/mL"),
        _record(3, validity="non_scalar_measurement"),
    ]
    assignments, registry, audit = assign_endpoint_policies_v2(records)
    assert list(assignments[0]) == [
        "normalized_record_id",
        "measurement_subtype",
        "endpoint_policy_key",
        "policy_assignment_status",
    ]
    assert assignments[0]["endpoint_policy_key"] == (
        "bioavailability_ma/auc0_inf/auc_exposure/v2"
    )
    assert assignments[1]["endpoint_policy_key"] is None
    assert assignments[2]["measurement_subtype"] is None
    assert all(audit["validations"].values())
    assert registry["scope"]["benchmark_outcome_selection"] is False
    assert registry["distance_normalization"]["invariant_across_threshold_variants"]
    assert registry["threshold_profile_version"] == THRESHOLD_PROFILE_VERSION
    assert registry["threshold_profiles"] == THRESHOLD_PROFILES
    endpoint_policy = registry["endpoint_policies"][
        assignments[0]["endpoint_policy_key"]
    ]
    assert endpoint_policy["threshold_profile"] == "auc_exposure"
    assert endpoint_policy["normalization"]["raw_distance_anchor"] == pytest.approx(
        math.log10(10.0)
    )

    records_path = tmp_path / "records.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    out_dir = tmp_path / "v2"
    metadata = build_endpoint_policy_assignments_v2(
        records_path=records_path,
        out_dir=out_dir,
    )
    assert {path.name for path in out_dir.iterdir()} == {
        ASSIGNMENTS_FILENAME,
        REGISTRY_FILENAME,
        METADATA_FILENAME,
    }
    assert metadata["stats"]["assignment_records"] == len(records)


def test_v2_distribution_audit_emits_normalized_diagnostics(tmp_path):
    records = [
        {
            **_record(1),
            "canonical_smiles": "CCO",
            "finite_scalar_value": 1.0,
        },
        {
            **_record(2),
            "canonical_smiles": "CCN",
            "finite_scalar_value": 10.0,
        },
    ]
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)
    sidecar_path = tmp_path / "sidecar.parquet"
    pd.DataFrame(
        [
            {
                "normalized_record_id": record["normalized_record_id"],
                "pair_bucket_key": "bucket",
                "bucket_eligible": True,
                "source_id": record["source_id"],
                "canonical_endpoint": record["canonical_endpoint"],
                "canonical_unit": record["canonical_unit"],
            }
            for record in records
        ]
    ).to_parquet(sidecar_path, index=False)
    policy_dir = tmp_path / "v2"
    build_endpoint_policy_assignments_v2(
        records_path=records_path, out_dir=policy_dir
    )
    analysis_dir = tmp_path / "analysis"
    summary = audit_distributions(
        records_path=records_path,
        sidecar_path=sidecar_path,
        assignments_path=policy_dir / ASSIGNMENTS_FILENAME,
        registry_path=policy_dir / REGISTRY_FILENAME,
        out_dir=analysis_dir,
        policy_version="v2",
    )
    audit = pd.read_parquet(analysis_dir / AUDIT_FILENAME)
    assert audit.loc[0, "normalized_comparison_space_robust_span_0_100"] == (
        pytest.approx(90.0)
    )
    assert audit.loc[0, "normalization_saturated"] == False  # noqa: E712
    assert audit.loc[0, "primary_normalized_transfer_max_0_100"] == pytest.approx(
        100.0 * math.log10(2.0)
    )
    assert summary["distance_normalization"]["saturated_bucket_count"] == 0
    assert summary["distance_normalization"]["threshold_profile_summaries"]


def test_aggregate_pair_yields_excludes_same_parent_and_keeps_boundaries():
    # A-A is excluded. A-B distances are 1, 9, 20, 30 and B-B is excluded.
    counts = aggregate_pair_yields(
        values=pd.Series([0.0, 10.0, 1.0, 30.0]).to_numpy(),
        parent_keys=pd.Series(["A", "A", "B", "B"]).to_numpy(),
        transfer_max=1.0,
        not_transfer_min=20.0,
    )
    assert counts == {
        "cross_parent_pairs": 4,
        "transfer_pairs": 1,
        "deadband_pairs": 1,
        "not_transfer_pairs": 2,
    }
