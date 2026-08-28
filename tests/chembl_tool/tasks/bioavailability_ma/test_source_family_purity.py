from tools.chembl_tool.tasks.bioavailability_ma.source_family_purity import (
    direct_like_bioavailability_reason,
)


def test_numeric_overall_f_moves_from_mechanism_to_direct_surface():
    record = {
        "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
        "canonical_endpoint_name": "fg_substrate_outcome:source_token:cnt1",
        "canonical_measurement_text": "1",
        "support_text": (
            "The median oral bioavailability was 44.8% and genotype values "
            "were 42.0%, 41.4%, and 62.4%."
        ),
    }
    assert direct_like_bioavailability_reason(record) == (
        "support_reports_overall_bioavailability_percent"
    )


def test_relative_f_is_direct_like_for_distance_but_not_relabelled_as_gold():
    record = {
        "group_id": "Observed.nondirect_oral_bioavailability",
        "canonical_endpoint_name": "oral_bioavailability",
        "canonical_measurement_text": "12.7",
        "canonical_unit_text": "%",
        "canonical_bioavailability_evidence_scope": "nondirect",
        "support_text": "Relative bioavailability (F) was 12.7% versus injection.",
    }
    assert direct_like_bioavailability_reason(record) == (
        "structured_overall_bioavailability_value"
    )


def test_relative_auc_change_and_transporter_no_effect_are_not_overall_f_values():
    auc_change = {
        "group_id": "Observed.nondirect_oral_bioavailability",
        "canonical_endpoint_name": "oral_bioavailability",
        "canonical_measurement_text": "increased in sick rats",
        "support_text": (
            "Bioavailability was increased in sick rats, with AUClast being "
            "64.75% higher than in controls."
        ),
    }
    no_effect = {
        "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
        "canonical_endpoint_name": "oral_exposure_change_due_to_gut_wall",
        "canonical_measurement_text": "no systematic effect on F",
        "support_text": "ABCG2 genotype had no systematic effect on the F value.",
    }
    assert direct_like_bioavailability_reason(auc_change) == ""
    assert direct_like_bioavailability_reason(no_effect) == ""


def test_unitless_fractional_f_moves_but_fold_change_does_not():
    fractional_f = {
        "group_id": "Observed.nondirect_oral_bioavailability",
        "canonical_endpoint_name": "oral_bioavailability",
        "canonical_measurement_text": "0.84",
        "canonical_unit_text": "unspecified",
        "support_text": "The bioavailability (F) of alprazolam was 0.84.",
    }
    relative_fold = {
        "group_id": "Observed.nondirect_oral_bioavailability",
        "canonical_endpoint_name": "oral_bioavailability",
        "canonical_measurement_text": "3.25-fold increase",
        "canonical_unit_text": "unspecified",
        "support_text": "A 3.25-fold increase in relative bioavailability was observed.",
    }

    assert direct_like_bioavailability_reason(fractional_f)
    assert direct_like_bioavailability_reason(relative_fold) == ""
