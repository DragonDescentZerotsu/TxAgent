import pytest

from tools.chembl_tool.common.starling.build_bbb_experimental_meaningful_cns_access import (
    DEFAULT_OUTPUT_ROOT,
    _parse_args,
    _validate_build_scope,
)
from tools.chembl_tool.tasks.bbb_martins.experimental_meaningful_cns_access_benchmark import (
    SOURCE_REVISION,
    classify_scope,
    label_record,
)


def test_builder_rejects_revision_drift_and_partial_canonical_output() -> None:
    revision_drift = _parse_args(["--bbb-revision", "different"])
    with pytest.raises(ValueError, match="frozen BBB source revision"):
        _validate_build_scope(revision_drift)

    partial = _parse_args(["--max-rows", "10"])
    assert partial.bbb_revision == SOURCE_REVISION
    assert partial.output_root == str(DEFAULT_OUTPUT_ROOT)
    with pytest.raises(ValueError, match="non-canonical"):
        _validate_build_scope(partial)

    smoke = _parse_args(
        ["--max-rows", "10", "--output-root", "outputs/smoke/bbb-gold"]
    )
    _validate_build_scope(smoke)


def test_accepts_measured_brain_exposure() -> None:
    record = {
        "bbb_permeability_label": "high_permeability",
        "quant_metric": "Kp,brain",
        "quant_value": "0.95",
        "assay_model": "in-vivo pharmacokinetic study",
        "support_text": "Brain exposure was measured after intravenous dosing.",
    }
    label, method = label_record(record)
    assert label == 1
    assert ":brain_systemic_ratio:" in method


def test_accepts_measured_csf_restriction_as_proxy_outcome() -> None:
    record = {
        "bbb_permeability_label": "poor_penetration",
        "quant_metric": "CSF/plasma ratio",
        "quant_value": "0.04 to 0.08",
        "assay_model": "CSF sampling",
        "support_text": "The CSF/plasma ratio was measured after an oral dose.",
    }
    label, method = label_record(record)
    assert label == 0
    assert ":csf:" in method


def test_low_but_nonzero_exposure_can_remain_restricted() -> None:
    label, method = label_record(
        {
            "bbb_permeability_label": "poor_penetration",
            "quant_metric": "brain/plasma ratio",
            "quant_value": "0.01",
            "assay_model": "in vivo pharmacokinetic study",
            "support_text": "A low ratio was measured after systemic dosing.",
        }
    )
    assert label == 0
    assert ":brain_systemic_ratio:" in method


def test_rejects_computational_logbb() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "logBB",
            "quant_value": "0.75",
            "assay_model": "Clark's equation (in silico)",
            "support_text": "The calculated logBB predicts brain distribution.",
        }
    )
    assert label is None
    assert reason == "computational_or_predicted_result"


def test_rejects_calculated_logbb_without_prediction_keyword() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "logBB",
            "quant_value": "> -1",
            "assay_model": "calculation",
            "support_text": "The logBB value is within the suggested limit.",
        }
    )
    assert label is None
    assert reason == "computational_or_predicted_result"


def test_rejects_calculated_logbb_when_only_support_text_marks_calculation() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "logBB",
            "quant_value": "0.5",
            "support_text": "A logBB value of 0.5 was calculated for this molecule.",
            "pmid": "123",
        }
    )
    assert label is None
    assert reason == "computational_or_predicted_result"


def test_requires_provenance_for_metric_without_experimental_context() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "brain/plasma ratio",
            "quant_value": "1.2",
            "support_text": "The brain/plasma ratio was 1.2.",
        }
    )
    assert label is None
    assert reason == "no_explicit_experimental_basis"


def test_accepts_systemic_quantitative_metric_with_provenance() -> None:
    label, method = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "brain/plasma ratio",
            "quant_value": "1.2",
            "support_text": "In rats after oral dosing, the brain/plasma ratio was 1.2.",
            "species": "rat",
            "pmid": "123",
        }
    )
    assert label == 1
    assert method.endswith(":systemic_quantitative_direct_metric")


def test_applies_revision_bound_manual_source_exclusion() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "good_penetration",
            "quant_metric": "brain/plasma ratio",
            "quant_value": "1.2",
            "assay_model": "in vivo",
        },
        source_index=246846,
    )
    assert label is None
    assert reason == "manual_source_exclusion:in_silico_logbb_source"


def test_rejects_metal_complex_before_parent_normalization() -> None:
    label, reason = label_record(
        {
            "smiles": "N.N.[Cl-].[Cl-].[Pt+2]",
            "bbb_permeability_label": "poor_penetration",
            "quant_metric": "CSF/plasma ratio",
            "quant_value": "0.04",
            "assay_model": "clinical pharmacokinetic study",
        }
    )
    assert label is None
    assert reason == "unsupported_metal_complex_identity"


def test_rejects_pampa_and_cell_models() -> None:
    for assay_model in ("PAMPA-BBB", "MDCK-MDR1 monolayer"):
        label, reason = label_record(
            {
                "bbb_permeability_label": "permeable",
                "quant_metric": "Papp",
                "quant_value": "12",
                "assay_model": assay_model,
            }
        )
        assert label is None
        assert reason == "in_vitro_or_passive_permeability_result"


def test_rejects_unmeasured_generic_permeability_claim() -> None:
    scope = classify_scope(
        {
            "bbb_permeability_label": "permeable",
            "support_text": "This lipophilic compound can passively diffuse across the BBB.",
        }
    )
    assert scope.rejection_reason == "no_direct_cns_outcome"


def test_rejects_indirect_in_vivo_efficacy_inference() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "impermeable",
            "assay_model": "animal models",
            "support_text": (
                "Intravenous treatment had little effect, probably because the compound "
                "does not cross the BBB."
            ),
        }
    )
    assert label is None
    assert reason == "indirect_outcome_inference"


def test_rejects_bbb_claim_based_only_on_in_vivo_activity() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "permeable",
            "assay_model": "in vivo",
            "support_text": (
                "The compound demonstrated the ability to cross the BBB based on "
                "its in vivo activity after oral administration."
            ),
        }
    )
    assert label is None
    assert reason == "indirect_outcome_inference"


def test_rejects_csf_elimination_metric_without_entry_measurement() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "poor_penetration",
            "quant_metric": "CSF half-life",
            "quant_value": "<4 h",
            "support_text": "The active CSF-to-plasma efflux shortened CSF half-life.",
        }
    )
    assert label is None
    assert reason == "no_direct_cns_outcome"


def test_rejects_obvious_within_record_direction_conflict() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "permeable",
            "quant_metric": "Kp,uu",
            "quant_value": "low",
            "assay_model": "in vivo",
        }
    )
    assert label is None
    assert reason == "within_record_direction_conflict"


def test_rejects_non_systemic_or_altered_barrier_context() -> None:
    for text in (
        "The compound was injected intrathecally and detected in CSF.",
        "The compound was injected intracisternally before brain measurement.",
        "Focused ultrasound opened the BBB before brain measurement.",
    ):
        label, reason = label_record(
            {
                "bbb_permeability_label": "permeable",
                "assay_model": "in vivo",
                "support_text": text,
            }
        )
        assert label is None
        assert reason == "non_systemic_or_altered_barrier_context"


def test_rejects_interpretation_altering_qualifying_conditions() -> None:
    label, reason = label_record(
        {
            "bbb_permeability_label": "permeable",
            "quant_metric": "brain concentration",
            "quant_value": "10 ng/g",
            "assay_model": "in vivo",
            "qualifying_conditions": "meningitis",
        }
    )
    assert label is None
    assert reason == "interpretation_altering_qualifying_conditions"
