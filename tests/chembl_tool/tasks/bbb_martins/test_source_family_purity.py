from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
    BBBSourceFamilyClassifier,
    DIRECT_GROUP,
    EFFLUX_GROUP,
    INFLUX_GROUP,
    NEAR_DIRECT_GROUP,
    PASSIVE_GROUP,
    NearDirectReview,
    gold_contract_decision,
    gold_contract_decision_v4,
    mdck_target,
    nondirect_target,
    support_key,
)
from tools.chembl_tool.paper_experiments.review_bbb_near_direct_source_family import (
    _review_row,
)


def _row(**updates):
    row = {
        "group_id": DIRECT_GROUP,
        "source_index": 1,
        "canonical_smiles": "CCO",
        "pmid": "1",
        "canonical_assay_context": "",
        "canonical_endpoint_name": "",
        "support_text": "",
    }
    row.update(updates)
    return row


def test_prediction_only_pampa_is_retained_as_passive_evidence():
    row = _row(
        canonical_assay_context="PAMPA-BBB",
        support_text="A model predicted low PAMPA permeability.",
    )
    decision = BBBSourceFamilyClassifier({}, set())(row)
    assert decision.new_group == PASSIVE_GROUP
    assert decision.reason == "pampa_membrane_permeability_assay"


def test_pampa_context_does_not_hide_an_explicit_efflux_endpoint():
    row = _row(
        canonical_assay_context="PAMPA-BBB assay",
        canonical_endpoint_name="efflux_substrate_outcome",
        support_text="The compound is a P-gp substrate despite its PAMPA result.",
    )
    decision = BBBSourceFamilyClassifier({}, set())(row)
    assert decision.new_group == EFFLUX_GROUP
    assert decision.reason == "pampa_context_explicit_efflux_readout"


def test_sparse_mdck_is_retained_and_engineered_system_is_efflux():
    assert mdck_target(_row(assay_model="MDCK", support_text=""))[0] == PASSIVE_GROUP
    assert (
        mdck_target(
            _row(
                assay_model="MDCK",
                canonical_endpoint_name="papp_b_to_a",
                support_text="A single B-to-A apparent permeability was reported.",
            )
        )[0]
        == PASSIVE_GROUP
    )
    assert (
        mdck_target(_row(assay_model="MDCK-MDR1", support_text=""))[0] == EFFLUX_GROUP
    )
    assert (
        mdck_target(
            _row(
                assay_model="MDCK",
                bbb_transport_label="influx_substrate",
                support_text="uptake",
            )
        )[0]
        == INFLUX_GROUP
    )


def test_transporter_mediated_does_not_override_explicit_efflux_endpoint():
    decision = nondirect_target(
        _row(
            bbb_transport_label="transporter_mediated",
            canonical_endpoint_name="efflux_ratio",
            support_text="Mrp transport produced an efflux ratio of 2.0.",
        ),
        "no_direct_cns_outcome",
    )
    assert decision.new_group == EFFLUX_GROUP


def test_generic_transporter_mediated_row_is_not_forced_into_influx():
    decision = nondirect_target(
        _row(
            bbb_transport_label="transporter_mediated",
            canonical_endpoint_name="bbb_substrate_relationship",
            support_text="A transporter relationship was reported.",
        ),
        "no_direct_cns_outcome",
    )
    assert decision.new_group == NEAR_DIRECT_GROUP


def test_brain_to_plasma_ratio_is_not_an_efflux_ratio():
    decision = nondirect_target(
        _row(
            canonical_endpoint_name="brain_to_plasma_ratio",
            support_text="The brain-to-plasma ratio was measured after dosing.",
        ),
        "non_systemic_or_altered_barrier_context",
    )
    assert decision.new_group == NEAR_DIRECT_GROUP


def test_explicit_carrier_influx_remains_influx():
    decision = nondirect_target(
        _row(
            bbb_transport_label="transporter_mediated",
            canonical_transport_mechanism="carrier_mediated_influx",
            canonical_endpoint_name="blood_to_brain_transport",
            support_text="Carrier-mediated blood-to-brain transport was observed.",
        ),
        "no_direct_cns_outcome",
    )
    assert decision.new_group == INFLUX_GROUP


def test_reviewed_functional_proxy_moves_but_gold_provenance_stays_direct():
    row = _row(support_text="A systemic central pharmacodynamic response was observed.")
    review = NearDirectReview(
        1, "record-1", "move_to_near_direct", "reviewed_proxy", support_key(row)
    )
    decision = BBBSourceFamilyClassifier({1: review}, set())(row)
    assert decision.new_group == NEAR_DIRECT_GROUP

    classifier = BBBSourceFamilyClassifier({1: review}, {1})
    assert classifier(row) == ""


def test_explicit_prediction_is_not_direct_even_if_legacy_gold_voter():
    row = _row(
        bbb_permeability_label="low_permeability",
        canonical_endpoint_name="logbb",
        assay_model="ADME/T analysis",
        support_text="ADME/T analysis predicted LogBB = -1.2.",
    )
    decision = BBBSourceFamilyClassifier({}, {1})(row)
    assert decision.new_group == NEAR_DIRECT_GROUP
    assert decision.reason == "computational_or_predicted_bbb_proxy"


def test_reviewed_experimental_transporter_model_is_not_prediction_keyword_hit():
    row = _row(
        source_index=112971,
        bbb_permeability_label="restricted",
        canonical_assay_context="in silico prediction",
        assay_model="compartmental BBB model",
        support_text=(
            "The study showed functional cooperation of P-gp and Bcrp restricts "
            "measured brain distribution."
        ),
    )
    assert BBBSourceFamilyClassifier({}, {112971})(row) == ""


def test_gold_contract_eligible_experimental_outcome_stays_direct():
    row = _row(
        bbb_permeability_label="high_permeability",
        assay_model="in vivo pharmacokinetic study after intravenous dosing",
        species="rat",
        pmid="1",
        canonical_endpoint_name="kp_brain",
        canonical_measurement_text="0.95",
        support_text="Brain exposure was measured after intravenous dosing.",
    )
    assert gold_contract_decision(row)[0] == 1
    assert BBBSourceFamilyClassifier({}, set())(row) == ""


def test_v4_classifier_keeps_reviewed_missing_direction_voter_in_l1():
    row = _row(
        source_index=5820,
        bbb_permeability_label=None,
        species="rat",
        assay_model="in vivo",
        canonical_endpoint_name="kp_uu",
        canonical_measurement_text="very low",
        support_text="The unbound brain/blood ratio was very low in vivo.",
    )
    assert gold_contract_decision(row)[0] is None
    assert gold_contract_decision_v4(row)[0] == 0
    classifier = BBBSourceFamilyClassifier(
        {}, set(), gold_decision=gold_contract_decision_v4
    )
    assert classifier(row) == ""


def test_external_comparative_bbb_outcome_review_precedes_influx_assignment():
    row = _row(
        source_index=None,
        source_record_id="ext_1",
        canonical_record_id=(
            "11dc172aa9c4dd920fe28ffef2668698a41d3de96f63bdb437a836c6ea43c798"
        ),
        group_id=INFLUX_GROUP,
        canonical_endpoint_name="blood_to_brain_transport",
        canonical_transport_mechanism="unspecified_mediated_bbb_uptake",
        support_text=(
            "The cytotoxic query cannot penetrate the BBB. However, its "
            "4-amino analog readily crosses the BBB or is transported into "
            "the brain parenchyma."
        ),
    )
    review = NearDirectReview(
        source_index=None,
        canonical_record_id=row["canonical_record_id"],
        decision="move_to_near_direct",
        reason=(
            "comparative_qualitative_bbb_outcome_without_experimental_measurement"
        ),
        support_key=support_key(row),
    )
    decision = BBBSourceFamilyClassifier(
        {f"canonical_record_id:{row['canonical_record_id']}": review}, set()
    )(row)
    assert decision.new_group == NEAR_DIRECT_GROUP
    assert decision.reason == review.reason


def test_qikprop_and_boiled_egg_predictions_are_not_direct():
    qplogbb = _row(
        assay_model="QikProp",
        canonical_endpoint_name="logbb",
        support_text="QikProp predicted QPlogBB = -0.4.",
    )
    assert nondirect_target(qplogbb, "computational_or_predicted_result").new_group == (
        NEAR_DIRECT_GROUP
    )

    qppmdck = _row(
        assay_model="QikProp",
        canonical_endpoint_name="qppmdck",
        support_text="QikProp predicted QPPMDCK permeability.",
    )
    assert nondirect_target(qppmdck, "computational_or_predicted_result").new_group == (
        PASSIVE_GROUP
    )

    boiled = _row(
        assay_model="BOILED-Egg",
        canonical_endpoint_name="bbb_permeability_outcome",
        support_text="The molecule was predicted in the yolk passive BBB region.",
    )
    assert nondirect_target(boiled, "computational_or_predicted_result").new_group == (
        PASSIVE_GROUP
    )

    boiled_pgp = _row(
        assay_model="BOILED-Egg",
        canonical_endpoint_name="efflux_substrate_outcome",
        support_text="SwissADME predicted a P-gp substrate.",
    )
    assert nondirect_target(
        boiled_pgp, "computational_or_predicted_result"
    ).new_group == EFFLUX_GROUP


def test_missing_and_generic_nonmechanistic_rows_move_to_near_direct():
    classifier = BBBSourceFamilyClassifier({}, set())
    missing = classifier(_row(canonical_endpoint_name="missing_endpoint"))
    generic = classifier(_row(canonical_endpoint_name="bbb_permeability_outcome"))
    assert missing.new_group == NEAR_DIRECT_GROUP
    assert generic.new_group == NEAR_DIRECT_GROUP


def test_explicit_passive_mechanism_without_named_assay_is_passive():
    decision = BBBSourceFamilyClassifier({}, set())(
        _row(
            canonical_endpoint_name="membrane_transport",
            support_text="The compound crossed by passive diffusion.",
        )
    )
    assert decision.new_group == PASSIVE_GROUP


def test_explicit_functional_proxy_review_overrides_generic_outcome_scope():
    row = _row(
        source_index=224755,
        assay_model="oral administration",
        canonical_assay_context="oral administration",
        support_text=(
            "Candesartan inhibited centrally mediated AngII responses after oral "
            "administration, illustrating BBB crossing."
        ),
        extra_details="Permeability is inferred from a central response.",
    )
    reviewed = _review_row(row, set())
    assert reviewed["decision"] == "move_to_near_direct"
    assert reviewed["decision_reason"] == (
        "systemic_central_pharmacodynamic_response_proxy"
    )
