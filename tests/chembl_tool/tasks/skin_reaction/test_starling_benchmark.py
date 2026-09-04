from tools.chembl_tool.tasks.skin_reaction.starling_benchmark import label_record


def _record(**updates):
    row = {
        "reaction_type": "sensitization",
        "outcome_label": "positive",
        "assay_or_test": "local lymph node assay (LLNA)",
        "support_text": "Experimental LLNA gave a positive sensitization outcome.",
    }
    row.update(updates)
    return row


def test_experimental_direct_sensitization_outcome_can_vote():
    assert label_record(_record())[0] == 1


def test_model_prediction_cannot_vote_even_with_binary_outcome_label():
    label, reason = label_record(
        _record(
            assay_or_test="QSAR model prediction",
            support_text="A machine learning model predicted a positive value.",
        )
    )
    assert label is None
    assert reason == "prediction_only"


def test_photo_and_irritation_only_outcomes_cannot_vote():
    photo = label_record(
        _record(assay_or_test="photopatch test", support_text="UVA phototoxic response")
    )
    irritation = label_record(
        _record(
            assay_or_test="primary skin irritation test",
            support_text="Dermal erythema and edema were scored after exposure.",
        )
    )
    assert photo == (None, "out_of_scope_photo_hazard")
    assert irritation == (None, "out_of_scope_irritation")


def test_ps_llna_and_noncontact_drug_reactions_cannot_vote():
    photo_llna = label_record(
        _record(
            assay_or_test="PS LLNA",
            support_text="A positive result was listed under the PS LLNA column.",
        )
    )
    dress = label_record(
        _record(
            assay_or_test="clinical case report",
            support_text="The patient developed DRESS syndrome after treatment.",
        )
    )
    assert photo_llna == (None, "out_of_scope_photo_hazard")
    assert dress == (None, "out_of_scope_noncontact_cutaneous_reaction")
