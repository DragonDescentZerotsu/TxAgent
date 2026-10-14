from predict.tasks.bbb_martins.prediction_filter import prediction_only


def test_bbb_filter_keeps_measured_assay_outcomes_and_excludes_model_only_cards():
    cases = [
        ({"support_text": "QikProp predicted logBB = -1.2"}, True),
        (
            {"assay_context": "SwissADME", "support_text": "Predicted BBB negative"},
            True,
        ),
        (
            {
                "support_text": "Predicted CNS+; no PAMPA assay was performed",
                "experimental_details": {"assay_type": "PAMPA"},
            },
            True,
        ),
        (
            {"support_text": "PAMPA-BBB showed low Pe; classified as CNS negative"},
            False,
        ),
        (
            {
                "support_text": "Predicted permeability; measured brain uptake also reported"
            },
            False,
        ),
        (
            {
                "assay_context": "in vitro BBB model",
                "support_text": "The assay showed a permeability coefficient",
            },
            False,
        ),
    ]
    for card, expected in cases:
        assert prediction_only({"card": card}) is expected
