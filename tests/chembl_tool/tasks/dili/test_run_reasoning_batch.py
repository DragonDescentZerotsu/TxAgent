from tools.chembl_tool.tasks.dili.run_reasoning_batch import prediction_to_label


def test_prediction_to_label_mapping():
    assert prediction_to_label("dili_risk") == 1
    assert prediction_to_label("hepatotoxic") == 1
    assert prediction_to_label("no_dili_risk") == 0
    assert prediction_to_label("non-hepatotoxic") == 0
    assert prediction_to_label("uncertain") is None
