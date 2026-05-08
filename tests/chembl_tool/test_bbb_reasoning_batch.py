from tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch import compute_metrics, prediction_to_label


def test_prediction_to_label_mapping() -> None:
    assert prediction_to_label("fail") == 0
    assert prediction_to_label("pass") == 1


def test_compute_metrics_for_binary_predictions() -> None:
    rows = [
        {"status": "ok", "label": 0, "pred_label": 0, "correct": True, "bbb_prediction": "fail"},
        {"status": "ok", "label": 0, "pred_label": 1, "correct": False, "bbb_prediction": "pass"},
        {"status": "ok", "label": 1, "pred_label": 1, "correct": True, "bbb_prediction": "pass"},
        {"status": "ok", "label": 1, "pred_label": 0, "correct": False, "bbb_prediction": "fail"},
    ]

    metrics = compute_metrics(rows)

    assert metrics["accuracy"] == 0.5
    assert metrics["per_class"]["0"]["tp"] == 1
    assert metrics["per_class"]["0"]["fp"] == 1
    assert metrics["per_class"]["0"]["fn"] == 1
    assert metrics["per_class"]["1"]["tp"] == 1
    assert metrics["per_class"]["1"]["fp"] == 1
    assert metrics["per_class"]["1"]["fn"] == 1
    assert metrics["macro_f1"] == 0.5
