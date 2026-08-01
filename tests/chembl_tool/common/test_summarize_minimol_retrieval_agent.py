import json
from types import SimpleNamespace

from tools.chembl_tool.paper_experiments.summarize_minimol_retrieval_agent import (
    summarize_condition,
)


def _write_batch(path, predictions):
    path.mkdir(parents=True)
    accuracy = sum(
        row["pred_label"] == row.get("true_label", row.get("label"))
        for row in predictions
    ) / len(predictions)
    (path / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "n_failed_runs": 0}),
        encoding="utf-8",
    )
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )


def test_summarize_condition_pairs_the_same_starling_queries(tmp_path):
    gold = [0, 0, 1, 1]
    morgan_predictions = [0, 1, 0, 1]
    minimol_predictions = [0, 0, 1, 1]
    morgan_batch = tmp_path / "morgan"
    minimol_batch = tmp_path / "minimol"
    _write_batch(
        morgan_batch,
        [
            {"query_index": i, "true_label": label, "pred_label": prediction}
            for i, (label, prediction) in enumerate(zip(gold, morgan_predictions))
        ],
    )
    _write_batch(
        minimol_batch,
        [
            {"query_index": i, "true_label": label, "pred_label": prediction}
            for i, (label, prediction) in enumerate(zip(gold, minimol_predictions))
        ],
    )

    row = summarize_condition(
        "random",
        SimpleNamespace(
            task="bbb_martins",
            name="bbb_martins__chembl_direct",
            source="chembl",
            mode="direct",
        ),
        morgan_batch=morgan_batch,
        minimol_batch=minimol_batch,
        bootstrap_replicates=100,
    )

    assert row["morgan_macro_f1"] == 0.5
    assert row["minimol_macro_f1"] == 1.0
    assert row["delta_macro_f1"] == 0.5
    assert row["morgan_only_correct"] == 0
    assert row["minimol_only_correct"] == 2
    assert row["n_prediction_flips"] == 2


def test_summarize_condition_accepts_current_label_field(tmp_path):
    morgan_batch = tmp_path / "morgan"
    minimol_batch = tmp_path / "minimol"
    rows = [
        {"query_index": 0, "label": 0, "pred_label": 0},
        {"query_index": 1, "label": 1, "pred_label": 1},
    ]
    _write_batch(morgan_batch, rows)
    _write_batch(minimol_batch, rows)

    result = summarize_condition(
        "random",
        SimpleNamespace(
            task="bbb_martins",
            name="bbb_martins__chembl_direct",
            source="chembl",
            mode="direct",
        ),
        morgan_batch=morgan_batch,
        minimol_batch=minimol_batch,
        bootstrap_replicates=10,
    )

    assert result["morgan_macro_f1"] == 1.0
    assert result["minimol_macro_f1"] == 1.0
