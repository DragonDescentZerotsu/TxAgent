import csv
import json

import pytest

from tools.chembl_tool.paper_experiments.analyze_coverage_selector_retrieval_changes import (
    final_input_signature,
    neighbor_identity,
)
from tools.chembl_tool.paper_experiments.summarize_coverage_selector_llm_matrix import (
    binary_metrics,
    bootstrap_macro_f1_delta,
    exact_mcnemar_p,
    prediction_paths,
    read_predictions,
    run_paths,
)
from tools.chembl_tool.paper_experiments.plot_coverage_selector_llm_matrix import (
    CONDITIONS as PLOT_CONDITIONS,
    METHODS as PLOT_METHODS,
    render,
)


def test_binary_metrics_and_exact_mcnemar():
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert exact_mcnemar_p(0, 0) == pytest.approx(1.0)
    assert exact_mcnemar_p(3, 0) == pytest.approx(0.25)


def test_bootstrap_macro_f1_delta_is_seeded_and_validated():
    first = bootstrap_macro_f1_delta(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        iterations=50,
        seed=17,
    )
    second = bootstrap_macro_f1_delta(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        iterations=50,
        seed=17,
    )

    assert first == second
    with pytest.raises(ValueError, match="iterations must be positive"):
        bootstrap_macro_f1_delta([0], [0], [0], iterations=0, seed=17)


def test_read_predictions_requires_complete_successful_matched_rows(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "query_index": index,
            "status": "ok",
            "final_status": "ok",
            "label": index,
            "pred_label": index,
            "correct": True,
        }
        for index in range(2)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert sorted(read_predictions(path, expected_n=2)) == [0, 1]
    rows[1]["final_status"] = "error"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or failed rows"):
        read_predictions(path, expected_n=2)


def test_condition_paths_centralize_legacy_bbb_scaffold_location():
    control_root, coverage_root, control_prefix, coverage_prefix = run_paths(
        "scaffold", "bbb_martins"
    )
    control_predictions, coverage_predictions = prediction_paths(
        "scaffold", "bbb_martins"
    )

    assert control_root.name == "runs"
    assert coverage_root.name == "runs"
    assert control_prefix == "bbb_martins__starling_full_mechanism"
    assert coverage_prefix == "bbb_scaffold_starling_full_mechanism_pd_coverage_changed50_v1"
    assert control_predictions == control_root.parent / "predictions.jsonl"
    assert coverage_predictions == coverage_root.parent / "predictions.jsonl"
    assert "coverage_selector_pilot" in str(coverage_root)


def test_final_input_signature_ignores_only_generated_assistant_answer(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    base_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "same final input"},
    ]
    for path, answer in ((first, "pass"), (second, "fail")):
        row = {"messages": [*base_messages, {"role": "assistant", "content": answer}]}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert final_input_signature(first) == final_input_signature(second)
    assert neighbor_identity({"standard_inchi_key": "AAAA"}) == "AAAA"
    with pytest.raises(ValueError, match="stable molecule identity"):
        neighbor_identity({})


def test_coverage_selector_matrix_plot_reads_canonical_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.tsv"
    output = tmp_path / "matrix.svg"
    fieldnames = ["task", "benchmark_split", "method", "n", "accuracy", "macro_f1"]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for task, split, _ in PLOT_CONDITIONS:
            for method, _, _ in PLOT_METHODS:
                writer.writerow(
                    {
                        "task": task,
                        "benchmark_split": split,
                        "method": method,
                        "n": 10,
                        "accuracy": 0.7,
                        "macro_f1": 0.6,
                    }
                )

    render(metrics_path, output)

    svg = output.read_text(encoding="utf-8")
    assert "Retrieval Selector Comparison Across Starling" in svg
    assert "Morgan similarity retrieve" in svg
    assert "Coverage retrieve" in svg
