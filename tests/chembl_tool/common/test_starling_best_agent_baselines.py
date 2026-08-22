import csv
import json

import numpy as np

from tools.chembl_tool.paper_experiments.analyze_starling_best_agent_baselines import (
    analyze,
    holm_adjust,
    paired_macro_f1_significance,
)


def test_paired_permutation_preserves_direction() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8)
    agent = labels.copy()
    baseline = np.asarray([1, 0, 0, 1, 1, 0], dtype=np.int8)
    result = paired_macro_f1_significance(
        labels,
        agent,
        baseline,
        permutation_replicates=200,
        bootstrap_replicates=200,
        permutation_seed=7,
        bootstrap_seed=11,
    )

    assert result["delta_macro_f1_agent_minus_baseline"] > 0
    assert 0 <= result["p_value_one_sided"] <= 1
    assert result["agent_only_correct"] > result["baseline_only_correct"]


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.02])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0 <= value <= 1 for value in adjusted)


def test_analysis_selects_best_agent_and_all_baselines(tmp_path, monkeypatch) -> None:
    rows = []
    predictions = (
        ("mol-a", 0, 0),
        ("mol-b", 0, 1),
        ("mol-c", 1, 1),
        ("mol-d", 1, 1),
    )

    def write_predictions(directory, *, agent, values):
        directory.mkdir(parents=True)
        name = "predictions.jsonl" if agent else "valid_predictions.jsonl"
        with (directory / name).open("w", encoding="utf-8") as handle:
            for index, (molecule, label, prediction) in enumerate(values):
                row = (
                    {
                        "query_index": index,
                        "smiles": molecule,
                        "label": label,
                        "pred_label": prediction,
                        "final_status": "ok",
                    }
                    if agent
                    else {
                        "query_index": index,
                        "drug": molecule,
                        "Y": label,
                        "prediction": prediction,
                        "status": "ok",
                    }
                )
                handle.write(json.dumps(row) + "\n")
        return directory / "metrics.json"

    agent_low = write_predictions(tmp_path / "agent-low", agent=True, values=predictions)
    agent_best_values = tuple(
        (molecule, label, label) for molecule, label, _ in predictions
    )
    agent_best = write_predictions(
        tmp_path / "agent-best", agent=True, values=agent_best_values
    )
    baseline_specs = (
        ("minimol_train_all", "minimol"),
        ("morgan_knn_k3", "structure_knn"),
        ("minimol_embedding_cosine_knn_k3", "minimol_embedding_knn"),
    )
    baseline_paths = {
        method: write_predictions(
            tmp_path / method, agent=False, values=predictions
        )
        for method, _ in baseline_specs
    }
    rows.extend(
        [
            {
                "benchmark_split": "scaffold",
                "evaluation_subset": "valid",
                "task": "bbb_martins",
                "method": "none",
                "method_family": "molecular_evidence_agent",
                "model_label": "Agent",
                "macro_f1": "0.7333333333333334",
                "metrics_path": str(agent_low),
            },
            {
                "benchmark_split": "scaffold",
                "evaluation_subset": "valid",
                "task": "bbb_martins",
                "method": "starling_full_flat",
                "method_family": "molecular_evidence_agent",
                "model_label": "Agent",
                "macro_f1": "1.0",
                "metrics_path": str(agent_best),
            },
        ]
    )
    for method, family in baseline_specs:
        rows.append(
            {
                "benchmark_split": "scaffold",
                "evaluation_subset": "valid",
                "task": "bbb_martins",
                "method": method,
                "method_family": family,
                "model_label": "Agent",
                "macro_f1": "0.7333333333333334",
                "metrics_path": str(baseline_paths[method]),
            }
        )

    metrics = tmp_path / "metrics.tsv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "analysis" / "paired"
    monkeypatch.setattr(
        "tools.chembl_tool.paper_experiments.analyze_starling_best_agent_baselines.ROOT",
        tmp_path,
    )

    result = analyze(
        metrics,
        output,
        permutation_replicates=100,
        bootstrap_replicates=100,
    )

    assert len(result) == 3
    assert {row["baseline_method"] for row in result} == {
        method for method, _ in baseline_specs
    }
    assert {row["agent_method"] for row in result} == {"starling_full_flat"}
    assert output.with_suffix(".tsv").exists()
    assert output.with_suffix(".json").exists()
    assert output.with_suffix(".md").exists()

    experiment_metrics = tmp_path / "experiment_metrics.tsv"
    experiment_row = {
        "benchmark_split": "scaffold",
        "evaluation_subset": "valid",
        "task": "bbb_martins",
        "method": "deepseek_best",
        "model_label": "DeepSeek-v4-pro",
        "comparison_role": "candidate",
        "macro_f1": "1.0",
        "metrics_path": str(agent_best),
    }
    with experiment_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(experiment_row),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(experiment_row)

    deepseek_result = analyze(
        metrics,
        tmp_path / "analysis" / "deepseek_paired",
        agent_metrics_path=experiment_metrics,
        agent_model="DeepSeek-v4-pro",
        permutation_replicates=100,
        bootstrap_replicates=100,
    )

    assert {row["agent_model"] for row in deepseek_result} == {"DeepSeek-v4-pro"}
    assert {row["agent_method"] for row in deepseek_result} == {"deepseek_best"}
