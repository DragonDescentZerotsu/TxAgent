import csv

import pytest

from tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview import (
    SPLITS,
    TASKS,
    render,
)
from tools.chembl_tool.paper_experiments.plot_starling_model_comparison import (
    render as render_model_comparison,
)


def _write_metrics(
    path,
    *,
    omit=None,
    failed=None,
    splits=None,
    count_failure=False,
    model_label="Test model",
    evaluation_subset="test",
):
    rows = []
    for split, _ in splits or SPLITS:
        for task in TASKS:
            for index, method in enumerate(task.methods):
                key = (split, task.key, method.key)
                if key == omit:
                    continue
                rows.append(
                    {
                        "benchmark_split": split,
                        "task": task.key,
                        "method": method.key,
                        "method_family": (
                            "molecular_evidence_agent"
                            if method.source not in {"minimol", "knn", "minimol_knn"}
                            else method.source
                        ),
                        "model_label": model_label,
                        "evaluation_subset": evaluation_subset,
                        "n_test": 500 if task.key == "bbb_martins" else 380,
                        "n_failed": 1 if key == failed else 0,
                        "failure_policy": (
                            "count_as_incorrect_opposite_label"
                            if key == failed and count_failure
                            else ""
                        ),
                        "macro_f1": 0.5 + index / 100,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "benchmark_split",
                "task",
                "method",
                "method_family",
                "model_label",
                "evaluation_subset",
                "n_test",
                "n_failed",
                "failure_policy",
                "macro_f1",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_experiment_metrics(
    path,
    *,
    anchor_delta=0.0,
    base_model_label="",
    experiment_model_label="",
    method_prefix="",
):
    rows = []
    for task in TASKS:
        anchor_index = next(
            index
            for index, method in enumerate(task.methods)
            if method.key == "starling_full_mechanism"
        )
        anchor_f1 = 0.5 + anchor_index / 100 + anchor_delta
        n_test = 500 if task.key == "bbb_martins" else 380
        for method, label, delta, comparison_role in (
            ("morgan_standard", "Morgan anchor", 0.0, "anchor"),
            (
                f"{method_prefix}coverage_standard",
                "Coverage retrieve + standard context",
                0.01,
                "experiment",
            ),
            (
                f"{method_prefix}coverage_aware",
                "Coverage retrieve + coverage-aware context",
                0.02,
                "experiment",
            ),
        ):
            rows.append(
                {
                    "benchmark_split": "scaffold",
                    "task": task.key,
                    "method": method,
                    "method_label": label,
                    "n": n_test,
                    "macro_f1": anchor_f1 + delta,
                    "comparison_role": comparison_role,
                    "base_method": "starling_full_mechanism",
                    "base_model_label": base_model_label,
                    "model_label": experiment_model_label,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "benchmark_split",
                "task",
                "method",
                "method_label",
                "n",
                "macro_f1",
                "comparison_role",
                "base_method",
                "base_model_label",
                "model_label",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_chart_renders_all_tasks_splits_and_baselines(tmp_path):
    metrics = tmp_path / "metrics.tsv"
    output = tmp_path / "overview.svg"
    _write_metrics(metrics)

    render(metrics, output)
    svg = output.read_text(encoding="utf-8")

    assert "Starling Benchmark Performance" in svg
    assert "Random split · n =" in svg
    assert "Scaffold split · n =" in svg
    assert svg.count("MiniMol · Train all") == 6
    assert svg.count("Morgan KNN · k=3") == 7  # Six panels plus the legend.
    assert svg.count("MiniMol KNN · k=3") == 7  # Six panels plus the legend.
    assert "MiniMol/cosine" in svg
    for task in TASKS:
        assert svg.count(f">{task.title}</text>") == 2


def test_chart_rejects_missing_or_failed_rows(tmp_path):
    metrics = tmp_path / "metrics.tsv"
    output = tmp_path / "overview.svg"
    key = ("random", "bbb_martins", "none")
    _write_metrics(metrics, omit=key)
    with pytest.raises(ValueError, match="Missing Starling benchmark results"):
        render(metrics, output)

    _write_metrics(metrics, failed=key)
    with pytest.raises(ValueError, match="rows with failures"):
        render(metrics, output)


def test_chart_renders_single_valid_split_and_counted_failure(tmp_path):
    metrics = tmp_path / "metrics.tsv"
    output = tmp_path / "overview.svg"
    key = ("scaffold", "bioavailability_ma", "chembl_full_flat")
    _write_metrics(
        metrics,
        failed=key,
        splits=(("scaffold", "Scaffold split"),),
        count_failure=True,
    )

    render(metrics, output)
    svg = output.read_text(encoding="utf-8")

    assert "Scaffold test · Macro-F1" in svg
    assert "Random split · n =" not in svg
    assert "Failed pipeline samples are counted as incorrect" in svg


def test_model_comparison_renders_paired_agents_and_shared_baselines(tmp_path):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(
        reference,
        splits=splits,
        model_label="GPT-OSS-20B",
        evaluation_subset="valid",
    )
    _write_metrics(
        candidate,
        splits=splits,
        model_label="GPT-OSS-120B",
        evaluation_subset="valid",
    )

    render_model_comparison(reference, candidate, output)
    svg = output.read_text(encoding="utf-8")

    assert "GPT-OSS-20B vs GPT-OSS-120B" in svg
    assert "Shared train-label baseline" in svg
    assert svg.count("MiniMol · Train all") == 3
    assert svg.count("Scaffold split · n =") == 3


def test_model_comparison_adds_visible_series_as_grouped_bars(tmp_path):
    reference = tmp_path / "blind_20b.tsv"
    candidate = tmp_path / "blind_120b.tsv"
    visible_20b = tmp_path / "visible_20b.tsv"
    visible_120b = tmp_path / "visible_120b.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    for path, label in (
        (reference, "GPT-OSS-20B blind"),
        (candidate, "GPT-OSS-120B blind"),
        (visible_20b, "GPT-OSS-20B visible"),
        (visible_120b, "GPT-OSS-120B visible"),
    ):
        _write_metrics(
            path,
            splits=splits,
            model_label=label,
            evaluation_subset="valid",
        )

    render_model_comparison(
        reference,
        candidate,
        output,
        comparison_paths=(visible_20b, visible_120b),
    )
    svg = output.read_text(encoding="utf-8")

    assert "4 model/visibility settings" in svg
    assert "GPT-OSS-20B blind" in svg
    assert "GPT-OSS-120B visible" in svg
    assert svg.count("MiniMol · Train all") == 3


def test_model_comparison_validates_subset_and_supports_two_splits(tmp_path):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    output = tmp_path / "comparison.svg"
    _write_metrics(reference, model_label="GPT-OSS-20B", evaluation_subset="valid")
    _write_metrics(candidate, model_label="GPT-OSS-120B", evaluation_subset="valid")

    render_model_comparison(reference, candidate, output)
    svg = output.read_text(encoding="utf-8")
    assert svg.count("Random split · n =") == 3
    assert svg.count("Scaffold split · n =") == 3

    _write_metrics(candidate, model_label="GPT-OSS-120B", evaluation_subset="test")
    with pytest.raises(ValueError, match="same single evaluation subset"):
        render_model_comparison(reference, candidate, output)


def test_model_comparison_appends_matched_experiments_without_duplicate_anchor(
    tmp_path,
):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    experiments = tmp_path / "experiments.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(
        reference,
        splits=splits,
        model_label="GPT-OSS-20B",
        evaluation_subset="valid",
    )
    _write_metrics(
        candidate,
        splits=splits,
        model_label="GPT-OSS-120B",
        evaluation_subset="valid",
    )
    _write_experiment_metrics(experiments)

    render_model_comparison(reference, candidate, output, (experiments,))
    svg = output.read_text(encoding="utf-8")

    assert "Starling Benchmark Model &amp; Experiment Comparison" in svg
    assert "Matched GPT-OSS-120B experiment" in svg
    assert svg.count("Coverage · standard context") == 3
    assert svg.count("Coverage · coverage-aware") == 3
    assert "Morgan anchor" not in svg


def test_model_comparison_rejects_experiment_anchor_drift(tmp_path):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    experiments = tmp_path / "experiments.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(reference, splits=splits, evaluation_subset="valid")
    _write_metrics(candidate, splits=splits, evaluation_subset="valid")
    _write_experiment_metrics(experiments, anchor_delta=0.1)

    with pytest.raises(ValueError, match="anchor macro-F1 mismatch"):
        render_model_comparison(reference, candidate, output, (experiments,))


def test_model_comparison_supports_visible_experiment_anchor(tmp_path):
    reference = tmp_path / "blind_20b.tsv"
    candidate = tmp_path / "blind_120b.tsv"
    visible = tmp_path / "visible_120b.tsv"
    blind_experiments = tmp_path / "blind_experiments.tsv"
    visible_experiments = tmp_path / "visible_experiments.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    for path, label in (
        (reference, "GPT-OSS-20B blind"),
        (candidate, "GPT-OSS-120B blind"),
        (visible, "GPT-OSS-120B visible"),
    ):
        _write_metrics(
            path,
            splits=splits,
            model_label=label,
            evaluation_subset="valid",
        )
    _write_experiment_metrics(
        blind_experiments,
        experiment_model_label="GPT-OSS-120B blind",
    )
    _write_experiment_metrics(
        visible_experiments,
        base_model_label="GPT-OSS-120B visible",
        experiment_model_label="GPT-OSS-120B visible",
        method_prefix="visible_",
    )

    render_model_comparison(
        reference,
        candidate,
        output,
        experiment_paths=(blind_experiments, visible_experiments),
        comparison_paths=(visible,),
    )
    svg = output.read_text(encoding="utf-8")

    assert "Matched opt-in experiments" in svg
    assert svg.count("120B blind · Coverage · standard context") == 3
    assert svg.count("120B visible · Coverage · standard context") == 3
