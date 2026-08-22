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
from tools.chembl_tool.paper_experiments.starling_paired_figure import (
    _format_p_value,
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
    baseline_delta=0.0,
):
    rows = []
    for split, _ in splits or SPLITS:
        for task in TASKS:
            for index, method in enumerate(task.methods):
                key = (split, task.key, method.key)
                if key == omit:
                    continue
                method_family = (
                    "molecular_evidence_agent"
                    if method.source not in {"minimol", "knn", "minimol_knn"}
                    else method.source
                )
                rows.append(
                    {
                        "benchmark_split": split,
                        "task": task.key,
                        "method": method.key,
                        "method_family": method_family,
                        "model_label": model_label,
                        "evaluation_subset": evaluation_subset,
                        "n_test": 500 if task.key == "bbb_martins" else 380,
                        "n_failed": 1 if key == failed else 0,
                        "failure_policy": (
                            "count_as_incorrect_opposite_label"
                            if key == failed and count_failure
                            else ""
                        ),
                        "macro_f1": 0.5
                        + index / 100
                        + (
                            baseline_delta
                            if method_family != "molecular_evidence_agent"
                            else 0.0
                        ),
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


def _write_paired_ci_metrics(
    path,
    *,
    agent_model="GPT-OSS-120B",
    agent_method_override="",
    experiment_method="",
    experiment_delta=0.02,
    delta_offset=0.0,
):
    rows = []
    for task in TASKS:
        agent_methods = [
            method
            for method in task.methods
            if method.source not in {"minimol", "knn", "minimol_knn"}
        ]
        anchor_method = next(
            method for method in task.methods if method.key == "starling_full_mechanism"
        )
        if experiment_method:
            agent_method_key = experiment_method
            agent_f1 = (
                0.5 + task.methods.index(anchor_method) / 100 + experiment_delta
            )
        else:
            agent_method = next(
                (
                    method
                    for method in agent_methods
                    if method.key == agent_method_override
                ),
                agent_methods[-1],
            )
            agent_method_key = agent_method.key
            agent_f1 = 0.5 + task.methods.index(agent_method) / 100
        n = 500 if task.key == "bbb_martins" else 380
        for baseline_method, baseline_label in (
            ("minimol_train_all", "MiniMol train-all"),
            ("morgan_knn_k3", "Morgan KNN k=3"),
            ("minimol_embedding_cosine_knn_k3", "MiniMol KNN k=3"),
        ):
            baseline = next(
                method for method in task.methods if method.key == baseline_method
            )
            baseline_f1 = 0.5 + task.methods.index(baseline) / 100
            delta = agent_f1 - baseline_f1
            rows.append(
                {
                    "benchmark_split": "scaffold",
                    "evaluation_subset": "valid",
                    "task": task.key,
                    "agent_model": agent_model,
                    "agent_method": agent_method_key,
                    "baseline": baseline_label,
                    "baseline_method": baseline_method,
                    "n": n,
                    "agent_macro_f1": agent_f1,
                    "baseline_macro_f1": baseline_f1,
                    "delta_macro_f1_agent_minus_baseline": delta + delta_offset,
                    "paired_ci95_low": delta - 0.04,
                    "paired_ci95_high": delta + 0.04,
                    "bootstrap_replicates": 10000,
                    "alternative": "agent_greater_than_baseline",
                    "p_value_one_sided": 0.123,
                    "permutation_replicates": 100000,
                    "permutation_seed": 29,
                    "analysis_status": "exploratory_test_fixture",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
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


def test_model_comparison_renders_one_latest_series_with_baselines(tmp_path):
    latest = tmp_path / "latest.tsv"
    output = tmp_path / "latest.svg"
    _write_metrics(
        latest,
        splits=(("scaffold", "Scaffold split"),),
        model_label="Latest GPT-OSS-120B",
        evaluation_subset="valid",
    )
    with latest.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["method"]
            in {
                "none",
                "starling_direct",
                "starling_direct_full",
                "starling_full_flat",
                "starling_full_mechanism",
                "minimol_train_all",
                "morgan_knn_k3",
                "minimol_embedding_cosine_knn_k3",
            }
        ]
    with latest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    render_model_comparison(
        latest,
        tmp_path / "ignored.tsv",
        output,
        series_label_overrides=("Latest GPT-OSS-120B",),
        single_series=True,
    )
    svg = output.read_text(encoding="utf-8")

    assert "Latest GPT-OSS-120B" in svg
    assert "Shared train-label baseline" in svg
    assert svg.count("MiniMol · Train all") == 3


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


def test_model_comparison_supports_dataset_version_agent_only_view(tmp_path):
    old_blind = tmp_path / "old_blind.tsv"
    new_blind = tmp_path / "new_blind.tsv"
    old_visible = tmp_path / "old_visible.tsv"
    new_visible = tmp_path / "new_visible.tsv"
    output = tmp_path / "dataset_comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    for path, label in (
        (old_blind, "GPT-OSS-120B"),
        (new_blind, "GPT-OSS-120B"),
        (old_visible, "GPT-OSS-120B visible"),
        (new_visible, "GPT-OSS-120B visible"),
    ):
        _write_metrics(
            path,
            splits=splits,
            model_label=label,
            evaluation_subset="valid",
        )

    render_model_comparison(
        old_blind,
        new_blind,
        output,
        comparison_paths=(old_visible, new_visible),
        method_family="molecular_evidence_agent",
        series_label_overrides=(
            "Original · Blind",
            "Record-supported v2 · Blind",
            "Original · Visible",
            "Record-supported v2 · Visible",
        ),
        chart_title="GPT-OSS-120B Across Dataset Versions",
        chart_subtitle="Scaffold valid · Parent-disjoint retrieval · Macro-F1",
        context_lines=(
            "Record-supported v2 prioritizes multi-record held-out molecules.",
            "Old and new valid sets contain different molecules.",
        ),
        comparison_title_override="Original vs record-supported v2",
    )
    svg = output.read_text(encoding="utf-8")

    assert "GPT-OSS-120B Across Dataset Versions" in svg
    assert "Record-supported v2 prioritizes multi-record held-out molecules." in svg
    assert "Original vs record-supported v2" in svg
    assert "Record-supported v2 · Visible" in svg
    assert "MiniMol · Train all" not in svg
    assert svg.count("Scaffold split · n =") == 3


def test_model_comparison_can_plot_lineage_specific_baseline_series(tmp_path):
    old = tmp_path / "old.tsv"
    new = tmp_path / "new.tsv"
    output = tmp_path / "dataset_comparison_with_baselines.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(old, splits=splits, model_label="Old", evaluation_subset="valid")
    _write_metrics(
        new,
        splits=splits,
        model_label="New",
        evaluation_subset="valid",
        baseline_delta=0.05,
    )

    with pytest.raises(ValueError, match="Baseline macro_f1 mismatch"):
        render_model_comparison(old, new, output)

    render_model_comparison(
        old,
        new,
        output,
        series_label_overrides=("Original v1", "Record-supported v2"),
        baseline_display="series",
    )
    svg = output.read_text(encoding="utf-8")

    assert svg.count("MiniMol · Train all") == 3
    assert svg.count("Morgan KNN · k=3") == 3
    assert svg.count("MiniMol KNN · k=3") == 3
    assert "Shared train-label baseline" not in svg


def test_model_comparison_collapses_visibility_duplicate_baselines(tmp_path):
    paths = [tmp_path / name for name in ("ob.tsv", "nb.tsv", "ov.tsv", "nv.tsv")]
    splits = (("scaffold", "Scaffold split"),)
    for path, delta in zip(paths, (0.0, 0.05, 0.0, 0.05), strict=True):
        _write_metrics(
            path,
            splits=splits,
            evaluation_subset="valid",
            baseline_delta=delta,
        )
    output = tmp_path / "collapsed_baselines.svg"
    render_model_comparison(
        paths[0],
        paths[1],
        output,
        comparison_paths=(paths[2], paths[3]),
        series_label_overrides=("OB", "NB", "OV", "NV"),
        baseline_display="series",
        baseline_series_groups=("old", "new", "old", "new"),
    )
    assert output.exists()

    _write_metrics(
        paths[2],
        splits=splits,
        evaluation_subset="valid",
        baseline_delta=0.01,
    )
    with pytest.raises(ValueError, match="grouped under the same lineage must match"):
        render_model_comparison(
            paths[0],
            paths[1],
            output,
            comparison_paths=(paths[2], paths[3]),
            series_label_overrides=("OB", "NB", "OV", "NV"),
            baseline_display="series",
            baseline_series_groups=("old", "new", "old", "new"),
        )


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


def test_model_comparison_supports_experiment_legend_override(tmp_path):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    experiments = tmp_path / "experiments.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(reference, splits=splits, evaluation_subset="valid")
    _write_metrics(candidate, splits=splits, evaluation_subset="valid")
    _write_experiment_metrics(experiments)

    render_model_comparison(
        reference,
        candidate,
        output,
        experiment_paths=(experiments,),
        experiment_legend="Additional current-valid results",
    )

    svg = output.read_text(encoding="utf-8")
    assert "Additional current-valid results" in svg
    assert "Teal rows: Additional current-valid results" in svg
    assert "Matched GPT-OSS-120B experiment" not in svg


def test_model_comparison_adds_paired_bootstrap_intervals(tmp_path):
    reference = tmp_path / "blind_20b.tsv"
    candidate = tmp_path / "blind_120b.tsv"
    paired = tmp_path / "paired_ci.tsv"
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
    _write_paired_ci_metrics(paired)

    render_model_comparison(
        reference,
        candidate,
        output,
        paired_ci_path=paired,
    )
    svg = output.read_text(encoding="utf-8")

    assert "Best agent paired Δ · 95% CI" in svg
    assert svg.count("Paired Δ macro-F1: best agent − baseline (95% CI)") == 3
    assert svg.count("10,000 paired resamples") == 3
    assert svg.count("MiniMol train-all") == 3
    assert svg.count("Morgan KNN k=3") == 3
    assert svg.count("MiniMol KNN k=3") == 3
    assert str(paired) in svg


def test_model_comparison_rejects_stale_paired_delta(tmp_path):
    reference = tmp_path / "blind_20b.tsv"
    candidate = tmp_path / "blind_120b.tsv"
    paired = tmp_path / "paired_ci.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(reference, splits=splits, evaluation_subset="valid")
    _write_metrics(
        candidate,
        splits=splits,
        model_label="GPT-OSS-120B",
        evaluation_subset="valid",
    )
    _write_paired_ci_metrics(paired, delta_offset=0.01)

    with pytest.raises(ValueError, match="Paired-CI delta mismatch"):
        render_model_comparison(
            reference,
            candidate,
            output,
            paired_ci_path=paired,
        )


def test_model_comparison_can_show_only_one_sided_pvalues(tmp_path):
    reference = tmp_path / "blind_20b.tsv"
    candidate = tmp_path / "blind_120b.tsv"
    paired = tmp_path / "paired_stats.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(reference, splits=splits, evaluation_subset="valid")
    _write_metrics(
        candidate,
        splits=splits,
        model_label="GPT-OSS-120B",
        evaluation_subset="valid",
    )
    _write_paired_ci_metrics(paired)

    render_model_comparison(
        reference,
        candidate,
        output,
        paired_ci_path=paired,
        paired_significance_display="pvalue",
    )
    svg = output.read_text(encoding="utf-8")

    assert svg.count("Exploratory one-sided paired permutation p-value") == 3
    assert svg.count("100,000 permutations") == 3
    assert svg.count("p = 0.123") == 9
    assert "Best agent paired Δ · 95% CI" not in svg
    assert "Paired Δ macro-F1" not in svg


def test_model_comparison_can_pair_best_experiment_agent_with_baselines(tmp_path):
    reference = tmp_path / "reference.tsv"
    candidate = tmp_path / "candidate.tsv"
    experiments = tmp_path / "experiments.tsv"
    paired = tmp_path / "paired.tsv"
    output = tmp_path / "comparison.svg"
    splits = (("scaffold", "Scaffold split"),)
    _write_metrics(reference, splits=splits, evaluation_subset="valid")
    _write_metrics(candidate, splits=splits, evaluation_subset="valid")
    _write_experiment_metrics(
        experiments,
        experiment_model_label="DeepSeek-v4-pro",
    )
    _write_paired_ci_metrics(
        paired,
        agent_model="DeepSeek-v4-pro",
        experiment_method="coverage_aware",
    )

    render_model_comparison(
        reference,
        candidate,
        output,
        experiment_paths=(experiments,),
        paired_ci_path=paired,
        paired_significance_display="pvalue",
    )

    svg = output.read_text(encoding="utf-8")
    assert svg.count("Best: DeepSeek-v4-pro · Coverage · coverage-aware") == 3
    assert svg.count("p = 0.123") == 9


def test_pvalue_formatter_does_not_render_small_values_as_zero():
    assert _format_p_value(0.00046) == "p < 0.001"
    assert _format_p_value(0.02955) == "p = 0.030"


def test_single_series_partial_matrix_can_show_pvalues(tmp_path):
    metrics = tmp_path / "current.tsv"
    paired = tmp_path / "paired.tsv"
    output = tmp_path / "comparison.svg"
    _write_metrics(
        metrics,
        splits=(("scaffold", "Scaffold split"),),
        model_label="GPT-OSS-120B",
        evaluation_subset="valid",
    )
    rows = list(csv.DictReader(metrics.open(encoding="utf-8"), delimiter="\t"))
    keep = {
        "none",
        "starling_direct",
        "starling_full_flat",
        "starling_full_mechanism",
        "minimol_train_all",
        "morgan_knn_k3",
        "minimol_embedding_cosine_knn_k3",
    }
    rows = [row for row in rows if row["method"] in keep]
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    _write_paired_ci_metrics(paired)

    render_model_comparison(
        metrics,
        metrics,
        output,
        paired_ci_path=paired,
        paired_significance_display="pvalue",
        single_series=True,
    )

    svg = output.read_text(encoding="utf-8")
    assert svg.count("p = 0.123") == 9
