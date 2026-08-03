import csv

import pytest

from tools.chembl_tool.paper_experiments.plot_starling_benchmark_overview import (
    SPLITS,
    TASKS,
    render,
)


def _write_metrics(path, *, omit=None, failed=None):
    rows = []
    for split, _ in SPLITS:
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
                        "n_test": 500 if task.key == "bbb_martins" else 380,
                        "n_failed": 1 if key == failed else 0,
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
                "n_test",
                "n_failed",
                "macro_f1",
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
