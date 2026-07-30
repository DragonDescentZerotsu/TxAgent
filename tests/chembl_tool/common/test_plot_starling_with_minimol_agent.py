import csv
import json

import pytest

from tools.chembl_tool.paper_experiments import (
    plot_starling_with_minimol_agent as plot,
)


def _write_existing(path):
    rows = []
    for split, _ in plot.SPLITS:
        for task in plot.TASKS:
            for index, method in enumerate(task.methods):
                rows.append(
                    {
                        "benchmark_split": split,
                        "task": task.key,
                        "method": method.key,
                        "neighbor_identity_policy": (
                            "operational" if method.key == "none" else "parent_disjoint"
                        ),
                        "n_test": 500,
                        "n_failed": 0,
                        "macro_f1": 0.5 + index / 100,
                        "metrics_path": f"existing/{split}/{task.key}/{method.key}",
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _operational_rows(*, failed=False):
    rows = {}
    for split, _ in plot.SPLITS:
        for task in plot.TASKS:
            for index, method in enumerate(task.methods):
                if method.source not in plot.AGENT_SOURCES:
                    continue
                rows[(split, task.key, method.key)] = {
                    "n_evaluable": 500,
                    "n_failed_runs": int(
                        failed
                        and split == "random"
                        and task.key == "bbb_martins"
                        and method.key == "chembl_direct"
                    ),
                    "macro_f1": 0.6 + index / 100,
                    "metrics_path": f"minimol/{split}/{task.key}/{method.key}",
                }
    return rows


def test_render_comparison_and_data(monkeypatch, tmp_path):
    metrics = tmp_path / "metrics.tsv"
    output = tmp_path / "comparison.svg"
    data_output = tmp_path / "comparison.tsv"
    _write_existing(metrics)
    monkeypatch.setattr(plot, "read_minimol_operational", _operational_rows)

    plot.render(metrics, output, data_output=data_output)

    svg = output.read_text(encoding="utf-8")
    assert "Starling Benchmark with MiniMol Agent Retrieval" in svg
    assert svg.count("Morgan agent · parent-disjoint") == 1
    assert svg.count("MiniMol agent · operational") == 1
    assert "0.600" in svg
    rows = list(csv.DictReader(data_output.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 100
    assert {row["result_series"] for row in rows} == {
        "No retrieval",
        "Morgan agent retrieval",
        "MiniMol agent retrieval",
        "MiniMol train-all head",
        "Morgan KNN",
        "MiniMol KNN",
    }


def test_operational_reader_rejects_failed_rows(monkeypatch, tmp_path):
    for split, _ in plot.SPLITS:
        root = tmp_path / split
        for task in plot.TASKS:
            for method in task.methods:
                if method.source not in plot.AGENT_SOURCES:
                    continue
                batch = (
                    root
                    / "runs_deployment_visible"
                    / task.key
                    / f"{task.key}__{method.key}"
                )
                batch.mkdir(parents=True)
                failed = (
                    split == "random"
                    and task.key == "bbb_martins"
                    and method.key == "chembl_direct"
                )
                (batch / "metrics.json").write_text(
                    json.dumps(
                        {
                            "n_evaluable": 500,
                            "n_failed_runs": int(failed),
                            "macro_f1": 0.6,
                        }
                    )
                )
    monkeypatch.setattr(
        plot,
        "paper_root_for_minimol_retrieval",
        lambda split: tmp_path / split,
    )

    with pytest.raises(ValueError, match="rows with failures"):
        plot.read_minimol_operational()
