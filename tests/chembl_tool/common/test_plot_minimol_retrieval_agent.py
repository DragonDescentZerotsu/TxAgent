import csv

import pytest

from tools.chembl_tool.paper_experiments.plot_minimol_retrieval_agent import (
    SPLITS,
    TASKS,
    read_results,
    render,
)


def _write_metrics(path, *, omit=None, failed=None):
    rows = []
    for split, _ in SPLITS:
        for task in TASKS:
            for condition, _ in task.conditions:
                key = (split, task.key, condition)
                if key == omit:
                    continue
                rows.append(
                    {
                        "split": split,
                        "task": task.key,
                        "condition": condition,
                        "n_total": 500,
                        "morgan_macro_f1": 0.6,
                        "minimol_macro_f1": 0.62,
                        "morgan_n_failed": int(key == failed),
                        "minimol_n_failed": 0,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_render_complete_comparison(tmp_path):
    metrics = tmp_path / "metrics.tsv"
    output = tmp_path / "figure.svg"
    _write_metrics(metrics)

    render(metrics, output)

    svg = output.read_text(encoding="utf-8")
    assert "Retrieval Feature Ablation" in svg
    assert "MiniMol embedding" in svg
    assert "+0.020" in svg


def test_read_results_rejects_missing_or_failed_rows(tmp_path):
    metrics = tmp_path / "metrics.tsv"
    first = ("random", "bbb_martins", "chembl_direct")
    _write_metrics(metrics, omit=first)
    with pytest.raises(ValueError, match="Missing"):
        read_results(metrics)

    _write_metrics(metrics, failed=first)
    with pytest.raises(ValueError, match="failed runs"):
        read_results(metrics)
