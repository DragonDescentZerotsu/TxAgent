import csv

from tools.chembl_tool.paper_experiments.plot_retrieval_claims_overview import (
    TASKS,
    render,
)


def _write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_valid_chart_uses_split_sizes_and_labels(tmp_path):
    analysis_dir = tmp_path / "analysis"
    experiment_rows = []
    parent_rows = []
    valid_sizes = {
        "bbb_martins": 195,
        "skin_reaction": 40,
        "clintox": 139,
        "bioavailability_ma": 64,
    }
    for task in TASKS:
        for index, condition in enumerate(task.conditions):
            experiment = f"{task.key}__{condition.suffix}"
            for visibility_mode, prefix in (
                ("identity_blind", ""),
                ("deployment_visible", "deployment_visible__"),
            ):
                experiment_rows.append(
                    {
                        "experiment": f"{prefix}{experiment}",
                        "visibility_mode": visibility_mode,
                        "task": task.key,
                        "n_total": valid_sizes[task.key],
                        "macro_f1": 0.5 + index / 100,
                    }
                )
            if condition.suffix != "none":
                parent_rows.append(
                    {
                        "experiment": experiment,
                        "parent_disjoint_macro_f1": 0.51 + index / 100,
                    }
                )

    _write_tsv(
        analysis_dir / "experiment_summary.tsv",
        ["experiment", "visibility_mode", "task", "n_total", "macro_f1"],
        experiment_rows,
    )
    _write_tsv(
        analysis_dir / "parent_disjoint_ablation/condition_results.tsv",
        ["experiment", "parent_disjoint_macro_f1"],
        parent_rows,
    )

    output = analysis_dir / "figures/retrieval_claims_overview.svg"
    render(analysis_dir, output, data_split="valid")
    svg = output.read_text(encoding="utf-8")

    assert "Values are valid macro-F1" in svg
    for title in (
        "BBB penetration",
        "Skin reaction",
        "Clinical toxicity",
        "Oral bioavailability",
    ):
        assert f">{title}</text>" in svg
    for size in valid_sizes.values():
        assert f">n = {size}</text>" in svg
