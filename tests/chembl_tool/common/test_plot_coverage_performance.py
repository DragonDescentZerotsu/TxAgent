import csv

from tools.chembl_tool.paper_experiments.plot_coverage_performance import render


def test_coverage_chart_renders_split_condition_and_class_denominators(tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    path = analysis_dir / "coverage_performance.tsv"
    rows = [
        {
            "experiment": "deployment_visible__bbb_martins__chembl_direct",
            "task": "bbb_martins",
            "visibility_mode": "deployment_visible",
            "condition": "chembl_direct",
            "condition_label": "ChEMBL · Direct",
            "source": "chembl",
            "retrieval_view": "direct",
            "n_total": 4,
            "n_paired": 4,
            "n_covered": 3,
            "coverage": 0.75,
            "n_negative": 2,
            "n_negative_covered": 1,
            "negative_coverage": 0.5,
            "n_positive": 2,
            "n_positive_covered": 2,
            "positive_coverage": 1.0,
            "baseline_macro_f1": 0.4,
            "retrieval_macro_f1": 0.6,
            "delta_macro_f1": 0.2,
            "delta_ci_low": 0.01,
            "delta_ci_high": 0.3,
        },
        {
            "experiment": "deployment_visible__skin_reaction__starling_full_flat",
            "task": "skin_reaction",
            "visibility_mode": "deployment_visible",
            "condition": "starling_full_flat",
            "condition_label": "Starling · Full / Flat",
            "source": "starling",
            "retrieval_view": "full_flat",
            "n_total": 6,
            "n_paired": 6,
            "n_covered": 4,
            "coverage": 0.6667,
            "n_negative": 4,
            "n_negative_covered": 3,
            "negative_coverage": 0.75,
            "n_positive": 2,
            "n_positive_covered": 1,
            "positive_coverage": 0.5,
            "baseline_macro_f1": 0.6,
            "retrieval_macro_f1": 0.55,
            "delta_macro_f1": -0.05,
            "delta_ci_low": -0.2,
            "delta_ci_high": 0.1,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    output = analysis_dir / "figures/coverage_performance_relationship.svg"
    render(analysis_dir, output, data_split="valid")
    svg = output.read_text(encoding="utf-8")

    assert "Retrieval Coverage and Macro-F1 Change" in svg
    assert "valid split" in svg
    assert "ChEMBL · Direct" in svg
    assert ">ChEMBL</text>" in svg
    assert ">Starling</text>" in svg
    assert 'data-source="chembl"' in svg
    assert 'data-source="starling"' in svg
    assert 'stroke-dasharray="7 5"' in svg
    assert "n0=2 · n1=2" in svg
