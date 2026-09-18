import json
from pathlib import Path

from tools.chembl_tool.paper_experiments.analyze_progressive_molecule_mentions import (
    run_analysis,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_molecule_mentions_use_only_separate_model_authored_streams(tmp_path):
    root = tmp_path / "run"
    _write_json(root / "matrix.json", {"settings": {"prompt": {"sha256": "prompt"}}})
    level = (
        root
        / "conditions/morgan_assay-transfer-trained_records50_query_prior"
        / "bbb_martins/queries/query_idx00000/levels/level_1"
    )
    _write_json(level / "prepared.json", {"n_active_molecules": 10})
    _write_json(
        level / "output.json",
        {
            "status": "ok",
            "llm": {
                "reasoning_content": "MOLECULE 1; molecule 10; MoLeCuLe 1",
                "raw_content": '{"summary":"molecule 2"}',
                "messages": [
                    {"content": " ".join(f"Molecule {number}" for number in range(1, 11))}
                ],
                "content": {"summary": "molecule 3"},
            },
            "state": {"decision_summary": "molecule 4"},
        },
    )
    missing = level.parents[2] / "query_idx00001/levels/level_1"
    _write_json(missing / "prepared.json", {"n_active_molecules": 10})

    output_dir = tmp_path / "analysis"
    manifest = run_analysis(
        [("v12", root)], {("bbb_martins", "cached")}, output_dir
    )
    rows = [
        row
        for row in _read_tsv(output_dir / "mention_coverage_by_position.tsv")
        if row["stream"] == "reasoning"
    ]
    by_position = {int(row["molecule_position"]): row for row in rows}
    assert by_position[1]["n_mentions"] == "2"
    assert by_position[10]["n_mentions"] == "1"
    assert by_position[2]["n_mentions"] == "0"

    final_rows = [
        row
        for row in _read_tsv(output_dir / "mention_coverage_by_position.tsv")
        if row["stream"] == "final"
    ]
    final_by_position = {int(row["molecule_position"]): row for row in final_rows}
    assert final_by_position[2]["n_mentions"] == "1"
    assert final_by_position[3]["n_mentions"] == "0"
    assert {row["stream"] for row in rows + final_rows} == {"reasoning", "final"}
    assert manifest["n_expected_by_version"] == {"v12": 2}
    assert manifest["n_available_by_version"] == {"v12": 1}
    assert manifest["sources"][0]["n_missing"] == 1
    assert manifest["sources"][0]["unavailable_traces"] == [
        {
            "version": "v12",
            "task": "bbb_martins",
            "retrieval": "morgan",
            "query_prior": "cached",
            "query_index": 1,
            "status": "missing",
        }
    ]
    assert (output_dir / "REPORT.md").is_file()
    summary = _read_tsv(output_dir / "molecule_mention_coverage_summary.tsv")
    reasoning_summary = next(row for row in summary if row["stream"] == "reasoning")
    assert reasoning_summary["n_expected_traces"] == "2"
    assert reasoning_summary["n_available_traces"] == "1"
    assert reasoning_summary["molecule_1_mentions"] == "2"
    assert reasoning_summary["molecule_10_mentions"] == "1"
    assert reasoning_summary["all_ten_coverage_fraction"] == "0.0"
    assert reasoning_summary["zero_position_coverage_fraction"] == "0.0"
    averages = _read_tsv(output_dir / "average_molecules_mentioned_per_query.tsv")
    assert averages == [
        {
            "condition": "v12_morgan",
            "average_number_of_molecules_mentioned_per_query": "2.0",
        }
    ]
    assert manifest["analysis"]["single_column_row_order"] == [
        {"version": "v12", "retrieval": "morgan"}
    ]
    distribution = _read_tsv(output_dir / "trace_coverage_distribution.tsv")
    assert len(distribution) == 22
    assert {(row["task"], row["query_prior"]) for row in distribution} == {
        ("bbb_martins", "cached")
    }
    assert (output_dir / "manifest.json").is_file()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
