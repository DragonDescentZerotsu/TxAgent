import argparse

from tools.chembl_tool.paper_experiments import plot_assay_retrieval_curve as plotter
from tools.chembl_tool.paper_experiments import run_assay_retrieval_curve as runner


def test_prefix_manifest_supports_legacy_and_task_specific_schedules():
    assert plotter._prefixes_by_task(
        {"tasks": ["a", "b"], "prefixes": [5, 20]}
    ) == {"a": [5, 20], "b": [5, 20]}
    assert plotter._prefixes_by_task(
        {
            "tasks": ["a", "b"],
            "prefixes_by_task": {"a": [5, 20, 21], "b": [5, 9]},
        }
    ) == {"a": [5, 20, 21], "b": [5, 9]}


def test_retrieval_counts_merge_molecules_and_sum_represented_records():
    retrieval = {
        "groups": [
            {
                "neighbors": [
                    {
                        "molecule_chembl_id": "m1",
                        "evidence_rows": [
                            {"source_record_count": 2},
                            {"source_record_count": 3},
                        ],
                    },
                    {
                        "molecule_chembl_id": "m2",
                        "evidence_rows": [{"source_record_count": 7}],
                    },
                ]
            }
        ]
    }
    assert plotter._retrieval_counts(retrieval) == (2, 3, 12)


def test_runner_default_prefix_plan_is_geometric_and_ends_at_each_catalog(monkeypatch):
    totals = {"bbb_martins": 22820, "bioavailability_ma": 1840}
    monkeypatch.setattr(runner, "_catalog_size", totals.__getitem__)
    assert runner.prefix_plan(list(totals)) == {
        "bbb_martins": [5, 20, 80, 320, 1280, 5120, 20480, 22820],
        "bioavailability_ma": [5, 20, 80, 320, 1280, 1840],
    }


def test_runner_explicit_prefixes_still_append_full_catalog(monkeypatch):
    monkeypatch.setattr(runner, "_catalog_size", lambda task: 100)
    assert runner.prefix_plan(["bbb_martins"], [5, 20, 500]) == {
        "bbb_martins": [5, 20, 100]
    }


def test_runner_reuses_single_analysis_from_actual_first_prefix(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "valid.jsonl"
    input_jsonl.write_text('{"drug":"CC"}\n', encoding="utf-8")
    monkeypatch.setitem(
        runner.TASKS,
        "bbb_martins",
        {
            "input": str(input_jsonl),
            "index": "index.pkl",
            "module": "task.module",
            "replay_root": "replays",
            "ranked_assays": "ranked.jsonl",
        },
    )
    args = argparse.Namespace(
        tasks=["bbb_martins"],
        output_root=str(tmp_path / "out"),
        include_complete_batches=False,
        python_executable="python",
        model="model",
        base_url="http://localhost/v1",
        api_key_env="KEY",
        tool_service_url="http://localhost/tools",
        max_tokens=100,
        timeout_s=30,
        max_stage_requeues=1,
        worst_case_smoke=False,
        limit=0,
        indices=[],
    )

    commands = runner._commands(args, {"bbb_martins": [7, 28]})
    second = commands[1].command

    source_index = second.index("--single-analysis-source-batch") + 1
    assert second[source_index].endswith("bbb_martins/assay_flat_top7")
