import argparse
import json

import pytest

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


def test_relevance_decay_uses_cumulative_top_n_mean(tmp_path):
    ranked = tmp_path / "ranked.jsonl"
    ranked.write_text(
        "\n".join(
            json.dumps({"relevance_score": score})
            for score in (100, 90, 70, 20)
        )
        + "\n",
        encoding="utf-8",
    )

    rows = plotter.collect_relevance_decay_data(
        ranked_assay_paths={"bbb_martins": ranked},
        prefixes_by_task={"bbb_martins": [2, 4]},
    )

    assert [row["mean_relevance_score"] for row in rows] == [95, 70]
    assert [row["boundary_relevance_score"] for row in rows] == [90, 20]


def test_analysis_records_actual_assay_retrieval_contract(tmp_path):
    plotter.write_analysis(
        analysis_dir=tmp_path,
        performance_rows=[{"task": "bbb_martins"}],
        retrieval_rows=[{"task": "bbb_martins"}],
        summary={
            "experiment": {
                "retrieval_contract": (
                    "direct_only_heldout_filtered_scaffold_disjoint"
                ),
                "reference_pool": "direct_only_heldout_filtered",
                "neighbor_identity_policy": "scaffold_disjoint",
            },
            "best": [],
        },
    )

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert (
        payload["comparison_contract"]["assay_reference_pool"]
        == "direct_only_heldout_filtered"
    )
    assert payload["comparison_contract"]["assay_retrieval_contract"] == (
        "direct_only_heldout_filtered_scaffold_disjoint"
    )
    assert (
        payload["comparison_contract"]["neighbor_identity_policy"]
        == "scaffold_disjoint"
    )


def test_incomplete_curve_gate_requires_every_zero_failure_prefix():
    experiment = {
        "tasks": ["bbb_martins"],
        "prefixes_by_task": {"bbb_martins": [5, 20, 80]},
    }
    rows = [
        {
            "task": "bbb_martins",
            "condition": "none_reused",
            "assay_count": 0,
            "n_total": 2,
            "n_failed_runs": 0,
        },
        {
            "task": "bbb_martins",
            "condition": "assay_level",
            "assay_count": 5,
            "n_total": 2,
            "n_failed_runs": 0,
        },
        {
            "task": "bbb_martins",
            "condition": "assay_level",
            "assay_count": 20,
            "n_total": 2,
            "n_failed_runs": 1,
        },
    ]

    assert plotter.incomplete_curve_conditions(rows, experiment) == [
        "bbb_martins__assay_flat_top20",
        "bbb_martins__assay_flat_top80",
    ]


def test_runner_default_prefix_plan_is_geometric_and_ends_at_each_catalog(monkeypatch):
    totals = {"bbb_martins": 22820, "bioavailability_ma": 1840}
    monkeypatch.setattr(runner, "_catalog_size", totals.__getitem__)
    assert runner.prefix_plan(list(totals)) == {
        "bbb_martins": [5, 20, 80, 320, 1280, 5120, 22820],
        "bioavailability_ma": [5, 20, 80, 320, 1840],
    }


def test_runner_default_is_direct_filtered_with_query_scaffold_disjointness():
    assert runner.DEFAULT_RETRIEVAL_CONTRACT == (
        "direct_only_heldout_filtered_scaffold_disjoint"
    )
    contract = runner.RETRIEVAL_CONTRACTS[runner.DEFAULT_RETRIEVAL_CONTRACT]
    spec = runner._task_spec("bbb_martins", runner.DEFAULT_RETRIEVAL_CONTRACT)
    assert "starling_assay_retrieval_v2" in spec["index"]
    assert "direct_only_heldout_filtered" in spec["index"]
    assert "starling_assay_retrieval_v5" in spec["replay_root"]
    assert contract.reference_pool == "direct_only_heldout_filtered"
    assert contract.neighbor_identity_policy == "scaffold_disjoint"


def test_runner_explicit_prefixes_still_append_full_catalog(monkeypatch):
    monkeypatch.setattr(runner, "_catalog_size", lambda task: 100)
    assert runner.prefix_plan(["bbb_martins"], [5, 20, 500]) == {
        "bbb_martins": [5, 20, 100]
    }


def test_runner_exact_prefixes_do_not_append_full_catalog(monkeypatch):
    monkeypatch.setattr(runner, "_catalog_size", lambda task: 100)
    assert runner.prefix_plan(
        ["bbb_martins"], [20], exact=True
    ) == {"bbb_martins": [20]}


def test_runner_manifest_scope_accumulates_compatible_resume_launches():
    shared = {
        "experiment": "starling_assay_retrieval_curve.v2",
        "evaluation_subset": "valid",
        "reference_pool": "direct_only_heldout_filtered",
        "model": "model",
        "visibility_mode": "identity_blind",
        "neighbor_identity_policy": "parent_disjoint",
        "top_k_per_assay": 3,
        "min_similarity": 0.3,
        "reasoning_effort": "",
        "thinking": "disabled",
    }
    previous = {
        **shared,
        "tasks": ["bbb_martins"],
        "prefixes_by_task": {"bbb_martins": [5, 20]},
        "assay_catalog_size_by_task": {"bbb_martins": 100},
        "inputs": {"bbb_martins": {"assay_index_sha256": "bbb"}},
        "launched_conditions": ["bbb_martins__assay_flat_top5"],
        "launched_single_source_by_condition": {"bbb_martins__assay_flat_top5": ""},
        "replay_audit": {"bbb_martins": {"prefixes": {"5": {"ok": True}}}},
    }
    current = {
        **shared,
        "tasks": ["bbb_martins", "bioavailability_ma"],
        "prefixes_by_task": {
            "bbb_martins": [80],
            "bioavailability_ma": [5],
        },
        "assay_catalog_size_by_task": {
            "bbb_martins": 100,
            "bioavailability_ma": 50,
        },
        "inputs": {
            "bbb_martins": {"assay_index_sha256": "bbb"},
            "bioavailability_ma": {"assay_index_sha256": "bio"},
        },
        "launched_conditions": ["bbb_martins__assay_flat_top80"],
        "launched_single_source_by_condition": {
            "bbb_martins__assay_flat_top80": "source"
        },
        "replay_audit": {
            "bbb_martins": {"prefixes": {"80": {"ok": True}}},
            "bioavailability_ma": {"prefixes": {"5": {"ok": True}}},
        },
    }

    merged = runner._merge_manifest_scope(previous, current)

    assert merged["tasks"] == ["bbb_martins", "bioavailability_ma"]
    assert merged["prefixes_by_task"] == {
        "bbb_martins": [5, 20, 80],
        "bioavailability_ma": [5],
    }
    assert set(merged["replay_audit"]["bbb_martins"]["prefixes"]) == {
        "5",
        "80",
    }
    assert merged["manifest_scope"] == "cumulative_compatible_launches"


def test_runner_manifest_scope_rejects_mixed_reference_pools():
    previous = {"reference_pool": "train_only"}
    current = {"reference_pool": "direct_only_heldout_filtered"}

    with pytest.raises(ValueError, match="different reference_pool"):
        runner._merge_manifest_scope(previous, current)


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


def test_runner_exact_resume_reuses_existing_complete_single_batch(tmp_path, monkeypatch):
    input_jsonl = tmp_path / "valid.jsonl"
    input_jsonl.write_text('{"drug":"CC"}\n{"drug":"CCC"}\n', encoding="utf-8")
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
    output_root = tmp_path / "out"
    source_batch = output_root / "bbb_martins" / "assay_flat_top5"
    source_batch.mkdir(parents=True)
    (source_batch / "manifest.json").write_text(
        json.dumps(
            {
                "input_jsonl": str(input_jsonl),
                "model": "model",
                "visibility_mode": "identity_blind",
                "harness_prefetch_tools": True,
                "n_items": 2,
            }
        ),
        encoding="utf-8",
    )
    for index in range(2):
        run_dir = source_batch / "runs" / f"assay_flat_top5_idx{index:05d}"
        run_dir.mkdir(parents=True)
        (run_dir / "single_molecule_reasoning_output.json").write_text(
            '{"status":"ok"}', encoding="utf-8"
        )
    args = argparse.Namespace(
        tasks=["bbb_martins"],
        output_root=str(output_root),
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

    command = runner._commands(args, {"bbb_martins": [28]})[0].command

    source_index = command.index("--single-analysis-source-batch") + 1
    assert command[source_index] == str(source_batch)


def test_best_panel_xlim_adapts_to_highest_score_and_leaves_annotation_room():
    rows = [
        {
            "group_best_macro_f1": 0.676241,
            "assay_best_macro_f1": 0.720856,
        },
        {
            "group_best_macro_f1": 0.616345,
            "assay_best_macro_f1": 0.626714,
        },
    ]

    lower, upper = plotter._best_panel_xlim(rows)

    assert lower == 0.60
    assert upper == 0.74
    assert upper > max(row["assay_best_macro_f1"] for row in rows) + 0.002


def test_best_panel_xlim_scales_beyond_previous_fixed_limit():
    _, upper = plotter._best_panel_xlim(
        [{"group_best_macro_f1": 0.81, "assay_best_macro_f1": 0.86}]
    )

    assert upper == 0.88


def test_conditioned_metric_readers_cover_agent_knn_and_minimol_head_receipts():
    assert plotter._metric_n({"n_total": 398}) == 398
    assert plotter._metric_n({"n_evaluated": 262}) == 262
    assert plotter._metric_n({"splits": {"evaluation": 246}}) == 246
    assert plotter._metric_value({"macro_f1": 0.62}, "macro_f1") == 0.62
    assert (
        plotter._metric_value(
            {"evaluation_metrics": {"macro_f1": 0.69}}, "macro_f1"
        )
        == 0.69
    )


def test_conditioned_agent_inclusion_gate_requires_full_zero_failure_metrics():
    assert (
        plotter._agent_metric_issue(
            {"n_total": 398, "n_failed_runs": 0}, expected_n=398
        )
        == ""
    )
    assert "n_failed_runs=2" == plotter._agent_metric_issue(
        {"n_total": 398, "n_failed_runs": 2}, expected_n=398
    )
    assert "expected=398" in plotter._agent_metric_issue(
        {"n_total": 397, "n_failed_runs": 0}, expected_n=398
    )


def test_conditioned_comparison_ylim_is_shared_and_data_driven():
    assert plotter._conditioned_comparison_ylim(
        [{"macro_f1": 0.506}, {"macro_f1": 0.728}]
    ) == (0.45, 0.8)


def test_conditioned_paired_predictions_preserve_same_molecule_condition_rows(
    tmp_path,
):
    agent_dir = tmp_path / "agent"
    baseline_dir = tmp_path / "baseline"
    agent_dir.mkdir()
    baseline_dir.mkdir()
    (agent_dir / "predictions.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "query_index": index,
                    "smiles": "CC",
                    "label": label,
                    "pred_label": prediction,
                    "final_status": "ok",
                }
            )
            for index, (label, prediction) in enumerate(((0, 0), (1, 1)))
        )
        + "\n",
        encoding="utf-8",
    )
    (baseline_dir / "valid_predictions.jsonl").write_text(
        "\n".join(
            json.dumps({"drug": "CC", "Y": label, "prediction": 0})
            for label in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )

    labels, agent, baseline, _, _ = plotter._read_paired_conditioned_predictions(
        agent_dir / "metrics.json", baseline_dir / "metrics.json"
    )

    assert labels.tolist() == [0, 1]
    assert agent.tolist() == [0, 1]
    assert baseline.tolist() == [0, 0]
