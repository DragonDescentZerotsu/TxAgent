import argparse
import json
from pathlib import Path

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






def test_progressive_resource_collector_separates_calls_from_carry_forward(tmp_path):
    root = tmp_path / "progressive"
    root.mkdir()
    level_contract = {
        "level": 1,
        "endpoint_group": "direct_brain_exposure",
        "cumulative_physical_assays": 5,
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "conditioned_assay_progressive_visible.v6",
                "n_failed_queries": 0,
                "tasks": ["bbb_martins"],
                "evaluation_subset": "valid",
                "visibility_mode": "deployment_visible_prefetched",
                "reference_pool": "direct_only_heldout_filtered",
                "neighbor_identity_policy": "scaffold_disjoint",
                "selection": {},
                "model": "model",
                "reasoning_effort": "omitted",
                "thinking": "provider_default",
                "evaluation_indices_by_task": {"bbb_martins": [0, 1]},
                "inputs": {"bbb_martins": {"levels": [level_contract]}},
            }
        ),
        encoding="utf-8",
    )
    metrics_dir = root / "bbb_martins/levels/level_1"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "n_total": 2,
                "n_successful": 2,
                "n_failed_runs": 0,
                "n_model_called": 1,
                "macro_f1": 0.6,
                "accuracy": 0.5,
            }
        ),
        encoding="utf-8",
    )
    for query_index, (molecules, cards, status, called) in enumerate(
        ((1, 2, "ok", True), (2, 3, "carried_forward", False))
    ):
        level_dir = (
            root
            / f"bbb_martins/queries/query_idx{query_index:05d}/levels/level_1"
        )
        level_dir.mkdir(parents=True)
        (level_dir / "prepared.json").write_text(
            json.dumps(
                {"n_active_molecules": molecules, "n_active_cards": cards}
            ),
            encoding="utf-8",
        )
        output = {"status": status, "model_called": called}
        if called:
            output["llm"] = {
                "reasoning_content": "x" * 400,
                "usage": {
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens": 200,
                    "completion_tokens": 120,
                },
            }
        (level_dir / "output.json").write_text(
            json.dumps(output), encoding="utf-8"
        )

    rows, summary = plotter.collect_conditioned_progressive_resource_data(
        progressive_root=root
    )

    assert len(rows) == 1
    assert rows[0]["mean_active_molecules"] == 1.5
    assert rows[0]["mean_active_record_cards"] == 2.5
    assert rows[0]["mean_cards_per_active_molecule"] == 1.75
    assert rows[0]["n_model_called"] == 1
    assert rows[0]["model_call_fraction"] == 0.5
    assert rows[0]["mean_reasoning_tokens_per_call"] == 0
    assert rows[0]["mean_reasoning_tokens_per_query"] == 0
    assert rows[0]["n_carried_forward"] == 1
    assert summary["comparison_contract"]["reasoning_mode"][
        "length_statistic"
    ] == "mean reasoning tokens among actual model calls"

    none_root = tmp_path / "none"
    none_dir = none_root / "bbb_martins/none"
    none_dir.mkdir(parents=True)
    complete = {
        "n_total": 2,
        "n_successful": 2,
        "n_evaluable": 2,
        "n_failed_runs": 0,
        "macro_f1": 0.55,
        "accuracy": 0.5,
    }
    (none_dir / "metrics.json").write_text(
        json.dumps(complete), encoding="utf-8"
    )
    baseline_root = tmp_path / "baselines"
    for _, _, relative_path in plotter.CONDITIONED_BASELINES:
        path = baseline_root / "BBB_Martins" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(complete), encoding="utf-8")

    overview_rows, overview_summary = (
        plotter.collect_conditioned_progressive_overview_data(
            progressive_roots_by_task={"bbb_martins": root},
            none_root=none_root,
            baseline_root=baseline_root,
        )
    )
    agent_level = next(
        row
        for row in overview_rows
        if row["result_type"] == "agent_level" and row["level"] == 1
    )
    assert agent_level["mean_cards_per_active_molecule"] == 1.75
    assert len(overview_rows) == 7
    assert overview_summary["comparison_contract"]["figure"] == (
        "conditioned_progressive_overview.v1"
    )
    overview_path = tmp_path / "overview.tsv"
    plotter._write_tsv(overview_path, overview_rows)
    assert "mean_cards_per_active_molecule" in overview_path.read_text(
        encoding="utf-8"
    ).splitlines()[0]


def test_progressive_agent_baseline_collector_supports_task_specific_roots(tmp_path):
    progressive_root = tmp_path / "progressive"
    none_root = tmp_path / "none"
    baseline_root = tmp_path / "baselines"
    progressive_root.mkdir()
    level_contract = {
        "level": 1,
        "endpoint_group": "direct_brain_exposure",
        "cumulative_physical_assays": 5,
    }
    (progressive_root / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "conditioned_assay_progressive_visible.v6",
                "n_failed_queries": 0,
                "tasks": ["bbb_martins"],
                "evaluation_subset": "valid",
                "visibility_mode": "deployment_visible_prefetched",
                "reference_pool": "direct_only_heldout_filtered",
                "neighbor_identity_policy": "scaffold_disjoint",
                "selection": {},
                "evaluation_indices_by_task": {"bbb_martins": [0, 1]},
                "inputs": {"bbb_martins": {"levels": [level_contract]}},
            }
        ),
        encoding="utf-8",
    )
    complete = {
        "n_total": 2,
        "n_successful": 2,
        "n_evaluable": 2,
        "n_failed_runs": 0,
        "macro_f1": 0.6,
        "accuracy": 0.5,
    }
    level_dir = progressive_root / "bbb_martins/levels/level_1"
    none_dir = none_root / "bbb_martins/none"
    level_dir.mkdir(parents=True)
    none_dir.mkdir(parents=True)
    (level_dir / "metrics.json").write_text(
        json.dumps({**complete, "macro_f1": 0.7}), encoding="utf-8"
    )
    (none_dir / "metrics.json").write_text(
        json.dumps(complete), encoding="utf-8"
    )
    for _, _, relative_path in plotter.CONDITIONED_BASELINES:
        path = baseline_root / "BBB_Martins" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(complete), encoding="utf-8")

    rows, summary = plotter.collect_conditioned_progressive_agent_baseline_data(
        progressive_roots_by_task={"bbb_martins": progressive_root},
        none_root=none_root,
        baseline_root=baseline_root,
    )

    assert [row["result_type"] for row in rows[:2]] == [
        "agent_level",
        "agent_level",
    ]
    assert [row["level"] for row in rows[:2]] == [0, 1]
    assert rows[1]["plot_label"] == "L1 (all)"
    assert len([row for row in rows if row["result_type"] == "baseline"]) == 5
    contract = summary["comparison_contract"]
    assert contract["tasks"] == ["bbb_martins"]
    assert contract["task_contracts"]["bbb_martins"]["n"] == 2

    alternate_root = tmp_path / "alternate_baselines"
    for _, _, relative_path in plotter.CONDITIONED_BASELINES:
        path = alternate_root / "BBB_Martins" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**complete, "macro_f1": 0.8}), encoding="utf-8"
        )
    override_rows, override_summary = (
        plotter.collect_conditioned_progressive_agent_baseline_data(
            progressive_roots_by_task={"bbb_martins": progressive_root},
            none_root=none_root,
            baseline_root=baseline_root,
            baseline_roots_by_task={"bbb_martins": alternate_root},
        )
    )
    override_baselines = [
        row for row in override_rows if row["result_type"] == "baseline"
    ]
    assert {row["macro_f1"] for row in override_baselines} == {0.8}
    assert override_summary["comparison_contract"]["baseline_lineage_by_task"] == {
        "bbb_martins": str(alternate_root)
    }

    mismatched = {**complete, "n_total": 3, "n_successful": 3, "n_evaluable": 3}
    for _, _, relative_path in plotter.CONDITIONED_BASELINES:
        path = baseline_root / "BBB_Martins" / relative_path
        path.write_text(json.dumps(mismatched), encoding="utf-8")

    omitted_rows, omitted_summary = (
        plotter.collect_conditioned_progressive_agent_baseline_data(
            progressive_roots_by_task={"bbb_martins": progressive_root},
            none_root=none_root,
            baseline_root=baseline_root,
            omit_mismatched_baselines=True,
        )
    )
    assert not [row for row in omitted_rows if row["result_type"] == "baseline"]
    omissions = omitted_summary["comparison_contract"]["baseline_omissions"]
    assert len(omissions) == len(plotter.CONDITIONED_BASELINES)
    assert {item["reason"] for item in omissions} == {"sample_count_mismatch"}
    assert {item["baseline_n"] for item in omissions} == {3}
    assert {item["agent_n"] for item in omissions} == {2}


def test_progressive_baseline_collector_accepts_current_minimol_head_path(tmp_path):
    baseline_root = tmp_path / "baselines"
    complete = {
        "n_total": 2,
        "n_evaluated": 2,
        "macro_f1": 0.7,
        "accuracy": 0.75,
    }
    for method, _, relative_path in plotter.CONDITIONED_BASELINES:
        if method == "minimol_head":
            relative_path = Path("minimol_head/final/metrics.json")
        path = baseline_root / "Bioavailability_Ma" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(complete), encoding="utf-8")

    rows: list[dict[str, object]] = []
    omissions = plotter._append_conditioned_baselines(
        rows,
        task="bioavailability_ma",
        task_label="Bioavailability",
        baseline_task="Bioavailability_Ma",
        baseline_root=baseline_root,
        expected_n=2,
    )

    assert not omissions
    assert len(rows) == len(plotter.CONDITIONED_BASELINES)
    head = next(row for row in rows if row["method"] == "minimol_head")
    assert head["metrics_path"].endswith("minimol_head/final/metrics.json")


def test_parse_task_path_overrides():
    assert plotter._parse_task_path_overrides(
        ["bbb_martins=/tmp/bbb", "skin_reaction=/tmp/skin"]
    ) == {
        "bbb_martins": Path("/tmp/bbb"),
        "skin_reaction": Path("/tmp/skin"),
    }
    with pytest.raises(ValueError, match="known conditioned task"):
        plotter._parse_task_path_overrides(["unknown=/tmp/value"])
    with pytest.raises(ValueError, match="Duplicate"):
        plotter._parse_task_path_overrides(
            ["bbb_martins=/tmp/one", "bbb_martins=/tmp/two"]
        )
