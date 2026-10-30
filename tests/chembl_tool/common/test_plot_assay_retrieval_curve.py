import argparse
import json
from pathlib import Path

import pytest

from tools.chembl_tool.paper_experiments import plot_assay_retrieval_curve as plotter
from tools.chembl_tool.paper_experiments import run_assay_retrieval_curve as runner


def test_indirect_comparison_means_and_square_export(tmp_path):
    from PIL import Image
    source, baseline = tmp_path/'ours.csv', tmp_path/'naive.csv'
    source.write_text('task,10,25\n' + ''.join(
        f'{task},{0.1 * i},0.8\n' for i, task in enumerate(plotter.TASK_SPECS)))
    baseline.write_text('task,10,25\n' + ''.join(
        f'{task},0.2,0.2\n' for task in plotter.TASK_SPECS))
    paths = plotter.plot_indirect_budget_comparison(source, baseline, tmp_path, output_stem='test')
    receipt = json.loads(Path(paths['receipt']).read_text())
    assert receipt['mean']['ours'] == pytest.approx([0.25, 0.8])
    assert receipt['mean']['naive'] == pytest.approx([0.2, 0.2])
    assert Image.open(paths['png']).size == (990, 990)
    assert receipt['point_count_per_method'] == 12
    for key in ('png', 'pdf', 'svg'):
        assert plotter.sha256_file(Path(paths[key])) == receipt['output_sha256'][key]
    source.write_text(baseline.read_text())  # Flat equal curves must retain a valid y range.
    plotter.plot_indirect_budget_comparison(source, baseline, tmp_path, output_stem='flat')


def test_indirect_comparison_rejects_missing_or_mismatched_inputs(tmp_path):
    with pytest.raises(ValueError, match='requires --indirect-budget-csv'):
        plotter.main(['--indirect-budget-baseline-csv', str(tmp_path/'missing.csv')])
    source, baseline = tmp_path/'ours.csv', tmp_path/'naive.csv'
    rows = ''.join(f'{task},0.5,0.6\n' for task in plotter.TASK_SPECS)
    source.write_text('task,10,25\n'+rows)
    baseline.write_text('task,10,50\n'+rows)
    with pytest.raises(ValueError, match='same budgets'):
        plotter.plot_indirect_budget_comparison(source, baseline, tmp_path, output_stem='bad')


@pytest.mark.parametrize('subset,visibility,label', [
    ('valid', 'deployment_visible_prefetched', 'identity visible'),
    ('test', 'identity_blind', 'identity blind'),
])
def test_budgeted_curve_labels_follow_contract(tmp_path, monkeypatch, subset, visibility, label):
    from tools.chembl_tool.paper_experiments import plot_starling_model_comparison as overview
    methods = [m for m, _, _ in plotter.CONDITIONED_BASELINES]
    methods += ['none', 'direct_fill', 'both_fill', 'direct_cap', 'both_cap', 'both_group']
    rows = [{'task': 'ames', 'method': m, 'family': 'fixture', 'n': 2,
             'macro_f1': .65, 'accuracy': .7} for m in methods]
    receipt = {'conditions': {'ames': {'n': 2}}, 'split_scheme': 'scaffold',
               'evaluation_subset': subset, 'visibility_mode': visibility,
               'min_similarity': 0, 'direct_budget': 50, 'indirect_budget': 50, 'tool_failures': 0}
    monkeypatch.setattr(overview, 'load_budgeted_comparison', lambda *a: (rows, receipt))
    paths = plotter.plot_budgeted_baseline_curves(
        tmp_path, tmp_path/'audit.json', tmp_path, output_stem='curves')
    svg = Path(paths['svg']).read_text()
    assert label in svg and 'no similarity cutoff' in svg
    assert f"same {'validation' if subset == 'valid' else 'test'} rows" in svg
    assert len(Path(paths['csv']).read_text().splitlines()) == 12


def test_scaling_curve_uses_declared_budget_stages(tmp_path):
    methods = [m for m, _, _ in plotter.CONDITIONED_BASELINES] + ['none','d25','d50','d100']
    rows = [dict(task='ames',method=m,family='fixture',n=2,macro_f1=.65,accuracy=.7) for m in methods]
    receipt = dict(conditions={'ames':{'n':2}}, split_scheme='scaffold',evaluation_subset='valid',
        visibility_mode='deployment_visible_prefetched',min_similarity=0,direct_budget='25/50/100',
        indirect_budget=0,tool_failures=0,title='Direct record scaling',
        stage_labels=['None','D25','D50','D100'],curve_settings={'Fill':['none','d25','d50','d100']},
        budget_label='Direct sweep; no indirect records')
    paths = plotter.plot_baseline_curve_points(rows,receipt,tmp_path,output_stem='scaling')
    svg = Path(paths['svg']).read_text()
    assert 'Direct record scaling' in svg and 'D100' in svg
    assert 'Direct sweep; no indirect records' in svg
    receipt['stage_labels'] = ['None','Direct']
    with pytest.raises(ValueError,match='stage labels'):
        plotter.plot_baseline_curve_points(rows,receipt,tmp_path,output_stem='invalid')


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

    called_output_path = (
        root / "bbb_martins/queries/query_idx00000/levels/level_1/output.json"
    )
    called_output = json.loads(called_output_path.read_text(encoding="utf-8"))
    called_output["llm"]["usage"].pop("prompt_tokens")
    called_output_path.write_text(json.dumps(called_output), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing prompt tokens"):
        plotter.collect_conditioned_progressive_resource_data(progressive_root=root)
    called_output["llm"]["usage"]["prompt_tokens"] = 200
    called_output_path.write_text(json.dumps(called_output), encoding="utf-8")

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


def test_ames_baseline_paths_keep_partial_coverage_explicit(tmp_path):
    for method, _, _ in plotter.CONDITIONED_BASELINES:
        relative = "minimol_head/final" if method == "minimol_head" else method
        path = tmp_path / "Ames" / relative / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n_total": 274,
                                    "n_evaluated": 252 if method.endswith("condition") else 274,
                                    "macro_f1": 0.6, "accuracy": 0.7}))
    kwargs = dict(task="ames", task_label="Ames", baseline_task="Ames",
                  baseline_root=tmp_path, expected_n=274)
    with pytest.raises(ValueError, match="sample-count mismatch"):
        plotter._append_conditioned_baselines([], **kwargs)
    rows = []
    omissions = plotter._append_conditioned_baselines(rows, **kwargs, omit_sample_count_mismatch=True)
    assert len(rows) == 3 and len(omissions) == 2
    assert {row["baseline_n"] for row in omissions} == {252}
    assert {row["agent_n"] for row in omissions} == {274}


def test_ames_fourth_column_keeps_unavailable_tasks_empty(tmp_path):
    roots = plotter._parse_configuration_task_path_overrides(["4/2:ames=/ames"])
    assert "ames" in roots["4/2"]
    svg = tmp_path / "four_tasks.svg"
    plotter.plot_conditioned_progressive_configuration_comparison(
        rows=[{"configuration": "4/2", "task": "ames", "level": 1, "n_queries": 274,
               "macro_f1": 0.72, "mean_active_molecules": 4.0,
               "mean_cards_per_active_molecule": 2.0, "mean_prompt_tokens_per_call": 1000.0,
               "mean_reasoning_tokens_per_call": 500.0}],
        output_svg=svg, output_png=tmp_path / "four_tasks.png",
        tasks=("bbb_martins", "bioavailability_ma", "skin_reaction", "ames"),
        configurations=("4/2", "8/4"),
    )
    text = svg.read_text()
    assert "Ames  (n=274)" in text and "Current results unavailable" in text
    assert text.index("BBB") < text.index("Bioavailability") < text.index("Skin") < text.index("Ames")


def test_progressive_baseline_collector_accepts_minimol_train_path(tmp_path):
    baseline_root = tmp_path / "baselines"
    complete = {
        "n_total": 2,
        "n_evaluated": 2,
        "macro_f1": 0.7,
        "accuracy": 0.75,
    }
    for method, _, relative_path in plotter.CONDITIONED_BASELINES:
        if method == "minimol_head":
            relative_path = Path("minimol_train/final/metrics.json")
        path = baseline_root / "BBB_Martins" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(complete), encoding="utf-8")

    rows: list[dict[str, object]] = []
    plotter._append_conditioned_baselines(
        rows,
        task="bbb_martins",
        task_label="BBB",
        baseline_task="BBB_Martins",
        baseline_root=baseline_root,
        expected_n=2,
    )

    head = next(row for row in rows if row["method"] == "minimol_head")
    assert head["metrics_path"].endswith("minimol_train/final/metrics.json")


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


def test_parse_configuration_task_path_overrides():
    assert plotter._parse_configuration_task_path_overrides(
        [
            "4/2:bbb_martins=/tmp/bbb_4_2",
            "8/4:bbb_martins=/tmp/bbb_8_4",
            "8/4:bbb_martins=/tmp/bbb_8_4_replay",
            "8/4:skin_reaction=/tmp/skin_8_4",
        ]
    ) == {
        "4/2": {"bbb_martins": [Path("/tmp/bbb_4_2")]},
        "8/4": {
            "bbb_martins": [
                Path("/tmp/bbb_8_4"),
                Path("/tmp/bbb_8_4_replay"),
            ],
            "skin_reaction": [Path("/tmp/skin_8_4")],
        },
    }
    with pytest.raises(ValueError, match="CONFIG:TASK=PATH"):
        plotter._parse_configuration_task_path_overrides(
            ["bbb_martins=/tmp/value"]
        )
    with pytest.raises(ValueError, match="Duplicate"):
        plotter._parse_configuration_task_path_overrides(
            [
                "4/2:bbb_martins=/tmp/one",
                "4/2:bbb_martins=/tmp/one",
            ]
        )


def test_selection_comparison_normalizes_legacy_text_and_ignores_only_card_limits():
    legacy = {
        "level_1": "top 10 molecules by Morgan similarity; at most 4 cards per molecule",
        "later_new_pool": "top 3 previously unseen molecules; at most 2 newly unlocked cards each",
        "append_only": True,
    }
    structured = {
        "level_1": {"molecule_limit": 10, "card_limit_per_molecule": 8},
        "later_new_pool": {
            "molecule_limit": 3,
            "delta_card_limit_per_molecule": 4,
        },
        "append_only": True,
    }
    assert plotter._selection_without_card_limits(legacy) == (
        plotter._selection_without_card_limits(structured)
    )


def test_progressive_replicate_rows_report_mean_and_observed_range():
    base = {
        "task": "bbb_martins",
        "task_label": "BBB",
        "level": 1,
        "family": "direct_brain_exposure",
        "cumulative_assays": 5,
        "n_queries": 2,
        "macro_f1": 0.6,
        "accuracy": 0.5,
        "metrics_path": "/tmp/run_one.json",
    }
    replay = {
        **base,
        "macro_f1": 0.8,
        "accuracy": 0.75,
        "metrics_path": "/tmp/run_two.json",
    }

    rows = plotter._aggregate_progressive_replicate_rows([[base], [replay]])

    assert rows[0]["macro_f1"] == pytest.approx(0.7)
    assert rows[0]["macro_f1_min"] == 0.6
    assert rows[0]["macro_f1_max"] == 0.8
    assert rows[0]["macro_f1_sd"] == pytest.approx(0.2 / 2**0.5)
    assert rows[0]["n_replicates"] == 2
    assert rows[0]["metrics_paths"] == [
        "/tmp/run_one.json",
        "/tmp/run_two.json",
    ]

    single = plotter._aggregate_progressive_replicate_rows([[base]])[0]
    assert single["macro_f1"] == 0.6
    assert single["n_replicates"] == 1
    assert "macro_f1_min" not in single
    assert "macro_f1_sd" not in single

    third = {**base, "macro_f1": 0.7}
    triple = plotter._aggregate_progressive_replicate_rows([[base], [replay], [third]])[0]
    assert triple["macro_f1"] == pytest.approx(0.7)
    assert triple["macro_f1_sd"] == pytest.approx(0.1)
    assert plotter._replicate_bounds(triple, "macro_f1", "sd") == pytest.approx((0.6, 0.8))
    assert plotter._replicate_bounds(rows[0], "macro_f1", "sd") == pytest.approx(
        (0.7 - 0.2 / 2**0.5, 0.7 + 0.2 / 2**0.5)
    )
    assert plotter._replicate_bounds(rows[0], "macro_f1", "range") == (0.6, 0.8)
    with pytest.raises(ValueError, match="at least two"):
        plotter._replicate_bounds(single, "macro_f1", "sd")


def test_progressive_configuration_comparison_validates_lineage(monkeypatch):
    changed_prepared = False
    def fake_load(path):
        return {
            "task": "bbb_martins",
            "split_scheme": "scaffold",
            "evaluation_subset": "valid",
            "artifacts": {
                "retrieval_index_sha256_before": "index-4/2",
                "retrieval_index_sha256_after": "index-8/4",
            },
            "results": {"selected_retrieval_surfaces_equal": True},
        }

    def fake_collect(*, progressive_roots_by_task=None, **_):
        root = str(next(iter(progressive_roots_by_task.values())))
        configuration = "4/2" if "4_2" in root else "8/4"
        selection = {
            "level_1": {
                "molecule_limit": 10,
                "card_limit_per_molecule": 4 if configuration == "4/2" else 8,
            },
            "later_new_pool": {
                "molecule_limit": 3,
                "delta_card_limit_per_molecule": (
                    2 if configuration == "4/2" else 4
                ),
            },
            "append_only": True,
        }
        contract = {
            "progressive_root": root,
            "experiment": "conditioned_assay_progressive_visible.v8",
            "evaluation_subset": "valid",
            "visibility_mode": "identity_blind",
            "reference_pool": "full_flat",
            "neighbor_identity_policy": "scaffold_disjoint",
            "selection": selection,
            "selection_without_card_limits": (
                plotter._selection_without_card_limits(selection)
            ),
            "candidate_generation": None,
            "min_similarity": 0.3,
            "prompt_profile": "prompt-v1",
            "condition_policy": "condition-v1",
            "max_tokens": 100,
            "temperature": 0,
            "thinking": "default",
            "reasoning_effort": "omitted",
            "tool_prefetch_complete": True,
            "agent_model": "model-alias",
            "model_identity": "model-identity",
            "input_sha256": "input",
            "prepared_inputs_sha256": "changed" if changed_prepared and "replay" in root else configuration,
            "evaluation_indices_sha256": "indices",
            "index_sha256": f"index-{configuration}",
            "family_manifest_sha256": "family",
            "execution_base_urls": ["https://provider.test/v1"],
            "n": 2,
            "split_scheme": "scaffold",
        }
        rows = [{
            "task": "bbb_martins",
            "task_label": "BBB",
            "level": 1,
            "family": "direct_brain_exposure",
            "cumulative_assays": 5,
            "n_queries": 2,
            "macro_f1": 0.7 if configuration == "4/2" else 0.68,
            "accuracy": 0.7,
        }]
        return rows, {
            "comparison_contract": {
                "split_scheme": "scaffold",
                "task_contracts": {"bbb_martins": contract},
            }
        }

    monkeypatch.setattr(plotter, "_load_json", fake_load)
    monkeypatch.setattr(
        plotter, "collect_conditioned_progressive_resource_data", fake_collect
    )
    monkeypatch.setattr(plotter, "sha256_file", lambda path: "receipt-hash")
    configuration_roots = {
        "4/2": {"bbb_martins": Path("/tmp/bbb_4_2")},
        "8/4": {
            "bbb_martins": [
                Path("/tmp/bbb_8_4"),
                Path("/tmp/bbb_8_4_replay"),
            ]
        },
    }
    with pytest.raises(ValueError, match="zero-change receipt"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=configuration_roots,
        )
    rows, summary = plotter.collect_conditioned_progressive_configuration_data(
        configuration_roots=configuration_roots,
        lineage_receipts_by_task={"bbb_martins": Path("/tmp/receipt.json")},
    )
    assert [row["configuration"] for row in rows] == ["4/2", "8/4"]
    assert rows[1]["n_replicates"] == 2
    contract = summary["comparison_contract"]
    assert contract["configurations"] == ["4/2", "8/4"]
    assert contract["task_audits"]["bbb_martins"]["index_sha256_equal"] is False
    assert contract["task_audits"]["bbb_martins"][
        "retrieval_lineage_receipt"
    ] == "/tmp/receipt.json"
    assert "prompt_profile" in contract["strict_invariants"]
    assert "evaluation_indices_sha256" in contract["strict_invariants"]
    assert contract["configurations_contract"]["8/4"]["task_contracts"][
        "bbb_martins"
    ]["n_replicates"] == 2

    changed_prepared = True
    with pytest.raises(ValueError, match="prepared_inputs_sha256"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=configuration_roots,
            lineage_receipts_by_task={"bbb_martins": Path("/tmp/receipt.json")},
        )

    changed_prepared = False
    drift = {}
    def cross_model_collect(*, progressive_roots_by_task=None, **kwargs):
        rows, summary = fake_collect(progressive_roots_by_task=progressive_roots_by_task, **kwargs)
        contract = summary["comparison_contract"]["task_contracts"]["bbb_martins"]
        contract["prepared_evidence_sha256"] = "same-evidence"
        contract["prompt_comparison_evidence_sha256"] = "same-tools-cards"
        root = str(next(iter(progressive_roots_by_task.values())))
        if "gpt" in root:
            contract.update(model_identity="gpt", temperature=None, reasoning_effort="medium")
            if "changed" in root:
                contract.update(drift)
        return rows, summary

    monkeypatch.setattr(plotter, "collect_conditioned_progressive_resource_data", cross_model_collect)
    roots = {"reference": {"bbb_martins": Path('/tmp/4_2')},
             "gpt": {"bbb_martins": Path('/tmp/gpt_4_2')}}
    with pytest.raises(ValueError, match="Incompatible progressive configurations"):
        plotter.collect_conditioned_progressive_configuration_data(configuration_roots=roots)
    _, summary = plotter.collect_conditioned_progressive_configuration_data(
        configuration_roots=roots, allow_model_comparison=True)
    audit = summary["comparison_contract"]["task_audits"]["bbb_martins"]
    assert audit["model_generation_differences"]["gpt"]["reasoning_effort"]["comparison"] == "medium"
    roots["gpt_changed"] = {"bbb_martins": Path('/tmp/gpt_changed_4_2')}
    for field, value in (("temperature", 0), ("reasoning_effort", "high")):
        drift.clear()
        drift[field] = value
        with pytest.raises(ValueError, match=f"Same-model comparison differs in {field}"):
            plotter.collect_conditioned_progressive_configuration_data(
                configuration_roots=roots, allow_model_comparison=True)
    roots.pop("gpt")
    drift.clear()
    drift["prepared_evidence_sha256"] = "changed"
    with pytest.raises(ValueError, match="Cross-model comparison differs in prepared_evidence_sha256"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=roots, allow_model_comparison=True)
    drift.clear()
    drift["reasoning_policy"] = {"prompt_adaptation": {"profile": "gpt_calibration.v1"}}
    with pytest.raises(ValueError, match="Prompt-policy ablations"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=roots, allow_model_comparison=True)
    _, summary = plotter.collect_conditioned_progressive_configuration_data(
        configuration_roots=roots, allow_model_comparison=True, allow_prompt_comparison=True)
    assert summary["comparison_contract"]["task_audits"]["bbb_martins"]["prompt_comparisons"]["gpt_changed"]["evidence_and_tools_equal"]
    drift["prompt_comparison_evidence_sha256"] = "changed-cards"
    with pytest.raises(ValueError, match="Prompt comparison differs in prompt_comparison_evidence_sha256"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=roots, allow_model_comparison=True, allow_prompt_comparison=True)
    drift["prompt_comparison_evidence_sha256"] = "same-tools-cards"
    drift["reasoning_policy"]["unrelated_policy"] = True
    with pytest.raises(ValueError, match="beyond prompt_adaptation"):
        plotter.collect_conditioned_progressive_configuration_data(
            configuration_roots=roots, allow_model_comparison=True, allow_prompt_comparison=True)


def test_prompt_comparison_preserves_evidence_and_unrelated_policy():
    prepared = {"query_prior": "generated", "cards": ["card-a"],
                "reasoning_policy": {"prompt_adaptation": {"profile": "new"}, "grounding": True}}
    expected = {"cards": ["card-a"], "reasoning_policy": {"grounding": True}}
    assert plotter._prompt_comparison_evidence(prepared) == expected
    assert "prompt_adaptation" in prepared["reasoning_policy"]


@pytest.mark.parametrize("mismatch", [None, "prepared_inputs_sha256", "selection", "index_sha256", "prompt_profile"])
@pytest.mark.parametrize("profile", ["progressive_compact_tools_short_aliases.v2", "progressive_evidence_revision.v5"])
def test_matched_full_flat_comparison_requires_same_evidence_and_frozen_profiles(mismatch, profile):
    progressive = {
        "experiment": "conditioned_assay_progressive_visible.v8",
        "prompt_profile": profile,
        "selection": {"level_1": {"card_limit_per_molecule": 4}},
        "prepared_inputs_sha256": "same-cards-tools-and-query-priors",
        "index_sha256": "same-index",
        "family_manifest_sha256": "same-families",
    }
    flat = {
        **progressive,
        "experiment": "conditioned_assay_matched_full_flat.v1",
        "prompt_profile": "independent_cumulative_tools_short_aliases.v1",
    }
    if mismatch:
        flat[mismatch] = "changed"
        with pytest.raises(ValueError):
            plotter._matched_organization_comparison(progressive, flat)
    else:
        assert plotter._matched_organization_comparison(progressive, flat)
        assert plotter._matched_organization_comparison(flat, progressive)
        assert not plotter._matched_organization_comparison(progressive, progressive)


def test_progressive_configuration_references_require_shared_none_and_baselines(
    tmp_path,
):
    configuration_roots = {}
    complete = {
        "n_total": 2,
        "n_successful": 2,
        "n_evaluable": 2,
        "n_failed_runs": 0,
        "macro_f1": 0.55,
        "accuracy": 0.5,
    }
    for configuration in ("4_2", "8_4"):
        root = tmp_path / configuration
        root.mkdir()
        (root / "experiment_manifest.json").write_text(
            json.dumps(
                {
                    "evaluation_indices_by_task": {"bbb_martins": [0, 1]},
                    "model": "same-model",
                }
            ),
            encoding="utf-8",
        )
        none_path = root / "bbb_martins/none/metrics.json"
        none_path.parent.mkdir(parents=True)
        none_path.write_text(json.dumps(complete), encoding="utf-8")
        configuration_roots[configuration] = {"bbb_martins": root}
    baseline_root = tmp_path / "baselines"
    for _, _, relative_path in plotter.CONDITIONED_BASELINES:
        path = baseline_root / "BBB_Martins" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**complete, "macro_f1": 0.6}), encoding="utf-8"
        )

    rows, summary = (
        plotter.collect_conditioned_progressive_configuration_reference_data(
            configuration_roots=configuration_roots,
            none_root=tmp_path / "fallback_none",
            baseline_root=baseline_root,
        )
    )

    assert len(rows) == 1 + len(plotter.CONDITIONED_BASELINES)
    assert rows[0]["result_type"] == "none"
    assert rows[0]["macro_f1"] == 0.55
    assert {row["macro_f1"] for row in rows[1:]} == {0.6}
    assert summary["none_audits"]["bbb_martins"]["status"] == (
        "matched_across_configurations"
    )
    none_only, omitted = plotter.collect_conditioned_progressive_configuration_reference_data(
        configuration_roots=configuration_roots, none_root=tmp_path / "fallback_none",
        baseline_root=tmp_path / "unavailable_baselines", omit_baseline_tasks=("bbb_martins",),
    )
    assert len(none_only) == 1 and none_only[0]["result_type"] == "none"
    assert omitted["baseline_omissions"] == [
        {"task": "bbb_martins", "reason": "explicit_task_baseline_omission"}
    ]

    mismatched_none = (
        configuration_roots["8_4"]["bbb_martins"]
        / "bbb_martins/none/metrics.json"
    )
    mismatched_none.write_text(
        json.dumps({**complete, "macro_f1": 0.54}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="No-retrieval metric mismatch"):
        plotter.collect_conditioned_progressive_configuration_reference_data(
            configuration_roots=configuration_roots,
            none_root=tmp_path / "fallback_none",
            baseline_root=baseline_root,
        )
    prompt_rows, prompt_audit = plotter.collect_conditioned_progressive_configuration_reference_data(
        configuration_roots=configuration_roots, none_root=tmp_path / "fallback_none",
        baseline_root=baseline_root, allow_prompt_comparison=True)
    assert sorted(row["macro_f1"] for row in prompt_rows if row["result_type"] == "none") == [0.54, 0.55]
    assert prompt_audit["none_shared_within_model"] is False

    for model, tasks in configuration_roots.items():
        path = tasks["bbb_martins"] / "experiment_manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps({**manifest, "model": model, "model_identity": model}))
    rows, audit = plotter.collect_conditioned_progressive_configuration_reference_data(
        configuration_roots=configuration_roots, none_root=tmp_path / "fallback_none",
        baseline_root=baseline_root, allow_model_comparison=True,
    )
    assert sorted(row["macro_f1"] for row in rows if row["result_type"] == "none") == [0.54, 0.55]
    assert sum(row["result_type"] == "baseline" for row in rows) == len(plotter.CONDITIONED_BASELINES)
    assert audit["none_shared_within_model"] is True
    mismatched_none.write_text(json.dumps(complete))
    for alias, tasks in zip(("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-v4-flash-0731"), configuration_roots.values()):
        path = tasks["bbb_martins"] / "experiment_manifest.json"
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps({**manifest, "model": alias, "model_identity": alias}))
    rows, _ = plotter.collect_conditioned_progressive_configuration_reference_data(
        configuration_roots=configuration_roots, none_root=tmp_path / "fallback_none",
        baseline_root=baseline_root, allow_model_comparison=True,
    )
    assert sum(row["result_type"] == "none" for row in rows) == 1


def test_mixed_repeats_never_estimate_sd_from_one_run():
    single = {"macro_f1": 0.7, "n_replicates": 1}
    assert plotter._replicate_bounds(single, "macro_f1", "sd_if_repeated") == (0.7, 0.7)
    with pytest.raises(ValueError, match="at least two"):
        plotter._replicate_bounds(single, "macro_f1", "sd")
    assert plotter._replicate_bounds({**single, "n_replicates": 3, "macro_f1_sd": 0.1},
                                    "macro_f1", "sd_if_repeated") == pytest.approx((0.6, 0.8))


def test_cross_model_organization_requires_identical_tools_and_evidence():
    a = {"experiment": "conditioned_assay_progressive_visible.v8",
         "prompt_profile": "progressive_compact_tools_short_aliases.v2",
         "selection": {"cards": 4}, "prepared_inputs_sha256": "prior-A",
         "prepared_evidence_sha256": "same-tools-cards", "index_sha256": "same-index",
         "family_manifest_sha256": "same-families"}
    b = {**a, "experiment": "conditioned_assay_matched_full_flat.v1",
         "prompt_profile": "independent_cumulative_tools_short_aliases.v1",
         "prepared_inputs_sha256": "prior-B"}
    assert plotter._matched_organization_comparison(a, b, cross_model=True)
    with pytest.raises(ValueError, match="prepared_inputs_sha256"):
        plotter._matched_organization_comparison(a, b)
    with pytest.raises(ValueError, match="prepared_evidence_sha256"):
        plotter._matched_organization_comparison(a, {**b, "prepared_evidence_sha256": "changed"}, cross_model=True)


@pytest.mark.parametrize("interval", ["range", "sd_if_repeated"])
def test_four_model_method_curves_keep_separate_none_and_mixed_run_counts(tmp_path, interval):
    rows, references = [], []
    for model, count in [("flash", 3), ("pro", 1)]:
        for organization in ["full_flat", "progressive"]:
            values = {"macro_f1": 0.7, "mean_active_molecules": 4.0,
                      "mean_cards_per_active_molecule": 2.0,
                      "mean_prompt_tokens_per_call": 1000.0, "mean_reasoning_tokens_per_call": 500.0}
            rows.append({"configuration": f"{model} {organization}", "model_identity": model,
                         "organization": organization, "task": "bbb_martins", "level": 1,
                         "n_queries": 2, "n_replicates": count, **values,
                         **({f"{key}_sd": value * 0.01 for key, value in values.items()} if count > 1 else {})})
        references.append({"task": "bbb_martins", "result_type": "none", "method": "none",
                           "model_identity": model, "macro_f1": 0.5 if count > 1 else 0.6})
    svg = tmp_path / "models.svg"
    plotter.plot_conditioned_progressive_configuration_comparison(
        rows=rows, output_svg=svg, output_png=tmp_path / "models.png",
        tasks=("bbb_martins",), configurations=tuple(row["configuration"] for row in rows),
        reference_rows=references, organization_comparison=True, performance_only=True,
        replicate_interval=interval,
    )
    text = svg.read_text()
    assert "1 run, no interval" in text and "3 runs, mean" in text
    assert "default 4/2 card budget" not in text
    if interval == "range":
        assert "observed min–max" in text and "sample SD" not in text
    else:
        assert "sample SD" in text
    assert "0.500" in text and "0.600" in text


@pytest.mark.parametrize("interval", ["range", "sd"])
def test_progressive_configuration_plot_supports_full_resource_panels(tmp_path, interval):
    rows = []
    for configuration, macro_f1, cards in (
        ("4/2", 0.72, 2.0),
        ("8/4", 0.70, 3.0),
    ):
        rows.append(
            {
                "configuration": configuration,
                "task": "bbb_martins",
                "level": 1,
                "n_queries": 2,
                "macro_f1": macro_f1,
                "macro_f1_min": macro_f1 - (0.02 if configuration == "8/4" else 0),
                "macro_f1_max": macro_f1 + (0.02 if configuration == "8/4" else 0),
                "mean_active_molecules": 4.0,
                "mean_cards_per_active_molecule": cards,
                "mean_prompt_tokens_per_call": 1000.0 * cards,
                "mean_reasoning_tokens_per_call": 500.0 * cards,
            }
        )
    if interval == "sd":
        for row in rows:
            row["n_replicates"] = 3
            for field in ("macro_f1", "mean_active_molecules", "mean_cards_per_active_molecule",
                          "mean_prompt_tokens_per_call", "mean_reasoning_tokens_per_call"):
                row[f"{field}_sd"] = row[field] * 0.05
    output_svg = tmp_path / "comparison.svg"
    output_png = tmp_path / "comparison.png"
    plotter.plot_conditioned_progressive_configuration_comparison(
        rows=rows,
        output_svg=output_svg,
        output_png=output_png,
        tasks=("bbb_martins",),
        configurations=("2/1", "4/2", "8/4"),
        replicate_interval=interval,
        reference_rows=[
            {
                "task": "bbb_martins",
                "result_type": "none",
                "method": "none",
                "macro_f1": 0.55,
            },
            {
                "task": "bbb_martins",
                "result_type": "baseline",
                "method": "minimol_head",
                "macro_f1": 0.6,
            },
        ],
    )
    assert output_svg.is_file()
    assert output_png.is_file()
    svg = output_svg.read_text(encoding="utf-8")
    assert "None" in svg
    assert "MiniMol head" in svg
    assert "Unavailable on current lineage: 2/1" in svg
    assert ("whiskers show observed min–max" if interval == "range" else "sample SD (ddof=1); 3 runs per point") in svg
    for row in rows:
        row["n_replicates"] = 1
    plotter.plot_conditioned_progressive_configuration_comparison(
        rows=rows, output_svg=output_svg, output_png=output_png,
        tasks=("bbb_martins",), configurations=("4/2", "8/4"),
        performance_only=True, replicate_interval="sd_if_repeated",
    )
    svg = output_svg.read_text(encoding="utf-8")
    assert "single run per setting" in svg
    assert "mean ± 1 sample SD" not in svg


def test_split_scoped_baselines_do_not_mix_equal_sized_valid_and_test(tmp_path):
    for subset, score in (("valid", 0.8), ("test", 0.6)):
        for method, _, _ in plotter.CONDITIONED_BASELINES:
            path = tmp_path / "DILI" / subset / method / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"n_evaluated": 103, "macro_f1": score, "accuracy": score}))
    for subset, expected in (("valid", 0.8), ("test", 0.6)):
        rows = []
        plotter._append_conditioned_baselines(
            rows, task="dili", task_label="DILI", baseline_task="DILI",
            baseline_root=tmp_path, expected_n=103, evaluation_subset=subset,
        )
        assert len(rows) == 5
        assert {row["macro_f1"] for row in rows} == {expected}
        assert all(f"/{subset}/" in row["metrics_path"] for row in rows)
    with pytest.raises(ValueError, match="explicit evaluation subset"):
        plotter._append_conditioned_baselines(
            [], task="dili", task_label="DILI", baseline_task="DILI",
            baseline_root=tmp_path, expected_n=103,
        )


@pytest.mark.parametrize('benchmark', ['conditioned', 'tdc'])
def test_reported_curve_source_resolves_relative_to_metadata(tmp_path, monkeypatch, benchmark):
    source = tmp_path / 'ours.json'
    curves = {
        'conditioned': [dict(label='Ours', source='user table', scores={'ames': [None, .7, .8]})],
        'tdc': [dict(label='Ours', source='user table', scores={'ames': [None, .6, .9]})],
    }
    source.write_text(json.dumps({'curves': curves}))
    folder = tmp_path / benchmark
    folder.mkdir()
    metadata = folder / 'metadata.json'
    config = {'reported_curves_source': {'path': '../ours.json', 'benchmark': benchmark}}
    metadata.write_text(json.dumps(config))
    points = folder / 'points.csv'
    points.write_text('task,method,family,n,macro_f1,accuracy\names,none,LLM,2,0.5,0.5\n')
    monkeypatch.setattr(plotter, 'plot_baseline_curve_points',
                        lambda rows, receipt, *args, **kwargs: receipt)
    receipt = plotter.plot_baseline_curves_csv(points, metadata, folder, output_stem='test')
    assert receipt['reported_curves'] == curves[benchmark]
    assert receipt['reported_curves_source']['sha256'] == plotter.sha256_file(source)
    assert receipt['reported_curves_source']['resolved_path'] == str(source)
    assert receipt['metadata_source']['sha256'] == plotter.sha256_file(metadata)
    # A stale inline copy must never silently override the shared result record.
    metadata.write_text(json.dumps({**config, 'reported_curves': curves[benchmark]}))
    with pytest.raises(ValueError, match='either inline'):
        plotter.plot_baseline_curves_csv(points, metadata, folder, output_stem='test')


def test_metadata_baselines_include_both_training_pools(tmp_path):
    rows = [dict(task='ames', method=m, family='fixture', n=2, macro_f1=.5+i*.03, accuracy=.7)
            for i,m in enumerate(['none','direct','both','rf','mixed_rf'])]
    receipt = dict(conditions={'ames':{'n':2}}, split_scheme='scaffold',evaluation_subset='test',
        visibility_mode='deployment_visible_prefetched',min_similarity=0,direct_budget=50,
        indirect_budget=50,tool_failures=0,cohort_label='Original TDC test',curve_settings={'Model':['none','direct','both']},
        baseline_settings=[dict(method=m,label=label,tick='RF',marker='v',face=face,edge='#333333')
                           for m,label,face in [('rf','RF train','#333333'),('mixed_rf','RF mixed','white')]])
    paths = plotter.plot_baseline_curve_points(rows,receipt,tmp_path,output_stem='all_ml')
    saved=json.loads(Path(paths['receipt']).read_text())
    assert {r['method'] for r in saved['scores']} == {r['method'] for r in rows}
    svg=Path(paths['svg']).read_text()
    assert 'RF train' in svg and 'RF mixed' in svg and 'Original TDC test' in svg
    with pytest.raises(ValueError,match='missing curve or ML'):
        plotter.plot_baseline_curve_points(rows[:-1],receipt,tmp_path,output_stem='missing')
    with pytest.raises(ValueError,match='omitted'):
        plotter.plot_baseline_curve_points(rows+[dict(rows[0],method='unlisted')],receipt,tmp_path,output_stem='extra')


def test_grouped_test_maxima_keep_training_pools_metrics_and_ties_separate(tmp_path, monkeypatch):
    rows=[]
    for task in ['ames','dili']:
        for method,f1,accuracy in [('none',.4,.5),('direct',.5,.6),('both',.6,.7),
                                  ('head',.6,.9),('finetuned',.7,.5),
                                  ('mixed_head',.8,.7),('mixed_finetuned',.8,.99)]:
            rows.append(dict(task=task,method=method,family='LLM' if method in ['none','direct','both'] else 'baseline',n=2,macro_f1=f1,accuracy=accuracy))
    receipt=dict(conditions={t:{'n':2} for t in ['ames','dili']},split_scheme='scaffold',evaluation_subset='test',
        visibility_mode='deployment_visible_prefetched',min_similarity=0,direct_budget=50,indirect_budget=50,
        tool_failures=0,curve_settings={'Model':['none','direct','both']},baseline_selection='max_test_macro_f1',
        baseline_settings=[dict(method=prefix+'mm_direct',members=[prefix+'head',prefix+'finetuned'],
            label=prefix+'MM Direct',tick='MM Direct',marker='D',face=face,edge='#333333')
            for prefix,face in [('', '#333333'),('mixed_', 'white')]])
    receipt['reported_curves']=[dict(label='Ours',source='User supplied final scores',
        scores={'ames':[None,.91,None],'dili':[None,.85,.92]})]
    reported_lines=[]
    original_plot=plotter.plt.Axes.plot
    def capture_reported(ax, *args, **kwargs):
        if kwargs.get('label')=='Ours':
            reported_lines.append(args[1])
        return original_plot(ax, *args, **kwargs)
    monkeypatch.setattr(plotter.plt.Axes,'plot',capture_reported)
    paths=plotter.plot_baseline_curve_points(rows,receipt,tmp_path,output_stem='grouped')
    saved=json.loads(Path(paths['receipt']).read_text())
    assert len(saved['scores'])==10 and len(saved['source_scores'])==14
    assert saved['point_counts']=={'audited_baselines':10,'reported':3,'total':13}
    assert saved['reported_curves']==receipt['reported_curves']
    assert 'Ours' not in saved['curve_settings']
    assert saved['macro_f1_limits'][1]>=.92
    assert str(reported_lines[0][0])=='nan' and str(reported_lines[0][2])=='nan'
    assert reported_lines[0][1]==.91 and reported_lines[1][1:]==[.85,.92]
    with pytest.raises(ValueError,match='invalid reported'):
        plotter.plot_baseline_curve_points(rows,{**receipt,'reported_curves':[
            dict(label='Ours',source='test',scores={'ames':[None,1.1,None],'dili':[None,.85,.92]})]},tmp_path,output_stem='bad_reported')
    selected={r['group']:r for r in saved['baseline_selections'] if r['task']=='ames'}
    assert selected['mm_direct']['selected_method']=='finetuned'
    assert selected['mixed_mm_direct']['selected_method']=='mixed_head'
    assert selected['mixed_mm_direct']['tied_methods']==['mixed_head','mixed_finetuned']
    winners={r['method']:r for r in saved['scores'] if r['task']=='ames'}
    assert winners['mm_direct']['accuracy']==.5  # Preserve the F1 winner, not a separate accuracy maximum.
    assert winners['mixed_mm_direct']['accuracy']==.7
    assert [r for r in saved['scores'] if r['family']=='LLM']==[r for r in rows if r['family']=='LLM']
    with pytest.raises(ValueError,match='explicit max_test'):
        plotter.plot_baseline_curve_points(rows,{**receipt,'baseline_selection':'validation'},tmp_path,output_stem='invalid')
    bad=[dict(r,n=3) if r['method']=='head' else r for r in rows]
    with pytest.raises(ValueError,match='same cohort'):
        plotter._baseline_group_maxima(bad,receipt['baseline_settings'],receipt['conditions'])
    with pytest.raises(ValueError,match='missing baseline group candidate'):
        plotter._baseline_group_maxima(rows[1:], [dict(method='group',members=['none','direct'])],receipt['conditions'])
    with pytest.raises(ValueError,match='disjoint'):
        plotter._baseline_group_maxima(rows,[dict(method='a',members=['head']),dict(method='b',members=['head'])],receipt['conditions'])
    with pytest.raises(ValueError,match='declared cohorts'):
        plotter._baseline_group_maxima(rows+[dict(rows[3],task='undeclared')],receipt['baseline_settings'],receipt['conditions'])


@pytest.mark.parametrize('layout,content', [('mixed', 'all'), ('horizontal', 'all'),
    ('mixed', 'tasks'), ('horizontal', 'tasks'), ('mixed', 'mean')])
@pytest.mark.parametrize('exclude_indirect', [False, True])
def test_bar_comparison_preserves_scores_missing_stages_and_local_scales(tmp_path, monkeypatch, layout, content, exclude_indirect):
    rows = [dict(task=task, method=method, family='LLM' if method != 'ml' else 'baseline',
                 n=2 if task == 'ames' else 200, macro_f1=score, accuracy=score)
            for task, offset in [('ames', 0), ('dili', .3)]
            for method, score in [('none', .2+offset), ('direct', .3+offset),
                                  ('both', .4+offset), ('ml', .25+offset)]]
    receipt = dict(conditions={'ames': {'n': 2}, 'dili': {'n': 200}},
        chart_style='bars', bar_layout=layout, bar_content=content, y_scale='task', curve_settings={'Model': ['none', 'direct', 'both']},
        reported_curve_backbones={'Ours': 'Model'},
        baseline_settings=[dict(method='ml', label='ML', tick='ML', face='#748396')],
        reported_curves=[dict(label='Ours', source='User supplied',
                             scores={'ames': [None, .35, .45], 'dili': [None, .65, .75]})])
    if exclude_indirect:
        receipt['excluded_bar_settings'] = {'Model': ['Ours · Direct + Indirect']}
    captured = []
    original_close = plotter.plt.close
    monkeypatch.setattr(plotter.plt, 'close', lambda fig: captured.append(fig))
    paths = plotter.plot_baseline_curve_points(rows, receipt, tmp_path, output_stem='bars')
    saved = json.loads(Path(paths['receipt']).read_text())
    assert saved['point_counts'] == {'audited_baselines': 8, 'reported': 4, 'total': 12}
    assert saved['task_macro_f1_limits']['ames'] != saved['task_macro_f1_limits']['dili']
    fig = captured[-1]
    assert len(fig.axes) == (1 if content == 'mean' else 2 if content == 'tasks' else 3)
    assert saved['rendered_counts'] == {'task_bars': 0 if content == 'mean' else 10 if exclude_indirect else 12,
                                       'mean_bars': 0 if content == 'tasks' else 5 if exclude_indirect else 6}
    for ax, task in zip(fig.axes, ('mean',) if content == 'mean' else ('ames', 'dili')):
        expected = [r['macro_f1'] for r in rows if r['task'] == task]
        reported = receipt['reported_curves'][0]['scores'].get(task, [])
        expected += [v for v in (reported[:2] if exclude_indirect else reported) if v is not None]
        if task == 'mean':
            expected = [.35, .45, .55, .4, .5] + ([] if exclude_indirect else [.6])
        endpoints = [p.get_y()+p.get_height() if layout == 'mixed' else p.get_x()+p.get_width() for p in ax.patches]
        assert sorted(endpoints) == pytest.approx(sorted(expected))
    audited_means = saved['excluded_bar_scores'] if exclude_indirect else saved['task_means']
    ours_mean = next(r for r in audited_means if r['setting'] == 'Ours · Direct + Indirect')
    assert ours_mean['mean'] == pytest.approx(.6)  # Equal tasks, not sample-size weighting.
    legend_labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    assert ('Ours · Direct + Indirect' in legend_labels) is not exclude_indirect
    if layout == 'mixed':
        labels = fig.axes[-1 if content == 'tasks' else 0].get_xticklabels()
        if content != 'tasks' or labels:
            assert [t.get_text() for t in labels] == ['N', 'D', 'D+I', 'D'] + ([] if exclude_indirect else ['D+I']) + ['ML']
            assert labels[3].get_color() == '#AA775D'
    assert len([r for r in saved['bar_scores'] if r['setting'] == 'None']) == 1
    svg = Path(paths['svg']).read_text()
    for removed in ('Independent task scales', 'Grouped by backbone', 'shared starting point', 'not reported'):
        assert removed not in svg
    original_close(fig)


def test_bar_backbone_binding_and_missing_means_are_explicit():
    tasks = ['ames', 'dili']
    receipt = dict(curve_settings={'DeepSeek': ['n', 'd', 'i'], 'Gemini': ['gn', 'gd', 'gi']},
                   reported_curves=[dict(label='Ours', scores={'ames': [None, .7, .8], 'dili': [None, None, .9]})],
                   reported_curve_backbones={'Ours': 'DeepSeek'})
    values = {(name, task): [.3, .4, .5] for name in receipt['curve_settings'] for task in tasks}
    values.update({('Ours','ames'): [float('nan'), .7, .8], ('Ours','dili'): [float('nan'), float('nan'), .9]})
    rows = plotter._baseline_bar_rows(tasks, values, {}, [], receipt)
    assert next(r for r in rows if r['group']=='DeepSeek' and r['stage']==3)['mean'] is None
    assert next(r for r in rows if r['group']=='DeepSeek' and r['stage']==4)['mean'] == pytest.approx(.85)
    assert all(r['mean'] is None for r in rows if r['group']=='Gemini' and r['stage']>=3)
    with pytest.raises(ValueError, match='bind every'):
        plotter._baseline_bar_rows(tasks, values, {}, [], {**receipt, 'reported_curve_backbones': {}})
    with pytest.raises(ValueError, match='existing group/setting'):
        plotter._baseline_bar_rows(tasks, values, {}, [], {**receipt,
            'excluded_bar_settings': {'unknown': ['Ours · Direct + Indirect']}})
    # ML positions come from baseline_settings; accepting this option for ML
    # would remove the bars while leaving their ticks and group heading behind.
    with pytest.raises(ValueError, match='configure ML groups with baseline_settings'):
        plotter._baseline_bar_rows(tasks, values, {(task, 'ml'): .4 for task in tasks},
            [dict(method='ml', tick='ML', face='#748396')], {**receipt,
                'excluded_bar_settings': {'ML baselines': ['ml']}})
    receipt['reported_curves'][0]['scores']['ames'][0] = .3
    with pytest.raises(ValueError, match='shared, not duplicated'):
        plotter._baseline_bar_rows(tasks, values, {}, [], receipt)
