"""Validate the fixed two-task Morgan-flat matrix contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from predict.harnesses.branches import matrix as flat_matrix
from predict.harnesses.branches import runner as flat_runner
from predict.harnesses.branches.runtime import synchronize_configured_single_reuse
from predict.api_client.pool import ProviderPoolConfig
from predict.utils.json import sha256_file


def test_matrix_splits_full_capacity_evenly() -> None:
    providers = flat_matrix.load_provider_pool_config(
        flat_matrix.DEFAULT_PROVIDER_CONFIG
    ).providers
    capacity = sum(provider.max_inflight for provider in providers)
    assert flat_matrix.endpoint_allocations(capacity) == [
        (provider.name, provider.max_inflight) for provider in providers
    ]


def test_upstream_v5_matrix_uses_reviewed_safety_levels(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "dili", tmp_path, limit=0, context_v5_six_tasks=True,
        six_task_prompt_version=flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        all_levels=True,
        assay_transfer_cache=Path(
            "ranked_level_retrieval_tdc_v1_v27_successors_v1.yaml"
        ),
    )
    assert args.max_level == 7
    assert "L7" in args.record_limits_by_level
    historical = flat_matrix._selection_args(
        "dili", tmp_path, limit=0, context_v5_six_tasks=True,
        six_task_prompt_version=flat_matrix.flat.CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
        all_levels=True,
    )
    assert historical.max_level == 2
    assert flat_matrix._selection_args(
        "dili", tmp_path, limit=0, context_v5_six_tasks=True,
        six_task_prompt_version=flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        all_levels=True,
    ).max_level == 2


def test_upstream_v2_routes_preselected_test_without_changing_budgets(tmp_path: Path) -> None:
    version = flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION
    assets = flat_matrix.flat.prompt_asset_manifest(version)
    assert assets["version"] == version
    args = flat_matrix._selection_args(
        "bbb_martins", tmp_path, limit=0, context_v5_six_tasks=True,
        six_task_prompt_version=version, all_levels=True,
        evaluation_subset="test", flat_mixed_selection=tmp_path / "mixed.json",
    )
    assert args.prompt_version == version
    assert args.harness_version == flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_HARNESS_VERSION
    assert args.evaluation_subset == "test"
    assert args.l1_molecules == 10


def test_recovery_subset_still_validates_full_direct_manifest(tmp_path: Path) -> None:
    records = tmp_path / "records.tsv"
    records.write_text(
        "benchmark_row_id\tselection_rank\tcontext_id\tparent_id\tgold_label\n"
        "q1\t1\tc1\tp1\tpass\nq1\t2\tc2\tp2\tfail\n"
        "q2\t1\tc3\tp3\tpass\nq2\t2\tc4\tp4\tfail\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "gold_direct_context_selection.v1", "status": "complete",
        "task_id": "bbb_martins", "budget": 2,
        "benchmark_row_ids": ["q1", "q2"],
        "records": {"path": records.name, "sha256": sha256_file(records), "row_count": 4},
    }), encoding="utf-8")
    load = flat_matrix.flat._load_preselected_contexts
    with pytest.raises(ValueError, match="queries differ"):
        load(manifest, task="bbb_martins", queries={"q1": "C"}, budget=2)
    selected, _ = load(
        manifest, task="bbb_martins", queries={"q1": "C"}, budget=2,
        allow_subset=True,
    )
    assert selected == {"q1": ["c1", "c2"]}
    with pytest.raises(ValueError, match="queries differ"):
        load(manifest, task="bbb_martins", queries={"q3": "C"}, budget=2,
             allow_subset=True)
    records.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load(manifest, task="bbb_martins", queries={"q1": "C"}, budget=2,
             allow_subset=True)


def test_gold_upstream_prior_current_validates_execution(tmp_path: Path, monkeypatch) -> None:
    execution = tmp_path / "batch" / "execution.json"
    execution.parent.mkdir()
    execution.write_text(json.dumps({
        "status": "complete",
        "subset": "test",
        "prompt_version": flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    }), encoding="utf-8")
    pointer = tmp_path / "CURRENT.json"
    pointer.write_text(json.dumps({
        "schema_version": "query_prior_current.v1",
        "gold_v1": {"test": {
            "batch_root": str(execution.parent),
            "execution_sha256": sha256_file(execution),
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(flat_matrix.flat, "QUERY_PRIOR_CURRENT", pointer)

    assert flat_matrix.flat.default_query_prior_root(
        flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        "gold", "test",
    ) == execution.parent
    assert flat_matrix.flat.default_query_prior_root("historical", "gold", "test") == (
        flat_matrix.flat.DEFAULT_QUERY_PRIOR_ROOT
    )
    execution.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        flat_matrix.flat.default_query_prior_root(
            flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
            "gold", "test",
        )


def test_matrix_keeps_two_routes_on_one_url_distinct() -> None:
    config = ProviderPoolConfig.from_mapping({
        "version": "openai_provider_pool.v1",
        "providers": [
            {"name": "together", "base_url": "https://openrouter.ai/api/v1",
             "model": "same", "api_key_env": "KEY", "max_inflight": 768},
            {"name": "cohere", "base_url": "https://openrouter.ai/api/v1",
             "model": "same", "api_key_env": "KEY", "max_inflight": 256},
        ],
    })
    assert flat_matrix.endpoint_allocations(1024, config=config) == [
        ("together", 768), ("cohere", 256),
    ]


def test_gpt6_matrix_omits_unsupported_temperature(tmp_path: Path) -> None:
    from predict.harnesses.branches.runner import _parse_args
    from predict.harnesses.branches.tasks.bbb_martins.contract import CONFIG

    args = flat_matrix._selection_args("bbb_martins", tmp_path, limit=0)
    command = flat_matrix._batch_command(
        args, tmp_path, model="openai/gpt-6-luna",
    ).command
    assert command[command.index("--temperature") + 1] == "none"
    assert _parse_args(CONFIG, ["--temperature", "none"]).temperature is None
    assert flat_matrix._model_temperature("deepseek/deepseek-v4-flash-0731") == 0.0


def test_provider_split_uses_selected_query_order_across_profiles(tmp_path: Path) -> None:
    indices = [0, 7, 8, 10, 20, 27, 35, 99]
    batches = {
        profile: SimpleNamespace(
            batch_dir=tmp_path / profile,
            items=[SimpleNamespace(index=index) for index in indices],
        )
        for profile in ("a", "b")
    }
    routes = flat_matrix.assign_provider_routes(batches)
    for batch in batches.values():
        selected = [routes[(str(batch.batch_dir), index)] for index in indices]
        assert selected == ["together", "together", "together", "cohere"] * 2


def test_matrix_builds_progressive_matched_record10_command(tmp_path: Path) -> None:
    args = flat_matrix._selection_args("bbb_martins", tmp_path, limit=0)
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(args, source).command

    assert args.records_per_level == 10
    assert args.l1_molecules == 10
    assert args.l1_records_per_molecule == 10
    assert args.max_level == 0
    assert args.reranking == "morgan"
    assert args.record_pool == "all"
    assert args.allow_frozen_l1_vote_scores is True
    assert flat_matrix._selection_args(
        "bioavailability_ma", tmp_path, limit=0
    ).allow_frozen_l1_vote_scores is False
    assert command[2] == "predict.harnesses.branches.tasks.bbb_martins.contract"
    assert command[command.index("--experiment-mode") + 1] == "full_flat"
    for pair in (
        ("--flat-prompt-version", "joseph_flat_v2"),
        ("--flat-reranking", "morgan"),
        ("--max-tokens", "262144"),
        ("--reasoning-effort", "high"),
        ("--limit", "0"),
    ):
        index = command.index(pair[0])
        assert command[index + 1] == pair[1]
    assert "--disable-flat-tools" in command
    assert "--enable-thinking" in command
    assert "--skip-existing" in command


def test_matrix_wires_test_split_and_explicit_prior_root(tmp_path: Path) -> None:
    prior_root = tmp_path / "test_priors"
    args = flat_matrix._selection_args(
        "bbb_martins",
        tmp_path,
        limit=0,
        context_width=25,
        context_v5=True,
        all_levels=True,
        evaluation_subset="test",
        prior_root=prior_root,
    )
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(args, source).command

    assert args.evaluation_subset == "test"
    assert args.input_jsonl.name == "test_molecule_condition_labels.jsonl"
    assert command[command.index("--input-jsonl") + 1] == str(args.input_jsonl)
    assert command[command.index("--single-analysis-source-batch") + 1] == str(
        prior_root.resolve() / "bbb_martins" / "bbb_martins__none"
    )


def test_matrix_wires_explicit_cache_bundle(tmp_path: Path) -> None:
    cache = tmp_path / "v9.yaml"
    args = flat_matrix._selection_args(
        "bbb_martins", tmp_path, limit=0, context_width=25, context_v5=True,
        min_contrast=1, assay_transfer_cache=cache,
    )

    assert args.assay_transfer_cache == cache.resolve()
    assert args.max_level == 1
    assert args.l1_min_contrast == 1


def test_matrix_builds_assay_transfer_all_pool_command(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "bioavailability_ma",
        tmp_path,
        limit=0,
        reranking="assay-transfer",
        record_pool="all",
        records_per_level=10,
    )
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(
        args,
        source,
        model="deepseek/deepseek-v4-flash-0731",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPEN_ROUTER_KEY",
        timeout_s=3600,
        reasoning_effort="",
        enable_thinking=False,
    ).command

    assert args.batch_id == (
        "bioavailability_ma__assay-transfer_all_records10"
    )
    assert args.reranking == "assay-transfer"
    assert args.record_pool == "all"
    for option, value in (
        ("--flat-reranking", "assay-transfer"),
        ("--model", "deepseek/deepseek-v4-flash-0731"),
        ("--base-url", "https://openrouter.ai/api/v1"),
        ("--timeout-s", "3600"),
    ):
        assert command[command.index(option) + 1] == value
    assert "--disable-thinking" in command


def test_matrix_builds_l1_uid_context_command(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "bbb_martins", tmp_path, limit=0,
        reranking="assay-transfer-contrastive", context_width=15,
        min_contrast=2,
    )
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(args, source, max_tokens=65_536).command

    assert args.batch_id == "bbb_martins__k10_w15_m2"
    assert args.max_level == 1
    assert command[command.index("--flat-prompt-version") + 1] == (
        "joseph_flat_context_v4_v1"
    )
    assert command[command.index("--flat-reranking") + 1] == (
        "assay-transfer-contrastive"
    )
    assert command[command.index("--flat-layout") + 1] == "global"
    assert command[command.index("--flat-query-prior") + 1] == "cached"
    assert command[command.index("--max-tokens") + 1] == "65536"


def test_preselected_command_forwards_exact_indices(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "bbb_martins", tmp_path, limit=0,
        context_width=25, context_v5=True, all_levels=True,
    )
    args.indices = ["3", "8"]
    command = flat_matrix._batch_command(
        args, tmp_path / "source", max_tokens=65_536,
    ).command

    index = command.index("--indices")
    assert command[index + 1:index + 3] == ["3", "8"]
    assert "--start" not in command
    assert "--limit" not in command


def test_replay_subset_uses_hash_pinned_source_retrievals(tmp_path: Path) -> None:
    source = tmp_path / "cache_matched_retrieval"
    retrieval = source / "runs" / f"{source.name}_idx00001" / "retrieval.json"
    retrieval.parent.mkdir(parents=True)
    retrieval.write_text('{"query": 1}', encoding="utf-8")
    input_jsonl = tmp_path / "queries.jsonl"
    input_jsonl.write_text('{"drug": "C"}\n', encoding="utf-8")
    prompt_version = flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
    manifest = {
        "harness_version": flat_runner.JOSEPH_PROMPT_HARNESSES[prompt_version],
        "prompt_version": prompt_version,
        "task": "bbb_martins",
        "reranking": "morgan",
        "indices": [0, 1],
        "input_jsonl": str(input_jsonl.resolve()),
        "input_sha256": sha256_file(input_jsonl),
        "layout": "global",
        "query_prior": "cached",
        "record_pool": "all",
        "cache_pool": "all",
        "selection_contract_sha256": "pinned",
        "retrieval_sha256_by_index": {"1": sha256_file(retrieval)},
        "evidence_projection": flat_runner.EVIDENCE_PROJECTION,
        "extra_details_policy": flat_runner.EXTRA_DETAILS_POLICY,
        "molecule_name_visible": False,
        "group_tools_enabled": False,
    }
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        flat_prompt_version=prompt_version,
        flat_selection_manifest=str(path),
        retrieval_replay_source_batch=str(source),
        flat_reranking="morgan",
        input_jsonl=str(input_jsonl),
        flat_layout="global",
        flat_query_prior="cached",
        flat_replay_subset=True,
    )
    config = SimpleNamespace(pipeline_module="predict.harnesses.branches.tasks.bbb_martins.contract")
    assert flat_runner._validated_flat_selection_manifest(config, args, [1]) == manifest
    with pytest.raises(SystemExit, match="not a subset"):
        flat_runner._validated_flat_selection_manifest(config, args, [2])
    retrieval.write_text("changed", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed or is missing"):
        flat_runner._validated_flat_selection_manifest(config, args, [1])
    args.flat_replay_subset = False
    with pytest.raises(SystemExit, match="manifest mismatch"):
        flat_runner._validated_flat_selection_manifest(config, args, [1])


def test_chained_recovery_accepts_only_previous_missing_cohort(tmp_path: Path, monkeypatch) -> None:
    input_jsonl = tmp_path / "queries.jsonl"
    input_jsonl.write_text('{"drug":"C"}\n{"drug":"N"}\n{"drug":"O"}\n', encoding="utf-8")
    mixed = tmp_path / "mixed.json"
    mixed.write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache.yaml"
    cache.write_text("cache", encoding="utf-8")
    source = tmp_path / "previous" / "cache_matched_retrieval"
    source.mkdir(parents=True)
    args = SimpleNamespace(
        task="bbb_martins", evaluation_subset="valid", input_jsonl=input_jsonl,
        prompt_version="full_flat_context_v5_six_tasks_upstream_v1",
        flat_mixed_selection=mixed, assay_transfer_cache=cache,
    )
    manifest = {
        "task": args.task, "evaluation_subset": args.evaluation_subset,
        "input_jsonl": str(input_jsonl),
        "input_sha256": sha256_file(input_jsonl),
        "prompt_version": args.prompt_version,
        "prompt_assets": {"sha256": "frozen-prompt"},
        "flat_mixed_selection": str(mixed),
        "flat_mixed_selection_sha256": sha256_file(mixed),
        "assay_transfer_cache": str(cache), "indices": [1, 2],
    }
    path = source / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    previous = {"path": str(path), "sha256": sha256_file(path),
                "assay_transfer_cache_sha256": sha256_file(cache)}
    monkeypatch.setattr(flat_matrix.flat, "_preselected_query_indices", lambda *a, **kw: [0, 1, 2])
    monkeypatch.setattr(flat_matrix.flat, "prompt_asset_manifest", lambda *a: {"sha256": "frozen-prompt"})
    monkeypatch.setattr(flat_matrix, "collect_completed_item", lambda *a: None)
    assert flat_matrix._missing_from_previous_batch(previous, args, [2]) == ([2], 0)
    monkeypatch.setattr(flat_matrix, "collect_completed_item", lambda *a: {"run_dir": str(tmp_path)})
    assert flat_matrix._missing_from_previous_batch(previous, args, [2]) == ([], 1)
    with pytest.raises(ValueError, match="different query cohort"):
        flat_matrix._missing_from_previous_batch(previous, args, [0])
    monkeypatch.setattr(flat_matrix.flat, "prompt_asset_manifest", lambda *a: {"sha256": "changed"})
    assert flat_matrix._missing_from_previous_batch(previous, args, [2]) == ([2], 0)
    batch_dir = source.parent
    (batch_dir / "manifest.json").write_text(json.dumps({
        "task_prompt_profile": "test", "flat_layout": "global", "flat_reranking": "morgan",
    }), encoding="utf-8")
    retrieval = source / "runs" / "cache_matched_retrieval_idx00002" / "retrieval.json"
    retrieval.parent.mkdir(parents=True)
    retrieval.write_text("{}", encoding="utf-8")
    run = batch_dir / "runs" / "previous_idx00002"
    run.mkdir(parents=True)
    (run / "single_molecule_reasoning_output.json").write_text("{}", encoding="utf-8")
    request = {"schema_version": "joseph_flat_context_request.v1",
               "messages": [{"role": "user", "content": "same"}],
               "message_char_count": 4}
    (run / "request.json").write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(flat_matrix, "attach_external_condition", lambda retrieval, record: retrieval)
    monkeypatch.setattr(flat_matrix, "validated_branch_content", lambda prior: prior)
    monkeypatch.setattr(flat_matrix.flat, "build_flat_context_request", lambda *a, **kw: (
        request["messages"], {},
    ))
    assert flat_matrix._missing_from_previous_batch(previous, args, [2]) == ([], 1)
    request["messages"][0]["content"] = "different"
    (run / "request.json").write_text(json.dumps(request), encoding="utf-8")
    assert flat_matrix._missing_from_previous_batch(previous, args, [2]) == ([2], 0)


def test_mixed_test_grids_allow_task_specific_winners(tmp_path: Path) -> None:
    grids = []
    for task, profile in (("bbb_martins", "ga075_mc010_sr010_sd010_ld010"),
                          ("bioavailability_ma", "ga100_mc025_sr050_sd025_ld025")):
        root = tmp_path / task
        indirect = root / "indirect" / profile / task / "manifest.json"
        indirect.parent.mkdir(parents=True)
        indirect.write_text('{}\n', encoding="utf-8")
        mixed = root / "mixed" / profile / task / "manifest.json"
        mixed.parent.mkdir(parents=True)
        mixed.write_text(json.dumps({
            "schema_version": "gold_mixed_selection.v1", "status": "complete",
            "task_id": task, "benchmark": "gold_v1", "subset": "test",
            "indirect": {"sha256": sha256_file(indirect)},
        }), encoding="utf-8")
        grid = root / "indirect_grid_manifest.json"
        grid.write_text(json.dumps({
            "schema_version": "gold_submodular_selection_grid.v1",
            "status": "complete", "kind": "indirect", "benchmark": "gold_v1",
            "subset": "test", "tasks": [task], "profile_count": 1,
            "profiles": [{"name": profile, "task_manifests": {
                task: {"path": str(indirect.relative_to(root)), "sha256": sha256_file(indirect)}
            }}],
        }), encoding="utf-8")
        grids.append(grid)

    loaded = flat_matrix.load_mixed_selection_grids(grids, benchmark="gold", subset="test")
    assert len(loaded) == 2
    assert {task for row in loaded for task in row["task_manifests"]} == {
        "bbb_martins", "bioavailability_ma",
    }
    with pytest.raises(ValueError, match="Incompatible mixed selection grid"):
        flat_matrix.load_mixed_selection_grids(grids, benchmark="gold", subset="valid")


@pytest.mark.parametrize("budget", (25, 100))
def test_mixed_selection_uses_manifest_indirect_budget(tmp_path: Path, monkeypatch, budget: int) -> None:
    direct = tmp_path / "direct.json"
    direct.write_text(json.dumps({"benchmark": "gold_v1", "task_id": "bbb_martins",
                                  "subset": "test", "benchmark_row_ids": ["q"], "budget": 10}))
    cache = tmp_path / "cache.yaml"
    index = tmp_path / "RELEASE_INDEX.json"
    cache.write_text("cache")
    index.write_text("index")
    records = tmp_path / "uids.tsv"
    records.write_text("benchmark_row_id\tselection_rank\tlevel\tsource_row_uid\n" + "".join(
        f"q\t{i}\tL2\tu{i}\n" for i in range(1, budget + 1)
    ))
    indirect = tmp_path / "indirect.json"
    indirect.write_text(json.dumps({
        "schema_version": "gold_joint_indirect_uid_selection.v1",
        "benchmark": "gold_v1", "task_id": "bbb_martins", "subset": "test",
        "benchmark_row_ids": ["q"], "budget": budget,
        "inputs": {"cache_bundle_sha256": sha256_file(cache),
                   "cache": {"release_index_sha256": sha256_file(index)}},
        "records": {"path": records.name, "sha256": sha256_file(records), "row_count": budget},
    }))
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({
        "schema_version": "gold_mixed_selection.v1", "status": "complete",
        "benchmark": "gold_v1", "task_id": "bbb_martins", "subset": "test",
        "benchmark_row_ids": ["q"], "direct_budget": 10, "indirect_budget": budget,
        "direct": {"path": str(direct), "sha256": sha256_file(direct)},
        "indirect": {"path": str(indirect), "sha256": sha256_file(indirect)},
    }))
    monkeypatch.setattr(flat_matrix.flat, "_load_preselected_contexts",
                        lambda *args, **kwargs: ({"q": [f"c{i}" for i in range(10)]}, {}))
    contexts, uids, receipt = flat_matrix.flat._load_mixed_selection(
        mixed, task="bbb_martins", benchmark="gold", subset="test",
        queries={"q": "CCO"}, levels={"L2"}, cache_index=index, cache_config=cache,
    )
    assert len(contexts["q"]) == 10
    assert len(uids["q"]["L2"]) == receipt["indirect_budget"] == budget


def test_preselected_grid_uses_oral_high_low_v6_bundle(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "bioavailability_ma", tmp_path, limit=0,
        context_width=25, context_v6=True, all_levels=True,
    )

    assert args.harness_version == flat_matrix.flat.CONTEXT_V6_HARNESS_VERSION
    assert args.prompt_version == flat_matrix.flat.CONTEXT_V6_PROMPT_VERSION
    assert args.max_level == 6
    assert args.morgan_primary_parent_width == 25
    assert args.l1_min_contrast == 0


def test_preselected_grid_can_use_v5_bundle(tmp_path: Path) -> None:
    args = flat_matrix._selection_args(
        "bioavailability_ma", tmp_path, limit=0,
        context_width=25, context_v5=True, all_levels=True,
    )

    assert args.harness_version == flat_matrix.flat.CONTEXT_V5_HARNESS_VERSION
    assert args.prompt_version == flat_matrix.flat.CONTEXT_V5_PROMPT_VERSION
    assert args.layout == "level-grouped"
    assert args.max_level == 6
    assert args.morgan_primary_parent_width == 25
    assert args.l1_min_contrast == 0


def test_direct_grid_uses_six_task_successor_and_context_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    prior_root = tmp_path / "priors"
    args = flat_matrix._selection_args(
        "skin_reaction", tmp_path, limit=0,
        reranking="assay-transfer-contrastive",
        context_width=25,
        flat_preselected_contexts=manifest,
        context_v5_six_tasks=True,
        six_task_prompt_version=(
            flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
        ),
        prior_root=prior_root,
    )
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(
        args, source, max_tokens=65_536,
    ).command

    assert args.harness_version == (
        flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_HARNESS_VERSION
    )
    assert args.prompt_version == (
        flat_matrix.flat.CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
    )
    assert args.flat_preselected_contexts == manifest
    assert args.max_level == 1
    assert args.layout == "level-grouped"
    assert command[command.index("--max-tokens") + 1] == "65536"
    assert command[command.index("--skin-prompt-profile") + 1] == (
        "skin_sensitization_contact_allergy.v2"
    )
    assert command[command.index("--single-analysis-source-batch") + 1] == str(
        prior_root.resolve() / "skin_reaction" / "skin_reaction__none"
    )


def test_preselected_grid_accepts_explicit_profile_subset(tmp_path: Path) -> None:
    task_manifests = {}
    for task in flat_matrix.TASKS:
        path = tmp_path / f"{task}.json"
        path.write_text("{}\n", encoding="utf-8")
        task_manifests[task] = {"path": path.name, "sha256": sha256_file(path)}
    screen = tmp_path / "screen.json"
    screen.write_text(json.dumps({
        "schema_version": flat_matrix.GRID_SCREEN_SCHEMA,
        "status": "complete",
        "selected_profiles": [{
            "name": "top_profile",
            "task_manifests": task_manifests,
        }],
    }), encoding="utf-8")

    profiles = flat_matrix.load_preselected_grid(screen)

    assert [profile["name"] for profile in profiles] == ["top_profile"]


def test_direct_grid_preserves_its_evaluation_subset(tmp_path: Path) -> None:
    task_manifests = {}
    tasks = flat_matrix.TASKS
    for task in tasks:
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps({"subset": "test"}) + "\n", encoding="utf-8")
        task_manifests[task] = {"path": path.name, "sha256": sha256_file(path)}
    grid = tmp_path / "direct.json"
    grid.write_text(json.dumps({
        "schema_version": "direct_context_selection_grid.v2",
        "status": "complete", "kind": "direct", "profile_count": 1,
        "tasks": tasks,
        "profiles": [{"name": "winner", "task_manifests": task_manifests}],
    }), encoding="utf-8")

    profiles = flat_matrix.load_preselected_direct_grid(grid)

    assert profiles[0]["subset"] == "test"
    assert profiles[0]["tasks"] == tasks


def test_direct_grid_accepts_task_specific_profiles(tmp_path: Path) -> None:
    tasks = ("bbb_martins", "ames")
    profiles = []
    for task in tasks:
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps({"subset": "valid"}) + "\n", encoding="utf-8")
        profiles.append({
            "name": f"winner_{task}",
            "task_manifests": {task: {"path": path.name, "sha256": sha256_file(path)}},
        })
    grid = tmp_path / "direct.json"
    grid.write_text(json.dumps({
        "schema_version": "direct_context_selection_grid.v2",
        "status": "complete", "kind": "direct", "profile_count": 2,
        "tasks": tasks, "profiles": profiles,
    }), encoding="utf-8")

    loaded = flat_matrix.load_preselected_direct_grid(grid)

    assert [profile["tasks"] for profile in loaded] == [(task,) for task in tasks]


def test_top_up_uses_largest_observed_load_and_preserves_no_failover() -> None:
    config = ProviderPoolConfig.from_mapping({
        "version": "openai_provider_pool.v1",
        "providers": [
            {"name": "a", "base_url": "http://a/v1", "model": "m", "max_inflight": 512},
            {"name": "b", "base_url": "http://b/v1", "model": "m", "max_inflight": 512},
        ],
        "max_failovers": 0,
    })
    samples = [
        {"provider": "a", "total": 100},
        {"provider": "a", "total": 120},
        {"provider": "b", "total": 512},
    ]

    active, allocations = flat_matrix.top_up_provider_config(
        config, samples, target_total=512,
    )

    assert [(row["provider"], row["launcher_slots"]) for row in allocations] == [
        ("a", 392), ("b", 0),
    ]
    assert [(provider.name, provider.max_inflight) for provider in active.providers] == [
        ("a", 392),
    ]
    assert active.max_failovers == 0


def test_synchronized_query_prior_is_not_rewritten(tmp_path: Path) -> None:
    batch = tmp_path / "source"
    source = batch / "runs" / "source_idx00000"
    source.mkdir(parents=True)
    (source / "single_molecule_reasoning_output.json").write_text("{}\n")
    run = tmp_path / "target"
    run.mkdir()
    target = run / "single_molecule_reasoning_output.json"
    target.write_text('{"already": "frozen"}\n')
    (run / "manifest.json").write_text(json.dumps({
        "single_analysis_source_run_dir": str(source),
    }))

    synchronize_configured_single_reuse(
        SimpleNamespace(args=SimpleNamespace(single_analysis_source_batch=str(batch))),
        SimpleNamespace(index=0),
        run,
    )

    assert json.loads(target.read_text()) == {"already": "frozen"}
