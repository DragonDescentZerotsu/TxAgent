"""Validate the fixed two-task Morgan-flat matrix contract."""

import json
from pathlib import Path

from predict.harnesses.branches import matrix as flat_matrix
from predict.api_client.pool import ProviderPoolConfig
from predict.utils.json import sha256_file


def test_matrix_splits_full_capacity_evenly() -> None:
    providers = flat_matrix.load_provider_pool_config(
        flat_matrix.DEFAULT_PROVIDER_CONFIG
    ).providers
    capacity = sum(provider.max_inflight for provider in providers)
    assert flat_matrix.endpoint_allocations(capacity) == [
        (provider.base_url, provider.max_inflight) for provider in providers
    ]


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
        prior_root=prior_root,
    )
    source = args.batch_root / args.batch_id / "cache_matched_retrieval"
    command = flat_matrix._batch_command(
        args, source, max_tokens=65_536,
    ).command

    assert args.harness_version == flat_matrix.flat.CONTEXT_V5_SIX_TASKS_HARNESS_VERSION
    assert args.prompt_version == flat_matrix.flat.CONTEXT_V5_SIX_TASKS_PROMPT_VERSION
    assert args.flat_preselected_contexts == manifest
    assert args.max_level == 1
    assert args.layout == "level-grouped"
    assert command[command.index("--max-tokens") + 1] == "65536"
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
    for task in flat_matrix.DIRECT_GRID_TASKS:
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps({"subset": "test"}) + "\n", encoding="utf-8")
        task_manifests[task] = {"path": path.name, "sha256": sha256_file(path)}
    grid = tmp_path / "direct.json"
    grid.write_text(json.dumps({
        "schema_version": flat_matrix.DIRECT_GRID_SCHEMA,
        "status": "complete", "kind": "direct", "profile_count": 1,
        "profiles": [{"name": "winner", "task_manifests": task_manifests}],
    }), encoding="utf-8")

    profiles = flat_matrix.load_preselected_direct_grid(grid)

    assert profiles[0]["subset"] == "test"


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
