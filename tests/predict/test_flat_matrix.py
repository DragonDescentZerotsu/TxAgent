"""Validate the fixed two-task Morgan-flat matrix contract."""

from pathlib import Path

from predict.harnesses.branches import matrix as flat_matrix


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
