from pathlib import Path

from predict.api_client.pool import load_provider_pool_config
from predict.harnesses.progressive import query_priors


def test_test_prior_commands_share_the_frozen_visible_contract(tmp_path):
    commands = query_priors.build_commands(
        tmp_path / "priors", Path("providers.json"), limit=2
    )

    assert [command.experiment_name for command in commands] == [
        "bbb_martins__none",
        "bioavailability_ma__none",
    ]
    for command in commands:
        assert "--experiment-mode" in command.command
        assert command.command[command.command.index("--experiment-mode") + 1] == "none"
        assert "--harness-prefetch-tools" in command.command
        assert command.command[command.command.index("--reasoning-effort") + 1] == "high"
        assert command.command[command.command.index("--limit") + 1] == "2"

    hashes = query_priors.validate_visible_payload_contract()
    assert set(hashes) == set(query_priors.TASKS)
    assert all(len(value) == 64 for value in hashes.values())


def test_test_prior_provider_pool_is_the_pinned_four_endpoint_pool():
    config = load_provider_pool_config(query_priors.DEFAULT_PROVIDER_CONFIG)

    query_priors._validate_provider_config(config, parallelism=512)
    assert tuple(provider.base_url for provider in config.providers) == (
        query_priors.EXPECTED_PROVIDER_URLS
    )
