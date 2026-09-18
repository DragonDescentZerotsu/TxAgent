from dataclasses import replace
import json

from semantic_buckets import (
    semantic_l2_retrieval_eligibility as eligibility,
)
from predict.api_client.pool import ProviderSelection, load_provider_pool_config


def test_provider_pool_caps_every_selected_endpoint_at_32(tmp_path, monkeypatch):
    source = load_provider_pool_config(
        "predict/api_client/providers/"
        "progressive_deepseek_v4_flash_four_endpoint_high.json"
    )
    source_path = tmp_path / "providers.json"
    source_path.write_text(json.dumps({
        "version": "openai_provider_pool.v1",
        "providers": [provider.public_dict() for provider in source.providers],
    }))

    def select(config, requested):
        assert requested == 32 * len(config.providers)
        assert all(provider.max_inflight == 32 for provider in config.providers)
        active = replace(config, providers=config.providers[:3])
        return ProviderSelection(active, requested, 96, ())

    monkeypatch.setattr(eligibility, "select_healthy_providers", select)
    endpoints, receipt = eligibility._select_provider_endpoints(
        source_path, 32, source.providers[0].model
    )
    assert len(endpoints) == 3
    assert all(endpoint["max_inflight"] == 32 for endpoint in endpoints)
    assert receipt["effective_parallelism"] == 96
