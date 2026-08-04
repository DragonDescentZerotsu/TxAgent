from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.chembl_tool.tasks.skin_reaction.data_processing import (
    build_embedding_bucket_mapping as builder,
)
from tools.chembl_tool.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers import (
    reconciliation,
)


def _extraction() -> builder.ExtractionSpec:
    return builder.ExtractionSpec(
        source_id="direct_skin_reaction",
        input_column="assay_or_test",
        output_name="global_context",
        output_column="canonical_assay_context",
        prompt="Normalize the values.",
        label_source="llm",
        reviewed_mapping_path=None,
        null_sentinel=None,
        bucket_pattern=None,
        clustering="lloyd",
    )


def _severity_extraction() -> builder.ExtractionSpec:
    return builder.ExtractionSpec(
        source_id="direct_skin_reaction",
        input_column="effect_metric",
        output_name="global_severity_grade",
        output_column="canonical_severity_grade",
        prompt="Normalize explicit severity grades.",
        label_source="llm",
        reviewed_mapping_path=None,
        null_sentinel="no explicit grade",
        bucket_pattern=r"[0-4]",
        clustering="lloyd",
    )


class _FakeCompletions:
    def __init__(self, *, total_tokens: int | None = 10):
        self.total_tokens = total_tokens
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        assert "json" in json.dumps(request["messages"]).casefold()
        item_ids = [
            item["id"]
            for item in json.loads(request["messages"][1]["content"])["items"]
        ]
        usage = (
            SimpleNamespace(
                total_tokens=self.total_tokens,
                prompt_tokens=max((self.total_tokens or 0) - 2, 0),
                completion_tokens=min(self.total_tokens or 0, 2),
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            )
            if self.total_tokens is not None
            else SimpleNamespace(total_tokens=None)
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"mapping": {item_id: "patch test" for item_id in item_ids}}
                        )
                    )
                )
            ],
            usage=usage,
        )


class _FakeHolder:
    def __init__(self, completions):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

    def get(self):
        return self.client


class _QuotaError(RuntimeError):
    status_code = 429


class _QuotaCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        raise _QuotaError("credit_balance_exhausted: no credits")


def _budget(tmp_path, *, limit=10):
    ledger = builder.TokenLedger(
        tmp_path / "ledger.json", model="test-model", reasoning_effort="medium"
    )
    return builder.TokenBudget(limit=limit, ledger=ledger)


def test_budget_stop_caches_completed_cluster_and_resume_only_calls_remaining(tmp_path):
    clusters = [
        builder.Cluster("cluster_a", ("assay a",)),
        builder.Cluster("cluster_b", ("assay b",)),
    ]
    cache = tmp_path / "clusters.jsonl"
    first_client = _FakeCompletions(total_tokens=10)
    with pytest.raises(builder.BudgetExhausted, match="remaining_clusters=1"):
        builder._map_clusters(
            client_holder=_FakeHolder(first_client),
            extraction=_extraction(),
            clusters=clusters,
            model="test-model",
            reasoning_effort="medium",
            workers=1,
            max_retries=1,
            retry_delay=0,
            cache_path=cache,
            budget=_budget(tmp_path),
        )
    assert first_client.calls == 1
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1

    second_client = _FakeCompletions(total_tokens=10)
    mapping = builder._map_clusters(
        client_holder=_FakeHolder(second_client),
        extraction=_extraction(),
        clusters=clusters,
        model="test-model",
        reasoning_effort="medium",
        workers=1,
        max_retries=1,
        retry_delay=0,
        cache_path=cache,
        budget=_budget(tmp_path),
    )
    assert second_client.calls == 1
    assert mapping == {"assay a": "patch test", "assay b": "patch test"}
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["lifetime_usage"]["total_tokens"] == 20
    assert ledger["attempts"] == 2


def test_missing_provider_usage_stops_but_retains_valid_cache(tmp_path):
    cache = tmp_path / "clusters.jsonl"
    with pytest.raises(builder.UsageUnavailable, match="omitted total_tokens"):
        builder._map_clusters(
            client_holder=_FakeHolder(_FakeCompletions(total_tokens=None)),
            extraction=_extraction(),
            clusters=[builder.Cluster("cluster_a", ("assay a",))],
            model="test-model",
            reasoning_effort="medium",
            workers=1,
            max_retries=1,
            retry_delay=0,
            cache_path=cache,
            budget=_budget(tmp_path),
        )
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1


def test_non_retryable_provider_error_stops_queued_clusters(tmp_path):
    completions = _QuotaCompletions()
    clusters = [
        builder.Cluster(f"cluster_{index}", (f"assay {index}",))
        for index in range(8)
    ]
    with pytest.raises(builder.NonRetryableAPIError, match="credit_balance_exhausted"):
        builder._map_clusters(
            client_holder=_FakeHolder(completions),
            extraction=_extraction(),
            clusters=clusters,
            model="test-model",
            reasoning_effort="low",
            workers=2,
            max_retries=5,
            retry_delay=0,
            cache_path=tmp_path / "clusters.jsonl",
            budget=_budget(tmp_path, limit=1000),
        )
    assert 1 <= completions.calls <= 2
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["status_counts"]["api_error"] == completions.calls


def test_numeric_json_severity_grades_are_normalized_to_strings():
    extraction = _severity_extraction()
    assert builder._validate_response(
        json.dumps({"mapping": {"v0000": 0, "v0001": 4}}),
        item_ids={"v0000": "negative", "v0001": "severe"},
        extraction=extraction,
    ) == {"v0000": "0", "v0001": "4"}


def test_free_form_reconciliation_still_rejects_numeric_json_labels():
    with pytest.raises(ValueError, match="bucket is not a string"):
        builder._validate_response(
            json.dumps({"mapping": {"v0000": 1}}),
            item_ids={"v0000": "patch test"},
            extraction=_extraction(),
        )


def test_builder_defaults_to_low_reasoning_for_the_neighborhood_lineage():
    assert builder.DEFAULT_MODEL == "gpt-5.4-mini"
    assert builder.DEFAULT_REASONING_EFFORT == "low"
    assert builder.PROMPT_VERSION == "starling_skin_embedding_bucket_mapping.v3"


def test_token_ledger_rejects_a_different_prompt_lineage(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "ledger_version": builder.LEDGER_VERSION,
                "prompt_version": "starling_skin_embedding_bucket_mapping.v2",
                "model": "test-model",
                "reasoning_effort": "low",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt identity mismatch"):
        builder.TokenLedger(path, model="test-model", reasoning_effort="low")


def test_prompt_registry_is_derived_from_every_llm_output_spec():
    expected = {
        (source_id, output.output_name)
        for source_id, source in builder.SOURCE_SPECS.items()
        for output in source.outputs
        if output.label_source == "llm"
    }
    actual = {
        (source_id, output_name)
        for source_id, outputs in builder._PROMPT_REGISTRY["prompts"].items()
        for output_name in outputs
    }
    assert actual == expected
    assert len(actual) == 9


def test_open_vocabulary_prompts_require_neighborhood_level_reconciliation():
    required = (
        "Consider all supplied values together",
        "smallest scientifically defensible set of canonical labels",
        "Multiple or all values may map to the same label",
        "Return every supplied ID exactly once",
    )
    for source_id, source in builder.SOURCE_SPECS.items():
        for output in source.outputs:
            if output.label_source != "llm" or output.bucket_pattern is not None:
                continue
            prompt = builder._PROMPT_REGISTRY["prompts"][source_id][
                output.output_name
            ]["prompt"]
            assert all(fragment in prompt for fragment in required), (
                source_id,
                output.output_name,
            )


def test_species_prompts_require_base_species_not_population_subgroups():
    for source_id, source in builder.SOURCE_SPECS.items():
        for output in source.outputs:
            if output.output_name != "global_species_context":
                continue
            prompt = builder._PROMPT_REGISTRY["prompts"][source_id][
                output.output_name
            ]["prompt"]
            assert "base" in prompt.casefold()
            assert "population descriptions" in prompt or "population qualifiers" in prompt


def test_only_phototoxicity_assay_inventory_uses_minibatch_clustering():
    algorithms = {
        (source_id, output.output_name): output.clustering
        for source_id, source in builder.SOURCE_SPECS.items()
        for output in source.outputs
        if output.label_source == "llm"
    }
    assert algorithms[
        ("phototoxicity_irritation_local_damage", "global_context")
    ] == "minibatch"
    assert all(
        algorithm == "lloyd"
        for key, algorithm in algorithms.items()
        if key != ("phototoxicity_irritation_local_damage", "global_context")
    )


def test_only_phototoxicity_context_uses_250_label_capped_clusters():
    configurations = {
        (source_id, output.output_name): (
            output.cluster_target_size,
            output.max_labels_per_call,
        )
        for source_id, source in builder.SOURCE_SPECS.items()
        for output in source.outputs
        if output.label_source == "llm"
    }
    photo_key = ("phototoxicity_irritation_local_damage", "global_context")
    assert configurations[photo_key] == (250, 250)
    assert all(
        configuration == (100, None)
        for key, configuration in configurations.items()
        if key != photo_key
    )


def test_oversized_embedding_cluster_is_split_at_the_call_cap(monkeypatch):
    class _OneCluster:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, embeddings):
            return np.zeros(len(embeddings), dtype=np.int64)

    monkeypatch.setattr(builder, "MiniBatchKMeans", _OneCluster)
    values = [f"value {index:03d}" for index in range(600)]
    clusters = builder._clusters_from_embeddings(
        values,
        np.zeros((600, 2), dtype=np.float32),
        clustering="minibatch",
        target_size=250,
        max_labels_per_call=250,
    )
    assert sorted(len(cluster.values) for cluster in clusters) == [100, 250, 250]
    assert {value for cluster in clusters for value in cluster.values} == set(values)


def test_finalization_can_load_an_explicit_compatible_snapshot_model(monkeypatch):
    calls = []

    def _fake_load(cache_dir, source_id, *, model, reasoning_effort, embedding_model):
        calls.append(model)
        return {"assay": {}} if model == "gpt-5.4" else None

    monkeypatch.setattr(builder, "_load_snapshot", _fake_load)
    loaded = builder._load_snapshot_from_models(
        None,
        "direct_skin_reaction",
        models=["gpt-5.4-mini", "gpt-5.4"],
        reasoning_effort="low",
        embedding_model="minilm",
    )
    assert loaded == ({"assay": {}}, "gpt-5.4")
    assert calls == ["gpt-5.4-mini", "gpt-5.4"]


def test_endpoint_aliases_merge_wording_not_distinct_nucleophiles_or_markers():
    output = next(
        item
        for item in reconciliation.SOURCE_SPECS["sensitization_aop"].outputs
        if item.output_name == "global_endpoint_context"
    )
    reconcile = lambda label: reconciliation._reconcile_output(
        "sensitization_aop", output, (label,), label
    )
    assert reconcile("cysteine peptide depletion") == "cysteine depletion"
    assert reconcile("lysine depletion") == "lysine depletion"
    assert reconcile("glutathione depletion") == "gsh depletion"
    assert reconcile("CD86 expression") == "cd86"
    assert reconcile("CD54 expression") == "cd54"
    assert reconcile("IL-8 expression") == "il-8"


def test_alias_resolver_accepts_identity_alias_but_rejects_a_real_cycle():
    assert reconciliation._resolve_alias(
        "draize test", {"draize test": "draize test"}
    ) == "draize test"
    with pytest.raises(ValueError, match="alias cycle"):
        reconciliation._resolve_alias("a", {"a": "b", "b": "a"})
