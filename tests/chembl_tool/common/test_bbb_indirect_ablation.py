"""Scientific selection and portable replay contracts; no network calls."""

from argparse import Namespace
from dataclasses import replace
import json

import pytest

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.openai_provider_pool import (
    ProviderPoolConfig,
    ProviderSpec,
)
from tools.chembl_tool.common.record_budget import content_hash
from tools.chembl_tool.paper_experiments import prepared_evidence as replay
from tools.chembl_tool.tasks.bbb_martins.indirect_ablation import (
    prediction_only,
    select_indirect,
)


@pytest.mark.parametrize(
    "text,context,removed",
    [
        ("QikProp predicted logBB = -1.2", "", True),
        ("SwissADME result", "", True),
        ("An AdaBoost prediction", "", True),
        ("Predicted permeability", "Experimental assay", False),
        ("Predicted permeability; measured exposure also reported", "", False),
        ("No permeability study was reported", "", False),
        # This known lexical limitation is intentional, not a semantic audit.
        ("Predicted permeability; not measured", "", False),
    ],
)
def test_prediction_filter(text, context, removed):
    assert (
        prediction_only({"card": {"support_text": text, "assay_context": context}})
        is removed
    )


def test_balanced_budgets_are_nested_and_donor_capped():
    records = [
        {
            "id": f"{m}:{i}",
            "molecule_id": str(m),
            "group": f"group{i % 4}",
            "similarity": 1 - m / 100,
            "card": {"support_text": "Measured uptake"},
        }
        for m in range(25)
        for i in range(8)
    ]
    small = select_indirect(records, [], "balanced20")
    large = select_indirect(records, [], "balanced50")
    assert small == large[:20] and len(large) == 50
    assert all(sum(r["molecule_id"] == str(m) for r in large) <= 3 for m in range(25))
    assert select_indirect(list(reversed(records)), [], "balanced50") == large


def test_neighbor_fill_filter_does_not_replenish():
    records = [
        {"id": "predicted", "card": {"support_text": "QikProp prediction"}},
        {"id": "measured", "card": {"support_text": "Measured uptake"}},
        {"id": "unused", "card": {"support_text": "Measured uptake"}},
    ]
    assert select_indirect(records, ["predicted", "measured"], "guard50") == [
        records[1]
    ]
    assert select_indirect(records, [], "direct_guard") == []


def test_family_replay_runs_and_resumes_without_selecting_by_gold(
    tmp_path, monkeypatch
):
    from tools.chembl_tool.paper_experiments import (
        run_conditioned_assay_family_curve as cli,
    )

    source = tmp_path / "test.jsonl"
    source.write_text('{"drug":"CCO","Y":1}\n')
    provider = ProviderPoolConfig(
        providers=(
            ProviderSpec(
                name="test",
                base_url="http://example.invalid",
                model="model",
                api_key_env="",
                max_inflight=1,
                request_extra_body={},
            ),
        ),
        max_failovers=0,
    )
    manifest = {
        "version": replay.VERSION,
        "model": "model",
        "max_tokens": 100,
        "generation": replay.runtime._generation_settings(provider),
        "request_extra_body": {},
        "inputs": {
            "bbb_martins": {
                "input": {"path": str(source), "sha256": sha256_file(source)}
            }
        },
        "setting": "direct",
        "selection": {"policy": "neighbor_fill", "budget": 0},
    }
    messages = [{"role": "user", "content": "Predict the BBB label of CCO."}]
    directory = tmp_path / "runs/bbb_martins/query_00000"
    write_json_atomic(tmp_path / "experiment_manifest.json", manifest)
    write_json_atomic(
        directory / "prepared.json",
        {
            "manifest_hash": content_hash(manifest),
            "messages": messages,
            "request_hash": content_hash(messages),
            "context_ready": True,
            "card_alias_map": {},
            "query_hash": content_hash(
                {
                    "benchmark_profile": "",
                    "canonical_smiles": "CCO",
                    "condition_group": "",
                }
            ),
        },
    )
    write_json_atomic(directory / "evaluation.json", {"label": 1})

    class Client:
        calls = 0

        def chat_json(self, actual):
            assert actual == messages
            self.calls += 1
            return {
                "content": {
                    "bbb_prediction": "fail",
                    "confidence": "low",
                    "claims": [],
                    "supportive_card_ids": [],
                    "contradictory_card_ids": [],
                    "prediction_basis_card_ids": [],
                    "evidence_gaps": [],
                    "decision_summary": "A structure-only inference with limited evidence.",
                }
            }

        def close(self):
            pass

    client = Client()
    monkeypatch.setattr(
        replay.runtime, "_resolve_provider_pool_config", lambda args: provider
    )
    monkeypatch.setattr(replay.runtime, "_make_client", lambda *args: client)
    args = [
        "--run-prepared",
        "--tasks",
        "bbb_martins",
        "--output-root",
        str(tmp_path),
        "--model",
        "model",
        "--max-tokens",
        "100",
        "--parallelism",
        "1",
        "--retry-race-width",
        "1",
    ]
    assert cli.main(args) == 0
    assert cli.main(args) == 0
    assert client.calls == 1
    metrics = json.loads((tmp_path / "summary/bbb_martins/metrics.json").read_text())
    assert metrics["n_successful"] == 1 and metrics["accuracy"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "none",
        "model",
        "provider_model",
        "input",
        "request",
        "label",
        "profile",
        "benchmark_input",
        "capacity",
        "race",
    ],
)
def test_replay_rejects_invalid_contract_before_network(
    tmp_path, monkeypatch, corruption
):
    source = tmp_path / "test.jsonl"
    source.write_text('{"drug":"CCO","Y":1}\n')
    benchmark = tmp_path / "benchmark.json"
    write_json_atomic(
        benchmark,
        {
            "profile": "wrong"
            if corruption == "profile"
            else "tdc_scaffold_train_labels.v1",
            "tasks": {
                "bbb_martins": {
                    "test": {
                        "path": str(source),
                        "sha256": "wrong"
                        if corruption == "benchmark_input"
                        else sha256_file(source),
                    }
                }
            },
        },
    )
    spec = ProviderSpec(
        name="test",
        base_url="http://example.invalid",
        model="model",
        api_key_env="",
        max_inflight=1,
        request_extra_body={},
    )
    if corruption == "provider_model":
        spec = replace(spec, model="other")
    if corruption == "capacity":
        spec = replace(spec, max_inflight=2)
    provider = ProviderPoolConfig(providers=(spec,), max_failovers=0)
    manifest = {
        "version": replay.VERSION,
        "model": "model",
        "max_tokens": 100,
        "benchmark_profile": "tdc_scaffold_train_labels.v1",
        "generation": replay.runtime._generation_settings(provider),
        "request_extra_body": {},
        "inputs": {
            "bbb_martins": {
                "input": {"path": str(source), "sha256": sha256_file(source)}
            }
        },
        "setting": "direct_indirect",
        "selection": {"policy": "group_balanced", "budget": 20},
    }
    messages = [{"role": "user", "content": "evidence"}]
    prepared = {
        "manifest_hash": content_hash(manifest),
        "messages": messages,
        "request_hash": content_hash(messages),
        "query_hash": content_hash(
            {
                "benchmark_profile": manifest["benchmark_profile"],
                "canonical_smiles": "CCO",
                "condition_group": "",
            }
        ),
    }
    if corruption == "input":
        source.write_text('{"drug":"CCC","Y":1}\n')
    if corruption == "request":
        prepared["messages"][0]["content"] = "changed"
    directory = tmp_path / "runs/bbb_martins/query_00000"
    write_json_atomic(tmp_path / "experiment_manifest.json", manifest)
    write_json_atomic(directory / "prepared.json", prepared)
    write_json_atomic(
        directory / "evaluation.json", {"label": 0 if corruption == "label" else 1}
    )
    args = Namespace(
        output_root=str(tmp_path),
        tasks=["bbb_martins"],
        model="other" if corruption == "model" else "model",
        max_tokens=100,
        matched_progressive_root="",
        skip_tool_prefetch=False,
        retry_race_width=2 if corruption == "race" else 1,
        parallelism=1,
        endpoint_concurrency_budget=1,
        benchmark_manifest=str(benchmark),
        indices=None,
        limit=0,
        prepare_only=corruption == "none",
    )
    monkeypatch.setattr(
        replay.runtime, "_resolve_provider_pool_config", lambda args: provider
    )
    monkeypatch.setattr(
        replay.runtime,
        "_make_client",
        lambda *args: pytest.fail("network must not be reached"),
    )
    if corruption == "none":

        def check_round(args, queries, client, root, run_query):
            assert client is None and len(queries) == 1
            assert run_query(args, queries[0], client)["status"] == "ok"
            return []

        monkeypatch.setattr(replay.runtime, "_run_query_rounds", check_round)
        assert replay._run_prepared(args) == 0
    else:
        with pytest.raises(ValueError):
            replay._run_prepared(args)
