"""Scientific selection and portable replay contracts; no network calls."""

from argparse import Namespace
from dataclasses import replace
import gzip
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
    PROMPT,
    prepare,
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
        ("Predicted permeability; not measured", "", True),
    ],
)
def test_prediction_filter(text, context, removed):
    assert (
        prediction_only({"card": {"support_text": text, "assay_context": context}})
        is removed
    )


@pytest.mark.parametrize(
    "card,removed",
    [
        # Archived Joseph #217 Record 22-1 and #472 Record 31-1: classifications
        # derived from assays, not computational-only predictions.
        (
            {
                "support_text": "In PAMPA-BBB using a brain lipid porcine membrane, compound 17 "
                "showed low Pe and was predicted not to penetrate the CNS "
                "(classification threshold: Pe > 3.95 × 10^-6 cm s^-1 for CNS+)."
            },
            False,
        ),
        (
            {
                "assay_context": "Parallel Artificial Membrane Permeability assay (PAMPA)",
                "support_text": "Using the Parallel Artificial Membrane Permeability assay "
                "(PAMPA) method, indolylpiperidine 3 was predicted to be able to cross "
                "the human blood-brain barrier (BBB) by passive permeation (classified as CNS+).",
            },
            False,
        ),
        (
            {
                "support_text": "Predicted CNS+ from a Pe of 4.7e-6 cm/s",
                "experimental_details": {"assay_type": "PAMPA-BBB"},
            },
            False,
        ),
        ({"support_text": "MDCK-MDR1 permeability predicted CNS access."}, False),
        (
            {"assay_context": "SwissADME", "support_text": "Predicted BBB negative."},
            True,
        ),
        ({"assay_context": "BOILED-Egg model", "reported_value": "permeable"}, True),
        ({"assay_context": "pkCSM", "reported_value": 0.7}, True),
        ({"support_text": "A QSAR model predicted PAMPA permeability."}, True),
        (
            {
                "assay_context": "mdck",
                "qualifying_conditions": "in silico prediction using QikProp",
                "support_text": "QikProp predicted permeability to MDCK cells.",
            },
            True,
        ),
        (
            {
                "assay_context": "in vitro BBB model",
                "support_text": "In an in vitro BBB model, "
                "metopimazine showed a permeability coefficient of 9.35e-3 cm/min.",
            },
            False,
        ),
        (
            {
                "assay_context": "in silico prediction",
                "support_text": "PAMPA permeability = 4.7",
                "experimental_details": {
                    "assay_type": "PAMPA",
                    "biological_system": "in vitro",
                },
            },
            True,
        ),
        (
            {
                "support_text": "Predicted CNS+; no PAMPA assay was performed.",
                "experimental_details": {"assay_type": "PAMPA"},
            },
            True,
        ),
        ({"support_text": "Predicted BBB positive; PAMPA was not performed."}, True),
        (
            {
                "support_text": "PAMPA was not performed, but MDCK assay showed low permeability; "
                "SwissADME predicted CNS+."
            },
            False,
        ),
        (
            {"support_text": "Predicted BBB positive; not experimentally measured."},
            True,
        ),
        (
            {
                "support_text": "PAMPA was not performed, but measured brain uptake was low; "
                "SwissADME predicted BBB positive."
            },
            False,
        ),
        (
            {
                "assay_context": "SwissADME",
                "support_text": "Measured brain exposure was "
                "low and SwissADME predicted BBB positive.",
            },
            False,
        ),
        (
            {
                "support_text": "Predicted BBB positive. Microdialysis showed brain exposure."
            },
            False,
        ),
        (
            {
                "support_text": "The compound is not a P-gp substrate in a measured transport assay; "
                "SwissADME predicted CNS+."
            },
            False,
        ),
        (
            {"support_text": "Permeability not reported", "reported_value": "positive"},
            False,
        ),
    ],
)
def test_filter_preserves_assay_classifications_and_removes_computational_only(
    card, removed
):
    assert prediction_only({"card": card}) is removed


@pytest.mark.parametrize(
    "card",
    [
        {
            "assay_context": "PAMPA-BBB",
            "support_text": "In vitro studies that simulate "
            "the BBB (PAMPA-BBB) demonstrated null permeability.",
        },
        {
            "assay_context": "Bidirectional transport studies in Caco-2 cell monolayers",
            "support_text": "Transport studies showed substrate activity.",
            "extra_details": "A computational prediction also supports substrate activity.",
        },
        {
            "assay_context": "parallel artificial membrane permeation assay",
            "support_text": "The parallel artificial membrane permeation assay was performed "
            "and Pe predicted CNS+.",
        },
        {
            "support_text": "Carrier-mediated brain entry was reported.",
            "extra_details": "Despite predicted lipophobicity.",
        },
    ],
)
def test_experimental_models_and_incidental_predictions_are_not_computational_only(
    card,
):
    assert not prediction_only({"card": card})


def test_prepare_pins_current_filter_and_rejects_old_manifest(tmp_path):
    bundle = tmp_path / "inputs.json.gz"
    record = {
        "id": "assay",
        "kind": "indirect",
        "similarity": 0.7,
        "canonical_smiles": "CCO",
        "group": "passive_permeability",
        "card": {"support_text": "PAMPA predicted CNS+"},
    }
    data = {
        "system_prompt_hash": content_hash(PROMPT.read_text()),
        "manifest": {"benchmark_profile": "test"},
        "evaluation_rows": [{"drug": "CCO", "Y": 0}],
        "records": {"assay": record},
        "queries": [
            {
                "query": {"smiles": "CCO"},
                "query_prior": {},
                "required_json_schema": {},
                "prepared_base": {},
                "direct": [],
                "candidates": ["assay"],
                "original_fill_ids": ["assay"],
            }
        ],
    }
    with gzip.open(bundle, "wt") as stream:
        json.dump(data, stream)
    write_json_atomic(
        bundle.with_suffix(".manifest.json"), {"sha256": sha256_file(bundle)}
    )
    root = tmp_path / "prepared"
    prepare(bundle, root, ["guard50"])
    receipt = json.loads((root / "guard50/preparation_receipt.json").read_text())
    assert receipt["prediction_filter"] == "v2"
    assert receipt["indirect_counts"] == [1]
    manifest_path = root / "guard50/experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["preparation"]["prediction_filter"] == "v2"
    manifest["preparation"]["prediction_filter"] = "v1"
    write_json_atomic(manifest_path, manifest)
    with pytest.raises(ValueError, match="Changed prepared input"):
        prepare(bundle, root, ["guard50"])


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
