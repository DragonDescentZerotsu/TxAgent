"""New task runtime contracts and compatible default retry races."""

import json
from pathlib import Path

import pytest

from tools.chembl_tool.common.conditioned_query_prior import build_query_prior_messages
from tools.chembl_tool.common.reasoning_validation import allowed_values_from_required_schema
from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runner


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_new_task_levels_and_conditioned_prior_contract(task):
    contract = runner._task_contract(task)
    assert len(runner._levels(task)) == 7
    for scheme in ("scaffold", "random"):
        valid = runner._progressive_task_specs(scheme, "valid")[task]
        test = runner._progressive_task_specs(scheme, "test")[task]
        assert valid.index == test.index
        assert valid.index.is_file()
        assert valid.input_jsonl.name == "valid.jsonl"
        assert test.input_jsonl.name == "test.jsonl"
        assert valid.index == runner.SOURCE_PURITY_ROOT / "indices" / task / scheme / "assay_neighbor_index.pkl"
    query = {"external_condition": "reported experimental exposure"}
    messages = build_query_prior_messages(contract, "final", query, {"reasoning_summary": "prior"})
    payload = json.loads(messages[-1]["content"])
    assert payload["query"] == query
    assert payload["instructions"][:len(contract.task_instructions)] == list(contract.task_instructions)
    assert set(allowed_values_from_required_schema(messages)[contract.prediction_field]) == contract.prediction_values
    single = build_query_prior_messages(contract, "single", {"properties": "property tool text"})
    assert "required_json_schema" in json.loads(single[-1]["content"])
    assert "gold" not in json.dumps(messages).lower()


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_cli_new_tasks_default_race_with_small_budget(monkeypatch, task):
    monkeypatch.setattr(runner, "run", lambda args: args)
    monkeypatch.setattr(family, "_run_matched_curve", lambda args: args)
    progressive = runner.main(["--tasks", task, "--parallelism", "256"])
    assert progressive.retry_race_width == 6
    assert progressive.transport_max_retries == 0
    small = runner.main(["--tasks", task, "--parallelism", "2"])
    assert small.retry_race_width == 2
    flat = family.main(["--tasks", task, "--matched-progressive-root", "frozen",
                        "--output-root", "matched", "--parallelism", "256"])
    assert flat.retry_race_width == 6


def test_legacy_family_default_remains_usable(monkeypatch):
    monkeypatch.setattr(family, "run", lambda args: args)
    args = family.main(["--tasks", "bbb_martins"])
    assert args.retry_race_width == 1


def test_new_task_legacy_family_requires_matched_inputs():
    with pytest.raises(SystemExit):
        family.main(["--tasks", "dili"])


def test_query_structure_map_preserves_gold_and_rejects_different_molecule(tmp_path):
    from types import SimpleNamespace
    from tools.chembl_tool.common.starling.new_task_identity import normalize_new_task_identity
    from tools.chembl_tool.common.json_utils import sha256_file
    source = "O=c1cccc[nH]1"
    identity = normalize_new_task_identity("carcinogens", source)
    rows = [{"drug": identity["drug"], "Y": 1, "molecule_identity_key": identity["molecule_identity_key"],
             "condition_group": "species=rodent", "benchmark_row_id": "frozen"}]
    inp = tmp_path / "valid.jsonl"; inp.write_text(json.dumps(rows[0]) + "\n")
    mapping = {"version": "verified_query_structure.v1", "task": "carcinogens",
               "input_sha256": sha256_file(inp), "structures": [{"benchmark_smiles": identity["drug"],
               "source_parent_smiles": identity["source_parent_smiles"]}]}
    path = tmp_path / "structures.json"; path.write_text(json.dumps(mapping))
    args = SimpleNamespace(query_structure_map=str(path))
    changed, receipt = runner._query_structure_records(args, "carcinogens", inp, rows)
    assert {k:v for k,v in changed[0].items() if k != "reasoning_smiles"} == rows[0]
    assert "reasoning_smiles" not in rows[0]
    assert receipt["sha256"] == sha256_file(path)
    mapping["structures"][0]["source_parent_smiles"] = "CCO"
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="exact frozen tautomer identity"):
        runner._query_structure_records(args, "carcinogens", inp, rows)
    mapping["input_sha256"] = "stale"; path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="task/input mismatch"):
        runner._query_structure_records(args, "carcinogens", inp, rows)


@pytest.mark.parametrize("flags", [[], ["--fresh-query-priors", "--prepare-only"]])
def test_query_priors_only_rejects_incompatible_mode(flags):
    with pytest.raises(SystemExit):
        runner.main(["--tasks", "dili", "--query-priors-only", *flags])


@pytest.mark.parametrize("prior_failed", [0, 1])
def test_query_priors_only_never_prepares_retrieval(monkeypatch, prior_failed):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(runner, "_resolve_provider_pool_config",
                        lambda args: SimpleNamespace(providers=[]))
    monkeypatch.setattr(runner, "_prepare_fresh_query_priors",
                        lambda *args: calls.append("priors") or prior_failed)
    monkeypatch.setattr(runner, "_validate_inputs",
                        lambda *args: pytest.fail("prior-only must not prepare retrieval or levels"))
    code = runner.main(["--tasks", "dili", "--fresh-query-priors", "--query-priors-only"])
    assert code == prior_failed
    assert calls == ["priors"]


@pytest.fixture
def prior_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(runner, "run", lambda args: args)
    args = runner.main(["--tasks", "dili", "--output-root", str(tmp_path / "priors")])
    source = tmp_path / "valid.jsonl"
    source.write_text('{"drug":"CCO","Y":0}\n')
    specs = {"dili": SimpleNamespace(input_jsonl=source)}
    monkeypatch.setattr(runner, "_make_client", lambda *args: object())
    monkeypatch.setattr(runner, "_run_query_rounds", lambda *args, **kwargs: 0)
    assert runner._prepare_fresh_query_priors(args, specs, None) == 0
    manifest = tmp_path / "priors/single_cache/dili/none/manifest.json"
    return args, specs, source, manifest


def test_prior_resume_transport_migration_preserves_history(prior_resume):
    args, specs, _, manifest = prior_resume
    previous = json.loads(manifest.read_text())
    previous["runner_sha256"] = "previous-runner-hash"
    manifest.write_text(json.dumps(previous))
    args.base_url = "http://127.0.0.1:50002/v1"
    assert runner._prepare_fresh_query_priors(args, specs, None) == 0
    current = json.loads(manifest.read_text())
    assert current["base_url"] == args.base_url
    assert current["execution_history"] == [previous]
    assert current["model"] == previous["model"]
    assert current["input_sha256"] == previous["input_sha256"]


@pytest.mark.parametrize("change", ["model", "input"])
def test_prior_resume_still_rejects_scientific_changes(prior_resume, change):
    args, specs, source, manifest = prior_resume
    old_bytes = manifest.read_bytes()
    args.base_url = "http://127.0.0.1:50002/v1"
    if change == "model":
        args.model = "different-model"
    else:
        source.write_text('{"drug":"CCC","Y":0}\n')
    with pytest.raises(ValueError, match="changed fresh prior inputs/settings"):
        runner._prepare_fresh_query_priors(args, specs, None)
    assert manifest.read_bytes() == old_bytes


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
@pytest.mark.parametrize("independent", [False, True])
def test_frozen_cards_render_scientific_family_labels_without_mutation(task, independent):
    from copy import deepcopy
    from tools.chembl_tool.common.progressive_assay_reasoning import build_progressive_messages

    root = Path("outputs/paper/starling_conditioned_dili_carcinogens_scaffold_4_2_v1")
    query_root = root / task / "valid/frozen_inputs" / task / "queries"
    paths = sorted(query_root.glob("*/levels/level_1/prepared.json"))
    if not paths:
        pytest.skip("Requires the frozen new-task preparation artifacts")
    prepared = next((row for path in paths if (row := json.loads(path.read_text()))["active_evidence"]), None)
    assert prepared is not None
    active = deepcopy(prepared["active_evidence"])
    levels = runner._levels(task)
    messages = build_progressive_messages(
        contract=runner._task_contract(task), levels=levels, current_level=1,
        query_smiles=prepared["query_smiles"], condition_sentence=prepared["condition_sentence"],
        query_prior=prepared["query_prior"], query_tool_summary=prepared["query_tool_summary"],
        active=active, prior_state=None, independent=independent,
    )
    payload = json.loads(messages[-1]["content"])
    text = json.dumps(payload).lower()
    assert all(term not in text for term in ("actual_voter", "nonvoter", "source votes", "study-level carcinogenicity votes"))
    assert active == prepared["active_evidence"]
    assert levels[0]["endpoint_group"] == prepared["level_definition"]["endpoint_group"]
    assert levels[0]["source_groups"] == prepared["level_definition"]["source_groups"]
    assert payload["level_context"]["full_level_plan"][0] == runner._level_prompt_definition(levels[0])
    original = sorted((card["support_text"], card["endpoint"], card.get("reported_value"))
                      for analog in active.values() for card in analog["cards"].values())
    rendered = sorted((card["support_text"], card["endpoint"], card.get("reported_value"))
                      for analog in payload["active_evidence"] for card in analog["evidence_cards"])
    assert rendered == original
    assert all(card["evidence_family"] == levels[0]["family_label"]
               for analog in payload["active_evidence"] for card in analog["evidence_cards"])


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_display_change_preserves_real_none_messages(task):
    root = Path(f"outputs/paper/starling_conditioned_{task}_gold_v4_scaffold_4_2_v1")
    run = root / task / "valid/priors/single_cache" / task / "none/runs/none_idx00000"
    if not (run / "final_reasoning_output.json").exists():
        pytest.skip("Requires completed new-task prior traces")
    for branch, filename in [("single", "single_molecule_reasoning_output.json"),
                             ("final", "final_reasoning_output.json")]:
        llm = json.loads((run / filename).read_text())["llm"]
        old = [message for message in llm["messages"] if message["role"] in {"system", "user"}][:2]
        payload = json.loads(old[1]["content"])
        current = build_query_prior_messages(runner._task_contract(task), branch, payload["query"],
                                             payload.get("single_molecule_analysis"))
        assert current == old


def test_matched_reuse_needs_no_retired_alias_guard(tmp_path):
    from types import SimpleNamespace
    from tools.chembl_tool.common.json_utils import sha256_file

    payload = tmp_path / 'input.jsonl'
    payload.write_text('{}\n')
    (tmp_path / 'manifest.json').write_text('{}')
    spec = SimpleNamespace(input_jsonl=payload, index=payload, family_manifest=payload)
    checks = []
    runtime = SimpleNamespace(
        _progressive_task_specs=lambda *_: {'dili': spec},
        _levels=lambda _: [1, 2],
        _heldout_filter_validation=lambda **kwargs: checks.append(kwargs),
        task_root=lambda *_: tmp_path,
    )
    inputs = {key: sha256_file(payload) for key in (
        'input_sha256', 'index_sha256', 'family_manifest_sha256')}
    inputs['levels'] = [1, 2]
    source = {'split_scheme': 'scaffold', 'evaluation_subset': 'valid',
              'inputs': {'dili': inputs}}
    assert family._validate_matched_task(runtime, source, 'dili') is spec
    assert len(checks) == 1
    inputs['heldout_direct_alias_guard'] = {'legacy': True}
    with pytest.raises(ValueError, match='heldout direct-alias guard'):
        family._validate_matched_task(runtime, source, 'dili')
