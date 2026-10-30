"""New task runtime contracts and compatible default retry races."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.conditioned_query_prior import build_query_prior_messages
from tools.chembl_tool.common.reasoning_validation import allowed_values_from_required_schema
from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runner


TEST_PROVIDER = SimpleNamespace(providers=[SimpleNamespace(temperature=0.0, reasoning_effort="")])


@pytest.mark.parametrize("independent", [False, True])
def test_gpt_adaptation_only_changes_instructions_and_is_opt_in(tmp_path, independent):
    from dataclasses import asdict
    from tools.chembl_tool.common.progressive_assay_reasoning import build_progressive_messages

    base = runner._task_contract("carcinogens")
    adapted = runner._task_contract("carcinogens", "gpt_calibration.v1")
    changed = {k for k, v in asdict(base).items() if v != asdict(adapted)[k]}
    assert changed == {"endpoint_name", "task_instructions", "query_prior_instructions"}
    assert base == runner._task_contract("carcinogens")
    payloads = []
    for contract in (base, adapted):
        messages = build_progressive_messages(
            contract=contract, levels=runner._levels("carcinogens"), current_level=1,
            query_smiles="CCO", condition_sentence="rodent", query_prior={"endpoint_prior": "mixed_or_unclear"},
            query_tool_summary={"content": "frozen tools"}, active={}, prior_state=None, independent=independent,
        )
        payloads.append(json.loads(messages[1]["content"]))
    assert {k for k in payloads[0] if payloads[0][k] != payloads[1][k]} == {"task_definition"}
    assert any("mixed_or_unclear is uncertainty" in s for s in payloads[1]["task_definition"]["instructions"])
    for branch in ("single", "final"):
        original, changed = [json.loads(build_query_prior_messages(c, branch, {"properties": "frozen"})[1]["content"])
                             for c in (base, adapted)]
        assert {k for k in original if original[k] != changed[k]} == {"task", "instructions"}


def test_adaptation_rejects_unsupported_tasks_and_old_priors_before_inference(monkeypatch):
    monkeypatch.setattr(runner, "_resolve_provider_pool_config", lambda *_: pytest.fail("no provider work"))
    for tasks in (["dili"], ["carcinogens", "ames"]):
        with pytest.raises(ValueError, match="unsupported prompt adaptation"):
            runner._prompt_adaptation("gpt_calibration.v1", tasks)
    with pytest.raises(ValueError, match="requires --fresh-query-priors"):
        runner.main(["--tasks", "carcinogens", "--prompt-adaptation", "gpt_calibration.v1"])
    with pytest.raises(SystemExit):
        family.main(["--tasks", "carcinogens", "--prompt-adaptation", "gpt_calibration.v1"])
    policy = {"prompt_adaptation": runner._prompt_adaptation("gpt_calibration.v1", ["carcinogens"])}
    for old, new in (({}, policy), (policy, {})):
        with pytest.raises(ValueError, match="reasoning_policy"):
            runner._merge_resume_manifest({"reasoning_policy": old}, {"reasoning_policy": new})
        with pytest.raises(ValueError, match="reasoning_policy"):
            family._merge_matched_resume_manifest(runner, {"reasoning_policy": old}, {"reasoning_policy": new})


def test_progressive_runner_uses_and_validates_prepared_adaptation(tmp_path):
    from copy import deepcopy

    frozen = Path("outputs/paper/carcinogens_scaffold_valid_gpt56_sol_batch_20260913/run/replicate_01/progressive/carcinogens/queries/query_idx00005/levels/level_1/prepared.json")
    if not frozen.exists():
        pytest.skip("Requires frozen Carcinogens inputs")
    prepared = json.loads(frozen.read_text())
    original = deepcopy(prepared)
    prepared["reasoning_policy"] = {"prompt_adaptation": runner._prompt_adaptation("gpt_calibration.v1", ["carcinogens"])}
    level = tmp_path / "levels/level_1"
    level.mkdir(parents=True)
    target = level / "prepared.json"
    target.write_text(json.dumps(prepared))
    args = SimpleNamespace(prepare_only=True, max_level=1)
    runner._run_query(args, runner.PreparedQuery("carcinogens", 5, tmp_path), None)
    request = json.loads((level / "request.json").read_text())
    text = request["messages"][1]["content"]
    assert "mixed_or_unclear is uncertainty" in text
    assert json.loads(target.read_text())["active_evidence"] == original["active_evidence"]
    prepared["reasoning_policy"]["prompt_adaptation"]["task_contract_sha256"]["carcinogens"] = "stale"
    target.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match="changed prepared prompt adaptation"):
        runner._run_query(args, runner.PreparedQuery("carcinogens", 5, tmp_path), None)


def test_matched_adaptation_materializes_same_cards_and_resumes_without_calls(tmp_path, monkeypatch):
    from copy import deepcopy

    frozen = Path("outputs/paper/carcinogens_scaffold_valid_gpt56_sol_batch_20260913/run/replicate_01/progressive")
    if not frozen.exists():
        pytest.skip("Requires frozen Carcinogens inputs")
    task, profile = "carcinogens", "gpt_calibration.v1"
    source, output, priors = [tmp_path / name for name in ("source", "output", "single_cache")]
    source.mkdir()
    manifest = json.loads((frozen / "experiment_manifest.json").read_text())
    manifest["tasks"] = [task]
    manifest["evaluation_indices_by_task"] = {task: [5]}
    (source / "experiment_manifest.json").write_text(json.dumps(manifest))
    originals = {}
    for level in range(1, 8):
        relative = Path(f"{task}/queries/query_idx00005/levels/level_{level}/prepared.json")
        prepared = json.loads((frozen / relative).read_text())
        originals[relative] = prepared
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_text(json.dumps(prepared))
    sample = next(iter(originals.values()))
    new_prior = (dict(endpoint_prior="mixed_or_unclear"), sample["query_tool_summary"], sample["reused_none_final"], 5)
    prior_manifest = priors / task / "none/manifest.json"
    prior_manifest.parent.mkdir(parents=True)
    prior_manifest.write_text(json.dumps({"model": manifest["model"], "indices": [5],
        "prompt_adaptation": runner._prompt_adaptation(profile, [task])}))
    monkeypatch.setattr(runner, "_load_reused_query_prior", lambda *a: deepcopy(new_prior))
    monkeypatch.setattr(runner, "_make_client", lambda *a: pytest.fail("no inference in preparation"))
    run_matched = family._run_matched_curve
    monkeypatch.setattr(family, "_run_matched_curve", lambda args: args)
    args = family.main(["--tasks", task, "--matched-progressive-root", str(source),
        "--output-root", str(output), "--matched-organizations", "progressive", "--prepare-only",
        "--model", manifest["model"], "--max-tokens", str(manifest["max_tokens"])])
    args.prompt_adaptation, args.matched_single_root = profile, str(priors)
    assert run_matched(args) == 0
    assert run_matched(args) == 0
    for relative, original in originals.items():
        current = json.loads((output / relative).read_text())
        assert current["active_evidence"] == original["active_evidence"]
        assert current["query_tool_summary"] == original["query_tool_summary"]
        assert current["query_prior"] == new_prior[0]
        assert current["reasoning_policy"]["prompt_adaptation"]["profile"] == profile
    requests = list(output.rglob("request.json"))
    assert requests and all("mixed_or_unclear is uncertainty" in p.read_text() for p in requests)
    args.prompt_adaptation = ""
    with pytest.raises(ValueError, match="incompatible refreshed prior prompt adaptation"):
        run_matched(args)


def test_alternate_retrieval_indices_preserve_benchmark_and_catalog(tmp_path):
    for scheme in ("scaffold", "random"):
        for subset in ("valid", "test"):
            old = runner._progressive_task_specs(scheme, subset)
            new = runner._progressive_task_specs(scheme, subset, tmp_path)
            for task, spec in new.items():
                assert spec.input_jsonl == old[task].input_jsonl
                assert spec.family_manifest == old[task].family_manifest
                assert spec.index == tmp_path / task / scheme / "assay_neighbor_index.pkl"


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
    monkeypatch.setattr(runner, "_make_client", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(runner, "_run_query_rounds", lambda *args, **kwargs: 0)
    assert runner._prepare_fresh_query_priors(args, specs, TEST_PROVIDER) == 0
    manifest = tmp_path / "priors/single_cache/dili/none/manifest.json"
    return args, specs, source, manifest


def test_prior_resume_transport_migration_preserves_history(prior_resume):
    args, specs, _, manifest = prior_resume
    previous = json.loads(manifest.read_text())
    previous["runner_sha256"] = "previous-runner-hash"
    manifest.write_text(json.dumps(previous))
    args.base_url = "http://127.0.0.1:50002/v1"
    assert runner._prepare_fresh_query_priors(args, specs, TEST_PROVIDER) == 0
    current = json.loads(manifest.read_text())
    assert current["base_url"] == args.base_url
    assert current["execution_history"] == [previous]
    assert current["model"] == previous["model"]
    assert current["input_sha256"] == previous["input_sha256"]


@pytest.mark.parametrize("task,profile", [
    ("bbb_martins", ""), ("bioavailability_ma", ""), ("skin_reaction", ""),
    ("carcinogens", "gpt_calibration.v1"),
])
def test_fresh_native_priors_reuse_messages_and_race_validation_retries(task, profile, monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace
    from tools.chembl_tool.common.experiment_retrieval import _query_only_retrieval
    from tools.chembl_tool.common.reasoning_payload import llm_query_payload

    monkeypatch.setattr(runner, "run", lambda args: args)
    args = runner.main(["--tasks", task, "--output-root", str(tmp_path / "priors"), "--retry-race-width", "6"])
    args.prompt_adaptation = profile
    source = tmp_path / "valid.jsonl"
    source.write_text('{"drug":"CCO","Y":0}\n')
    directory = runner._source_run_dir(task, 0, tmp_path / "priors/single_cache")
    directory.mkdir(parents=True)
    retrieval = _query_only_retrieval("CCO", mode="none")
    retrieval["query"].update(tools_prefetched=True, prefetched_molecule_properties={
        "tool_name": "molecule_properties", "status": "ok",
        "content": "[molecule_properties]\nexact molecular weight: 46\n[functional_group_tree]\nalcohol: 1",
    })
    (directory / "retrieval.json").write_text(json.dumps(retrieval))

    def response(messages):
        schema = json.loads(messages[1]["content"])["required_json_schema"]
        allowed = allowed_values_from_required_schema(messages)
        content = {k: sorted(allowed[k])[0] if k in allowed else ([] if isinstance(v, list) else "test")
                   for k, v in schema.items()}
        return {"content": content, "messages": messages}

    class Race:
        def __init__(self):
            self.widths, self.messages = [], []
        def chat_validated(self, messages, *, width, validate, **kwargs):
            self.widths.append(width)
            self.messages.append(messages)
            result = {"content": {}} if len(self.widths) == 1 else response(messages)
            assert bool(validate(result)) == (len(self.widths) == 1)
            return result
        def close(self):
            pass

    client = Race()
    monkeypatch.setattr(runner, "ParallelRetryClient", Race)
    monkeypatch.setattr(runner, "_make_client", lambda *a: client)
    monkeypatch.setattr(runner, "_run_query_rounds", lambda a, queries, client, root, run_query:
                        0 if run_query(a, queries[0], client)["status"] == "ok" else 1)
    assert runner._prepare_fresh_query_priors(args, {task: SimpleNamespace(input_jsonl=source)}, TEST_PROVIDER) == 0
    assert client.widths == [1, 6, 1]
    if profile:
        single_payload = json.loads(client.messages[0][1]["content"])
        final_payload = json.loads(client.messages[-1][1]["content"])
        assert "Distinguish features" in single_payload["instructions"][-1]
        assert any("mixed_or_unclear is uncertainty" in s for s in final_payload["instructions"])
        manifest = directory.parent.parent / "manifest.json"
        assert json.loads(manifest.read_text())["prompt_adaptation"] == runner._prompt_adaptation(profile, [task])
        args.prompt_adaptation = ""
        with pytest.raises(ValueError, match="changed fresh prior inputs/settings"):
            runner._prepare_fresh_query_priors(args, {task: SimpleNamespace(input_jsonl=source)}, TEST_PROVIDER)
        assert len(client.messages) == 3
        return
    native = importlib.import_module(f"tools.chembl_tool.tasks.{task}.run_reasoning_pipeline")
    direct_client = SimpleNamespace(chat_json=response)
    single = native._reason_single_molecule(direct_client, llm_query_payload(retrieval["query"]))
    final = native._run_final_reasoning(direct_client, retrieval, single, [])
    assert client.messages[0] == single["llm"]["messages"]
    assert client.messages[-1] == final["llm"]["messages"]
    assert json.loads((directory / "final_reasoning_output.json").read_text())["status"] == "ok"


def test_prior_resume_rejects_receipts_from_before_the_tree_tool_profile(prior_resume):
    args, specs, _, manifest = prior_resume
    previous = json.loads(manifest.read_text())
    previous.pop("tool_text_profile")
    manifest.write_text(json.dumps(previous))
    old_bytes = manifest.read_bytes()
    with pytest.raises(ValueError, match="changed fresh prior inputs/settings"):
        runner._prepare_fresh_query_priors(args, specs, TEST_PROVIDER)
    assert manifest.read_bytes() == old_bytes


@pytest.mark.parametrize("status", ["error", "ok"])
def test_cached_prior_tools_must_succeed_and_include_a_tree(prior_resume, monkeypatch, status):
    args, specs, _, _ = prior_resume
    directory = runner._source_run_dir("dili", 0, Path(args.single_source_root))
    directory.mkdir(parents=True)
    (directory / "retrieval.json").write_text(json.dumps({"query": {
        "input_smiles": "CCO", "prefetched_molecule_properties": {
            "status": status, "content": "failed call" if status == "error" else "old flat properties",
        },
    }}))
    monkeypatch.setattr(runner, "_run_query_rounds", lambda a, queries, client, root, run_query:
                        run_query(a, queries[0], client))
    with pytest.raises(ValueError, match="Fresh query priors require"):
        runner._prepare_fresh_query_priors(args, specs, TEST_PROVIDER)
    assert not (directory / "single_molecule_reasoning_output.json").exists()


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
        runner._prepare_fresh_query_priors(args, specs, TEST_PROVIDER)
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


def test_matched_rebuilt_index_is_hash_checked_and_heldout_validated(tmp_path):
    from types import SimpleNamespace
    from tools.chembl_tool.common.json_utils import sha256_file

    original = tmp_path / 'original.pkl'
    original.write_text('old index')
    rebuilt = tmp_path / 'rebuilt' / 'index.pkl'
    rebuilt.parent.mkdir()
    rebuilt.write_text('uncapped records')
    (rebuilt.parent / 'manifest.json').write_text('{"rebuilt":true}')
    spec = runner.ProgressiveTaskSpec(original, original, original)
    checks = []
    runtime = SimpleNamespace(
        _progressive_task_specs=lambda *_: {'ames': spec},
        _levels=lambda _: [1],
        _heldout_filter_validation=lambda **kwargs: checks.append(kwargs),
        task_root=lambda *_: tmp_path,
    )
    inputs = {'index': str(rebuilt), 'index_sha256': sha256_file(rebuilt),
              'input_sha256': sha256_file(original),
              'family_manifest_sha256': sha256_file(original), 'levels': [1]}
    source = {'split_scheme': 'scaffold', 'evaluation_subset': 'valid',
              'inputs': {'ames': inputs}}
    assert family._validate_matched_task(runtime, source, 'ames').index == rebuilt
    assert checks[0]['index_manifest'] == {'rebuilt': True}
    catalog = rebuilt.parent / 'family_manifest.json'
    catalog.write_text('{"corrected_source":true}')
    inputs.update(family_manifest=str(catalog), family_manifest_sha256=sha256_file(catalog))
    assert family._validate_matched_task(runtime, source, 'ames').family_manifest == catalog
    catalog.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='family_manifest_sha256'):
        family._validate_matched_task(runtime, source, 'ames')
    catalog.write_text('{"corrected_source":true}')
    rebuilt.write_text('changed')
    with pytest.raises(ValueError, match='index_sha256'):
        family._validate_matched_task(runtime, source, 'ames')
