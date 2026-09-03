import argparse
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import pytest

from tools.chembl_tool.paper_experiments import molecular_evidence_agent as agent_runner
from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    DEPLOYMENT_VISIBLE_PREFETCHED,
    EXPERIMENTS,
    IDENTITY_BLIND,
    PARENT_DISJOINT,
    SCAFFOLD_DISJOINT,
    _command,
    _parse_args,
    _prepare_policy_selection,
    experiment_for_split,
    experiment_run_root,
    paper_root_for_split,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    _benchmark_provenance,
    _matrix_manifest_path,
    _paper_root_for_evaluation_subset,
    _validate_concurrency,
    _validate_inputs,
    _validate_reference_pool,
    _validate_retrieval_ablation_args,
    _write_json_atomic,
    experiments_for_starling_benchmark,
)
from tools.chembl_tool.paper_experiments.minimol_retrieval_contract import (
    paper_root_for_minimol_retrieval,
)
from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
    INDEX_SPECS,
    _apply_source_evidence_overrides,
    _collect_existing_index_meta,
    _load_existing_summary,
    heldout_labels_path,
    normalize_heldout_subsets,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _group_prompt_payload,
    _llm_query_payload,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING as BIO_STARLING
from tools.chembl_tool.tasks.bbb_martins.experiment_config import STARLING as BBB_STARLING
from tools.chembl_tool.tasks.skin_reaction.experiment_config import (
    CHEMBL as SKIN_CHEMBL,
    STARLING as SKIN_STARLING,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_evidence_library import (
    DEFAULT_INDEX_PATH as DEFAULT_SKIN_INDEX_PATH,
)


def test_frozen_matrix_has_unique_expected_conditions():
    names = [experiment.name for experiment in EXPERIMENTS]
    assert len(names) == 29
    assert len(names) == len(set(names))
    assert "bioavailability_ma__starling_full_mechanism" in names
    assert "bbb_martins__starling_direct" in names
    assert "bbb_martins__starling_full_flat" in names
    assert "bbb_martins__starling_full_mechanism" in names
    assert "skin_reaction__starling_direct" in names
    assert "skin_reaction__starling_full_flat" in names
    assert "skin_reaction__starling_full_mechanism" in names
    assert "clintox__starling_direct" in names
    assert "clintox__starling_full_flat" in names
    assert "clintox__starling_full_mechanism" in names


def test_skin_defaults_use_canonical_direct_aop_source():
    skin_experiments = [item for item in EXPERIMENTS if item.task == "skin_reaction"]
    starling_experiments = [item for item in skin_experiments if item.source == "starling"]
    assert starling_experiments
    assert {item.index for item in starling_experiments} == {str(DEFAULT_SKIN_INDEX_PATH)}

    skin_specs = [item for item in INDEX_SPECS if item["task"] == "Skin_Reaction"]
    assert skin_specs == [
        {
            "name": "skin_reaction_starling_v7",
            "task": "Skin_Reaction",
            "task_id": "skin_reaction",
            "normalized_root": (
                "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
                "starling_normalized_v7"
                ),
                "view": "full",
                "filter_source_id": "direct_skin_reaction",
            }
        ]


def test_starling_benchmark_matrix_reuses_conditions_but_replaces_inputs_and_indices():
    experiments = experiments_for_starling_benchmark("random")
    assert len(experiments) == 29
    assert {experiment.task for experiment in experiments} == {
        "bbb_martins",
        "bioavailability_ma",
        "clintox",
        "skin_reaction",
    }
    assert all("/random/test.jsonl" in experiment.input_jsonl for experiment in experiments)

    chembl = next(item for item in experiments if item.name == "bbb_martins__chembl_direct")
    starling = next(item for item in experiments if item.name == "bbb_martins__starling_direct")
    assert chembl.index == EXPERIMENTS[1].index
    assert "molecular_evidence_agent_starling_random_record_agreement70_split811_v1/evidence" in starling.index
    assert starling.index.endswith("bbb_starling_v7/08_neighbor_index")
    v7_starling = [
        item
        for item in experiments
        if item.source == "starling" and item.task != "clintox"
    ]
    assert all("starling_v7" in item.index for item in v7_starling)
    assert all("_v3" not in item.index and not item.index.endswith(".pkl") for item in v7_starling)
    clintox = next(
        item for item in experiments if item.name == "clintox__starling_direct"
    )
    assert clintox.index.endswith(
        "clintox_starling_full/starling_clintox_neighbor_index.pkl"
    )


def test_query_only_none_does_not_require_its_placeholder_index(tmp_path):
    input_jsonl = tmp_path / "test.jsonl"
    input_jsonl.write_text('{"drug":"CCO","Y":1}\n', encoding="utf-8")
    none = next(item for item in EXPERIMENTS if item.mode == "none")
    experiment = replace(
        none,
        input_jsonl=str(input_jsonl),
        index=str(tmp_path / "absent-placeholder.pkl"),
    )

    _validate_inputs([experiment])


def test_starling_benchmark_matrix_can_select_valid_without_changing_indices():
    test_experiments = experiments_for_starling_benchmark("random")
    valid_experiments = experiments_for_starling_benchmark(
        "random",
        evaluation_subset="valid",
    )

    assert all("/random/valid.jsonl" in item.input_jsonl for item in valid_experiments)
    assert [item.index for item in valid_experiments] == [
        item.index for item in test_experiments
    ]


def test_starling_benchmark_matrix_accepts_isolated_data_and_index_lineage(tmp_path):
    data_root = tmp_path / "processed_starling_record_supported_v2"
    index_root = tmp_path / "molecular_evidence_agent_starling_scaffold_record_supported_v2"

    experiments = experiments_for_starling_benchmark(
        "scaffold",
        evaluation_subset="valid",
        data_root=data_root,
        canonical_paper_root=index_root,
    )

    assert all(
        item.input_jsonl.startswith(str(data_root)) for item in experiments
    )
    starling = next(
        item for item in experiments if item.name == "bbb_martins__starling_direct"
    )
    assert starling.index.startswith(str(index_root / "evidence"))


def test_partial_starling_index_rebuild_can_preserve_existing_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"random": {"bbb": {"ok": True}}}), encoding="utf-8")

    assert _load_existing_summary(summary) == {"random": {"bbb": {"ok": True}}}
    assert _load_existing_summary(tmp_path / "missing.json") == {}


def test_starling_index_source_override_is_explicit_and_nonmutating():
    specs = [{"name": "skin", "source_evidence": "historical.jsonl"}]

    updated = _apply_source_evidence_overrides(specs, ["skin=aligned.jsonl"])

    assert updated[0]["source_evidence"] == "aligned.jsonl"
    assert specs[0]["source_evidence"] == "historical.jsonl"


def test_starling_index_source_override_rejects_unselected_name():
    with pytest.raises(SystemExit, match="does not match a selected index"):
        _apply_source_evidence_overrides(
            [{"name": "skin", "source_evidence": "historical.jsonl"}],
            ["bbb=other.jsonl"],
        )


def test_index_summary_can_be_rebuilt_after_parallel_partial_builds(tmp_path):
    meta_path = (
        tmp_path
        / "molecular_evidence_agent_starling_random_record_agreement70_split811_v1"
        / "evidence"
        / "example"
        / "meta.json"
    )
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({"zero_parent_overlap": True}), encoding="utf-8")

    collected = _collect_existing_index_meta(
        splits=["random"],
        specs=[{"name": "example", "meta_filename": "meta.json"}],
        output_root=tmp_path,
    )

    assert collected == {"random": {"example": {"zero_parent_overlap": True}}}


def test_heldout_subset_scope_is_canonical_and_rejects_duplicates(tmp_path):
    assert normalize_heldout_subsets(["test", "valid"]) == ("valid", "test")
    assert heldout_labels_path(tmp_path, "BBB_Martins", "scaffold", ["test"]).name == (
        "test_molecule_labels.jsonl"
    )
    with pytest.raises(ValueError, match="Unsupported held-out subsets"):
        normalize_heldout_subsets(["test", "test"])
    with pytest.raises(ValueError, match="train pool"):
        normalize_heldout_subsets(["valid"])


@pytest.mark.parametrize(
    ("reference_pool", "heldout_filename", "should_pass"),
    [
        ("train", "heldout_molecule_labels.jsonl", True),
        ("train", "test_molecule_labels.jsonl", False),
        ("train_valid", "test_molecule_labels.jsonl", True),
        ("train_valid", "heldout_molecule_labels.jsonl", False),
    ],
)
def test_matrix_reference_pool_matches_index_scope(
    tmp_path,
    reference_pool,
    heldout_filename,
    should_pass,
):
    index_path = tmp_path / "starling.pkl"
    index_path.write_bytes(b"placeholder")
    index_path.with_suffix(".meta.json").write_text(
        json.dumps({"source": {"heldout_labels_jsonl": str(tmp_path / heldout_filename)}})
    )
    experiment = replace(
        next(item for item in EXPERIMENTS if item.name == "bbb_martins__starling_direct"),
        index=str(index_path),
    )
    args = argparse.Namespace(
        reference_pool=reference_pool,
        evaluation_subset="test",
        retrieval_feature="morgan",
    )
    if should_pass:
        _validate_reference_pool([experiment], args)
    else:
        with pytest.raises(SystemExit, match="requires an index excluding"):
            _validate_reference_pool([experiment], args)


def test_starling_minimol_matrix_uses_descriptors_and_isolated_output_root():
    experiments = experiments_for_starling_benchmark(
        "scaffold",
        retrieval_feature="minimol",
    )
    none = next(item for item in experiments if item.name == "bbb_martins__none")
    direct = next(item for item in experiments if item.name == "bbb_martins__chembl_direct")

    assert none.index == EXPERIMENTS[0].index
    assert direct.index.endswith(
        "minimol_retrieval_features_v7_record_agreement70_split811_v1/scaffold/descriptors/"
        "bbb_martins__chembl_direct.json"
    )
    assert paper_root_for_minimol_retrieval("scaffold").name == (
        "molecular_evidence_agent_starling_scaffold_record_agreement70_split811_v1_minimol_retrieval"
    )


def test_starling_minimol_matrix_accepts_lineage_specific_feature_root(tmp_path):
    feature_root = tmp_path / "current-minimol-features"
    experiments = experiments_for_starling_benchmark(
        "scaffold",
        evaluation_subset="valid",
        retrieval_feature="minimol",
        minimol_feature_root=feature_root,
    )

    direct = next(item for item in experiments if item.name == "bbb_martins__chembl_direct")
    assert direct.index == str(
        feature_root / "scaffold/descriptors/bbb_martins__chembl_direct.json"
    )


def test_starling_matrix_uses_selection_specific_atomic_manifests(tmp_path):
    experiments = experiments_for_starling_benchmark("random")
    args = argparse.Namespace(
        visibility_mode="identity_blind",
        neighbor_identity_policy="operational",
        experiments=["bbb_martins__none"],
    )
    bbb = next(item for item in experiments if item.name == "bbb_martins__none")
    bio = next(item for item in experiments if item.name == "bioavailability_ma__none")
    bbb_path = _matrix_manifest_path(tmp_path, args, [bbb])
    bio_path = _matrix_manifest_path(tmp_path, args, [bio])

    assert bbb_path != bio_path
    assert "bbb_martins" in bbb_path.name
    assert "bioavailability_ma" in bio_path.name

    _write_json_atomic(bbb_path, {"selected_experiments": [bbb.name]})
    assert json.loads(bbb_path.read_text()) == {
        "selected_experiments": ["bbb_martins__none"]
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_starling_matrix_provenance_hashes_split_inputs(tmp_path):
    experiments = experiments_for_starling_benchmark("random")
    task_dir = tmp_path / "BBB_Martins"
    split_dir = task_dir / "random"
    split_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol_version": "test.protocol.v1",
                "identity_normalizer_version": "test.identity.v1",
                "seed": 7,
                "parent_label_policy": {"agreement_threshold": 0.7},
                "split_size_policy": {"valid_fraction": 0.1, "test_fraction": 0.1},
                "source_metadata": {"revision": "abc"},
            }
        )
    )
    (split_dir / "summary.json").write_text(
        json.dumps({"train_test_identity_overlap": 0})
    )
    (split_dir / "test.jsonl").write_text('{"drug":"CCO","Y":1}\n')
    (split_dir / "valid.jsonl").write_text('{"drug":"CCN","Y":0}\n')
    (split_dir / "valid_molecule_labels.jsonl").write_text(
        '{"drug":"CCN","Y":0,"molecule_identity_key":"QUSNBJAOOMFDIB-UHFFFAOYSA-N"}\n'
    )
    (split_dir / "test_molecule_labels.jsonl").write_text(
        '{"drug":"CCO","Y":1,"molecule_identity_key":"LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}\n'
    )
    (split_dir / "heldout_molecule_labels.jsonl").write_text(
        '{"drug":"CCN","Y":0,"molecule_identity_key":"QUSNBJAOOMFDIB-UHFFFAOYSA-N"}\n'
        '{"drug":"CCO","Y":1,"molecule_identity_key":"LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}\n'
    )

    provenance = _benchmark_provenance(
        "random",
        [next(item for item in experiments if item.task == "bbb_martins")],
        data_root=tmp_path,
    )

    bbb = provenance["bbb_martins"]
    assert bbb["protocol_version"] == "test.protocol.v1"
    assert bbb["source_metadata"] == {"revision": "abc"}
    assert bbb["parent_label_policy"] == {"agreement_threshold": 0.7}
    assert bbb["split_size_policy"] == {"valid_fraction": 0.1, "test_fraction": 0.1}
    assert bbb["split_summary"]["train_test_identity_overlap"] == 0
    assert len(bbb["test_jsonl_sha256"]) == 64
    assert len(bbb["valid_jsonl_sha256"]) == 64
    assert len(bbb["valid_molecule_labels_sha256"]) == 64
    assert len(bbb["test_molecule_labels_sha256"]) == 64
    assert len(bbb["heldout_molecule_labels_sha256"]) == 64


def test_starling_matrix_accepts_conditioned_provenance_names(tmp_path):
    experiment = next(
        item for item in experiments_for_starling_benchmark("scaffold")
        if item.task == "bbb_martins"
    )
    split_dir = tmp_path / "BBB_Martins" / "scaffold"
    split_dir.mkdir(parents=True)
    (split_dir / "summary.json").write_text(
        json.dumps({"benchmark": "conditioned_benchmark", "contract": "conditioned_benchmark.v1"})
    )
    for subset in ("valid", "test"):
        (split_dir / f"{subset}.jsonl").write_text('{"drug":"CCO","Y":1}\n')
        (split_dir / f"{subset}_molecule_condition_labels.jsonl").write_text(
            '{"drug":"CCO","Y":1,"condition_group":"none"}\n'
        )
    (split_dir / "heldout_molecule_condition_labels.jsonl").write_text(
        '{"drug":"CCO","Y":1,"condition_group":"none"}\n'
    )

    provenance = _benchmark_provenance(
        "scaffold", [experiment], data_root=tmp_path
    )["bbb_martins"]
    assert provenance["benchmark"] == "conditioned_benchmark"
    assert provenance["contract"] == "conditioned_benchmark.v1"
    assert provenance["task_summary_path"] == str(split_dir / "summary.json")
    assert provenance["heldout_molecule_labels_jsonl"].endswith(
        "heldout_molecule_condition_labels.jsonl"
    )


def test_starling_matrix_accepts_manifest_only_mode():
    from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
        _parse_args as parse_starling_args,
    )

    args = parse_starling_args(
        ["--benchmark-split", "random", "--manifest-only"]
    )

    assert args.manifest_only is True
    assert args.base_url is None
    assert args.model == "nvidia/GLM-5.2-NVFP4"
    assert args.reasoning_effort == ""
    assert args.api_key_env == "LITELLM_API_KEY"
    assert args.evaluation_subset == "valid"
    assert args.visibility_mode == IDENTITY_BLIND
    assert args.neighbor_identity_policy == PARENT_DISJOINT
    assert args.parallelism == 128
    assert args.retrieval_preparation_workers == 8
    assert args.neighbor_selector == "similarity"
    assert args.neighbor_context_profile == "standard"
    assert args.top_k_per_group == 3
    assert args.min_similarity == 0.3
    assert args.tool_service_url == "http://127.0.0.1:8766"


def test_starling_matrix_isolates_nonstandard_retrieval_profiles():
    from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
        _parse_args as parse_starling_args,
    )

    missing_root = parse_starling_args(
        [
            "--benchmark-split",
            "scaffold",
            "--neighbor-selector",
            "query_feature_coverage",
            "--neighbor-context-profile",
            "coverage_aware",
        ]
    )
    try:
        _validate_retrieval_ablation_args(missing_root)
    except SystemExit as error:
        assert "explicit --output-root" in str(error)
    else:
        raise AssertionError("Expected a nonstandard profile to require an isolated root")

    isolated = parse_starling_args(
        [
            "--benchmark-split",
            "scaffold",
            "--neighbor-selector",
            "query_feature_coverage",
            "--neighbor-context-profile",
            "coverage_aware",
            "--output-root",
            "outputs/test-coverage-aware",
        ]
    )
    _validate_retrieval_ablation_args(isolated)

    no_floor_without_root = parse_starling_args(
        ["--benchmark-split", "scaffold", "--min-similarity", "0.0"]
    )
    with pytest.raises(SystemExit, match="explicit --output-root"):
        _validate_retrieval_ablation_args(no_floor_without_root)

    no_floor_k7 = parse_starling_args(
        [
            "--benchmark-split",
            "scaffold",
            "--top-k-per-group",
            "7",
            "--min-similarity",
            "0.0",
            "--output-root",
            "outputs/test-morgan-k7-no-floor",
        ]
    )
    _validate_retrieval_ablation_args(no_floor_k7)
    experiment = next(
        item
        for item in experiments_for_starling_benchmark(
            "scaffold", evaluation_subset="valid"
        )
        if item.name == "bbb_martins__starling_full_mechanism"
    )
    no_floor_k7.paper_root = no_floor_k7.output_root
    no_floor_k7.split = "valid"
    no_floor_k7.fresh_parent_disjoint = True
    command = _command(experiment, no_floor_k7)
    assert command[command.index("--top-k-per-group") + 1] == "7"
    assert command[command.index("--min-similarity") + 1] == "0.0"
    assert command[command.index("--tool-service-url") + 1] == "http://127.0.0.1:8766"

    blind_mmp_ledger = parse_starling_args(
        [
            "--benchmark-split",
            "scaffold",
            "--neighbor-selector",
            "query_feature_coverage",
            "--neighbor-context-profile",
            "coverage_mmp_ledger",
            "--output-root",
            "outputs/test-coverage-mmp-ledger",
        ]
    )
    with pytest.raises(SystemExit, match="visible-only"):
        _validate_retrieval_ablation_args(blind_mmp_ledger)

    visible_mmp_ledger = parse_starling_args(
        [
            "--benchmark-split",
            "scaffold",
            "--visibility-mode",
            "deployment_visible",
            "--neighbor-selector",
            "query_feature_coverage",
            "--neighbor-context-profile",
            "coverage_mmp_ledger",
            "--output-root",
            "outputs/test-coverage-mmp-ledger",
        ]
    )
    _validate_retrieval_ablation_args(visible_mmp_ledger)


def test_starling_matrix_enforces_single_endpoint_concurrency_budget():
    from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
        _parse_args as parse_starling_args,
    )

    accepted = parse_starling_args(
        ["--benchmark-split", "random", "--parallelism", "256"]
    )
    _validate_concurrency(accepted)

    rejected = parse_starling_args(
        ["--benchmark-split", "random", "--parallelism", "513"]
    )
    try:
        _validate_concurrency(rejected)
    except SystemExit as error:
        assert "exceeds the configured endpoint budget" in str(error)
    else:
        raise AssertionError("Expected an over-budget launcher shape to be rejected")

    deepseek = parse_starling_args(
        [
            "--benchmark-split",
            "random",
            "--parallelism",
            "2048",
            "--endpoint-concurrency-budget",
            "2048",
        ]
    )
    _validate_concurrency(deepseek)


def test_starling_valid_root_is_isolated_from_formal_test_root(tmp_path):
    canonical = tmp_path / "starling_random_v4"
    model_root = tmp_path / "starling_random_v4_valid_gpt_oss_20b"

    assert _paper_root_for_evaluation_subset(canonical, "test") == canonical
    assert _paper_root_for_evaluation_subset(canonical, "valid") == tmp_path / "starling_random_v4_valid"
    assert _paper_root_for_evaluation_subset(
        canonical,
        "valid",
        output_root=model_root,
    ) == model_root


def test_fresh_identity_blind_parent_disjoint_keeps_none_and_avoids_group_reuse(tmp_path):
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_LOCAL_API_KEY",
        parallelism=512,
        visibility_mode=IDENTITY_BLIND,
        neighbor_identity_policy=PARENT_DISJOINT,
        fresh_parent_disjoint=True,
        paper_root=str(tmp_path),
        split="valid",
        limit=0,
        experiments=[],
    )
    selected = _prepare_policy_selection(list(EXPERIMENTS), args)
    assert any(item.mode == "none" for item in selected)

    command = _command(EXPERIMENTS[1], args)
    batch_root = command[command.index("--batch-root") + 1]
    single_root = command[command.index("--single-analysis-source-batch") + 1]
    assert "runs_identity_blind_parent_disjoint" in batch_root
    assert "runs_identity_blind_parent_disjoint" in single_root
    assert "--group-analysis-source-batch" not in command
    assert "--identity-blind" in command
    assert experiment_run_root(
        IDENTITY_BLIND,
        PARENT_DISJOINT,
        paper_root=tmp_path,
    ).name == "runs_identity_blind_parent_disjoint"


def test_fresh_deployment_visible_parent_disjoint_keeps_none_and_avoids_reuse(tmp_path):
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GPT_OSS_LOCAL_API_KEY",
        parallelism=256,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy=PARENT_DISJOINT,
        fresh_parent_disjoint=True,
        paper_root=str(tmp_path),
        split="valid",
        timeout_s=600,
        limit=0,
        experiments=[],
    )
    selected = _prepare_policy_selection(list(EXPERIMENTS), args)
    assert any(item.mode == "none" for item in selected)

    command = _command(EXPERIMENTS[1], args)
    batch_root = command[command.index("--batch-root") + 1]
    single_root = command[command.index("--single-analysis-source-batch") + 1]
    assert "runs_deployment_visible_parent_disjoint" in batch_root
    assert "runs_deployment_visible_parent_disjoint" in single_root
    assert "--group-analysis-source-batch" not in command
    assert "--identity-blind" not in command
    assert command[command.index("--timeout-s") + 1] == "600"


def test_scaffold_disjoint_always_runs_fresh_in_an_isolated_root(tmp_path):
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GPT_OSS_LOCAL_API_KEY",
        parallelism=128,
        visibility_mode=IDENTITY_BLIND,
        neighbor_identity_policy=SCAFFOLD_DISJOINT,
        fresh_parent_disjoint=False,
        paper_root=str(tmp_path),
        split="valid",
        timeout_s=600,
        limit=0,
        experiments=[],
    )

    selected = _prepare_policy_selection(list(EXPERIMENTS), args)
    command = _command(EXPERIMENTS[1], args)

    assert any(item.mode == "none" for item in selected)
    assert "runs_identity_blind_scaffold_disjoint" in command[
        command.index("--batch-root") + 1
    ]
    assert "runs_identity_blind_scaffold_disjoint" in command[
        command.index("--single-analysis-source-batch") + 1
    ]
    assert "--group-analysis-source-batch" not in command
    assert experiment_run_root(
        IDENTITY_BLIND,
        SCAFFOLD_DISJOINT,
        paper_root=tmp_path,
    ).name == "runs_identity_blind_scaffold_disjoint"


def test_runner_defaults_to_parent_disjoint_primary_and_excludes_none():
    args = _parse_args([])

    assert args.visibility_mode == DEPLOYMENT_VISIBLE
    assert args.neighbor_identity_policy == PARENT_DISJOINT
    assert args.base_url is None
    assert args.model == "nvidia/GLM-5.2-NVFP4"
    assert args.reasoning_effort == ""
    assert args.api_key_env == "LITELLM_API_KEY"
    selected = _prepare_policy_selection(list(EXPERIMENTS), args)
    assert selected
    assert all(experiment.mode != "none" for experiment in selected)


def test_litellm_endpoint_and_key_resolve_from_local_keys(monkeypatch):
    settings = SimpleNamespace(
        LITELLM_BASE_URL="https://litellm.example.test/v1/",
        LITELLM_API_KEY="test-only-secret",
    )
    monkeypatch.setattr(agent_runner, "_load_local_keys", lambda: settings)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    base_url = agent_runner.resolve_endpoint_base_url(None)
    agent_runner.ensure_endpoint_api_key("LITELLM_API_KEY", base_url)

    assert base_url == "https://litellm.example.test/v1"
    assert os.environ["LITELLM_API_KEY"] == "test-only-secret"


def test_explicit_endpoint_and_exported_key_take_precedence(monkeypatch):
    monkeypatch.setattr(
        agent_runner,
        "_load_local_keys",
        lambda: pytest.fail("keys.py should not be loaded for explicit settings"),
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "already-exported")

    base_url = agent_runner.resolve_endpoint_base_url("https://custom.example.test/v1/")
    agent_runner.ensure_endpoint_api_key("CUSTOM_API_KEY", base_url)

    assert base_url == "https://custom.example.test/v1"
    assert os.environ["CUSTOM_API_KEY"] == "already-exported"


def test_missing_local_litellm_settings_fail_without_secret_values(monkeypatch):
    monkeypatch.setattr(agent_runner, "_load_local_keys", lambda: SimpleNamespace())
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="LITELLM_BASE_URL"):
        agent_runner.resolve_endpoint_base_url(None)
    with pytest.raises(SystemExit, match="LITELLM_API_KEY"):
        agent_runner.ensure_endpoint_api_key(
            "LITELLM_API_KEY",
            "https://litellm.example.test/v1",
        )


def test_missing_local_keys_module_has_actionable_error(monkeypatch):
    def missing_keys(_name):
        raise ModuleNotFoundError("No module named 'keys'", name="keys")

    monkeypatch.setattr(agent_runner.importlib, "import_module", missing_keys)

    with pytest.raises(SystemExit, match="Missing ignored keys.py"):
        agent_runner.resolve_endpoint_base_url(None)


def test_loopback_endpoint_retains_non_secret_placeholder(monkeypatch):
    monkeypatch.delenv("GLM_LOCAL_API_KEY", raising=False)
    agent_runner.ensure_endpoint_api_key(
        "GLM_LOCAL_API_KEY",
        "http://127.0.0.1:50000/v1",
    )

    assert os.environ["GLM_LOCAL_API_KEY"] == "local"


def test_explicit_parent_disjoint_none_is_rejected():
    args = _parse_args(["--experiments", "bbb_martins__none"])

    try:
        _prepare_policy_selection([EXPERIMENTS[0]], args)
    except SystemExit as error:
        assert "Query-only conditions" in str(error)
    else:
        raise AssertionError("Expected explicit parent-disjoint none selection to be rejected")


def test_starling_mechanism_views_match_each_binary_endpoint():
    expected = [
        "direct_brain_exposure",
        "passive_permeability",
        "efflux_transport",
        "influx_transport",
    ]
    assert [group.endpoint_group for group in BBB_STARLING.mechanism_groups] == expected

    expected_skin = [
        "direct_skin_reaction",
        "sensitisation_aop",
    ]
    assert [group.endpoint_group for group in SKIN_STARLING.mechanism_groups] == expected_skin
    assert [group.source_groups for group in SKIN_STARLING.mechanism_groups] == [
        ("Direct.skin_reaction",),
        ("Mechanism.sensitization_aop",),
    ]


def test_skin_paper_view_excludes_standalone_weak_context_branch():
    endpoint_groups = [group.endpoint_group for group in SKIN_CHEMBL.mechanism_groups]

    assert endpoint_groups == [
        "direct_skin_reaction",
        "sensitisation_aop",
        "phototoxicity_irritation_local_damage",
        "skin_exposure",
    ]
    assert "context_background" not in endpoint_groups


def test_bioavailability_starling_mechanism_groups_are_reiterable():
    expected = [
        "Observed.direct_oral_bioavailability",
        "Observed.nondirect_oral_bioavailability",
        "Observed.oral_auc_cmax_exposure",
        "Fa.absorption_solubility_permeability",
        "Fg.gut_wall_efflux_intestinal_metabolism",
        "Fh.hepatic_clearance_metabolic_stability",
    ]

    assert isinstance(BIO_STARLING.mechanism_groups, tuple)
    assert [group.group_id for group in BIO_STARLING.mechanism_groups] == expected
    assert [group.group_id for group in BIO_STARLING.mechanism_groups] == expected


def test_matrix_command_freezes_glm_and_identity_conditions():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode="identity_blind",
    )
    command = _command(EXPERIMENTS[0], args)

    assert "--identity-blind" in command
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--max-tokens") + 1] == "20480"
    assert command[command.index("--api-key-env") + 1] == "GLM_API_KEY"
    assert "--no-combine-traces" in command
    assert "--single-analysis-source-batch" not in command
    assert not any(part.startswith("sk-") for part in command)

    direct_command = _command(EXPERIMENTS[1], args)
    assert "--single-analysis-source-batch" in direct_command
    assert direct_command[direct_command.index("--single-analysis-source-batch") + 1].endswith(
        "bbb_martins__none"
    )


def test_matrix_command_forwards_retrieval_count_and_threshold():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GPT_OSS_LOCAL_API_KEY",
        parallelism=8,
        visibility_mode=IDENTITY_BLIND,
        neighbor_identity_policy=PARENT_DISJOINT,
        fresh_parent_disjoint=True,
        paper_root="outputs/test-minimol-top5",
        split="valid",
        top_k_per_group=5,
        min_similarity=0.0,
    )

    command = _command(EXPERIMENTS[1], args)

    assert command[command.index("--top-k-per-group") + 1] == "5"
    assert command[command.index("--min-similarity") + 1] == "0.0"


def test_valid_split_changes_only_dataset_and_isolates_output_root():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
        split="valid",
        paper_root="",
    )

    command = _command(EXPERIMENTS[1], args)
    valid_experiment = experiment_for_split(EXPERIMENTS[1], "valid")

    assert valid_experiment.input_jsonl.endswith("/valid.jsonl")
    assert command[command.index("--input-jsonl") + 1].endswith("/valid.jsonl")
    assert str(paper_root_for_split("valid")) in command[command.index("--batch-root") + 1]
    assert EXPERIMENTS[1].index == valid_experiment.index
    assert command[command.index("--model") + 1] == "nvidia/GLM-5.2-NVFP4"
    assert command[command.index("--base-url") + 1] == "https://litellm.parcc.upenn.edu/v1"
    assert command[command.index("--reasoning-effort") + 1] == ""
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--max-tokens") + 1] == "20480"


def test_deployment_visible_command_reuses_pipeline_without_blind_redaction():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
    )

    none_command = _command(EXPERIMENTS[0], args)
    direct_command = _command(EXPERIMENTS[1], args)

    assert "--identity-blind" not in none_command
    assert "runs_deployment_visible" in none_command[none_command.index("--batch-root") + 1]
    assert "--single-analysis-source-batch" in direct_command
    assert "runs_deployment_visible" in direct_command[
        direct_command.index("--single-analysis-source-batch") + 1
    ]


def test_parent_disjoint_command_uses_separate_root_and_operational_single_prior():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy=PARENT_DISJOINT,
    )

    command = _command(EXPERIMENTS[1], args)

    assert command[command.index("--neighbor-identity-policy") + 1] == "parent_disjoint"
    assert "runs_deployment_visible_parent_disjoint" in command[command.index("--batch-root") + 1]
    single_source = command[command.index("--single-analysis-source-batch") + 1]
    assert "runs_deployment_visible_parent_disjoint" not in single_source
    assert "runs_deployment_visible" in single_source
    group_source = command[command.index("--group-analysis-source-batch") + 1]
    assert group_source.endswith("bbb_martins__chembl_direct")


def test_parent_disjoint_command_can_reuse_external_single_without_crossing_group_features():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy=PARENT_DISJOINT,
        paper_root="/tmp/minimol-paper-root",
        single_analysis_root="/tmp/frozen-morgan-single-root",
        split="test",
    )

    command = _command(EXPERIMENTS[1], args)

    single_source = command[command.index("--single-analysis-source-batch") + 1]
    group_source = command[command.index("--group-analysis-source-batch") + 1]
    assert single_source.startswith("/tmp/frozen-morgan-single-root/")
    assert group_source.startswith("/tmp/minimol-paper-root/runs_deployment_visible/")


def test_operational_feature_ablation_reuses_only_hash_identical_morgan_groups():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy="operational",
        paper_root="/tmp/minimol-paper-root",
        single_analysis_root="/tmp/frozen-morgan-operational",
        group_analysis_root="/tmp/frozen-morgan-operational",
        split="test",
    )

    command = _command(EXPERIMENTS[1], args)

    single_source = command[command.index("--single-analysis-source-batch") + 1]
    group_source = command[command.index("--group-analysis-source-batch") + 1]
    assert single_source.startswith("/tmp/frozen-morgan-operational/")
    assert group_source.startswith("/tmp/frozen-morgan-operational/")


def test_matrix_command_passes_explicit_smoke_limit_only_when_requested():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=1,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy="operational",
        limit=1,
    )

    command = _command(EXPERIMENTS[0], args)

    assert command[command.index("--limit") + 1] == "1"


def test_matched_prefetch_command_is_visible_but_disables_agentic_tool_choice():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE_PREFETCHED,
    )

    command = _command(EXPERIMENTS[0], args)

    assert "--identity-blind" not in command
    assert "--harness-prefetch-tools" in command
    assert "runs_deployment_visible_prefetched" in command[command.index("--batch-root") + 1]


def test_visible_parent_disjoint_can_prefetch_tools_without_hiding_structures():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="OPENROUTER_API_KEY",
        parallelism=2,
        visibility_mode=DEPLOYMENT_VISIBLE,
        neighbor_identity_policy=PARENT_DISJOINT,
        fresh_parent_disjoint=True,
        fresh_disjoint=True,
        harness_prefetch_tools=True,
        paper_root="/tmp/visible-prefetch-paper-root",
        split="test",
    )

    command = _command(EXPERIMENTS[0], args)

    assert "--identity-blind" not in command
    assert "--harness-prefetch-tools" in command
    assert "runs_deployment_visible_parent_disjoint" in command[
        command.index("--batch-root") + 1
    ]


def test_deployment_prompt_hides_query_name_but_preserves_structures_and_neighbor_name():
    query = {
        "input_smiles": "CCO",
        "canonical_smiles": "CCO",
        "molecule_name": "QueryNameMustNotAppear",
    }
    group = {
        "group_id": "Observed.direct_oral_bioavailability",
        "tier": "Observed",
        "endpoint_group": "direct_oral_bioavailability",
        "transfer_neighbor_selection": {"reranker": "assay_transfer", "raw_pool_size": 100},
        "neighbors": [
            {
                "rank": 1,
                "structural_rank": 8,
                "transfer_selection_rank": 1,
                "transfer_selection_score": 0.987,
                "transfer_winning_record_id": "audit-only-record",
                "molecule_chembl_id": "STARLING_1",
                "canonical_smiles": "CCN",
                "similarity": 0.8,
                "similarity_bucket": "close_analog",
                "evidence_rows": [
                    {
                        "molecule_chembl_id": "STARLING_1",
                        "canonical_smiles": "CCN",
                        "source_molecule_names": ["VisibleNeighbor"],
                        "group_id": "Observed.direct_oral_bioavailability",
                        "standard_type": "Oral bioavailability",
                        "standard_value": 50,
                        "standard_units": "%",
                    }
                ],
            }
        ],
    }

    query_payload = _llm_query_payload(query)
    group_payload = _group_prompt_payload(query_payload, group)
    serialized = str(group_payload)

    assert query_payload == {"input_smiles": "CCO", "canonical_smiles": "CCO"}
    assert "QueryNameMustNotAppear" not in serialized
    assert "CCO" in serialized
    assert "CCN" in serialized
    assert "VisibleNeighbor" in serialized
    assert "transfer_selection" not in serialized
    assert "structural_rank" not in serialized
    assert "audit-only-record" not in serialized


def test_visible_prefetched_query_payload_preserves_structure_and_prefetched_properties():
    query = {
        "input_smiles": "CCO",
        "canonical_smiles": "CCO",
        "tools_prefetched": True,
        "prefetched_molecule_properties": {"tool_name": "molecule_properties", "status": "ok"},
    }

    payload = _llm_query_payload(query)

    assert payload["canonical_smiles"] == "CCO"
    assert payload["tools_prefetched"] is True
    assert payload["prefetched_molecule_properties"]["status"] == "ok"
