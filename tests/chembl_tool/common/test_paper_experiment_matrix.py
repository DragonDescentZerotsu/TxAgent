import argparse
import json

from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    DEPLOYMENT_VISIBLE_PREFETCHED,
    EXPERIMENTS,
    PARENT_DISJOINT,
    _command,
    _parse_args,
    _prepare_policy_selection,
    experiment_for_split,
    paper_root_for_split,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    _benchmark_provenance,
    _matrix_manifest_path,
    _write_json_atomic,
    experiments_for_starling_benchmark,
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


def test_frozen_matrix_has_unique_expected_conditions():
    names = [experiment.name for experiment in EXPERIMENTS]
    assert len(names) == 26
    assert len(names) == len(set(names))
    assert "bioavailability_ma__starling_full_mechanism" in names
    assert "bbb_martins__starling_direct" in names
    assert "bbb_martins__starling_full_flat" in names
    assert "bbb_martins__starling_full_mechanism" in names
    assert "skin_reaction__starling_direct" in names
    assert "skin_reaction__starling_full_flat" in names
    assert "skin_reaction__starling_full_mechanism" in names


def test_starling_benchmark_matrix_reuses_conditions_but_replaces_inputs_and_indices():
    experiments = experiments_for_starling_benchmark("random")
    assert len(experiments) == 22
    assert {experiment.task for experiment in experiments} == {
        "bbb_martins",
        "bioavailability_ma",
        "skin_reaction",
    }
    assert all("/random/test.jsonl" in experiment.input_jsonl for experiment in experiments)

    chembl = next(item for item in experiments if item.name == "bbb_martins__chembl_direct")
    starling = next(item for item in experiments if item.name == "bbb_martins__starling_direct")
    assert chembl.index == EXPERIMENTS[1].index
    assert "molecular_evidence_agent_starling_random/evidence" in starling.index
    assert starling.index.endswith("bbb_starling_direct/starling_bbb_neighbor_index.pkl")


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
                "source_metadata": {"revision": "abc"},
            }
        )
    )
    (split_dir / "summary.json").write_text(
        json.dumps({"train_test_identity_overlap": 0})
    )
    (split_dir / "test.jsonl").write_text('{"drug":"CCO","Y":1}\n')
    (split_dir / "test_molecule_labels.jsonl").write_text(
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
    assert bbb["split_summary"]["train_test_identity_overlap"] == 0
    assert len(bbb["test_jsonl_sha256"]) == 64
    assert len(bbb["test_molecule_labels_sha256"]) == 64


def test_starling_matrix_accepts_manifest_only_mode():
    from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
        _parse_args as parse_starling_args,
    )

    args = parse_starling_args(
        ["--benchmark-split", "random", "--manifest-only"]
    )

    assert args.manifest_only is True


def test_runner_defaults_to_parent_disjoint_primary_and_excludes_none():
    args = _parse_args([])

    assert args.visibility_mode == DEPLOYMENT_VISIBLE
    assert args.neighbor_identity_policy == PARENT_DISJOINT
    selected = _prepare_policy_selection(list(EXPERIMENTS), args)
    assert selected
    assert all(experiment.mode != "none" for experiment in selected)


def test_explicit_parent_disjoint_none_is_rejected():
    args = _parse_args(["--experiments", "bbb_martins__none"])

    try:
        _prepare_policy_selection([EXPERIMENTS[0]], args)
    except SystemExit as error:
        assert "Query-only conditions" in str(error)
    else:
        raise AssertionError("Expected explicit parent-disjoint none selection to be rejected")


def test_new_starling_sources_match_the_four_paper_mechanism_families():
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
        "phototoxicity_irritation_local_damage",
        "skin_exposure",
    ]
    assert [group.endpoint_group for group in SKIN_STARLING.mechanism_groups] == expected_skin


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
        group_workers=3,
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


def test_valid_split_changes_only_dataset_and_isolates_output_root():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        group_workers=3,
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
    assert command[command.index("--model") + 1] == "zai-org/GLM-5.2-FP8"
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--max-tokens") + 1] == "20480"


def test_deployment_visible_command_reuses_pipeline_without_blind_redaction():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        group_workers=3,
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
        group_workers=3,
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


def test_matched_prefetch_command_is_visible_but_disables_agentic_tool_choice():
    args = argparse.Namespace(
        python_executable="python",
        api_key_env="GLM_API_KEY",
        parallelism=2,
        group_workers=3,
        visibility_mode=DEPLOYMENT_VISIBLE_PREFETCHED,
    )

    command = _command(EXPERIMENTS[0], args)

    assert "--identity-blind" not in command
    assert "--harness-prefetch-tools" in command
    assert "runs_deployment_visible_prefetched" in command[command.index("--batch-root") + 1]


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
        "neighbors": [
            {
                "rank": 1,
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
