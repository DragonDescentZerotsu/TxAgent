import argparse

from tools.chembl_tool.paper_experiments.molecular_evidence_agent import (
    DEPLOYMENT_VISIBLE,
    DEPLOYMENT_VISIBLE_PREFETCHED,
    EXPERIMENTS,
    PARENT_DISJOINT,
    _command,
    experiment_for_split,
    paper_root_for_split,
)
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import (
    _group_prompt_payload,
    _llm_query_payload,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING as BIO_STARLING
from tools.chembl_tool.tasks.skin_reaction.experiment_config import CHEMBL as SKIN_CHEMBL


def test_frozen_matrix_has_unique_expected_conditions():
    names = [experiment.name for experiment in EXPERIMENTS]
    assert len(names) == 21
    assert len(names) == len(set(names))
    assert "bioavailability_ma__starling_full_mechanism" in names
    assert "bbb_martins__starling_direct" in names


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
