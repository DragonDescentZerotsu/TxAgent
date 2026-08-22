import json

from tools.chembl_tool.common.retrieval_ablation import (
    changed_group_ids,
    load_reusable_group_outputs,
    materialize_reused_run,
    retrieval_prompt_hash,
)


def _retrieval(molecule_id="CHEMBL1", relation="structural_analog"):
    return {
        "query": {"input_smiles": "CCO", "canonical_smiles": "CCO", "audit": "ignored"},
        "experiment": {"neighbor_identity_policy": "operational"},
        "groups": [
            {
                "group_id": "Direct.outcome",
                "tier": "Direct",
                "endpoint_group": "outcome",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": molecule_id,
                        "canonical_smiles": "CCN",
                        "similarity": 0.8,
                        "similarity_bucket": "close_analog",
                        "molecule_relation": relation,
                        "evidence_rows": [
                            {
                                "molecule_chembl_id": molecule_id,
                                "canonical_smiles": "CCN",
                                "group_id": "Direct.outcome",
                                "standard_type": "outcome",
                                "standard_value": 1,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_hash_ignores_audit_policy_metadata_but_detects_neighbor_changes():
    baseline = _retrieval(relation="same_parent")
    same_prompt = _retrieval(relation="structural_analog")
    same_prompt["experiment"]["neighbor_identity_policy"] = "parent_disjoint"

    assert retrieval_prompt_hash(baseline) == retrieval_prompt_hash(same_prompt)
    assert retrieval_prompt_hash(baseline) != retrieval_prompt_hash(_retrieval("CHEMBL2"))


def test_hash_detects_external_condition_changes():
    baseline = _retrieval()
    conditioned = _retrieval()
    conditioned["query"]["external_condition"] = "Under a fasted state."
    assert retrieval_prompt_hash(baseline) != retrieval_prompt_hash(conditioned)


def test_changed_group_ids_is_branch_granular():
    assert changed_group_ids(_retrieval(), _retrieval("CHEMBL2")) == ["Direct.outcome"]


def test_materialized_reuse_keeps_source_immutable_and_records_provenance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "final_reasoning_output.json").write_text('{"status":"ok"}\n')
    (source / "manifest.json").write_text('{"neighbor_identity_policy":"operational"}\n')
    target = tmp_path / "target"

    materialize_reused_run(
        source,
        target,
        {
            "reused_from": str(source),
            "reuse_reason": "same_hash",
            "neighbor_identity_policy": "parent_disjoint",
        },
    )

    reuse = json.loads((target / "reuse.json").read_text())
    target_manifest = json.loads((target / "manifest.json").read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert reuse["reused_from"] == str(source)
    assert target_manifest["neighbor_identity_policy"] == "parent_disjoint"
    assert target_manifest["artifact_reuse"]["reuse_reason"] == "same_hash"
    assert source_manifest["neighbor_identity_policy"] == "operational"
    assert not (source / "reuse.json").exists()


def test_load_reusable_group_outputs_reuses_only_unchanged_independent_branches(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    baseline = _retrieval()
    second = json.loads(json.dumps(baseline["groups"][0]))
    second["group_id"] = "Mechanism.second"
    second["endpoint_group"] = "second"
    baseline["groups"].append(second)
    (source / "retrieval.json").write_text(json.dumps(baseline))
    (source / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Direct.outcome", "status": "ok"})
        + "\n"
        + json.dumps({"group_id": "Mechanism.second", "status": "ok"})
        + "\n"
    )
    target = json.loads(json.dumps(baseline))
    target["groups"][0]["neighbors"][0]["molecule_chembl_id"] = "CHEMBL_CHANGED"

    outputs = load_reusable_group_outputs(str(source), target)

    assert [output["group_id"] for output in outputs] == ["Mechanism.second"]
    assert outputs[0]["reuse_reason"] == "identical_llm_visible_group_input"


def test_group_reuse_requires_matching_neighbor_context_profile(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "retrieval.json").write_text(json.dumps(_retrieval()))
    (source / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Direct.outcome", "status": "ok"}) + "\n"
    )
    (source / "manifest.json").write_text(
        json.dumps({"neighbor_context_profile": "standard"})
    )

    outputs = load_reusable_group_outputs(
        str(source),
        _retrieval(),
        target_neighbor_context_profile="coverage_aware",
    )

    assert outputs == []
