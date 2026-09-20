import json

from jinja2 import Environment, StrictUndefined
import pandas as pd

from semantic_buckets import source_local_semantic_v4 as workflow


def _render(name, payload):
    path = workflow.ROOT / "semantic_buckets/prompts/source_local_semantic_v4" / f"{name}.jinja"
    return Environment(undefined=StrictUndefined).from_string(path.read_text()).render(
        payload=payload, purpose="semantic"
    )


def test_skin_policy_uses_only_approved_semantic_columns():
    selected = set().union(*map(set, workflow.REFINEMENT_COLUMNS.values()))
    assert selected == {
        "canonical_endpoint_concept",
        "canonical_assay_or_test",
        "canonical_measurement_scale_id",
        "canonical_aop_event",
        "canonical_assay_type",
        "canonical_study_design",
    }


def test_markdown_value_prompt_omits_counts_and_raw_payload():
    prompt = _render("merge_values", {
        "task": "Skin_Reaction", "level": "L2", "source_id": "sensitization_aop",
        "bucket_id": "sb_test", "column": "canonical_assay_type",
        "column_description": "assay type",
        "current_bucket_identity": {"canonical_endpoint_concept": ["sensitization"]},
        "values": [{"value_id": "v000000", "value": "llna", "record_count": 99}],
    })
    assert prompt.startswith("# Reconcile selected values")
    assert "`v000000`: `llna`" in prompt
    assert "record_count" not in prompt
    assert "compact_payload_json" not in prompt


def test_establish_requires_review_hash_and_materializes_record_map(tmp_path, monkeypatch):
    run = tmp_path / "semantic_run"
    run.mkdir()
    input_path = tmp_path / "record_relevance_map.parquet"
    pairs = [
        '["direct_skin_reaction","sensitization","free-text","llna","mouse","__unknown__"]',
        '["direct_skin_reaction","sensitization","free-text","patch test","human","__unknown__"]',
    ]
    atoms = [workflow.core._stable_id("atom", "L2", "direct_skin_reaction", pair) for pair in pairs]
    pd.DataFrame({
        "canonical_record_id": ["c1", "c2"], "source_row_uid": ["u1", "u2"],
        "level": ["L2", "L2"], "source_id": ["direct_skin_reaction"] * 2,
        "pair_bucket_key": pairs,
    }).to_parquet(input_path, index=False)
    candidate = run / "source_semantic_bucket_map.parquet"
    pd.DataFrame({
        "level": ["L2", "L2"], "source_id": ["direct_skin_reaction"] * 2,
        "source_semantic_bucket_id": ["sb1", "sb2"], "atom_id": atoms,
    }).to_parquet(candidate, index=False)
    pd.DataFrame({
        "level": ["L2", "L2"], "source_id": ["direct_skin_reaction"] * 2,
        "atom_id": atoms,
    }).to_parquet(run / "input_atoms.parquet", index=False)
    (run / "manifest.json").write_text(json.dumps({"status": "awaiting_agentic_review"}))
    review = run / "agentic_review.json"
    review.write_text(json.dumps({
        "version": f"{workflow.VERSION}.agentic_review.v1", "reviewer": "test",
        "candidate_map_sha256": workflow.file_sha256(candidate), "decision": "approve",
        "rationale": "test approval", "audit": {},
    }))
    release = tmp_path / "v10_test"
    monkeypatch.setattr(
        workflow, "configure", lambda _root: {
            "run": run, "input": input_path, "release": release,
        },
    )
    monkeypatch.setattr(workflow.core, "EXPECTED_ATOM_COUNT", 2)
    monkeypatch.setattr(workflow.core, "EXPECTED_RECORD_COUNT", 2)

    result = workflow.establish(tmp_path, review)

    assert result["status"] == "complete_reviewed"
    assert result["publication_status"] == "reviewed_unselected"
    assert pd.read_parquet(run / "semantic_bucket_map.parquet").semantic_bucket_id.tolist() == ["sb1", "sb2"]
    assert pd.read_parquet(run / "record_semantic_bucket_map.parquet").source_row_uid.tolist() == ["u1", "u2"]
    assert json.loads((run / "semantic_bucket_map_manifest.json").read_text())["status"] == "complete_reviewed"
