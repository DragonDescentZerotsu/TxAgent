import json

from tools.chembl_tool.paper_experiments.carry_forward_assay_prefix import (
    carry_forward_run,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _retrieval(prefix, endpoint="same"):
    return {
        "status": "ok",
        "query": {"input_smiles": "CC", "canonical_smiles": "CC"},
        "coverage": {"n_selected_assays": prefix, "n_neighbors_total": 1},
        "experiment": {"mode": "assay_flat", "assay_prefix": prefix},
        "groups": [
            {
                "group_id": "Flat.assay_ranked_evidence",
                "tier": "Flat",
                "endpoint_group": "assay_ranked_evidence",
                "neighbors": [
                    {
                        "rank": 1,
                        "molecule_chembl_id": "M1",
                        "canonical_smiles": "CCC",
                        "similarity": 0.8,
                        "similarity_bucket": "high",
                        "evidence_rows": [{"standard_type": endpoint}],
                    }
                ],
            }
        ],
    }


def _source_run(path):
    path.mkdir()
    _write_json(path / "retrieval.json", _retrieval(5))
    _write_json(path / "single_molecule_reasoning_output.json", {"status": "ok"})
    (path / "group_reasoning_outputs.jsonl").write_text(
        json.dumps({"group_id": "Flat.assay_ranked_evidence", "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        path / "final_reasoning_output.json",
        {"status": "ok", "llm": {"content": {"prediction": "positive"}}},
    )


def test_carry_forward_ignores_selected_assay_count_when_evidence_is_identical(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _source_run(source)
    target.mkdir()
    _write_json(target / "retrieval.json", _retrieval(20))
    _write_json(target / "manifest.json", {"stage_pool": {"events": []}})

    assert carry_forward_run(source, target)
    reuse = _read(target / "reuse.json")
    assert reuse["target_assay_prefix"] == 20
    assert reuse["final_prompt_byte_identical"] is False
    assert reuse["changed_final_coverage_fields_ignored_by_policy"] == [
        "n_selected_assays"
    ]
    assert _read(target / "final_reasoning_output.json")["status"] == "ok"


def test_carry_forward_rejects_changed_visible_evidence(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _source_run(source)
    target.mkdir()
    _write_json(target / "retrieval.json", _retrieval(20, endpoint="changed"))
    _write_json(target / "manifest.json", {})

    assert not carry_forward_run(source, target)
    assert not (target / "final_reasoning_output.json").exists()


def test_carry_forward_does_not_overwrite_target_reasoning(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _source_run(source)
    target.mkdir()
    _write_json(target / "retrieval.json", _retrieval(20))
    _write_json(target / "manifest.json", {})
    (target / "group_reasoning_outputs.jsonl").write_text(
        json.dumps(
            {
                "group_id": "Flat.assay_ranked_evidence",
                "status": "ok",
                "analysis": "fresh target output",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert not carry_forward_run(source, target)
    assert "fresh target output" in (
        target / "group_reasoning_outputs.jsonl"
    ).read_text(encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))
