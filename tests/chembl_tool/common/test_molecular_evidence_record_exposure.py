from __future__ import annotations

import json

from tools.chembl_tool.paper_experiments import (
    summarize_molecular_evidence_record_exposure as exposure,
)


def _evidence_row(group: str, count: int, n_examples: int, prefix: str):
    return {
        "molecule_chembl_id": f"MOL_{prefix}",
        "canonical_smiles": "CCO",
        "group_id": group,
        "endpoint_group": group.split(".", 1)[-1],
        "standard_type": "example endpoint",
        "evidence_source": "test source",
        "source_record_count": count,
        "source_record_examples": [
            {
                "source_id": "source",
                "source_index": index,
                "source_record_id": f"{prefix}-{index}",
                "support_text": f"record {index}",
            }
            for index in range(n_examples)
        ],
    }


def _neighbor(molecule_id: str, rows: list[dict]):
    return {
        "molecule_chembl_id": molecule_id,
        "evidence_rows": rows,
    }


def test_exposure_counts_visible_examples_per_row_and_detects_shortfalls(tmp_path):
    root = tmp_path / "condition"
    batch = (
        root
        / exposure.RUNS_DIRECTORY
        / "bioavailability_ma"
        / "bioavailability_ma__starling_full_mechanism"
    )
    run = batch / "runs" / "condition_idx00000"
    run.mkdir(parents=True)
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "batch_id": batch.name,
                "input_jsonl": "valid.jsonl",
                "n_items": 1,
                "experiment_mode": "full_mechanism",
                "retrieval_source": "starling",
                "retrieval_strategy": "morgan_fingerprint",
                "neighbor_identity_policy": "parent_disjoint",
                "top_k_per_group": 3,
                "min_similarity": 0.0,
            }
        )
    )
    direct = "Observed.direct_oral_bioavailability"
    mechanism = "Fa.absorption_solubility_permeability"
    retrieval = {
        "status": "ok",
        "experiment": {
            "mode": "full_mechanism",
            "source": "starling",
            "retrieval_feature": {"feature": "morgan_fingerprint"},
        },
        "coverage": {"top_k_per_group": 3, "min_similarity": 0.0},
        "groups": [
            {
                "group_id": direct,
                "neighbors": [
                    _neighbor(
                        "MOL_A",
                        [
                            _evidence_row("Observed.direct", 10, 8, "direct"),
                            _evidence_row("Observed.nondirect", 20, 7, "nondirect"),
                        ],
                    ),
                    _neighbor("MOL_B", [_evidence_row(direct, 2, 2, "small")]),
                ],
            },
            {
                "group_id": mechanism,
                "neighbors": [
                    _neighbor("MOL_A", [_evidence_row(mechanism, 4, 4, "mech-a")]),
                    _neighbor("MOL_A", [_evidence_row(mechanism, 1, 1, "mech-b")]),
                ],
            },
        ],
    }
    (run / "retrieval.json").write_text(json.dumps(retrieval))

    summary = exposure.summarize_condition("k3", root)
    task = summary["tasks"]["bioavailability_ma"]
    direct_row = task["families"][direct]
    overall = task["overall"]

    assert direct_row["visible_records_per_molecule_slot"] == {
        "n": 2,
        "mean": 7.0,
        "median": 7.0,
        "p95": 12,
        "max": 12,
    }
    assert direct_row["underlying_records_per_molecule_slot"]["mean"] == 16.0
    assert direct_row["fraction_evidence_rows_at_six_record_cap"] == 2 / 3
    assert overall["short_family_prompts"] == 2
    assert overall["duplicate_molecule_slots_within_family"] == 1
    assert overall["visible_records_per_query"]["mean"] == 19.0
    assert overall["unique_molecules_per_query"]["mean"] == 2.0

    output = tmp_path / "analysis"
    assert exposure.main(
        ["--condition", f"k3={root}", "--output-dir", str(output)]
    ) == 0
    assert (output / "record_exposure_summary.json").exists()
    assert (output / "record_exposure.tsv").exists()
    assert (output / "report.md").exists()


def test_empty_condition_fails_closed(tmp_path):
    root = tmp_path / "empty"
    run_root = root / exposure.RUNS_DIRECTORY
    run_root.mkdir(parents=True)
    output = tmp_path / "analysis"

    try:
        exposure.main(["--condition", f"k3={root}", "--output-dir", str(output)])
    except SystemExit as exc:
        assert "No Morgan full-mechanism batches" in str(exc)
    else:
        raise AssertionError("Expected an empty condition to fail closed")
