import json
from types import SimpleNamespace

from tools.chembl_tool.paper_experiments.parent_disjoint_ablation import (
    build_experiment_plan,
    _same_parent_exposure_row,
    _same_parent_stats,
)


def test_same_parent_stats_count_llm_visible_slots_and_query_local_unique_neighbors():
    salt = {
        "molecule_chembl_id": "SALT1",
        "canonical_smiles": "CC[NH3+].[Cl-]",
    }
    retrieval = {
        "groups": [
            {
                "group_id": "Mechanism.one",
                "neighbors": [salt, {"molecule_chembl_id": "ANALOG", "canonical_smiles": "CCO"}],
            },
            {
                "group_id": "Mechanism.two",
                "neighbors": [
                    {"molecule_chembl_id": "ANALOG2", "canonical_smiles": "CCCN"},
                    dict(salt),
                ],
            },
        ]
    }

    stats = _same_parent_stats("CCN", retrieval)

    assert stats == {
        "group_ids": ["Mechanism.one", "Mechanism.two"],
        "n_retrieved_neighbor_slots": 4,
        "n_same_parent_neighbor_slots": 2,
        "n_same_parent_unique_neighbors": 1,
        "n_same_parent_rank1_slots": 1,
    }


def test_same_parent_exposure_row_is_a_stable_condition_level_schema():
    row = _same_parent_exposure_row(
        {
            "experiment": "task__source_full_flat",
            "task": "task",
            "source": "source",
            "mode": "full_flat",
            "n_total": 10,
            "n_queries_with_same_parent": 2,
            "n_groups_with_same_parent": 2,
            "n_retrieved_neighbor_slots": 30,
            "n_same_parent_neighbor_slots": 3,
            "same_parent_neighbor_slot_fraction": 0.1,
            "n_same_parent_unique_neighbors_summed_per_query": 2,
            "n_same_parent_rank1_slots": 2,
            "n_changed": 2,
            "n_reused": 8,
            "changed_indices": [1, 4],
        }
    )

    assert row["n_same_parent_neighbor_slots"] == 3
    assert row["same_parent_neighbor_slot_fraction"] == 0.1
    assert "changed_indices" not in row


def test_plan_does_not_load_neighbor_index_without_same_parent_exposure(tmp_path):
    experiment = SimpleNamespace(
        name="task__chembl_direct",
        task="task",
        source="chembl",
        mode="direct",
        input_jsonl=str(tmp_path / "input.jsonl"),
        index=str(tmp_path / "missing-index.pkl"),
    )
    operational_root = tmp_path / "operational"
    source_batch = operational_root / experiment.task / experiment.name
    source_run = source_batch / "runs" / f"{experiment.name}_idx00000"
    source_run.mkdir(parents=True)
    (source_batch / "manifest.json").write_text(
        json.dumps({"top_k_per_group": 3, "min_similarity": 0.3}),
        encoding="utf-8",
    )
    (tmp_path / "input.jsonl").write_text(
        json.dumps({"drug": "CCO", "Y": 0}) + "\n",
        encoding="utf-8",
    )
    (source_run / "retrieval.json").write_text(
        json.dumps(
            {
                "query": {"input_smiles": "CCO", "canonical_smiles": "CCO"},
                "groups": [
                    {
                        "group_id": "Direct.one",
                        "tier": "Direct",
                        "endpoint_group": "one",
                        "neighbors": [
                            {
                                "rank": 1,
                                "molecule_chembl_id": "ANALOG",
                                "canonical_smiles": "CCCN",
                                "similarity": 0.4,
                                "similarity_bucket": "weak_analog",
                                "evidence_rows": [],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows, summary = build_experiment_plan(
        experiment,
        operational_root=operational_root,
        target_root=tmp_path / "target",
        materialize=False,
    )

    assert rows[0]["changed"] is False
    assert summary["n_changed"] == 0
