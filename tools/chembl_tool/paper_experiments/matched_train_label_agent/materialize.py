"""Materialize exact frozen-KNN neighbors as label-visible direct-agent retrievals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate, policy_metadata
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import similarity_bucket

from .contract import DEFAULT_OUTPUT_ROOT, SCHEMA_VERSION, TaskSpec, replay_batch, selected_specs


def materialize_task(spec: TaskSpec, *, output_root: str | Path) -> dict[str, Any]:
    labels = _read_jsonl(spec.input_jsonl)
    knn_rows = _read_jsonl(spec.knn_predictions)
    if len(labels) != len(knn_rows):
        raise ValueError(f"{spec.task} label/KNN row mismatch: {len(labels)} != {len(knn_rows)}")

    batch = replay_batch(output_root, spec)
    relation_counts: dict[str, int] = {}
    for query_index, (label_row, knn_row) in enumerate(zip(labels, knn_rows, strict=True)):
        _validate_alignment(spec, query_index, label_row, knn_row)
        query_smiles = str(label_row["drug"])
        query_identity = normalize_molecule_identity(query_smiles)
        neighbors = []
        for rank, source_neighbor in enumerate(knn_row["neighbors"], start=1):
            candidate_smiles = str(source_neighbor["drug"])
            decision = decide_candidate(
                query_identity,
                {"canonical_smiles": candidate_smiles},
                "parent_disjoint",
            )
            relation_counts[decision.relation.value] = relation_counts.get(decision.relation.value, 0) + 1
            if decision.excluded:
                raise ValueError(
                    f"{spec.task} query {query_index} KNN neighbor {rank} violates parent_disjoint: "
                    f"{decision.relation.value}"
                )
            neighbors.append(_neighbor(spec, source_neighbor, rank, decision.relation.value))

        retrieval = _retrieval(
            spec,
            query_smiles,
            neighbors,
            int(knn_row.get("n_eligible_neighbors") or len(knn_row.get("neighbors") or [])),
        )
        run_id = f"{batch.name}_idx{query_index:05d}"
        write_json_atomic(batch / "runs" / run_id / "retrieval.json", retrieval)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": spec.task,
        "evaluation_subset": "valid",
        "input_jsonl": str(spec.input_jsonl),
        "knn_predictions": str(spec.knn_predictions),
        "n_queries": len(labels),
        "k": 3,
        "retrieval_feature": spec.retrieval_feature,
        "retrieval_similarity": spec.retrieval_similarity,
        "retrieval_similarity_metric": spec.retrieval_similarity_metric,
        "neighbor_selector": "similarity.v1",
        "neighbor_identity_policy": "parent_disjoint",
        "neighbor_set_contract": spec.retrieval_neighbor_set_contract,
        "visible_neighbor_supervision": "frozen_train_Y_and_task_label_meaning",
        "external_starling_records_visible": False,
        "relation_counts": relation_counts,
        "batch": str(batch),
    }
    write_json_atomic(batch / "manifest.json", manifest)
    return manifest


def _neighbor(
    spec: TaskSpec,
    source: dict[str, Any],
    rank: int,
    relation: str,
) -> dict[str, Any]:
    train_index = int(source["train_index"])
    label = int(source["Y"])
    smiles = str(source["drug"])
    identity = normalize_molecule_identity(smiles)
    molecule_id = f"train_neighbor_{train_index:08d}"
    evidence = {
        "contract_version": "minimal_evidence.v1",
        "source": {"name": "frozen_benchmark_train_label", "record_id": str(train_index)},
        "molecule": {
            "id": molecule_id,
            "canonical_smiles": identity.canonical_smiles,
            "names": [],
        },
        "group": {
            "id": spec.source_group_id,
            "tier": spec.tier,
            "endpoint_group": spec.endpoint_group,
        },
        "endpoint": {
            "name": spec.endpoint_name,
            "measurement": {"relation": "=", "value": spec.label_text(label), "unit": ""},
        },
        "text": {
            "evidence": (
                "This molecule is a frozen scaffold-train example with observed benchmark outcome "
                f"{spec.label_text(label)}. Treat it as a supervised analog outcome and judge "
                "chemical transferability to the query; do not assume majority vote is always correct."
            ),
            "context": "benchmark split=train; label is hidden for the query and visible only for this neighbor",
        },
        "annotations": {
            "evidence_role": "direct_outcome",
            "scope": {"dataset_split": ["train"], "task": [spec.data_name]},
            "transferability": "not_assessed",
            "uncertainty": [],
        },
        "quality": {"confidence": "frozen_gold_label"},
        "provenance": {"assay_id": "", "source_record_count": 1},
        "examples": [],
    }
    return {
        "rank": rank,
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": identity.canonical_smiles,
        "standard_inchi_key": identity.standard_inchi_key,
        "similarity": round(float(source["similarity"]), 6),
        "similarity_bucket": similarity_bucket(float(source["similarity"])),
        "similarity_metric": spec.retrieval_similarity_metric,
        "molecule_relation": relation,
        "source_group_ids": [spec.source_group_id],
        "n_evidence_rows": 1,
        "evidence_rows": [
            {
                "evidence_source": "frozen_benchmark_train_label",
                "group_id": spec.source_group_id,
                "train_index": train_index,
                "train_label": label,
                "minimal_evidence": evidence,
            }
        ],
    }


def _retrieval(
    spec: TaskSpec,
    query_smiles: str,
    neighbors: list[dict[str, Any]],
    n_candidates: int,
) -> dict[str, Any]:
    identity = normalize_molecule_identity(query_smiles)
    return {
        "status": "ok",
        "evidence_source": {
            "type": "frozen_benchmark_train_labels",
            "dataset": spec.data_name,
            "split": "train",
        },
        "experiment": {
            "mode": "direct",
            "source": "matched_train_label_knn",
            "resolved_group_mapping": {spec.group_id: [spec.source_group_id]},
            "retrieval_feature": _retrieval_feature_metadata(spec),
            "matched_knn_neighbor_set": True,
            "train_labels_visible": True,
            "external_starling_records_visible": False,
            **policy_metadata("parent_disjoint"),
        },
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": identity.canonical_smiles,
            "standard_inchi_key": identity.standard_inchi_key,
            **_query_feature_metadata(spec),
        },
        "groups": [
            {
                "group_id": spec.group_id,
                "tier": spec.tier,
                "endpoint_group": spec.endpoint_group,
                "source_group_ids": [spec.source_group_id],
                "n_candidate_molecules": n_candidates,
                "neighbors": neighbors,
            }
        ],
        "coverage": {
            "n_groups": 1,
            "n_groups_with_neighbors": 1,
            "n_neighbors_total": len(neighbors),
            "min_similarity": None,
            "top_k_per_group": 3,
        },
    }


def _retrieval_feature_metadata(spec: TaskSpec) -> dict[str, Any]:
    if spec.retrieval_feature == "morgan_fingerprint":
        return {
            "feature": "morgan_fingerprint",
            "similarity": spec.retrieval_similarity,
            "radius": 2,
            "n_bits": 2048,
        }
    if spec.retrieval_feature == "minimol_embedding":
        return {
            "feature": "minimol_embedding",
            "model": "MiniMol",
            "model_version": "minimol_v1",
            "dimension": 512,
            "normalization": "L2",
            "similarity": spec.retrieval_similarity,
        }
    raise ValueError(f"Unsupported matched retrieval feature: {spec.retrieval_feature}")


def _query_feature_metadata(spec: TaskSpec) -> dict[str, Any]:
    feature = _retrieval_feature_metadata(spec)
    if spec.retrieval_feature == "morgan_fingerprint":
        return {"fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048}}
    return {"retrieval_embedding": feature}


def _validate_alignment(
    spec: TaskSpec,
    query_index: int,
    label_row: dict[str, Any],
    knn_row: dict[str, Any],
) -> None:
    if int(knn_row.get("query_index", -1)) != query_index:
        raise ValueError(f"{spec.task} KNN query index mismatch at {query_index}")
    if str(knn_row.get("drug") or "") != str(label_row.get("drug") or ""):
        raise ValueError(f"{spec.task} molecule mismatch at {query_index}")
    if int(knn_row.get("Y", -1)) != int(label_row.get("Y", -2)):
        raise ValueError(f"{spec.task} gold mismatch at {query_index}")
    if knn_row.get("status") != "ok" or len(knn_row.get("neighbors") or []) != 3:
        raise ValueError(f"{spec.task} query {query_index} does not have exact KNN top-3")
    expected = int(sum(int(row["Y"]) for row in knn_row["neighbors"]) >= 2)
    if int(knn_row.get("prediction", -1)) != expected:
        raise ValueError(f"{spec.task} query {query_index} KNN prediction is not majority vote")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args(argv)
    manifests = [materialize_task(spec, output_root=args.output_root) for spec in selected_specs(args.tasks)]
    write_json_atomic(
        args.output_root / "retrieval_materialization_manifest.json",
        {"schema_version": SCHEMA_VERSION, "tasks": manifests},
    )
    print(json.dumps(manifests, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
