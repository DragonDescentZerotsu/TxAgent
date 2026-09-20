"""Build the offline Morgan top-10 cache for TDC L1 evaluation.

Each TDC validation/test query is ranked against one card-level union of frozen
Gold-v1 training contexts and TDC training labels.  The cache stores no model
scores and excludes exact-parent and nonempty-scaffold overlap with the query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from data.processing.gold_labels.conditioned_benchmark import task_root, tdc_task_root
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


PROFILE = "tdc_mixed_l1_v1"
SCHEMA = "tdc_mixed_l1_morgan.v1"
ROOT = Path(__file__).resolve().parents[3]
TASKS = ("bbb_martins", "bioavailability_ma")
SUBSETS = ("valid", "test")
TOP_K = 10


def _display_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _fingerprint(smiles: str, generator: Any) -> Any:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid published parent SMILES: {smiles}")
    return generator.GetFingerprint(molecule)


def _candidates(task: str) -> list[dict[str, Any]]:
    sources = (
        ("gold_v1", task_root(task, "v1") / "train_molecule_condition_labels.jsonl"),
        ("tdc_v1", tdc_task_root(task) / "train_molecule_condition_labels.jsonl"),
    )
    rows = []
    for source_kind, path in sources:
        for row in read_jsonl(path):
            rows.append({
                "source_kind": source_kind,
                "benchmark_row_id": str(row["benchmark_row_id"]),
                "drug": str(row["drug"]),
                "molecule_identity_key": str(row["molecule_identity_key"]),
                "bemis_murcko_scaffold": str(row.get("bemis_murcko_scaffold") or ""),
                "condition_group": str(row.get("condition_group") or ""),
                "Y": int(row["Y"]),
                "label_counts": dict(row.get("label_counts") or {str(row["Y"]): 1}),
                "source_record_ids": list(row.get("source_record_ids") or []),
            })
    return rows


def _rank_query(query: dict[str, Any], candidates: list[dict[str, Any]], fingerprints: list[Any], generator: Any) -> list[dict[str, Any]]:
    query_fingerprint = _fingerprint(str(query["drug"]), generator)
    similarities = DataStructs.BulkTanimotoSimilarity(query_fingerprint, fingerprints)
    query_parent = str(query["molecule_identity_key"])
    query_scaffold = str(query.get("bemis_murcko_scaffold") or "")
    ranked = []
    for candidate, similarity in zip(candidates, similarities):
        if candidate["molecule_identity_key"] == query_parent:
            continue
        if query_scaffold and candidate["bemis_murcko_scaffold"] == query_scaffold:
            continue
        ranked.append({**candidate, "morgan_similarity": round(float(similarity), 8)})
    ranked.sort(key=lambda row: (
        -row["morgan_similarity"],
        0 if row["source_kind"] == "gold_v1" else 1,
        row["benchmark_row_id"],
    ))
    if len(ranked) < TOP_K:
        raise ValueError(f"TDC query has only {len(ranked)} disjoint L1 cards")
    return ranked[:TOP_K]


def build_task(task: str, output_root: Path) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    candidates = _candidates(task)
    fingerprints = [_fingerprint(row["drug"], generator) for row in candidates]
    task_root_path = output_root / task
    source_paths = {
        "gold_train": task_root(task, "v1") / "train_molecule_condition_labels.jsonl",
        "tdc_train": tdc_task_root(task) / "train_molecule_condition_labels.jsonl",
    }
    manifests = {}
    for subset in SUBSETS:
        query_path = tdc_task_root(task) / f"{subset}.jsonl"
        rows = [{
            "benchmark_row_id": str(query["benchmark_row_id"]),
            "query_drug": str(query["drug"]),
            "query_molecule_identity_key": str(query["molecule_identity_key"]),
            "query_scaffold": str(query.get("bemis_murcko_scaffold") or ""),
            "cards": _rank_query(query, candidates, fingerprints, generator),
        } for query in read_jsonl(query_path)]
        output = task_root_path / f"{subset}.jsonl"
        write_jsonl_atomic(output, rows)
        manifests[subset] = {
            "queries": len(rows),
            "cache": output.name,
            "cache_sha256": sha256_file(output),
            "query_input": _display_path(query_path),
            "query_input_sha256": sha256_file(query_path),
        }
    manifest = {
        "schema_version": SCHEMA,
        "status": "complete",
        "profile": PROFILE,
        "task": task,
        "ranking": "morgan",
        "selection_unit": "source_specific_label_card",
        "top_k": TOP_K,
        "neighbor_identity_policy": "parent_and_nonempty_scaffold_disjoint",
        "candidate_sources": {
            name: {"path": _display_path(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "candidate_cards": len(candidates),
        "subsets": manifests,
    }
    write_json_atomic(task_root_path / "manifest.json", manifest)
    return manifest


def main() -> int:
    output = cache_profile_root(PROFILE)
    results = {task: build_task(task, output) for task in TASKS}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
