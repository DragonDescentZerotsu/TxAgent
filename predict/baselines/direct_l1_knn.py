"""Compare Morgan and direct assay-transfer KNN on frozen L1 cards."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score

from data.processing.gold_labels.conditioned_benchmark import tdc_task_root
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.utils.json import atomic_output_path, read_jsonl, sha256_file, write_json_atomic


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
SPLITS = ("valid", "test")
METHODS = ("morgan", "assay_transfer")
EXPECTED_QUERIES = {
    "gold_v1": {
        "bbb_martins": {"valid": 397, "test": 393},
        "bioavailability_ma": {"valid": 262, "test": 269},
        "skin_reaction": {"valid": 239, "test": 241},
    },
    "tdc_v1": {
        "bbb_martins": {"valid": 197, "test": 530},
        "bioavailability_ma": {"valid": 64, "test": 128},
        "skin_reaction": {"valid": 40, "test": 82},
    },
}
GOLD_PROFILES = {
    "bbb_martins": "v10_3_best_scaffold_morgan100_v1",
    "bioavailability_ma": "v10_3_best_scaffold_morgan100_v1",
    "skin_reaction": "v9_skin_gold_v1_scaffold_morgan100_v1",
}
TDC_PROFILES = {
    "bbb_martins": "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
    "bioavailability_ma": "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
    "skin_reaction": "ranked_level_retrieval_tdc_v1_assay_skin_v9_v1",
}


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def _vote(labels: Iterable[int]) -> tuple[int, float]:
    values = list(labels)
    score = sum(values) / len(values)
    return int(score >= 0.5), score


def _gold_neighbors(task: str, split: str, method: str, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = cache_profile_root(GOLD_PROFILES[task]) / task / "scaffold" / split
    version_path, rankings_path = root / "VERSION.json", root / "rankings.parquet"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if (version.get("status") != "complete"
            or (version.get("inputs") or {}).get("neighbor_identity_policy") != "scaffold_disjoint"
            or version.get("rankings_sha256") != sha256_file(rankings_path)):
        raise ValueError(f"Incompatible Gold L1 cache: {root}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(rankings_path).to_pylist():
        grouped[str(row["query_record_id"])].append(row)
    output = []
    for query_id, rows in sorted(grouped.items()):
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_parent[str(row["retrieval_molecule_identity_key"])].append(row)
        if len(by_parent) != 100:
            raise ValueError(f"{task}/{split}/{query_id} has {len(by_parent)} Gold L1 parents")
        if method == "morgan":
            cards = [min(values, key=lambda row: (int(row["retrieval_parent_context_index"]), str(row["retrieval_record_id"]))) for values in by_parent.values()]
            cards.sort(key=lambda row: (int(row["retrieval_parent_rank"]), str(row["retrieval_molecule_identity_key"])))
        else:
            cards = [max(values, key=lambda row: (float(row["model_score"]), -int(row["retrieval_parent_context_index"]))) for values in by_parent.values()]
            cards.sort(key=lambda row: (-float(row["model_score"]), int(row["retrieval_parent_rank"]), str(row["retrieval_molecule_identity_key"])))
        output.append({"query_id": query_id, "target": int(rows[0]["query_gold_Y"]), "cards": cards[:k]})
    provenance = {
        "profile": GOLD_PROFILES[task], "version": str(version_path.resolve()),
        "version_sha256": sha256_file(version_path), "rankings": str(rankings_path.resolve()),
        "rankings_sha256": sha256_file(rankings_path), "model": version["model"],
    }
    return output, provenance


def _tdc_neighbors(task: str, split: str, method: str, k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = cache_profile_root(TDC_PROFILES[task]) / task
    index_path = root / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index["splits"][split]["levels"]["L1"]
    version_path = root / entry["manifest"]
    version = json.loads(version_path.read_text(encoding="utf-8"))
    database = version_path.with_name(str(version["database"]))
    if (index.get("status") != "complete"
            or index.get("ranking_modes") != ["morgan", "assay-transfer"]
            or index.get("neighbor_identity_policy_by_level", {}).get("L1") != "scaffold_disjoint"
            or sha256_file(version_path) != entry["manifest_sha256"]
            or sha256_file(database) != version["database_sha256"]):
        raise ValueError(f"Incompatible TDC L1 cache: {root}")
    rank, context = (("morgan_rank", "morgan_context_id") if method == "morgan"
                     else ("assay_rank", "assay_context_id"))
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        labels = {str(row[0]): int(row[1]) for row in connection.execute("SELECT context_id,gold_label FROM contexts")}
        targets = {
            str(row["benchmark_row_id"]): int(row["Y"])
            for row in read_jsonl(tdc_task_root(task) / f"{split}_molecule_condition_labels.jsonl")
        }
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in connection.execute(
            f"SELECT benchmark_row_id,item_id,parent_smiles,morgan_similarity,{rank} AS selected_rank,{context} AS context_id,assay_transfer_score FROM rankings WHERE {rank}<=? ORDER BY benchmark_row_id,{rank}",
            (k,),
        ):
            item = dict(row)
            item["retrieval_gold_Y"] = labels[str(item["context_id"])]
            item["retrieval_molecule_identity_key"] = str(item["item_id"])
            item["retrieval_record_id"] = str(item["context_id"])
            grouped[str(item["benchmark_row_id"])].append(item)
    output = [{"query_id": query_id, "target": targets[query_id], "cards": cards}
              for query_id, cards in sorted(grouped.items())]
    if any(len(row["cards"]) != k for row in output):
        raise ValueError(f"{task}/{split}/{method} lacks {k} TDC L1 cards")
    provenance = {
        "profile": TDC_PROFILES[task], "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path), "version": str(version_path.resolve()),
        "version_sha256": sha256_file(version_path), "database": str(database.resolve()),
        "database_sha256": sha256_file(database), "model": index["model"],
    }
    return output, provenance


def run(output_dir: Path, *, k: int = 3, width: int = 100) -> dict[str, Any]:
    if k != 3 or width != 100:
        raise ValueError("This frozen study requires K=3 and W=100")
    metrics, predictions, neighbors, inputs = [], [], [], {}
    for benchmark in EXPECTED_QUERIES:
        for task in TASKS:
            for split in SPLITS:
                for method in METHODS:
                    loader = _gold_neighbors if benchmark == "gold_v1" else _tdc_neighbors
                    queries, provenance = loader(task, split, method, k)
                    if len(queries) != EXPECTED_QUERIES[benchmark][task][split]:
                        raise ValueError(f"Unexpected query count for {benchmark}/{task}/{split}")
                    key = f"{benchmark}/{task}/{split}"
                    inputs.setdefault(key, provenance)
                    rows = []
                    for query in queries:
                        prediction, score = _vote(int(card["retrieval_gold_Y"]) for card in query["cards"])
                        rows.append({"target": query["target"], "prediction": prediction})
                        predictions.append({
                            "benchmark": benchmark, "task": task, "split": split,
                            "method": method, "query_id": query["query_id"],
                            "target": query["target"], "prediction": prediction,
                            "positive_vote_fraction": score,
                        })
                        for rank_index, card in enumerate(query["cards"], 1):
                            neighbors.append({
                                "benchmark": benchmark, "task": task, "split": split,
                                "method": method, "query_id": query["query_id"],
                                "neighbor_rank": rank_index,
                                "context_id": card["retrieval_record_id"],
                                "item_id": card["retrieval_molecule_identity_key"],
                                "label": int(card["retrieval_gold_Y"]),
                            })
                    targets = [row["target"] for row in rows]
                    predicted = [row["prediction"] for row in rows]
                    class_f1 = f1_score(targets, predicted, labels=[0, 1], average=None, zero_division=0.0)
                    model = provenance["model"]
                    metrics.append({
                        "benchmark": benchmark, "task": task, "split": split,
                        "method": method, "k": k, "width": width, "n": len(rows),
                        "macro_f1": float(f1_score(targets, predicted, labels=[0, 1], average="macro", zero_division=0.0)),
                        "accuracy": float(accuracy_score(targets, predicted)),
                        "negative_f1": float(class_f1[0]), "positive_f1": float(class_f1[1]),
                        "model": model["model"], "model_revision": model["revision"],
                        "neighbor_identity_policy": "scaffold_disjoint",
                    })
    lookup = {(row["benchmark"], row["task"], row["split"], row["method"]): row for row in metrics}
    comparison = []
    for benchmark in EXPECTED_QUERIES:
        for task in TASKS:
            for split in SPLITS:
                morgan = lookup[(benchmark, task, split, "morgan")]
                assay = lookup[(benchmark, task, split, "assay_transfer")]
                comparison.append({
                    "benchmark": benchmark, "task": task, "split": split, "n": assay["n"],
                    "morgan_macro_f1": morgan["macro_f1"],
                    "assay_transfer_macro_f1": assay["macro_f1"],
                    "macro_f1_delta": assay["macro_f1"] - morgan["macro_f1"],
                    "morgan_accuracy": morgan["accuracy"],
                    "assay_transfer_accuracy": assay["accuracy"],
                    "accuracy_delta": assay["accuracy"] - morgan["accuracy"],
                })
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_tsv(output_dir / "metrics.tsv", metrics)
    _write_tsv(output_dir / "comparison.tsv", comparison)
    _write_tsv(output_dir / "predictions.tsv", predictions)
    _write_tsv(output_dir / "neighbors.tsv", neighbors)
    write_json_atomic(output_dir / "manifest.json", {
        "status": "complete", "study": "direct_l1_knn_gold_tdc_k3_w100_v1",
        "k": k, "width": width, "vote": "unweighted_context_card_majority",
        "neighbor_identity_policy": "scaffold_disjoint", "inputs": inputs,
        "code_sha256": sha256_file(Path(__file__)),
    })
    lines = ["# Direct L1 KNN: assay transfer versus Morgan", "",
             "All L1 candidate universes are scaffold-disjoint; K=3 and W=100.", "",
             "| Benchmark | Task | Split | N | Morgan F1 | Assay F1 | Delta |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {row['benchmark']} | {row['task']} | {row['split']} | {row['n']} | {row['morgan_macro_f1']:.4f} | {row['assay_transfer_macro_f1']:.4f} | {row['macro_f1_delta']:+.4f} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "complete", "output_dir": str(output_dir), "comparison": comparison}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, k=args.k, width=args.width), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
