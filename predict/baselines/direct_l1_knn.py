"""Compare Morgan and direct assay-transfer KNN on frozen L1 cards."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from sklearn.metrics import accuracy_score, f1_score

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_task_root
from predict.retrieval.assay_reranking.runtime import cache_profile_root
from predict.retrieval.assay_reranking.v9 import model_profile
from predict.utils.json import atomic_output_path, read_jsonl, sha256_file, write_json_atomic


GOLD_TASKS = (
    "bbb_martins", "bioavailability_ma", "skin_reaction",
    "ames", "dili", "carcinogens",
)
TDC_TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens")
SPLITS = ("valid", "test")
METHODS = ("morgan", "assay_transfer")
EXPECTED_QUERIES = {
    "gold_v1": {
        "bbb_martins": {"valid": 397, "test": 393},
        "bioavailability_ma": {"valid": 262, "test": 269},
        "skin_reaction": {"valid": 239, "test": 241},
        "ames": {"valid": 274, "test": 274},
        "dili": {"valid": 402, "test": 402},
        "carcinogens": {"valid": 469, "test": 469},
    },
    "tdc_v1": {
        "bbb_martins": {"valid": 197, "test": 530},
        "bioavailability_ma": {"valid": 64, "test": 128},
        "skin_reaction": {"valid": 40, "test": 82},
        "ames": {"valid": 720, "test": 1449},
        "dili": {"valid": 47, "test": 96},
        "carcinogens": {"valid": 27, "test": 56},
    },
}
GOLD_PROFILES = {
    "bbb_martins": "ranked_level_retrieval_v4",
    "bioavailability_ma": "ranked_level_retrieval_v4",
    "skin_reaction": "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2",
    "ames": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
    "dili": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
    "carcinogens": "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
}
TDC_PROFILES = {
    task: (
        f"flat_v5/tdc_v1/{task}/l1/assay_transfer/v10_3/tdc_pinned_v1"
        if task in {"ames", "dili", "carcinogens"}
        else "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1"
    )
    for task in TDC_TASKS
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


def _neighbors(
    benchmark: str, task: str, split: str, method: str, k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = GOLD_PROFILES[task] if benchmark == "gold_v1" else TDC_PROFILES[task]
    root = cache_profile_root(profile) / task
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
        raise ValueError(f"Incompatible L1 cache: {root}")
    rank, context = (("morgan_rank", "morgan_context_id") if method == "morgan"
                     else ("assay_rank", "assay_context_id"))
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        labels = {str(row[0]): int(row[1]) for row in connection.execute("SELECT context_id,gold_label FROM contexts")}
        target_path = (
            split_path(task, split, version="v1")
            if benchmark == "gold_v1"
            else tdc_task_root(task) / f"{split}_molecule_condition_labels.jsonl"
        )
        targets = {
            str(row["benchmark_row_id"]): int(row["Y"])
            for row in read_jsonl(target_path)
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
        raise ValueError(f"{benchmark}/{task}/{split}/{method} lacks {k} L1 cards")
    if set(grouped) != set(targets):
        raise ValueError(f"{benchmark}/{task}/{split} query identities differ")
    selected_model = index.get("model") or version.get("model")
    if selected_model is None:
        selected_model = model_profile(task, "v10_3_best")
    provenance = {
        "profile": profile, "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path), "version": str(version_path.resolve()),
        "version_sha256": sha256_file(version_path), "database": str(database.resolve()),
        "database_sha256": sha256_file(database), "target": str(target_path.resolve()),
        "target_sha256": sha256_file(target_path), "model": selected_model,
    }
    return output, provenance


def run(output_dir: Path, *, k: int = 3, width: int = 100) -> dict[str, Any]:
    if k not in {3, 5} or width != 100:
        raise ValueError("This study supports K=3 or K=5 with W=100")
    metrics, predictions, neighbors, inputs = [], [], [], {}
    tasks_by_benchmark = {"gold_v1": GOLD_TASKS, "tdc_v1": TDC_TASKS}
    for benchmark, tasks in tasks_by_benchmark.items():
        for task in tasks:
            for split in SPLITS:
                for method in METHODS:
                    queries, provenance = _neighbors(benchmark, task, split, method, k)
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
    for benchmark, tasks in tasks_by_benchmark.items():
        for task in tasks:
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
        "status": "complete", "study": f"direct_l1_knn_gold6_tdc6_k{k}_w100_v3",
        "k": k, "width": width, "vote": "unweighted_context_card_majority",
        "neighbor_identity_policy": "scaffold_disjoint", "inputs": inputs,
        "code_sha256": sha256_file(Path(__file__)),
    })
    lines = ["# Direct L1 KNN: assay transfer versus Morgan", "",
             f"All L1 candidate universes are scaffold-disjoint; K={k} and W=100.", "",
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
