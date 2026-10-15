"""Score frozen Gold or TDC direct-card selections under their benchmark contract."""

from __future__ import annotations

from predict.baselines.manual_evidence import add_manual_arguments, run_manual

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path
from predict.baselines.direct_l1_knn import _write_tsv
from predict.harnesses.branches.flat import _load_preselected_contexts, PRESELECTED_DIRECT_SCHEMAS
from predict.harnesses.branches.matrix import load_preselected_direct_grid
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic


def run(
    grid_path: Path, output_dir: Path, *, benchmark: str, subset: str, task: str,
    neighbor_order: str, ks: tuple[int, ...], ambiguous_card_policy: str | None,
) -> Path:
    if benchmark not in {"gold_v1", "tdc"} or neighbor_order not in {"assay-rank", "policy-relevance"}:
        raise ValueError("Neighbor order is incompatible with the requested benchmark")
    if not ks or any(k < 1 or k > 10 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("K values must be distinct integers from 1 to 10")
    grid_path = grid_path.resolve()
    profiles = load_preselected_direct_grid(grid_path, benchmark=benchmark, subset=subset)
    if not profiles or any(task not in profile["tasks"] for profile in profiles):
        raise ValueError(f"Grid does not cover {task} in every profile")
    subset = profiles[0]["subset"]
    target_path = (split_path if benchmark == "gold_v1" else tdc_split_path)(task, subset)
    targets = {str(row["benchmark_row_id"]): int(row["Y"]) for row in read_jsonl(target_path)}
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in read_jsonl(target_path)}
    metrics, predictions, neighbors, sources = [], [], [], []
    cache_identity = None
    ranks_by_cache = {}
    for profile in profiles:
        name = profile["name"]
        leaf_path = Path(profile["task_manifests"][task]["path"])
        leaf = json.loads(leaf_path.read_text(encoding="utf-8"))
        if leaf.get("schema_version") not in PRESELECTED_DIRECT_SCHEMAS or leaf.get("status") != "complete":
            raise ValueError(f"Incomplete direct selection: {leaf_path}")
        inputs = leaf["inputs"]
        cache = inputs["direct_cache"]
        policy = leaf.get("score_policy")
        score_policy_name = (policy or {}).get("version")
        label_release = "v1" if benchmark == "gold_v1" else {"benchmark": "tdc"}
        if (leaf["benchmark"] != benchmark or leaf["subset"] != subset
                or leaf["task_id"] != task or leaf["profile"] != name
                or Path(inputs["queries"]).resolve() != target_path.resolve()
                or inputs["queries_sha256"] != sha256_file(target_path)
                or cache["label_release"] != label_release):
            raise ValueError(f"Selection identity or score policy differs: {name}")
        for field in ("release_index", "l1_manifest"):
            if sha256_file(Path(cache[field])) != cache[f"{field}_sha256"]:
                raise ValueError(f"Selected L1 cache manifest changed: {name}/{field}")
        identity = (cache["release_index_sha256"], cache["database_sha256"])
        if cache_identity is not None and identity != cache_identity:
            raise ValueError("Direct profiles use different L1 cache releases")
        cache_identity = identity
        _, receipt = _load_preselected_contexts(
            leaf_path, task=task, queries=queries, budget=10, benchmark=benchmark,
        )
        bands = receipt.get("score_visibility") or {}
        selected_path = leaf_path.parent / leaf["records"]["path"]
        with selected_path.open(encoding="utf-8", newline="") as handle:
            selected = list(csv.DictReader(handle, delimiter="\t"))
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["benchmark_row_id"]].append(row)
        if set(grouped) != set(targets):
            raise ValueError(f"Selected query identities differ: {name}")
        if identity not in ranks_by_cache:
            with sqlite3.connect(f"file:{Path(cache['database']).resolve()}?mode=ro&immutable=1", uri=True) as db:
                ranks = defaultdict(dict)
                for qid, cid, parent, label, assay_rank, assay_score, similarity in db.execute(
                    "SELECT r.benchmark_row_id, c.context_id, c.parent_id, c.gold_label, "
                    "CASE WHEN c.context_id=r.assay_context_id THEN r.assay_rank END, "
                    "CASE WHEN c.context_id=r.assay_context_id THEN r.assay_transfer_score END, "
                    "r.morgan_similarity FROM rankings r JOIN contexts c "
                    "ON c.context_id=r.morgan_context_id OR c.context_id=r.assay_context_id"
                ):
                    ranks[qid][cid] = (parent, int(label), assay_rank, assay_score, float(similarity))
            ranks_by_cache[identity] = ranks
        ranks = ranks_by_cache[identity]
        for qid in sorted(targets):
            cards = grouped[qid]
            if len(cards) != 10:
                raise ValueError(f"Expected ten selected cards: {name}/{qid}")
            ordered = []
            for card in cards:
                cid = card["context_id"]
                parent, label, assay_rank, assay_score, similarity = ranks[qid][cid]
                if parent != card["parent_id"] or label != int(card["gold_label"]):
                    raise ValueError(f"Frozen card identity differs: {name}/{qid}/{cid}")
                if neighbor_order == "assay-rank":
                    if assay_rank is None or assay_score is None:
                        raise ValueError(f"Assay rank or score is missing: {name}/{qid}/{cid}")
                    relevance = -int(assay_rank)
                else:
                    raw = card.get("assay_transfer_score")
                    exact = (float(raw) if raw not in (None, "") else None) if policy else assay_score
                    weight = float(leaf["objective"]["gated_assay_lambda"])
                    relevance = (similarity if bands.get(qid) == "low" or exact is None else
                                 (similarity + weight * similarity * exact) / (1 + weight))
                ordered.append((relevance, cid, label, card, similarity))
            ordered.sort(key=lambda item: (-item[0], item[1]))
            if any(label == -1 for _, _, label, _, _ in ordered) and ambiguous_card_policy != "abstain":
                raise ValueError("Ambiguous cards require --ambiguous-card-policy abstain")
            eligible = []
            for rank, (relevance, cid, label, card, similarity) in enumerate(ordered, 1):
                neighbors.append({"benchmark": benchmark, "subset": subset, "task": task,
                                  "profile": name, "query_id": qid, "neighbor_rank": rank,
                                  "context_id": cid, "label": label, "selection_rank": card["selection_rank"],
                                  "coverage_band": bands.get(qid, "complete"), "similarity": similarity,
                                  "relevance": relevance, "exact_score": card.get("assay_transfer_score", "")})
                if label in (0, 1):
                    eligible.append(label)
            for k in ks:
                votes = eligible[:k]
                if not votes:
                    raise ValueError(f"No binary-labeled cards: {name}/{qid}")
                fraction = sum(votes) / len(votes)
                predictions.append({"benchmark": benchmark, "subset": subset, "task": task,
                                    "profile": name, "neighbor_order": neighbor_order, "k": k,
                                    "query_id": qid, "target": targets[qid],
                                    "prediction": int(fraction >= .5), "positive_vote_fraction": fraction,
                                    "effective_k": len(votes), "coverage_band": bands.get(qid, "complete")})
        sources.append({"profile": name, "selection_manifest": str(leaf_path),
                        "selection_manifest_sha256": sha256_file(leaf_path),
                        "cache_database_sha256_from_manifest": cache["database_sha256"],
                        "score_policy": score_policy_name})
    for name in (profile["name"] for profile in profiles):
        for k in ks:
            rows = [row for row in predictions if row["profile"] == name and row["k"] == k]
            y = [row["target"] for row in rows]
            p = [row["prediction"] for row in rows]
            counts = Counter(zip(y, p))
            metrics.append({"benchmark": benchmark, "subset": subset, "task": task,
                            "profile": name, "neighbor_order": neighbor_order, "k": k,
                            "n": len(rows), "macro_f1": f1_score(y, p, labels=[0, 1],
                                                                   average="macro", zero_division=0),
                            "accuracy": accuracy_score(y, p), "tn": counts[(0, 0)],
                            "fp": counts[(0, 1)], "fn": counts[(1, 0)], "tp": counts[(1, 1)],
                            "queries_below_k": sum(row["effective_k"] < k for row in rows)})
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("metrics.tsv", metrics), ("predictions.tsv", predictions),
                       ("neighbors.tsv", neighbors)):
        _write_tsv(output_dir / name, rows)
    report = (f"# Selected-card kNN: {benchmark} {task} {subset}\n\n"
              f"Neighbor order: `{neighbor_order}`. Ambiguous-card policy: "
              f"`{ambiguous_card_policy or 'none'}`. Full results: [metrics.tsv](metrics.tsv).\n")
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    write_json_atomic(output_dir / "manifest.json", {
        "schema_version": "selected_direct_knn.v1", "benchmark": benchmark,
        "subset": subset, "task": task, "neighbor_order": neighbor_order,
        "ambiguous_card_policy": ambiguous_card_policy, "k": ks,
        "grid_path": str(grid_path), "grid_sha256": sha256_file(grid_path),
        "target_path": str(target_path.resolve()), "target_sha256": sha256_file(target_path),
        "profiles": sources,
        "outputs": {name: sha256_file(output_dir / name) for name in
                    ("metrics.tsv", "predictions.tsv", "neighbors.tsv", "report.md")},
    })
    return output_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-grid", type=Path)
    parser.add_argument("--benchmark", choices=("gold_v1", "tdc"))
    parser.add_argument("--subset", choices=("valid", "test"))
    parser.add_argument("--task")
    parser.add_argument("--neighbor-order", choices=("assay-rank", "policy-relevance"))
    parser.add_argument("--k", type=int, nargs="+", required=True)
    parser.add_argument("--ambiguous-card-policy", choices=("abstain",))
    parser.add_argument("--output-dir", type=Path, required=True)
    add_manual_arguments(parser)
    args = parser.parse_args()
    if args.uids is not None or args.uids_file is not None:
        print(run_manual(args))
        return 0
    if args.selection_grid is None or not all((args.benchmark, args.subset, args.task, args.neighbor_order)):
        parser.error("automatic selection requires --selection-grid, --benchmark, --subset, --task and --neighbor-order")
    print(run(args.selection_grid, args.output_dir, benchmark=args.benchmark,
              subset=args.subset, task=args.task, neighbor_order=args.neighbor_order,
              ks=tuple(args.k), ambiguous_card_policy=args.ambiguous_card_policy))


if __name__ == "__main__":
    main()
