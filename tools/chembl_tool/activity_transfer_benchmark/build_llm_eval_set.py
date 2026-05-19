"""Build a stratified LLM eval set for ChEMBL activity transfer."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = (
    "outputs/chembl_tool/activity_transfer_benchmark/"
    "chembl36_activity_transfer_dynamic_v1/continuous_pairs.tsv.gz"
)
DEFAULT_ENDPOINT_SUMMARY = (
    "outputs/chembl_tool/activity_transfer_benchmark/"
    "chembl36_activity_transfer_dynamic_v1/continuous_assay_endpoint_summary.tsv"
)
DEFAULT_MCS_RESULTS = (
    "outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime/"
    "dynamic_v1_mcs_full_t2_w128_stream/mcs_sample_results.tsv.tmp"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/llm_eval_sets"
TANIMOTO_BUCKETS = [
    "very_close_analog",
    "close_analog",
    "moderate_analog",
    "weak_analog",
    "distant_analog",
    "very_distant_analog",
]
LABELS = ["similar", "different"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    run_id = args.run_id
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_meta = read_endpoint_meta(Path(args.endpoint_summary))
    mcs_rows = read_mcs_rows(Path(args.mcs_results))
    selected = stratified_select(
        Path(args.pairs),
        endpoint_meta=endpoint_meta,
        mcs_rows=mcs_rows,
        n_total=args.n_total,
        seed=args.seed,
        max_per_assay=args.max_per_assay,
        min_abs_delta_similar=args.min_abs_delta_similar,
        min_abs_delta_different=args.min_abs_delta_different,
    )
    rng.shuffle(selected)
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index

    jsonl_path = out_dir / "eval_pairs.jsonl"
    tsv_path = out_dir / "eval_pairs.tsv"
    write_jsonl(jsonl_path, selected)
    write_tsv(tsv_path, selected)
    summary = summarize(selected, args)
    summary["files"] = {"jsonl": str(jsonl_path), "tsv": str(tsv_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir / "report_zh.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--endpoint-summary", default=DEFAULT_ENDPOINT_SUMMARY)
    parser.add_argument("--mcs-results", default=DEFAULT_MCS_RESULTS)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="dynamic_v1_llm_3k")
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--max-per-assay", type=int, default=8)
    parser.add_argument("--min-abs-delta-similar", type=float, default=0.0)
    parser.add_argument("--min-abs-delta-different", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=20260517)
    return parser.parse_args(argv)


def read_endpoint_meta(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[(row["assay_chembl_id"], row["standard_type"])] = row
    return out


def read_mcs_rows(path: Path) -> dict[int, dict[str, str]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            out[int(row["index"])] = row
    return out


def stratified_select(
    pair_path: Path,
    *,
    endpoint_meta: dict[tuple[str, str], dict[str, str]],
    mcs_rows: dict[int, dict[str, str]],
    n_total: int,
    seed: int,
    max_per_assay: int,
    min_abs_delta_similar: float,
    min_abs_delta_different: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    quota = build_quota(n_total)
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in quota}
    seen: Counter = Counter()
    with gzip.open(pair_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for pair_index, row in enumerate(reader):
            label = row["label"]
            if label not in LABELS:
                continue
            abs_delta = float(row["abs_activity_delta"])
            if label == "similar" and abs_delta > 0.5 - min_abs_delta_similar:
                continue
            if label == "different" and abs_delta < min_abs_delta_different:
                continue
            mcs = mcs_rows.get(pair_index)
            if not mcs:
                continue
            key = (label, row["similarity_bucket"])
            if key not in reservoirs:
                continue
            seen[key] += 1
            candidate = build_sample_record(pair_index, row, mcs, endpoint_meta)
            reservoir = reservoirs[key]
            target = quota[key]
            if len(reservoir) < target:
                reservoir.append(candidate)
            else:
                replace_index = rng.randrange(seen[key])
                if replace_index < target:
                    reservoir[replace_index] = candidate

    selected: list[dict[str, Any]] = []
    per_assay: Counter = Counter()
    overflow: list[dict[str, Any]] = []
    for key in quota:
        for row in reservoirs[key]:
            assay_key = f"{row['assay_chembl_id']}::{row['standard_type']}"
            if per_assay[assay_key] < max_per_assay:
                selected.append(row)
                per_assay[assay_key] += 1
            else:
                overflow.append(row)

    target_n = sum(quota.values())
    if len(selected) < target_n:
        rng.shuffle(overflow)
        selected.extend(overflow[: target_n - len(selected)])

    if len(selected) < target_n:
        raise RuntimeError(f"Only selected {len(selected)} rows, expected {target_n}.")
    return selected[:target_n]


def build_quota(n_total: int) -> dict[tuple[str, str], int]:
    base = n_total // (len(LABELS) * len(TANIMOTO_BUCKETS))
    remainder = n_total - base * len(LABELS) * len(TANIMOTO_BUCKETS)
    quota = {}
    keys = [(label, bucket) for label in LABELS for bucket in TANIMOTO_BUCKETS]
    for index, key in enumerate(keys):
        quota[key] = base + (1 if index < remainder else 0)
    return quota


def build_sample_record(
    pair_index: int,
    pair: dict[str, str],
    mcs: dict[str, str],
    endpoint_meta: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    meta = endpoint_meta.get((pair["assay_chembl_id"], pair["standard_type"]), {})
    tanimoto = float(pair["tanimoto"])
    mcs_coverage = float(mcs["mean_mcs_coverage"])
    label = pair["label"]
    return {
        "pair_index": pair_index,
        "label": label,
        "assay_chembl_id": pair["assay_chembl_id"],
        "assay_id": int(pair["assay_id"]),
        "standard_type": pair["standard_type"],
        "target_chembl_id": pair["target_chembl_id"],
        "target_pref_name": pair["target_pref_name"],
        "target_type": meta.get("target_type", ""),
        "target_organism": meta.get("target_organism", ""),
        "assay_type": meta.get("assay_type", ""),
        "assay_test_type": meta.get("assay_test_type", ""),
        "assay_category": meta.get("assay_category", ""),
        "confidence_score": meta.get("confidence_score", ""),
        "relationship_type": meta.get("relationship_type", ""),
        "assay_description": meta.get("description", ""),
        "query_molecule_chembl_id": pair["molecule_b_chembl_id"],
        "reference_molecule_chembl_id": pair["molecule_a_chembl_id"],
        "query_smiles": pair["molecule_b_smiles"],
        "reference_smiles": pair["molecule_a_smiles"],
        "reference_pchembl_value": float(pair["activity_a"]),
        "hidden_query_pchembl_value": float(pair["activity_b"]),
        "abs_activity_delta": float(pair["abs_activity_delta"]),
        "tanimoto": tanimoto,
        "similarity_bucket": pair["similarity_bucket"],
        "mean_mcs_coverage": mcs_coverage,
        "query_mcs_coverage": float(mcs["query_mcs_coverage"]),
        "reference_mcs_coverage": float(mcs["reference_mcs_coverage"]),
        "mcs_timed_out": parse_bool(mcs["timed_out"]),
        "baseline_tanimoto_0_50_prediction": "similar" if tanimoto >= 0.50 else "different",
        "baseline_tanimoto_0_48_prediction": "similar" if tanimoto >= 0.48 else "different",
        "baseline_mcs_0_70_prediction": "similar" if mcs_coverage >= 0.70 else "different",
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    label_counts = Counter(row["label"] for row in rows)
    bucket_counts = Counter(row["similarity_bucket"] for row in rows)
    label_bucket_counts = Counter((row["label"], row["similarity_bucket"]) for row in rows)
    assay_counts = Counter(f"{row['assay_chembl_id']}::{row['standard_type']}" for row in rows)
    return {
        "run_id": args.run_id,
        "n": len(rows),
        "parameters": vars(args),
        "label_counts": dict(label_counts),
        "similarity_bucket_counts": dict(bucket_counts),
        "label_bucket_counts": {f"{label}|{bucket}": count for (label, bucket), count in label_bucket_counts.items()},
        "n_assay_endpoints": len(assay_counts),
        "max_pairs_per_assay_endpoint_observed": max(assay_counts.values()) if assay_counts else 0,
        "baseline_metrics": {
            "tanimoto_0_50": compute_prediction_metrics(rows, "baseline_tanimoto_0_50_prediction"),
            "tanimoto_0_48": compute_prediction_metrics(rows, "baseline_tanimoto_0_48_prediction"),
            "mcs_0_70": compute_prediction_metrics(rows, "baseline_mcs_0_70_prediction"),
        },
    }


def compute_prediction_metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred_similar = row[prediction_key] == "similar"
        true_similar = row["label"] == "similar"
        if pred_similar and true_similar:
            tp += 1
        elif pred_similar and not true_similar:
            fp += 1
        elif not pred_similar and true_similar:
            fn += 1
        else:
            tn += 1
    return metrics_from_counts(tp, fp, tn, fn)


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    precision_similar = safe_div(tp, tp + fp)
    recall_similar = safe_div(tp, tp + fn)
    precision_different = safe_div(tn, tn + fn)
    recall_different = safe_div(tn, tn + fp)
    f1_similar = safe_div(2 * precision_similar * recall_similar, precision_similar + recall_similar)
    f1_different = safe_div(2 * precision_different * recall_different, precision_different + recall_different)
    return {
        "n": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": (recall_similar + recall_different) / 2.0,
        "macro_f1": (f1_similar + f1_different) / 2.0,
        "precision_similar": precision_similar,
        "recall_similar": recall_similar,
        "precision_different": precision_different,
        "recall_different": recall_different,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Activity-transfer LLM eval set",
        "",
        f"- samples: {summary['n']:,}",
        f"- assay endpoints: {summary['n_assay_endpoints']:,}",
        f"- max pairs per assay endpoint: {summary['max_pairs_per_assay_endpoint_observed']:,}",
        f"- label counts: {summary['label_counts']}",
        f"- similarity bucket counts: {summary['similarity_bucket_counts']}",
        "",
        "## Baselines",
        "",
        "| baseline | accuracy | balanced accuracy | macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["baseline_metrics"].items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
