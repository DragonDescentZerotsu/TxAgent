"""Build HF-compatible MLP train/validation/test splits from dynamic_v1 pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
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
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/"
    "dynamic_v1_mlp_splits/endpoint_disjoint_7_1_2"
)

SPLITS = ("train", "validation", "test")
LABEL_TO_COMPLETION = {"similar": "A", "different": "B"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_summary = read_endpoint_summary(Path(args.endpoint_summary))
    group_counts = count_groups(
        Path(args.pairs),
        progress_every=args.progress_every,
        similar_delta=args.similar_delta,
        different_delta=args.different_delta,
    )
    ratios = parse_ratios(args.ratios)
    label_tag = delta_tag(args.similar_delta, args.different_delta)
    split_by_group = assign_group_splits(
        group_counts,
        ratios=ratios,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    summary = write_splits(
        pairs_path=Path(args.pairs),
        out_dir=out_dir,
        endpoint_summary=endpoint_summary,
        split_by_group=split_by_group,
        ratios=ratios,
        split_version=f"dynamic_v1_{label_tag}_{args.split_mode}_{ratio_tag(args.ratios)}",
        similar_delta=args.similar_delta,
        different_delta=args.different_delta,
        progress_every=args.progress_every,
    )
    summary.update(
        {
            "created_at_unix": time.time(),
            "wall_s": round(time.time() - started, 3),
            "parameters": {
                "pairs": args.pairs,
                "endpoint_summary": args.endpoint_summary,
                "out_dir": args.out_dir,
                "ratios": args.ratios,
                "seed": args.seed,
                "split_mode": args.split_mode,
                "similar_delta": args.similar_delta,
                "different_delta": args.different_delta,
            },
            "files": {split: str(out_dir / f"{split}.jsonl") for split in SPLITS},
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": summary["split_counts"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--endpoint-summary", default=DEFAULT_ENDPOINT_SUMMARY)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--ratios", default="0.7,0.1,0.2", help="train,validation,test ratios.")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--similar-delta", type=float, default=0.5)
    parser.add_argument("--different-delta", type=float, default=1.0)
    parser.add_argument(
        "--split-mode",
        choices=["endpoint_disjoint", "pair_random"],
        default="endpoint_disjoint",
        help="endpoint_disjoint keeps assay_id+standard_type groups in one split.",
    )
    parser.add_argument("--progress-every", type=int, default=200000)
    return parser.parse_args(argv)


def parse_ratios(raw: str) -> dict[str, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--ratios must contain exactly three values: train,validation,test")
    total = sum(values)
    if total <= 0:
        raise ValueError("--ratios must sum to a positive value")
    return {split: value / total for split, value in zip(SPLITS, values, strict=True)}


def ratio_tag(raw: str) -> str:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if values == [0.7, 0.1, 0.2]:
        return "7_1_2"
    return "_".join(f"{value:g}".replace(".", "p") for value in values)


def delta_tag(similar_delta: float, different_delta: float) -> str:
    return f"pchembl_sim_le_{similar_delta:g}_diff_ge_{different_delta:g}".replace(".", "p")


def read_endpoint_summary(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[endpoint_key(row)] = row
    return rows


def count_groups(
    path: Path,
    *,
    progress_every: int,
    similar_delta: float,
    different_delta: float,
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.time()
    completed = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = label_from_delta(row.get("abs_activity_delta"), similar_delta, different_delta)
            if label is None:
                continue
            counts[endpoint_key(row)][label] += 1
            completed += 1
            if progress_every > 0 and completed % progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "stage": "count_groups",
                            "completed_non_ambiguous": completed,
                            "groups": len(counts),
                            "rate_per_s": round(completed / elapsed, 2),
                            "elapsed_s": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )
    return counts


def assign_group_splits(
    group_counts: dict[str, Counter[str]],
    *,
    ratios: dict[str, float],
    seed: int,
    split_mode: str,
) -> dict[str, str]:
    if split_mode == "pair_random":
        return {}

    items = [
        (group, int(counter["similar"] + counter["different"]))
        for group, counter in group_counts.items()
        if int(counter["similar"] + counter["different"]) > 0
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    total = sum(weight for _, weight in items)
    targets = {split: total * ratio for split, ratio in ratios.items()}
    assigned_weight = {split: 0 for split in SPLITS}
    split_by_group: dict[str, str] = {}
    for group, weight in items:
        split = max(SPLITS, key=lambda name: targets[name] - assigned_weight[name])
        split_by_group[group] = split
        assigned_weight[split] += weight
    return split_by_group


def write_splits(
    *,
    pairs_path: Path,
    out_dir: Path,
    endpoint_summary: dict[str, dict[str, str]],
    split_by_group: dict[str, str],
    ratios: dict[str, float],
    split_version: str,
    similar_delta: float,
    different_delta: float,
    progress_every: int,
) -> dict[str, Any]:
    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    bucket_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    assay_type_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    endpoint_counts: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing_endpoint_context = 0
    written = 0
    started = time.time()
    rng = random.Random(17)
    try:
        with gzip.open(pairs_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for source_index, row in enumerate(reader):
                label = label_from_delta(row.get("abs_activity_delta"), similar_delta, different_delta)
                if label is None:
                    continue
                key = endpoint_key(row)
                split = split_by_group.get(key) or random_split(rng, ratios)
                endpoint = endpoint_summary.get(key)
                if endpoint is None:
                    missing_endpoint_context += 1
                    endpoint = {}

                record = build_record(row, endpoint, split, split_version, label, similar_delta, different_delta, source_index)
                handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                split_counts[split][label] += 1
                bucket_counts[split][str(row.get("similarity_bucket") or "")] += 1
                assay_type_counts[split][str(endpoint.get("assay_type") or "")] += 1
                endpoint_counts[split].add(key)
                written += 1
                if progress_every > 0 and written % progress_every == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "stage": "write_splits",
                                "written": written,
                                "rate_per_s": round(written / elapsed, 2),
                                "elapsed_s": round(elapsed, 1),
                            }
                        ),
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "n": written,
        "split_counts": {
            split: {
                "total": int(sum(split_counts[split].values())),
                "label_counts": dict(sorted(split_counts[split].items())),
                "similarity_bucket_counts": dict(sorted(bucket_counts[split].items())),
                "assay_type_counts": dict(sorted(assay_type_counts[split].items())),
                "n_endpoint_groups": len(endpoint_counts[split]),
            }
            for split in SPLITS
        },
        "missing_endpoint_context": missing_endpoint_context,
        "label_rule": {
            "label_type": "pchembl_delta",
            "similar_delta": similar_delta,
            "different_delta": different_delta,
            "ambiguous": "excluded",
        },
    }


def random_split(rng: random.Random, ratios: dict[str, float]) -> str:
    value = rng.random()
    train_cutoff = ratios["train"]
    validation_cutoff = train_cutoff + ratios["validation"]
    if value < train_cutoff:
        return "train"
    if value < validation_cutoff:
        return "validation"
    return "test"


def build_record(
    row: dict[str, str],
    endpoint: dict[str, str],
    split: str,
    split_version: str,
    label: str,
    similar_delta: float,
    different_delta: float,
    source_index: int,
) -> dict[str, Any]:
    assay_id = parse_int(row.get("assay_id"))
    standard_type = row.get("standard_type") or endpoint.get("standard_type") or ""
    endpoint_id = f"{assay_id}|{standard_type}|continuous"
    pair_id = (
        f"{endpoint_id}|{row.get('molecule_a_chembl_id', '')}"
        f"<>{row.get('molecule_b_chembl_id', '')}"
    )
    metadata = {
        "sample_id": f"dynamic_v1:{source_index}",
        "pair_id": pair_id,
        "source_index": source_index,
        "source_run_id": "chembl36_activity_transfer_dynamic_v1",
        "split": split,
        "split_version": split_version,
        "endpoint_scope": "assay",
        "endpoint_id": endpoint_id,
        "assay_chembl_id": row.get("assay_chembl_id"),
        "assay_id": assay_id,
        "standard_type": standard_type,
        "assay_type": endpoint.get("assay_type"),
        "assay_test_type": endpoint.get("assay_test_type"),
        "assay_category": endpoint.get("assay_category"),
        "confidence_score": parse_int(endpoint.get("confidence_score")),
        "relationship_type": endpoint.get("relationship_type"),
        "target_chembl_id": row.get("target_chembl_id") or endpoint.get("target_chembl_id"),
        "target_name": row.get("target_pref_name") or endpoint.get("target_pref_name"),
        "target_type": endpoint.get("target_type"),
        "target_organism": endpoint.get("target_organism"),
        "label_type": "pchembl_delta",
        "similar_delta": similar_delta,
        "different_delta": different_delta,
        "original_dynamic_v1_label": row.get("label"),
        "molecule_a_chembl_id": row.get("molecule_a_chembl_id"),
        "molecule_b_chembl_id": row.get("molecule_b_chembl_id"),
        "activity_a": parse_float(row.get("activity_a")),
        "activity_b": parse_float(row.get("activity_b")),
        "abs_activity_delta": parse_float(row.get("abs_activity_delta")),
        "weighted_tanimoto": parse_float(row.get("tanimoto")),
        "similarity_bucket": row.get("similarity_bucket"),
        "direction": "molecule_a_to_molecule_b",
    }
    return {
        "prompt": build_prompt(row, endpoint),
        "completion": LABEL_TO_COMPLETION[label],
        "metadata": metadata,
    }


def label_from_delta(value: Any, similar_delta: float, different_delta: float) -> str | None:
    delta = parse_float(value)
    if delta is None:
        return None
    if delta <= similar_delta:
        return "similar"
    if delta >= different_delta:
        return "different"
    return None


def build_prompt(row: dict[str, str], endpoint: dict[str, str]) -> str:
    endpoint_lines = [
        f"- Measurement type: {display_value(row.get('standard_type') or endpoint.get('standard_type'))}",
        f"- Assay type: {display_value(endpoint.get('assay_type'))}",
        f"- Assay test type: {display_value(endpoint.get('assay_test_type'))}",
        f"- Assay category: {display_value(endpoint.get('assay_category'))}",
        f"- Assay description: {display_value(endpoint.get('description'))}",
        f"- Target name: {display_value(row.get('target_pref_name') or endpoint.get('target_pref_name'))}",
        f"- Target type: {display_value(endpoint.get('target_type'))}",
        f"- Target organism: {display_value(endpoint.get('target_organism'))}",
        f"- Confidence score: {display_value(endpoint.get('confidence_score'))}",
        f"- Relationship type: {display_value(endpoint.get('relationship_type'))}",
    ]
    return (
        "You are given two molecules and an assay endpoint. Decide whether the endpoint behavior "
        "should transfer between the molecules.\n\n"
        "## Endpoint\n"
        + "\n".join(endpoint_lines)
        + "\n\n## Molecule A\n"
        + f"- SMILES: {row.get('molecule_a_smiles', '')}\n\n"
        + "## Molecule B\n"
        + f"- SMILES: {row.get('molecule_b_smiles', '')}\n\n"
        + "Should this assay endpoint transfer between Molecule A and Molecule B?\n"
        + "(A) transfer\n"
        + "(B) not transfer\n\n"
        + "Answer:"
    )


def endpoint_key(row: dict[str, Any]) -> str:
    return f"{row.get('assay_id')}|{row.get('standard_type')}"


def display_value(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "not specified"


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


if __name__ == "__main__":
    raise SystemExit(main())
