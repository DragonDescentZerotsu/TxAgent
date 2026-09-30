"""Materialize HF prompt/completion validation splits for activity-transfer runs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_REPOS = os.environ.get("TXAGENT_TRANSFER_DATASETS", "").split()
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/hf_transfer_valid20k"


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    for repo in args.repos:
        materialize_repo(repo, args.split, args.limit, out_root)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS or None, required=not DEFAULT_REPOS)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def materialize_repo(repo: str, split: str, limit: int, out_root: Path) -> None:
    dataset_name = repo.split("/", 1)[-1]
    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split}.jsonl"
    summary_path = out_dir / "summary.json"

    label_counts: Counter[str] = Counter()
    similarity_bucket_counts: Counter[str] = Counter()
    assay_type_counts: Counter[str] = Counter()
    eval_subset_counts: Counter[str] = Counter()
    split_version_counts: Counter[str] = Counter()
    weighted_tanimoto_by_bucket: dict[str, list[float]] = defaultdict(list)
    columns: list[str] = []
    n_rows = 0

    ds = load_dataset(repo, split=split, streaming=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            if not columns:
                columns = list(row.keys())
            validate_row(repo, row)
            metadata = row.get("metadata") or {}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

            label_counts[str(row.get("completion", ""))] += 1
            count_metadata(metadata, "similarity_bucket", similarity_bucket_counts)
            count_metadata(metadata, "assay_type", assay_type_counts)
            count_metadata(metadata, "eval_subset", eval_subset_counts)
            count_metadata(metadata, "split_version", split_version_counts)
            bucket = metadata.get("similarity_bucket")
            tanimoto = parse_float(metadata.get("weighted_tanimoto"))
            if bucket is not None and tanimoto is not None:
                weighted_tanimoto_by_bucket[str(bucket)].append(tanimoto)

            if limit > 0 and n_rows >= limit:
                break

    summary = {
        "repo": repo,
        "split": split,
        "limit": limit,
        "num_rows": n_rows,
        "columns": columns,
        "label_counts": dict(sorted(label_counts.items())),
        "similarity_bucket_counts": dict(sorted(similarity_bucket_counts.items())),
        "assay_type_counts": dict(sorted(assay_type_counts.items())),
        "eval_subset_counts": dict(sorted(eval_subset_counts.items())),
        "split_version_counts": dict(sorted(split_version_counts.items())),
        "weighted_tanimoto_by_bucket": summarize_float_groups(weighted_tanimoto_by_bucket),
        "files": {"jsonl": str(jsonl_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repo": repo, "rows": n_rows, "jsonl": str(jsonl_path)}, ensure_ascii=False), flush=True)


def validate_row(repo: str, row: dict[str, Any]) -> None:
    missing = {"prompt", "completion", "metadata"} - set(row)
    if missing:
        raise ValueError(f"{repo} row is missing required columns: {sorted(missing)}")
    if not isinstance(row.get("metadata"), dict):
        raise TypeError(f"{repo} metadata must be a dict, got {type(row.get('metadata')).__name__}")


def count_metadata(metadata: dict[str, Any], key: str, counter: Counter[str]) -> None:
    value = metadata.get(key)
    if value is not None:
        counter[str(value)] += 1


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_float_groups(groups: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        ordered = sorted(values)
        n = len(ordered)
        summary[key] = {
            "n": n,
            "min": ordered[0],
            "q25": ordered[int((n - 1) * 0.25)],
            "median": ordered[int((n - 1) * 0.50)],
            "mean": sum(ordered) / n,
            "q75": ordered[int((n - 1) * 0.75)],
            "max": ordered[-1],
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
