"""Sample HF prompt/completion JSONL from exhaustive compact pair shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits import (
    LABEL_TO_COMPLETION,
    build_prompt,
    parse_ratios,
    ratio_tag,
)
from tools.chembl_tool.activity_transfer_benchmark.build_single_endpoint_exhaustive_raw_pairs import (
    SPLITS,
)


DEFAULT_COMPACT_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer/exhaustive_compact/"
    "CHEMBL4513218_inhibition_full_range_raw_std_sim_le_0p5_diff_ge_1p5_"
    "molecule_disjoint_7_1_2_exhaustive_nonambig_compact"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compact_dir = Path(args.compact_dir)
    summary = json.loads((compact_dir / "summary.json").read_text(encoding="utf-8"))
    ratios = parse_ratios(args.ratios)
    targets = split_targets(args.n_pairs, ratios)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out_root) / build_run_id(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = summary["endpoint"]
    endpoint_for_prompt = endpoint_prompt_context(endpoint)
    sample_summary = {
        "source_compact_dir": str(compact_dir),
        "parameters": vars(args),
        "targets": targets,
        "source_summary": {
            "value_stats": summary.get("value_stats"),
            "similar_delta": summary.get("similar_delta"),
            "different_delta": summary.get("different_delta"),
            "planned_split_counts": summary.get("planned_split_counts"),
            "planned_totals": summary.get("planned_totals"),
        },
        "splits": {},
    }

    started_all = time.time()
    for split in SPLITS:
        if args.splits and split not in args.splits:
            continue
        split_summary = sample_split(
            split=split,
            target=targets[split],
            compact_dir=compact_dir,
            out_path=out_dir / f"{split}.jsonl",
            endpoint=endpoint,
            endpoint_for_prompt=endpoint_for_prompt,
            source_total=int(summary["planned_split_counts"][split]["non_ambiguous_pairs"]),
            source_run_id=build_run_id(args),
            split_version=build_split_version(args),
            seed=args.seed + stable_text_int(split),
            progress_every=args.progress_every,
            max_source_rows=args.max_source_rows_per_split,
        )
        sample_summary["splits"][split] = split_summary

    sample_summary["elapsed_s"] = round(time.time() - started_all, 3)
    (out_dir / "summary.json").write_text(json.dumps(sample_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sample_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", default=DEFAULT_COMPACT_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-pairs", type=int, default=20_000_000)
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--splits", nargs="*", default=[], choices=list(SPLITS), help="Optional subset for smoke tests.")
    parser.add_argument("--max-source-rows-per-split", type=int, default=0, help="Smoke-test cap; 0 scans the full source split.")
    return parser.parse_args(argv)


def split_targets(n_pairs: int, ratios: dict[str, float]) -> dict[str, int]:
    train_n = int(n_pairs * ratios["train"])
    validation_n = int(n_pairs * ratios["validation"])
    return {
        "train": train_n,
        "validation": validation_n,
        "test": n_pairs - train_n - validation_n,
    }


def sample_split(
    *,
    split: str,
    target: int,
    compact_dir: Path,
    out_path: Path,
    endpoint: dict[str, Any],
    endpoint_for_prompt: dict[str, Any],
    source_total: int,
    source_run_id: str,
    split_version: str,
    seed: int,
    progress_every: int,
    max_source_rows: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    if max_source_rows > 0:
        source_total = min(source_total, max_source_rows)
    if target > source_total:
        raise ValueError(f"{split}: target={target:,} exceeds source_total={source_total:,}")
    remaining_total = source_total
    remaining_needed = target
    seen = 0
    written = 0
    label_counts: Counter[str] = Counter()
    started = time.time()
    shard_paths = sorted((compact_dir / "shards").glob(f"{split}.part*.tsv.gz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for split={split} in {compact_dir / 'shards'}")

    with out_path.open("w", encoding="utf-8") as out_handle:
        for shard_path in shard_paths:
            with gzip.open(shard_path, "rt", encoding="utf-8", newline="") as shard_handle:
                reader = csv.DictReader(shard_handle, delimiter="\t")
                for row in reader:
                    if max_source_rows > 0 and seen >= max_source_rows:
                        break
                    seen += 1
                    if remaining_needed <= 0:
                        remaining_total -= 1
                        continue
                    take = rng.random() < (remaining_needed / remaining_total)
                    remaining_total -= 1
                    if not take:
                        continue
                    record = build_record(
                        row,
                        endpoint=endpoint,
                        endpoint_for_prompt=endpoint_for_prompt,
                        source_run_id=source_run_id,
                        split_version=split_version,
                        source_index=seen - 1,
                    )
                    out_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    remaining_needed -= 1
                    label_counts[str(row["label"])] += 1
                    if progress_every > 0 and written % progress_every == 0:
                        elapsed = max(time.time() - started, 1e-9)
                        print(
                            json.dumps(
                                {
                                    "stage": "sample_exhaustive_compact",
                                    "split": split,
                                    "seen": seen,
                                    "written": written,
                                    "target": target,
                                    "rate_written_per_s": round(written / elapsed, 2),
                                    "elapsed_s": round(elapsed, 1),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            if max_source_rows > 0 and seen >= max_source_rows:
                break
    if written != target:
        raise RuntimeError(f"{split}: wrote {written:,}/{target:,}; source_total={source_total:,}, seen={seen:,}")
    return {
        "target": target,
        "seen": seen,
        "written": written,
        "label_counts": dict(sorted(label_counts.items())),
        "output": str(out_path),
        "elapsed_s": round(time.time() - started, 3),
    }


def build_record(
    row: dict[str, str],
    *,
    endpoint: dict[str, Any],
    endpoint_for_prompt: dict[str, Any],
    source_run_id: str,
    split_version: str,
    source_index: int,
) -> dict[str, Any]:
    label = str(row["label"])
    pair_id = (
        f"{row['assay_id']}|{row['standard_type']}|raw_std_delta|{row['value_version']}|"
        f"{row['molecule_a_chembl_id']}<>{row['molecule_b_chembl_id']}"
    )
    metadata = {
        "sample_id": f"{source_run_id}:{row['split']}:{source_index}",
        "pair_id": pair_id,
        "source_index": source_index,
        "source_run_id": source_run_id,
        "split": row["split"],
        "split_version": split_version,
        "endpoint_scope": "assay",
        "endpoint_id": f"{row['assay_id']}|{row['standard_type']}|raw_std_delta|{row['value_version']}",
        "assay_chembl_id": row["assay_chembl_id"],
        "assay_id": parse_int(row["assay_id"]),
        "standard_type": row["standard_type"],
        "raw_units": row.get("raw_units", ""),
        "assay_type": endpoint.get("assay_type"),
        "assay_test_type": endpoint.get("assay_test_type"),
        "assay_category": endpoint.get("assay_category"),
        "confidence_score": endpoint.get("confidence_score"),
        "relationship_type": endpoint.get("relationship_type"),
        "target_chembl_id": endpoint.get("target_chembl_id"),
        "target_name": endpoint.get("target_pref_name"),
        "target_type": endpoint.get("target_type"),
        "target_organism": endpoint.get("target_organism"),
        "doc_type": endpoint.get("doc_type"),
        "pubmed_id": endpoint.get("pubmed_id"),
        "doi": endpoint.get("doi"),
        "title": endpoint.get("title"),
        "label_type": "raw_std_delta",
        "value_version": row["value_version"],
        "molecule_a_chembl_id": row["molecule_a_chembl_id"],
        "molecule_b_chembl_id": row["molecule_b_chembl_id"],
        "activity_a": parse_float(row["raw_activity_a"]),
        "activity_b": parse_float(row["raw_activity_b"]),
        "abs_activity_delta": parse_float(row["abs_activity_delta"]),
        "normalized_delta": parse_float(row["normalized_delta"]),
        "direction": "molecule_a_to_molecule_b",
    }
    prompt_row = {
        "standard_type": row["standard_type"],
        "molecule_a_smiles": row["molecule_a_smiles"],
        "molecule_b_smiles": row["molecule_b_smiles"],
        "target_pref_name": endpoint.get("target_pref_name"),
    }
    return {
        "prompt": build_prompt(prompt_row, endpoint_for_prompt),
        "completion": LABEL_TO_COMPLETION[label],
        "metadata": metadata,
    }


def endpoint_prompt_context(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "standard_type": endpoint.get("standard_type"),
        "assay_type": endpoint.get("assay_type"),
        "assay_test_type": endpoint.get("assay_test_type"),
        "assay_category": endpoint.get("assay_category"),
        "description": endpoint.get("description"),
        "target_pref_name": endpoint.get("target_pref_name"),
        "target_type": endpoint.get("target_type"),
        "target_organism": endpoint.get("target_organism"),
        "confidence_score": endpoint.get("confidence_score"),
        "relationship_type": endpoint.get("relationship_type"),
    }


def build_run_id(args: argparse.Namespace) -> str:
    return (
        f"CHEMBL4513218_inhibition_full_range_raw_std_sim_le_0p5_diff_ge_1p5_"
        f"molecule_disjoint_{ratio_tag(args.ratios)}_n{args.n_pairs}"
    ).replace(".", "p")


def build_split_version(args: argparse.Namespace) -> str:
    return f"{build_run_id(args)}_sampled_from_exhaustive_compact"


def stable_text_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
