"""Export oral bioavailability benchmark splits for Hugging Face upload."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/chembl_tool/activity_transfer_benchmark")
TRANSFER_RUN_ID = "absolute_unspecified_systemic_directed_molecule_disjoint_pairratio_7_1_2_v1"
DIRECTION_RUN_ID = "absolute_unspecified_systemic_directed_molecule_disjoint_higher_lower_pairratio_7_1_2_v1"
SPLITS = ("train", "validation", "test")


DATASETS = {
    "transfer": {
        "repo_id": "Kiria-Nozan/Starling-bioavailability-transfer",
        "title": "Starling Oral Bioavailability Transfer",
        "run_id": TRANSFER_RUN_ID,
        "completion_a_label": "similar",
        "completion_b_label": "different",
        "task": "Decide whether oral bioavailability transfers between Molecule A and Molecule B.",
        "label_rule": (
            "similar if absolute oral bioavailability difference is <= 10 percentage points; "
            "different if absolute difference is >= 30 percentage points; ambiguous pairs are excluded."
        ),
    },
    "direction": {
        "repo_id": "Kiria-Nozan/Starling-bioavailability-direction",
        "title": "Starling Oral Bioavailability Direction",
        "run_id": DIRECTION_RUN_ID,
        "completion_a_label": "query_higher",
        "completion_b_label": "query_lower",
        "task": "Predict whether Molecule B has higher or lower oral bioavailability than Molecule A.",
        "label_rule": (
            "query_higher if Molecule B oral bioavailability is greater than Molecule A; "
            "query_lower if Molecule B is lower; exact ties are excluded."
        ),
    },
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    export_root = Path(args.export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        export_dataset(dataset_name, args, export_root)
    print(
        json.dumps(
            {
                "export_root": str(export_root),
                "datasets": args.datasets,
                "elapsed_s": round(time.time() - started, 3),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_ROOT / "oral_bioavailability_hf_pair_splits"))
    parser.add_argument("--export-root", default=str(DEFAULT_ROOT / "hf_upload_exports"))
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["transfer", "direction"])
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args(argv)


def export_dataset(dataset_name: str, args: argparse.Namespace, export_root: Path) -> None:
    spec = DATASETS[dataset_name]
    source_dir = Path(args.source_root) / str(spec["run_id"])
    out_dir = export_root / dataset_name
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    split_summaries: dict[str, Any] = {}

    for split in SPLITS:
        in_path = source_dir / f"{split}.jsonl"
        out_path = data_dir / f"{split}.jsonl.gz"
        split_summaries[split] = export_split(
            in_path=in_path,
            out_path=out_path,
            split=split,
            dataset_name=dataset_name,
            spec=spec,
            progress_every=args.progress_every,
            compresslevel=args.compression_level,
        )

    export_summary = {
        "dataset_name": dataset_name,
        "repo_id": spec["repo_id"],
        "source_run_id": spec["run_id"],
        "source_dir": str(source_dir),
        "created_at_unix": time.time(),
        "splits": split_summaries,
        "source_summary": summary,
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(source_dir / "summary.json", out_dir / "source_summary.json")
    write_readme(out_dir / "README.md", dataset_name, spec, export_summary)


def export_split(
    *,
    in_path: Path,
    out_path: Path,
    split: str,
    dataset_name: str,
    spec: dict[str, Any],
    progress_every: int,
    compresslevel: int,
) -> dict[str, Any]:
    started = time.time()
    n = 0
    completion_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    a_label = str(spec["completion_a_label"])
    b_label = str(spec["completion_b_label"])
    completion_to_label = {"A": a_label, "B": b_label}
    with in_path.open("r", encoding="utf-8") as src, gzip.open(
        out_path,
        "wt",
        encoding="utf-8",
        compresslevel=compresslevel,
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            completion = str(row.get("completion", ""))
            label_text = completion_to_label.get(completion, completion)
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "benchmark_name": f"starling_bioavailability_{dataset_name}",
                    "benchmark_version": str(spec["run_id"]),
                    "benchmark_split": split,
                    "source_dataset": "starling-labs/Oral_Bioavailability",
                    "source_dataset_split": "train",
                    "split_mode": "molecule_disjoint",
                    "molecule_disjoint": True,
                    "completion_a_label": a_label,
                    "completion_b_label": b_label,
                    "label_text": label_text,
                    "hf_repo_id": spec["repo_id"],
                }
            )
            row["metadata"] = metadata
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
            completion_counts[completion] = completion_counts.get(completion, 0) + 1
            label_counts[label_text] = label_counts.get(label_text, 0) + 1
            if progress_every > 0 and n % progress_every == 0:
                print(f"[{dataset_name}:{split}] exported {n:,} rows", flush=True)
    return {
        "rows": n,
        "completion_counts": completion_counts,
        "label_counts": label_counts,
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "elapsed_s": round(time.time() - started, 3),
    }


def write_readme(path: Path, dataset_name: str, spec: dict[str, Any], export_summary: dict[str, Any]) -> None:
    splits = export_summary["splits"]
    split_lines = "\n".join(
        f"| {split} | {info['rows']} | `{Path(info['path']).name}` | {info['label_counts']} |"
        for split, info in splits.items()
    )
    readme = f"""---
language:
- en
task_categories:
- text-classification
tags:
- chemistry
- admet
- oral-bioavailability
- molecular-property-prediction
- pairwise-comparison
pretty_name: {spec["title"]}
size_categories:
- 1M<n<10M
---

# {spec["title"]}

This dataset is a molecule-disjoint pairwise benchmark derived from
`starling-labs/Oral_Bioavailability`. Each row is a prompt/completion example with
metadata for audit and reproducibility.

## Task

{spec["task"]}

Completion labels:

- `A`: `{spec["completion_a_label"]}`
- `B`: `{spec["completion_b_label"]}`

Label rule: {spec["label_rule"]}

## Splits

| split | rows | file | label counts |
|---|---:|---|---|
{split_lines}

## Schema

Each line in `data/*.jsonl.gz` is a JSON object:

```json
{{
  "prompt": "string",
  "completion": "A or B",
  "metadata": {{
    "label_text": "string",
    "reference_activity_value": 0.0,
    "hidden_query_activity_value": 0.0,
    "reference_aggregate_id": "string",
    "query_aggregate_id": "string",
    "condition_key": "string",
    "weighted_tanimoto": 0.0
  }}
}}
```

The prompt hides Molecule B's oral bioavailability value. The hidden value and
the resolved label are kept in `metadata` for supervised training and auditing.

## Construction Notes

- Source dataset: `starling-labs/Oral_Bioavailability`, split `train`.
- Clean rows keep `absolute`, `unspecified`, and `systemic_availability` report types when the oral
  bioavailability value can be parsed as an explicit numeric percent.
- Split mode is molecule-disjoint by canonical SMILES. The same molecule does not appear across
  train, validation, and test.
- Experimental context fields are preserved in the prompt and metadata.
- Original local source run: `{spec["run_id"]}`.

## Citation / Attribution

Please cite or attribute the upstream `starling-labs/Oral_Bioavailability` dataset when using this benchmark.
"""
    path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
