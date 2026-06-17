"""Export clean numeric oral bioavailability data for Hugging Face upload."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/chembl_tool/activity_transfer_benchmark")
DEFAULT_SOURCE_RUN_ID = "absolute_unspecified_systemic_broad_condition_full_text_v1"
DEFAULT_REPO_ID = "Kiria-Nozan/Starling-bioavailability-clean"


TABLES = {
    "aggregate_molecules": {
        "source_file": "aggregate_molecules.jsonl",
        "description": "One row per canonical molecule SMILES and broad experimental condition.",
    },
    "molecule_records": {
        "source_file": "molecule_records.jsonl",
        "description": "One row per original source row that passed numeric parsing and SMILES validation.",
    },
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.export_root) / "clean"
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    table_summaries: dict[str, Any] = {}
    for table_name, spec in TABLES.items():
        in_path = source_dir / str(spec["source_file"])
        out_path = data_dir / f"{table_name}.jsonl.gz"
        table_summaries[table_name] = export_table(
            table_name=table_name,
            in_path=in_path,
            out_path=out_path,
            source_run_id=args.source_run_id,
            repo_id=args.repo_id,
            progress_every=args.progress_every,
            compresslevel=args.compression_level,
        )

    copy_if_exists(source_dir / "summary.json", out_dir / "source_summary.json")
    copy_if_exists(source_dir / "report_zh.md", out_dir / "source_report_zh.md")
    copy_if_exists(source_dir / "figures" / "value_distribution.svg", out_dir / "figures" / "value_distribution.svg")
    copy_if_exists(source_dir / "figures" / "value_distribution.png", out_dir / "figures" / "value_distribution.png")

    export_summary = {
        "repo_id": args.repo_id,
        "source_dataset": "starling-labs/Oral_Bioavailability",
        "source_dataset_split": "train",
        "source_dataset_sha": args.source_dataset_sha,
        "source_license_note": "No explicit license field was detected in the upstream dataset metadata at export time.",
        "source_run_id": args.source_run_id,
        "source_dir": str(source_dir),
        "created_at_unix": time.time(),
        "elapsed_s": round(time.time() - started, 3),
        "tables": table_summaries,
        "source_summary": source_summary,
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(out_dir / "README.md", export_summary)
    print(json.dumps({"export_dir": str(out_dir), "tables": table_summaries}, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_source_dir = (
        DEFAULT_ROOT / "oral_bioavailability_hf" / DEFAULT_SOURCE_RUN_ID
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--source-dir", default=str(default_source_dir))
    parser.add_argument("--export-root", default=str(DEFAULT_ROOT / "hf_upload_exports"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--source-dataset-sha",
        default="0f011349db5b47437a933e8681be25e74a97c0f7",
        help="Upstream starling-labs/Oral_Bioavailability commit observed at export time.",
    )
    parser.add_argument("--progress-every", type=int, default=25_000)
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args(argv)


def export_table(
    *,
    table_name: str,
    in_path: Path,
    out_path: Path,
    source_run_id: str,
    repo_id: str,
    progress_every: int,
    compresslevel: int,
) -> dict[str, Any]:
    started = time.time()
    n = 0
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
            row["benchmark_metadata"] = {
                "hf_repo_id": repo_id,
                "table_name": table_name,
                "source_dataset": "starling-labs/Oral_Bioavailability",
                "source_dataset_split": "train",
                "source_run_id": source_run_id,
                "cleaning_version": source_run_id,
                "value_unit": "percent",
            }
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
            if progress_every > 0 and n % progress_every == 0:
                print(f"[{table_name}] exported {n:,} rows", flush=True)
    return {
        "rows": n,
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "description": TABLES[table_name]["description"],
        "elapsed_s": round(time.time() - started, 3),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_readme(path: Path, export_summary: dict[str, Any]) -> None:
    source = export_summary["source_summary"]
    tables = export_summary["tables"]
    table_lines = "\n".join(
        f"| {name} | {info['rows']} | `{Path(info['path']).name}` | {info['description']} |"
        for name, info in tables.items()
    )
    parse_methods = source.get("parse_method_counts", {})
    drop_reasons = source.get("drop_reasons", {})
    value_summary = source.get("value_percent_summary", {})
    readme = f"""---
language:
- en
tags:
- chemistry
- admet
- oral-bioavailability
- molecular-property-prediction
- clean-data
pretty_name: Starling Oral Bioavailability Clean Numeric
size_categories:
- 10K<n<100K
---

# Starling Oral Bioavailability Clean Numeric

This is a cleaned numeric version of `starling-labs/Oral_Bioavailability`.
It keeps rows whose oral bioavailability value can be parsed into an explicit
numeric percent and whose SMILES can be validated. It is intended as a reusable
base table for molecular property modeling, benchmark construction, and audit.

## Tables

| table | rows | file | description |
|---|---:|---|---|
{table_lines}

Recommended default table: `aggregate_molecules`, where repeated source rows for
the same molecule and broad experimental condition are aggregated.

## Important Fields

- `smiles`: canonical molecule SMILES used by the benchmark pipeline.
- `oral_bioavailability_value_percent`: cleaned numeric oral bioavailability value.
- `condition_text`: all experimental condition fields rendered as text.
- `condition_key`: broad condition key used for grouping compatible observations.
- `parse_method` / `parse_modifier`: how the original value string was converted.
- `metadata`: original upstream fields, including `pmid`, `support_text`, molecule name,
  original value string, report type, condition columns, and source SMILES.

`molecule_records` is line-level provenance. `aggregate_molecules` contains
`source_indices` and a `metadata` list pointing back to contributing source rows.

## Cleaning Summary

- Source dataset: `starling-labs/Oral_Bioavailability`, split `train`
- Upstream commit observed at export time: `{export_summary["source_dataset_sha"]}`
- Source rows: {source.get("source_rows")}
- Clean numeric rows: {source.get("clean_rows")}
- Aggregate molecule-condition rows: {source.get("aggregate_molecules")}
- Allowed report types: `absolute`, `unspecified`, `systemic_availability`
- Value unit: percent oral bioavailability

Value distribution:

```text
min={value_summary.get("min")}
median={value_summary.get("median")}
mean={value_summary.get("mean")}
max={value_summary.get("max")}
std={value_summary.get("std")}
```

Parse methods:

```text
{json.dumps(parse_methods, ensure_ascii=False)}
```

Drop reasons:

```text
{json.dumps(drop_reasons, ensure_ascii=False)}
```

## Parsing Rules

- `about`, `~`, and approximately values keep the main numeric value.
- `per cent` and `percent` are normalized to `%`.
- explicit mean/average/median values are preferred when marked.
- `x ± y` keeps `x`.
- ranges such as `x to y` use the midpoint.
- values without `%` in `[0, 1.5]` are treated as fractions and multiplied by 100.
- AUC-only, fold-change, higher/lower/comparable relative descriptions are dropped.
- Values outside `[0, 1000]` percent are dropped.

## License and Attribution

This is a derived dataset from `starling-labs/Oral_Bioavailability`. No explicit
license field was detected in the upstream Hugging Face dataset metadata at export
time. Please check the upstream dataset page and cite/attribute the original source
when using this derived clean table.
"""
    path.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
