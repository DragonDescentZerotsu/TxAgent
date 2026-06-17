"""Build exhaustive split-internal raw-value transfer pairs for one endpoint.

The output is compact TSV.GZ shards, not prompt/completion JSONL. It keeps the
same molecule-disjoint split policy as build_single_endpoint_raw_transfer_splits
and writes only non-ambiguous pairs:

similar:   abs(raw_a - raw_b) <= similar_std * sample_std
different: abs(raw_a - raw_b) >= different_std * sample_std
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sqlite3
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.activity_transfer_benchmark.build_single_endpoint_raw_transfer_splits import (
    DEFAULT_CHEMBL_SQLITE,
    DEFAULT_FPS_GZ,
    DEFAULT_OUT_ROOT,
    SPLITS,
    VERSION_FULL,
    MoleculeActivity,
    assign_molecule_splits,
    filter_version_molecules,
    load_endpoint_context,
    load_molecule_activities,
    parse_ratios,
    stable_text_int,
    summarize_values,
)
from tools.chembl_tool.activity_transfer_benchmark.run_benchmark import load_fingerprints


FIELDS = [
    "split",
    "label",
    "assay_chembl_id",
    "assay_id",
    "standard_type",
    "raw_units",
    "value_version",
    "molecule_a_chembl_id",
    "molecule_b_chembl_id",
    "molecule_a_smiles",
    "molecule_b_smiles",
    "raw_activity_a",
    "raw_activity_b",
    "abs_activity_delta",
    "normalized_delta",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    out_dir = Path(args.out_root) / build_run_id(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.chembl_sqlite)
    conn.row_factory = sqlite3.Row
    try:
        endpoint = load_endpoint_context(conn, args.assay_chembl_id, args.standard_type)
        raw_molecules = load_molecule_activities(conn, endpoint["assay_id"], endpoint["standard_type"])
    finally:
        conn.close()

    needed_ids = {molecule["molecule_chembl_id"] for molecule in raw_molecules}
    fingerprints = load_fingerprints(Path(args.fps_gz), needed_ids)
    molecules = [
        MoleculeActivity(
            molecule_chembl_id=molecule["molecule_chembl_id"],
            smiles=molecule["smiles"],
            raw_value=float(molecule["raw_value"]),
            n_records=int(molecule["n_records"]),
            fp_int=fingerprints[molecule["molecule_chembl_id"]][0],
            fp_popcount=fingerprints[molecule["molecule_chembl_id"]][1],
        )
        for molecule in raw_molecules
        if molecule["molecule_chembl_id"] in fingerprints
    ]
    version_molecules = filter_version_molecules(molecules, args.value_version)
    value_stats = summarize_values([molecule.raw_value for molecule in version_molecules])
    sample_std = float(value_stats["sample_std"])
    similar_delta = args.similar_std * sample_std
    different_delta = args.different_std * sample_std

    ratios = parse_ratios(args.ratios)
    split_seed = args.seed + stable_text_int(args.value_version)
    molecules_by_split = assign_molecule_splits(version_molecules, ratios=ratios, seed=split_seed)

    planned = {}
    for split, split_molecules in molecules_by_split.items():
        planned[split] = count_non_ambiguous_pairs(
            [molecule.raw_value for molecule in split_molecules],
            similar_delta=similar_delta,
            different_delta=different_delta,
        )
    if args.count_only:
        summary = build_summary(
            args=args,
            endpoint=endpoint,
            value_stats=value_stats,
            molecules_by_split=molecules_by_split,
            planned=planned,
            written={},
            elapsed_s=time.time() - started,
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    written = {}
    for split in SPLITS:
        if args.splits and split not in args.splits:
            continue
        written[split] = write_split_shards(
            split=split,
            molecules=molecules_by_split[split],
            endpoint=endpoint,
            value_version=args.value_version,
            sample_std=sample_std,
            similar_delta=similar_delta,
            different_delta=different_delta,
            out_dir=shard_dir,
            rows_per_shard=args.rows_per_shard,
            compresslevel=args.compresslevel,
            max_rows=args.max_rows_per_split,
            progress_every=args.progress_every,
        )

    summary = build_summary(
        args=args,
        endpoint=endpoint,
        value_stats=value_stats,
        molecules_by_split=molecules_by_split,
        planned=planned,
        written=written,
        elapsed_s=time.time() - started,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--fps-gz", default=DEFAULT_FPS_GZ)
    parser.add_argument("--out-root", default=f"{DEFAULT_OUT_ROOT}/exhaustive_compact")
    parser.add_argument("--assay-chembl-id", default="CHEMBL4513218")
    parser.add_argument("--standard-type", default="inhibition")
    parser.add_argument("--value-version", choices=[VERSION_FULL], default=VERSION_FULL)
    parser.add_argument("--ratios", default="0.7,0.1,0.2")
    parser.add_argument("--similar-std", type=float, default=0.5)
    parser.add_argument("--different-std", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--rows-per-shard", type=int, default=5_000_000)
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="Smoke-test cap; 0 means exhaustive.")
    parser.add_argument("--splits", nargs="*", default=[], choices=list(SPLITS), help="Optional subset of splits.")
    parser.add_argument("--count-only", action="store_true")
    return parser.parse_args(argv)


def count_non_ambiguous_pairs(
    values: list[float],
    *,
    similar_delta: float,
    different_delta: float,
) -> dict[str, int]:
    values = sorted(values)
    n = len(values)
    total = n * (n - 1) // 2
    similar = 0
    different = 0
    for i, value in enumerate(values):
        similar += max(0, bisect_right(values, value + similar_delta) - i - 1)
        different += max(0, n - bisect_left(values, value + different_delta))
    return {
        "n_molecules": n,
        "total_pairs": total,
        "similar_pairs": similar,
        "different_pairs": different,
        "non_ambiguous_pairs": similar + different,
        "ambiguous_pairs": total - similar - different,
    }


def write_split_shards(
    *,
    split: str,
    molecules: list[MoleculeActivity],
    endpoint: dict[str, Any],
    value_version: str,
    sample_std: float,
    similar_delta: float,
    different_delta: float,
    out_dir: Path,
    rows_per_shard: int,
    compresslevel: int,
    max_rows: int,
    progress_every: int,
) -> dict[str, Any]:
    sorted_molecules = sorted(molecules, key=lambda molecule: molecule.raw_value)
    values = [molecule.raw_value for molecule in sorted_molecules]
    writer = ShardWriter(
        out_dir=out_dir,
        split=split,
        rows_per_shard=rows_per_shard,
        compresslevel=compresslevel,
    )
    started = time.time()
    try:
        for row in iter_similar_pairs(
            split=split,
            molecules=sorted_molecules,
            values=values,
            endpoint=endpoint,
            value_version=value_version,
            sample_std=sample_std,
            similar_delta=similar_delta,
        ):
            writer.write(row)
            if should_stop_or_report(writer.rows_written, max_rows, progress_every, split, started, "similar"):
                if max_rows and writer.rows_written >= max_rows:
                    break
        if not max_rows or writer.rows_written < max_rows:
            for row in iter_different_pairs(
                split=split,
                molecules=sorted_molecules,
                values=values,
                endpoint=endpoint,
                value_version=value_version,
                sample_std=sample_std,
                different_delta=different_delta,
            ):
                writer.write(row)
                if should_stop_or_report(writer.rows_written, max_rows, progress_every, split, started, "different"):
                    if max_rows and writer.rows_written >= max_rows:
                        break
    finally:
        writer.close()
    return {
        "rows_written": writer.rows_written,
        "shards": [str(path) for path in writer.shards],
        "elapsed_s": round(time.time() - started, 3),
    }


def iter_similar_pairs(
    *,
    split: str,
    molecules: list[MoleculeActivity],
    values: list[float],
    endpoint: dict[str, Any],
    value_version: str,
    sample_std: float,
    similar_delta: float,
) -> Iterable[list[Any]]:
    for i, mol_a in enumerate(molecules[:-1]):
        end = bisect_right(values, mol_a.raw_value + similar_delta)
        for j in range(i + 1, end):
            yield pair_row(split, "similar", mol_a, molecules[j], endpoint, value_version, sample_std)


def iter_different_pairs(
    *,
    split: str,
    molecules: list[MoleculeActivity],
    values: list[float],
    endpoint: dict[str, Any],
    value_version: str,
    sample_std: float,
    different_delta: float,
) -> Iterable[list[Any]]:
    n = len(molecules)
    for i, mol_a in enumerate(molecules[:-1]):
        start = bisect_left(values, mol_a.raw_value + different_delta)
        for j in range(start, n):
            yield pair_row(split, "different", mol_a, molecules[j], endpoint, value_version, sample_std)


def pair_row(
    split: str,
    label: str,
    mol_a: MoleculeActivity,
    mol_b: MoleculeActivity,
    endpoint: dict[str, Any],
    value_version: str,
    sample_std: float,
) -> list[Any]:
    delta = abs(mol_a.raw_value - mol_b.raw_value)
    return [
        split,
        label,
        endpoint["assay_chembl_id"],
        endpoint["assay_id"],
        endpoint["standard_type"],
        endpoint.get("raw_units") or "",
        value_version,
        mol_a.molecule_chembl_id,
        mol_b.molecule_chembl_id,
        mol_a.smiles,
        mol_b.smiles,
        format_float(mol_a.raw_value),
        format_float(mol_b.raw_value),
        format_float(delta),
        format_float(delta / sample_std),
    ]


class ShardWriter:
    def __init__(self, *, out_dir: Path, split: str, rows_per_shard: int, compresslevel: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.rows_per_shard = rows_per_shard
        self.compresslevel = compresslevel
        self.rows_written = 0
        self.shard_index = 0
        self.handle = None
        self.writer = None
        self.shards: list[Path] = []

    def write(self, row: list[Any]) -> None:
        if self.handle is None or self.rows_written % self.rows_per_shard == 0:
            self.open_next()
        assert self.writer is not None
        self.writer.writerow(row)
        self.rows_written += 1

    def open_next(self) -> None:
        self.close()
        self.shard_index += 1
        path = self.out_dir / f"{self.split}.part{self.shard_index:05d}.tsv.gz"
        self.shards.append(path)
        self.handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=self.compresslevel)
        self.writer = csv.writer(self.handle, delimiter="\t", lineterminator="\n")
        self.writer.writerow(FIELDS)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.writer = None


def should_stop_or_report(
    rows_written: int,
    max_rows: int,
    progress_every: int,
    split: str,
    started: float,
    phase: str,
) -> bool:
    if progress_every > 0 and rows_written > 0 and rows_written % progress_every == 0:
        elapsed = max(time.time() - started, 1e-9)
        print(
            json.dumps(
                {
                    "stage": "write_exhaustive_pairs",
                    "split": split,
                    "phase": phase,
                    "rows_written": rows_written,
                    "rate_per_s": round(rows_written / elapsed, 2),
                    "elapsed_s": round(elapsed, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return bool(max_rows and rows_written >= max_rows)


def build_summary(
    *,
    args: argparse.Namespace,
    endpoint: dict[str, Any],
    value_stats: dict[str, float | int],
    molecules_by_split: dict[str, list[MoleculeActivity]],
    planned: dict[str, dict[str, int]],
    written: dict[str, dict[str, Any]],
    elapsed_s: float,
) -> dict[str, Any]:
    totals = {
        key: sum(row[key] for row in planned.values())
        for key in ("total_pairs", "similar_pairs", "different_pairs", "non_ambiguous_pairs", "ambiguous_pairs")
    }
    return {
        "endpoint": endpoint,
        "parameters": vars(args),
        "value_stats": value_stats,
        "similar_delta": args.similar_std * float(value_stats["sample_std"]),
        "different_delta": args.different_std * float(value_stats["sample_std"]),
        "molecule_split_counts": {split: len(molecules_by_split[split]) for split in SPLITS},
        "planned_split_counts": planned,
        "planned_totals": totals,
        "written": written,
        "elapsed_s": round(elapsed_s, 3),
    }


def build_run_id(args: argparse.Namespace) -> str:
    return (
        f"{args.assay_chembl_id}_{args.standard_type}_{args.value_version}_"
        f"raw_std_sim_le_{args.similar_std:g}_diff_ge_{args.different_std:g}_"
        f"molecule_disjoint_{ratio_tag(args.ratios)}_exhaustive_nonambig_compact"
    ).replace(".", "p").replace("/", "_")


def ratio_tag(raw: str) -> str:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if values == [0.7, 0.1, 0.2]:
        return "7_1_2"
    return "_".join(f"{value:g}".replace(".", "p") for value in values)


def format_float(value: float) -> str:
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
