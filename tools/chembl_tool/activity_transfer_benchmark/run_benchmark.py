"""Build a ChEMBL assay-activity transferability benchmark.

The benchmark asks a deliberately simple first-version question:

Given two molecules measured in the same assay endpoint, how well can a
structural Tanimoto threshold predict whether their measured activity is
similar enough to transfer?

The main continuous endpoint uses ChEMBL pchembl_value. Conservative binary
activity_comment handling is included as a separate secondary analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import os
import random
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DEFAULT_CHEMBL_SQLITE = "tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db"
DEFAULT_FPS_GZ = "tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark"

SIMILARITY_BUCKETS = [
    ("very_close_analog", 0.95, 1.01),
    ("close_analog", 0.80, 0.95),
    ("moderate_analog", 0.60, 0.80),
    ("weak_analog", 0.40, 0.60),
    ("distant_analog", 0.20, 0.40),
    ("very_distant_analog", -0.01, 0.20),
]

DEFAULT_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

ACTIVE_COMMENTS = {
    "active",
    "active compound",
    "active compounds",
}

INACTIVE_COMMENTS = {
    "inactive",
    "inactive compound",
    "inactive compounds",
    "not active",
    "no activity",
    "no inhibition",
}


@dataclass(frozen=True)
class EndpointMeta:
    assay_id: int
    assay_chembl_id: str
    standard_type: str
    n_molecules_sql: int
    n_records_sql: int
    assay_type: str | None
    assay_test_type: str | None
    assay_category: str | None
    assay_organism: str | None
    confidence_score: int | None
    relationship_type: str | None
    target_chembl_id: str | None
    target_pref_name: str | None
    target_type: str | None
    target_organism: str | None
    description: str | None
    pchembl_min: float = 0.0
    pchembl_q25: float = 0.0
    pchembl_median: float = 0.0
    pchembl_q75: float = 0.0
    pchembl_max: float = 0.0
    pchembl_range: float = 0.0
    pchembl_iqr: float = 0.0


@dataclass(frozen=True)
class GroupWorkItem:
    assay_id: int
    assay_chembl_id: str
    standard_type: str
    target_chembl_id: str | None
    target_pref_name: str | None
    n_molecules_sql: int
    n_molecules_with_fp: int
    pair_budget: int
    values_are_binary: bool
    molecules: tuple[tuple[str, str, float | int, int, int], ...]
    seed: int


class Confusion:
    __slots__ = ("tp", "fp", "tn", "fn")

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, truth_similar: bool, pred_similar: bool) -> None:
        if truth_similar and pred_similar:
            self.tp += 1
        elif truth_similar and not pred_similar:
            self.fn += 1
        elif not truth_similar and pred_similar:
            self.fp += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, float | int]:
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        total = tp + fp + tn + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        neg_precision = _safe_div(tn, tn + fn)
        neg_recall = specificity
        neg_f1 = _safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall)
        return {
            "n": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": _safe_div(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_similar": f1,
            "f1_different": neg_f1,
            "macro_f1": (f1 + neg_f1) / 2.0,
            "balanced_accuracy": (recall + specificity) / 2.0,
        }


def main(argv: list[str] | None = None) -> int:
    global CURRENT_SIMILAR_DELTA, CURRENT_DIFFERENT_DELTA
    args = parse_args(argv)
    CURRENT_SIMILAR_DELTA = args.similar_delta
    CURRENT_DIFFERENT_DELTA = args.different_delta
    started = time.monotonic()
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    log(f"output directory: {out_dir}")
    log("opening ChEMBL SQLite")
    conn = connect_sqlite(args.chembl_sqlite)
    try:
        endpoint_meta = load_continuous_endpoint_meta(conn, args)
        if args.assay_endpoint_limit:
            endpoint_meta = dict(list(endpoint_meta.items())[: args.assay_endpoint_limit])
        log(f"continuous assay-endpoints retained by SQL filter: {len(endpoint_meta):,}")
        continuous_groups = load_continuous_groups(conn, endpoint_meta, args)
        continuous_groups, endpoint_meta, dynamic_filter_summary = apply_dynamic_range_filter(
            continuous_groups,
            endpoint_meta,
            args,
        )
        if dynamic_filter_summary["filter_enabled"]:
            log(
                "dynamic range filter retained "
                f"{dynamic_filter_summary['retained_assay_endpoints']:,}/"
                f"{dynamic_filter_summary['input_assay_endpoints']:,} assay-endpoints "
                f"(min_range={args.min_pchembl_range:g}, min_iqr={args.min_pchembl_iqr:g})"
            )
    finally:
        conn.close()

    needed_ids = {
        molecule[0]
        for molecules in continuous_groups.values()
        for molecule in molecules
    }
    log(f"unique molecules needing fingerprints: {len(needed_ids):,}")
    fingerprints = load_fingerprints(Path(args.fps_gz), needed_ids)
    log(f"fingerprints loaded for retained molecules: {len(fingerprints):,}")

    continuous_items, continuous_endpoint_rows = prepare_continuous_work_items(
        continuous_groups,
        endpoint_meta,
        fingerprints,
        args,
    )
    log(f"continuous assay-endpoints with enough fingerprints: {len(continuous_items):,}")
    log(f"continuous pair budget: {sum(item.pair_budget for item in continuous_items):,}")

    continuous_paths = OutputPaths(
        pairs=out_dir / "continuous_pairs.tsv.gz",
        endpoint_summary=out_dir / "continuous_assay_endpoint_summary.tsv",
        threshold_metrics=out_dir / "continuous_threshold_metrics.tsv",
        bucket_summary=out_dir / "continuous_similarity_bucket_summary.tsv",
        enrichment_detail=out_dir / "continuous_assay_bucket_enrichment.tsv",
        enrichment_summary=out_dir / "continuous_enrichment_summary.tsv",
    )
    write_endpoint_summary(continuous_paths.endpoint_summary, continuous_endpoint_rows)
    continuous_result = run_pair_benchmark(
        continuous_items,
        thresholds,
        continuous_paths,
        args.workers,
        args.progress_every_groups,
        similar_label="similar",
        different_label="different",
    )

    binary_result = None
    binary_paths = None
    binary_summary = {"status": "skipped"}
    if not args.skip_binary:
        binary_result, binary_paths, binary_summary = run_binary_analysis(args, thresholds, out_dir, fingerprints)

    summary = {
        "run_id": run_id,
        "elapsed_s": round(time.monotonic() - started, 3),
        "chembl_sqlite": args.chembl_sqlite,
        "fps_gz": args.fps_gz,
        "continuous": continuous_result,
        "dynamic_range_filter": dynamic_filter_summary,
        "binary": binary_summary,
        "parameters": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    write_figures(
        continuous_result,
        continuous_paths,
        figures_dir,
        title_prefix="Continuous pChEMBL",
    )
    if binary_result and binary_paths:
        write_figures(binary_result, binary_paths, figures_dir, title_prefix="Binary comments", stem_prefix="binary_")

    write_report(out_dir / "report_zh.md", summary, continuous_paths, binary_paths, figures_dir)
    log(f"finished in {format_elapsed(time.monotonic() - started)}")
    log(f"report: {out_dir / 'report_zh.md'}")
    return 0


@dataclass(frozen=True)
class OutputPaths:
    pairs: Path
    endpoint_summary: Path
    threshold_metrics: Path
    bucket_summary: Path
    enrichment_detail: Path
    enrichment_summary: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--fps-gz", default=DEFAULT_FPS_GZ)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--min-molecules-per-assay", type=int, default=20)
    parser.add_argument("--min-molecules-with-fp", type=int, default=20)
    parser.add_argument("--max-pairs-per-assay", type=int, default=5000)
    parser.add_argument("--max-total-pairs", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--similar-delta", type=float, default=0.5)
    parser.add_argument("--different-delta", type=float, default=1.0)
    parser.add_argument("--min-pchembl-range", type=float, default=0.0, help="Keep continuous assay-endpoints with molecule-level pChEMBL max-min at least this value.")
    parser.add_argument("--min-pchembl-iqr", type=float, default=0.0, help="Keep continuous assay-endpoints with molecule-level pChEMBL Q75-Q25 at least this value.")
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--assay-endpoint-limit", type=int, default=0, help="Smoke-test limit after SQL filtering.")
    parser.add_argument("--progress-every-groups", type=int, default=250)
    parser.add_argument("--include-invalid-activities", action="store_true")
    parser.add_argument("--include-potential-duplicates", action="store_true")
    parser.add_argument("--skip-binary", action="store_true")
    parser.add_argument("--binary-max-total-pairs", type=int, default=500_000)
    return parser.parse_args(argv)


def parse_thresholds(raw: str) -> list[float]:
    thresholds = sorted({round(float(item.strip()), 6) for item in raw.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def load_continuous_endpoint_meta(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[tuple[int, str], EndpointMeta]:
    validity_filter = "" if args.include_invalid_activities else "AND act.data_validity_comment IS NULL"
    duplicate_filter = "" if args.include_potential_duplicates else "AND COALESCE(act.potential_duplicate, 0) = 0"
    query = f"""
        SELECT
          act.assay_id AS assay_id,
          a.chembl_id AS assay_chembl_id,
          LOWER(TRIM(act.standard_type)) AS standard_type,
          COUNT(DISTINCT act.molregno) AS n_molecules,
          COUNT(*) AS n_records,
          a.assay_type,
          a.assay_test_type,
          a.assay_category,
          a.assay_organism,
          a.confidence_score,
          a.relationship_type,
          a.description,
          td.chembl_id AS target_chembl_id,
          td.pref_name AS target_pref_name,
          td.target_type,
          td.organism AS target_organism
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        LEFT JOIN target_dictionary td ON a.tid = td.tid
        WHERE act.pchembl_value IS NOT NULL
          AND act.standard_relation = '='
          AND act.standard_flag = 1
          AND act.molregno IS NOT NULL
          AND act.standard_type IS NOT NULL
          {validity_filter}
          {duplicate_filter}
        GROUP BY act.assay_id, LOWER(TRIM(act.standard_type))
        HAVING n_molecules >= ?
        ORDER BY n_molecules DESC, n_records DESC
    """
    meta: dict[tuple[int, str], EndpointMeta] = {}
    for row in conn.execute(query, (args.min_molecules_per_assay,)):
        standard_type = str(row["standard_type"] or "").strip().lower()
        if not standard_type:
            continue
        key = (int(row["assay_id"]), standard_type)
        meta[key] = EndpointMeta(
            assay_id=int(row["assay_id"]),
            assay_chembl_id=str(row["assay_chembl_id"]),
            standard_type=standard_type,
            n_molecules_sql=int(row["n_molecules"] or 0),
            n_records_sql=int(row["n_records"] or 0),
            assay_type=none_or_str(row["assay_type"]),
            assay_test_type=none_or_str(row["assay_test_type"]),
            assay_category=none_or_str(row["assay_category"]),
            assay_organism=none_or_str(row["assay_organism"]),
            confidence_score=none_or_int(row["confidence_score"]),
            relationship_type=none_or_str(row["relationship_type"]),
            target_chembl_id=none_or_str(row["target_chembl_id"]),
            target_pref_name=none_or_str(row["target_pref_name"]),
            target_type=none_or_str(row["target_type"]),
            target_organism=none_or_str(row["target_organism"]),
            description=none_or_str(row["description"]),
        )
    return meta


def load_continuous_groups(
    conn: sqlite3.Connection,
    endpoint_meta: dict[tuple[int, str], EndpointMeta],
    args: argparse.Namespace,
) -> dict[tuple[int, str], list[tuple[str, str, float, int]]]:
    if not endpoint_meta:
        return {}
    selected_assay_ids = sorted({key[0] for key in endpoint_meta})
    selected_keys = set(endpoint_meta)
    validity_filter = "" if args.include_invalid_activities else "AND act.data_validity_comment IS NULL"
    duplicate_filter = "" if args.include_potential_duplicates else "AND COALESCE(act.potential_duplicate, 0) = 0"
    groups: dict[tuple[int, str], list[tuple[str, str, float, int]]] = defaultdict(list)
    accumulator: dict[tuple[tuple[int, str], int], list[float | int]] = {}
    started = time.monotonic()
    for index, chunk in enumerate(chunks(selected_assay_ids, 500), start=1):
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
            SELECT
              act.assay_id AS assay_id,
              LOWER(TRIM(act.standard_type)) AS standard_type,
              act.molregno AS molregno,
              CAST(act.pchembl_value AS REAL) AS pchembl_value
            FROM activities act
            WHERE act.assay_id IN ({placeholders})
              AND act.pchembl_value IS NOT NULL
              AND act.standard_relation = '='
              AND act.standard_flag = 1
              AND act.molregno IS NOT NULL
              AND act.standard_type IS NOT NULL
              {validity_filter}
              {duplicate_filter}
        """
        for row in conn.execute(query, chunk):
            key = (int(row["assay_id"]), str(row["standard_type"]).strip().lower())
            if key not in selected_keys:
                continue
            molregno = row["molregno"]
            pchembl = row["pchembl_value"]
            if molregno is not None and pchembl is not None:
                molecule_key = (key, int(molregno))
                entry = accumulator.get(molecule_key)
                if entry is None:
                    accumulator[molecule_key] = [float(pchembl), 1]
                else:
                    entry[0] = float(entry[0]) + float(pchembl)
                    entry[1] = int(entry[1]) + 1
        if index % 20 == 0:
            log(
                "loaded continuous molecule activities "
                f"chunks={index:,}/{math.ceil(len(selected_assay_ids) / 500):,} "
                f"molecule_endpoint_values={len(accumulator):,} elapsed={format_elapsed(time.monotonic() - started)}"
            )
    molregnos = sorted({molregno for _key, molregno in accumulator})
    log(f"loading molecule IDs and SMILES for continuous values: molregnos={len(molregnos):,}")
    molecule_lookup = load_molecule_lookup(conn, molregnos)
    for (key, molregno), (value_sum, n_records) in accumulator.items():
        lookup = molecule_lookup.get(molregno)
        if lookup is None:
            continue
        molecule_id, smiles = lookup
        groups[key].append((molecule_id, smiles, float(value_sum) / int(n_records), int(n_records)))
    return groups


def apply_dynamic_range_filter(
    groups: dict[tuple[int, str], list[tuple[str, str, float, int]]],
    meta: dict[tuple[int, str], EndpointMeta],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, str], list[tuple[str, str, float, int]]], dict[tuple[int, str], EndpointMeta], dict[str, Any]]:
    min_range = float(args.min_pchembl_range or 0.0)
    min_iqr = float(args.min_pchembl_iqr or 0.0)
    filter_enabled = min_range > 0.0 or min_iqr > 0.0
    filtered_groups: dict[tuple[int, str], list[tuple[str, str, float, int]]] = {}
    updated_meta: dict[tuple[int, str], EndpointMeta] = {}
    input_count = len(groups)
    removed_low_range = 0
    removed_low_iqr = 0
    removed_empty = 0

    for key, molecules in groups.items():
        endpoint = meta[key]
        values = sorted(float(molecule[2]) for molecule in molecules)
        if not values:
            removed_empty += 1
            continue
        q25 = quantile(values, 0.25)
        q50 = quantile(values, 0.50)
        q75 = quantile(values, 0.75)
        pchembl_min = values[0]
        pchembl_max = values[-1]
        pchembl_range = pchembl_max - pchembl_min
        pchembl_iqr = q75 - q25
        endpoint_with_stats = endpoint_with_dynamic_stats(
            endpoint,
            pchembl_min=pchembl_min,
            pchembl_q25=q25,
            pchembl_median=q50,
            pchembl_q75=q75,
            pchembl_max=pchembl_max,
            pchembl_range=pchembl_range,
            pchembl_iqr=pchembl_iqr,
        )
        low_range = pchembl_range < min_range
        low_iqr = pchembl_iqr < min_iqr
        if low_range:
            removed_low_range += 1
        if low_iqr:
            removed_low_iqr += 1
        if filter_enabled and (low_range or low_iqr):
            continue
        filtered_groups[key] = molecules
        updated_meta[key] = endpoint_with_stats

    summary = {
        "filter_enabled": filter_enabled,
        "min_pchembl_range": min_range,
        "min_pchembl_iqr": min_iqr,
        "input_assay_endpoints": input_count,
        "retained_assay_endpoints": len(filtered_groups),
        "removed_assay_endpoints": input_count - len(filtered_groups),
        "removed_empty": removed_empty,
        "below_min_pchembl_range": removed_low_range,
        "below_min_pchembl_iqr": removed_low_iqr,
    }
    return filtered_groups, updated_meta, summary


def endpoint_with_dynamic_stats(endpoint: EndpointMeta, **stats: float) -> EndpointMeta:
    return EndpointMeta(
        assay_id=endpoint.assay_id,
        assay_chembl_id=endpoint.assay_chembl_id,
        standard_type=endpoint.standard_type,
        n_molecules_sql=endpoint.n_molecules_sql,
        n_records_sql=endpoint.n_records_sql,
        assay_type=endpoint.assay_type,
        assay_test_type=endpoint.assay_test_type,
        assay_category=endpoint.assay_category,
        assay_organism=endpoint.assay_organism,
        confidence_score=endpoint.confidence_score,
        relationship_type=endpoint.relationship_type,
        target_chembl_id=endpoint.target_chembl_id,
        target_pref_name=endpoint.target_pref_name,
        target_type=endpoint.target_type,
        target_organism=endpoint.target_organism,
        description=endpoint.description,
        pchembl_min=float(stats["pchembl_min"]),
        pchembl_q25=float(stats["pchembl_q25"]),
        pchembl_median=float(stats["pchembl_median"]),
        pchembl_q75=float(stats["pchembl_q75"]),
        pchembl_max=float(stats["pchembl_max"]),
        pchembl_range=float(stats["pchembl_range"]),
        pchembl_iqr=float(stats["pchembl_iqr"]),
    )


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def load_molecule_lookup(conn: sqlite3.Connection, molregnos: list[int]) -> dict[int, tuple[str, str]]:
    lookup: dict[int, tuple[str, str]] = {}
    for chunk in chunks(molregnos, 1000):
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
            SELECT
              md.molregno AS molregno,
              md.chembl_id AS molecule_chembl_id,
              cs.canonical_smiles AS canonical_smiles
            FROM molecule_dictionary md
            JOIN compound_structures cs ON md.molregno = cs.molregno
            WHERE md.molregno IN ({placeholders})
        """
        for row in conn.execute(query, chunk):
            if row["molecule_chembl_id"] and row["canonical_smiles"]:
                lookup[int(row["molregno"])] = (str(row["molecule_chembl_id"]), str(row["canonical_smiles"]))
    return lookup


def load_fingerprints(fps_path: Path, needed_ids: set[str]) -> dict[str, tuple[int, int]]:
    if not needed_ids:
        return {}
    found: dict[str, tuple[int, int]] = {}
    started = time.monotonic()
    scanned = 0
    with gzip.open(fps_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            scanned += 1
            try:
                fp_hex, chembl_id = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            if chembl_id in needed_ids:
                fp_int = int(fp_hex, 16)
                found[chembl_id] = (fp_int, fp_int.bit_count())
                if len(found) == len(needed_ids):
                    break
            if scanned % 1_000_000 == 0:
                log(
                    f"fingerprint scan scanned={scanned:,} found={len(found):,}/{len(needed_ids):,} "
                    f"elapsed={format_elapsed(time.monotonic() - started)}"
                )
    return found


def prepare_continuous_work_items(
    groups: dict[tuple[int, str], list[tuple[str, str, float, int]]],
    meta: dict[tuple[int, str], EndpointMeta],
    fingerprints: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[GroupWorkItem], list[dict[str, Any]]]:
    raw_items: list[GroupWorkItem] = []
    endpoint_rows: list[dict[str, Any]] = []
    for key, molecules in groups.items():
        endpoint = meta[key]
        with_fp = []
        for molecule_id, smiles, pchembl, _n_records in molecules:
            fp = fingerprints.get(molecule_id)
            if fp is None:
                continue
            with_fp.append((molecule_id, smiles, pchembl, fp[0], fp[1]))
        n = len(with_fp)
        total_pairs = n * (n - 1) // 2
        row = endpoint_summary_row(endpoint, len(molecules), n, total_pairs)
        endpoint_rows.append(row)
        if n < args.min_molecules_with_fp:
            continue
        budget = min(args.max_pairs_per_assay, total_pairs)
        if budget <= 0:
            continue
        raw_items.append(
            GroupWorkItem(
                assay_id=endpoint.assay_id,
                assay_chembl_id=endpoint.assay_chembl_id,
                standard_type=endpoint.standard_type,
                target_chembl_id=endpoint.target_chembl_id,
                target_pref_name=endpoint.target_pref_name,
                n_molecules_sql=endpoint.n_molecules_sql,
                n_molecules_with_fp=n,
                pair_budget=budget,
                values_are_binary=False,
                molecules=tuple(with_fp),
                seed=args.seed + stable_int(endpoint.assay_chembl_id + endpoint.standard_type),
            )
        )
    items = allocate_pair_budgets(raw_items, args.max_total_pairs)
    endpoint_rows.sort(key=lambda row: (-int(row["n_molecules_with_fp"]), str(row["assay_chembl_id"]), str(row["standard_type"])))
    return items, endpoint_rows


def endpoint_summary_row(endpoint: EndpointMeta, n_loaded: int, n_with_fp: int, total_pairs: int) -> dict[str, Any]:
    return {
        "assay_chembl_id": endpoint.assay_chembl_id,
        "assay_id": endpoint.assay_id,
        "standard_type": endpoint.standard_type,
        "n_molecules_sql": endpoint.n_molecules_sql,
        "n_records_sql": endpoint.n_records_sql,
        "n_molecules_loaded": n_loaded,
        "n_molecules_with_fp": n_with_fp,
        "all_possible_pairs_with_fp": total_pairs,
        "pchembl_min": endpoint.pchembl_min,
        "pchembl_q25": endpoint.pchembl_q25,
        "pchembl_median": endpoint.pchembl_median,
        "pchembl_q75": endpoint.pchembl_q75,
        "pchembl_max": endpoint.pchembl_max,
        "pchembl_range": endpoint.pchembl_range,
        "pchembl_iqr": endpoint.pchembl_iqr,
        "assay_type": endpoint.assay_type,
        "assay_test_type": endpoint.assay_test_type,
        "assay_category": endpoint.assay_category,
        "confidence_score": endpoint.confidence_score,
        "relationship_type": endpoint.relationship_type,
        "target_chembl_id": endpoint.target_chembl_id,
        "target_pref_name": endpoint.target_pref_name,
        "target_type": endpoint.target_type,
        "target_organism": endpoint.target_organism,
        "description": endpoint.description,
    }


def allocate_pair_budgets(items: list[GroupWorkItem], max_total_pairs: int) -> list[GroupWorkItem]:
    total = sum(item.pair_budget for item in items)
    if total <= max_total_pairs:
        return items
    scale = max_total_pairs / total
    allocated: list[GroupWorkItem] = []
    used = 0
    for item in items:
        budget = max(1, int(item.pair_budget * scale))
        if used + budget > max_total_pairs:
            budget = max_total_pairs - used
        if budget <= 0:
            break
        allocated.append(replace_budget(item, budget))
        used += budget
    return allocated


def replace_budget(item: GroupWorkItem, budget: int) -> GroupWorkItem:
    return GroupWorkItem(
        assay_id=item.assay_id,
        assay_chembl_id=item.assay_chembl_id,
        standard_type=item.standard_type,
        target_chembl_id=item.target_chembl_id,
        target_pref_name=item.target_pref_name,
        n_molecules_sql=item.n_molecules_sql,
        n_molecules_with_fp=item.n_molecules_with_fp,
        pair_budget=budget,
        values_are_binary=item.values_are_binary,
        molecules=item.molecules,
        seed=item.seed,
    )


def run_pair_benchmark(
    items: list[GroupWorkItem],
    thresholds: list[float],
    paths: OutputPaths,
    workers: int,
    progress_every_groups: int,
    *,
    similar_label: str,
    different_label: str,
) -> dict[str, Any]:
    threshold_confusions = {threshold: Confusion() for threshold in thresholds}
    bucket_stats: dict[str, dict[str, Any]] = {
        name: {
            "n_pairs": 0,
            "similar": 0,
            "different": 0,
            "ambiguous": 0,
            "delta_sum": 0.0,
            "similarity_sum": 0.0,
            "deltas": [],
        }
        for name, _lo, _hi in SIMILARITY_BUCKETS
    }
    label_counts = Counter()
    pchembl_delta_values: list[float] = []
    enrichment_detail_rows: list[dict[str, Any]] = []
    enrichment_accumulator = new_enrichment_accumulator()
    started = time.monotonic()
    completed = 0
    pair_count = 0
    assay_count = len(items)

    with gzip.open(paths.pairs, "wt", encoding="utf-8", newline="") as pair_handle:
        fieldnames = [
            "assay_chembl_id",
            "assay_id",
            "standard_type",
            "target_chembl_id",
            "target_pref_name",
            "molecule_a_chembl_id",
            "molecule_b_chembl_id",
            "molecule_a_smiles",
            "molecule_b_smiles",
            "activity_a",
            "activity_b",
            "abs_activity_delta",
            "tanimoto",
            "similarity_bucket",
            "label",
        ]
        writer = csv.DictWriter(pair_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if workers <= 1:
            for item in items:
                rows = build_group_pairs(item)
                completed, pair_count = consume_pair_rows(
                    rows,
                    writer,
                    thresholds,
                    threshold_confusions,
                    bucket_stats,
                    enrichment_detail_rows,
                    enrichment_accumulator,
                    label_counts,
                    pchembl_delta_values,
                    completed,
                    pair_count,
                    assay_count,
                    progress_every_groups,
                    started,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending = {}
                item_iter = iter(items)
                for _ in range(min(len(items), workers * 4)):
                    item = next(item_iter)
                    pending[executor.submit(build_group_pairs, item)] = item
                while pending:
                    future = next(as_completed(pending))
                    pending.pop(future)
                    rows = future.result()
                    completed, pair_count = consume_pair_rows(
                        rows,
                        writer,
                        thresholds,
                        threshold_confusions,
                        bucket_stats,
                        enrichment_detail_rows,
                        enrichment_accumulator,
                        label_counts,
                        pchembl_delta_values,
                        completed,
                        pair_count,
                        assay_count,
                        progress_every_groups,
                        started,
                    )
                    try:
                        item = next(item_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(build_group_pairs, item)] = item

    threshold_rows = []
    best_row: dict[str, Any] | None = None
    for threshold in thresholds:
        row = {"threshold": threshold, **threshold_confusions[threshold].metrics()}
        threshold_rows.append(row)
        if best_row is None or float(row["macro_f1"]) > float(best_row["macro_f1"]):
            best_row = row
    write_tsv(paths.threshold_metrics, threshold_rows)

    bucket_rows = []
    for bucket, stats in bucket_stats.items():
        n = int(stats["n_pairs"])
        deltas = stats["deltas"]
        bucket_rows.append(
            {
                "similarity_bucket": bucket,
                "n_pairs": n,
                "similar": int(stats["similar"]),
                "different": int(stats["different"]),
                "ambiguous": int(stats["ambiguous"]),
                "similar_rate": _safe_div(int(stats["similar"]), n),
                "different_rate": _safe_div(int(stats["different"]), n),
                "ambiguous_rate": _safe_div(int(stats["ambiguous"]), n),
                "mean_abs_delta": _safe_div(float(stats["delta_sum"]), n),
                "median_abs_delta": median(deltas) if deltas else 0.0,
                "mean_tanimoto": _safe_div(float(stats["similarity_sum"]), n),
            }
        )
    write_tsv(paths.bucket_summary, bucket_rows)
    enrichment_summary_rows = finalize_enrichment_summary(enrichment_accumulator)
    write_tsv(paths.enrichment_detail, enrichment_detail_rows)
    write_tsv(paths.enrichment_summary, enrichment_summary_rows)

    non_ambiguous = label_counts[similar_label] + label_counts[different_label]
    majority_baseline = _safe_div(max(label_counts[similar_label], label_counts[different_label]), non_ambiguous)
    return {
        "n_assay_endpoints": assay_count,
        "n_pairs": pair_count,
        "label_counts": dict(label_counts),
        "n_non_ambiguous_pairs": non_ambiguous,
        "majority_baseline_accuracy": majority_baseline,
        "best_threshold_by_macro_f1": best_row or {},
        "mean_abs_activity_delta": _safe_div(sum(pchembl_delta_values), len(pchembl_delta_values)),
        "median_abs_activity_delta": median(pchembl_delta_values) if pchembl_delta_values else 0.0,
        "files": {
            "pairs": str(paths.pairs),
            "endpoint_summary": str(paths.endpoint_summary),
            "threshold_metrics": str(paths.threshold_metrics),
            "bucket_summary": str(paths.bucket_summary),
            "enrichment_detail": str(paths.enrichment_detail),
            "enrichment_summary": str(paths.enrichment_summary),
        },
    }


def consume_pair_rows(
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    thresholds: list[float],
    threshold_confusions: dict[float, Confusion],
    bucket_stats: dict[str, dict[str, Any]],
    enrichment_detail_rows: list[dict[str, Any]],
    enrichment_accumulator: dict[str, dict[str, Any]],
    label_counts: Counter,
    activity_delta_values: list[float],
    completed: int,
    pair_count: int,
    assay_count: int,
    progress_every_groups: int,
    started: float,
) -> tuple[int, int]:
    for enrichment_row in assay_enrichment_rows(rows):
        enrichment_detail_rows.append(enrichment_row)
        update_enrichment_accumulator(enrichment_accumulator, enrichment_row)

    for row in rows:
        writer.writerow(row)
        pair_count += 1
        label = str(row["label"])
        label_counts[label] += 1
        bucket = str(row["similarity_bucket"])
        tanimoto = float(row["tanimoto"])
        delta = float(row["abs_activity_delta"])
        stats = bucket_stats[bucket]
        stats["n_pairs"] += 1
        stats["similarity_sum"] += tanimoto
        stats["delta_sum"] += delta
        stats["deltas"].append(delta)
        activity_delta_values.append(delta)
        if label in ("similar", "same_class"):
            stats["similar"] += 1
            truth_similar = True
        elif label in ("different", "different_class"):
            stats["different"] += 1
            truth_similar = False
        else:
            stats["ambiguous"] += 1
            continue
        for threshold in thresholds:
            threshold_confusions[threshold].update(truth_similar, tanimoto >= threshold)
    completed += 1
    if progress_every_groups and (completed % progress_every_groups == 0 or completed == assay_count):
        log(
            f"pair benchmark progress groups={completed:,}/{assay_count:,} pairs={pair_count:,} "
            f"elapsed={format_elapsed(time.monotonic() - started)}"
        )
    return completed, pair_count


def assay_enrichment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    first = rows[0]
    baseline = summarize_pair_rows(rows)
    output_rows = []
    for bucket, _lo, _hi in SIMILARITY_BUCKETS:
        bucket_rows = [row for row in rows if row["similarity_bucket"] == bucket]
        if not bucket_rows:
            continue
        stats = summarize_pair_rows(bucket_rows)
        output_rows.append(
            {
                "assay_chembl_id": first["assay_chembl_id"],
                "assay_id": first["assay_id"],
                "standard_type": first["standard_type"],
                "target_chembl_id": first["target_chembl_id"],
                "target_pref_name": first["target_pref_name"],
                "similarity_bucket": bucket,
                "baseline_n_pairs": baseline["n_pairs"],
                "bucket_n_pairs": stats["n_pairs"],
                "baseline_similar_rate": baseline["similar_rate"],
                "bucket_similar_rate": stats["similar_rate"],
                "delta_lift_similar_rate": stats["similar_rate"] - baseline["similar_rate"],
                "fold_lift_similar_rate": _safe_div(stats["similar_rate"], baseline["similar_rate"]),
                "baseline_different_rate": baseline["different_rate"],
                "bucket_different_rate": stats["different_rate"],
                "delta_lift_different_rate": stats["different_rate"] - baseline["different_rate"],
                "baseline_mean_abs_delta": baseline["mean_abs_delta"],
                "bucket_mean_abs_delta": stats["mean_abs_delta"],
                "mean_abs_delta_reduction": baseline["mean_abs_delta"] - stats["mean_abs_delta"],
                "baseline_median_abs_delta": baseline["median_abs_delta"],
                "bucket_median_abs_delta": stats["median_abs_delta"],
                "median_abs_delta_reduction": baseline["median_abs_delta"] - stats["median_abs_delta"],
                "bucket_mean_tanimoto": stats["mean_tanimoto"],
            }
        )
    return output_rows


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    similar = 0
    different = 0
    ambiguous = 0
    deltas: list[float] = []
    tanimoto_sum = 0.0
    for row in rows:
        label = str(row["label"])
        if label in ("similar", "same_class"):
            similar += 1
        elif label in ("different", "different_class"):
            different += 1
        else:
            ambiguous += 1
        deltas.append(float(row["abs_activity_delta"]))
        tanimoto_sum += float(row["tanimoto"])
    return {
        "n_pairs": n,
        "similar": similar,
        "different": different,
        "ambiguous": ambiguous,
        "similar_rate": _safe_div(similar, n),
        "different_rate": _safe_div(different, n),
        "ambiguous_rate": _safe_div(ambiguous, n),
        "mean_abs_delta": _safe_div(sum(deltas), n),
        "median_abs_delta": median(deltas) if deltas else 0.0,
        "mean_tanimoto": _safe_div(tanimoto_sum, n),
    }


def new_enrichment_accumulator() -> dict[str, dict[str, Any]]:
    return {
        bucket: {
            "n_assay_endpoints": 0,
            "sampled_pairs": 0,
            "baseline_n_pairs": 0,
            "baseline_similar_rate": [],
            "bucket_similar_rate": [],
            "delta_lift_similar_rate": [],
            "fold_lift_similar_rate": [],
            "baseline_different_rate": [],
            "bucket_different_rate": [],
            "delta_lift_different_rate": [],
            "baseline_mean_abs_delta": [],
            "bucket_mean_abs_delta": [],
            "mean_abs_delta_reduction": [],
            "baseline_median_abs_delta": [],
            "bucket_median_abs_delta": [],
            "median_abs_delta_reduction": [],
            "bucket_mean_tanimoto": [],
        }
        for bucket, _lo, _hi in SIMILARITY_BUCKETS
    }


def update_enrichment_accumulator(accumulator: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    stats = accumulator[str(row["similarity_bucket"])]
    stats["n_assay_endpoints"] += 1
    stats["sampled_pairs"] += int(row["bucket_n_pairs"])
    stats["baseline_n_pairs"] += int(row["baseline_n_pairs"])
    for key in (
        "baseline_similar_rate",
        "bucket_similar_rate",
        "delta_lift_similar_rate",
        "fold_lift_similar_rate",
        "baseline_different_rate",
        "bucket_different_rate",
        "delta_lift_different_rate",
        "baseline_mean_abs_delta",
        "bucket_mean_abs_delta",
        "mean_abs_delta_reduction",
        "baseline_median_abs_delta",
        "bucket_median_abs_delta",
        "median_abs_delta_reduction",
        "bucket_mean_tanimoto",
    ):
        stats[key].append(float(row[key]))


def finalize_enrichment_summary(accumulator: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for bucket, _lo, _hi in SIMILARITY_BUCKETS:
        stats = accumulator[bucket]
        rows.append(
            {
                "similarity_bucket": bucket,
                "n_assay_endpoints": int(stats["n_assay_endpoints"]),
                "sampled_pairs": int(stats["sampled_pairs"]),
                "baseline_n_pairs_sum": int(stats["baseline_n_pairs"]),
                "macro_baseline_similar_rate": mean_or_zero(stats["baseline_similar_rate"]),
                "macro_bucket_similar_rate": mean_or_zero(stats["bucket_similar_rate"]),
                "macro_delta_lift_similar_rate": mean_or_zero(stats["delta_lift_similar_rate"]),
                "macro_fold_lift_similar_rate": mean_or_zero(stats["fold_lift_similar_rate"]),
                "macro_baseline_different_rate": mean_or_zero(stats["baseline_different_rate"]),
                "macro_bucket_different_rate": mean_or_zero(stats["bucket_different_rate"]),
                "macro_delta_lift_different_rate": mean_or_zero(stats["delta_lift_different_rate"]),
                "macro_baseline_mean_abs_delta": mean_or_zero(stats["baseline_mean_abs_delta"]),
                "macro_bucket_mean_abs_delta": mean_or_zero(stats["bucket_mean_abs_delta"]),
                "macro_mean_abs_delta_reduction": mean_or_zero(stats["mean_abs_delta_reduction"]),
                "macro_baseline_median_abs_delta": mean_or_zero(stats["baseline_median_abs_delta"]),
                "macro_bucket_median_abs_delta": mean_or_zero(stats["bucket_median_abs_delta"]),
                "macro_median_abs_delta_reduction": mean_or_zero(stats["median_abs_delta_reduction"]),
                "macro_bucket_mean_tanimoto": mean_or_zero(stats["bucket_mean_tanimoto"]),
            }
        )
    return rows


def build_group_pairs(item: GroupWorkItem) -> list[dict[str, Any]]:
    molecules = item.molecules
    n = len(molecules)
    rng = random.Random(item.seed)
    total_pairs = n * (n - 1) // 2
    if item.pair_budget >= total_pairs:
        pairs = ((i, j) for i in range(n - 1) for j in range(i + 1, n))
    else:
        selected: set[tuple[int, int]] = set()
        max_attempts = max(item.pair_budget * 20, 1000)
        attempts = 0
        while len(selected) < item.pair_budget and attempts < max_attempts:
            i = rng.randrange(0, n - 1)
            j = rng.randrange(i + 1, n)
            selected.add((i, j))
            attempts += 1
        pairs = iter(selected)

    rows = []
    for i, j in pairs:
        mol_a = molecules[i]
        mol_b = molecules[j]
        tanimoto = tanimoto_int(mol_a[3], mol_a[4], mol_b[3], mol_b[4])
        delta = abs(float(mol_a[2]) - float(mol_b[2]))
        if item.values_are_binary:
            label = "same_class" if int(mol_a[2]) == int(mol_b[2]) else "different_class"
        else:
            if delta <= CURRENT_SIMILAR_DELTA:
                label = "similar"
            elif delta >= CURRENT_DIFFERENT_DELTA:
                label = "different"
            else:
                label = "ambiguous"
        rows.append(
            {
                "assay_chembl_id": item.assay_chembl_id,
                "assay_id": item.assay_id,
                "standard_type": item.standard_type,
                "target_chembl_id": item.target_chembl_id or "",
                "target_pref_name": item.target_pref_name or "",
                "molecule_a_chembl_id": mol_a[0],
                "molecule_b_chembl_id": mol_b[0],
                "molecule_a_smiles": mol_a[1],
                "molecule_b_smiles": mol_b[1],
                "activity_a": format_float(float(mol_a[2])),
                "activity_b": format_float(float(mol_b[2])),
                "abs_activity_delta": format_float(delta),
                "tanimoto": format_float(tanimoto),
                "similarity_bucket": similarity_bucket(tanimoto),
                "label": label,
            }
        )
    return rows


# ProcessPool workers receive module globals, so main updates these before work starts.
CURRENT_SIMILAR_DELTA = 0.5
CURRENT_DIFFERENT_DELTA = 1.0


def tanimoto_int(fp_a: int, pop_a: int, fp_b: int, pop_b: int) -> float:
    intersection = (fp_a & fp_b).bit_count()
    union = pop_a + pop_b - intersection
    return intersection / union if union else 0.0


def similarity_bucket(value: float) -> str:
    for name, lo, hi in SIMILARITY_BUCKETS:
        if lo <= value < hi:
            return name
    return "unknown"


def run_binary_analysis(
    args: argparse.Namespace,
    thresholds: list[float],
    out_dir: Path,
    existing_fingerprints: dict[str, tuple[int, int]],
) -> tuple[dict[str, Any] | None, OutputPaths | None, dict[str, Any]]:
    log("starting conservative binary activity_comment analysis")
    conn = connect_sqlite(args.chembl_sqlite)
    try:
        groups, meta = load_binary_groups(conn, args)
    finally:
        conn.close()
    if not groups:
        return None, None, {"status": "no conservative binary groups found"}
    needed_ids = {molecule[0] for molecules in groups.values() for molecule in molecules}
    missing_ids = needed_ids.difference(existing_fingerprints)
    fingerprints = dict(existing_fingerprints)
    if missing_ids:
        fingerprints.update(load_fingerprints(Path(args.fps_gz), missing_ids))
    items, endpoint_rows = prepare_binary_work_items(groups, meta, fingerprints, args)
    if not items:
        return None, None, {"status": "no binary groups with enough fingerprints", "n_raw_groups": len(groups)}

    paths = OutputPaths(
        pairs=out_dir / "binary_pairs.tsv.gz",
        endpoint_summary=out_dir / "binary_assay_endpoint_summary.tsv",
        threshold_metrics=out_dir / "binary_threshold_metrics.tsv",
        bucket_summary=out_dir / "binary_similarity_bucket_summary.tsv",
        enrichment_detail=out_dir / "binary_assay_bucket_enrichment.tsv",
        enrichment_summary=out_dir / "binary_enrichment_summary.tsv",
    )
    write_endpoint_summary(paths.endpoint_summary, endpoint_rows)
    result = run_pair_benchmark(
        items,
        thresholds,
        paths,
        args.workers,
        args.progress_every_groups,
        similar_label="same_class",
        different_label="different_class",
    )
    summary = {
        "status": "ok",
        "n_assay_endpoints": result["n_assay_endpoints"],
        "n_pairs": result["n_pairs"],
        "label_counts": result["label_counts"],
        "best_threshold_by_macro_f1": result["best_threshold_by_macro_f1"],
        "files": result["files"],
    }
    return result, paths, summary


def load_binary_groups(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, str], list[tuple[str, str, int, int]]], dict[tuple[int, str], EndpointMeta]]:
    duplicate_filter = "" if args.include_potential_duplicates else "AND COALESCE(act.potential_duplicate, 0) = 0"
    allowed_comments = sorted(ACTIVE_COMMENTS | INACTIVE_COMMENTS)
    placeholders = ",".join("?" for _ in allowed_comments)
    query = f"""
        SELECT
          act.assay_id,
          LOWER(TRIM(COALESCE(act.standard_type, act.type, 'activity_comment'))) AS standard_type,
          act.molregno AS molregno,
          LOWER(TRIM(act.activity_comment)) AS activity_comment,
          COUNT(*) AS n_records
        FROM activities act
        WHERE act.pchembl_value IS NULL
          AND act.activity_comment IS NOT NULL
          AND act.molregno IS NOT NULL
          AND LOWER(TRIM(act.activity_comment)) IN ({placeholders})
          {duplicate_filter}
        GROUP BY act.assay_id, LOWER(TRIM(COALESCE(act.standard_type, act.type, 'activity_comment'))), act.molregno, LOWER(TRIM(act.activity_comment))
    """
    per_molecule: dict[tuple[int, str, int], set[int]] = defaultdict(set)
    molecule_record_counts: Counter = Counter()
    for row in conn.execute(query, allowed_comments):
        label = normalize_activity_comment(row["activity_comment"])
        if label is None:
            continue
        standard_type = str(row["standard_type"] or "activity_comment").strip().lower()
        key = (int(row["assay_id"]), standard_type)
        molecule_key = (key[0], key[1], int(row["molregno"]))
        per_molecule[molecule_key].add(label)
        molecule_record_counts[molecule_key] += int(row["n_records"] or 0)

    molregnos = sorted({molecule_key[2] for molecule_key in per_molecule})
    log(f"loading molecule IDs and SMILES for binary values: molregnos={len(molregnos):,}")
    molecule_lookup = load_molecule_lookup(conn, molregnos)
    groups_by_molregno: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    record_counts = Counter()
    for molecule_key, labels in per_molecule.items():
        if len(labels) != 1:
            continue
        key = (molecule_key[0], molecule_key[1])
        molregno = molecule_key[2]
        if molregno not in molecule_lookup:
            continue
        label = next(iter(labels))
        groups_by_molregno[key].append((molregno, label, molecule_record_counts[molecule_key]))
        record_counts[key] += molecule_record_counts[molecule_key]
    groups_by_molregno = {
        key: molecules
        for key, molecules in groups_by_molregno.items()
        if len(molecules) >= args.min_molecules_per_assay
    }

    groups: dict[tuple[int, str], list[tuple[str, str, int, int]]] = defaultdict(list)
    for key, molecules in groups_by_molregno.items():
        for molregno, label, n_records in molecules:
            molecule_id, smiles = molecule_lookup[molregno]
            groups[key].append((molecule_id, smiles, label, n_records))

    meta = load_endpoint_meta_for_keys(conn, groups.keys())
    filtered_meta: dict[tuple[int, str], EndpointMeta] = {}
    for key, endpoint in meta.items():
        if key not in groups_by_molregno:
            continue
        filtered_meta[key] = EndpointMeta(
            assay_id=endpoint.assay_id,
            assay_chembl_id=endpoint.assay_chembl_id,
            standard_type=endpoint.standard_type,
            n_molecules_sql=len(groups_by_molregno[key]),
            n_records_sql=record_counts[key],
            assay_type=endpoint.assay_type,
            assay_test_type=endpoint.assay_test_type,
            assay_category=endpoint.assay_category,
            assay_organism=endpoint.assay_organism,
            confidence_score=endpoint.confidence_score,
            relationship_type=endpoint.relationship_type,
            target_chembl_id=endpoint.target_chembl_id,
            target_pref_name=endpoint.target_pref_name,
            target_type=endpoint.target_type,
            target_organism=endpoint.target_organism,
            description=endpoint.description,
        )
    log(f"binary assay-endpoints retained: {len(groups):,}")
    return groups, filtered_meta


def load_endpoint_meta_for_keys(
    conn: sqlite3.Connection,
    keys: Iterable[tuple[int, str]],
) -> dict[tuple[int, str], EndpointMeta]:
    keys = list(keys)
    if not keys:
        return {}
    assay_ids = sorted({key[0] for key in keys})
    standard_types_by_assay: dict[int, set[str]] = defaultdict(set)
    for assay_id, standard_type in keys:
        standard_types_by_assay[assay_id].add(standard_type)
    meta: dict[tuple[int, str], EndpointMeta] = {}
    for chunk in chunks(assay_ids, 1000):
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
            SELECT
              a.assay_id,
              a.chembl_id AS assay_chembl_id,
              a.assay_type,
              a.assay_test_type,
              a.assay_category,
              a.assay_organism,
              a.confidence_score,
              a.relationship_type,
              a.description,
              td.chembl_id AS target_chembl_id,
              td.pref_name AS target_pref_name,
              td.target_type,
              td.organism AS target_organism
            FROM assays a
            LEFT JOIN target_dictionary td ON a.tid = td.tid
            WHERE a.assay_id IN ({placeholders})
        """
        for row in conn.execute(query, chunk):
            assay_id = int(row["assay_id"])
            for standard_type in standard_types_by_assay[assay_id]:
                meta[(assay_id, standard_type)] = EndpointMeta(
                    assay_id=assay_id,
                    assay_chembl_id=str(row["assay_chembl_id"]),
                    standard_type=standard_type,
                    n_molecules_sql=0,
                    n_records_sql=0,
                    assay_type=none_or_str(row["assay_type"]),
                    assay_test_type=none_or_str(row["assay_test_type"]),
                    assay_category=none_or_str(row["assay_category"]),
                    assay_organism=none_or_str(row["assay_organism"]),
                    confidence_score=none_or_int(row["confidence_score"]),
                    relationship_type=none_or_str(row["relationship_type"]),
                    target_chembl_id=none_or_str(row["target_chembl_id"]),
                    target_pref_name=none_or_str(row["target_pref_name"]),
                    target_type=none_or_str(row["target_type"]),
                    target_organism=none_or_str(row["target_organism"]),
                    description=none_or_str(row["description"]),
                )
    return meta


def prepare_binary_work_items(
    groups: dict[tuple[int, str], list[tuple[str, str, int, int]]],
    meta: dict[tuple[int, str], EndpointMeta],
    fingerprints: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[GroupWorkItem], list[dict[str, Any]]]:
    raw_items: list[GroupWorkItem] = []
    endpoint_rows: list[dict[str, Any]] = []
    for key, molecules in groups.items():
        endpoint = meta[key]
        with_fp = []
        for molecule_id, smiles, label, _n_records in molecules:
            fp = fingerprints.get(molecule_id)
            if fp is None:
                continue
            with_fp.append((molecule_id, smiles, label, fp[0], fp[1]))
        n = len(with_fp)
        total_pairs = n * (n - 1) // 2
        endpoint_rows.append(endpoint_summary_row(endpoint, len(molecules), n, total_pairs))
        if n < args.min_molecules_with_fp:
            continue
        budget = min(args.max_pairs_per_assay, total_pairs)
        raw_items.append(
            GroupWorkItem(
                assay_id=endpoint.assay_id,
                assay_chembl_id=endpoint.assay_chembl_id,
                standard_type=endpoint.standard_type,
                target_chembl_id=endpoint.target_chembl_id,
                target_pref_name=endpoint.target_pref_name,
                n_molecules_sql=endpoint.n_molecules_sql,
                n_molecules_with_fp=n,
                pair_budget=budget,
                values_are_binary=True,
                molecules=tuple(with_fp),
                seed=args.seed + stable_int("binary" + endpoint.assay_chembl_id + endpoint.standard_type),
            )
        )
    return allocate_pair_budgets(raw_items, args.binary_max_total_pairs), endpoint_rows


def normalize_activity_comment(value: str | None) -> int | None:
    if not value:
        return None
    normalized = " ".join(str(value).strip().lower().replace("-", " ").replace("_", " ").split())
    if normalized in ACTIVE_COMMENTS:
        return 1
    if normalized in INACTIVE_COMMENTS:
        return 0
    return None


def write_endpoint_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_tsv(path, rows)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_figures(
    result: dict[str, Any],
    paths: OutputPaths,
    figures_dir: Path,
    *,
    title_prefix: str,
    stem_prefix: str = "",
) -> None:
    threshold_rows = read_tsv_rows(paths.threshold_metrics)
    bucket_rows = read_tsv_rows(paths.bucket_summary)
    enrichment_rows = read_tsv_rows(paths.enrichment_summary)
    if threshold_rows:
        write_line_svg(
            figures_dir / f"{stem_prefix}threshold_metrics.svg",
            threshold_rows,
            x_key="threshold",
            series=[
                ("macro_f1", "#2563eb"),
                ("balanced_accuracy", "#16a34a"),
                ("accuracy", "#dc2626"),
            ],
            title=f"{title_prefix}: threshold baseline metrics",
            x_label="Tanimoto threshold",
            y_label="score",
        )
    if bucket_rows:
        write_grouped_bar_svg(
            figures_dir / f"{stem_prefix}label_rates_by_bucket.svg",
            bucket_rows,
            category_key="similarity_bucket",
            series=[
                ("similar_rate", "#2563eb"),
                ("different_rate", "#dc2626"),
                ("ambiguous_rate", "#9ca3af"),
            ],
            title=f"{title_prefix}: label rates by similarity bucket",
            y_label="rate",
        )
        write_bar_svg(
            figures_dir / f"{stem_prefix}median_delta_by_bucket.svg",
            bucket_rows,
            category_key="similarity_bucket",
            value_key="median_abs_delta",
            title=f"{title_prefix}: median absolute activity delta",
            y_label="median abs delta",
            color="#7c3aed",
        )
        write_bar_svg(
            figures_dir / f"{stem_prefix}pair_counts_by_bucket.svg",
            bucket_rows,
            category_key="similarity_bucket",
            value_key="n_pairs",
            title=f"{title_prefix}: sampled pair counts",
            y_label="pairs",
            color="#0f766e",
        )
    if enrichment_rows:
        write_signed_bar_svg(
            figures_dir / f"{stem_prefix}delta_lift_similar_rate_by_bucket.svg",
            enrichment_rows,
            category_key="similarity_bucket",
            value_key="macro_delta_lift_similar_rate",
            title=f"{title_prefix}: macro similar-rate lift over assay baseline",
            y_label="bucket similar rate - assay baseline",
            color="#2563eb",
        )
        write_bar_svg(
            figures_dir / f"{stem_prefix}fold_lift_similar_rate_by_bucket.svg",
            enrichment_rows,
            category_key="similarity_bucket",
            value_key="macro_fold_lift_similar_rate",
            title=f"{title_prefix}: macro fold lift over assay baseline",
            y_label="bucket similar rate / assay baseline",
            color="#16a34a",
        )
        write_signed_bar_svg(
            figures_dir / f"{stem_prefix}median_delta_reduction_by_bucket.svg",
            enrichment_rows,
            category_key="similarity_bucket",
            value_key="macro_median_abs_delta_reduction",
            title=f"{title_prefix}: macro median-delta reduction vs assay baseline",
            y_label="baseline median delta - bucket median delta",
            color="#7c3aed",
        )


def write_line_svg(
    path: Path,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    series: list[tuple[str, str]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 920, 520
    left, right, top, bottom = 82, 30, 56, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(row[x_key]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_max = max(float(row[key]) for row in rows for key, _color in series)
    y_max = max(1.0, math.ceil(y_max * 10) / 10)

    def sx(value: float) -> float:
        return left + _safe_div(value - x_min, x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - _safe_div(value, y_max)) * plot_h

    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, x_label, y_label))
    for key, color in series:
        points = " ".join(f"{sx(float(row[x_key])):.1f},{sy(float(row[key])):.1f}" for row in rows)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
    parts.extend(legend_svg(series, width - 260, top + 8))
    for tick in xs:
        parts.append(f'<text x="{sx(tick):.1f}" y="{height - 42}" text-anchor="middle" font-size="11">{tick:g}</text>')
    for value in [0, y_max / 2, y_max]:
        parts.append(f'<text x="{left - 10}" y="{sy(value) + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_bar_svg(
    path: Path,
    rows: list[dict[str, str]],
    *,
    category_key: str,
    series: list[tuple[str, str]],
    title: str,
    y_label: str,
) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 86, 28, 58, 126
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(1.0, max(float(row[key]) for row in rows for key, _color in series))
    group_w = plot_w / max(len(rows), 1)
    bar_w = group_w / (len(series) + 1)
    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, "similarity bucket", y_label))
    for i, row in enumerate(rows):
        group_x = left + i * group_w
        for j, (key, color) in enumerate(series):
            value = float(row[key])
            bar_h = _safe_div(value, y_max) * plot_h
            x = group_x + (j + 0.5) * bar_w
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        label = str(row[category_key]).replace("_analog", "")
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{height - 72}" text-anchor="end" '
            f'font-size="11" transform="rotate(-32 {group_x + group_w / 2:.1f},{height - 72})">{escape(label)}</text>'
        )
    for value in [0, y_max / 2, y_max]:
        y = top + (1 - _safe_div(value, y_max)) * plot_h
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    parts.extend(legend_svg(series, width - 300, top + 8))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(
    path: Path,
    rows: list[dict[str, str]],
    *,
    category_key: str,
    value_key: str,
    title: str,
    y_label: str,
    color: str,
) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 86, 28, 58, 126
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row[value_key]) for row in rows]
    y_max = max(values) if values else 1.0
    y_max = max(1.0, y_max * 1.08)
    bar_w = plot_w / max(len(rows), 1) * 0.58
    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, "similarity bucket", y_label))
    for i, row in enumerate(rows):
        value = float(row[value_key])
        x_center = left + (i + 0.5) * plot_w / max(len(rows), 1)
        bar_h = _safe_div(value, y_max) * plot_h
        x = x_center - bar_w / 2
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x_center:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11">{value:.2g}</text>')
        label = str(row[category_key]).replace("_analog", "")
        parts.append(
            f'<text x="{x_center:.1f}" y="{height - 72}" text-anchor="end" '
            f'font-size="11" transform="rotate(-32 {x_center:.1f},{height - 72})">{escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_signed_bar_svg(
    path: Path,
    rows: list[dict[str, str]],
    *,
    category_key: str,
    value_key: str,
    title: str,
    y_label: str,
    color: str,
) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 92, 28, 58, 126
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row[value_key]) for row in rows]
    min_value = min(0.0, min(values) if values else 0.0)
    max_value = max(0.0, max(values) if values else 0.0)
    if math.isclose(min_value, max_value):
        min_value, max_value = -1.0, 1.0
    pad = (max_value - min_value) * 0.08
    y_min = min_value - pad
    y_max = max_value + pad
    bar_w = plot_w / max(len(rows), 1) * 0.58

    def sy(value: float) -> float:
        return top + (1.0 - _safe_div(value - y_min, y_max - y_min)) * plot_h

    zero_y = sy(0.0)
    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, "similarity bucket", y_label))
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#6b7280" stroke-width="1"/>')
    for i, row in enumerate(rows):
        value = float(row[value_key])
        x_center = left + (i + 0.5) * plot_w / max(len(rows), 1)
        y_value = sy(value)
        y = min(y_value, zero_y)
        bar_h = abs(zero_y - y_value)
        x = x_center - bar_w / 2
        fill = color if value >= 0 else "#dc2626"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{fill}"/>')
        label_y = y - 6 if value >= 0 else y + bar_h + 14
        parts.append(f'<text x="{x_center:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="11">{value:.2g}</text>')
        label = str(row[category_key]).replace("_analog", "")
        parts.append(
            f'<text x="{x_center:.1f}" y="{height - 72}" text-anchor="end" '
            f'font-size="11" transform="rotate(-32 {x_center:.1f},{height - 72})">{escape(label)}</text>'
        )
    for value in [y_min, 0.0, y_max]:
        parts.append(f'<text x="{left - 10}" y="{sy(value) + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" font-weight="700">{escape(title)}</text>',
    ]


def axis_svg(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> str:
    bottom_y = top + plot_h
    return "\n".join(
        [
            f'<line x1="{left}" y1="{bottom_y}" x2="{left + plot_w}" y2="{bottom_y}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{bottom_y + 56}" text-anchor="middle" font-size="13">{escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 22,{top + plot_h / 2:.1f})">{escape(y_label)}</text>',
        ]
    )


def legend_svg(series: list[tuple[str, str]], x: int, y: int) -> list[str]:
    parts = []
    for i, (label, color) in enumerate(series):
        yy = y + i * 22
        parts.append(f'<rect x="{x}" y="{yy - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{yy + 1}" font-size="12">{escape(label)}</text>')
    return parts


def format_enrichment_report_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["enrichment 汇总尚未生成。"]
    output = [
        "| similarity bucket | assay endpoints | macro baseline similar_rate | macro bucket similar_rate | delta lift | fold lift | median delta reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        output.append(
            "| {bucket} | {n:,} | {base:.3f} | {bucket_rate:.3f} | {delta:+.3f} | {fold:.3f} | {delta_red:+.3f} |".format(
                bucket=str(row["similarity_bucket"]).replace("_analog", ""),
                n=int(float(row["n_assay_endpoints"])),
                base=float(row["macro_baseline_similar_rate"]),
                bucket_rate=float(row["macro_bucket_similar_rate"]),
                delta=float(row["macro_delta_lift_similar_rate"]),
                fold=float(row["macro_fold_lift_similar_rate"]),
                delta_red=float(row["macro_median_abs_delta_reduction"]),
            )
        )
    close = next((row for row in rows if row["similarity_bucket"] == "close_analog"), None)
    distant = next((row for row in rows if row["similarity_bucket"] == "distant_analog"), None)
    if close and distant:
        output.extend(
            [
                "",
                "直观解读：",
                "",
                f"- close analog 的 macro similar-rate lift 为 {float(close['macro_delta_lift_similar_rate']):+.3f}，fold lift 为 {float(close['macro_fold_lift_similar_rate']):.3f}。",
                f"- distant analog 的 macro similar-rate lift 为 {float(distant['macro_delta_lift_similar_rate']):+.3f}，fold lift 为 {float(distant['macro_fold_lift_similar_rate']):.3f}。",
                "- 如果 lift 接近 0，说明该 bucket 的高 similar_rate 大多来自 assay 自身背景；如果 lift 明显为正，才说明结构相似带来了额外 transfer 信息。",
            ]
        )
    return output


def format_dynamic_filter_report_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return []
    if not summary.get("filter_enabled"):
        return ["dynamic range filter: 未启用。"]
    return [
        "dynamic range filter:",
        "",
        f"- min pChEMBL range: {float(summary.get('min_pchembl_range', 0.0)):.3f}",
        f"- min pChEMBL IQR: {float(summary.get('min_pchembl_iqr', 0.0)):.3f}",
        f"- input assay-endpoints: {int(summary.get('input_assay_endpoints', 0)):,}",
        f"- retained assay-endpoints: {int(summary.get('retained_assay_endpoints', 0)):,}",
        f"- removed assay-endpoints: {int(summary.get('removed_assay_endpoints', 0)):,}",
        f"- below min range: {int(summary.get('below_min_pchembl_range', 0)):,}",
        f"- below min IQR: {int(summary.get('below_min_pchembl_iqr', 0)):,}",
    ]


def write_report(
    path: Path,
    summary: dict[str, Any],
    continuous_paths: OutputPaths,
    binary_paths: OutputPaths | None,
    figures_dir: Path,
) -> None:
    continuous = summary["continuous"]
    best = continuous.get("best_threshold_by_macro_f1") or {}
    labels = continuous.get("label_counts") or {}
    binary = summary.get("binary") or {}
    dynamic_filter = summary.get("dynamic_range_filter") or {}
    enrichment_rows = read_tsv_rows(continuous_paths.enrichment_summary)
    lines = [
        "# ChEMBL assay activity transferability 阈值基线报告",
        "",
        "## 目标",
        "",
        "这个第一版 benchmark 不调用 LLM，只测试一个最简单的结构相似度基线：在同一个 ChEMBL assay endpoint 中，两个分子的 Morgan Tanimoto 相似度是否足以判断 activity 是否可以 transfer。",
        "",
        "## 连续值主分析",
        "",
        "筛选规则：`pchembl_value` 非空，`standard_relation='='`，`standard_flag=1`，默认排除 `data_validity_comment` 非空记录和 `potential_duplicate=1` 记录；同一 molecule 在同一 assay endpoint 的重复值用平均 pChEMBL 汇总。",
        "",
        f"- assay-endpoints: {int(continuous.get('n_assay_endpoints', 0)):,}",
        f"- sampled molecule pairs: {int(continuous.get('n_pairs', 0)):,}",
        f"- non-ambiguous pairs: {int(continuous.get('n_non_ambiguous_pairs', 0)):,}",
        f"- label counts: {json.dumps(labels, ensure_ascii=False, sort_keys=True)}",
        f"- mean |delta pChEMBL|: {float(continuous.get('mean_abs_activity_delta', 0.0)):.3f}",
        f"- median |delta pChEMBL|: {float(continuous.get('median_abs_activity_delta', 0.0)):.3f}",
        "",
    ]
    lines.extend(format_dynamic_filter_report_lines(dynamic_filter))
    lines.extend(
        [
            "",
            "标签定义：",
        "",
        "- `similar`: `|delta pChEMBL| <= 0.5`，约等于 3.16 倍以内。",
        "- `different`: `|delta pChEMBL| >= 1.0`，约等于 10 倍以上。",
        "- `ambiguous`: 中间区域，不进入阈值分类主指标。",
        "",
        "最佳 Tanimoto 阈值按 macro-F1 选择：",
        "",
        f"- threshold: {float(best.get('threshold', 0.0)):.3f}",
        f"- macro-F1: {float(best.get('macro_f1', 0.0)):.3f}",
        f"- balanced accuracy: {float(best.get('balanced_accuracy', 0.0)):.3f}",
        f"- accuracy: {float(best.get('accuracy', 0.0)):.3f}",
        f"- precision(similar): {float(best.get('precision', 0.0)):.3f}",
        f"- recall(similar): {float(best.get('recall', 0.0)):.3f}",
        f"- majority baseline accuracy: {float(continuous.get('majority_baseline_accuracy', 0.0)):.3f}",
        "",
        "结论：这个单一 Tanimoto threshold 可以提供一个很弱的排序信号，但不适合单独作为 activity transfer 判据。尤其当 majority baseline accuracy 高于 threshold accuracy 时，应优先看 macro-F1 / balanced accuracy，而不是只看 accuracy。",
        "",
        "## Assay-specific enrichment 分析",
        "",
        "这里先在每个 assay endpoint 内计算本 assay 的随机 pair 背景 similar_rate / median delta，再比较各 similarity bucket 相对本 assay 背景的提升。这个分析用于回答：高 similar_rate 是来自结构相似，还是 assay 自身本来就容易 similar。",
        "",
        ]
    )
    lines.extend(format_enrichment_report_lines(enrichment_rows))
    lines.extend(
        [
            "",
            "## 非连续值保守分析",
            "",
        ]
    )
    if binary.get("status") == "ok":
        bbest = binary.get("best_threshold_by_macro_f1") or {}
        lines.extend(
            [
                "只使用非常保守的 `activity_comment` 映射：`active` / `active compound` 视为 active，`inactive` / `not active` / `no activity` / `no inhibition` 视为 inactive；冲突 molecule 被丢弃。",
                "",
                f"- assay-endpoints: {int(binary.get('n_assay_endpoints', 0)):,}",
                f"- sampled molecule pairs: {int(binary.get('n_pairs', 0)):,}",
                f"- label counts: {json.dumps(binary.get('label_counts', {}), ensure_ascii=False, sort_keys=True)}",
                f"- best threshold: {float(bbest.get('threshold', 0.0)):.3f}",
                f"- macro-F1: {float(bbest.get('macro_f1', 0.0)):.3f}",
                f"- balanced accuracy: {float(bbest.get('balanced_accuracy', 0.0)):.3f}",
                "",
            ]
        )
    else:
        lines.extend([f"状态：{binary.get('status', 'skipped')}", ""])
    lines.extend(
        [
            "## 图表",
            "",
            f"- 连续值阈值指标：`{figures_dir / 'threshold_metrics.svg'}`",
            f"- 连续值各相似度桶标签比例：`{figures_dir / 'label_rates_by_bucket.svg'}`",
            f"- 连续值各相似度桶 median delta：`{figures_dir / 'median_delta_by_bucket.svg'}`",
            f"- 连续值各相似度桶 pair 数：`{figures_dir / 'pair_counts_by_bucket.svg'}`",
            f"- 连续值 similar-rate enrichment：`{figures_dir / 'delta_lift_similar_rate_by_bucket.svg'}`",
            f"- 连续值 fold enrichment：`{figures_dir / 'fold_lift_similar_rate_by_bucket.svg'}`",
            f"- 连续值 median delta reduction：`{figures_dir / 'median_delta_reduction_by_bucket.svg'}`",
        ]
    )
    if binary_paths is not None:
        lines.extend(
            [
                f"- binary 阈值指标：`{figures_dir / 'binary_threshold_metrics.svg'}`",
                f"- binary 各相似度桶标签比例：`{figures_dir / 'binary_label_rates_by_bucket.svg'}`",
                f"- binary similar-rate enrichment：`{figures_dir / 'binary_delta_lift_similar_rate_by_bucket.svg'}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 连续值 pair 数据：`{continuous_paths.pairs}`",
            f"- 连续值 assay endpoint 汇总：`{continuous_paths.endpoint_summary}`",
            f"- 连续值阈值指标：`{continuous_paths.threshold_metrics}`",
            f"- 连续值相似度桶汇总：`{continuous_paths.bucket_summary}`",
            f"- 连续值 assay-specific enrichment 明细：`{continuous_paths.enrichment_detail}`",
            f"- 连续值 assay-specific enrichment 汇总：`{continuous_paths.enrichment_summary}`",
            f"- manifest：`{path.parent / 'manifest.json'}`",
            "",
            "## 初步解释",
            "",
            "这个结果应作为后续 LLM assay-transfer 判断的最低 baseline。LLM 版本至少需要超过“只看 Tanimoto 阈值”的 macro-F1 / balanced accuracy，并且最好在 close/moderate analog 区间给出更好的 confidence calibration，才说明它真正利用了 assay context、结构变化和性质差异。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def stable_int(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def format_float(value: float) -> str:
    return f"{value:.6g}"


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def none_or_int(value: Any) -> int | None:
    return None if value is None else int(value)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def log(message: str) -> None:
    print(f"[activity_transfer_benchmark] {message}", file=sys.stderr, flush=True)


def format_elapsed(seconds: float) -> str:
    seconds_int = int(seconds)
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


if __name__ == "__main__":
    parsed_args = parse_args()
    CURRENT_SIMILAR_DELTA = parsed_args.similar_delta
    CURRENT_DIFFERENT_DELTA = parsed_args.different_delta
    raise SystemExit(main(sys.argv[1:]))
