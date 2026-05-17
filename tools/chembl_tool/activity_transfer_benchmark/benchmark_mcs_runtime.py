"""Benchmark RDKit MCS coverage runtime on sampled activity-transfer pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = (
    "outputs/chembl_tool/activity_transfer_benchmark/"
    "chembl36_activity_transfer_dynamic_v1/continuous_pairs.tsv.gz"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/mcs_runtime"
SIMILARITY_BUCKETS = [
    "very_close_analog",
    "close_analog",
    "moderate_analog",
    "weak_analog",
    "distant_analog",
    "very_distant_analog",
]
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RDKIT_NUM_THREADS",
)
RESULT_FIELDS = [
    "index",
    "similarity_bucket",
    "tanimoto",
    "label",
    "status",
    "timed_out",
    "elapsed_s",
    "query_heavy_atoms",
    "reference_heavy_atoms",
    "mcs_atoms",
    "query_mcs_coverage",
    "reference_mcs_coverage",
    "mean_mcs_coverage",
    "error",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("mcs_runtime_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_path = out_dir / "mcs_sample_results.tsv"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report_zh.md"
    missing_path = out_dir / "missing_result_indices.tsv"

    if args.finalize_existing:
        result_path = pair_path if pair_path.exists() else pair_path.with_suffix(pair_path.suffix + ".tmp")
        if not result_path.exists():
            raise FileNotFoundError(f"No existing result TSV found at {pair_path} or {result_path}")
        log(f"finalizing existing result file: {result_path}")
        started = time.perf_counter()
        rows, observed_indices = read_result_rows(result_path)
        bucket_totals, total_pairs = scan_pair_bucket_totals(Path(args.pairs), max_scan_pairs=args.max_scan_pairs)
        missing_indices = write_missing_indices(missing_path, observed_indices, total_pairs)
        wall_s = time.perf_counter() - started
        summary = summarize_results(rows, bucket_totals, total_pairs, args.workers, wall_s)
        summary["completion"] = {
            "mode": "finalize_existing",
            "result_file_is_partial": str(result_path).endswith(".tmp") or len(rows) < total_pairs,
            "result_rows": len(rows),
            "expected_pairs": total_pairs,
            "missing_result_count": len(missing_indices),
            "completion_rate": safe_div(len(rows), total_pairs),
        }
        summary["files"] = {
            "sample_results": str(result_path),
            "missing_result_indices": str(missing_path),
            "summary": str(summary_path),
            "report": str(report_path),
        }
        summary["parameters"] = vars(args)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_report(report_path, summary)
        log(
            "finalized existing results "
            f"rows={len(rows):,}/{total_pairs:,} missing={len(missing_indices):,} "
            f"report={report_path}"
        )
        return 0

    if args.full_scan:
        log(f"streaming all pairs from {args.pairs}")
        started = time.perf_counter()
        summary = run_mcs_full_scan(
            Path(args.pairs),
            output_path=pair_path,
            workers=args.workers,
            timeout_s=args.timeout_s,
            progress_every=args.progress_every,
            max_scan_pairs=args.max_scan_pairs,
        )
        wall_s = time.perf_counter() - started
        summary["files"] = {
            "sample_results": str(pair_path),
            "summary": str(summary_path),
            "report": str(report_path),
        }
        summary["parameters"] = vars(args)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_report(report_path, summary)
        log(f"finished wall_s={wall_s:.2f} throughput={summary['overall']['observed_pairs_per_s']:.2f} pairs/s")
        log(f"report: {report_path}")
        return 0

    log(f"sampling pairs from {args.pairs}")
    sampled, bucket_totals, total_pairs = sample_pairs(
        Path(args.pairs),
        per_bucket=args.per_bucket,
        seed=args.seed,
        max_scan_pairs=args.max_scan_pairs,
    )
    log(
        f"sampled={len(sampled):,} total_scanned_pairs={total_pairs:,} "
        f"bucket_totals={dict(bucket_totals)}"
    )
    if not sampled:
        raise RuntimeError("No pairs sampled.")

    started = time.perf_counter()
    rows = run_mcs_jobs(
        sampled,
        workers=args.workers,
        timeout_s=args.timeout_s,
        chunksize=args.chunksize,
        progress_every=args.progress_every,
    )
    wall_s = time.perf_counter() - started
    summary = summarize_results(rows, bucket_totals, total_pairs, args.workers, wall_s)

    write_tsv(pair_path, rows)
    summary["files"] = {
        "sample_results": str(pair_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    summary["parameters"] = vars(args)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(report_path, summary)
    log(f"finished wall_s={wall_s:.2f} throughput={summary['overall']['observed_pairs_per_s']:.2f} pairs/s")
    log(f"report: {report_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--per-bucket", type=int, default=1000)
    parser.add_argument("--full-scan", action="store_true", help="Stream and compute every pair without reservoir sampling.")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--timeout-s", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-scan-pairs", type=int, default=0, help="Optional scan limit for smoke tests.")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Build summary/report from an existing mcs_sample_results.tsv or .tmp file without running MCS.",
    )
    return parser.parse_args(argv)


def sample_pairs(
    path: Path,
    *,
    per_bucket: int,
    seed: int,
    max_scan_pairs: int,
) -> tuple[list[dict[str, str]], Counter, int]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, str]]] = {bucket: [] for bucket in SIMILARITY_BUCKETS}
    bucket_totals: Counter = Counter()
    total_pairs = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_pairs += 1
            bucket = row["similarity_bucket"]
            bucket_totals[bucket] += 1
            reservoir = reservoirs.setdefault(bucket, [])
            seen = bucket_totals[bucket]
            if len(reservoir) < per_bucket:
                reservoir.append(row)
            else:
                replace_index = rng.randrange(seen)
                if replace_index < per_bucket:
                    reservoir[replace_index] = row
            if max_scan_pairs and total_pairs >= max_scan_pairs:
                break
    sampled: list[dict[str, str]] = []
    for bucket in SIMILARITY_BUCKETS:
        sampled.extend(reservoirs.get(bucket, []))
    return sampled, bucket_totals, total_pairs


def scan_pair_bucket_totals(path: Path, *, max_scan_pairs: int) -> tuple[Counter, int]:
    bucket_totals: Counter = Counter()
    total_pairs = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_pairs += 1
            bucket_totals[row["similarity_bucket"]] += 1
            if max_scan_pairs and total_pairs >= max_scan_pairs:
                break
    return bucket_totals, total_pairs


def read_result_rows(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    rows: list[dict[str, Any]] = []
    indices: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            normalized = normalize_result_row(row)
            normalized["index"] = int(normalized["index"])
            normalized["tanimoto"] = float(normalized["tanimoto"])
            normalized["timed_out"] = parse_bool(normalized["timed_out"])
            normalized["elapsed_s"] = float(normalized["elapsed_s"] or 0.0)
            normalized["query_heavy_atoms"] = int(float(normalized["query_heavy_atoms"] or 0))
            normalized["reference_heavy_atoms"] = int(float(normalized["reference_heavy_atoms"] or 0))
            normalized["mcs_atoms"] = int(float(normalized["mcs_atoms"] or 0))
            normalized["query_mcs_coverage"] = float(normalized["query_mcs_coverage"] or 0.0)
            normalized["reference_mcs_coverage"] = float(normalized["reference_mcs_coverage"] or 0.0)
            normalized["mean_mcs_coverage"] = float(normalized["mean_mcs_coverage"] or 0.0)
            rows.append(normalized)
            indices.add(int(normalized["index"]))
    return rows, indices


def write_missing_indices(path: Path, observed_indices: set[int], total_pairs: int) -> list[int]:
    missing = [index for index in range(total_pairs) if index not in observed_indices]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["index"])
        for index in missing:
            writer.writerow([index])
    return missing


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def run_mcs_jobs(
    pairs: list[dict[str, str]],
    *,
    workers: int,
    timeout_s: int,
    chunksize: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    tasks = [task_from_pair_row(index, row, timeout_s) for index, row in enumerate(pairs)]
    if chunksize <= 1:
        return run_mcs_jobs_progress(tasks, workers=workers, progress_every=progress_every)

    batches = [tasks[index : index + chunksize] for index in range(0, len(tasks), chunksize)]
    rows: list[dict[str, Any]] = []
    total = len(tasks)
    completed = 0
    timed_out = 0
    started = time.perf_counter()
    last_reported = 0
    max_pending = max(workers * 4, workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        pending = {}
        batch_iter = iter(batches)
        for _ in range(min(max_pending, len(batches))):
            batch = next(batch_iter)
            pending[executor.submit(compute_mcs_batch, batch)] = len(batch)
        while pending:
            done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                batch_size = pending.pop(future)
                batch_rows = future.result()
                rows.extend(batch_rows)
                completed += batch_size
                timed_out += sum(1 for row in batch_rows if row.get("timed_out"))
                if should_report_progress(completed, total, last_reported, progress_every):
                    log_progress(completed, total, timed_out, started)
                    last_reported = completed
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    continue
                pending[executor.submit(compute_mcs_batch, batch)] = len(batch)
    return rows


def run_mcs_full_scan(
    path: Path,
    *,
    output_path: Path,
    workers: int,
    timeout_s: int,
    progress_every: int,
    max_scan_pairs: int,
) -> dict[str, Any]:
    output_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    accumulator = new_stream_accumulator()
    completed = 0
    timed_out = 0
    submitted = 0
    started = time.perf_counter()
    last_reported = 0
    max_pending = max(workers * 8, workers)

    with output_tmp.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
            pending = {}
            with gzip.open(path, "rt", encoding="utf-8", newline="") as input_handle:
                reader = csv.DictReader(input_handle, delimiter="\t")
                reader_iter = iter(reader)
                exhausted = False
                while pending or not exhausted:
                    while not exhausted and len(pending) < max_pending:
                        if max_scan_pairs and submitted >= max_scan_pairs:
                            exhausted = True
                            break
                        try:
                            pair_row = next(reader_iter)
                        except StopIteration:
                            exhausted = True
                            break
                        task = task_from_pair_row(submitted, pair_row, timeout_s)
                        pending[executor.submit(compute_mcs_row, task)] = None
                        submitted += 1
                        accumulator["bucket_totals"][pair_row["similarity_bucket"]] += 1
                    if not pending:
                        continue
                    done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        row = normalize_result_row(future.result())
                        writer.writerow(row)
                        update_stream_accumulator(accumulator, row)
                        completed += 1
                        if row.get("timed_out"):
                            timed_out += 1
                        if should_report_progress(completed, submitted, last_reported, progress_every):
                            log_progress(completed, submitted, timed_out, started)
                            last_reported = completed
                    if progress_every and completed % (progress_every * 10) == 0:
                        output_handle.flush()
    output_tmp.replace(output_path)
    wall_s = time.perf_counter() - started
    return finalize_stream_summary(accumulator, total_pairs=completed, workers=workers, wall_s=wall_s)


def task_from_pair_row(index: int, row: dict[str, str], timeout_s: int) -> tuple[int, str, str, str, str, str, int]:
    return (
        index,
        row["molecule_a_smiles"],
        row["molecule_b_smiles"],
        row["similarity_bucket"],
        row["tanimoto"],
        row["label"],
        timeout_s,
    )


def run_mcs_jobs_progress(
    tasks: list[tuple[int, str, str, str, str, str, int]],
    *,
    workers: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(tasks)
    completed = 0
    timed_out = 0
    started = time.perf_counter()
    last_reported = 0
    max_pending = max(workers * 8, workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        pending = {}
        task_iter = iter(tasks)
        for _ in range(min(max_pending, total)):
            task = next(task_iter)
            pending[executor.submit(compute_mcs_row, task)] = None
        while pending:
            done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                row = future.result()
                rows.append(row)
                completed += 1
                if row.get("timed_out"):
                    timed_out += 1
                if should_report_progress(completed, total, last_reported, progress_every):
                    log_progress(completed, total, timed_out, started)
                    last_reported = completed
                try:
                    task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(compute_mcs_row, task)] = None
    return rows


def compute_mcs_batch(tasks: list[tuple[int, str, str, str, str, str, int]]) -> list[dict[str, Any]]:
    return [compute_mcs_row(task) for task in tasks]


def should_report_progress(completed: int, total: int, last_reported: int, progress_every: int) -> bool:
    if completed >= total:
        return True
    return progress_every > 0 and completed - last_reported >= progress_every


def log_progress(completed: int, total: int, timed_out: int, started: float) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_s = safe_div(remaining, rate)
    pct = 100.0 * safe_div(completed, total)
    log(
        f"progress completed={completed:,}/{total:,} ({pct:.2f}%) "
        f"rate={rate:.2f}/s elapsed={format_seconds(elapsed)} "
        f"eta={format_seconds(eta_s)} timeouts={timed_out:,}"
    )


def worker_init() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def compute_mcs_row(task: tuple[int, str, str, str, str, str, int]) -> dict[str, Any]:
    index, smiles_a, smiles_b, bucket, tanimoto, label, timeout_s = task
    start = time.perf_counter()
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFMCS

        RDLogger.DisableLog("rdApp.*")
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return error_row(index, bucket, tanimoto, label, start, "invalid_smiles")
        heavy_a = mol_a.GetNumHeavyAtoms()
        heavy_b = mol_b.GetNumHeavyAtoms()
        result = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=max(1, int(timeout_s)),
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            matchValences=False,
        )
        elapsed_s = time.perf_counter() - start
        mcs_atoms = int(result.numAtoms or 0)
        query_coverage = safe_div(mcs_atoms, heavy_a)
        reference_coverage = safe_div(mcs_atoms, heavy_b)
        return {
            "index": index,
            "similarity_bucket": bucket,
            "tanimoto": float(tanimoto),
            "label": label,
            "status": "ok",
            "timed_out": bool(result.canceled),
            "elapsed_s": elapsed_s,
            "query_heavy_atoms": heavy_a,
            "reference_heavy_atoms": heavy_b,
            "mcs_atoms": mcs_atoms,
            "query_mcs_coverage": query_coverage,
            "reference_mcs_coverage": reference_coverage,
            "mean_mcs_coverage": (query_coverage + reference_coverage) / 2.0,
        }
    except Exception as exc:  # pragma: no cover - diagnostics script
        return error_row(index, bucket, tanimoto, label, start, f"{type(exc).__name__}: {exc}")


def error_row(index: int, bucket: str, tanimoto: str, label: str, start: float, error: str) -> dict[str, Any]:
    return {
        "index": index,
        "similarity_bucket": bucket,
        "tanimoto": float(tanimoto),
        "label": label,
        "status": "error",
        "timed_out": False,
        "elapsed_s": time.perf_counter() - start,
        "query_heavy_atoms": 0,
        "reference_heavy_atoms": 0,
        "mcs_atoms": 0,
        "query_mcs_coverage": 0.0,
        "reference_mcs_coverage": 0.0,
        "mean_mcs_coverage": 0.0,
        "error": error,
    }


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    normalized["timed_out"] = parse_bool(normalized["timed_out"])
    return normalized


def new_stream_accumulator() -> dict[str, Any]:
    return {
        "rows": [],
        "bucket_totals": Counter(),
        "by_bucket": {
            bucket: {
                "n": 0,
                "ok": 0,
                "errors": 0,
                "timeouts": 0,
                "elapsed": [],
                "coverages": [],
            }
            for bucket in SIMILARITY_BUCKETS
        },
    }


def update_stream_accumulator(accumulator: dict[str, Any], row: dict[str, Any]) -> None:
    accumulator["rows"].append(
        {
            "status": row["status"],
            "timed_out": row["timed_out"],
            "elapsed_s": float(row["elapsed_s"]),
            "mean_mcs_coverage": float(row["mean_mcs_coverage"] or 0.0),
            "similarity_bucket": row["similarity_bucket"],
        }
    )
    bucket_stats = accumulator["by_bucket"][row["similarity_bucket"]]
    bucket_stats["n"] += 1
    bucket_stats["elapsed"].append(float(row["elapsed_s"]))
    if row["status"] == "ok":
        bucket_stats["ok"] += 1
        bucket_stats["coverages"].append(float(row["mean_mcs_coverage"] or 0.0))
        if row["timed_out"]:
            bucket_stats["timeouts"] += 1
    else:
        bucket_stats["errors"] += 1


def finalize_stream_summary(
    accumulator: dict[str, Any],
    *,
    total_pairs: int,
    workers: int,
    wall_s: float,
) -> dict[str, Any]:
    rows = accumulator["rows"]
    bucket_totals = accumulator["bucket_totals"]
    summary = summarize_results(rows, bucket_totals, total_pairs, workers, wall_s)
    for bucket in SIMILARITY_BUCKETS:
        bucket_stats = accumulator["by_bucket"][bucket]
        summary["by_bucket"][bucket].update(
            {
                "n": bucket_stats["n"],
                "ok": bucket_stats["ok"],
                "errors": bucket_stats["errors"],
                "timeouts": bucket_stats["timeouts"],
                "mean_elapsed_s": mean(bucket_stats["elapsed"]),
                "median_elapsed_s": percentile(bucket_stats["elapsed"], 0.50),
                "p90_elapsed_s": percentile(bucket_stats["elapsed"], 0.90),
                "p99_elapsed_s": percentile(bucket_stats["elapsed"], 0.99),
                "mean_mcs_coverage": mean(bucket_stats["coverages"]),
                "median_mcs_coverage": percentile(bucket_stats["coverages"], 0.50),
            }
        )
    return summary


def summarize_results(
    rows: list[dict[str, Any]],
    bucket_totals: Counter,
    total_pairs: int,
    workers: int,
    wall_s: float,
) -> dict[str, Any]:
    overall = summarize_subset(rows)
    observed_pairs_per_s = safe_div(len(rows), wall_s)
    overall.update(
        {
            "sampled_pairs": len(rows),
            "total_pairs": total_pairs,
            "workers": workers,
            "wall_s": wall_s,
            "observed_pairs_per_s": observed_pairs_per_s,
            "estimated_full_wall_s_observed_throughput": safe_div(total_pairs, observed_pairs_per_s),
        }
    )

    by_bucket = {}
    weighted_serial_s = 0.0
    for bucket in SIMILARITY_BUCKETS:
        bucket_rows = [row for row in rows if row["similarity_bucket"] == bucket]
        stats = summarize_subset(bucket_rows)
        total_bucket_pairs = int(bucket_totals.get(bucket, 0))
        stats["total_bucket_pairs"] = total_bucket_pairs
        stats["estimated_serial_s_for_bucket"] = stats["mean_elapsed_s"] * total_bucket_pairs
        weighted_serial_s += stats["estimated_serial_s_for_bucket"]
        by_bucket[bucket] = stats
    overall["bucket_weighted_estimated_serial_s"] = weighted_serial_s
    overall["bucket_weighted_estimated_parallel_s"] = safe_div(weighted_serial_s, workers)
    return {
        "overall": overall,
        "by_bucket": by_bucket,
    }


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    elapsed = [float(row["elapsed_s"]) for row in rows]
    coverages = [float(row["mean_mcs_coverage"]) for row in ok_rows]
    return {
        "n": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "timeouts": sum(1 for row in ok_rows if row["timed_out"]),
        "mean_elapsed_s": mean(elapsed),
        "median_elapsed_s": percentile(elapsed, 0.50),
        "p90_elapsed_s": percentile(elapsed, 0.90),
        "p99_elapsed_s": percentile(elapsed, 0.99),
        "mean_mcs_coverage": mean(coverages),
        "median_mcs_coverage": percentile(coverages, 0.50),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    completion = summary.get("completion", {})
    lines = [
        "# MCS runtime benchmark 报告",
        "",
        "## 总览",
        "",
        f"- sampled pairs: {int(overall['sampled_pairs']):,}",
        f"- total scanned pairs: {int(overall['total_pairs']):,}",
        f"- workers: {int(overall['workers']):,}",
        f"- wall time: {float(overall['wall_s']):.2f} s",
        f"- observed throughput: {float(overall['observed_pairs_per_s']):.2f} pairs/s",
        f"- observed-throughput full estimate: {format_seconds(float(overall['estimated_full_wall_s_observed_throughput']))}",
        f"- bucket-weighted ideal parallel estimate: {format_seconds(float(overall['bucket_weighted_estimated_parallel_s']))}",
        f"- timeout count: {int(overall['timeouts']):,}",
    ]
    if completion:
        lines.extend(
            [
                f"- completion mode: {completion.get('mode', 'unknown')}",
                f"- result file is partial: {completion.get('result_file_is_partial', False)}",
                f"- result rows / expected pairs: {int(completion['result_rows']):,} / {int(completion['expected_pairs']):,}",
                f"- missing result count: {int(completion['missing_result_count']):,}",
                f"- completion rate: {100.0 * float(completion['completion_rate']):.4f}%",
                "- note: wall time and observed throughput are for finalizing existing TSV, not the original MCS run.",
            ]
        )
    lines.extend(
        [
            "",
            "## By Bucket",
            "",
            "| bucket | sample n | total pairs | mean s/pair | p90 s | p99 s | timeout | mean MCS coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in SIMILARITY_BUCKETS:
        stats = summary["by_bucket"][bucket]
        lines.append(
            "| {bucket} | {n:,} | {total:,} | {mean_s:.4f} | {p90:.4f} | {p99:.4f} | {timeouts:,} | {coverage:.3f} |".format(
                bucket=bucket.replace("_analog", ""),
                n=int(stats["n"]),
                total=int(stats["total_bucket_pairs"]),
                mean_s=float(stats["mean_elapsed_s"]),
                p90=float(stats["p90_elapsed_s"]),
                p99=float(stats["p99_elapsed_s"]),
                timeouts=int(stats["timeouts"]),
                coverage=float(stats["mean_mcs_coverage"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h{minutes:02d}m{secs:04.1f}s"
    if minutes:
        return f"{minutes}m{secs:04.1f}s"
    return f"{secs:.1f}s"


def log(message: str) -> None:
    print(f"[mcs_runtime] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
