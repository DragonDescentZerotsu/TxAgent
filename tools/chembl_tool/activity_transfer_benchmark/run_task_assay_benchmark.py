"""Build task-scoped ChEMBL assay activity-transfer benchmarks.

This variant restricts endpoints to the assay/activity evidence gathered for
the four downstream task pipelines. It supports raw-value transfer labels based
on assay-internal robust sigma, plus the legacy pChEMBL delta label where
pChEMBL values are available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from tools.chembl_tool.activity_transfer_benchmark.run_benchmark import (
    DEFAULT_FPS_GZ,
    DEFAULT_OUT_ROOT,
    DEFAULT_THRESHOLDS,
    SIMILARITY_BUCKETS,
    Confusion,
    format_elapsed,
    format_float,
    load_fingerprints,
    log,
    mean_or_zero,
    parse_thresholds,
    quantile,
    similarity_bucket,
    stable_int,
    tanimoto_int,
    write_tsv,
)


DEFAULT_TASK_EVIDENCE = [
    "bbb_martins=outputs/chembl_tool/tasks/bbb_martins/assay_screening/v6/bbb_activity_evidence.csv",
    "bioavailability_ma=outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_activity_evidence.csv",
    "clintox=outputs/chembl_tool/tasks/clintox/assay_screening/v6/clintox_activity_evidence.csv",
    "skin_reaction=outputs/chembl_tool/tasks/skin_reaction/assay_screening/v1/skin_reaction_activity_evidence.csv",
]
DEFAULT_LABEL_MODES = ("raw_robust_z", "log_raw_robust_z", "pchembl_delta")


@dataclass(frozen=True)
class AssayMeta:
    task_name: str
    assay_chembl_id: str
    assay_id: str
    tier: str
    score: str
    assay_type: str
    description: str
    target_chembl_id: str
    target_pref_name: str
    organism: str
    confidence_score: str
    relationship_type: str
    reason: str


@dataclass(frozen=True)
class MoleculeValue:
    chembl_id: str
    smiles: str
    raw_value: float
    log_raw_value: float | None
    pchembl_value: float | None
    n_records: int
    fp_int: int = 0
    fp_popcount: int = 0


@dataclass(frozen=True)
class EndpointStats:
    task_name: str
    assay_chembl_id: str
    assay_id: str
    standard_type: str
    raw_units: str
    raw_value_min: float
    raw_value_q25: float
    raw_value_median: float
    raw_value_q75: float
    raw_value_max: float
    raw_value_std: float
    raw_value_iqr: float
    raw_value_mad: float
    raw_value_robust_sigma: float
    log_raw_value_min: float | None
    log_raw_value_q25: float | None
    log_raw_value_median: float | None
    log_raw_value_q75: float | None
    log_raw_value_max: float | None
    log_raw_value_std: float | None
    log_raw_value_iqr: float | None
    log_raw_value_mad: float | None
    log_raw_value_robust_sigma: float | None
    pchembl_min: float | None
    pchembl_q25: float | None
    pchembl_median: float | None
    pchembl_q75: float | None
    pchembl_max: float | None
    pchembl_std: float | None
    pchembl_iqr: float | None
    pchembl_mad: float | None
    pchembl_robust_sigma: float | None
    n_molecules_loaded: int
    n_records_loaded: int
    meta: AssayMeta | None


@dataclass(frozen=True)
class WorkItem:
    label_mode: str
    stats: EndpointStats
    n_molecules_with_fp: int
    pair_budget: int
    molecules: tuple[MoleculeValue, ...]
    seed: int


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    run_id = args.run_id or datetime.now().strftime("task_assay_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"task assay benchmark output directory: {out_dir}")

    task_sources = parse_task_sources(args.task_evidence or list(DEFAULT_TASK_EVIDENCE))
    label_modes = parse_label_modes(args.label_modes)
    thresholds = parse_thresholds(args.thresholds)

    metas = load_assay_metadata(task_sources)
    groups, source_summary = load_task_activity_groups(task_sources, args)
    if args.endpoint_limit:
        groups = dict(list(groups.items())[: args.endpoint_limit])
    log(f"loaded task endpoint groups: {len(groups):,}")

    stats_by_key, molecule_groups, endpoint_rows = finalize_groups(groups, metas, args)
    log(f"retained endpoint groups after molecule filters: {len(molecule_groups):,}")
    write_tsv(out_dir / "endpoint_summary.tsv", endpoint_rows)

    needed_ids = {
        molecule.chembl_id
        for molecules in molecule_groups.values()
        for molecule in molecules
    }
    log(f"unique molecules needing fingerprints: {len(needed_ids):,}")
    fingerprints = load_fingerprints(Path(args.fps_gz), needed_ids)
    log(f"fingerprints loaded: {len(fingerprints):,}/{len(needed_ids):,}")

    mode_summaries: dict[str, Any] = {}
    for label_mode in label_modes:
        mode_dir = out_dir / label_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        items, mode_endpoint_rows = prepare_work_items(label_mode, molecule_groups, stats_by_key, fingerprints, args)
        mode_endpoint_rows.sort(key=lambda row: (-int(row["n_molecules_with_fp"]), row["task_name"], row["assay_chembl_id"], row["standard_type"], row["raw_units"]))
        write_tsv(mode_dir / "endpoint_summary.tsv", mode_endpoint_rows)
        log(
            f"{label_mode}: endpoints={len(items):,} "
            f"pair_budget={sum(item.pair_budget for item in items):,}"
        )
        if not items:
            mode_summaries[label_mode] = {"status": "no endpoints with enough fingerprints/sigma"}
            continue
        result = run_mode_benchmark(label_mode, items, thresholds, mode_dir, args)
        mode_summaries[label_mode] = result

    summary = {
        "run_id": run_id,
        "elapsed_s": round(time.monotonic() - started, 3),
        "task_sources": {task: str(path) for task, path in task_sources.items()},
        "source_summary": source_summary,
        "label_modes": mode_summaries,
        "parameters": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out_dir / "report_zh.md", summary)
    log(f"finished task assay benchmark in {format_elapsed(time.monotonic() - started)}")
    log(f"report: {out_dir / 'report_zh.md'}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps-gz", default=DEFAULT_FPS_GZ)
    parser.add_argument("--out-root", default=f"{DEFAULT_OUT_ROOT}/task_assay_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--task-evidence", action="append", default=[], help="Task evidence as task_name=path. May be repeated. Defaults to the four current task evidence CSVs when omitted.")
    parser.add_argument("--label-modes", default=",".join(DEFAULT_LABEL_MODES), help="Comma-separated subset of raw_robust_z,log_raw_robust_z,pchembl_delta.")
    parser.add_argument("--min-molecules-per-endpoint", type=int, default=10)
    parser.add_argument("--min-molecules-with-fp", type=int, default=10)
    parser.add_argument("--max-pairs-per-endpoint", type=int, default=5000)
    parser.add_argument("--max-total-pairs-per-mode", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=max(1, min(128, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--similar-z", type=float, default=0.5)
    parser.add_argument("--different-z", type=float, default=1.0)
    parser.add_argument("--similar-delta", type=float, default=0.5)
    parser.add_argument("--different-delta", type=float, default=1.0)
    parser.add_argument("--min-robust-sigma", type=float, default=1e-12)
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--endpoint-limit", type=int, default=0, help="Smoke-test limit after grouping.")
    parser.add_argument("--progress-every-endpoints", type=int, default=250)
    parser.add_argument("--include-invalid-activities", action="store_true")
    parser.add_argument("--include-non-equal-relations", action="store_true")
    return parser.parse_args(argv)


def parse_task_sources(raw_sources: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for raw in raw_sources:
        if "=" not in raw:
            raise ValueError(f"Expected task evidence as task_name=path, got: {raw}")
        task, path = raw.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Empty task name in task evidence argument: {raw}")
        evidence_path = Path(path.strip())
        if not evidence_path.exists():
            raise FileNotFoundError(f"Task evidence CSV not found for {task}: {evidence_path}")
        sources[task] = evidence_path
    return sources


def parse_label_modes(raw: str) -> list[str]:
    allowed = set(DEFAULT_LABEL_MODES)
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(modes).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown label mode(s): {', '.join(unknown)}")
    if not modes:
        raise ValueError("At least one label mode is required.")
    return modes


def load_assay_metadata(task_sources: dict[str, Path]) -> dict[tuple[str, str], AssayMeta]:
    metas: dict[tuple[str, str], AssayMeta] = {}
    for task_name, evidence_path in task_sources.items():
        candidates_path = candidate_path_for_evidence(evidence_path)
        if not candidates_path.exists():
            log(f"{task_name}: assay candidate metadata not found: {candidates_path}")
            continue
        with candidates_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                assay_chembl_id = clean_text(row.get("assay_chembl_id"))
                if not assay_chembl_id:
                    continue
                metas[(task_name, assay_chembl_id)] = AssayMeta(
                    task_name=task_name,
                    assay_chembl_id=assay_chembl_id,
                    assay_id=clean_text(row.get("assay_id")),
                    tier=clean_text(row.get("tier")),
                    score=clean_text(row.get("score")),
                    assay_type=clean_text(row.get("assay_type")),
                    description=clean_text(row.get("description")),
                    target_chembl_id=clean_text(row.get("target_chembl_id")),
                    target_pref_name=clean_text(row.get("target_pref_name")),
                    organism=clean_text(row.get("organism")),
                    confidence_score=clean_text(row.get("confidence_score")),
                    relationship_type=clean_text(row.get("relationship_type")),
                    reason=clean_text(row.get("reason")),
                )
    return metas


def candidate_path_for_evidence(evidence_path: Path) -> Path:
    name = evidence_path.name
    if name.endswith("_activity_evidence.csv"):
        return evidence_path.with_name(name.replace("_activity_evidence.csv", "_assay_candidates.csv"))
    return evidence_path.with_name("assay_candidates.csv")


def load_task_activity_groups(
    task_sources: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str, str, str], dict[str, dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    source_summary: dict[str, Any] = {}
    for task_name, evidence_path in task_sources.items():
        rows_seen = 0
        rows_used = 0
        rows_bad_value = 0
        rows_bad_units = 0
        rows_non_equal = 0
        rows_invalid = 0
        with evidence_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_seen += 1
                relation = clean_text(row.get("standard_relation"))
                if relation and relation != "=" and not args.include_non_equal_relations:
                    rows_non_equal += 1
                    continue
                if clean_text(row.get("data_validity_comment")) and not args.include_invalid_activities:
                    rows_invalid += 1
                    continue
                value = parse_float(row.get("standard_value"))
                if value is None or not math.isfinite(value):
                    rows_bad_value += 1
                    continue
                normalized = normalize_standard_value(value, row.get("standard_units"))
                if normalized is None:
                    rows_bad_units += 1
                    continue
                normalized_value, normalized_units = normalized
                assay_chembl_id = clean_text(row.get("assay_chembl_id"))
                molecule_chembl_id = clean_text(row.get("molecule_chembl_id"))
                smiles = clean_text(row.get("canonical_smiles"))
                standard_type = normalize_standard_type(row.get("standard_type"))
                if not assay_chembl_id or not molecule_chembl_id or not smiles or not standard_type:
                    rows_bad_value += 1
                    continue
                key = (task_name, assay_chembl_id, standard_type, normalized_units)
                entry = groups[key].get(molecule_chembl_id)
                if entry is None:
                    entry = {
                        "smiles_counts": Counter(),
                        "raw_values": [],
                        "pchembl_values": [],
                    }
                    groups[key][molecule_chembl_id] = entry
                entry["smiles_counts"][smiles] += 1
                entry["raw_values"].append(normalized_value)
                pchembl = parse_float(row.get("pchembl_value"))
                if pchembl is not None and math.isfinite(pchembl):
                    entry["pchembl_values"].append(pchembl)
                rows_used += 1
        source_summary[task_name] = {
            "path": str(evidence_path),
            "rows_seen": rows_seen,
            "rows_used": rows_used,
            "rows_bad_value": rows_bad_value,
            "rows_bad_units": rows_bad_units,
            "rows_non_equal_relation": rows_non_equal,
            "rows_invalid": rows_invalid,
        }
        log(f"{task_name}: used {rows_used:,}/{rows_seen:,} activity evidence rows")
    return groups, source_summary


def normalize_standard_type(value: Any) -> str:
    return " ".join(clean_text(value).lower().split())


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_standard_value(value: float, units: Any) -> tuple[float, str] | None:
    unit = normalize_unit_text(units)
    if not unit:
        unit = "unitless"
    concentration_factors = {
        "pm": 0.001,
        "nm": 1.0,
        "um": 1000.0,
        "mm": 1_000_000.0,
        "m": 1_000_000_000.0,
    }
    if unit in concentration_factors:
        return value * concentration_factors[unit], "nM"
    scaled_units = {
        "cm s-1": (1.0, "cm/s"),
        "cm/s": (1.0, "cm/s"),
        "10^-6 cm/s": (1e-6, "cm/s"),
        "10'-6 cm/s": (1e-6, "cm/s"),
        "1e-6 cm/s": (1e-6, "cm/s"),
        "ucm/s": (1e-6, "cm/s"),
        "um/s": (1e-6, "cm/s"),
        "nm/s": (1e-7, "cm/s"),
        "nm s-1": (1e-7, "cm/s"),
        "ml/min": (1.0, "mL/min"),
        "ml.min-1": (1.0, "mL/min"),
        "ul/min": (0.001, "mL/min"),
        "ul.min-1": (0.001, "mL/min"),
        "ml.min-1.kg-1": (1.0, "mL/min/kg"),
        "ml/min/kg": (1.0, "mL/min/kg"),
        "ml/min.kg": (1.0, "mL/min/kg"),
        "ul.min-1.kg-1": (0.001, "mL/min/kg"),
        "ul/min/kg": (0.001, "mL/min/kg"),
        "ml.min-1.g-1": (1.0, "mL/min/g"),
        "ml/min/g": (1.0, "mL/min/g"),
        "ml/min.g": (1.0, "mL/min/g"),
        "ul.min-1.g-1": (0.001, "mL/min/g"),
        "ul/min/g": (0.001, "mL/min/g"),
        "ul.min-1.(10^6cells)-1": (0.001, "mL/min/10^6cells"),
        "ul/min/10^6cells": (0.001, "mL/min/10^6cells"),
        "microl/min/mg": (0.001, "mL/min/mg"),
        "ul/min/mg": (0.001, "mL/min/mg"),
        "ul.min-1.mg-1": (0.001, "mL/min/mg"),
    }
    if unit in scaled_units:
        factor, normalized_unit = scaled_units[unit]
        return value * factor, normalized_unit
    aliases = {
        "%": "%",
        "percent": "%",
        "ratio": "ratio",
        "fold": "fold",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "s-1": "s^-1",
        "ug/ml": "ug/mL",
        "ug.ml-1": "ug/mL",
        "ug.mL-1": "ug/mL",
        "ug ml-1": "ug/mL",
        "mu/ml": "ug/mL",
        "ng/ml": "ng/mL",
        "ng.ml-1": "ng/mL",
        "ng ml-1": "ng/mL",
        "mg kg-1": "mg/kg",
        "mg.kg-1": "mg/kg",
        "um kg-1": "umol/kg",
        "umol.kg-1": "umol/kg",
        "umol/min/mg": "umol/min/mg",
        "nmol/min/mg": "nmol/min/mg",
        "nmol/mg.min": "nmol/min/mg",
        "nmol/mg/min": "nmol/min/mg",
        "ng.hr.ml-1": "ng*h/mL",
        "ng*h/ml": "ng*h/mL",
        "unitless": "unitless",
    }
    return value, aliases.get(unit, unit)


def normalize_unit_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("−", "-").replace("–", "-")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    lower = text.lower()
    lower = lower.replace("per ", "/")
    lower = lower.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
    lower = lower.replace("u l", "ul")
    lower = lower.replace("ml.", "ml.")
    if lower in {"nanomolar", "nanomole/l", "nmol/l", "nmol l-1"}:
        return "nm"
    if lower in {"micromolar", "umol/l", "umol l-1"}:
        return "um"
    if lower in {"millimolar", "mmol/l", "mmol l-1"}:
        return "mm"
    return lower


def finalize_groups(
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]],
    metas: dict[tuple[str, str], AssayMeta],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str, str, str], EndpointStats], dict[tuple[str, str, str, str], list[MoleculeValue]], list[dict[str, Any]]]:
    stats_by_key: dict[tuple[str, str, str, str], EndpointStats] = {}
    molecule_groups: dict[tuple[str, str, str, str], list[MoleculeValue]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    for key, raw_molecules in groups.items():
        task_name, assay_chembl_id, standard_type, raw_units = key
        molecules = []
        n_records = 0
        for molecule_chembl_id, entry in raw_molecules.items():
            raw_values = sorted(float(v) for v in entry["raw_values"])
            if not raw_values:
                continue
            pchembl_values = sorted(float(v) for v in entry["pchembl_values"])
            smiles = entry["smiles_counts"].most_common(1)[0][0]
            raw_value = median(raw_values)
            molecules.append(
                MoleculeValue(
                    chembl_id=molecule_chembl_id,
                    smiles=smiles,
                    raw_value=raw_value,
                    log_raw_value=math.log10(raw_value) if raw_value > 0 else None,
                    pchembl_value=median(pchembl_values) if pchembl_values else None,
                    n_records=len(raw_values),
                )
            )
            n_records += len(raw_values)
        if len(molecules) < args.min_molecules_per_endpoint:
            continue
        raw_values = [m.raw_value for m in molecules]
        log_values = [m.log_raw_value for m in molecules if m.log_raw_value is not None]
        pchembl_values = [m.pchembl_value for m in molecules if m.pchembl_value is not None]
        meta = metas.get((task_name, assay_chembl_id))
        stats = EndpointStats(
            task_name=task_name,
            assay_chembl_id=assay_chembl_id,
            assay_id=meta.assay_id if meta else "",
            standard_type=standard_type,
            raw_units=raw_units,
            n_molecules_loaded=len(molecules),
            n_records_loaded=n_records,
            meta=meta,
            **stats_fields("raw_value", raw_values),
            **nullable_stats_fields("log_raw_value", log_values),
            **nullable_stats_fields("pchembl", pchembl_values),
        )
        stats_by_key[key] = stats
        molecule_groups[key] = molecules
        endpoint_rows.append(endpoint_summary_row(stats, 0, 0, "all"))
    endpoint_rows.sort(key=lambda row: (-int(row["n_molecules_loaded"]), row["task_name"], row["assay_chembl_id"], row["standard_type"], row["raw_units"]))
    return stats_by_key, molecule_groups, endpoint_rows


def stats_fields(prefix: str, values: list[float]) -> dict[str, float]:
    values = sorted(values)
    q25 = quantile(values, 0.25)
    q50 = quantile(values, 0.50)
    q75 = quantile(values, 0.75)
    iqr = q75 - q25
    mad = median([abs(value - q50) for value in values])
    return {
        f"{prefix}_min": values[0],
        f"{prefix}_q25": q25,
        f"{prefix}_median": q50,
        f"{prefix}_q75": q75,
        f"{prefix}_max": values[-1],
        f"{prefix}_std": standard_deviation(values),
        f"{prefix}_iqr": iqr,
        f"{prefix}_mad": mad,
        f"{prefix}_robust_sigma": robust_sigma(iqr, mad),
    }


def nullable_stats_fields(prefix: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_q25": None,
            f"{prefix}_median": None,
            f"{prefix}_q75": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
            f"{prefix}_iqr": None,
            f"{prefix}_mad": None,
            f"{prefix}_robust_sigma": None,
        }
    return stats_fields(prefix, values)


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def robust_sigma(iqr: float, mad: float) -> float:
    if iqr > 0:
        return iqr / 1.349
    if mad > 0:
        return mad * 1.4826
    return 0.0


def endpoint_summary_row(stats: EndpointStats, n_with_fp: int, total_pairs: int, label_mode: str) -> dict[str, Any]:
    meta = stats.meta
    return {
        "task_name": stats.task_name,
        "assay_chembl_id": stats.assay_chembl_id,
        "assay_id": stats.assay_id,
        "standard_type": stats.standard_type,
        "raw_units": stats.raw_units,
        "label_mode": label_mode,
        "n_molecules_loaded": stats.n_molecules_loaded,
        "n_records_loaded": stats.n_records_loaded,
        "n_molecules_with_fp": n_with_fp,
        "all_possible_pairs_with_fp": total_pairs,
        "raw_value_min": fmt(stats.raw_value_min),
        "raw_value_q25": fmt(stats.raw_value_q25),
        "raw_value_median": fmt(stats.raw_value_median),
        "raw_value_q75": fmt(stats.raw_value_q75),
        "raw_value_max": fmt(stats.raw_value_max),
        "raw_value_std": fmt(stats.raw_value_std),
        "raw_value_iqr": fmt(stats.raw_value_iqr),
        "raw_value_mad": fmt(stats.raw_value_mad),
        "raw_value_robust_sigma": fmt(stats.raw_value_robust_sigma),
        "log_raw_value_median": fmt(stats.log_raw_value_median),
        "log_raw_value_iqr": fmt(stats.log_raw_value_iqr),
        "log_raw_value_robust_sigma": fmt(stats.log_raw_value_robust_sigma),
        "pchembl_median": fmt(stats.pchembl_median),
        "pchembl_iqr": fmt(stats.pchembl_iqr),
        "tier": meta.tier if meta else "",
        "score": meta.score if meta else "",
        "assay_type": meta.assay_type if meta else "",
        "target_chembl_id": meta.target_chembl_id if meta else "",
        "target_pref_name": meta.target_pref_name if meta else "",
        "organism": meta.organism if meta else "",
        "confidence_score": meta.confidence_score if meta else "",
        "relationship_type": meta.relationship_type if meta else "",
        "description": meta.description if meta else "",
        "reason": meta.reason if meta else "",
    }


def prepare_work_items(
    label_mode: str,
    molecule_groups: dict[tuple[str, str, str, str], list[MoleculeValue]],
    stats_by_key: dict[tuple[str, str, str, str], EndpointStats],
    fingerprints: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[list[WorkItem], list[dict[str, Any]]]:
    raw_items: list[WorkItem] = []
    endpoint_rows: list[dict[str, Any]] = []
    for key, molecules in molecule_groups.items():
        stats = stats_by_key[key]
        with_fp = []
        for molecule in molecules:
            if label_mode == "log_raw_robust_z" and molecule.log_raw_value is None:
                continue
            if label_mode == "pchembl_delta" and molecule.pchembl_value is None:
                continue
            fp = fingerprints.get(molecule.chembl_id)
            if fp is None:
                continue
            with_fp.append(replace(molecule, fp_int=fp[0], fp_popcount=fp[1]))
        n = len(with_fp)
        total_pairs = n * (n - 1) // 2
        endpoint_rows.append(endpoint_summary_row(stats, n, total_pairs, label_mode))
        if n < args.min_molecules_with_fp:
            continue
        if label_mode == "raw_robust_z" and stats.raw_value_robust_sigma <= args.min_robust_sigma:
            continue
        if label_mode == "log_raw_robust_z" and (stats.log_raw_value_robust_sigma is None or stats.log_raw_value_robust_sigma <= args.min_robust_sigma):
            continue
        budget = min(args.max_pairs_per_endpoint, total_pairs)
        if budget <= 0:
            continue
        raw_items.append(
            WorkItem(
                label_mode=label_mode,
                stats=stats,
                n_molecules_with_fp=n,
                pair_budget=budget,
                molecules=tuple(with_fp),
                seed=args.seed + stable_int("|".join(key) + "|" + label_mode),
            )
        )
    return allocate_pair_budgets(raw_items, args.max_total_pairs_per_mode), endpoint_rows


def allocate_pair_budgets(items: list[WorkItem], max_total_pairs: int) -> list[WorkItem]:
    total = sum(item.pair_budget for item in items)
    if total <= max_total_pairs:
        return items
    scale = max_total_pairs / total
    allocated: list[WorkItem] = []
    used = 0
    for item in items:
        budget = max(1, int(item.pair_budget * scale))
        if used + budget > max_total_pairs:
            budget = max_total_pairs - used
        if budget <= 0:
            break
        allocated.append(replace(item, pair_budget=budget))
        used += budget
    return allocated


def run_mode_benchmark(label_mode: str, items: list[WorkItem], thresholds: list[float], mode_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    pair_path = mode_dir / "pairs.tsv.gz"
    threshold_path = mode_dir / "threshold_metrics.tsv"
    bucket_path = mode_dir / "similarity_bucket_summary.tsv"
    task_path = mode_dir / "task_summary.tsv"
    fieldnames = pair_fieldnames()
    threshold_confusions = {threshold: Confusion() for threshold in thresholds}
    bucket_stats = new_bucket_stats()
    task_stats = defaultdict(new_label_stats)
    label_counts = Counter()
    deltas: list[float] = []
    pair_count = 0
    completed = 0
    started = time.monotonic()
    with gzip.open(pair_path, "wt", encoding="utf-8", newline="") as pair_handle:
        writer = csv.DictWriter(pair_handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        if args.workers <= 1:
            for item in items:
                rows = build_group_pairs(item, args.similar_z, args.different_z, args.similar_delta, args.different_delta)
                completed, pair_count = consume_rows(rows, writer, thresholds, threshold_confusions, bucket_stats, task_stats, label_counts, deltas, completed, pair_count, len(items), args.progress_every_endpoints, started)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = {}
                item_iter = iter(items)
                for _ in range(min(len(items), args.workers * 4)):
                    item = next(item_iter)
                    pending[executor.submit(build_group_pairs, item, args.similar_z, args.different_z, args.similar_delta, args.different_delta)] = item
                while pending:
                    future = next(as_completed(pending))
                    pending.pop(future)
                    rows = future.result()
                    completed, pair_count = consume_rows(rows, writer, thresholds, threshold_confusions, bucket_stats, task_stats, label_counts, deltas, completed, pair_count, len(items), args.progress_every_endpoints, started)
                    try:
                        item = next(item_iter)
                    except StopIteration:
                        continue
                    pending[executor.submit(build_group_pairs, item, args.similar_z, args.different_z, args.similar_delta, args.different_delta)] = item
    threshold_rows = []
    best_row: dict[str, Any] | None = None
    for threshold in thresholds:
        row = {"threshold": threshold, **threshold_confusions[threshold].metrics()}
        threshold_rows.append(row)
        if best_row is None or float(row["macro_f1"]) > float(best_row["macro_f1"]):
            best_row = row
    write_tsv(threshold_path, threshold_rows)
    write_tsv(bucket_path, finalize_label_stats(bucket_stats, "similarity_bucket"))
    write_tsv(task_path, finalize_label_stats(task_stats, "task_name"))
    non_ambiguous = label_counts["similar"] + label_counts["different"]
    return {
        "status": "ok",
        "n_endpoint_groups": len(items),
        "n_pairs": pair_count,
        "label_counts": dict(label_counts),
        "n_non_ambiguous_pairs": non_ambiguous,
        "majority_baseline_accuracy": max(label_counts["similar"], label_counts["different"]) / non_ambiguous if non_ambiguous else 0.0,
        "best_threshold_by_macro_f1": best_row or {},
        "mean_abs_activity_delta": mean_or_zero(deltas),
        "median_abs_activity_delta": median(deltas) if deltas else 0.0,
        "files": {
            "pairs": str(pair_path),
            "endpoint_summary": str(mode_dir / "endpoint_summary.tsv"),
            "threshold_metrics": str(threshold_path),
            "bucket_summary": str(bucket_path),
            "task_summary": str(task_path),
        },
    }


def pair_fieldnames() -> list[str]:
    return [
        "task_name",
        "assay_chembl_id",
        "assay_id",
        "standard_type",
        "raw_units",
        "label_mode",
        "molecule_a_chembl_id",
        "molecule_b_chembl_id",
        "molecule_a_smiles",
        "molecule_b_smiles",
        "raw_activity_a",
        "raw_activity_b",
        "log_raw_activity_a",
        "log_raw_activity_b",
        "pchembl_a",
        "pchembl_b",
        "activity_a",
        "activity_b",
        "abs_activity_delta",
        "normalized_delta",
        "robust_sigma",
        "tanimoto",
        "similarity_bucket",
        "label",
        "target_chembl_id",
        "target_pref_name",
        "assay_tier",
        "assay_description",
    ]


def build_group_pairs(item: WorkItem, similar_z: float, different_z: float, similar_delta: float, different_delta: float) -> list[dict[str, Any]]:
    molecules = item.molecules
    n = len(molecules)
    rng = random.Random(item.seed)
    total_pairs = n * (n - 1) // 2
    if item.pair_budget >= total_pairs:
        pairs = ((i, j) for i in range(n - 1) for j in range(i + 1, n))
    else:
        selected: set[tuple[int, int]] = set()
        max_attempts = max(item.pair_budget * 30, 1000)
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
        value_a, value_b, sigma = mode_values(item.label_mode, item.stats, mol_a, mol_b)
        if value_a is None or value_b is None:
            continue
        delta = abs(value_a - value_b)
        normalized_delta = ""
        if item.label_mode == "pchembl_delta":
            if delta <= similar_delta:
                label = "similar"
            elif delta >= different_delta:
                label = "different"
            else:
                label = "ambiguous"
        else:
            if not sigma or sigma <= 0:
                continue
            z_delta = delta / sigma
            normalized_delta = z_delta
            if z_delta <= similar_z:
                label = "similar"
            elif z_delta >= different_z:
                label = "different"
            else:
                label = "ambiguous"
        tanimoto = tanimoto_int(mol_a.fp_int, mol_a.fp_popcount, mol_b.fp_int, mol_b.fp_popcount)
        meta = item.stats.meta
        rows.append(
            {
                "task_name": item.stats.task_name,
                "assay_chembl_id": item.stats.assay_chembl_id,
                "assay_id": item.stats.assay_id,
                "standard_type": item.stats.standard_type,
                "raw_units": item.stats.raw_units,
                "label_mode": item.label_mode,
                "molecule_a_chembl_id": mol_a.chembl_id,
                "molecule_b_chembl_id": mol_b.chembl_id,
                "molecule_a_smiles": mol_a.smiles,
                "molecule_b_smiles": mol_b.smiles,
                "raw_activity_a": fmt(mol_a.raw_value),
                "raw_activity_b": fmt(mol_b.raw_value),
                "log_raw_activity_a": fmt(mol_a.log_raw_value),
                "log_raw_activity_b": fmt(mol_b.log_raw_value),
                "pchembl_a": fmt(mol_a.pchembl_value),
                "pchembl_b": fmt(mol_b.pchembl_value),
                "activity_a": fmt(value_a),
                "activity_b": fmt(value_b),
                "abs_activity_delta": fmt(delta),
                "normalized_delta": fmt(normalized_delta),
                "robust_sigma": fmt(sigma),
                "tanimoto": fmt(tanimoto),
                "similarity_bucket": similarity_bucket(tanimoto),
                "label": label,
                "target_chembl_id": meta.target_chembl_id if meta else "",
                "target_pref_name": meta.target_pref_name if meta else "",
                "assay_tier": meta.tier if meta else "",
                "assay_description": meta.description if meta else "",
            }
        )
    return rows


def mode_values(label_mode: str, stats: EndpointStats, mol_a: MoleculeValue, mol_b: MoleculeValue) -> tuple[float | None, float | None, float | None]:
    if label_mode == "raw_robust_z":
        return mol_a.raw_value, mol_b.raw_value, stats.raw_value_robust_sigma
    if label_mode == "log_raw_robust_z":
        return mol_a.log_raw_value, mol_b.log_raw_value, stats.log_raw_value_robust_sigma
    if label_mode == "pchembl_delta":
        return mol_a.pchembl_value, mol_b.pchembl_value, None
    raise ValueError(f"Unsupported label mode: {label_mode}")


def consume_rows(
    rows: list[dict[str, Any]],
    writer: csv.DictWriter,
    thresholds: list[float],
    threshold_confusions: dict[float, Confusion],
    bucket_stats: dict[str, dict[str, Any]],
    task_stats: dict[str, dict[str, Any]],
    label_counts: Counter,
    deltas: list[float],
    completed: int,
    pair_count: int,
    endpoint_count: int,
    progress_every: int,
    started: float,
) -> tuple[int, int]:
    for row in rows:
        writer.writerow(row)
        pair_count += 1
        label = str(row["label"])
        label_counts[label] += 1
        delta = float(row["abs_activity_delta"])
        deltas.append(delta)
        for stats in (bucket_stats[str(row["similarity_bucket"])], task_stats[str(row["task_name"])]):
            stats["n_pairs"] += 1
            stats[label] += 1
            stats["delta_sum"] += delta
            stats["deltas"].append(delta)
            stats["tanimoto_sum"] += float(row["tanimoto"])
        if label == "ambiguous":
            continue
        truth_similar = label == "similar"
        tanimoto = float(row["tanimoto"])
        for threshold in thresholds:
            threshold_confusions[threshold].update(truth_similar, tanimoto >= threshold)
    completed += 1
    if progress_every and (completed % progress_every == 0 or completed == endpoint_count):
        log(
            f"pair benchmark progress endpoints={completed:,}/{endpoint_count:,} "
            f"pairs={pair_count:,} elapsed={format_elapsed(time.monotonic() - started)}"
        )
    return completed, pair_count


def new_bucket_stats() -> dict[str, dict[str, Any]]:
    return {bucket: new_label_stats() for bucket, _lo, _hi in SIMILARITY_BUCKETS}


def new_label_stats() -> dict[str, Any]:
    return {
        "n_pairs": 0,
        "similar": 0,
        "different": 0,
        "ambiguous": 0,
        "delta_sum": 0.0,
        "deltas": [],
        "tanimoto_sum": 0.0,
    }


def finalize_label_stats(stats_by_name: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    rows = []
    for name, stats in stats_by_name.items():
        n = int(stats["n_pairs"])
        deltas = stats["deltas"]
        rows.append(
            {
                key_name: name,
                "n_pairs": n,
                "similar": int(stats["similar"]),
                "different": int(stats["different"]),
                "ambiguous": int(stats["ambiguous"]),
                "similar_rate": int(stats["similar"]) / n if n else 0.0,
                "different_rate": int(stats["different"]) / n if n else 0.0,
                "ambiguous_rate": int(stats["ambiguous"]) / n if n else 0.0,
                "mean_abs_delta": float(stats["delta_sum"]) / n if n else 0.0,
                "median_abs_delta": median(deltas) if deltas else 0.0,
                "mean_tanimoto": float(stats["tanimoto_sum"]) / n if n else 0.0,
            }
        )
    rows.sort(key=lambda row: str(row[key_name]))
    return rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Task-scoped assay activity-transfer benchmark",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- elapsed_s: {summary['elapsed_s']}",
        "",
        "## Data sources",
        "",
    ]
    for task, task_summary in summary["source_summary"].items():
        lines.append(
            f"- `{task}`: used {task_summary['rows_used']:,}/{task_summary['rows_seen']:,} rows from `{task_summary['path']}`"
        )
    lines.extend(["", "## Label modes", ""])
    for label_mode, result in summary["label_modes"].items():
        lines.append(f"### {label_mode}")
        if result.get("status") != "ok":
            lines.append(f"- status: {result.get('status')}")
            lines.append("")
            continue
        best = result.get("best_threshold_by_macro_f1", {})
        lines.extend(
            [
                f"- endpoint groups: {result['n_endpoint_groups']:,}",
                f"- pairs: {result['n_pairs']:,}",
                f"- label_counts: `{result['label_counts']}`",
                f"- non-ambiguous pairs: {result['n_non_ambiguous_pairs']:,}",
                f"- majority baseline accuracy: {float(result['majority_baseline_accuracy']):.3f}",
                f"- best Tanimoto threshold: {float(best.get('threshold', 0.0)):.2f}",
                f"- best macro-F1: {float(best.get('macro_f1', 0.0)):.3f}",
                f"- pairs file: `{result['files']['pairs']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format_float(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
