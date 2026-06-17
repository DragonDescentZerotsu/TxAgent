"""Build raw-value assay-transfer splits for one ChEMBL assay endpoint.

This is a small companion to the broader ChEMBL activity-transfer builders. It
targets one assay_id/standard_type endpoint, labels random molecule pairs by raw
activity delta divided by endpoint sample std, and writes HF-compatible
prompt/completion JSONL splits.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.chembl_tool.activity_transfer_benchmark.build_dynamic_v1_mlp_splits import (
    LABEL_TO_COMPLETION,
    build_prompt,
    parse_ratios,
    random_split,
    ratio_tag,
)
from tools.chembl_tool.activity_transfer_benchmark.run_benchmark import (
    DEFAULT_FPS_GZ,
    load_fingerprints,
    load_molecule_lookup,
    quantile,
    similarity_bucket,
    tanimoto_int,
)


DEFAULT_CHEMBL_SQLITE = "tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/activity_transfer_benchmark/single_endpoint_raw_transfer"
SPLITS = ("train", "validation", "test")
VERSION_FULL = "full_range"
VERSION_NO_LT_MINUS_100 = "no_lt_minus_100"


@dataclass(frozen=True)
class MoleculeActivity:
    molecule_chembl_id: str
    smiles: str
    raw_value: float
    n_records: int
    fp_int: int
    fp_popcount: int


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    ratios = parse_ratios(args.ratios)
    versions = parse_versions(args.versions)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

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
    if len(molecules) < 2:
        raise ValueError(f"Need at least two molecules with fingerprints, got {len(molecules)}")

    summaries = {}
    for version in versions:
        version_molecules = filter_version_molecules(molecules, version)
        if len(version_molecules) < 2:
            raise ValueError(f"{version}: need at least two molecules after filtering, got {len(version_molecules)}")
        stats = summarize_values([molecule.raw_value for molecule in version_molecules])
        if stats["sample_std"] <= 0:
            raise ValueError(f"{version}: sample std must be positive, got {stats['sample_std']}")

        run_id = build_run_id(
            args.assay_chembl_id,
            endpoint["standard_type"],
            version,
            args.similar_std,
            args.different_std,
            args.ratios,
            args.n_pairs,
            args.split_mode,
        )
        out_dir = out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        pair_rows, molecule_split_counts = build_pairs_for_split_mode(
            version_molecules,
            endpoint=endpoint,
            value_stats=stats,
            version=version,
            split_mode=args.split_mode,
            n_pairs=args.n_pairs,
            ratios=ratios,
            similar_std=args.similar_std,
            different_std=args.different_std,
            seed=args.seed + stable_text_int(version),
            progress_every=args.progress_every,
        )
        split_summary = write_outputs(
            out_dir=out_dir,
            pair_rows=pair_rows,
            endpoint=endpoint,
            value_stats=stats,
            version=version,
            ratios=ratios,
            ratios_raw=args.ratios,
            seed=args.seed,
            split_mode=args.split_mode,
            molecule_split_counts=molecule_split_counts,
            similar_std=args.similar_std,
            different_std=args.different_std,
            source_run_id=run_id,
            source_parameters=vars(args),
        )
        summaries[version] = {
            "run_id": run_id,
            "out_dir": str(out_dir),
            "n_molecules_before_version_filter": len(molecules),
            "n_molecules": len(version_molecules),
            "value_stats": stats,
            "split_summary": split_summary,
        }

    manifest = {
        "elapsed_s": round(time.time() - started, 3),
        "endpoint": endpoint,
        "n_raw_molecules": len(raw_molecules),
        "n_molecules_with_fingerprint": len(molecules),
        "versions": summaries,
    }
    (out_root / f"{args.assay_chembl_id}_{endpoint['standard_type']}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--fps-gz", default=DEFAULT_FPS_GZ)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--assay-chembl-id", default="CHEMBL4513218")
    parser.add_argument("--standard-type", default="inhibition")
    parser.add_argument("--versions", default=f"{VERSION_FULL},{VERSION_NO_LT_MINUS_100}")
    parser.add_argument("--n-pairs", type=int, default=200_000, help="Number of non-ambiguous pairs to materialize per version.")
    parser.add_argument("--ratios", default="0.7,0.1,0.2", help="train,validation,test ratios.")
    parser.add_argument(
        "--split-mode",
        choices=["molecule_disjoint", "pair_random"],
        default="molecule_disjoint",
        help="molecule_disjoint keeps every molecule in exactly one split.",
    )
    parser.add_argument("--similar-std", type=float, default=0.5)
    parser.add_argument("--different-std", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser.parse_args(argv)


def parse_versions(raw: str) -> list[str]:
    versions = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {VERSION_FULL, VERSION_NO_LT_MINUS_100}
    unknown = sorted(set(versions).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown version(s): {', '.join(unknown)}")
    if not versions:
        raise ValueError("At least one version is required.")
    return versions


def load_endpoint_context(conn: sqlite3.Connection, assay_chembl_id: str, standard_type: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          a.assay_id,
          a.chembl_id AS assay_chembl_id,
          LOWER(TRIM(act.standard_type)) AS standard_type,
          a.assay_type,
          a.assay_test_type,
          a.assay_category,
          a.assay_organism,
          a.confidence_score,
          a.relationship_type,
          a.description,
          d.doc_type,
          d.pubmed_id,
          d.doi,
          d.title,
          td.chembl_id AS target_chembl_id,
          td.pref_name AS target_pref_name,
          td.target_type,
          td.organism AS target_organism,
          GROUP_CONCAT(DISTINCT act.standard_units) AS raw_units,
          COUNT(*) AS n_activity_rows,
          COUNT(DISTINCT act.molregno) AS n_molecules
        FROM assays a
        JOIN activities act ON a.assay_id = act.assay_id
        JOIN docs d ON a.doc_id = d.doc_id
        LEFT JOIN target_dictionary td ON a.tid = td.tid
        WHERE a.chembl_id = ?
          AND LOWER(TRIM(act.standard_type)) = LOWER(TRIM(?))
          AND act.molregno IS NOT NULL
          AND act.standard_value IS NOT NULL
        GROUP BY a.assay_id, LOWER(TRIM(act.standard_type))
        """,
        (assay_chembl_id, standard_type),
    ).fetchone()
    if row is None:
        raise ValueError(f"No endpoint found for {assay_chembl_id} / {standard_type}")
    return {key: row[key] for key in row.keys()}


def load_molecule_activities(conn: sqlite3.Connection, assay_id: int, standard_type: str) -> list[dict[str, Any]]:
    accumulator: dict[int, list[float | int]] = {}
    for row in conn.execute(
        """
        SELECT act.molregno, CAST(act.standard_value AS REAL) AS raw_value
        FROM activities act
        WHERE act.assay_id = ?
          AND LOWER(TRIM(act.standard_type)) = ?
          AND act.molregno IS NOT NULL
          AND act.standard_value IS NOT NULL
        """,
        (assay_id, standard_type),
    ):
        molregno = int(row["molregno"])
        entry = accumulator.get(molregno)
        if entry is None:
            accumulator[molregno] = [float(row["raw_value"]), 1]
        else:
            entry[0] = float(entry[0]) + float(row["raw_value"])
            entry[1] = int(entry[1]) + 1
    lookup = load_molecule_lookup(conn, sorted(accumulator))
    molecules = []
    for molregno, (value_sum, n_records) in accumulator.items():
        molecule = lookup.get(molregno)
        if molecule is None:
            continue
        chembl_id, smiles = molecule
        molecules.append(
            {
                "molregno": molregno,
                "molecule_chembl_id": chembl_id,
                "smiles": smiles,
                "raw_value": float(value_sum) / int(n_records),
                "n_records": int(n_records),
            }
        )
    molecules.sort(key=lambda row: row["molecule_chembl_id"])
    return molecules


def filter_version_molecules(molecules: list[MoleculeActivity], version: str) -> list[MoleculeActivity]:
    if version == VERSION_FULL:
        return list(molecules)
    if version == VERSION_NO_LT_MINUS_100:
        return [molecule for molecule in molecules if molecule.raw_value >= -100.0]
    raise ValueError(f"Unsupported version: {version}")


def summarize_values(values: list[float]) -> dict[str, float | int]:
    sorted_values = sorted(values)
    n = len(sorted_values)
    mean = sum(sorted_values) / n
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in sorted_values) / (n - 1)) if n > 1 else 0.0
    population_std = math.sqrt(sum((value - mean) ** 2 for value in sorted_values) / n) if n else 0.0
    return {
        "n": n,
        "min": sorted_values[0],
        "p01": quantile(sorted_values, 0.01),
        "p05": quantile(sorted_values, 0.05),
        "p25": quantile(sorted_values, 0.25),
        "median": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p95": quantile(sorted_values, 0.95),
        "p99": quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": mean,
        "sample_std": sample_std,
        "population_std": population_std,
    }


def build_pairs_for_split_mode(
    molecules: list[MoleculeActivity],
    *,
    endpoint: dict[str, Any],
    value_stats: dict[str, float | int],
    version: str,
    split_mode: str,
    n_pairs: int,
    ratios: dict[str, float],
    similar_std: float,
    different_std: float,
    seed: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if split_mode == "pair_random":
        return (
            sample_labeled_pairs(
                molecules,
                endpoint=endpoint,
                value_stats=value_stats,
                version=version,
                split="",
                n_pairs=n_pairs,
                similar_std=similar_std,
                different_std=different_std,
                seed=seed,
                progress_every=progress_every,
            ),
            {},
        )
    if split_mode != "molecule_disjoint":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    molecules_by_split = assign_molecule_splits(molecules, ratios=ratios, seed=seed)
    targets = pair_targets(n_pairs, ratios)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_molecules = molecules_by_split[split]
        if len(split_molecules) < 2:
            raise ValueError(f"{split}: need at least two molecules for molecule_disjoint split")
        rows.extend(
            sample_labeled_pairs(
                split_molecules,
                endpoint=endpoint,
                value_stats=value_stats,
                version=version,
                split=split,
                n_pairs=targets[split],
                similar_std=similar_std,
                different_std=different_std,
                seed=seed + stable_text_int(split),
                progress_every=progress_every,
            )
        )
    return rows, {split: len(molecules_by_split[split]) for split in SPLITS}


def assign_molecule_splits(
    molecules: list[MoleculeActivity],
    *,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[MoleculeActivity]]:
    shuffled = list(molecules)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_n = int(n * ratios["train"])
    validation_n = int(n * ratios["validation"])
    return {
        "train": shuffled[:train_n],
        "validation": shuffled[train_n : train_n + validation_n],
        "test": shuffled[train_n + validation_n :],
    }


def pair_targets(n_pairs: int, ratios: dict[str, float]) -> dict[str, int]:
    train_n = int(n_pairs * ratios["train"])
    validation_n = int(n_pairs * ratios["validation"])
    return {
        "train": train_n,
        "validation": validation_n,
        "test": n_pairs - train_n - validation_n,
    }


def sample_labeled_pairs(
    molecules: list[MoleculeActivity],
    *,
    endpoint: dict[str, Any],
    value_stats: dict[str, float | int],
    version: str,
    split: str,
    n_pairs: int,
    similar_std: float,
    different_std: float,
    seed: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    n = len(molecules)
    sample_std = float(value_stats["sample_std"])
    selected: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    attempts = 0
    max_attempts = max(n_pairs * 100, 10_000)
    started = time.time()
    while len(rows) < n_pairs and attempts < max_attempts:
        i = rng.randrange(0, n - 1)
        j = rng.randrange(i + 1, n)
        attempts += 1
        key = (i, j)
        if key in selected:
            continue
        selected.add(key)
        mol_a = molecules[i]
        mol_b = molecules[j]
        delta = abs(mol_a.raw_value - mol_b.raw_value)
        normalized_delta = delta / sample_std
        if normalized_delta <= similar_std:
            label = "similar"
        elif normalized_delta >= different_std:
            label = "different"
        else:
            continue
        tanimoto = tanimoto_int(mol_a.fp_int, mol_a.fp_popcount, mol_b.fp_int, mol_b.fp_popcount)
        label_counts[label] += 1
        rows.append(
            {
                "task_name": "single_endpoint_raw_transfer",
                "assay_chembl_id": endpoint["assay_chembl_id"],
                "assay_id": endpoint["assay_id"],
                "standard_type": endpoint["standard_type"],
                "raw_units": endpoint.get("raw_units") or "",
                "label_mode": "raw_std_delta",
                "value_version": version,
                "split": split,
                "molecule_a_chembl_id": mol_a.molecule_chembl_id,
                "molecule_b_chembl_id": mol_b.molecule_chembl_id,
                "molecule_a_smiles": mol_a.smiles,
                "molecule_b_smiles": mol_b.smiles,
                "raw_activity_a": format_float(mol_a.raw_value),
                "raw_activity_b": format_float(mol_b.raw_value),
                "activity_a": format_float(mol_a.raw_value),
                "activity_b": format_float(mol_b.raw_value),
                "abs_activity_delta": format_float(delta),
                "normalized_delta": format_float(normalized_delta),
                "std": format_float(sample_std),
                "tanimoto": format_float(tanimoto),
                "similarity_bucket": similarity_bucket(tanimoto),
                "label": label,
                "target_chembl_id": endpoint.get("target_chembl_id") or "",
                "target_pref_name": endpoint.get("target_pref_name") or "",
                "assay_description": endpoint.get("description") or "",
            }
        )
        if progress_every > 0 and len(rows) % progress_every == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                json.dumps(
                    {
                        "stage": "sample_pairs",
                        "version": version,
                        "split": split,
                        "written_candidates": len(rows),
                        "attempts": attempts,
                        "label_counts": dict(label_counts),
                        "rate_per_s": round(len(rows) / elapsed, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(rows) < n_pairs:
        raise RuntimeError(f"{version}: sampled only {len(rows):,}/{n_pairs:,} non-ambiguous pairs after {attempts:,} attempts")
    return rows


def write_outputs(
    *,
    out_dir: Path,
    pair_rows: list[dict[str, Any]],
    endpoint: dict[str, Any],
    value_stats: dict[str, float | int],
    version: str,
    ratios: dict[str, float],
    ratios_raw: str,
    seed: int,
    split_mode: str,
    molecule_split_counts: dict[str, int],
    similar_std: float,
    different_std: float,
    source_run_id: str,
    source_parameters: dict[str, Any],
) -> dict[str, Any]:
    rng = random.Random(seed)
    split_version = (
        f"{source_run_id}_raw_std_sim_le_{similar_std:g}_diff_ge_{different_std:g}_"
        f"{split_mode}_{ratio_tag(ratios_raw)}"
    ).replace(".", "p")
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    bucket_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    try:
        with gzip.open(out_dir / "pairs.tsv.gz", "wt", encoding="utf-8", newline="") as pair_handle:
            writer = csv.DictWriter(pair_handle, fieldnames=list(pair_rows[0]), delimiter="\t")
            writer.writeheader()
            for source_index, row in enumerate(pair_rows):
                writer.writerow(row)
                split = str(row.get("split") or "") if split_mode == "molecule_disjoint" else ""
                if split not in SPLITS:
                    split = random_split(rng, ratios)
                record = build_record(
                    row,
                    endpoint,
                    split=split,
                    split_version=split_version,
                    source_run_id=source_run_id,
                    source_index=source_index,
                    value_stats=value_stats,
                    version=version,
                    similar_std=similar_std,
                    different_std=different_std,
                )
                handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                split_counts[split][row["label"]] += 1
                bucket_counts[split][row["similarity_bucket"]] += 1
    finally:
        for handle in handles.values():
            handle.close()

    endpoint_summary = endpoint_summary_row(endpoint, value_stats, version, len(pair_rows), split_mode, similar_std, different_std)
    write_tsv(out_dir / "endpoint_summary.tsv", [endpoint_summary])
    summary = {
        "source_run_id": source_run_id,
        "source_parameters": source_parameters,
        "label_rule": {
            "label_type": "raw_std_delta",
            "std": "sample_std",
            "similar_if_normalized_delta_le": similar_std,
            "different_if_normalized_delta_ge": different_std,
            "ambiguous": "excluded",
        },
        "value_version": version,
        "split_mode": split_mode,
        "molecule_split_counts": molecule_split_counts,
        "value_stats": value_stats,
        "n": len(pair_rows),
        "split_counts": {
            split: {
                "total": int(sum(split_counts[split].values())),
                "label_counts": dict(sorted(split_counts[split].items())),
                "similarity_bucket_counts": dict(sorted(bucket_counts[split].items())),
            }
            for split in SPLITS
        },
        "files": {split: str(out_dir / f"{split}.jsonl") for split in SPLITS}
        | {
            "pairs_tsv_gz": str(out_dir / "pairs.tsv.gz"),
            "endpoint_summary": str(out_dir / "endpoint_summary.tsv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_record(
    row: dict[str, Any],
    endpoint: dict[str, Any],
    *,
    split: str,
    split_version: str,
    source_run_id: str,
    source_index: int,
    value_stats: dict[str, float | int],
    version: str,
    similar_std: float,
    different_std: float,
) -> dict[str, Any]:
    assay_id = parse_int(row.get("assay_id"))
    standard_type = str(row.get("standard_type") or "")
    endpoint_id = f"{assay_id}|{standard_type}|raw_std_delta|{version}"
    pair_id = (
        f"{endpoint_id}|{row.get('molecule_a_chembl_id', '')}"
        f"<>{row.get('molecule_b_chembl_id', '')}"
    )
    endpoint_for_prompt = {
        "standard_type": standard_type,
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
    metadata = {
        "sample_id": f"{source_run_id}:{source_index}",
        "pair_id": pair_id,
        "source_index": source_index,
        "source_run_id": source_run_id,
        "split": split,
        "split_version": split_version,
        "endpoint_scope": "assay",
        "endpoint_id": endpoint_id,
        "assay_chembl_id": row.get("assay_chembl_id"),
        "assay_id": assay_id,
        "standard_type": standard_type,
        "raw_units": row.get("raw_units"),
        "assay_type": endpoint.get("assay_type"),
        "assay_test_type": endpoint.get("assay_test_type"),
        "assay_category": endpoint.get("assay_category"),
        "confidence_score": parse_int(endpoint.get("confidence_score")),
        "relationship_type": endpoint.get("relationship_type"),
        "target_chembl_id": endpoint.get("target_chembl_id"),
        "target_name": endpoint.get("target_pref_name"),
        "target_type": endpoint.get("target_type"),
        "target_organism": endpoint.get("target_organism"),
        "doc_type": endpoint.get("doc_type"),
        "pubmed_id": parse_int(endpoint.get("pubmed_id")),
        "doi": endpoint.get("doi"),
        "title": endpoint.get("title"),
        "label_type": "raw_std_delta",
        "similar_std": similar_std,
        "different_std": different_std,
        "std": value_stats["sample_std"],
        "value_version": version,
        "molecule_a_chembl_id": row.get("molecule_a_chembl_id"),
        "molecule_b_chembl_id": row.get("molecule_b_chembl_id"),
        "activity_a": parse_float(row.get("activity_a")),
        "activity_b": parse_float(row.get("activity_b")),
        "abs_activity_delta": parse_float(row.get("abs_activity_delta")),
        "normalized_delta": parse_float(row.get("normalized_delta")),
        "weighted_tanimoto": parse_float(row.get("tanimoto")),
        "similarity_bucket": row.get("similarity_bucket"),
        "direction": "molecule_a_to_molecule_b",
    }
    return {
        "prompt": build_prompt(row, endpoint_for_prompt),
        "completion": LABEL_TO_COMPLETION[str(row["label"])],
        "metadata": metadata,
    }


def endpoint_summary_row(
    endpoint: dict[str, Any],
    value_stats: dict[str, float | int],
    version: str,
    n_pairs: int,
    split_mode: str,
    similar_std: float,
    different_std: float,
) -> dict[str, Any]:
    row = {
        "task_name": "single_endpoint_raw_transfer",
        "assay_chembl_id": endpoint.get("assay_chembl_id"),
        "assay_id": endpoint.get("assay_id"),
        "standard_type": endpoint.get("standard_type"),
        "raw_units": endpoint.get("raw_units") or "",
        "label_mode": "raw_std_delta",
        "value_version": version,
        "n_pairs": n_pairs,
        "split_mode": split_mode,
        "similar_std": similar_std,
        "different_std": different_std,
        "assay_type": endpoint.get("assay_type") or "",
        "assay_test_type": endpoint.get("assay_test_type") or "",
        "assay_category": endpoint.get("assay_category") or "",
        "description": endpoint.get("description") or "",
        "target_chembl_id": endpoint.get("target_chembl_id") or "",
        "target_pref_name": endpoint.get("target_pref_name") or "",
        "doc_type": endpoint.get("doc_type") or "",
        "pubmed_id": endpoint.get("pubmed_id") or "",
        "doi": endpoint.get("doi") or "",
        "title": endpoint.get("title") or "",
    }
    for key, value in value_stats.items():
        row[f"raw_value_{key}"] = value
    return row


def build_run_id(
    assay_chembl_id: str,
    standard_type: str,
    version: str,
    similar_std: float,
    different_std: float,
    ratios: str,
    n_pairs: int,
    split_mode: str,
) -> str:
    return (
        f"{assay_chembl_id}_{standard_type}_{version}_"
        f"raw_std_sim_le_{similar_std:g}_diff_ge_{different_std:g}_"
        f"{split_mode}_{ratio_tag(ratios)}_n{n_pairs}"
    ).replace(".", "p").replace("/", "_")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


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


def stable_text_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


if __name__ == "__main__":
    raise SystemExit(main())
