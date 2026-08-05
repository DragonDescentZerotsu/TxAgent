"""Shared evidence-library and neighbor-index builder for ChEMBL reasoning tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity


DEFAULT_CHEMBL_FPS = "tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz"
FP_RADIUS = 2
FP_BITS = 2048
StandardizedIndexMolecule = tuple[
    str,
    str,
    DataStructs.ExplicitBitVect | None,
    dict[str, Any],
]

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


@dataclass(frozen=True)
class EvidenceLibraryConfig:
    description: str
    default_assays_csv: str
    default_activities_csv: str
    default_out_dir: str
    evidence_filename: str
    index_filename: str
    meta_filename: str
    index_version: str
    assign_endpoint_group: Callable[[dict[str, Any]], Any]


def main(config: EvidenceLibraryConfig, argv: list[str] | None = None) -> int:
    args = _parse_args(config, argv)
    started = time.monotonic()
    out_dir = ensure_dir(args.out_dir)

    _log(f"loading assay metadata: {args.assays_csv}")
    assays = _load_assays(Path(args.assays_csv))
    _log(f"loaded assay metadata rows={len(assays):,}")

    _log(f"building evidence rows: {args.activities_csv}")
    evidence_rows = build_evidence_rows(
        Path(args.activities_csv),
        assays,
        config.assign_endpoint_group,
        progress_every=args.progress_every,
    )
    _log(f"built evidence rows={len(evidence_rows):,}")

    needed_molecule_ids = {row["molecule_chembl_id"] for row in evidence_rows if row.get("molecule_chembl_id")}
    fps_by_molecule: dict[str, DataStructs.ExplicitBitVect] = {}
    if args.chembl_fps and not args.no_chembl_fps:
        fps_path = Path(args.chembl_fps)
        if fps_path.exists():
            _log(f"loading precomputed ChEMBL fps for evidence molecules: {fps_path}")
            fps_by_molecule = _load_fps_subset(fps_path, needed_molecule_ids, progress_every=args.progress_every)
            _log(f"loaded precomputed fps={len(fps_by_molecule):,}")
        else:
            _log(f"precomputed fps path does not exist; falling back to SMILES fingerprints: {fps_path}")

    _log("building neighbor index")
    index = build_neighbor_index(
        evidence_rows,
        fps_by_molecule=fps_by_molecule,
        index_version=config.index_version,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    _log(
        "built neighbor index "
        f"molecules={len(index['molecules']):,} groups={len(index['group_to_molecule_indices']):,}"
    )

    evidence_jsonl = out_dir / config.evidence_filename
    index_pkl = out_dir / config.index_filename
    meta_json = out_dir / config.meta_filename

    _write_jsonl(evidence_jsonl, evidence_rows)
    with index_pkl.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "assays_csv": str(args.assays_csv),
        "activities_csv": str(args.activities_csv),
        "chembl_fps": None if args.no_chembl_fps else str(args.chembl_fps),
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "n_groups": len(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "workers": args.workers,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _log(f"wrote {evidence_jsonl}")
    _log(f"wrote {index_pkl}")
    _log(f"wrote {meta_json}")
    return 0


def build_evidence_rows(
    activities_csv: Path,
    assays: dict[str, dict[str, Any]],
    assign_endpoint_group: Callable[[dict[str, Any]], Any],
    *,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_rows = max(0, _line_count(activities_csv) - 1) if progress_every else 0
    started = time.monotonic()
    with activities_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for i, activity in enumerate(csv.DictReader(handle), start=1):
            assay = assays.get(activity.get("assay_chembl_id", ""), {})
            row = _merged_evidence_row(activity, assay)
            assignment = assign_endpoint_group(row)
            row.update(
                {
                    "assay_tier": assignment.tier,
                    "endpoint_group": assignment.endpoint_group,
                    "group_id": assignment.group_id,
                    "endpoint_group_reason": assignment.reason,
                    "evidence_direction": assignment.evidence_direction,
                    "evidence_strength": assignment.evidence_strength,
                }
            )
            row.setdefault("evidence_source", "ChEMBL")
            attach_minimal_evidence(row)
            rows.append(row)
            if progress_every and i % progress_every == 0:
                _log(_progress_message("evidence rows", i, total_rows, started, extra=f"kept={len(rows):,}"))
    if progress_every and rows:
        _log(_progress_message("evidence rows", len(rows), total_rows, started, extra=f"kept={len(rows):,}"))
    return rows


def build_neighbor_index(
    evidence_rows: list[dict[str, Any]],
    fps_by_molecule: dict[str, DataStructs.ExplicitBitVect] | None = None,
    *,
    index_version: str,
    progress_every: int = 0,
    workers: int = 1,
    standardized_by_molecule: Mapping[str, StandardizedIndexMolecule] | None = None,
) -> dict[str, Any]:
    fps_by_molecule = fps_by_molecule or {}
    molecule_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        molecule_id = str(row.get("molecule_chembl_id") or "").strip()
        smiles = str(row.get("canonical_smiles") or "").strip()
        if not molecule_id or not smiles:
            continue
        molecule_rows[molecule_id].append(row)

    molecules: list[dict[str, Any]] = []
    fingerprints: list[DataStructs.ExplicitBitVect] = []
    evidence_by_molecule_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    group_to_molecule_indices: dict[str, list[int]] = defaultdict(list)

    molecule_ids = sorted(molecule_rows)
    standardized = dict(standardized_by_molecule or {})
    missing_ids = [
        molecule_id for molecule_id in molecule_ids if molecule_id not in standardized
    ]
    if missing_ids:
        standardized.update(
            _standardize_molecules_for_index(
                missing_ids,
                molecule_rows,
                fps_by_molecule,
                workers=workers,
                progress_every=progress_every,
            )
        )
    for molecule_id in molecule_ids:
        rows = molecule_rows[molecule_id]
        smiles = _first_nonempty(row.get("canonical_smiles") for row in rows)
        canonical_smiles, inchi_key, fallback_fp, molecule_identity = standardized.get(
            molecule_id, ("", "", None, None)
        )
        if molecule_id in fps_by_molecule:
            fp = fps_by_molecule[molecule_id]
        else:
            fp = fallback_fp
        if fp is None:
            continue
        for row in rows:
            row["canonical_smiles"] = canonical_smiles or smiles
            row["standard_inchi_key"] = inchi_key
            attach_minimal_evidence(row)
        molecule_index = len(molecules)
        group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_rows[str(row["group_id"])].append(row)
        for group_id in sorted(group_rows):
            group_to_molecule_indices[group_id].append(molecule_index)
        molecules.append(
            {
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": canonical_smiles or smiles,
                "standard_inchi_key": inchi_key,
                "n_evidence_rows": len(rows),
                "groups": sorted(group_rows),
                "fingerprint_source": "chembl_fps" if molecule_id in fps_by_molecule else "canonical_smiles",
                "molecule_identity": (
                    molecule_identity
                    if molecule_identity is not None
                    else normalize_molecule_identity(canonical_smiles or smiles).to_dict()
                ),
            }
        )
        fingerprints.append(fp)
        evidence_by_molecule_group[molecule_id] = {group_id: group_rows[group_id] for group_id in group_rows}

    return {
        "version": index_version,
        "fingerprint": fingerprint_metadata(),
        "molecules": molecules,
        "fingerprints": fingerprints,
        "group_to_molecule_indices": dict(group_to_molecule_indices),
        "evidence_by_molecule_group": evidence_by_molecule_group,
    }


def standardize_index_molecules(
    molecule_smiles: Mapping[str, str],
    *,
    workers: int = 1,
    progress_every: int = 0,
) -> dict[str, StandardizedIndexMolecule]:
    """Standardize one shared molecule union for several compatible indices."""
    molecule_rows = {
        str(molecule_id): [{"canonical_smiles": str(smiles)}]
        for molecule_id, smiles in molecule_smiles.items()
    }
    return _standardize_molecules_for_index(
        sorted(molecule_rows),
        molecule_rows,
        {},
        workers=workers,
        progress_every=progress_every,
    )


def _standardize_molecules_for_index(
    molecule_ids: list[str],
    molecule_rows: dict[str, list[dict[str, Any]]],
    fps_by_molecule: dict[str, DataStructs.ExplicitBitVect],
    *,
    workers: int,
    progress_every: int,
) -> dict[str, StandardizedIndexMolecule]:
    tasks = [
        (
            molecule_id,
            _first_nonempty(row.get("canonical_smiles") for row in molecule_rows[molecule_id]),
            molecule_id not in fps_by_molecule,
        )
        for molecule_id in molecule_ids
    ]
    total = len(tasks)
    started = time.monotonic()
    results: dict[str, StandardizedIndexMolecule] = {}
    workers = max(1, int(workers or 1))
    if workers == 1 or total < 2:
        for i, task in enumerate(tasks, start=1):
            molecule_id, canonical_smiles, inchi_key, fp, identity = (
                _standardize_molecule_task(task)
            )
            results[molecule_id] = (canonical_smiles, inchi_key, fp, identity)
            if progress_every and i % progress_every == 0:
                _log(_progress_message("index molecule standardization", i, total, started))
    else:
        chunksize = max(1, min(1000, total // (workers * 4) if total >= workers * 4 else 1))
        _log(f"standardizing index molecules with workers={workers} chunksize={chunksize}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for i, result in enumerate(executor.map(_standardize_molecule_task, tasks, chunksize=chunksize), start=1):
                molecule_id, canonical_smiles, inchi_key, fp, identity = result
                results[molecule_id] = (canonical_smiles, inchi_key, fp, identity)
                if progress_every and i % progress_every == 0:
                    _log(_progress_message("index molecule standardization", i, total, started))
    if progress_every:
        _log(_progress_message("index molecule standardization", total, total, started))
    return results


def _standardize_molecule_task(
    task: tuple[str, str, bool],
) -> tuple[str, str, DataStructs.ExplicitBitVect | None, dict[str, Any]]:
    molecule_id, smiles, needs_fp = task
    if needs_fp:
        canonical_smiles, inchi_key, fp = standardize_smiles_and_fp(smiles)
        identity = normalize_molecule_identity(canonical_smiles or smiles).to_dict()
        return molecule_id, canonical_smiles, inchi_key, fp, identity
    canonical_smiles, inchi_key = standardize_smiles(smiles)
    identity = normalize_molecule_identity(canonical_smiles or smiles).to_dict()
    return molecule_id, canonical_smiles, inchi_key, None, identity


def fingerprint_metadata() -> dict[str, Any]:
    return {
        "type": "RDKit-Morgan",
        "radius": FP_RADIUS,
        "n_bits": FP_BITS,
        "useFeatures": False,
        "useChirality": False,
        "useBondTypes": True,
    }


def standardize_smiles_and_fp(smiles: str) -> tuple[str, str, DataStructs.ExplicitBitVect | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", "", None
    canonical_smiles, inchi_key = _standardize_mol(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
        useFeatures=False,
        useChirality=False,
        useBondTypes=True,
    )
    return canonical_smiles, inchi_key, fp


def standardize_smiles(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    return _standardize_mol(mol)


def _standardize_mol(mol: Chem.Mol) -> tuple[str, str]:
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    try:
        inchi_key = inchi.MolToInchiKey(mol)
    except Exception:
        inchi_key = ""
    return canonical_smiles, inchi_key


def _load_assays(path: Path) -> dict[str, dict[str, Any]]:
    assays: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        for row in csv.DictReader(handle, delimiter=delimiter):
            assay_id = str(row.get("assay_chembl_id") or "").strip()
            if not assay_id:
                continue
            assay = {
                "assay_chembl_id": assay_id,
                "assay_id": row.get("assay_id", ""),
                "assay_tier": row.get("tier", ""),
                "assay_score": row.get("score", ""),
                "assay_type": row.get("assay_type", ""),
                "assay_description": row.get("description", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
                "target_pref_name": row.get("target_pref_name", ""),
                "target_genes": row.get("target_genes", ""),
                "target_synonyms": row.get("target_synonyms", ""),
                "organism": row.get("organism", ""),
                "confidence_score": row.get("confidence_score", ""),
                "relationship_type": row.get("relationship_type", ""),
                "assay_cell_type": row.get("assay_cell_type", ""),
                "assay_tissue": row.get("assay_tissue", ""),
                "assay_standard_types": row.get("standard_types", ""),
                "matched_keywords": row.get("matched_keywords", ""),
                "matched_endpoints": row.get("matched_endpoints", ""),
                "matched_targets": row.get("matched_targets", ""),
                "assay_reason": row.get("reason", ""),
            }
            for field in (
                "distance_level",
                "distance_family_id",
                "distance_tree_node_id",
                "parent_c_family_id",
                "source_group_id",
                "measured_node",
                "scope_match",
                "quality_status",
                "effect_direction",
                "mapping_reason",
            ):
                if row.get(field) not in (None, ""):
                    assay[field] = row[field]
            assays[assay_id] = assay
    return assays


def load_assays(path: Path) -> dict[str, dict[str, Any]]:
    """Public assay-manifest loader; TSV is supported for distance manifests."""
    return _load_assays(path)


def _merged_evidence_row(activity: dict[str, Any], assay: dict[str, Any]) -> dict[str, Any]:
    row = {
        "molecule_chembl_id": activity.get("molecule_chembl_id", ""),
        "canonical_smiles": activity.get("canonical_smiles", ""),
        "assay_chembl_id": activity.get("assay_chembl_id", ""),
        "standard_type": activity.get("standard_type", ""),
        "standard_relation": activity.get("standard_relation", ""),
        "standard_value": activity.get("standard_value", ""),
        "standard_units": activity.get("standard_units", ""),
        "pchembl_value": activity.get("pchembl_value", ""),
        "data_validity_comment": activity.get("data_validity_comment", ""),
        "activity_comment": activity.get("activity_comment", ""),
    }
    row.update(assay)
    return row


def _load_fps_subset(
    path: Path,
    molecule_ids: set[str],
    *,
    progress_every: int = 0,
) -> dict[str, DataStructs.ExplicitBitVect]:
    fps: dict[str, DataStructs.ExplicitBitVect] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    started = time.monotonic()
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            fps_text, molecule_id = parts[0], parts[1]
            if molecule_id in molecule_ids:
                fps[molecule_id] = DataStructs.CreateFromFPSText(fps_text)
                if len(fps) == len(molecule_ids):
                    break
            if progress_every and i % progress_every == 0:
                _log(_fps_progress_message(i, len(fps), len(molecule_ids), started))
    if progress_every:
        _log(_fps_progress_message(i if "i" in locals() else 0, len(fps), len(molecule_ids), started))
    return fps


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _progress_message(
    label: str,
    done: int,
    total: int | None,
    started: float,
    *,
    extra: str = "",
) -> str:
    elapsed = max(0.001, time.monotonic() - started)
    rate = done / elapsed
    parts = [f"{label} processed={done:,}"]
    if total:
        pct = 100 * done / total
        remaining = max(0, total - done)
        eta_s = remaining / rate if rate > 0 else 0
        parts.append(f"total={total:,} pct={pct:.1f}% eta={_format_duration(eta_s)}")
    parts.append(f"elapsed={_format_duration(elapsed)} rate={rate:,.1f}/s")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _fps_progress_message(scanned: int, matched: int, needed: int, started: float) -> str:
    elapsed = max(0.001, time.monotonic() - started)
    scan_rate = scanned / elapsed
    match_rate = matched / elapsed
    parts = [
        f"fps scan lines={scanned:,}",
        f"matched={matched:,}/{needed:,}",
        f"elapsed={_format_duration(elapsed)}",
        f"scan_rate={scan_rate:,.1f}/s",
        f"match_rate={match_rate:,.1f}/s",
    ]
    if needed and matched:
        remaining = max(0, needed - matched)
        parts.append(f"eta_by_match_rate={_format_duration(remaining / match_rate)}")
    else:
        parts.append("eta_by_match_rate=unknown")
    return " ".join(parts)


def _first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _parse_args(config: EvidenceLibraryConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--assays-csv", default=config.default_assays_csv)
    parser.add_argument("--activities-csv", default=config.default_activities_csv)
    parser.add_argument("--out-dir", default=config.default_out_dir)
    parser.add_argument("--chembl-fps", default=DEFAULT_CHEMBL_FPS)
    parser.add_argument("--no-chembl-fps", action="store_true", help="Compute all fingerprints from SMILES.")
    parser.add_argument("--progress-every", type=int, default=100000, help="Print progress every N rows; 0 disables.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for RDKit molecule standardization/indexing.")
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[build_evidence_library] {message}", file=sys.stderr, flush=True)
