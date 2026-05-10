"""Build a molecule-level oral bioavailability evidence library and neighbor index."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.tasks.bioavailability_ma.endpoint_groups import assign_endpoint_group


DEFAULT_ASSAYS_CSV = "outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_assay_candidates.csv"
DEFAULT_ACTIVITIES_CSV = "outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_activity_evidence.csv"
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library"
DEFAULT_CHEMBL_FPS = "tools/chembl_tool/chembl_data/chembl_36_fps/chembl_36.fps.gz"

FP_RADIUS = 2
FP_BITS = 2048

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    out_dir = ensure_dir(args.out_dir)

    _log(f"loading assay metadata: {args.assays_csv}")
    assays = _load_assays(Path(args.assays_csv))
    _log(f"loaded assay metadata rows={len(assays):,}")

    _log(f"building evidence rows: {args.activities_csv}")
    evidence_rows = build_evidence_rows(Path(args.activities_csv), assays)
    _log(f"built evidence rows={len(evidence_rows):,}")

    needed_molecule_ids = {row["molecule_chembl_id"] for row in evidence_rows if row.get("molecule_chembl_id")}
    fps_by_molecule: dict[str, DataStructs.ExplicitBitVect] = {}
    if args.chembl_fps and not args.no_chembl_fps:
        fps_path = Path(args.chembl_fps)
        if fps_path.exists():
            _log(f"loading precomputed ChEMBL fps for evidence molecules: {fps_path}")
            fps_by_molecule = _load_fps_subset(fps_path, needed_molecule_ids)
            _log(f"loaded precomputed fps={len(fps_by_molecule):,}")
        else:
            _log(f"precomputed fps path does not exist; falling back to SMILES fingerprints: {fps_path}")

    _log("building neighbor index")
    index = build_neighbor_index(evidence_rows, fps_by_molecule=fps_by_molecule)
    _log(
        "built neighbor index "
        f"molecules={len(index['molecules']):,} groups={len(index['group_to_molecule_indices']):,}"
    )

    evidence_jsonl = out_dir / "bioavailability_molecule_evidence.jsonl"
    index_pkl = out_dir / "bioavailability_neighbor_index.pkl"
    meta_json = out_dir / "bioavailability_neighbor_index.meta.json"

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
        "fingerprint": {
            "type": "RDKit-Morgan",
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
            "useFeatures": False,
            "useChirality": False,
            "useBondTypes": True,
        },
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _log(f"wrote {evidence_jsonl}")
    _log(f"wrote {index_pkl}")
    _log(f"wrote {meta_json}")
    return 0


def build_evidence_rows(activities_csv: Path, assays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with activities_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        for activity in csv.DictReader(handle):
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
            rows.append(row)
    return rows


def build_neighbor_index(
    evidence_rows: list[dict[str, Any]],
    fps_by_molecule: dict[str, DataStructs.ExplicitBitVect] | None = None,
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

    for molecule_id in sorted(molecule_rows):
        rows = molecule_rows[molecule_id]
        smiles = _first_nonempty(row.get("canonical_smiles") for row in rows)
        canonical_smiles, inchi_key = standardize_smiles(smiles)
        if molecule_id in fps_by_molecule:
            fp = fps_by_molecule[molecule_id]
        else:
            _, _, fp = standardize_smiles_and_fp(smiles)
        if fp is None:
            continue
        for row in rows:
            row["canonical_smiles"] = canonical_smiles or smiles
            row["standard_inchi_key"] = inchi_key
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
            }
        )
        fingerprints.append(fp)
        evidence_by_molecule_group[molecule_id] = {group_id: group_rows[group_id] for group_id in group_rows}

    return {
        "version": "bioavailability_ma_neighbor_index.v1",
        "fingerprint": {
            "type": "RDKit-Morgan",
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
            "useFeatures": False,
            "useChirality": False,
            "useBondTypes": True,
        },
        "molecules": molecules,
        "fingerprints": fingerprints,
        "group_to_molecule_indices": dict(group_to_molecule_indices),
        "evidence_by_molecule_group": evidence_by_molecule_group,
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
        for row in csv.DictReader(handle):
            assay_id = str(row.get("assay_chembl_id") or "").strip()
            if not assay_id:
                continue
            assays[assay_id] = {
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
    return assays


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


def _load_fps_subset(path: Path, molecule_ids: set[str]) -> dict[str, DataStructs.ExplicitBitVect]:
    fps: dict[str, DataStructs.ExplicitBitVect] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
    return fps


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def _first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays-csv", default=DEFAULT_ASSAYS_CSV)
    parser.add_argument("--activities-csv", default=DEFAULT_ACTIVITIES_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--chembl-fps", default=DEFAULT_CHEMBL_FPS)
    parser.add_argument("--no-chembl-fps", action="store_true", help="Compute all fingerprints from SMILES.")
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[build_evidence_library] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
