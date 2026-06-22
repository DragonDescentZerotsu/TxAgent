"""Build a combined ChEMBL Tier 1 plus Starling oral bioavailability neighbor index."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)


DEFAULT_CHEMBL_INDEX = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl"
)
DEFAULT_STARLING_INDEX = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/"
    "starling_oral_bioavailability_neighbor_index.pkl"
)
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/combined_tier1_starling"
CHEMBL_GROUPS = (
    "Tier 1.direct_absolute_bioavailability",
    "Tier 1.context_dependent",
)
STARLING_GROUPS = ("Starling.direct_oral_bioavailability",)
COMBINED_GROUP_ID = "Combined.chembl_tier1_and_starling"
INDEX_VERSION = "bioavailability_ma_combined_tier1_starling_neighbor_index.v2"
EVIDENCE_FILENAME = "combined_tier1_starling_evidence.jsonl"
INDEX_FILENAME = "combined_tier1_starling_neighbor_index.pkl"
META_FILENAME = "combined_tier1_starling_neighbor_index.meta.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    chembl_index = _load_index(Path(args.chembl_index))
    starling_index = _load_index(Path(args.starling_index))

    evidence_rows, stats = build_combined_evidence_rows(chembl_index, starling_index)
    print(
        "[build_combined_tier1_starling] "
        f"entities={stats['n_combined_molecules']:,} shared={stats['n_shared_molecules']:,} "
        f"evidence_rows={len(evidence_rows):,}",
        flush=True,
    )
    index = build_neighbor_index(
        evidence_rows,
        index_version=INDEX_VERSION,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "combined_chembl_tier1_starling",
        "sources": ["ChEMBL 36 Bioavailability_Ma Tier 1", "starling-labs/Oral_Bioavailability"],
        "chembl_index": args.chembl_index,
        "starling_index": args.starling_index,
        "chembl_groups": list(CHEMBL_GROUPS),
        "starling_groups": list(STARLING_GROUPS),
        "group_id": COMBINED_GROUP_ID,
        "molecule_merge_key": "InChIKey connectivity layer",
        "exact_query_exclusion": True,
    }

    out_dir = ensure_dir(args.out_dir)
    evidence_path = out_dir / EVIDENCE_FILENAME
    index_path = out_dir / INDEX_FILENAME
    meta_path = out_dir / META_FILENAME
    _write_jsonl(evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "index_version": INDEX_VERSION,
        "chembl_index": args.chembl_index,
        "starling_index": args.starling_index,
        "chembl_groups": list(CHEMBL_GROUPS),
        "starling_groups": list(STARLING_GROUPS),
        "combined_group_id": COMBINED_GROUP_ID,
        **stats,
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "n_groups": len(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "exact_query_exclusion": True,
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {
            "evidence_jsonl": str(evidence_path),
            "index_pkl": str(index_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_combined_evidence_rows(
    chembl_index: dict[str, Any],
    starling_index: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entities: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    _collect_source(
        entities,
        chembl_index,
        groups=CHEMBL_GROUPS,
        source_name="ChEMBL",
        source_counts=source_counts,
    )
    _collect_source(
        entities,
        starling_index,
        groups=STARLING_GROUPS,
        source_name="starling-labs/Oral_Bioavailability",
        source_counts=source_counts,
    )

    evidence_rows: list[dict[str, Any]] = []
    n_shared = 0
    n_chembl_only = 0
    n_starling_only = 0
    for merge_key in sorted(entities):
        entity = entities[merge_key]
        sources = set(entity["sources"])
        if len(sources) > 1:
            n_shared += 1
        elif "ChEMBL" in sources:
            n_chembl_only += 1
        else:
            n_starling_only += 1
        molecule_id = "COMBINED_" + hashlib.sha1(merge_key.encode("utf-8")).hexdigest()[:16].upper()
        canonical_smiles = _representative_smiles(entity["smiles"])
        for source_name, source_molecule_id, source_group_id, source_smiles, row in entity["evidence"]:
            combined = copy.deepcopy(row)
            combined.update(
                {
                    "molecule_chembl_id": molecule_id,
                    "canonical_smiles": canonical_smiles,
                    "assay_tier": "Combined",
                    "endpoint_group": "chembl_tier1_and_starling",
                    "group_id": COMBINED_GROUP_ID,
                    "evidence_source": source_name,
                    "source_molecule_id": source_molecule_id,
                    "source_group_id": source_group_id,
                    "source_canonical_smiles": source_smiles,
                    "combined_molecule_sources": sorted(sources),
                    "combined_merge_key": merge_key,
                }
            )
            evidence_rows.append(combined)

    return evidence_rows, {
        "n_combined_molecules": len(entities),
        "n_shared_molecules": n_shared,
        "n_chembl_only_molecules": n_chembl_only,
        "n_starling_only_molecules": n_starling_only,
        "n_chembl_source_molecules": source_counts["ChEMBL"],
        "n_starling_source_molecules": source_counts["starling-labs/Oral_Bioavailability"],
    }


def _collect_source(
    entities: dict[str, dict[str, Any]],
    index: dict[str, Any],
    *,
    groups: tuple[str, ...],
    source_name: str,
    source_counts: Counter[str],
) -> None:
    selected_indices: set[int] = set()
    for group_id in groups:
        selected_indices.update(index.get("group_to_molecule_indices", {}).get(group_id, []))
    for molecule_index in sorted(selected_indices):
        molecule = index["molecules"][molecule_index]
        molecule_id = str(molecule.get("molecule_chembl_id") or "")
        smiles = str(molecule.get("canonical_smiles") or "")
        inchi_key = str(molecule.get("standard_inchi_key") or "")
        merge_key = _merge_key(inchi_key, smiles)
        if not merge_key:
            continue
        source_counts[source_name] += 1
        entity = entities.setdefault(merge_key, {"sources": set(), "smiles": [], "evidence": []})
        entity["sources"].add(source_name)
        entity["smiles"].append(smiles)
        rows_by_group = index.get("evidence_by_molecule_group", {}).get(molecule_id, {})
        for group_id in groups:
            for row in rows_by_group.get(group_id, []):
                entity["evidence"].append((source_name, molecule_id, group_id, smiles, row))


def _merge_key(inchi_key: str, smiles: str) -> str:
    connectivity = inchi_key.strip().split("-", 1)[0]
    if connectivity:
        return f"inchi_connectivity:{connectivity}"
    if smiles:
        return f"canonical_smiles:{smiles}"
    return ""


def _representative_smiles(smiles_values: list[str]) -> str:
    values = sorted({value for value in smiles_values if value}, key=lambda value: (len(value), value))
    return values[0] if values else ""


def _load_index(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-index", default=DEFAULT_CHEMBL_INDEX)
    parser.add_argument("--starling-index", default=DEFAULT_STARLING_INDEX)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
