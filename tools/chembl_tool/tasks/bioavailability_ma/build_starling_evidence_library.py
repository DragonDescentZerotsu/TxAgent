"""Build a Starling-only oral bioavailability evidence library and neighbor index."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.starling import oral_bioavailability as oral_cleaning
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)


DEFAULT_SOURCE_JSONL = (
    "outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf/"
    "absolute_unspecified_systemic_broad_condition_full_text_v1/molecule_records.jsonl"
)
DEFAULT_DROPPED_JSONL = (
    "outputs/chembl_tool/activity_transfer_benchmark/oral_bioavailability_hf/"
    "absolute_unspecified_systemic_broad_condition_full_text_v1/dropped_rows.jsonl"
)
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling"
EVIDENCE_FILENAME = "starling_oral_bioavailability_evidence.jsonl"
INDEX_FILENAME = "starling_oral_bioavailability_neighbor_index.pkl"
META_FILENAME = "starling_oral_bioavailability_neighbor_index.meta.json"
GROUP_ID = "Observed.direct_oral_bioavailability"
INDEX_VERSION = "bioavailability_ma_starling_neighbor_index.v2"
SOURCE_DATASET = "starling-labs/Oral_Bioavailability"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    source_path = Path(args.source_jsonl) if args.source_mode == "prepared-jsonl" else None
    dropped_path = Path(args.dropped_jsonl) if args.dropped_jsonl else None
    out_dir = ensure_dir(args.out_dir)

    if args.source_mode == "disabled":
        evidence_rows, source_stats = [], {"source_mode": "disabled", "n_source_rows": 0, "n_source_rows_kept": 0}
    elif args.source_mode == "pinned-hf":
        evidence_rows, source_stats = build_starling_evidence_rows_from_pinned_hf(
            min_value_percent=args.min_value_percent, max_value_percent=args.max_value_percent,
            max_record_examples=args.max_record_examples, evidence_content=args.evidence_content,
        )
    else:
        assert source_path is not None
        evidence_rows, source_stats = build_starling_evidence_rows(
            source_path, dropped_jsonl=dropped_path if args.evidence_content == "full" else None,
            min_value_percent=args.min_value_percent, max_value_percent=args.max_value_percent,
            max_record_examples=args.max_record_examples,
        )
    print(
        f"[build_starling_evidence_library] evidence molecules={len(evidence_rows):,} "
        f"source rows kept={source_stats['n_source_rows_kept']:,}",
        flush=True,
    )

    index = build_neighbor_index(
        evidence_rows,
        index_version=INDEX_VERSION,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "starling_oral_bioavailability",
        "dataset": SOURCE_DATASET,
        "source_mode": args.source_mode,
        "source_revision": oral_cleaning.ORAL_BIOAVAILABILITY_REVISION if args.source_mode == "pinned-hf" else "",
        "source_jsonl": str(source_path) if source_path else "",
        "dropped_jsonl": str(dropped_path) if dropped_path else "",
        "group_id": GROUP_ID,
        "exact_query_exclusion": True,
    }

    evidence_path = out_dir / EVIDENCE_FILENAME
    index_path = out_dir / INDEX_FILENAME
    meta_path = out_dir / META_FILENAME
    _write_jsonl(evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "index_version": INDEX_VERSION,
        "source_dataset": SOURCE_DATASET,
        "source_mode": args.source_mode,
        "source_jsonl": str(source_path) if source_path else "",
        "source_revision": oral_cleaning.ORAL_BIOAVAILABILITY_REVISION if args.source_mode == "pinned-hf" else "",
        "dropped_jsonl": str(dropped_path) if dropped_path else "",
        "group_id": GROUP_ID,
        "min_value_percent": args.min_value_percent,
        "max_value_percent": args.max_value_percent,
        **source_stats,
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


def build_starling_evidence_rows(
    source_jsonl: Path,
    *,
    dropped_jsonl: Path | None = None,
    min_value_percent: float = 0.0,
    max_value_percent: float = 100.0,
    max_record_examples: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    numeric_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualitative_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_source_rows = 0
    n_source_rows_kept = 0
    n_out_of_range = 0
    with source_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_source_rows += 1
            row = json.loads(line)
            report_type = str((row.get("metadata") or {}).get("bioavailability_report_type") or "").strip()
            if report_type not in oral_cleaning.ALLOWED_REPORT_TYPES:
                continue
            smiles = _canonicalize_smiles(str(row.get("smiles") or ""))
            try:
                value = float(row.get("oral_bioavailability_value_percent"))
            except (TypeError, ValueError):
                continue
            if not smiles or value < min_value_percent or value > max_value_percent:
                n_out_of_range += 1
                continue
            row["_value_percent"] = value
            numeric_by_smiles[smiles].append(row)
            n_source_rows_kept += 1

    qualitative_stats = _load_qualitative_rows(dropped_jsonl, qualitative_by_smiles)
    return _build_evidence_rows(numeric_by_smiles, qualitative_by_smiles, {
        "n_source_rows": n_source_rows,
        "n_source_rows_kept": n_source_rows_kept,
        "n_source_rows_out_of_range_or_invalid": n_out_of_range,
        **qualitative_stats,
    }, max_record_examples=max_record_examples)


def build_starling_evidence_rows_from_pinned_hf(
    *, min_value_percent: float = 0.0, max_value_percent: float = 100.0,
    max_record_examples: int = 6, evidence_content: str = "full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Directly ingest the pinned upstream dataset, retaining qualitative rows only in full mode."""
    dataset = oral_cleaning.load_pinned_oral_bioavailability_dataset()
    clean_rows, dropped = oral_cleaning.clean_oral_bioavailability_rows(
        dataset, min_value_percent=min_value_percent, max_value_percent=max_value_percent,
    )
    numeric_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualitative_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        numeric_by_smiles[row.canonical_smiles].append(_clean_row_to_record(row))
    qualitative_stats = (
        _load_qualitative_dropped_records(dropped, qualitative_by_smiles)
        if evidence_content == "full"
        else {"n_dropped_rows_scanned": 0, "n_qualitative_rows_kept": 0,
              "n_qualitative_rows_invalid_smiles": 0, "n_qualitative_rows_empty": 0}
    )
    rows, stats = _build_evidence_rows(numeric_by_smiles, qualitative_by_smiles, {
        "n_source_rows": len(clean_rows) + len(dropped), "n_source_rows_kept": len(clean_rows),
        "n_source_rows_out_of_range_or_invalid": sum(1 for item in dropped if item["drop_reason"] in {"value_out_of_range", "invalid_smiles"}),
        **qualitative_stats,
    }, max_record_examples=max_record_examples)
    stats.update({"source_mode": "pinned-hf", "dataset": oral_cleaning.ORAL_BIOAVAILABILITY_DATASET,
                  "revision": oral_cleaning.ORAL_BIOAVAILABILITY_REVISION})
    return rows, stats


def _build_evidence_rows(
    numeric_by_smiles: dict[str, list[dict[str, Any]]], qualitative_by_smiles: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any], *, max_record_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_smiles = sorted(set(numeric_by_smiles) | set(qualitative_by_smiles))
    evidence_rows = [
        _summarize_molecule(
            smiles,
            numeric_by_smiles.get(smiles, []),
            qualitative_by_smiles.get(smiles, []),
            max_record_examples=max_record_examples,
        )
        for smiles in all_smiles
    ]
    return evidence_rows, {
        **stats,
        "n_unique_numeric_source_smiles": len(numeric_by_smiles),
        "n_unique_source_smiles": len(all_smiles),
        "n_molecules_with_numeric_evidence": sum(smiles in numeric_by_smiles for smiles in all_smiles),
        "n_molecules_with_qualitative_evidence": sum(smiles in qualitative_by_smiles for smiles in all_smiles),
        "n_molecules_with_qualitative_only_evidence": sum(
            smiles not in numeric_by_smiles and smiles in qualitative_by_smiles for smiles in all_smiles
        ),
    }


def _clean_row_to_record(row: oral_cleaning.CleanOralBioavailabilityRow) -> dict[str, Any]:
    return {"source_index": row.source_index, "molecule_name": row.molecule_name, "smiles": row.canonical_smiles,
            "_value_percent": row.value_percent, "condition_text": row.condition_text,
            "parse_modifier": row.parse_modifier, "metadata": row.raw_row}


def _summarize_molecule(
    smiles: str,
    numeric_rows: list[dict[str, Any]],
    qualitative_rows: list[dict[str, Any]],
    *,
    max_record_examples: int,
) -> dict[str, Any]:
    values = sorted(float(row["_value_percent"]) for row in numeric_rows)
    median_value = float(statistics.median(values)) if values else None
    all_rows = [*numeric_rows, *qualitative_rows]
    names = _unique_text(row.get("molecule_name") for row in all_rows)
    report_types = _unique_text((row.get("metadata") or {}).get("bioavailability_report_type") for row in all_rows)
    species = _unique_text((row.get("metadata") or {}).get("species_or_population") for row in all_rows)
    pmids = _unique_text((row.get("metadata") or {}).get("pmid") for row in all_rows)
    support_texts = _unique_text((row.get("metadata") or {}).get("support_text") for row in all_rows)
    record_examples = _representative_numeric_record_examples(numeric_rows, limit=max_record_examples)
    qualitative_examples = _representative_qualitative_examples(qualitative_rows, limit=max_record_examples)
    molecule_id = "STARLING_" + hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16].upper()
    relation = _summary_relation(numeric_rows) if numeric_rows else ""
    if numeric_rows:
        activity_comment = (
            f"Starling literature summary over {len(numeric_rows)} numeric records: "
            f"median {median_value:.2f}%, range {values[0]:.2f}-{values[-1]:.2f}%."
        )
        if qualitative_rows:
            activity_comment += f" Also retained {len(qualitative_rows)} qualitative/contextual records."
        standard_type = "Oral bioavailability"
        standard_value: float | str = round(median_value, 6)
        standard_units = "%"
    else:
        activity_comment = (
            f"Starling qualitative/contextual oral bioavailability summary over "
            f"{len(qualitative_rows)} records; no reliable numeric F% was parsed."
        )
        standard_type = "Oral bioavailability (qualitative/contextual)"
        standard_value = ""
        standard_units = ""
    row = {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "assay_chembl_id": "STARLING_ORAL_BIOAVAILABILITY",
        "assay_tier": "Observed",
        "endpoint_group": "direct_oral_bioavailability",
        "group_id": GROUP_ID,
        "standard_type": standard_type,
        "standard_relation": relation,
        "standard_value": standard_value,
        "standard_units": standard_units,
        "pchembl_value": "",
        "activity_comment": activity_comment,
        "data_validity_comment": "",
        "assay_description": _record_examples_text(
            record_examples,
            qualitative_examples,
            max_chars=5000,
        ),
        "target_pref_name": "oral bioavailability",
        "target_genes": "",
        "organism": "; ".join(species[:8]),
        "confidence_score": "",
        "relationship_type": "",
        "evidence_source": SOURCE_DATASET,
        "evidence_role": "direct_outcome",
        "evidence_scope": {
            "species_or_population": species[:8],
            "report_types": report_types,
        },
        "transferability": "not_assessed",
        "uncertainty": ["qualitative_only_no_numeric_measurement"] if not numeric_rows else [],
        "source_molecule_names": names[:10],
        "source_record_count": len(all_rows),
        "source_numeric_record_count": len(numeric_rows),
        "source_qualitative_record_count": len(qualitative_rows),
        "source_value_min_percent": round(values[0], 6) if values else "",
        "source_value_median_percent": round(median_value, 6) if values else "",
        "source_value_max_percent": round(values[-1], 6) if values else "",
        "source_report_types": report_types,
        "source_pmids": pmids[:20],
        "source_support_texts": support_texts[:max_record_examples],
        "source_record_examples": record_examples,
        "source_qualitative_examples": qualitative_examples,
    }
    return attach_minimal_evidence(row)


def _summary_relation(rows: list[dict[str, Any]]) -> str:
    modifiers = {str(row.get("parse_modifier") or "").strip() for row in rows}
    if modifiers == {"lower_bound"}:
        return ">"
    if modifiers == {"upper_bound"}:
        return "<"
    return "="


def _unique_text(values: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _representative_numeric_record_examples(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in sorted(rows, key=lambda item: (float(item["_value_percent"]), int(item.get("source_index", 0)))):
        metadata = row.get("metadata") or {}
        key = (
            round(float(row["_value_percent"]), 8),
            str(row.get("condition_text") or "").strip(),
            str(metadata.get("support_text") or "").strip(),
        )
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return [_numeric_record_example(row) for row in _evenly_spaced_items(unique_rows, limit)]


def _numeric_record_example(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "source_index": row.get("source_index", ""),
        "molecule_name": str(row.get("molecule_name") or "").strip(),
        "oral_bioavailability_value_percent": round(float(row["_value_percent"]), 6),
        "parse_modifier": str(row.get("parse_modifier") or "").strip(),
        "condition_text": str(row.get("condition_text") or "").strip(),
        "species_or_population": str(metadata.get("species_or_population") or "").strip(),
        "dose": str(metadata.get("dose") or "").strip(),
        "oral_exposure_mode": str(metadata.get("oral_exposure_mode") or "").strip(),
        "qualifying_conditions": str(metadata.get("qualifying_conditions") or "").strip(),
        "comparator": str(metadata.get("comparator") or "").strip(),
        "extra_details": str(metadata.get("extra_details") or "").strip(),
        "pmid": str(metadata.get("pmid") or "").strip(),
        "support_text": str(metadata.get("support_text") or "").strip(),
        "bioavailability_report_type": str(metadata.get("bioavailability_report_type") or "").strip(),
    }


def _representative_qualitative_examples(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        key = (
            str(row.get("_qualitative_value_text") or "").strip().lower(),
            str(row.get("condition_text") or "").strip(),
            str(metadata.get("support_text") or "").strip(),
        )
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return [_qualitative_record_example(row) for row in _evenly_spaced_items(unique_rows, limit)]


def _qualitative_record_example(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "source_index": row.get("source_index", ""),
        "molecule_name": str(row.get("molecule_name") or "").strip(),
        "oral_bioavailability_value_text": str(row.get("_qualitative_value_text") or "").strip(),
        "condition_text": str(row.get("condition_text") or "").strip(),
        "species_or_population": str(metadata.get("species_or_population") or "").strip(),
        "dose": str(metadata.get("dose") or "").strip(),
        "oral_exposure_mode": str(metadata.get("oral_exposure_mode") or "").strip(),
        "qualifying_conditions": str(metadata.get("qualifying_conditions") or "").strip(),
        "comparator": str(metadata.get("comparator") or "").strip(),
        "extra_details": str(metadata.get("extra_details") or "").strip(),
        "pmid": str(metadata.get("pmid") or "").strip(),
        "support_text": str(metadata.get("support_text") or "").strip(),
        "bioavailability_report_type": str(metadata.get("bioavailability_report_type") or "").strip(),
        "source_drop_reason": str(row.get("_source_drop_reason") or "").strip(),
    }


def _evenly_spaced_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in dict.fromkeys(indices)]


def _record_examples_text(
    numeric_examples: list[dict[str, Any]],
    qualitative_examples: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    blocks = []
    for example in numeric_examples:
        value = example["oral_bioavailability_value_percent"]
        blocks.append(
            f"oral_bioavailability_value_percent: {value:g}\n"
            f"{example['condition_text']}"
        )
    for example in qualitative_examples:
        blocks.append(
            f"oral_bioavailability_value_text: {example['oral_bioavailability_value_text'] or 'not specified'}\n"
            f"{example['condition_text']}\n"
            f"support_text: {example['support_text']}"
        )
    text = "\n\n".join(blocks)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _load_qualitative_rows(
    dropped_jsonl: Path | None,
    qualitative_by_smiles: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    stats = {
        "n_dropped_rows_scanned": 0,
        "n_qualitative_rows_kept": 0,
        "n_qualitative_rows_invalid_smiles": 0,
        "n_qualitative_rows_empty": 0,
    }
    if dropped_jsonl is None or not dropped_jsonl.exists():
        return stats
    allowed_reasons = {"unparseable_or_non_numeric_value", "report_type_not_allowed"}
    with dropped_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["n_dropped_rows_scanned"] += 1
            dropped = json.loads(line)
            reason = str(dropped.get("drop_reason") or "")
            if reason not in allowed_reasons:
                continue
            _append_qualitative_dropped_record(dropped, qualitative_by_smiles, stats)
    return stats


def _load_qualitative_dropped_records(
    dropped_rows: list[dict[str, Any]], qualitative_by_smiles: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    stats = {"n_dropped_rows_scanned": 0, "n_qualitative_rows_kept": 0,
             "n_qualitative_rows_invalid_smiles": 0, "n_qualitative_rows_empty": 0}
    for dropped in dropped_rows:
        stats["n_dropped_rows_scanned"] += 1
        _append_qualitative_dropped_record(dropped, qualitative_by_smiles, stats)
    return stats


def _append_qualitative_dropped_record(
    dropped: dict[str, Any], qualitative_by_smiles: dict[str, list[dict[str, Any]]], stats: dict[str, int],
) -> None:
    reason = str(dropped.get("drop_reason") or "")
    if reason not in {"unparseable_or_non_numeric_value", "report_type_not_allowed"}:
        return
    raw = dropped.get("raw_row") or {}
    # Relative comparisons are intentionally excluded, including from full evidence.
    if str(raw.get("bioavailability_report_type") or "").strip() not in oral_cleaning.ALLOWED_REPORT_TYPES:
        return
    value_text = str(raw.get("oral_bioavailability_value") or "").strip()
    support_text = str(raw.get("support_text") or "").strip()
    if not value_text and not support_text:
        stats["n_qualitative_rows_empty"] += 1
        return
    smiles = _canonicalize_smiles(str(raw.get("smiles") or ""))
    if not smiles:
        stats["n_qualitative_rows_invalid_smiles"] += 1
        return
    qualitative_by_smiles[smiles].append({
        "source_index": dropped.get("source_index", ""), "molecule_name": str(raw.get("molecule_name") or "").strip(),
        "condition_text": _condition_text(raw), "metadata": raw, "_qualitative_value_text": value_text,
        "_source_drop_reason": reason,
    })
    stats["n_qualitative_rows_kept"] += 1


def _canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _condition_text(row: dict[str, Any]) -> str:
    fields = [
        "species_or_population",
        "dose",
        "oral_exposure_mode",
        "qualifying_conditions",
        "comparator",
        "extra_details",
    ]
    return "\n".join(f"{field}: {str(row.get(field) or 'not specified').strip()}" for field in fields)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--dropped-jsonl", default=DEFAULT_DROPPED_JSONL)
    parser.add_argument("--source-mode", choices=["pinned-hf", "prepared-jsonl", "disabled"], default="prepared-jsonl")
    parser.add_argument("--evidence-content", choices=["numeric_only", "full"], default="full")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-value-percent", type=float, default=0.0)
    parser.add_argument("--max-value-percent", type=float, default=100.0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
