"""Build the in-distribution Bioavailability_Ma evidence index from starling's
normalized `hf_cleaned` records (the exact data the assay-transfer model trained on).

Unlike TxAgent's evidence library (which aggregates raw extractions and later
reconstructs scoring records with synthetic `index.*` endpoint keys), this builder emits
one evidence row per **normalized** starling record, preserving the real
`canonical_endpoint_key`, `scalar_value`, `unit_basis`, context, provenance, and raw
narrative. Those rows feed:

* the neighbor index (Morgan retrieval over the starling molecules), and
* the in-distribution assay-transfer catalog (scored with the exact v6_5 template), and
* the LLM group/final presentation (raw metadata).

See `IN_DISTRIBUTION_DESIGN.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Iterator

from pathlib import Path as _Path

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    CATALOG_SCHEMA_VERSION,
    TEMPLATE_BY_CONCEPT,
    V6_5_TEMPLATE_PROFILE,
    full_record_example,
    template_bundle_hash,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_assay_transfer_rerank_catalog import (
    _context,
    _v3_scoring_profile,
)

# Authoritative per-record training source: assay_concept + pre-mapped context_<field>
# columns + normalized scalar/unit/endpoint -> exact v6_5 training-prompt fidelity.
DEFAULT_STARLING_ROOT = os.getenv(
    "TXAGENT_STARLING_ROOT", "/data1/joseph/starling_assay_transfer"
)
DEFAULT_HF_CLEANED_DIR = str(
    Path(DEFAULT_STARLING_ROOT)
    / "datasets/eligible/assay_transfer_soft_evidence_v6_5/records.parquet"
)
# Canonical base the eligible v6.5 records were built from; carries the full support_text
# narrative, joinable by child_id (100% coverage). Used for LLM presentation only.
DEFAULT_SUPPORT_TEXT_BASE = str(
    Path(DEFAULT_STARLING_ROOT) / "datasets/base/canonical_endpoints_v3"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution"
)
EVIDENCE_FILENAME = "starling_in_distribution_evidence.jsonl"
INDEX_FILENAME = "starling_in_distribution_neighbor_index.pkl"
CATALOG_FILENAME = "starling_in_distribution_catalog.jsonl"
META_FILENAME = "starling_in_distribution_neighbor_index.meta.json"
INDEX_VERSION = "bioavailability_ma_starling_in_distribution_neighbor_index.v1"

# canonical_endpoint_key prefix -> (group_id, tier, endpoint_group). q1 splits on the
# second component (oral_bioavailability vs oral_exposure); q2/q3/q4 map to Fa/Fg/Fh.
GROUP_BY_CONCEPT: dict[str, tuple[str, str, str]] = {
    "oral_bioavailability": (
        "Observed.direct_oral_bioavailability", "Observed", "direct_oral_bioavailability",
    ),
    "oral_exposure": (
        "Observed.oral_auc_cmax_exposure", "Observed", "oral_auc_cmax_exposure",
    ),
    "Fa": ("Fa.absorption_solubility_permeability", "Fa", "absorption_solubility_permeability"),
    "Fg": ("Fg.gut_wall_efflux_intestinal_metabolism", "Fg", "gut_wall_efflux_intestinal_metabolism"),
    "Fh": ("Fh.hepatic_clearance_metabolic_stability", "Fh", "hepatic_clearance_metabolic_stability"),
}

# Normalized scoring/template fields carried verbatim onto each evidence row's
# `starling_record` (mirrors the fields the v6_5 template + presentation consume).
STARLING_RECORD_FIELDS = (
    "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
    "metric_type", "scalar_value", "unit_normalized", "scalar_is_approximate",
    "variation_value", "variation_type", "direction", "target",
    "species_exact", "support_text", "extra_details",
    # raw context (present in hf_cleaned; used for the template context_* slots + presentation)
    "assay_system", "study_context", "condition_medium", "biological_context",
    "formulation_or_solid_form", "transporter_or_enzyme", "substrate_status",
    "intestinal_site", "molecular_form", "enzyme_or_pathway", "qualifying_conditions",
    "comparator", "comparator_exposure", "statistic_type", "oral_dose", "dose",
    "species_or_population", "pmid",
    # provenance
    "record_id", "parent_provenance_id", "input_sha256", "source_id", "source_row_number",
)


def _record_concept(record: dict[str, Any]) -> str | None:
    """The starling assay_concept for a record: the authoritative column if present,
    else derived from the canonical endpoint key prefix."""
    concept = str(record.get("assay_concept") or "")
    if concept in GROUP_BY_CONCEPT:
        return concept
    return concept_from_endpoint_key(str(record.get("canonical_endpoint_key") or ""))


def _record_smiles(record: dict[str, Any]) -> str:
    return str(record.get("canonical_smiles") or record.get("smiles") or "").strip()


def load_support_text_by_child_id(base_dir: Path) -> dict[str, str]:
    """Map child_id -> full support_text narrative from the canonical base (all sources).

    The eligible v6.5 records drop support_text; the canonical base they were built from
    (canonical_endpoints_v3) retains it, joinable by child_id. Returns {} if absent.
    """
    import pyarrow.parquet as pq

    if not base_dir.exists():
        return {}
    mapping: dict[str, str] = {}
    for path in sorted(base_dir.glob("*/records.parquet")):
        table = pq.read_table(path, columns=["child_id", "support_text"])
        for child_id, support_text in zip(
            table.column("child_id").to_pylist(), table.column("support_text").to_pylist()
        ):
            if child_id and support_text:
                mapping[str(child_id)] = str(support_text)
    return mapping


def _resolve_support_text(record: dict[str, Any], support_by_id: dict[str, str] | None) -> str:
    """Full support_text via the child_id join, falling back to context_extra_details."""
    child_id = str(record.get("child_id") or record.get("record_id") or "")
    if support_by_id and child_id in support_by_id:
        return support_by_id[child_id]
    return str(record.get("support_text") or record.get("context_extra_details") or "")


def concept_from_endpoint_key(canonical_endpoint_key: str) -> str | None:
    """Return the starling assay_concept for a canonical endpoint key, or None."""
    parts = str(canonical_endpoint_key or "").split(".")
    if len(parts) < 2:
        return None
    collection, second = parts[0], parts[1]
    if collection == "q2":
        return "Fa"
    if collection == "q3":
        return "Fg"
    if collection == "q4":
        return "Fh"
    if collection == "q1":
        if second == "oral_bioavailability":
            return "oral_bioavailability"
        if second == "oral_exposure":
            return "oral_exposure"
    return None


def _first(record: dict[str, Any], *names: str) -> Any:
    """First non-empty value among the named columns (mirrors compose_v3._first)."""
    return next((record.get(name) for name in names if record.get(name) not in (None, "")), None)


# The 16 v6_5 template context fields (order per the template).
_TEMPLATE_CONTEXT_FIELDS = (
    "species_or_population", "report_or_statistic_type", "dose", "study_or_assay_system",
    "measured_process", "biological_context", "medium", "formulation_or_solid_form",
    "transporter_or_enzyme", "substrate_status", "intestinal_site", "molecular_form",
    "enzyme_or_pathway", "qualifying_conditions", "comparator", "extra_details",
)


def template_context_from_record(record: dict[str, Any]) -> dict[str, str]:
    """Return the 16 v6_5 template context fields for a normalized record.

    Prefers the authoritative pre-mapped `context_<field>` columns (present in the
    starling eligible v6.5 training records — exact training fidelity). Falls back to the
    `compose_v3._context` mapping over raw columns when those aren't present.
    """
    if any(f"context_{field}" in record for field in _TEMPLATE_CONTEXT_FIELDS):
        return _context(**{field: record.get(f"context_{field}") for field in _TEMPLATE_CONTEXT_FIELDS})
    mapped = {
        "species_or_population": _first(record, "species_or_population", "species", "species_exact"),
        "report_or_statistic_type": _first(record, "bioavailability_report_type", "statistic_type"),
        "dose": _first(record, "dose", "oral_dose"),
        "study_or_assay_system": _first(record, "study_context", "oral_exposure_mode", "assay_system"),
        "measured_process": _first(record, "exposure_measure", "endpoint_category", "gut_wall_process", "metric_type"),
        "biological_context": record.get("biological_context"),
        "medium": record.get("condition_medium"),
        "formulation_or_solid_form": record.get("formulation_or_solid_form"),
        "transporter_or_enzyme": record.get("transporter_or_enzyme"),
        "substrate_status": record.get("substrate_status"),
        "intestinal_site": record.get("intestinal_site"),
        "molecular_form": record.get("molecular_form"),
        "enzyme_or_pathway": record.get("enzyme_or_pathway"),
        "qualifying_conditions": record.get("qualifying_conditions"),
        "comparator": _first(record, "comparator", "comparator_exposure"),
        "extra_details": record.get("extra_details"),
    }
    return _context(**mapped)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def in_distribution_catalog_record(
    record: dict[str, Any], support_by_id: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Build a v6_5-renderable catalog record from a normalized record.

    Returns None for records without a finite scalar (e.g. categorical substrate-status),
    which are correctly unscoreable under the scalar-only v6_5 template. `support_by_id`
    supplies the full narrative for LLM presentation (scoring is unaffected).
    """
    canonical_endpoint_key = str(record.get("canonical_endpoint_key") or "")
    concept = _record_concept(record)
    if concept is None:
        return None
    scalar = _finite_float(record.get("scalar_value"))
    if scalar is None:
        return None
    smiles = _record_smiles(record)
    if not smiles:
        return None
    endpoint_subtype = str(record.get("endpoint_subtype") or "")
    unit_basis = str(record.get("unit_basis") or "")
    metric_type, threshold_display = _v3_scoring_profile(
        concept=concept, endpoint_subtype=endpoint_subtype, unit_basis=unit_basis
    )
    # child_id is unique per scalar emission; record_id is the (shared) parent. Use
    # child_id so scalar-split records don't collide in the catalog.
    catalog_record_id = str(record.get("child_id") or record.get("record_id") or "")
    return {
        "record_type": "assay_record",
        "record_id": catalog_record_id,
        "source_id": str(record.get("source_id") or ""),
        "original_smiles": smiles,
        "canonical_smiles": smiles,
        "assay_concept": concept,
        "canonical_endpoint_key": canonical_endpoint_key,
        "endpoint_family": str(record.get("endpoint_family") or ""),
        "endpoint_subtype": endpoint_subtype,
        "unit_basis": unit_basis,
        "unit_normalized": str(record.get("unit_normalized") or ""),
        "metric_type": metric_type,
        "threshold_display": threshold_display,
        "value": scalar,
        "value_display": str(record.get("scalar_value")).strip(),
        "measurement_label": endpoint_subtype.replace("_", " "),
        # scientific detail surfaced by the per-source `full` presentation style.
        "direction": str(record.get("direction") or ""),
        "variation_type": str(record.get("variation_type") or ""),
        "variation_value": (
            "" if record.get("variation_value") in (None, "")
            else str(record.get("variation_value")).strip()
        ),
        "statistic_type": str(record.get("statistic_type") or ""),
        # full narrative for the LLM presentation (joined from the canonical base by
        # child_id; not used by the scoring template).
        "support_text": _resolve_support_text(record, support_by_id),
        "extra_details": str(record.get("extra_details") or record.get("context_extra_details") or ""),
        "template_id": _Path(TEMPLATE_BY_CONCEPT[concept]).stem,
        "template_context": template_context_from_record(record),
        "source_provenance": {
            "child_id": catalog_record_id,
            "record_id": str(record.get("record_id") or ""),
            "parent_provenance_id": str(record.get("parent_provenance_id") or ""),
            "input_sha256": str(record.get("input_sha256") or ""),
            "source_id": str(record.get("source_id") or ""),
        },
    }


def molecule_id_for(smiles: str) -> str:
    """Stable per-molecule id from the (raw) SMILES; index re-canonicalizes for grouping."""
    import hashlib

    return "STARLING_ID_" + hashlib.sha1(str(smiles).strip().encode("utf-8")).hexdigest()[:16].upper()


def _hf_rows(hf_cleaned_dir: Path, *, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = [hf_cleaned_dir] if hf_cleaned_dir.is_file() else sorted(hf_cleaned_dir.glob("**/*.parquet"))
    emitted = 0
    for path in files:
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row
            emitted += 1
            if max_rows and emitted >= max_rows:
                return


def evidence_row_from_record(
    record: dict[str, Any], support_by_id: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """Convert one normalized record to an in-distribution evidence row."""
    canonical_endpoint_key = str(record.get("canonical_endpoint_key") or "")
    concept = _record_concept(record)
    if concept is None:
        return None
    smiles = _record_smiles(record)
    if not smiles:
        return None
    group_id, tier, endpoint_group = GROUP_BY_CONCEPT[concept]
    support_text = _resolve_support_text(record, support_by_id)
    starling_record = {field: record.get(field) for field in STARLING_RECORD_FIELDS}
    starling_record["assay_concept"] = concept
    starling_record["support_text"] = support_text
    catalog_record = in_distribution_catalog_record(record, support_by_id)
    # Legacy minimal example (5 fields) for the default presentation, enriched with the
    # full scientific fields (from the catalog record) so `--presentation-style full` can
    # show them. The legacy policy ignores the extra keys, so the legacy view is unchanged.
    example = {
        "endpoint_type": str(record.get("endpoint_subtype") or canonical_endpoint_key),
        "reported_value": record.get("scalar_value"),
        "reported_units": record.get("unit_basis"),
        "context": template_context_from_record(record),
        "support_text": support_text,
    }
    if catalog_record is not None:
        example.update(full_record_example(catalog_record))
    return {
        "molecule_chembl_id": molecule_id_for(smiles),
        "canonical_smiles": smiles,
        "group_id": group_id,
        "assay_tier": tier,
        "endpoint_group": endpoint_group,
        "evidence_source": f"starling-in-distribution/{concept}",
        "source_record_id": record.get("record_id"),
        "source_index": record.get("source_row_number"),
        "source_molecule_names": [record.get("molecule_name")] if record.get("molecule_name") else [],
        # minimal_evidence.v1 contract fields (normalized value/endpoint/narrative)
        "standard_type": canonical_endpoint_key,
        "standard_value": record.get("scalar_value"),
        "standard_units": record.get("unit_basis"),
        "support_text": support_text,
        "extra_details": record.get("extra_details") or record.get("context_extra_details"),
        "confidence_score": record.get("confidence"),
        # one minimal-evidence example so the morganfingerprint presentation
        # (evidence_for_llm -> examples) shows this normalized record.
        "source_record_examples": [example],
        # full normalized record for the catalog + template + presentation
        "starling_record": starling_record,
        # v6_5-renderable scoring record (None when unscoreable, e.g. categorical-only)
        "catalog_record": catalog_record,
    }


def build_in_distribution_evidence_rows(
    hf_cleaned_dir: Path, *, max_rows: int = 0, support_by_id: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_group: dict[str, int] = {}
    scanned = 0
    skipped = 0
    n_support = 0
    for record in _hf_rows(hf_cleaned_dir, max_rows=max_rows):
        scanned += 1
        row = evidence_row_from_record(record, support_by_id)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        per_group[row["group_id"]] = per_group.get(row["group_id"], 0) + 1
        if row.get("support_text"):
            n_support += 1
    stats = {
        "n_records_scanned": scanned,
        "n_evidence_rows": len(rows),
        "n_skipped": skipped,
        "n_rows_with_support_text": n_support,
        "rows_per_group": per_group,
    }
    return rows, stats


def write_in_distribution_catalog(evidence_rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Write a reranker-loadable catalog (metadata + assay_record rows) from evidence rows.

    Dedups scoreable records by their catalog record id (child_id). The catalog is
    profile-agnostic; the renderer picks legacy vs v6_5 at scoring time.
    """
    import hashlib

    by_id: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        catalog_record = row.get("catalog_record")
        if not catalog_record:
            continue
        record_id = str(catalog_record["record_id"])
        if record_id and record_id not in by_id:
            by_id[record_id] = catalog_record
    records = sorted(by_id.values(), key=lambda r: str(r["record_id"]))
    digest = hashlib.sha256(
        "\n".join(json.dumps(r, sort_keys=True) for r in records).encode("utf-8")
    ).hexdigest()
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": f"{CATALOG_SCHEMA_VERSION}:in_distribution:{digest}",
        "template_hash": template_bundle_hash(profile=V6_5_TEMPLATE_PROFILE),
        "template_profile": V6_5_TEMPLATE_PROFILE,
        "source_mode": "starling_in_distribution_hf_cleaned",
        "n_records": len(records),
    }
    with path.open("w", encoding="utf-8") as handle:
        for row in (metadata, *records):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"n_catalog_records": len(records), "catalog_version": metadata["catalog_version"]}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    out_dir = ensure_dir(args.out_dir)

    support_by_id = load_support_text_by_child_id(Path(args.support_text_base))
    evidence_rows, stats = build_in_distribution_evidence_rows(
        Path(args.hf_cleaned_dir), max_rows=args.max_rows, support_by_id=support_by_id
    )
    stats["n_support_text_map"] = len(support_by_id)
    index = build_neighbor_index(
        evidence_rows,
        index_version=INDEX_VERSION,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "starling_in_distribution_index",
        "dataset": "starling_assay_transfer/hf_cleaned",
        "groups": sorted(index.get("group_to_molecule_indices", {})),
        "exact_query_exclusion": True,
    }

    evidence_path = out_dir / EVIDENCE_FILENAME
    index_path = out_dir / INDEX_FILENAME
    meta_path = out_dir / META_FILENAME
    with evidence_path.open("w", encoding="utf-8") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    catalog_path = out_dir / CATALOG_FILENAME
    catalog_stats = write_in_distribution_catalog(evidence_rows, catalog_path)
    meta = {
        "index_version": INDEX_VERSION,
        "hf_cleaned_dir": args.hf_cleaned_dir,
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "build_stats": stats,
        "catalog_stats": catalog_stats,
        "paths_catalog": str(catalog_path),
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {"evidence_jsonl": str(evidence_path), "index_pkl": str(index_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cleaned-dir", default=DEFAULT_HF_CLEANED_DIR)
    parser.add_argument(
        "--support-text-base",
        default=DEFAULT_SUPPORT_TEXT_BASE,
        help="Canonical base (canonical_endpoints_v3) providing full support_text, joined by child_id for LLM presentation. Optional.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug cap on records scanned; 0 = all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
