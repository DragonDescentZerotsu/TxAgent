"""Build flat Starling assay-context catalogs.

The experiment treats ``canonical_assay_context`` as the assay unit.  Endpoint
names remain part of the assay description, rather than splitting one context
into many assay units.  Records without a canonical context use a stable
endpoint fallback so that direct outcome evidence is not silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.common.json_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


CATALOG_VERSION = "starling_assay_context_catalog.v1"
RAW_ASSAY_FIELDS = (
    "assay_system",
    "assay_model",
    "assay_or_test",
    "assay_method",
    "canonical_assay_type",
    "canonical_assay_or_test",
    "canonical_assay_method",
    "canonical_study_design",
)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {
        "none",
        "null",
        "nan",
        "<na>",
        "nat",
        "unknown",
        "__unknown__",
    }:
        return None
    return text


def assay_unit(context: Any, endpoint: Any) -> tuple[str, str]:
    """Return the context-level assay key and its provenance type."""

    clean_context = _clean_text(context)
    if clean_context:
        return clean_context, "canonical_assay_context"
    clean_endpoint = _clean_text(endpoint) or "__unknown_endpoint__"
    return f"endpoint_fallback::{clean_endpoint}", "endpoint_fallback"


def assay_id(task: str, unit: str) -> str:
    payload = f"{CATALOG_VERSION}\0{task}\0{unit}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _top_values(
    frame: Any,
    column: str,
    *,
    limit: int,
    include_molecules: bool = False,
) -> list[dict[str, Any]]:
    clean = frame.loc[frame[column].notna(), [column, "molecule_id"]]
    if clean.empty:
        return []
    grouped = clean.groupby(column, dropna=False).agg(
        record_count=(column, "size"),
        unique_molecule_count=("molecule_id", "nunique"),
    )
    grouped = grouped.sort_values(
        ["record_count", "unique_molecule_count"], ascending=False
    ).head(limit)
    rows = []
    for value, summary in grouped.iterrows():
        row: dict[str, Any] = {
            "value": str(value),
            "record_count": int(summary["record_count"]),
        }
        if include_molecules:
            row["unique_molecule_count"] = int(summary["unique_molecule_count"])
        rows.append(row)
    return rows


def build_catalog(
    *,
    task: str,
    task_definition: str,
    records_path: str | Path,
    membership_path: str | Path,
    min_molecules: int = 5,
    top_endpoint_limit: int = 12,
    top_metadata_limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a retrieval-eligible context-level assay catalog."""

    import pyarrow.parquet as pq

    records_path = Path(records_path)
    membership_path = Path(membership_path)
    schema_names = set(pq.read_schema(records_path).names)
    required = {
        "canonical_record_id",
        "molecule_id",
        "canonical_endpoint_name",
        "canonical_assay_context",
    }
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"Records parquet is missing required columns: {missing}")
    optional = [
        name
        for name in (
            "canonical_species_context",
            "measurement_kind",
            *RAW_ASSAY_FIELDS,
        )
        if name in schema_names
    ]
    columns = sorted(required) + optional
    records = pq.read_table(records_path, columns=columns).to_pandas()
    member_ids = set(
        pq.read_table(membership_path, columns=["canonical_record_id"])
        .column(0)
        .to_pylist()
    )
    records = records[records["canonical_record_id"].isin(member_ids)].copy()
    for column in columns:
        if column not in {"canonical_record_id", "molecule_id"}:
            records[column] = records[column].map(_clean_text).astype("string")

    units = records.apply(
        lambda row: assay_unit(
            row["canonical_assay_context"], row["canonical_endpoint_name"]
        ),
        axis=1,
    )
    records["assay_unit"] = [item[0] for item in units]
    records["assay_unit_source"] = [item[1] for item in units]

    support = records.groupby("assay_unit")["molecule_id"].nunique()
    eligible_units = set(support[support >= min_molecules].index)
    eligible = records[records["assay_unit"].isin(eligible_units)]
    catalog: list[dict[str, Any]] = []
    for unit, frame in eligible.groupby("assay_unit", sort=True):
        source = str(frame["assay_unit_source"].iloc[0])
        row = {
            "catalog_version": CATALOG_VERSION,
            "task": task,
            "task_definition": task_definition,
            "assay_id": assay_id(task, str(unit)),
            "assay_context": str(unit),
            "assay_unit_source": source,
            "record_count": int(len(frame)),
            "unique_molecule_count": int(frame["molecule_id"].nunique()),
            "endpoints": _top_values(
                frame,
                "canonical_endpoint_name",
                limit=top_endpoint_limit,
                include_molecules=True,
            ),
            "species_contexts": _top_values(
                frame,
                "canonical_species_context",
                limit=top_metadata_limit,
            )
            if "canonical_species_context" in frame
            else [],
            "measurement_kinds": _top_values(
                frame, "measurement_kind", limit=top_metadata_limit
            )
            if "measurement_kind" in frame
            else [],
            "assay_descriptions": _collect_descriptions(
                frame, optional, limit=top_metadata_limit
            ),
        }
        catalog.append(row)
    catalog.sort(
        key=lambda row: (-row["unique_molecule_count"], row["assay_context"])
    )

    manifest = {
        "catalog_version": CATALOG_VERSION,
        "task": task,
        "task_definition": task_definition,
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "membership_path": str(membership_path.resolve()),
        "membership_sha256": sha256_file(membership_path),
        "assay_unit_definition": (
            "canonical_assay_context; endpoint_fallback::<canonical_endpoint_name> "
            "only when context is missing"
        ),
        "endpoint_role": "descriptive metadata; endpoints do not split assay units",
        "quality_gate": "none",
        "min_unique_molecules": min_molecules,
        "n_stage07_records": int(len(records)),
        "n_stage07_molecules": int(records["molecule_id"].nunique()),
        "n_all_assay_units": int(records["assay_unit"].nunique()),
        "n_eligible_assay_units": len(catalog),
        "n_context_units": sum(
            row["assay_unit_source"] == "canonical_assay_context" for row in catalog
        ),
        "n_endpoint_fallback_units": sum(
            row["assay_unit_source"] == "endpoint_fallback" for row in catalog
        ),
    }
    return catalog, manifest


def _collect_descriptions(frame: Any, columns: Iterable[str], *, limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for column in columns:
        if column not in RAW_ASSAY_FIELDS or column not in frame:
            continue
        for value, count in frame[column].value_counts(dropna=True).items():
            text = _clean_text(value)
            if text:
                counts[text] = counts.get(text, 0) + int(count)
    return [
        value
        for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    ]


def _build_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    catalog, manifest = build_catalog(
        task=args.task,
        task_definition=args.task_definition,
        records_path=args.records,
        membership_path=args.membership,
        min_molecules=args.min_molecules,
        top_endpoint_limit=args.top_endpoint_limit,
        top_metadata_limit=args.top_metadata_limit,
    )
    write_jsonl_atomic(output_dir / "assay_catalog.jsonl", catalog)
    write_json_atomic(output_dir / "catalog_manifest.json", manifest)
    print(
        f"[{args.task}] wrote {len(catalog):,} eligible assay units to "
        f"{output_dir / 'assay_catalog.jsonl'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("build-catalog")
    catalog.add_argument("--task", required=True)
    catalog.add_argument("--task-definition", required=True)
    catalog.add_argument("--records", required=True)
    catalog.add_argument("--membership", required=True)
    catalog.add_argument("--output-dir", required=True)
    catalog.add_argument("--min-molecules", type=int, default=5)
    catalog.add_argument("--top-endpoint-limit", type=int, default=12)
    catalog.add_argument("--top-metadata-limit", type=int, default=8)
    catalog.set_defaults(func=_build_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
