"""Add stable assay contexts to residual HF bioavailability evidence.

The normalized v7 HF source has no physical ``assay_system`` column.  Leaving
all nondirect rows null collapses more than one hundred thousand literature
records into one endpoint fallback assay.  This overlay derives a small,
interpretable context vocabulary from genuine structured source fields without
changing endpoint, measurement, condition, molecule, or support-text content.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.tasks.bioavailability_ma.source_identity_review import (
    reviewed_hf_identity_exclusion_reason,
)


CONTEXT_VERSION = "bioavailability_nondirect_assay_context.v1"
SOURCE_GROUP = "Observed.nondirect_oral_bioavailability"
DEFAULT_INPUT = Path(
    "/data1/joseph/TxAgent/outputs/chembl_tool/tasks/bioavailability_ma/"
    "evidence_library/starling_normalized_v7/03_records/records.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "bioavailability_nondirect_assay_context_v1"
)

_MISSING = {"", "<na>", "n/a", "na", "nan", "none", "null", "unknown", "__unknown__"}


def _text(value: Any) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip().lower()
    return "" if text in _MISSING else text


def population_context(value: Any) -> str:
    """Map free-text population fields to a bounded experimental system."""

    text = _text(value)
    # Check animal systems before human demographic words such as ``male``.
    patterns = (
        (r"\brats?\b|sprague|wistar", "rat"),
        (r"\bmice\b|\bmouse\b|murine", "mouse"),
        (r"\bdogs?\b|beagle|canine", "dog"),
        (r"monkey|macaque|primate", "nonhuman primate"),
        (r"rabbits?", "rabbit"),
        (r"\bpigs?\b|porcine|swine", "pig"),
        (r"horses?|equine", "horse"),
        (r"pediatric|paediatric|children?|infants?|adolesc|neonat", "human pediatric"),
        (r"elderly|geriatric|\baged\b", "human elderly"),
        (r"healthy|volunteers?", "healthy human"),
        (
            r"humans?|patients?|subjects?|participants?|adults?|\bmen\b|women|"
            r"\bman\b|woman|males?|females?",
            "human clinical",
        ),
    )
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "other reported population" if text else "population not reported"


def exposure_context(value: Any) -> str:
    """Map oral exposure descriptions to a bounded dosage-system vocabulary."""

    text = _text(value)
    patterns = (
        (r"nano|nanocrystal", "nanoformulation"),
        (r"sedds|snedds|smedds|self.?emuls|lipid|liposom", "lipid or self-emulsifying formulation"),
        (r"tablets?", "tablet"),
        (r"capsules?", "capsule"),
        (r"suspension", "suspension"),
        (r"solution", "solution"),
        (r"gavage|intragastr", "gavage or intragastric dosing"),
        (r"food|diet|feed", "food or dietary exposure"),
        (r"formulation|emulsion|vehicle|dispersion|powder|pellet", "other oral formulation"),
    )
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "oral exposure not further specified"


def report_context(value: Any) -> str:
    report_type = _text(value)
    return {
        "relative_comparison": "relative comparison",
        "apparent": "apparent bioavailability",
    }.get(report_type, "qualitative or unspecified report")


def nondirect_assay_context(
    *,
    report_type: Any,
    species_or_population: Any,
    oral_exposure_mode: Any,
) -> str:
    """Build the LLM-readable context used as the physical assay unit."""

    return "; ".join(
        (
            "literature oral bioavailability",
            f"report={report_context(report_type)}",
            f"population={population_context(species_or_population)}",
            f"exposure={exposure_context(oral_exposure_mode)}",
        )
    )


def build_overlay(input_path: Path, output_dir: Path) -> dict[str, Any]:
    table = pq.read_table(input_path)
    required = {
        "canonical_assay_context",
        "canonical_bioavailability_report_type",
        "canonical_smiles",
        "group_id",
        "molecule_name",
        "oral_exposure_mode",
        "pmid",
        "retrieval_eligible",
        "source_id",
        "source_record_id",
        "species_or_population",
    }
    missing = sorted(required - set(table.schema.names))
    if missing:
        raise ValueError(f"Bioavailability records lack required columns: {missing}")

    fields = table.select(sorted(required)).to_pandas()
    exclusion_reasons = [
        (
            reviewed_hf_identity_exclusion_reason(record)
            if str(record.get("source_id") or "") == "hf_bioavailability"
            else ""
        )
        for record in fields.to_dict(orient="records")
    ]
    exclude = pa.array([bool(reason) for reason in exclusion_reasons])
    excluded = table.filter(exclude)
    keep = pa.compute.invert(exclude)
    table = table.filter(keep)
    fields = table.select(sorted(required)).to_pandas()
    target = fields["group_id"].eq(SOURCE_GROUP)
    if not target.any():
        raise ValueError(f"No records found for {SOURCE_GROUP}")
    wrong_source = set(fields.loc[target, "source_id"].dropna().astype(str)) - {
        "hf_bioavailability"
    }
    if wrong_source:
        raise ValueError(f"Unexpected source ids for {SOURCE_GROUP}: {sorted(wrong_source)}")

    missing_context = fields["canonical_assay_context"].map(_text).eq("")
    enrich = target & missing_context
    contexts = [
        nondirect_assay_context(
            report_type=report_type,
            species_or_population=population,
            oral_exposure_mode=exposure,
        )
        for report_type, population, exposure in zip(
            fields.loc[enrich, "canonical_bioavailability_report_type"],
            fields.loc[enrich, "species_or_population"],
            fields.loc[enrich, "oral_exposure_mode"],
            strict=True,
        )
    ]
    context_values = table.column("canonical_assay_context").to_pylist()
    for row_index, context in zip(fields.index[enrich], contexts, strict=True):
        context_values[int(row_index)] = context
    context_field = table.schema.field("canonical_assay_context")
    context_array = pa.array(context_values, type=context_field.type, from_pandas=True)
    context_index = table.schema.get_field_index("canonical_assay_context")
    enriched = table.set_column(context_index, context_field, context_array)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.parquet"
    exclusions_path = output_dir / "reviewed_source_exclusions.parquet"
    temporary_path = output_dir / ".records.parquet.tmp"
    pq.write_table(enriched, temporary_path, compression="zstd")
    os.replace(temporary_path, output_path)
    if excluded.num_rows:
        excluded = excluded.append_column(
            "reviewed_exclusion_reason",
            pa.array([reason for reason in exclusion_reasons if reason]),
        )
    pq.write_table(excluded, exclusions_path, compression="zstd")

    distribution = Counter(contexts)
    manifest = {
        "context_version": CONTEXT_VERSION,
        "source_group": SOURCE_GROUP,
        "input_records": str(input_path.resolve()),
        "input_records_sha256": sha256_file(input_path),
        "output_records": str(output_path.resolve()),
        "output_records_sha256": sha256_file(output_path),
        "n_input_rows": table.num_rows + excluded.num_rows,
        "n_rows": table.num_rows,
        "n_reviewed_identity_exclusions": excluded.num_rows,
        "reviewed_source_exclusions": str(exclusions_path.resolve()),
        "reviewed_source_exclusions_sha256": sha256_file(exclusions_path),
        "n_target_rows": int(target.sum()),
        "n_target_retrieval_eligible_rows": int(
            (target & fields["retrieval_eligible"].eq(True)).sum()  # noqa: E712
        ),
        "n_contexts_added": len(distribution),
        "n_rows_enriched": int(enrich.sum()),
        "n_target_rows_with_context_after": int(target.sum()),
        "context_fields": [
            "canonical_bioavailability_report_type",
            "species_or_population",
            "oral_exposure_mode",
        ],
        "context_distribution": dict(sorted(distribution.items())),
        "non_target_rows_modified": 0,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-records", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    manifest = build_overlay(Path(args.input_records), Path(args.output_dir))
    print(
        f"enriched {manifest['n_rows_enriched']:,} rows into "
        f"{manifest['n_contexts_added']:,} assay contexts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
