"""Oral-Bioavailability row-level reference-semantics policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_COMPARATOR,
    REFERENCE_SCOPE_NOT_APPLICABLE,
    REFERENCE_SCOPE_UNKNOWN,
    ReferenceAssignment,
    ReferenceSemanticsConfig,
    ReferenceSourcePromptSpec,
    deterministic_default_assignment,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    normalize_bioavailability_report_type,
)


TASK_ROOT = Path(__file__).resolve().parent
PROMPT_VERSION = "bioavailability_reference_semantics_prompt.v2"
MAPPING_VERSION = "bioavailability_reference_semantics.v2"
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/reference_semantics_v2/reference_semantics.parquet"
)
DEFAULT_RECORDS_PATH = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7/02_canonicalized/records.parquet"
)

# These permeability records were previously classified as non-scalar because
# U+2010 scientific-minus notation was unrecognized or because the explicit
# 10^-6 factor appeared only in support text. Their reviewed endpoint is an
# absolute permeability coefficient. Invalid-structure rows remain
# assay-transfer-ineligible independently in Stage 04.
_REVIEWED_ABSOLUTE_PERMEABILITY_RECORDS = frozenset(
    {
        "8e6b73794e5f9ff8200230a467f4547c51de10184601a7aac1c0476303927ef3",
        "9229fd4b814e6bc0219efecb6a51085b69b0cdb8dfb7ffe36cb166d66b2b2bcc",
        "4f89f2058ad71b81ad61d897e52c6523764666ea09d93a12bb8bf20d07e3e479",
        "c460a5a57a630070ff2cd48849973c86107580b3649a87a3ddc6a487ff9a0201",
        "a46a6adac4d48ff615540e93eefd7faa41a87e0a8b611c42276fb9967ad3b71e",
        "259c86ac22b749bb0ff00cf809ddd852b2290fc89a9cb2a69bb094340c7ebba6",
        "144c503d0a309147db1ae892b8b9bfc6bf1d4a749dca876ef5e751ca7a0c8f83",
        "5c895959a23bdc5887850a7e22de8b621d928a6e8701c8488eb7c5ed2490993e",
        "1a5745b8d1cb2bf23619856a01ca34e3ec66b580e1fb215a9952ad4bc894a88e",
        "9f47a273c85807322ab1e2712a939241998e68a2ee466f321351a597e9f4826c",
        "76ff578e7b15d366a9e741857b4bc645f36f4f4c313de7410b5d559fd76057c5",
        "7317c21c05b793859bb1fbdaacb9dbe1bde00b34acb9aebea5c5ebbfcdf8cc3d",
        "2ea30caf5e8193682646e5e70215ebaf6468919134975c86ae5e34aad6edd63f",
        "ff5afd1a6485163bda6385d7f5873ac15d70305725b7f741a8e71b15cc43f670",
        "9675e4d747292270bf53b29ea47da7117170155d9c7040fca7d05cac0f347b01",
    }
)


def deterministic_assignment(record: Mapping[str, Any]) -> ReferenceAssignment | None:
    common = deterministic_default_assignment(record, output_basis=False)
    if common is not None:
        return common
    if str(record.get("cleaned_record_id") or "") in (
        _REVIEWED_ABSOLUTE_PERMEABILITY_RECORDS
    ):
        return ReferenceAssignment(
            REFERENCE_SCOPE_ABSOLUTE,
            None,
            "reviewed_absolute_permeability_scale_repair",
        )
    if str(record.get("source_id") or "") != "hf_bioavailability":
        return None
    report_type = normalize_bioavailability_report_type(
        record.get("bioavailability_report_type")
    )
    if report_type in {"absolute", "systemic_availability", "extent_f"}:
        return ReferenceAssignment(
            REFERENCE_SCOPE_ABSOLUTE, None, "authoritative_hf_report_type"
        )
    if report_type in {"relative_comparison", "apparent"}:
        return ReferenceAssignment(
            REFERENCE_SCOPE_COMPARATOR, None, "authoritative_hf_report_type"
        )
    return ReferenceAssignment(
        REFERENCE_SCOPE_UNKNOWN, None, "unresolved_hf_report_type"
    )


REFERENCE_SEMANTICS_CONFIG = ReferenceSemanticsConfig(
    task_id="bioavailability_ma",
    canonical_records_path=DEFAULT_RECORDS_PATH,
    mapping_path=DEFAULT_MAPPING_PATH,
    prompt_registry_path=TASK_ROOT / "data_processing/reference_semantics_prompts.json",
    prompt_version=PROMPT_VERSION,
    output_basis=False,
    source_specs={
        "oral_exposure": ReferenceSourcePromptSpec(
            "oral_exposure",
            (
                "statistic_type",
                "oral_dose",
                "study_context",
                "comparator_exposure",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fa": ReferenceSourcePromptSpec(
            "fa",
            (
                "assay_system",
                "condition_medium",
                "biological_context",
                "formulation_or_solid_form",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fg": ReferenceSourcePromptSpec(
            "fg",
            (
                "transporter_or_enzyme",
                "substrate_status",
                "assay_system",
                "intestinal_site",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        "fh": ReferenceSourcePromptSpec(
            "fh",
            (
                "assay_system",
                "species",
                "molecular_form",
                "enzyme_or_pathway",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        # Deterministic HF handling means this spec is a defensive invariant.
        "hf_bioavailability": ReferenceSourcePromptSpec(
            "hf_bioavailability", ("bioavailability_report_type", "comparator")
        ),
    },
    deterministic_assignment=deterministic_assignment,
    eligible_scopes=(REFERENCE_SCOPE_ABSOLUTE, REFERENCE_SCOPE_NOT_APPLICABLE),
    batch_size=50,
    prompt_fields=("measurement_text", "support_text"),
    labels_only_output=True,
    batch_across_sources=False,
)


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_SEMANTICS_CONFIG",
    "deterministic_assignment",
]
