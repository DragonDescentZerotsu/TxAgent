"""Endpoint-aware measurement resolution for Skin's numeric indirect sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.starling.measurement_routing import SourceRoutingRules


TASK_ROOT = Path(__file__).resolve().parent
BBB_TEMPLATE = (
    TASK_ROOT.parent
    / "bbb_martins/measurement_resolution_templates/measurement_resolution_v5.jinja"
)

PROMPT_VERSION = "skin_reaction_measurement_resolution_prompt.v7"
MAPPING_VERSION = "skin_reaction_measurement_resolution.v1"
BATCH_SIZE = 10
_ROUTED_SOURCE_IDS = (
    "direct_skin_reaction",
    "sensitization_aop",
    "skin_exposure",
    "phototoxicity_irritation_local_damage",
)
SOURCE_IDS = _ROUTED_SOURCE_IDS
_EMBEDDED_UNIT_SOURCES = frozenset(
    {"direct_skin_reaction", "phototoxicity_irritation_local_damage"}
)

DEFAULT_CLEANED_RECORDS = Path(
    "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v1/measurement_resolution.parquet"
)
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"

SKIN_INSTRUCTIONS = """## Skin-specific endpoint semantics

These are indirect numeric Skin assays. A named readout defines its reference and is `ok`: LLNA/LLN stimulation index, EC3/pEC3, RFI/cMFI, Imax, peptide/GSH depletion, sensitization incidence or subject count, biomarker expression fold/log2-fold change, a named expression ratio, and a skin-exposure fraction defined against applied dose or recovered material. Do not call such a readout `relative` merely because its assay uses a control; a selected endpoint value also remains `ok` when support gives a comparator.

`measurement_text` is the primary value selector. Return every clearly resolved quantity it selects; Stage 02 will explode multiple returned quantities into separate rows. Do not suppress a resolved quantity merely because its unit may be incompatible with the canonical endpoint; the exact unit map decides mapping or exclusion. Use support only to identify each selected value's unit, readout, or reference semantics; never add a support-only number or an alternate representation. If support explicitly defines a selected value as a ratio, fold-change, enhancement factor, or percent against an experiment-specific control, it is `relative` unless the canonical endpoint or named readout defines that reference. A separate comparator in support does not make a selected absolute point relative. With `missing_endpoint`, an experimental comparator is never endpoint-defined. ALN cell proliferation reported as fold induction compared with vehicle is `relative`; `cell proliferation` does not define that vehicle reference. A percent of an exchangeable ion pool is definitional for flux; "control period" describes the study period, not the denominator.

Named readout units remain absolute-or-definitional: KeratinoSens Imax uses `fold`, LLNA pEC3 may use `log M`, and an endpoint explicitly labelled cMFI or RFI supplies that dimensionless readout unit.

The number must still measure the canonical endpoint. EC3 under a stimulation-index endpoint, stimulation index under broad sensitization, incidence under an erythema/edema score, organ distribution or excretion under permeation/absorption, and photobinding under penetration are `unavailable`. A pEC50 paired with explicit `log M` is inconsistent and `unsure`. Otherwise use `relative` only for an experiment-specific reference outside the canonical endpoint/readout. Category codes such as GHS category 1 are `unavailable`.

In output units, write an explicit scale as `10^-n` without a leading `x` or `×`, and write numeric exponents with `^`.
"""


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field="" if source_id in _EMBEDDED_UNIT_SOURCES else "unit_text",
        )
        for source_id in _ROUTED_SOURCE_IDS
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    if source_id in _EMBEDDED_UNIT_SOURCES:
        return ("endpoint_name", "measurement_text", "support_text")
    return ("endpoint_name", "measurement_text", "unit_text", "support_text")


def canonical_endpoint_record(record: Mapping[str, Any]) -> str:
    """Read the final endpoint materialized by the authoritative clean stage."""
    endpoint = record.get("canonical_endpoint_name")
    if not endpoint:
        raise ValueError(
            "Skin measurement resolution requires Stage-01 canonical_endpoint_name"
        )
    return str(endpoint)


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    """Reject the legacy partial-row fallback for Skin endpoint identity."""
    del source_id, endpoint_name
    raise ValueError("Skin endpoint identity must come from Stage-01 canonical_endpoint_name")


def _base_render(source_id: str, endpoint_profiles: tuple[str, ...]) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(BBB_TEMPLATE.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    rendered = environment.get_template(BBB_TEMPLATE.name).render(
        batch_size=BATCH_SIZE,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id not in _EMBEDDED_UNIT_SOURCES,
        endpoint_profiles=endpoint_profiles,
    )
    rendered = rendered.replace(
        "You extract measurement values from the BBB dataset.",
        "You extract measurement values from the Skin Reaction dataset.",
        1,
    )
    return rendered.replace("## Result", f"{SKIN_INSTRUCTIONS}\n## Result", 1)


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    if batch_size != BATCH_SIZE:
        raise ValueError(f"Skin measurement batches are frozen at {BATCH_SIZE}")
    return _base_render(source_id, endpoint_profiles)


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(BBB_TEMPLATE),
        "template_sha256": hashlib.sha256(BBB_TEMPLATE.read_bytes()).hexdigest(),
        "task_instructions_sha256": hashlib.sha256(SKIN_INSTRUCTIONS.encode()).hexdigest(),
        "batch_size": batch_size,
        "rendered_sha256": {
            source_id: hashlib.sha256(render_prompt(source_id).encode()).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "canonical_endpoint_name",
    "canonical_endpoint_record",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "source_routing_rules",
]
