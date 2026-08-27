"""Endpoint-aware measurement resolution for the seven ClinTox sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.starling.measurement_routing import SourceRoutingRules


TASK_ROOT = Path(__file__).resolve().parent
TEMPLATE = TASK_ROOT / "measurement_resolution_templates/measurement_resolution_v1.jinja"
PROMPT_VERSION = "clintox_measurement_resolution_prompt.v9"
MAPPING_VERSION = "clintox_measurement_resolution.v1"
BATCH_SIZE = 10
SOURCE_IDS = (
    "human_clinical_toxicity",
    "nonclinical_in_vivo_toxicity",
    "organ_specific_toxicity",
    "genotoxicity_carcinogenicity",
    "cellular_stress",
    "general_cytotoxicity",
    "off_target_ddi_exposure",
)
_UNIT_SOURCES = frozenset(
    {
        "nonclinical_in_vivo_toxicity",
        "general_cytotoxicity",
        "off_target_ddi_exposure",
    }
)

DEFAULT_CLEANED_RECORDS = Path(
    "outputs/chembl_tool/tasks/clintox/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v1/measurement_resolution.parquet"
)
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"

CLINTOX_INSTRUCTIONS = """## ClinTox-specific endpoint semantics

Extract only quantities that measure the supplied canonical endpoint. The seven endpoint identities are source-defined: human toxicity category, nonclinical evidence type, organ toxicity endpoint, genotoxicity endpoint, cellular stress endpoint, cytotoxicity endpoint type, and off-target result metric. Named targets, organs, assay types, evidence categories, and study context describe the experiment; they are not substitute endpoints.

Use `ok` for an absolute value or an endpoint-defined ratio/readout whose reference is stable inside that endpoint. Use `relative` for a value defined only against an experiment-specific control, comparator, baseline, genotype, or another compound. Use `unavailable` when the text is qualitative or the number measures dose, time, sample size, target identity, or another endpoint. Use `unsure` when the selected value, unit, or reference cannot be resolved without guessing. Never invent a unit from domain expectations.

`measurement_text` is the value selector. Return each clearly resolved selected quantity and do not add support-only quantities. Preserve explicit scientific-notation scale in the unit as `10^-n` and numeric exponents with `^`.

Apply these rules literally:
- A stated treatment-group point remains `ok` when a control/comparator value is also shown. Return the treatment value only; comparison by itself does not make an absolute point `relative` or `unsure`.
- `relative` requires a selected numeric change, ratio, or fold defined only against an experiment-specific reference. A qualitative phrase such as "lower incidence" with no numeric result is `unavailable`.
- Counts and rates that measure the endpoint are valid values with complete units such as `patients`, `cases`, `%`, or `per 10^6 person-years`.
- A dose, concentration, duration, threshold dose, sample size, p-value, or other experimental condition is `unavailable` even when it is the only number near an observed effect.
- If several values reuse the same unit for different outcomes, time points, tissues, or arms and bare output pairs would lose those identities, return `unsure`. If the canonical endpoint clearly selects one value, return only that value. Several clearly selected quantities with distinguishable complete units may all be returned.
- A bound, interval without a selected point, or endpoint-defined range is `unsure`; do not return its boundary as a point. A central estimate followed by spread in parentheses or with `±` is `ok` and returns the central estimate.
- With `missing_endpoint`, do not infer an endpoint from support text; return `unavailable`.
- A complete unit must retain any explicit power-of-ten scale and named endpoint-defined reference.
"""

SOURCE_INSTRUCTIONS = {
    "genotoxicity_carcinogenicity": """## Source-specific rules

The selected measurement is often a categorical call. If `measurement_text` is qualitative or categorical, return `unavailable` even when `support_text` contains numerical assay results. Do not replace the selected call with a supporting number.
""",
    "human_clinical_toxicity": """## Source-specific rules

Extract clearly named toxicity rates, patient counts, event counts, and severity-specific death counts. Multiple distinctly named toxicity outcomes from the same study may all be returned. Ignore co-reported sample sizes, doses, durations, laboratory stopping thresholds, and exposure concentrations; a serum drug concentration never measures an organ-toxicity endpoint. When the endpoint names one organ toxicity, return only that outcome and exclude numbers for other co-reported outcomes. A reported prevalence stated as "up to N%" may be returned as N%. If numbers combine different studies, reporting systems, severity grades, or time windows and no single context is selected, return `unsure`. Odds ratios against named comparators are `relative` and return no measurement pairs.
""",
    "nonclinical_in_vivo_toxicity": """## Source-specific rules

An explicit change, delay, or percent increase relative to controls is `relative`. A treatment-group absolute point paired with a control point is `ok`; return only the treatment point. A count of named adverse events is a valid endpoint measurement. Any unit described as inferred, assumed, or typical is unresolved, so return `unsure`. Exposure concentrations that merely caused an effect are `unavailable`, even when `measurement_text` contains only that concentration. For mortality or survival, return `unsure` when several time points are listed or when a percentage is paired with its count fraction; a single exact survival percentage remains `ok`. Approximate bounds such as "under N" or "about N" are `unsure`. Several co-reported biological outcomes under the broad `animal_toxicity_finding` endpoint are `unsure` when their identities would be lost.
""",
    "off_target_ddi_exposure": """## Source-specific rules

When the endpoint name itself is a ratio between named targets, its reference is stable: return the selected ratio as `ok` with unit `fold`. For relative potency against a named reference compound, preserve the power-of-ten scale and reference name in the unit. In contrast, clearance change, remaining activity, or a fold decrease relative to an experimental baseline is `relative`. A comparison-only result remains `relative` even when the only numerical value shown is the reference boundary; return no measurement pair in that case. For a central estimate followed by a parenthetical coefficient of variation, return the central estimate and its supplied physical unit, not the coefficient of variation. Incidence or threshold-exceedance counts with a denominator are `unsure` when the context contains a very small cohort or several competing threshold categories.
""",
    "organ_specific_toxicity": """## Source-specific rules

An explicit standalone count such as `N cases`, `N reports`, `N affected`, or `N patients` is `ok`; preserve the count noun from `measurement_text` as the unit and ignore co-reported doses. A zero count paired with a denominator, or nonzero fractions across groups, is `unsure`. Treat multiple arms, comparator incidences, or several co-reported outcomes as `unsure` when bare value-unit pairs cannot preserve their identities. A numeric increase or decrease against control is `relative`.
""",
}


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field="unit_text" if source_id in _UNIT_SOURCES else "",
        )
        for source_id in SOURCE_IDS
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    fields = ["endpoint_name", "measurement_text"]
    if source_id in _UNIT_SOURCES:
        fields.append("unit_text")
    fields.append("support_text")
    return tuple(fields)


def canonical_endpoint_record(record: Mapping[str, Any]) -> str:
    endpoint = record.get("canonical_endpoint_name") or record.get("endpoint_name")
    return str(endpoint or "missing_endpoint")


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    endpoint = str(endpoint_name or "").strip()
    return endpoint or "missing_endpoint"


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    if batch_size != BATCH_SIZE:
        raise ValueError(f"ClinTox measurement batches are frozen at {BATCH_SIZE}")
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    rendered = environment.get_template(TEMPLATE.name).render(
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id in _UNIT_SOURCES,
        endpoint_profiles=endpoint_profiles,
        task_instructions=CLINTOX_INSTRUCTIONS.rstrip(),
        source_instructions=SOURCE_INSTRUCTIONS.get(source_id, "").rstrip(),
    )
    return rendered


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE),
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        "task_instructions_sha256": hashlib.sha256(
            CLINTOX_INSTRUCTIONS.encode()
        ).hexdigest(),
        "source_instructions_sha256": {
            source_id: hashlib.sha256(
                SOURCE_INSTRUCTIONS.get(source_id, "").encode()
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
        "batch_size": batch_size,
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id).encode()
            ).hexdigest()
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
