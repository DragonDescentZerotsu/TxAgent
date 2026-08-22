"""BBB measurement-resolution routing rules and frozen-artifact configuration.

**BBB declares no declarative non-scalar rules, by measurement.**  A cleaned BBB
row is rejected only when its measurement column carries no digit.

Four candidate declarative rules were built and audited against the real corpus,
and all four were rejected.  Each was a column stating that a row's evidence was
qualitative or comparative, and each still covered rows carrying a hard absolute
measurement.  Counting rows whose support text names an absolute PK parameter
(Km, Ki, IC50, EC50, Papp, Vmax, clearance) with no relative marker anywhere:

=======================================================  =========  ===========  =======
candidate rule                                            ruled out  genuine abs  FP rate
=======================================================  =========  ===========  =======
``efflux.evidence_type == qualitative_transporter_claim``      2,823          794    28.1%
``influx.evidence_basis == comparative``                       6,033          142     2.4%
``influx.evidence_basis == inhibition or competition``         4,308          267     6.2%
``influx.evidence_basis == qualitative substrate claim``      18,828          291     1.5%
=======================================================  =========  ===========  =======

The failure is systematic rather than a matter of tuning: these columns describe
the *conclusion the study drew* -- substrate status, inhibitor status -- not whether
a quantity was measured.  A paper can report ``IC50 = 0.51 uM`` and still conclude
only that the compound is an inhibitor, so qualitative evidence and an absolute
number coexist freely.  Because the reject bucket discards a row with no extraction
pass behind it to catch the mistake, a 1.5% floor is already too high.

The same reasoning excluded the two obvious outcome labels:
``direct_bbb.bbb_permeability_label`` and
``passive_permeability.passive_bbb_interpretation`` say what a study concluded, and
a permeable compound still has a reported Papp.  Both remain inputs to the
controlled categorical encoder, which is untouched by routing.  Influx remains
positive-only evidence rather than a fabricated binary measurement: its digit-free
qualitative rows reject mechanically, while numeric prose still gets endpoint-aware
resolution.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.starling.measurement_routing import SourceRoutingRules
from tools.chembl_tool.common.starling.normalization.measurements import (
    canonicalize_endpoint,
)
from tools.chembl_tool.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING,
    EndpointNormalizer,
)


TASK_ROOT = Path(__file__).resolve().parent

PROMPT_VERSION = "bbb_measurement_resolution_prompt.v11"
MAPPING_VERSION = "bbb_martins_measurement_resolution.v1"

DEFAULT_MAPPING_PATH = (
    TASK_ROOT
    / "data_processing/measurement_resolution_v1/measurement_resolution.parquet"
)
TEMPLATE_DIR = TASK_ROOT / "measurement_resolution_templates"
TEMPLATE_NAME = "measurement_resolution_v5.jinja"
DEFAULT_CLEANED_RECORDS = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = Path(
    "outputs/chembl_tool/tasks/bbb_martins/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"

# Ten rows amortize the instruction block while limiting cross-row bleed.
BATCH_SIZE = 10

# efflux and influx have no unit column at all, so no bare-number rule can fire for
# them: a number with no unit cannot be given one mechanically.
_SOURCES_WITHOUT_A_UNIT_COLUMN = frozenset({"efflux_transport", "influx_transport"})

SOURCE_IDS = (
    "direct_bbb",
    "passive_permeability",
    "efflux_transport",
    "influx_transport",
)


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    """Build the source-column boundary for decimal bypass versus extraction."""
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field=(
                "" if source_id in _SOURCES_WITHOUT_A_UNIT_COLUMN else "unit_text"
            ),
        )
        for source_id in SOURCE_IDS
    }


#: Fields shown to the model, per source.  A source with no unit column is never
#: told about ``unit_text``: naming a field that is always null invites the model to
#: invent one.
def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    fields = ["endpoint_name", "measurement_text"]
    if source_id not in _SOURCES_WITHOUT_A_UNIT_COLUMN:
        fields.append("unit_text")
    fields.append("support_text")
    return tuple(fields)


@lru_cache(maxsize=1)
def _endpoint_normalizer() -> EndpointNormalizer:
    return EndpointNormalizer(DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING)


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    decision = _endpoint_normalizer().decision(source_id, str(endpoint_name or ""))
    return canonicalize_endpoint(decision.spacing_and_spelling_endpoint)


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    """Render the reviewed prompt for one source.

    The template is the editable source of truth; this only supplies the per-source
    facts.  ``StrictUndefined`` means a typo in a variable name raises at render
    time rather than silently emitting an empty string into a paid prompt.
    """
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return _environment().get_template(TEMPLATE_NAME).render(
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id not in _SOURCES_WITHOUT_A_UNIT_COLUMN,
        endpoint_profiles=endpoint_profiles,
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    """Provenance for the stage manifest: version plus template and render digests.

    The template digest is what changes when the file is edited; the per-source
    render digests are what actually reached the model.
    """
    template_path = TEMPLATE_DIR / TEMPLATE_NAME
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "batch_size": batch_size,
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "TEMPLATE_DIR",
    "TEMPLATE_NAME",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "source_routing_rules",
]
