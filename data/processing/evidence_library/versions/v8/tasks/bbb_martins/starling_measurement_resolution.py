"""BBB V8 measurement extraction with a strict one-column, one-value contract."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.versions.v8.measurement_routing import SourceRoutingRules
from data.processing.evidence_library.versions.v8.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.measurement_resolution_rules import (
    RULE_POLICY_VERSION,
    route_measurement,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING,
    EndpointNormalizer,
)
from data.processing.evidence_library.versions.v8.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_VERSION,
    load_unit_vocabulary,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256


TASK_ASSET_ROOT = Path(__file__).resolve().parent / "data_processing"
PROMPT_VERSION = "bbb_measurement_resolution_prompt.v21"
MAPPING_VERSION = "bbb_martins_measurement_resolution.v15"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
ALLOW_REBATCH_UNATTEMPTED = True
REASONING_EFFORT = "low"

DEFAULT_MAPPING_PATH = (
    TASK_ASSET_ROOT / "measurement_resolution_v15/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
TEMPLATE_DIR = PROMPT_ROOT / "measurement_resolution"
TEMPLATE_NAME = "bbb_v21.jinja"
DEFAULT_CLEANED_RECORDS = (
    evidence_library_root("bbb_martins", "v8") / "01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"

SOURCE_MEASUREMENT_FIELDS = {
    "direct_bbb": ("quant_value", "quant_units"),
    "passive_permeability": ("metric_value", "metric_units"),
    "efflux_transport": ("quantitative_value", ""),
    "influx_transport": ("reported_result", ""),
}
SOURCE_IDS = tuple(SOURCE_MEASUREMENT_FIELDS)


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    """Route the exact normalized projection of each declared source field."""
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            measurement_field="measurement_text",
            unit_field="unit_text" if unit_field else "",
        )
        for source_id, (_, unit_field) in SOURCE_MEASUREMENT_FIELDS.items()
    }


def validate_mapping_provenance(mapping_path: str | Path) -> None:
    """Reject a BBB V8 extraction that was not produced by the frozen run."""
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"BBB V8 extraction or manifest not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "mapping_version": MAPPING_VERSION,
        "model": "gpt-5.4-mini",
        "api_base_url": "https://api.openai.com/v1",
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    prompt = manifest.get("prompt") or {}
    if prompt.get("prompt_version") != PROMPT_VERSION:
        mismatches["prompt_version"] = {
            "expected": PROMPT_VERSION,
            "found": prompt.get("prompt_version"),
        }
    if manifest.get("base_mapping") is not None:
        mismatches["base_mapping"] = {
            "expected": None,
            "found": manifest.get("base_mapping"),
        }
    if not (manifest.get("validations") or {}).get("maximum_measurements_per_row"):
        mismatches["maximum_measurements_per_row"] = {
            "expected": True,
            "found": (manifest.get("validations") or {}).get(
                "maximum_measurements_per_row"
            ),
        }
    if mismatches:
        raise ValueError(f"v8 extraction provenance mismatch: {mismatches}")


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    _, unit_field = SOURCE_MEASUREMENT_FIELDS[source_id]
    return (
        ("endpoint_name", "measurement_text", "unit_text", "support_text")
        if unit_field
        else ("endpoint_name", "measurement_text", "support_text")
    )


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
    if source_id not in SOURCE_MEASUREMENT_FIELDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    declared_measurement_field, declared_unit_field = SOURCE_MEASUREMENT_FIELDS[source_id]
    return _environment().get_template(TEMPLATE_NAME).render(
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        measurement_field="measurement_text",
        unit_field="unit_text" if declared_unit_field else None,
        declared_measurement_field=declared_measurement_field,
        declared_unit_field=declared_unit_field or None,
        endpoint_profiles=(),
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    template_path = TEMPLATE_DIR / TEMPLATE_NAME
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "routing_rule_policy_version": RULE_POLICY_VERSION,
        "unit_vocabulary": {
            "version": VOCABULARY_VERSION,
            "path": str(DEFAULT_VOCABULARY_PATH),
            "sha256": file_sha256(DEFAULT_VOCABULARY_PATH),
            "units": len(load_unit_vocabulary()),
        },
        "source_measurement_fields": {
            source_id: {
                "measurement_field": fields[0],
                "unit_field": fields[1] or None,
            }
            for source_id, fields in SOURCE_MEASUREMENT_FIELDS.items()
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_BASE_MAPPING_PATH",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "MAPPING_VERSION",
    "MAX_MEASUREMENTS_PER_ROW",
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "SOURCE_MEASUREMENT_FIELDS",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_routing_rules",
    "validate_mapping_provenance",
]
