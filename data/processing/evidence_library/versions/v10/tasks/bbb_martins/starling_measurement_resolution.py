"""BBB V8 measurement extraction with a strict one-column, one-value contract."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.mapping_registry import (
    mapping_path,
)
from data.processing.evidence_library.versions.v10.measurement_routing import SourceRoutingRules
from data.processing.evidence_library.versions.v10.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.measurement_resolution_rules import (
    RULE_POLICY_VERSION,
    route_measurement,
)
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING,
    EndpointNormalizer,
)
from data.processing.evidence_library.versions.v10.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_VERSION,
    load_unit_vocabulary,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256


TASK_ASSET_ROOT = Path(__file__).resolve().parent / "data_processing"
PROMPT_VERSION = "bbb_measurement_resolution_prompt.v22_percentage_review"
MAPPING_VERSION = "bbb_martins_measurement_resolution.v15"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
ALLOW_REBATCH_UNATTEMPTED = True
REQUIRE_SOURCE_ROW_UID = True
REASONING_EFFORT = "low"

DEFAULT_MAPPING_PATH = mapping_path("measurement_resolution")
DEFAULT_BASE_MAPPING_PATH = None
SOURCE_UNIT_OVERRIDES = TASK_ASSET_ROOT / "measurement_resolution_v15/source_unit_overrides.v1.json"
TEMPLATE_DIR = PROMPT_ROOT / "measurement_resolution"
TEMPLATE_NAME = "bbb_v22.jinja"
DEFAULT_CLEANED_RECORDS = (
    evidence_library_root("bbb_martins", "v10") / "01_cleaned/records.parquet"
)
MAIN_UNIVERSE_CLEANED_RECORDS = (
    Path(__file__).resolve().parents[7]
    / "data/evidence_libraries/bbb_martins/v10_main_universe_v1/01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
COMPATIBILITY_LEDGER = Path(__file__).resolve().parents[2] / "level_mapping_compatibility.json"

SOURCE_MEASUREMENT_FIELDS = {
    "direct_bbb": ("quant_value", "quant_units"),
    "passive_permeability": ("metric_value", "metric_units"),
    "efflux_transport": ("quantitative_value", ""),
    "influx_transport": ("reported_result", ""),
}
SOURCE_IDS = tuple(SOURCE_MEASUREMENT_FIELDS)


def apply_reviewed_source_units(mapping_path: Path, output_path: Path) -> dict:
    """Replay the four user-reviewed source-unit corrections without model calls."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if mapping_path.resolve() == output_path.resolve():
        raise ValueError("The paid extraction must not be overwritten")
    decision_path = SOURCE_UNIT_OVERRIDES
    decision = json.loads(decision_path.read_text())
    manifest = json.loads(mapping_path.with_suffix(".manifest.json").read_text())
    if manifest["mapping_sha256"] != file_sha256(mapping_path):
        raise ValueError("Parent extraction hash mismatch")
    source_path = Path(manifest["cleaned_records_path"])
    if manifest["cleaned_records_sha256"] != file_sha256(source_path):
        raise ValueError("Source extraction input hash mismatch")
    expected = {row["source_row_uid"]: row["measurement"] for row in decision["rows"]}
    source = {row["source_row_uid"]: row for row in pq.read_table(
        source_path, columns=["source_row_uid", "cleaned_record_id", "unit_text"],
        filters=[("source_row_uid", "in", list(expected))]).to_pylist()}
    table = pq.read_table(mapping_path)
    rows = table.to_pylist()
    changed = []
    for row in rows:
        uid = row["source_row_uid"]
        if uid not in expected:
            continue
        entries = json.loads(row["measurements_json"])
        if (row["status"] != "ok" or len(entries) != 1 or
            entries[0] != {"measurement": expected[uid], "unit": decision["expected_model_unit"]} or
            source[uid]["unit_text"] != decision["expected_source_unit"] or
            source[uid]["cleaned_record_id"] != row["cleaned_record_id"]):
            raise ValueError(f"Reviewed source-unit input drift: {uid}")
        entries[0]["unit"] = source[uid]["unit_text"]
        row["measurements_json"] = json.dumps(entries, ensure_ascii=False)
        changed.append(uid)
    if len(changed) != len(expected) or set(changed) != set(expected):
        raise ValueError("Reviewed source-unit identity coverage mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output_path)
    manifest.update(mapping_path=str(output_path), mapping_sha256=file_sha256(output_path),
        source_unit_overrides={"path": str(decision_path), "sha256": file_sha256(decision_path),
            "parent_mapping_path": str(mapping_path), "parent_mapping_sha256": file_sha256(mapping_path),
            "source_row_uids": sorted(changed), "coefficients_statuses_and_raw_responses_unchanged": True})
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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
    """Validate the complete V10 delta and its independently hashed reuse input."""
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"BBB V8 extraction or manifest not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generation_version") == "stage1_measurement_resolution_replay.v1":
        mismatches = {}
        if manifest.get("task_id") != "bbb_martins":
            mismatches["task_id"] = manifest.get("task_id")
        if manifest.get("mapping_version") != MAPPING_VERSION:
            mismatches["mapping_version"] = manifest.get("mapping_version")
        if manifest.get("mapping_sha256") != file_sha256(path):
            mismatches["mapping_sha256"] = manifest.get("mapping_sha256")
        for name, source in (manifest.get("sources") or {}).items():
            source_path = Path(str(source.get("path") or ""))
            if not source_path.is_file() or source.get("sha256") != file_sha256(
                source_path
            ):
                mismatches[name] = "missing or hash-mismatched"
        if not all((manifest.get("validations") or {}).values()):
            mismatches["validations"] = manifest.get("validations")
        if mismatches:
            raise ValueError(f"BBB measurement replay provenance mismatch: {mismatches}")
        return
    override = manifest.get("source_unit_overrides")
    if override is not None:
        if (override.get("sha256") != file_sha256(SOURCE_UNIT_OVERRIDES) or
            override.get("parent_mapping_sha256") != file_sha256(Path(override["parent_mapping_path"]))):
            raise ValueError("Source-unit override decision or parent mapping hash mismatch")
    source_sha = file_sha256(DEFAULT_CLEANED_RECORDS)
    manifest_source = Path(str(manifest.get("cleaned_records_path") or ""))
    allowed_main_source = MAIN_UNIVERSE_CLEANED_RECORDS.resolve()
    protected_main_source = False
    cleaning_manifest_path = manifest_source.parent / "source_value_cleaning_manifest.json"
    source_inventory_path = (
        manifest_source.parents[1] / "00_source/source_inventory.json"
        if len(manifest_source.parents) > 1
        else Path("__missing_source_inventory__")
    )
    if cleaning_manifest_path.is_file() and source_inventory_path.is_file():
        cleaning_manifest = json.loads(cleaning_manifest_path.read_text(encoding="utf-8"))
        source_inventory = json.loads(source_inventory_path.read_text(encoding="utf-8"))
        authoritative = source_inventory.get("authoritative_source_seed") or {}
        protection = cleaning_manifest.get("gold_v1_voter_protection") or {}
        protected_main_source = (
            authoritative.get("version") == "main_source_universe.v1"
            and protection.get("all_protected_rows_present") is True
        )
    if (
        manifest_source.is_file()
        and (
            manifest_source.resolve() == allowed_main_source
            or protected_main_source
        )
        and manifest.get("cleaned_records_sha256") == file_sha256(manifest_source)
    ):
        source_sha = manifest["cleaned_records_sha256"]
    if source_sha != manifest.get("cleaned_records_sha256"):
        # A compatible rebuild does not rewrite the frozen extraction's input hash.
        spec = json.loads(COMPATIBILITY_LEDGER.read_text())["tasks"]["bbb_martins"]
        archived = Path(__file__).resolve().parents[7] / spec["previous_stage1_path"]
        if (manifest.get("cleaned_records_sha256") == spec["previous_stage1_sha256"]
                and archived.is_file()
                and file_sha256(archived) == spec["previous_stage1_sha256"]):
            source_sha = spec["previous_stage1_sha256"]
    expected = {
        "mapping_version": MAPPING_VERSION,
        "task_id": "bbb_martins",
        "mapping_sha256": file_sha256(path),
        "cleaned_records_sha256": source_sha,
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
    base = manifest.get("base_mapping") or {}
    base_path = Path(base.get("path") or "")
    if not base_path.is_file() or base.get("sha256") != file_sha256(base_path):
        mismatches["base_mapping"] = "Missing or mismatched reuse input hash"
    delta_models = (manifest.get("delta_inference") or {}).get("models")
    deepseek_delta = delta_models == ["deepseek/deepseek-v4-flash-0731"]
    high_effort_openai_delta = (
        delta_models == ["gpt-5.4-mini"]
        and (manifest.get("inference") or {}).get("reasoning_mode") == "high"
    )
    if not (deepseek_delta or high_effort_openai_delta):
        mismatches["delta_inference"] = (
            "Expected the reviewed DeepSeek/Baidu delta or gpt-5.4-mini high-effort successor"
        )
    for key in ("one_row_per_candidate", "unique_cleaned_record_ids",
                "only_ok_carries_measurements", "maximum_measurements_per_row"):
        if (manifest.get("validations") or {}).get(key) is not True:
            mismatches[key] = "Validation did not pass"
    import pyarrow.parquet as pq
    rows = pq.read_table(path, columns=["source_row_uid", "inference_source",
                        "served_provider", "inference_base_url"]).to_pylist()
    if any(not row["source_row_uid"] for row in rows):
        mismatches["source_row_uid"] = "Missing source identity"
    if deepseek_delta and any(row["inference_source"] == "delta_inference" and (
        row["served_provider"] != "Baidu" or
        row["inference_base_url"] != "https://openrouter.ai/api/v1"
    ) for row in rows):
        mismatches["delta_provider"] = "DeepSeek assignments must use Baidu on OpenRouter"
    if high_effort_openai_delta and any(
        row["inference_source"] == "delta_inference"
        and row["inference_base_url"] != "https://api.openai.com/v1"
        for row in rows
    ):
        mismatches["delta_provider"] = "gpt-5.4-mini assignments must use OpenAI"
    if mismatches:
        raise ValueError(f"v10 extraction provenance mismatch: {mismatches}")


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    _, unit_field = SOURCE_MEASUREMENT_FIELDS[source_id]
    fields = (
        ("endpoint_name", "measurement_text", "unit_text", "support_text")
        if unit_field
        else ("endpoint_name", "measurement_text", "support_text")
    )
    context = {
        'direct_bbb': ('assay_model', 'species', 'molecule_name', 'extra_details'),
        'passive_permeability': ('assay_system', 'qualifying_conditions', 'molecule_name', 'extra_details'),
        'efflux_transport': ('assay_type', 'biological_system', 'perturbation', 'transporter_identifier', 'qualifying_conditions', 'extra_details'),
        'influx_transport': ('assay_system', 'mediator_name', 'transport_mechanism', 'qualifying_conditions', 'extra_details'),
    }
    return (*fields, *context[source_id])


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
