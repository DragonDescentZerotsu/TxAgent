"""Layer 1: common-schema ingestion and meaning-preserving cleaning."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles
from .contracts import CLEANING_STAGE_VERSION, NormalizedSourceProfile


_NULL_TEXT = {"", "nan", "none", "null", "na", "n/a", "-", "unspecified"}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUPERSCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")


def clean_text(value: Any) -> str | None:
    """Normalize typography and whitespace without changing semantic content."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("μ", "µ")
    text = re.sub(r"\s+", " ", text).strip()
    return None if text.casefold() in _NULL_TEXT else text or None


def clean_measurement_text(value: Any) -> str | None:
    """Clean source measurement/unit text while preserving exponent boundaries."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    text = _SUPERSCRIPT_RE.sub(
        lambda match: "^" + match.group(0).translate(_SUPERSCRIPTS), text
    )
    return clean_text(text)


def normalize_endpoint_name(value: Any) -> str | None:
    return clean_text(value)


def clean_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float, bool)):
        return value
    return clean_text(value)


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_inventory_hash(values: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_structure(
    row: Mapping[str, Any],
    profile: NormalizedSourceProfile,
    smiles_mapping: Mapping[str, str] | None,
) -> tuple[str | None, str]:
    if profile.structure_mode == "mapped":
        identifier = clean_text(row.get("global_identifier"))
        raw_smiles = clean_text(smiles_mapping.get(identifier)) if identifier and smiles_mapping else None
        if not raw_smiles:
            return None, "unresolved_mapped_structure"
    else:
        raw_smiles = clean_text(row.get(profile.smiles_field))
        if not raw_smiles:
            return None, "missing_structure"
    canonical = standardize_smiles(raw_smiles)[0]
    return (canonical, "resolved") if canonical else (None, "invalid_structure")


def clean_source_rows(
    rows: Iterable[Mapping[str, Any]],
    profile: NormalizedSourceProfile,
    *,
    smiles_mapping: Mapping[str, str] | None,
    source_sha256: str = "",
) -> list[dict[str, Any]]:
    """Return exactly one cleaned record for every supplied source row."""
    cleaned_records: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str, str], tuple[str | None, str]] = {}
    for source_row_number, raw_mapping in enumerate(rows, start=1):
        raw_source = {str(key): value for key, value in dict(raw_mapping).items()}
        raw = {key: clean_scalar(value) for key, value in raw_source.items()}
        endpoint_name = normalize_endpoint_name(
            profile.endpoint_constant or raw.get(profile.endpoint_field)
        )
        source_record_id = clean_text(raw.get(profile.record_id_field)) or str(source_row_number)
        cleaned_record_id = stable_id(
            "cleaned",
            profile.source_id,
            source_sha256,
            source_row_number,
            source_record_id,
        )
        structure_value = (
            clean_text(raw.get("global_identifier"))
            if profile.structure_mode == "mapped"
            else clean_text(raw.get(profile.smiles_field))
        )
        source_smiles = (
            clean_text(smiles_mapping.get(structure_value))
            if profile.structure_mode == "mapped"
            and structure_value
            and smiles_mapping
            else clean_text(raw.get(profile.smiles_field))
        )
        structure_key = (profile.structure_mode, structure_value or "")
        if structure_key not in structure_cache:
            structure_cache[structure_key] = resolve_structure(raw, profile, smiles_mapping)
        smiles, structure_status = structure_cache[structure_key]
        if profile.unit_constant:
            unit_text = clean_measurement_text(profile.unit_constant)
        elif profile.unit_field:
            unit_text = clean_measurement_text(raw_source.get(profile.unit_field))
        else:
            unit_text = None
        measurement_text = clean_measurement_text(
            raw_source.get(profile.measurement_field)
        )
        context = {
            field: clean_scalar(raw.get(field))
            for field in profile.context_fields
            if clean_scalar(raw.get(field)) is not None
        }
        molecule_name = next(
            (
                text
                for text in (clean_text(raw.get(field)) for field in profile.name_fields)
                if text
            ),
            None,
        )
        confidence = finite_float(raw.get(profile.confidence_field))
        source_payload = dict(raw)
        source_payload["_source_endpoint"] = clean_scalar(
            profile.endpoint_constant or raw.get(profile.endpoint_field)
        )
        source_payload["_source_measurement"] = measurement_text
        source_payload["_source_unit"] = unit_text
        cleaned_record = {
                "cleaning_version": CLEANING_STAGE_VERSION,
                "source_id": profile.source_id,
                "source_name": profile.source_name,
                "source_revision": profile.source_revision or None,
                "source_path": profile.source_path or None,
                "source_sha256": source_sha256 or None,
                "source_row_number": source_row_number,
                "source_record_id": source_record_id,
                "cleaned_record_id": cleaned_record_id,
                "endpoint_name": endpoint_name,
                "measurement_text": measurement_text,
                "unit_text": unit_text,
                "embedded_unit": bool(profile.embedded_unit),
                "source_smiles": source_smiles,
                "canonical_smiles": smiles,
                "structure_status": structure_status,
                "molecule_id": starling_molecule_id(smiles) if smiles else None,
                "molecule_name": molecule_name,
                "confidence": confidence,
                "support_text": clean_text(raw.get(profile.support_text_field)),
                "evidence_context_json": json.dumps(
                    context, ensure_ascii=False, sort_keys=True, default=str
                ),
                "source_payload_json": json.dumps(
                    source_payload, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
        # Context fields remain source-facing display values at top level. Canonical
        # variants are added only by the normalization stage.
        cleaned_record.update(context)
        cleaned_records.append(cleaned_record)
    return cleaned_records


def finite_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "clean_scalar",
    "clean_source_rows",
    "clean_text",
    "endpoint_inventory_hash",
    "file_sha256",
    "finite_float",
    "normalize_endpoint_name",
    "resolve_structure",
    "stable_id",
]
