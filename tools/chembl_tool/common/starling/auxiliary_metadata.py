"""Attach globally reconciled assay-transfer metadata to normalized records.

The lookup, validation, and coverage audit are task-agnostic.  A task binds its
own mapping path, mapping version, and the sources the mapping applies to; every
other source is recorded explicitly as ``not_applicable`` rather than silently
left blank.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_scalar,
    file_sha256,
)


AUXILIARY_ATTACHMENT_VERSION = "starling_auxiliary_attachment.v1"
OUTPUT_FIELDS = ("global_context", "global_species_context")


@dataclass(frozen=True)
class _Lookup:
    source_columns: tuple[str, ...]
    values: Mapping[tuple[Any, ...], str | None]


class AuxiliaryMetadataAttacher:
    """Validated, immutable lookup over one task's reconciled mapping sidecar."""

    def __init__(
        self,
        path: str | Path,
        *,
        mapping_version: str,
        applicable_sources: Sequence[str],
        null_like: Sequence[str],
    ):
        self.path = Path(path)
        self.applicable_sources = tuple(applicable_sources)
        self._null_like = frozenset(null_like)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if set(payload) != {"mapping_version", "sources"}:
            raise ValueError("invalid globally reconciled mapping root")
        if payload.get("mapping_version") != mapping_version:
            raise ValueError(
                "globally reconciled mapping version mismatch: "
                f"expected {mapping_version!r}, found {payload.get('mapping_version')!r}"
            )
        sources = payload.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != set(self.applicable_sources):
            raise ValueError("globally reconciled mapping source inventory mismatch")

        self.mapping_version = mapping_version
        self.sha256 = file_sha256(self.path)
        self._lookups: dict[tuple[str, str], _Lookup] = {}
        collision_counts: Counter[str] = Counter()
        raw_key_counts: Counter[str] = Counter()
        cleaned_key_counts: Counter[str] = Counter()

        for source_id in self.applicable_sources:
            source_outputs = sources[source_id]
            if not isinstance(source_outputs, Mapping) or set(source_outputs) != set(OUTPUT_FIELDS):
                raise ValueError(f"auxiliary output inventory mismatch for {source_id}")
            for output_field in OUTPUT_FIELDS:
                section = source_outputs[output_field]
                columns = section.get("source_columns")
                raw_mapping = section.get("mapping")
                if (
                    not isinstance(columns, list)
                    or not columns
                    or not all(isinstance(column, str) for column in columns)
                    or not isinstance(raw_mapping, Mapping)
                ):
                    raise ValueError(
                        f"invalid auxiliary mapping section for {source_id}/{output_field}"
                    )
                cleaned_mapping: dict[tuple[Any, ...], str | None] = {}
                for serialized_key, value in raw_mapping.items():
                    try:
                        decoded = json.loads(str(serialized_key))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid tuple key for {source_id}/{output_field}"
                        ) from exc
                    if not isinstance(decoded, list) or len(decoded) != len(columns):
                        raise ValueError(
                            f"tuple width mismatch for {source_id}/{output_field}"
                        )
                    if value is not None and not isinstance(value, str):
                        raise ValueError(
                            f"non-string auxiliary value for {source_id}/{output_field}"
                        )
                    key = self._tuple_key(tuple(decoded))
                    if key in cleaned_mapping and cleaned_mapping[key] != value:
                        raise ValueError(
                            f"conflicting cleaned tuple mapping for {source_id}/"
                            f"{output_field}: {key!r}"
                        )
                    if key in cleaned_mapping:
                        collision_counts[f"{source_id}/{output_field}"] += 1
                    cleaned_mapping[key] = value
                name = f"{source_id}/{output_field}"
                raw_key_counts[name] = len(raw_mapping)
                cleaned_key_counts[name] = len(cleaned_mapping)
                self._lookups[(source_id, output_field)] = _Lookup(
                    source_columns=tuple(columns),
                    values=cleaned_mapping,
                )

        self._manifest = {
            "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
            "mapping_version": self.mapping_version,
            "mapping_path": str(self.path),
            "mapping_sha256": self.sha256,
            "tuple_key_cleaning": {
                "whitespace_and_null_cleaning": "clean_scalar",
                "null_like_values": sorted(self._null_like),
            },
            "output_fields": list(OUTPUT_FIELDS),
            "applicable_sources": list(self.applicable_sources),
            "sections": {
                f"{source}/{field}": {
                    "source_columns": list(self._lookups[(source, field)].source_columns),
                    "raw_tuple_keys": raw_key_counts[f"{source}/{field}"],
                    "cleaned_tuple_keys": cleaned_key_counts[f"{source}/{field}"],
                    "equivalent_cleaned_key_collisions": collision_counts[
                        f"{source}/{field}"
                    ],
                    "conflicting_cleaned_key_collisions": 0,
                }
                for source in self.applicable_sources
                for field in OUTPUT_FIELDS
            },
        }

    def _tuple_key(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        cleaned: list[Any] = []
        for value in values:
            item = clean_scalar(value)
            # The frozen mapping builder collapses source null sentinels before
            # it serializes tuple keys. Apply the identical null-equivalence
            # contract at attachment time; otherwise a cleaned source literal
            # such as ``unknown`` cannot join its authoritative ``null`` tuple.
            if isinstance(item, str) and item.casefold() in self._null_like:
                item = None
            cleaned.append(item)
        return tuple(cleaned)

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        source_id = str(record.get("source_id") or "")
        if source_id not in self.applicable_sources:
            return {
                "global_context": None,
                "global_species_context": None,
                "auxiliary_mapping_status": "not_applicable",
                "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
            }
        output: dict[str, Any] = {}
        for output_field in OUTPUT_FIELDS:
            lookup = self._lookups[(source_id, output_field)]
            key = self._tuple_key(
                tuple(record.get(column) for column in lookup.source_columns)
            )
            if key not in lookup.values:
                raise ValueError(
                    f"globally reconciled mapping lacks {source_id}/{output_field} "
                    f"tuple {key!r}"
                )
            output[output_field] = lookup.values[key]
        output.update(
            {
                "auxiliary_mapping_status": "mapped",
                "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
            }
        )
        return output

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest, sort_keys=True))

    def coverage_audit(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            source_id = str(record.get("source_id") or "")
            status = str(record.get("auxiliary_mapping_status") or "missing")
            source_counts[source_id][status] += 1
            if source_id in self.applicable_sources and status != "mapped":
                raise ValueError(
                    f"applicable source record lacks mapped auxiliary metadata: {source_id}"
                )
            if source_id not in self.applicable_sources and status != "not_applicable":
                raise ValueError(
                    f"inapplicable source record has unexpected auxiliary status: {source_id}"
                )
        return {
            "records": len(records),
            "source_status_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(source_counts.items())
            },
            "validations": {
                "all_applicable_records_mapped": True,
                "inapplicable_sources_explicit": True,
            },
        }


__all__ = [
    "AUXILIARY_ATTACHMENT_VERSION",
    "OUTPUT_FIELDS",
    "AuxiliaryMetadataAttacher",
]
