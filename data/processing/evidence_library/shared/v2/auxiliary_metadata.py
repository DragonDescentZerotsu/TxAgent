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

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_measurement_text,
    clean_scalar,
    file_sha256,
)


AUXILIARY_ATTACHMENT_VERSION = "starling_auxiliary_attachment.v1"
OUTPUT_FIELDS = ("global_context", "global_species_context")


@dataclass(frozen=True)
class _Lookup:
    source_columns: tuple[str, ...]
    values: Mapping[tuple[Any, ...], str | None]


def _resolve_output_fields(
    output_fields: Sequence[str] | Mapping[str, Sequence[str]],
    applicable_sources: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Normalize the declaration to one explicit field tuple per source."""
    if isinstance(output_fields, Mapping):
        missing = set(applicable_sources) - set(output_fields)
        if missing:
            raise ValueError(
                f"no auxiliary output fields declared for {sorted(missing)}"
            )
        extra = set(output_fields) - set(applicable_sources)
        if extra:
            raise ValueError(
                f"auxiliary output fields declared for inapplicable sources {sorted(extra)}"
            )
        resolved = {source: tuple(output_fields[source]) for source in applicable_sources}
    else:
        shared = tuple(output_fields)
        resolved = {source: shared for source in applicable_sources}
    for source, fields in resolved.items():
        if not fields:
            raise ValueError(f"source {source!r} declares no auxiliary output fields")
        if len(set(fields)) != len(fields):
            raise ValueError(f"source {source!r} declares duplicate auxiliary output fields")
    return resolved


class AuxiliaryMetadataAttacher:
    """Validated, immutable lookup over one task's reconciled mapping sidecar."""

    def __init__(
        self,
        path: str | Path,
        *,
        mapping_version: str,
        applicable_sources: Sequence[str],
        null_like: Sequence[str],
        output_fields: Sequence[str] | Mapping[str, Sequence[str]] = OUTPUT_FIELDS,
        attachment_version: str = AUXILIARY_ATTACHMENT_VERSION,
        source_column_aliases: Mapping[str, Mapping[str, str]] | None = None,
        non_null_outputs_when_input_present: Mapping[str, Sequence[str]] | None = None,
    ):
        """``output_fields`` is either one field set for every source, or a
        per-source mapping when sources reconcile different things -- one may
        carry a reconciled endpoint concept that another has no analogue for.
        """
        self.path = Path(path)
        self.applicable_sources = tuple(applicable_sources)
        self._null_like = frozenset(null_like)
        self.attachment_version = attachment_version
        self._source_column_aliases = {
            source: dict(aliases)
            for source, aliases in (source_column_aliases or {}).items()
        }
        unknown_alias_sources = set(self._source_column_aliases) - set(
            self.applicable_sources
        )
        if unknown_alias_sources:
            raise ValueError(
                "auxiliary aliases declared for inapplicable sources "
                f"{sorted(unknown_alias_sources)}"
            )
        self._non_null_outputs_when_input_present = {
            source: frozenset(fields)
            for source, fields in (non_null_outputs_when_input_present or {}).items()
        }
        self._per_source_output_fields = isinstance(output_fields, Mapping)
        self._output_fields_by_source = _resolve_output_fields(
            output_fields, self.applicable_sources
        )
        # Ordered union, used for the manifest and for the not-applicable
        # branch, which must return the same keys whatever the source.
        self._all_output_fields: tuple[str, ...] = tuple(
            dict.fromkeys(
                field
                for fields in self._output_fields_by_source.values()
                for field in fields
            )
        )
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
            expected_fields = self._output_fields_by_source[source_id]
            if not isinstance(source_outputs, Mapping) or set(source_outputs) != set(
                expected_fields
            ):
                raise ValueError(f"auxiliary output inventory mismatch for {source_id}")
            for output_field in expected_fields:
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
                    key = self._tuple_key(
                        source_id,
                        tuple(columns),
                        tuple(decoded),
                    )
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
            "attachment_version": self.attachment_version,
            "mapping_version": self.mapping_version,
            "mapping_path": str(self.path),
            "mapping_sha256": self.sha256,
            "tuple_key_cleaning": {
                "whitespace_and_null_cleaning": "clean_scalar",
                "measurement_role_cleaning": "clean_measurement_text",
                "null_like_values": sorted(self._null_like),
            },
            "source_column_aliases": {
                source: dict(sorted(aliases.items()))
                for source, aliases in sorted(self._source_column_aliases.items())
            },
            "non_null_outputs_when_input_present": {
                source: sorted(fields)
                for source, fields in sorted(
                    self._non_null_outputs_when_input_present.items()
                )
            },
            "output_fields": list(self._all_output_fields),
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
                for field in self._output_fields_by_source[source]
            },
        }
        # Preserve the frozen v1 manifest shape when every source uses the
        # historical shared field tuple.  Only task-specific field inventories
        # need the additional declaration.
        if self._per_source_output_fields:
            self._manifest["output_fields_by_source"] = {
                source: list(fields)
                for source, fields in sorted(self._output_fields_by_source.items())
            }

    def _tuple_key(
        self,
        source_id: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        cleaned: list[Any] = []
        aliases = self._source_column_aliases.get(source_id, {})
        for column, value in zip(columns, values, strict=True):
            target = aliases.get(column, column)
            item = (
                clean_measurement_text(value)
                if target in {"measurement_text", "unit_text"}
                else clean_scalar(value)
            )
            # The frozen mapping builder collapses source null sentinels before
            # it serializes tuple keys. Apply the identical null-equivalence
            # contract at attachment time; otherwise a cleaned source literal
            # such as ``unknown`` cannot join its authoritative ``null`` tuple.
            if isinstance(item, str) and item.casefold() in self._null_like:
                item = None
            cleaned.append(item)
        return tuple(cleaned)

    def _record_value(
        self, record: Mapping[str, Any], source_id: str, column: str
    ) -> Any:
        alias = self._source_column_aliases.get(source_id, {}).get(column, column)
        return record.get(alias)

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        source_id = str(record.get("source_id") or "")
        if source_id not in self.applicable_sources:
            return {
                **{field: None for field in self._all_output_fields},
                "auxiliary_mapping_status": "not_applicable",
                "auxiliary_attachment_version": self.attachment_version,
            }
        # Every source emits the full union so the persisted schema is stable;
        # fields this source does not reconcile stay explicitly null.
        output: dict[str, Any] = {field: None for field in self._all_output_fields}
        for output_field in self._output_fields_by_source[source_id]:
            lookup = self._lookups[(source_id, output_field)]
            key = self._tuple_key(
                source_id,
                lookup.source_columns,
                tuple(
                    self._record_value(record, source_id, column)
                    for column in lookup.source_columns
                ),
            )
            if key not in lookup.values:
                raise ValueError(
                    f"globally reconciled mapping lacks {source_id}/{output_field} "
                    f"tuple {key!r}"
                )
            output[output_field] = lookup.values[key]
            if (
                output_field
                in self._non_null_outputs_when_input_present.get(source_id, ())
                and any(value is not None for value in key)
                and output[output_field] is None
            ):
                raise ValueError(
                    f"globally reconciled mapping returned null for required "
                    f"{source_id}/{output_field} tuple {key!r}"
                )
        output.update(
            {
                "auxiliary_mapping_status": "mapped",
                "auxiliary_attachment_version": self.attachment_version,
            }
        )
        return output

    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest, sort_keys=True))

    @property
    def source_columns(self) -> dict[str, tuple[str, ...]]:
        """Return the frozen lookup inputs declared by each source."""
        return {
            source: tuple(
                dict.fromkeys(
                    column
                    for output in self._output_fields_by_source[source]
                    for column in self._lookups[(source, output)].source_columns
                )
            )
            for source in self.applicable_sources
        }

    def coverage_audit(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        output_counts: dict[str, Counter[str]] = defaultdict(Counter)
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
            for output_field in self._output_fields_by_source.get(source_id, ()):
                output_counts[f"{source_id}/{output_field}"][
                    "non_null" if record.get(output_field) is not None else "null"
                ] += 1
        return {
            "records": len(records),
            "source_status_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(source_counts.items())
            },
            "output_value_counts": {
                field: dict(sorted(counts.items()))
                for field, counts in sorted(output_counts.items())
            },
            "validations": {
                "all_applicable_records_mapped": True,
                "inapplicable_sources_explicit": True,
            },
        }


class SourceUniverseAuxiliaryAttacher:
    """Apply a frozen map to hash-pinned source-universe values by physical UID."""

    def __init__(
        self,
        path: str | Path,
        *,
        mapping_version: str,
        applicable_sources: Sequence[str],
        null_like: Sequence[str],
        universe_records_path: str | Path,
        universe_manifest_path: str | Path,
        task_id: str,
        output_fields: Sequence[str] | Mapping[str, Sequence[str]] = OUTPUT_FIELDS,
        source_aliases: Mapping[str, str] | None = None,
        attachment_version: str = AUXILIARY_ATTACHMENT_VERSION,
        non_null_outputs_when_input_present: Mapping[str, Sequence[str]] | None = None,
    ):
        self._attacher = AuxiliaryMetadataAttacher(
            path,
            mapping_version=mapping_version,
            applicable_sources=applicable_sources,
            null_like=null_like,
            output_fields=output_fields,
            attachment_version=attachment_version,
            non_null_outputs_when_input_present=non_null_outputs_when_input_present,
        )
        records_path = Path(universe_records_path)
        manifest_path = Path(universe_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        task = (manifest.get("tasks") or {}).get(task_id) or {}
        expected_path = manifest_path.parent / str(task.get("path") or "")
        if manifest.get("version") != "main_source_universe.v1":
            raise ValueError("source-universe manifest version changed")
        if records_path.resolve() != expected_path.resolve():
            raise ValueError("source-universe records path differs from its manifest")
        if file_sha256(records_path) != task.get("sha256"):
            raise ValueError("source-universe records hash differs from its manifest")

        aliases = dict(source_aliases or {})
        fields = tuple(
            dict.fromkeys(
                column
                for columns in self._attacher.source_columns.values()
                for column in columns
            )
        )
        table = pq.read_table(
            records_path,
            columns=["source_row_uid", "source_id", *fields],
        )
        if table.num_rows != int(task.get("rows") or -1):
            raise ValueError("source-universe row count differs from its manifest")
        self._fields = fields
        self._rows: dict[str, tuple[str, tuple[Any, ...]]] = {}
        applicable = set(applicable_sources)
        for row in table.to_pylist():
            uid = str(row.get("source_row_uid") or "")
            source = aliases.get(
                str(row.get("source_id") or ""), str(row.get("source_id") or "")
            )
            if not uid or uid in self._rows:
                raise ValueError("source universe contains a missing or duplicate UID")
            if source not in applicable:
                raise ValueError(f"source universe contains unknown source {source!r}")
            self._rows[uid] = (source, tuple(row.get(field) for field in fields))
        self._projection_manifest = {
            "version": "source_universe_auxiliary_projection.v1",
            "task_id": task_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "records_path": str(records_path),
            "records_sha256": str(task["sha256"]),
            "records": len(self._rows),
            "source_aliases": dict(sorted(aliases.items())),
            "projected_fields": list(fields),
            "join_key": "source_row_uid",
        }

    def _values(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        uid = str(record.get("source_row_uid") or "")
        if uid not in self._rows:
            raise ValueError(f"source-universe projection lacks UID {uid or '<missing>'}")
        source, values = self._rows[uid]
        if source != str(record.get("source_id") or ""):
            raise ValueError(f"source-universe source mismatch for UID {uid}")
        return values

    def _project(self, record: Mapping[str, Any]) -> dict[str, Any]:
        values = self._values(record)
        return {**record, **dict(zip(self._fields, values, strict=True))}

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._attacher.attach(self._project(record))

    def manifest(self) -> dict[str, Any]:
        return {
            **self._attacher.manifest(),
            "source_universe_projection": dict(self._projection_manifest),
        }

    def coverage_audit(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        for record in records:
            self._values(record)
        audit = self._attacher.coverage_audit(records)
        audit["validations"]["all_records_join_source_universe"] = True
        return audit


__all__ = [
    "AUXILIARY_ATTACHMENT_VERSION",
    "OUTPUT_FIELDS",
    "AuxiliaryMetadataAttacher",
    "SourceUniverseAuxiliaryAttacher",
]
