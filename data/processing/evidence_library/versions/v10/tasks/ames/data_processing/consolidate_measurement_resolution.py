"""Build and validate the provenance-preserving mixed Ames measurement mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import (
    SubmissionCache,
)


GENERATION_VERSION = "ames_measurement_resolution_mixed.v1"
MAPPING_VERSION = "ames_measurement_resolution.v6_mixed_retry"
RETRYABLE_PREFIXES = (
    "ambiguous_process_termination",
    "api_failure",
    "invalid_response_after_structural_retry",
    "invalid_row_response:",
    "worker_failure",
)
SOURCE_IDS = {
    "mutagenicity_outcomes",
    "fixed_mutation",
    "premutagenic_damage",
    "mutagenicity_mechanism",
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _extract_rows(records_path: Path) -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    columns = [
        "cleaned_record_id",
        "source_row_uid",
        "source_id",
        "measurement_resolution_route",
    ]
    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            if (
                row.get("measurement_resolution_route") != "extract"
                or row.get("source_id") not in SOURCE_IDS
            ):
                continue
            record_id = str(row.get("cleaned_record_id") or "")
            identity = (
                str(row.get("source_row_uid") or ""),
                str(row.get("source_id") or ""),
            )
            if not record_id or not identity[0] or record_id in output:
                raise ValueError("invalid or duplicate Ames extraction identity")
            output[record_id] = identity
    return output


def _is_publishable(assignment: Mapping[str, Any]) -> bool:
    return not str(assignment.get("assignment_method") or "").startswith(
        RETRYABLE_PREFIXES
    )


def _layer(
    spec: Mapping[str, Any], records_sha256: str, profile_sha256: str
) -> tuple[dict[str, Any], SubmissionCache, dict[str, Any]]:
    layer_id = str(spec.get("layer_id") or "")
    cache_path = Path(str(spec.get("cache_path") or ""))
    contract_path = Path(
        str(spec.get("contract_path") or cache_path.parent / "input_contract.json")
    )
    max_tokens = int(spec.get("max_completion_tokens") or 0)
    if not layer_id or not cache_path.is_file() or not contract_path.is_file():
        raise ValueError(f"invalid mixed-mapping layer: {layer_id!r}")
    if max_tokens < 1:
        raise ValueError(f"layer {layer_id} lacks max_completion_tokens")
    contract = _json(contract_path)
    if contract.get("task") != "ames":
        raise ValueError(f"non-Ames cache layer: {layer_id}")
    if contract.get("records_sha256") != records_sha256:
        raise ValueError(f"records hash mismatch in layer {layer_id}")
    if contract.get("profile_sha256") != profile_sha256:
        raise ValueError(f"profile hash mismatch in layer {layer_id}")
    prompt = contract.get("prompt") or {}
    if not prompt.get("prompt_version") or not prompt.get("template_sha256"):
        raise ValueError(f"prompt provenance missing in layer {layer_id}")
    reasoning_effort = str(contract.get("reasoning_effort") or "")
    if reasoning_effort not in {"low", "high"}:
        raise ValueError(f"reasoning provenance missing in layer {layer_id}")
    cache = SubmissionCache(cache_path)
    public = {
        "layer_id": layer_id,
        "cache_path": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "prompt": prompt,
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_tokens,
    }
    return public, cache, contract


def build_mixed_mapping(
    *,
    records_path: Path,
    profile_path: Path,
    layers: list[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Fill unresolved rows from ordered immutable caches and write one mapping."""
    records_sha256 = file_sha256(records_path)
    profile_sha256 = file_sha256(profile_path)
    identities = _extract_rows(records_path)
    selected: dict[str, dict[str, Any]] = {}
    public_layers: list[dict[str, Any]] = []
    for spec in layers:
        public, cache, _ = _layer(spec, records_sha256, profile_sha256)
        accepted = 0
        for record_id, assignment in cache.assignments.items():
            if record_id not in identities:
                raise ValueError(
                    f"layer {public['layer_id']} contains a noncandidate row: {record_id}"
                )
            if not _is_publishable(assignment):
                continue
            if record_id in selected:
                raise ValueError(
                    f"layer {public['layer_id']} would replace accepted row {record_id}"
                )
            if assignment.get("rejected_response_json") not in (None, ""):
                raise ValueError(f"publishable row retains rejection: {record_id}")
            source_row_uid, source_id = identities[record_id]
            provenance = cache.provenance.get(record_id, {})
            row = {
                **assignment,
                "source_row_uid": source_row_uid,
                "source_id": source_id,
                "inference_source": "delta_inference",
                "inference_model": str(
                    assignment.get("returned_model")
                    or provenance.get("inference_model")
                    or ""
                ),
                "inference_base_url": str(
                    assignment.get("base_url")
                    or provenance.get("inference_base_url")
                    or ""
                ),
                "inference_credential_env": str(
                    provenance.get("inference_credential_env") or ""
                ),
                "inference_layer_id": public["layer_id"],
                "inference_prompt_version": public["prompt"]["prompt_version"],
                "inference_prompt_template_sha256": public["prompt"][
                    "template_sha256"
                ],
                "inference_reasoning_effort": public["reasoning_effort"],
                "inference_max_completion_tokens": public[
                    "max_completion_tokens"
                ],
                "inference_cache_sha256": public["cache_sha256"],
                "inference_cache_contract_sha256": public["contract_sha256"],
            }
            selected[record_id] = row
            accepted += 1
        public["accepted_rows"] = accepted
        public_layers.append(public)
    missing = set(identities) - set(selected)
    if missing:
        raise ValueError(
            f"mixed mapping remains incomplete: {len(missing)} rows; first={min(missing)}"
        )
    rows = [selected[record_id] for record_id in sorted(selected)]
    columns = sorted(set().union(*(row.keys() for row in rows)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows]),
        temporary,
    )
    temporary.replace(output_path)
    status_counts = Counter(str(row["status"]) for row in rows)
    manifest = {
        "generation_version": GENERATION_VERSION,
        "mapping_version": MAPPING_VERSION,
        "task_id": "ames",
        "mapping_path": str(output_path),
        "mapping_sha256": file_sha256(output_path),
        "mapping_rows": len(rows),
        "cleaned_records_path": str(records_path),
        "cleaned_records_sha256": records_sha256,
        "endpoint_profile": {
            "path": str(profile_path),
            "sha256": profile_sha256,
        },
        "layers": public_layers,
        "layer_counts": dict(
            sorted(Counter(row["inference_layer_id"] for row in rows).items())
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "inference_model_counts": dict(
            sorted(Counter(row["inference_model"] for row in rows).items())
        ),
        "inference_base_url_counts": dict(
            sorted(Counter(row["inference_base_url"] for row in rows).items())
        ),
        "validations": {
            "one_row_per_candidate": len(rows) == len(identities),
            "unique_cleaned_record_ids": len(selected) == len(rows),
            "only_ok_carries_measurements": all(
                (int(row["quantity_count"]) > 0) == (row["status"] == "ok")
                for row in rows
            ),
            "zero_rejected_rows": all(
                row.get("rejected_response_json") in (None, "") for row in rows
            ),
        },
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_mixed_mapping(output_path)
    return manifest


def validate_mixed_mapping(
    mapping_path: str | Path, *, expected_record_ids: set[str] | None = None
) -> None:
    path = Path(mapping_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"mixed mapping or manifest not found: {path}")
    manifest = _json(manifest_path)
    if manifest.get("generation_version") != GENERATION_VERSION:
        raise ValueError("mixed mapping generation version mismatch")
    if manifest.get("mapping_version") != MAPPING_VERSION:
        raise ValueError("mixed mapping version mismatch")
    if manifest.get("mapping_sha256") != file_sha256(path):
        raise ValueError("mixed mapping hash mismatch")
    records_path = Path(str(manifest.get("cleaned_records_path") or ""))
    profile_path = Path(str((manifest.get("endpoint_profile") or {}).get("path") or ""))
    if manifest.get("cleaned_records_sha256") != file_sha256(records_path):
        raise ValueError("mixed mapping records hash mismatch")
    if (manifest.get("endpoint_profile") or {}).get("sha256") != file_sha256(
        profile_path
    ):
        raise ValueError("mixed mapping profile hash mismatch")
    identities = _extract_rows(records_path)
    expected = set(identities) if expected_record_ids is None else set(expected_record_ids)
    rows = pq.read_table(path).to_pylist()
    observed = [str(row.get("cleaned_record_id") or "") for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("mixed mapping candidate coverage mismatch")
    layer_by_id = {str(layer["layer_id"]): layer for layer in manifest["layers"]}
    layer_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for row in rows:
        record_id = str(row["cleaned_record_id"])
        if (str(row.get("source_row_uid") or ""), str(row.get("source_id") or "")) != identities[record_id]:
            raise ValueError(f"mixed mapping source identity mismatch: {record_id}")
        layer_id = str(row.get("inference_layer_id") or "")
        layer = layer_by_id.get(layer_id)
        if layer is None:
            raise ValueError(f"mixed mapping has unknown layer: {layer_id}")
        expected_provenance = (
            layer["prompt"]["prompt_version"],
            layer["prompt"]["template_sha256"],
            layer["reasoning_effort"],
            int(layer["max_completion_tokens"]),
            layer["cache_sha256"],
            layer["contract_sha256"],
        )
        found_provenance = (
            row.get("inference_prompt_version"),
            row.get("inference_prompt_template_sha256"),
            row.get("inference_reasoning_effort"),
            int(row.get("inference_max_completion_tokens") or 0),
            row.get("inference_cache_sha256"),
            row.get("inference_cache_contract_sha256"),
        )
        if found_provenance != expected_provenance:
            raise ValueError(f"mixed mapping row provenance mismatch: {record_id}")
        if row.get("rejected_response_json") not in (None, ""):
            raise ValueError(f"mixed mapping contains a rejected row: {record_id}")
        if (int(row.get("quantity_count") or 0) > 0) != (row.get("status") == "ok"):
            raise ValueError(f"mixed mapping measurement/status mismatch: {record_id}")
        layer_counts[layer_id] += 1
        status_counts[str(row["status"])] += 1
    if dict(sorted(layer_counts.items())) != manifest.get("layer_counts"):
        raise ValueError("mixed mapping layer counts mismatch")
    if dict(sorted(status_counts.items())) != manifest.get("status_counts"):
        raise ValueError("mixed mapping status counts mismatch")
    for layer in manifest["layers"]:
        if file_sha256(Path(layer["cache_path"])) != layer["cache_sha256"]:
            raise ValueError(f"mixed mapping cache changed: {layer['layer_id']}")
        if file_sha256(Path(layer["contract_path"])) != layer["contract_sha256"]:
            raise ValueError(f"mixed mapping contract changed: {layer['layer_id']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--layers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _json(args.layers)
    build_mixed_mapping(
        records_path=args.records,
        profile_path=args.profile,
        layers=list(payload.get("layers") or ()),
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
