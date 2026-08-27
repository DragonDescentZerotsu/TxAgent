"""Rebuild only the assay-transfer canonical tuple in frozen Stage-03 rows."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    canonicalize_assay_transfer_base,
    finalize_assay_transfer_measurement,
    load_measurement_policy,
    validate_final_assay_transfer_measurements,
)
from tools.chembl_tool.common.starling.build_normalized_evidence_library import (
    load_task_policy,
)
from tools.chembl_tool.common.starling.normalization.audit import (
    scalar_distribution_audit,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.retrieval_boundary import (
    FROZEN_RETRIEVAL_FIELDS,
)


_PREBASE_FIELDS = (
    "assay_transfer_prebase_measurement_text",
    "assay_transfer_prebase_unit_text",
    "assay_transfer_prebase_scalar_value",
    "assay_transfer_pretransform_variation_value",
    "assay_transfer_scale_factor",
)


def _restore_stage02_prebase(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Restore the untransformed tuple instead of reusing final Stage-03 values."""
    authority = pd.read_parquet(path, columns=["canonical_record_id", *_PREBASE_FIELDS])
    if not authority["canonical_record_id"].is_unique:
        raise ValueError("Stage-02 canonical record IDs are not unique")
    renamed = {
        field: f"_stage02_{field}" for field in _PREBASE_FIELDS
    }
    authority = authority.rename(columns=renamed)
    restored = frame.merge(
        authority,
        on="canonical_record_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not restored["_merge"].eq("both").all():
        raise ValueError("Stage-03 row lacks its canonical Stage-02 authority")
    restored["canonical_measurement_text"] = restored[
        renamed["assay_transfer_prebase_measurement_text"]
    ]
    restored["canonical_unit_text"] = restored[
        renamed["assay_transfer_prebase_unit_text"]
    ]
    restored["finite_scalar_value"] = restored[
        renamed["assay_transfer_prebase_scalar_value"]
    ]
    variation = pd.to_numeric(
        restored[renamed["assay_transfer_pretransform_variation_value"]],
        errors="coerce",
    )
    factor = pd.to_numeric(
        restored[renamed["assay_transfer_scale_factor"]], errors="coerce"
    )
    restored["variation_value"] = variation.where(factor.isna(), variation / factor)
    return restored.drop(columns=["_merge", *renamed.values()])


def _restore_stage03_pretransform(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore the persisted post-base tuple when older Stage 02 lacks prebase."""
    required = {
        "assay_transfer_pretransform_measurement_text",
        "assay_transfer_pretransform_unit_text",
        "assay_transfer_pretransform_scalar_value",
        "assay_transfer_pretransform_variation_value",
    }
    if missing := sorted(required - set(frame)):
        raise ValueError(f"Stage-03 pretransform authority lacks columns: {missing}")
    restored = frame.copy()
    restored["canonical_measurement_text"] = restored[
        "assay_transfer_pretransform_measurement_text"
    ]
    restored["canonical_unit_text"] = restored[
        "assay_transfer_pretransform_unit_text"
    ]
    restored["finite_scalar_value"] = restored[
        "assay_transfer_pretransform_scalar_value"
    ]
    restored["variation_value"] = restored[
        "assay_transfer_pretransform_variation_value"
    ]
    return restored


def _restore_stage02_source_projection(
    frame: pd.DataFrame, path: Path, *, fields: tuple[str, ...]
) -> pd.DataFrame:
    """Restore retrieval-visible source values from their Stage-02 authority."""
    authority = pd.read_parquet(path, columns=["canonical_record_id", *fields])
    if not authority["canonical_record_id"].is_unique:
        raise ValueError("Stage-02 canonical record IDs are not unique")
    renamed = {field: f"_stage02_source_{field}" for field in fields}
    restored = frame.merge(
        authority.rename(columns=renamed),
        on="canonical_record_id",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not restored["_merge"].eq("both").all():
        raise ValueError("Stage-03 row lacks its Stage-02 source projection")
    for field, source_field in renamed.items():
        restored[field] = restored[source_field]
    return restored.drop(columns=["_merge", *renamed.values()])


def _tuple_authority(root: Path) -> str:
    names = set(pq.read_schema(root / "02_canonicalized/records.parquet").names)
    return (
        "stage02_assay_transfer_prebase"
        if set(_PREBASE_FIELDS) <= names
        else "stage03_assay_transfer_pretransform"
    )


def _task_args(task_id: str, root: Path) -> Any:
    module = importlib.import_module(
        f"tools.chembl_tool.tasks.{task_id}.build_normalized_starling_evidence_library"
    )
    return module.parse_args(load_task_policy(task_id), ["--out-dir", str(root)])


def _canonicalize_record(
    row: dict[str, Any],
    *,
    contract: Any,
    hooks: Any,
    policy: dict[str, Any],
    apply_base: bool,
    retrieval_boundary_fields: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    working = contract.inflate_canonical(row)
    if apply_base:
        working, changed = canonicalize_assay_transfer_base(working, policy)
        if changed:
            working.update(dict(hooks.assay_transfer_revalidator(working)))
    projected = contract.canonical_projection(working)
    working, projected = finalize_assay_transfer_measurement(
        working, projected, record_contract=contract, policy=policy
    )
    projected.update({field: row.get(field) for field in retrieval_boundary_fields})
    return working, projected


def _rebuild_rows(task_id: str, root: Path) -> tuple[list[dict], list[dict], dict]:
    policy_binding = load_task_policy(task_id)
    contract = policy_binding.record_contract
    policy_path = policy_binding.assay_transfer_measurement_policy
    if contract is None or policy_path is None:
        raise ValueError(f"{task_id} lacks the assay-transfer v7 contract")
    hooks = policy_binding.build_hooks(_task_args(task_id, root))
    policy = load_measurement_policy(policy_path)
    path = root / "03_records/records.parquet"
    frame = pd.read_parquet(path)
    source_fields = tuple(
        sorted(
            {
                "cleaned_record_id",
                "group_id",
                "source_id",
                *(
                    field
                    for profile in contract.sources.values()
                    for field in profile.source_visible_fields
                ),
            }
        )
    )
    frame = _restore_stage02_source_projection(
        frame,
        root / "02_canonicalized/records.parquet",
        fields=source_fields,
    )
    authority = _tuple_authority(root)
    frame = (
        _restore_stage02_prebase(frame, root / "02_canonicalized/records.parquet")
        if authority == "stage02_assay_transfer_prebase"
        else _restore_stage03_pretransform(frame)
    )
    source_rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    working_rows, projected_rows = [], []
    for row in source_rows:
        working, projected = _canonicalize_record(
            row,
            contract=contract,
            hooks=hooks,
            policy=policy,
            apply_base=authority == "stage02_assay_transfer_prebase",
            retrieval_boundary_fields=tuple(
                dict.fromkeys((*source_fields, *FROZEN_RETRIEVAL_FIELDS))
            ),
        )
        working_rows.append(working)
        projected_rows.append(projected)
    errors = validate_final_assay_transfer_measurements(projected_rows)
    if errors:
        raise ValueError(f"invalid rebuilt Stage-03 tuple: {errors[0]}")
    return working_rows, projected_rows, policy


def _write_records(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    integer_fields: tuple[str, ...] = (),
) -> None:
    old_columns = pq.read_schema(path).names
    frame = pd.DataFrame.from_records(rows)
    for field in integer_fields:
        if field in frame:
            frame[field] = pd.array(frame[field], dtype="Int64")
    columns = [*old_columns, *(column for column in frame if column not in old_columns)]
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as f:
        temporary = Path(f.name)
    try:
        frame.loc[:, columns].to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_manifests(task_id: str, root: Path, policy: dict[str, Any]) -> None:
    records = root / "03_records/records.parquet"
    stage_path = root / "03_records/manifest.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["output"] = {"path": str(records), "sha256": file_sha256(records)}
    stage["assay_transfer_rebuild"] = {
        "scope": "stage03_canonical_measurement_tuple_only",
        "policy_version": policy["policy_version"],
        "retrieval_boundary": "frozen",
        "tuple_authority": _tuple_authority(root),
    }
    stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("record_build_cache", None)
    manifest["assay_transfer_stage03_rebuild"] = {
        "task_id": task_id,
        "records_sha256": file_sha256(records),
        "policy_version": policy["policy_version"],
        "retrieval_boundary": "frozen",
        "tuple_authority": _tuple_authority(root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def rebuild(task_id: str, root: Path) -> None:
    working, projected, policy = _rebuild_rows(task_id, root)
    records_path = root / "03_records/records.parquet"
    stage02_schema = pq.read_schema(root / "02_canonicalized/records.parquet")
    _write_records(
        records_path,
        projected,
        integer_fields=tuple(
            field.name for field in stage02_schema if pa.types.is_integer(field.type)
        ),
    )
    audit = scalar_distribution_audit(working)
    pd.DataFrame.from_records(audit).to_parquet(
        root / "03_records/scalar_distribution_audit.parquet", index=False
    )
    _update_manifests(task_id, root, policy)
    print(json.dumps({"task_id": task_id, "records": len(projected)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.task, args.root)


if __name__ == "__main__":
    main()
