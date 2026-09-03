"""Build the canonical Stage-3 pair buckets for one evidence-library release."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.build_pair_bucket_transfer_policy import (
    write_deterministic_gzip,
)
from data.processing.evidence_library.shared.v1.build_runtime import (
    assert_unpublished_build_root,
    starling_build_session,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v1.normalization.task_policy import (
    StarlingTaskPolicy,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    DEDUPLICATION_VERSION,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    DIRECT_MAPPING_FILENAME as DEDUP_DIRECT_MAPPING_FILENAME,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    DUPLICATES_FILENAME as DEDUP_DUPLICATES_FILENAME,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    PAIR_BUCKET_RECORDS_FILENAME as DEDUP_PAIR_BUCKET_RECORDS_FILENAME,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    RECORDS_FILENAME as DEDUP_RECORDS_FILENAME,
)
from data.processing.evidence_library.shared.v1.record_deduplication import (
    build_deduplicated_record_stage,
)
from data.processing.evidence_library.versions.v8.build_pair_bucket_distance_calibration import (
    CALIBRATION_FILENAME,
)

CORE_PAIR_BUCKET_STAGE = "03_pair_buckets"
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
PAIR_BUCKET_METADATA_FILENAME = "pair_bucket_metadata.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PairBucketBuildSpec:
    task_id: str
    policy: StarlingTaskPolicy
    pair_bucket_version: str
    build_sidecar: Callable[..., dict[str, Any]]
    build_transfer_policy: Callable[..., dict[str, Any]]
    direct_mapping_builder: (
        Callable[[Sequence[Mapping[str, Any]]], list[dict[str, Any]]] | None
    ) = None

    @property
    def pair_bucket_records_filename(self) -> str:
        return PAIR_BUCKET_RECORDS_FILENAME

    @property
    def pair_bucket_metadata_filename(self) -> str:
        return PAIR_BUCKET_METADATA_FILENAME


def build_canonical_artifacts(
    spec: PairBucketBuildSpec,
    *,
    normalized_root: str | Path,
    workers: int = 1,
    rebuild_request: Mapping[str, Any] | None = None,
    validation_level: str = "strict",
    cache_mode: str = "auto",
    apply_record_pruning: bool = True,
) -> dict[str, Any]:
    assert_unpublished_build_root(normalized_root)
    with starling_build_session(normalized_root):
        return _build_canonical_artifacts(
            spec,
            normalized_root=normalized_root,
            workers=workers,
            rebuild_request=rebuild_request,
            validation_level=validation_level,
            cache_mode=cache_mode,
            apply_record_pruning=apply_record_pruning,
        )


def _build_canonical_artifacts(
    spec: PairBucketBuildSpec,
    *,
    normalized_root: str | Path,
    workers: int = 1,
    rebuild_request: Mapping[str, Any] | None = None,
    validation_level: str = "strict",
    cache_mode: str = "auto",
    apply_record_pruning: bool = True,
) -> dict[str, Any]:
    """Build the active three-stage core through deduplicated pair buckets."""
    del validation_level
    started = time.monotonic()
    root = Path(normalized_root)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "02_canonicalized/records.parquet"
    auxiliary_manifest = root / "02_canonicalized/auxiliary_mapping_manifest.json"
    if not records_path.is_file() or not auxiliary_manifest.is_file():
        raise FileNotFoundError(
            "Stage 3 requires complete 02_canonicalized records and auxiliary manifest"
        )
    if "canonical_record_id" not in pq.read_schema(records_path).names:
        raise ValueError("Stage 3 requires canonical-v7 records")
    input_hashes = {
        "canonical_records": file_sha256(records_path),
        "auxiliary_mapping_manifest": file_sha256(auxiliary_manifest),
    }
    pruned_record_ineligibility: dict[str, str] = {}
    pruning_input: dict[str, Any] | None = None
    from data.processing.evidence_library.versions.v8.assay_transfer_record_pruning import (
        ARTIFACT_DIR as RECORD_PRUNING_DIR,
    )
    from data.processing.evidence_library.versions.v8.assay_transfer_record_pruning import (
        MANIFEST_FILENAME as RECORD_PRUNING_MANIFEST,
    )
    from data.processing.evidence_library.versions.v8.assay_transfer_record_pruning import (
        PRUNED_RECORDS_REMAIN_IN_LIBRARY,
        PRUNING_CAN_REJECT_BUCKETS,
        PRUNING_SCOPE,
        load_pruned_record_ineligibility,
    )

    pruning_manifest = root / RECORD_PRUNING_DIR / RECORD_PRUNING_MANIFEST
    if apply_record_pruning and pruning_manifest.is_file():
        pruned_record_ineligibility, pruning_input = load_pruned_record_ineligibility(
            pruning_manifest, task_id=spec.task_id, canonical_records_path=records_path
        )
        input_hashes["assay_transfer_record_pruning"] = pruning_input["manifest_sha256"]
    published = root / CORE_PAIR_BUCKET_STAGE
    published_manifest = published / MANIFEST_FILENAME
    if cache_mode == "auto" and published_manifest.is_file():
        cached = json.loads(published_manifest.read_text(encoding="utf-8"))
        if cached.get("input_hashes") == input_hashes and _core_outputs_match(
            published, cached.get("outputs") or {}
        ):
            return _read_json_if_present(root / MANIFEST_FILENAME)
    with tempfile.TemporaryDirectory(dir=root, prefix=".stage3-build-") as name:
        candidate = Path(name)
        raw_pair_dir = candidate / ".raw_pair_buckets"
        stage_dir = candidate / CORE_PAIR_BUCKET_STAGE
        pair_metadata = spec.build_sidecar(
            records_path=records_path,
            out_dir=raw_pair_dir,
            assay_transfer_record_ineligibility=pruned_record_ineligibility,
        )
        pair_metadata_path = stage_dir / PAIR_BUCKET_METADATA_FILENAME
        stage_dir.mkdir(parents=True, exist_ok=True)
        dedup = build_deduplicated_record_stage(
            task_id=spec.task_id,
            records_path=records_path,
            pair_bucket_records_path=raw_pair_dir / spec.pair_bucket_records_filename,
            out_dir=stage_dir,
            direct_mapping_builder=spec.direct_mapping_builder,
        )
        pair_metadata["output"] = {
            "path": str(
                root / CORE_PAIR_BUCKET_STAGE / DEDUP_PAIR_BUCKET_RECORDS_FILENAME
            ),
            "sha256": file_sha256(stage_dir / DEDUP_PAIR_BUCKET_RECORDS_FILENAME),
        }
        pair_metadata["deduplicated_records"] = dedup["summary"]
        _write_json(pair_metadata_path, pair_metadata)
        transfer_kwargs = {
            "records_path": stage_dir / DEDUP_RECORDS_FILENAME,
            "pair_bucket_records_path": stage_dir / DEDUP_PAIR_BUCKET_RECORDS_FILENAME,
            "pair_bucket_metadata_path": pair_metadata_path,
            "auxiliary_manifest_path": auxiliary_manifest,
            "out_dir": stage_dir,
        }
        if "workers" in inspect.signature(spec.build_transfer_policy).parameters:
            transfer_kwargs["workers"] = workers
        calibration = spec.build_transfer_policy(**transfer_kwargs)
        calibration = _project_paths(calibration, candidate, root)
        write_deterministic_gzip(stage_dir / CALIBRATION_FILENAME, calibration)
        outputs = {
            filename: file_sha256(stage_dir / filename)
            for filename in (
                DEDUP_RECORDS_FILENAME,
                DEDUP_PAIR_BUCKET_RECORDS_FILENAME,
                DEDUP_DIRECT_MAPPING_FILENAME,
                DEDUP_DUPLICATES_FILENAME,
                PAIR_BUCKET_METADATA_FILENAME,
                CALIBRATION_FILENAME,
            )
        }
        stage_manifest = {
            "version": "starling_core_stage3.v1",
            "task_id": spec.task_id,
            "input_hashes": input_hashes,
            "contract": {
                "pair_bucket_version": spec.pair_bucket_version,
                "record_deduplication_version": DEDUPLICATION_VERSION,
                "direct_vote_assay_transfer_eligibility": "normal_record_level_gates",
                "source_scoped_identity": True,
                "reference_scope_in_identity": False,
                "record_collapse": False,
                "residual_heterogeneity_gate": False,
                "minimum_records": 20,
                "minimum_distinct_molecules": 16,
                "pruning_scope": PRUNING_SCOPE,
                "pruning_can_reject_buckets": PRUNING_CAN_REJECT_BUCKETS,
                "pruned_records_remain_in_library": PRUNED_RECORDS_REMAIN_IN_LIBRARY,
                "bucket_eligibility_is_separate_from_pruning": True,
            },
            "summary": {
                "input_records": pair_metadata["stats"]["input_records"],
                **dedup["summary"],
                **calibration["summary"],
                "record_pruning": pruning_input,
            },
            "outputs": outputs,
            "validations": {
                **dedup["validations"],
                "one_stage3_directory": True,
                "record_collapse_absent": True,
                "residual_heterogeneity_gate_absent": True,
                "pruning_is_record_level_only": PRUNING_SCOPE
                == "record_level_assay_transfer_eligibility",
                "pruning_never_rejects_buckets": not PRUNING_CAN_REJECT_BUCKETS,
                "pruned_records_remain_in_library": PRUNED_RECORDS_REMAIN_IN_LIBRARY,
            },
        }
        _write_json(stage_dir / MANIFEST_FILENAME, stage_manifest)
        root_manifest = _read_json_if_present(root / MANIFEST_FILENAME)
        for legacy_key in (
            "compact_artifact_version",
            "index_version",
            "record_collapse_version",
            "semantic_aggregation_version",
            "final_endpoint_pruning_version",
        ):
            root_manifest.pop(legacy_key, None)
        root_manifest.update(
            {
                **spec.policy.manifest_versions(),
                "pipeline_layout_version": "starling_normalized_three_stage.v1",
                "artifact_scope": "canonical_split_independent",
                "completed_artifact_stages": [
                    "01_cleaned",
                    "02_canonicalized",
                    CORE_PAIR_BUCKET_STAGE,
                ],
                "canonical_scope": {
                    "cleaned_stage": "01_cleaned",
                    "canonicalized_stage": "02_canonicalized",
                    "pair_bucket_stage": CORE_PAIR_BUCKET_STAGE,
                    "paper_view_stage": None,
                },
                "canonical_stats": stage_manifest["summary"],
                "canonical_artifact_hashes": outputs,
                "rebuild_request": dict(
                    rebuild_request
                    or {
                        "from_stage": "03_pair_buckets",
                        "through_stage": "03_pair_buckets",
                    }
                ),
                "elapsed_stage3_s": round(time.monotonic() - started, 3),
            }
        )
        _write_json(candidate / MANIFEST_FILENAME, root_manifest)
        _validate_core_stage3(stage_dir)
        backup = Path(tempfile.mkdtemp(dir=root, prefix=".stage3-backup-"))
        active_manifest = root / MANIFEST_FILENAME
        had_stage = published.exists()
        had_manifest = active_manifest.exists()
        try:
            if had_stage:
                os.replace(published, backup / CORE_PAIR_BUCKET_STAGE)
            if had_manifest:
                os.replace(active_manifest, backup / MANIFEST_FILENAME)
            os.replace(stage_dir, published)
            os.replace(candidate / MANIFEST_FILENAME, active_manifest)
        except BaseException:
            if published.exists():
                os.replace(published, stage_dir)
            if had_stage:
                os.replace(backup / CORE_PAIR_BUCKET_STAGE, published)
            if had_manifest:
                os.replace(backup / MANIFEST_FILENAME, active_manifest)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)
    return root_manifest


def _core_outputs_match(root: Path, outputs: Mapping[str, Any]) -> bool:
    return bool(outputs) and all(
        (
            (root / filename).is_file()
            and file_sha256(root / filename) == str(expected)
            for filename, expected in outputs.items()
        )
    )


def _validate_core_stage3(stage_dir: Path) -> None:
    manifest = json.loads((stage_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not all((manifest.get("validations") or {}).values()):
        raise ValueError("Stage-3 validation failed")
    records = pq.ParquetFile(stage_dir / DEDUP_RECORDS_FILENAME).metadata.num_rows
    sidecar = pq.ParquetFile(
        stage_dir / DEDUP_PAIR_BUCKET_RECORDS_FILENAME
    ).metadata.num_rows
    if records != sidecar:
        raise ValueError("Stage-3 record and pair-bucket cardinality differ")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _project_paths(value: Any, candidate: Path, root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _project_paths(item, candidate, root) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_project_paths(item, candidate, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_project_paths(item, candidate, root) for item in value)
    return value.replace(str(candidate), str(root)) if isinstance(value, str) else value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CORE_PAIR_BUCKET_STAGE",
    "PAIR_BUCKET_METADATA_FILENAME",
    "PAIR_BUCKET_RECORDS_FILENAME",
    "PairBucketBuildSpec",
    "build_canonical_artifacts",
]
