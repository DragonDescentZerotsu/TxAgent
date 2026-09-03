"""Import and validate the immutable seven-source ClinTox send_v2 delivery."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization.cleaning import (
    file_sha256,
    standardize_smiles,
)


ARCHIVE_SHA256 = "bf6da36bf2ac347d4763e5c3a234292c7f5c04576cccc4b75f6a4741246ac4ea"
SOURCE_RELEASE = "clintox_send_v2"
DEFAULT_ARCHIVE = Path("/vast/projects/myatskar/lab/shared_docs/clintox_send_v2.tar.gz")
DEFAULT_DATA_ROOT = Path("data/starling_data/clintox/send_v2")
HUMAN_CLINICAL_SOURCE_ID = "human_clinical_toxicity"
# Historical name retained for callers that use it only as a source identifier.
DIRECT_SOURCE_ID = HUMAN_CLINICAL_SOURCE_ID


@dataclass(frozen=True)
class SourceSpec:
    archive_dir: str
    source_id: str
    title: str
    expected_rows: int
    parquet_sha256: str
    guidance_sha256: str
    columns: tuple[str, ...]


COMMON_PREFIX = ("paragraph_idx", "support_text")
COMMON_SUFFIX = (
    "confidence", "needs_more_context", "pmid", "extraction_id", "SMILES",
)
SOURCE_SPECS = (
    SourceSpec(
        "base", HUMAN_CLINICAL_SOURCE_ID, "Human clinical toxicity", 584_307,
        "e09d712cc87a12c0e72195b5050c8b82bc5e4908b84bd38f43a4a00753f4b157",
        "94b520acf0c58979580ba685e32a48d5c9f2853f7640e452e6c75adacd8dcb88",
        (
            "paragraph_idx", "support_text", "molecule_name", "toxicity_outcome",
            "toxicity_category", "outcome_measure", "clinical_context",
            "dose_or_exposure", "fda_approval_status", "approved_indication",
            "extra_details", "confidence", "needs_more_context", "pmid",
            "extraction_id", "SMILES",
        ),
    ),
    SourceSpec(
        "v1", "nonclinical_in_vivo_toxicity", "Nonclinical in vivo toxicity",
        578_856, "923c9b43ecf4513c1fe2ed130472117b088981b07d8d2d07d12b1946a64256dc",
        "d7a71fcd3ab868abb9403341e87dd39a205c79cbabe2de9d4cf7f83d07ebbadc",
        (*COMMON_PREFIX, "evidence_type", "endpoint_value", "endpoint_unit",
         "administered_dose", "animal_context", "exposure_context",
         "observed_effect", "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
    SourceSpec(
        "v2", "organ_specific_toxicity", "Organ-specific toxicity", 674_318,
        "d1c43db1e666ce253586cf52dfac8fd6de9320c639463a8ab3a84b1e7a70c0f6",
        "d2f4585c2488461ad089418a51d92150e5c073cb4653a2c8498737eb3e01ce2a",
        (*COMMON_PREFIX, "organ_system", "toxicity_endpoint", "effect_status",
         "evidence_context", "biological_system", "exposure_regimen",
         "quantitative_result", "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
    SourceSpec(
        "v3", "genotoxicity_carcinogenicity", "Genotoxicity and carcinogenicity",
        570_749, "99a5eca0e26b97211ce2e980cd481529d2d4d7f534520918a4ce6c5f2d43193a",
        "f5b4f73e79456a8d73dba48a48d1367565e123552daa661e6bc5439b028166dc",
        (*COMMON_PREFIX, "evidence_category", "assay_type", "study_context",
         "endpoint", "result_direction", "biological_system",
         "exposure_conditions", "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
    SourceSpec(
        "v4", "cellular_stress", "Cellular stress", 526_737,
        "5f1aae6e9f3a9d5b23d7ec1cfb46a7e18f09fdf8ba5bd8cca6957095de3cf3d9",
        "4250d7ea0bae7ab4aa7056eaaf7ae94ae650787225aadc1ec060c2fc944234d4",
        (*COMMON_PREFIX, "stress_endpoint", "effect_direction", "evidence_basis",
         "mechanistic_effect", "target_or_pathway", "biological_model",
         "dose_and_duration", "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
    SourceSpec(
        "v5", "general_cytotoxicity", "General cytotoxicity", 807_290,
        "d6716731c1a283b00dc1b09dfbecaabe9045ed1765132ba736d4b0144537a99a",
        "670643f5ebe7ae542fe8de5489e81f9fe8e72b9008b9dba6048977d25533f7da",
        (*COMMON_PREFIX, "cell_model", "endpoint_type", "result_value",
         "result_unit", "test_concentration", "exposure_time_h", "assay_method",
         "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
    SourceSpec(
        "v6", "off_target_ddi_exposure", "Off-target, DDI, and exposure",
        1_104_657, "c16b8051e2543ab73d47a40ad1d737b26a513f0c588f42d1c72479a5009ea2d1",
        "e1506374463b3683132d908ebbb5a9463b07da19244cdfe02ba1791d264484c1",
        (*COMMON_PREFIX, "evidence_type", "target_or_endpoint",
         "target_identifier", "result_metric", "result_value", "result_unit",
         "assay_context", "qualifying_conditions", "extra_details", *COMMON_SUFFIX),
    ),
)
SOURCE_BY_ID = {spec.source_id: spec for spec in SOURCE_SPECS}
EXPECTED_SOURCE_ROWS = {spec.source_id: spec.expected_rows for spec in SOURCE_SPECS}
EXPECTED_SOURCE_SHA256 = {spec.source_id: spec.parquet_sha256 for spec in SOURCE_SPECS}
SOURCE_COLUMNS = {spec.source_id: spec.columns for spec in SOURCE_SPECS}


def _extract_members(
    archive_path: Path, staging: Path, published_root: Path
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as handle:
        members = {member.name: member for member in handle.getmembers()}
        for spec in SOURCE_SPECS:
            target_dir = staging / spec.source_id
            target_dir.mkdir(parents=True)
            for name in ("extractions.parquet", "extraction_guidance.json"):
                member_name = f"clintox_send/{spec.archive_dir}/{name}"
                member = members.get(member_name)
                if member is None or not member.isfile():
                    raise FileNotFoundError(f"missing archive member: {member_name}")
                source = handle.extractfile(member)
                if source is None:
                    raise FileNotFoundError(f"cannot read archive member: {member_name}")
                target = target_dir / name
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                files.append(
                    _file_manifest(spec, member_name, target, published_root)
                )
    return files


def _file_manifest(
    spec: SourceSpec, member: str, path: Path, published_root: Path
) -> dict[str, Any]:
    digest = file_sha256(path)
    expected = spec.parquet_sha256 if path.suffix == ".parquet" else spec.guidance_sha256
    if digest != expected:
        raise ValueError(f"digest mismatch for {member}: {digest}")
    return {
        "source_id": spec.source_id,
        "archive_member": member,
        "path": str(published_root / spec.source_id / path.name),
        "size": path.stat().st_size,
        "sha256": digest,
    }


def _source_audit(spec: SourceSpec, staging: Path) -> dict[str, Any]:
    path = staging / spec.source_id / "extractions.parquet"
    parquet = pq.ParquetFile(path)
    columns = tuple(parquet.schema_arrow.names)
    if columns != spec.columns:
        raise ValueError(f"schema mismatch for {spec.source_id}")
    if parquet.metadata.num_rows != spec.expected_rows:
        raise ValueError(f"row-count mismatch for {spec.source_id}")
    keys: set[tuple[str, str]] = set()
    smiles_counts: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=["pmid", "extraction_id", "SMILES"]):
        for row in batch.to_pylist():
            key = (str(row["pmid"] or ""), str(row["extraction_id"] or ""))
            if key in keys:
                raise ValueError(f"duplicate PMID/extraction_id in {spec.source_id}: {key}")
            keys.add(key)
            smiles = str(row["SMILES"] or "").strip()
            smiles_counts[smiles] = smiles_counts.get(smiles, 0) + 1
    return _structure_audit(spec, smiles_counts)


def _structure_audit(spec: SourceSpec, counts: dict[str, int]) -> dict[str, Any]:
    invalid: list[str] = []
    valid_rows = 0
    for smiles, rows in counts.items():
        canonical = standardize_smiles(smiles)[0] if smiles else None
        if canonical:
            valid_rows += rows
        else:
            invalid.append(smiles)
    nonempty = spec.expected_rows - counts.get("", 0)
    if nonempty != spec.expected_rows:
        raise ValueError(f"blank SMILES found in {spec.source_id}")
    return {
        "source_id": spec.source_id,
        "rows": spec.expected_rows,
        "columns": list(spec.columns),
        "composite_record_key": "pmid + extraction_id",
        "composite_record_key_unique": True,
        "smiles_nonempty_rows": nonempty,
        "unique_raw_smiles": len(counts),
        "rdkit_valid_rows": valid_rows,
        "rdkit_invalid_rows": spec.expected_rows - valid_rows,
        "rdkit_invalid_unique_smiles": len(invalid),
        "rdkit_invalid_examples": sorted(invalid)[:20],
    }


def import_archive(
    archive: str | Path = DEFAULT_ARCHIVE,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically import the exact delivery and publish its validation manifest."""
    archive_path = Path(archive)
    root = Path(data_root)
    digest = file_sha256(archive_path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"unexpected ClinTox archive SHA-256: {digest}")
    if root.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing source root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".send_v2.", dir=root.parent))
    try:
        files = _extract_members(archive_path, staging, root)
        sources = [_source_audit(spec, staging) for spec in SOURCE_SPECS]
        manifest = _manifest(archive_path, digest, files, sources)
        _write_json(staging / "SOURCE_MANIFEST.json", manifest)
        (staging / "README.md").write_text(_readme(manifest), encoding="utf-8")
        if root.exists():
            shutil.rmtree(root)
        staging.replace(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _manifest(
    archive: Path,
    digest: str,
    files: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "clintox_source_manifest.v2",
        "source_release": SOURCE_RELEASE,
        "archive": {"provided_path": str(archive), "size": archive.stat().st_size, "sha256": digest},
        "raw_policy": "Archive members are copied byte-for-byte; derived artifacts never modify them.",
        "structure_policy": "Only source SMILES is used; no global identifier fallback is permitted.",
        "qualifying_conditions": {
            HUMAN_CLINICAL_SOURCE_ID: "unavailable_in_source_schema",
            "other_sources": "source_visible_when_reported",
        },
        "known_missing_provenance": [
            "upstream dataset revision", "extraction model and software version",
            "entity-mapping method", "license",
        ],
        "files": files,
        "sources": sources,
    }


def _readme(manifest: dict[str, Any]) -> str:
    rows = [
        "# ClinTox send_v2 source\n",
        "This directory is the active immutable seven-source ClinTox delivery.\n",
        f"Archive SHA-256: `{manifest['archive']['sha256']}`.\n",
        "| source | rows | RDKit-valid | RDKit-invalid |",
        "|---|---:|---:|---:|",
    ]
    rows.extend(
        f"| `{item['source_id']}` | {item['rows']:,} | {item['rdkit_valid_rows']:,} | {item['rdkit_invalid_rows']:,} |"
        for item in manifest["sources"]
    )
    rows.extend([
        "", "All source rows contain a nonempty `SMILES`. Invalid structures remain auditable and fail closed.",
        "The clinical source lacks `qualifying_conditions`; it remains indirect evidence.", "",
        "The exact supplied archive is tracked as verified sub-100 MB parts under",
        "`artifacts/chembl_tool/tasks/clintox/clintox_send_v2_source/`.",
        "Restore a fresh checkout with `python -m tools.chembl_tool.tasks.clintox.starling_source_artifact_store restore-source`.", "",
    ])
    return "\n".join(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = import_archive(args.archive, data_root=args.data_root, force=args.force)
    print(
        json.dumps(
            {
                "source_release": manifest["source_release"],
                "archive_sha256": manifest["archive"]["sha256"],
                "sources": [
                    {
                        "source_id": source["source_id"],
                        "rows": source["rows"],
                        "rdkit_valid_rows": source["rdkit_valid_rows"],
                        "rdkit_invalid_rows": source["rdkit_invalid_rows"],
                    }
                    for source in manifest["sources"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_SHA256", "DEFAULT_ARCHIVE", "DEFAULT_DATA_ROOT", "DIRECT_SOURCE_ID",
    "HUMAN_CLINICAL_SOURCE_ID",
    "EXPECTED_SOURCE_ROWS", "EXPECTED_SOURCE_SHA256", "SOURCE_COLUMNS",
    "SOURCE_RELEASE", "SOURCE_SPECS", "import_archive", "main",
]
