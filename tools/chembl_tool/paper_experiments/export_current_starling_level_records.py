"""Export collaborator-readable ledgers for the current Starling level records.

The export intentionally distinguishes two surfaces:

1. every retrieval-eligible source record and its static family level; and
2. the representative record cards actually materialized in each frozen
   scaffold/random assay-molecule index.

Per-query selected cards are run outputs and are not part of this static dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from tools.chembl_tool.common.build_runtime import local_input
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.starling.current_retrieval_artifacts import (
    DEFAULT_LOCAL_ROOT,
    TASKS,
)
from tools.chembl_tool.paper_experiments import rebuild_current_starling_retrieval as rebuild


EXPORT_VERSION = "current_starling_level_record_share.v1"
MAX_PARQUET_BYTES = 90_000_000
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/chembl_tool/starling/current_level_records"
)
_TASK_CONFIG_MODULES = {
    "bbb_martins": "tools.chembl_tool.tasks.bbb_martins.experiment_config",
    "bioavailability_ma": "tools.chembl_tool.tasks.bioavailability_ma.experiment_config",
    "skin_reaction": "tools.chembl_tool.tasks.skin_reaction.experiment_config",
    "ames": "tools.chembl_tool.tasks.ames.experiment_config",
    "dili": "tools.chembl_tool.tasks.dili.experiment_config",
    "carcinogens": "tools.chembl_tool.tasks.carcinogens.experiment_config",
}

_TEXT_COLUMNS = (
    "canonical_record_id",
    "source_id",
    "source_name",
    "source_record_id",
    "molecule_id",
    "molecule_name",
    "canonical_smiles",
    "canonical_assay_context",
    "canonical_endpoint_name",
    "canonical_measurement_text",
    "canonical_unit_text",
    "canonical_species_context",
    "qualifying_conditions",
    "support_text",
    "confidence",
    "source_family_original_group_id",
    "source_family_purity_reason",
)

SOURCE_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("task", pa.string()),
        pa.field("level", pa.int32()),
        pa.field("family_key", pa.string()),
        pa.field("level_description", pa.string()),
        pa.field("source_group_id", pa.string()),
        pa.field("legacy_family_id", pa.string()),
        *[pa.field(name, pa.string()) for name in _TEXT_COLUMNS],
        pa.field("card_fingerprint_sha256", pa.string()),
        pa.field("retrieval_eligible", pa.bool_()),
    ]
)

INDEXED_CARD_SCHEMA = pa.schema(
    [
        pa.field("task", pa.string()),
        pa.field("split_scheme", pa.string()),
        pa.field("level", pa.int32()),
        pa.field("family_key", pa.string()),
        pa.field("level_description", pa.string()),
        pa.field("legacy_family_id", pa.string()),
        pa.field("assay_first_level", pa.int32()),
        pa.field("assay_id", pa.string()),
        pa.field("molecule_id", pa.string()),
        pa.field("canonical_smiles", pa.string()),
        pa.field("card_position_in_assay_molecule", pa.int32()),
        pa.field("source_record_count_in_assay_molecule", pa.int64()),
        pa.field("card_fingerprint_sha256", pa.string()),
    ]
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int_or_zero(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _card_json(example: dict[str, Any]) -> str:
    return json.dumps(example, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _card_fingerprint(example: dict[str, Any]) -> str:
    return hashlib.sha256(_card_json(example).encode("utf-8")).hexdigest()


def _atomic_parquet_writer(path: Path, schema: pa.Schema) -> tuple[pq.ParquetWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    return (
        pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        ),
        temporary,
    )


def _publish_parquet(writer: pq.ParquetWriter, temporary: Path, path: Path) -> None:
    writer.close()
    temporary.replace(path)


def _source_file_inventory(path: Path) -> dict[str, Any]:
    """Keep large ledgers readable by Arrow while respecting ordinary Git limits."""
    if path.stat().st_size <= MAX_PARQUET_BYTES:
        return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    dataset = path.with_suffix("")
    parts = []
    with tempfile.TemporaryDirectory(prefix="level-records-", dir=path.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        parquet = pq.ParquetFile(path)
        for index, batch in enumerate(parquet.iter_batches(batch_size=65_536)):
            part = stage / f"part-{index:05d}.parquet"
            pq.write_table(pa.Table.from_batches([batch]), part, compression="zstd")
            size = part.stat().st_size
            if size > MAX_PARQUET_BYTES:
                raise ValueError(f"Source ledger shard exceeds Git file budget: {part}")
            parts.append({"path": str(dataset / part.name), "sha256": sha256_file(part), "size_bytes": size})
        parquet.close()
        previous = Path(tmp) / "previous"
        if dataset.exists():
            dataset.rename(previous)
        try:
            stage.rename(dataset)
        except BaseException:
            if previous.exists():
                previous.rename(dataset)
            raise
    path.unlink()
    return {"path": str(dataset), "parts": parts, "size_bytes": sum(part["size_bytes"] for part in parts)}


def _level_maps(catalog_manifest: Path) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    manifest = json.loads(catalog_manifest.read_text(encoding="utf-8"))
    by_source_group: dict[str, dict[str, Any]] = {}
    by_level: dict[int, dict[str, Any]] = {}
    for row in manifest.get("levels") or []:
        level = int(row["level"])
        normalized = {
            "level": level,
            "family_key": _text(row.get("endpoint_group")),
            "legacy_family_id": _text(row.get("family_id")),
        }
        by_level[level] = normalized
        for source_group in row.get("source_groups") or []:
            key = str(source_group)
            previous = by_source_group.get(key)
            if previous is not None and previous != normalized:
                raise ValueError(
                    f"source group {key!r} maps to multiple current levels in {catalog_manifest}"
                )
            by_source_group[key] = normalized
    if not by_source_group or not by_level:
        raise ValueError(f"catalog manifest has no level mappings: {catalog_manifest}")
    return by_source_group, by_level


def _level_descriptions(task: str, by_level: dict[int, dict[str, Any]]) -> dict[int, str]:
    module_name = _TASK_CONFIG_MODULES.get(task)
    if not module_name:
        return {level: "" for level in by_level}
    module = importlib.import_module(module_name)
    endpoint_descriptions = getattr(
        module, "PROGRESSIVE_ASSAY_ENDPOINT_DESCRIPTIONS", {}
    )
    level_descriptions = getattr(module, "PROGRESSIVE_ASSAY_LEVEL_DESCRIPTIONS", {})
    return {
        level: str(
            endpoint_descriptions.get(row["family_key"])
            or level_descriptions.get(level)
            or ""
        )
        for level, row in by_level.items()
    }


def _string_array(table: pa.Table, name: str) -> pa.Array | pa.ChunkedArray:
    # Some sources assign families during canonicalization rather than moving
    # pre-existing source groups. Preserve that absence instead of inventing one.
    if name == "source_family_original_group_id" and name not in table.column_names:
        return pa.nulls(table.num_rows, type=pa.string())
    if name == "source_family_purity_reason" and name not in table.column_names:
        return pc.cast(table["level_assignment_reason"], pa.string(), safe=False)
    return pc.cast(table[name], pa.string(), safe=False)


def export_source_membership(
    *,
    task: str,
    records_path: Path,
    catalog_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    by_source_group, by_level = _level_maps(catalog_manifest)
    descriptions = _level_descriptions(task, by_level)
    parquet = pq.ParquetFile(records_path)
    columns = ["group_id", "retrieval_eligible", *_TEXT_COLUMNS]
    if "source_family_purity_reason" not in parquet.schema_arrow.names:
        columns.remove("source_family_purity_reason")
        columns.append("level_assignment_reason")
    required = set(columns) - {"source_family_original_group_id"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing canonical record columns: {sorted(missing)}")
    columns = [name for name in columns if name in parquet.schema_arrow.names]
    writer, temporary = _atomic_parquet_writer(output_path, SOURCE_MEMBERSHIP_SCHEMA)
    counts: Counter[int] = Counter()
    n_rows = 0
    try:
        for batch in parquet.iter_batches(columns=columns, batch_size=65_536):
            table = pa.Table.from_batches([batch])
            groups = pc.cast(table["group_id"], pa.string(), safe=False)
            mask = pc.and_(
                pc.fill_null(table["retrieval_eligible"], False),
                pc.is_in(groups, value_set=pa.array(sorted(by_source_group))),
            )
            selected = table.filter(mask)
            if not selected.num_rows:
                continue
            source_groups = [_text(value) for value in selected["group_id"].to_pylist()]
            levels = [int(by_source_group[value]["level"]) for value in source_groups]
            family_keys = [
                by_source_group[value]["family_key"] for value in source_groups
            ]
            values = {
                name: [_text(value) for value in selected[name].to_pylist()]
                for name in (
                    "canonical_endpoint_name",
                    "canonical_measurement_text",
                    "canonical_unit_text",
                    "canonical_assay_context",
                    "canonical_species_context",
                    "qualifying_conditions",
                    "support_text",
                )
            }
            fingerprints = [
                _card_fingerprint(
                    {
                        "endpoint_type": values["canonical_endpoint_name"][index],
                        "reported_value": values["canonical_measurement_text"][index],
                        "reported_units": values["canonical_unit_text"][index],
                        "assay_context": values["canonical_assay_context"][index],
                        "species_context": values["canonical_species_context"][index],
                        "qualifying_conditions": values["qualifying_conditions"][index],
                        "support_text": values["support_text"][index],
                        "evidence_family": family_keys[index],
                        "evidence_family_level": levels[index],
                    }
                )
                for index in range(selected.num_rows)
            ]
            counts.update(levels)
            n_rows += selected.num_rows
            output = pa.Table.from_arrays(
                [
                    pa.array([task] * selected.num_rows, type=pa.string()),
                    pa.array(levels, type=pa.int32()),
                    pa.array(
                        family_keys,
                        type=pa.string(),
                    ),
                    pa.array([descriptions[level] for level in levels], type=pa.string()),
                    pa.array(source_groups, type=pa.string()),
                    pa.array(
                        [
                            by_source_group[value]["legacy_family_id"]
                            for value in source_groups
                        ],
                        type=pa.string(),
                    ),
                    *[_string_array(selected, name) for name in _TEXT_COLUMNS],
                    pa.array(fingerprints, type=pa.string()),
                    pa.array([True] * selected.num_rows, type=pa.bool_()),
                ],
                schema=SOURCE_MEMBERSHIP_SCHEMA,
            )
            writer.write_table(output)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    _publish_parquet(writer, temporary, output_path)
    return {
        **_source_file_inventory(output_path),
        "n_records": n_rows,
        "records_by_level": {str(level): counts[level] for level in sorted(counts)},
        "scope": "retrieval-eligible source records before split-specific direct heldout filtering",
    }


def _write_card_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=INDEXED_CARD_SCHEMA))
        rows.clear()


def export_indexed_cards(
    *,
    task: str,
    split_scheme: str,
    evidence_path: Path,
    catalog_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    _, by_level = _level_maps(catalog_manifest)
    descriptions = _level_descriptions(task, by_level)
    writer, temporary = _atomic_parquet_writer(output_path, INDEXED_CARD_SCHEMA)
    pending: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    n_assay_molecule_rows = 0
    n_cards = 0
    try:
        with evidence_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                evidence = json.loads(line)
                n_assay_molecule_rows += 1
                retrieval = evidence.get("assay_retrieval") or {}
                first_level = _int_or_zero(retrieval.get("first_level"))
                examples = evidence.get("source_record_examples") or []
                for card_position, example in enumerate(examples, start=1):
                    if not isinstance(example, dict):
                        raise ValueError(
                            f"non-object source_record_example at {evidence_path}:{line_number}"
                        )
                    level = _int_or_zero(example.get("evidence_family_level"))
                    if level not in by_level:
                        raise ValueError(
                            f"unknown evidence family level {level} at {evidence_path}:{line_number}"
                        )
                    family_key = _text(example.get("evidence_family"))
                    expected_family_key = by_level[level]["family_key"]
                    if family_key != expected_family_key:
                        raise ValueError(
                            f"level/family mismatch at {evidence_path}:{line_number}: "
                            f"{level}/{family_key!r} != {expected_family_key!r}"
                        )
                    pending.append(
                        {
                            "task": task,
                            "split_scheme": split_scheme,
                            "level": level,
                            "family_key": family_key,
                            "level_description": descriptions[level],
                            "legacy_family_id": by_level[level]["legacy_family_id"],
                            "assay_first_level": first_level,
                            "assay_id": _text(evidence.get("assay_chembl_id")),
                            "molecule_id": _text(evidence.get("molecule_chembl_id")),
                            "canonical_smiles": _text(evidence.get("canonical_smiles")),
                            "card_position_in_assay_molecule": card_position,
                            "source_record_count_in_assay_molecule": int(
                                evidence.get("source_record_count") or 0
                            ),
                            "card_fingerprint_sha256": _card_fingerprint(example),
                        }
                    )
                    counts[level] += 1
                    n_cards += 1
                    if len(pending) >= 50_000:
                        _write_card_rows(writer, pending)
        _write_card_rows(writer, pending)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    _publish_parquet(writer, temporary, output_path)
    return {
        **_source_file_inventory(output_path),
        "n_assay_molecule_rows": n_assay_molecule_rows,
        "n_representative_cards": n_cards,
        "cards_by_level": {str(level): counts[level] for level in sorted(counts)},
        "scope": "compact references to exact representative cards materialized in the frozen split index",
    }


def validate_card_links(
    source_membership_path: Path,
    indexed_card_paths: Iterable[Path],
) -> dict[str, int]:
    """Require every compact index-card reference to resolve to source records."""

    source_table = pq.read_table(
        source_membership_path,
        columns=["molecule_id", "card_fingerprint_sha256"],
    )
    source_keys = set(
        zip(
            source_table["molecule_id"].to_pylist(),
            source_table["card_fingerprint_sha256"].to_pylist(),
        )
    )
    n_cards = 0
    n_missing = 0
    for path in indexed_card_paths:
        for batch in ds.dataset(path, format="parquet").to_batches(
            columns=["molecule_id", "card_fingerprint_sha256"], batch_size=65_536,
        ):
            table = pa.Table.from_batches([batch])
            keys = zip(
                table["molecule_id"].to_pylist(),
                table["card_fingerprint_sha256"].to_pylist(),
            )
            for key in keys:
                n_cards += 1
                n_missing += key not in source_keys
    if n_missing:
        raise ValueError(
            f"{n_missing}/{n_cards} indexed cards do not resolve to source membership"
        )
    return {
        "n_unique_source_card_keys": len(source_keys),
        "n_indexed_cards_checked": n_cards,
        "n_missing_source_card_keys": 0,
    }


def _relative_dataset_paths(value: Any, output_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _relative_dataset_paths(item, output_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_dataset_paths(item, output_dir) for item in value]
    if isinstance(value, str):
        path = Path(value)
        try:
            return str(path.relative_to(output_dir))
        except ValueError:
            return value
    return value


def _readme() -> str:
    return """# Current Starling level-record collaborator dataset

This Git-tracked snapshot has two deliberately separate tables for each task.
It is stored as ordinary Parquet files so a collaborator receives it with a
normal repository clone; no archive unpacking or Git LFS installation is
required. Only the current adopted snapshot belongs here.

The current public naming model is `source_group_id -> family_key -> level`.
`legacy_family_id` is included only to locate older catalogs and traces; it is
not a second semantic classification.

- `source_record_level_membership.parquet` (or the same-named directory without
  the suffix, containing ordinary Parquet parts when the ledger exceeds 90 MB) contains every current
  `retrieval_eligible=True` source record whose purity-overlay `group_id` maps
  to a current progressive level. This is the static, pre-split level ledger.
- `<split>/indexed_representative_cards.parquet` (or a directory of parts above
  90 MB) contains compact references to
  the exact record cards materialized in that split's frozen assay-molecule
  index after direct heldout filtering. The index keeps at most three cards per
  assay×molecule; card text stays normalized in the source ledger rather than
  being duplicated in every split table.

Join the two tables on `molecule_id` plus `card_fingerprint_sha256` when source
record lineage is needed. More than one canonical record can have the same
visible card payload; all matching source rows remain in the membership table.
Export fails unless every indexed-card key resolves to at least one source row.

Read `manifest.json` first. It records row counts and SHA-256 hashes for every
file and for the frozen inputs. Pass each table's `path` to `pyarrow.parquet.read_table`;
both a single file and a directory of parts have the same schema. Sharded tables
list individual file hashes under `parts`. A physical assay can occur at several record
levels; `assay_first_level` is catalog metadata, not a retrieval gate.

Regenerate the complete directory from the repository root with:

```bash
python -m tools.chembl_tool.paper_experiments.export_current_starling_level_records
```

The exporter first verifies every current input against the frozen retrieval
contract. Commit a regenerated snapshot only when that current lineage changes;
do not keep parallel versioned copies of this directory.

These are candidate surfaces, not the cards selected for a particular query.
Exact model-visible cards for one completed query are stored in that run's
`<task>/queries/query_idxNNNNN/levels/level_N/prepared.json` under
`active_evidence`; `new_card_ids` identifies what was newly unlocked at that
level.
"""


def export_dataset(
    *,
    artifact_root: Path,
    records_root: Path,
    output_dir: Path,
    tasks: Iterable[str],
    splits: Iterable[str],
) -> dict[str, Any]:
    selected_tasks = tuple(tasks)
    selected_splits = tuple(splits)
    if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve() and (
        set(selected_tasks) != set(TASKS) or set(selected_splits) != {"scaffold", "random"}
    ):
        raise ValueError("Partial exports require a separate --output-dir; preserve the complete current dataset")
    verification = rebuild.verify(artifact_root, records_root=records_root, tasks=selected_tasks)
    contract = rebuild._contract()
    overlays = rebuild._overlay_paths(artifact_root, contract, records_root=records_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": EXPORT_VERSION,
        "exporter": str(Path(__file__).resolve().relative_to(rebuild.PROJECT_ROOT)),
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "current_retrieval_contract": str(
            rebuild.CONTRACT_PATH.relative_to(rebuild.PROJECT_ROOT)
        ),
        "current_retrieval_contract_sha256": sha256_file(rebuild.CONTRACT_PATH),
        "verified_current_inputs": bool(verification.get("ok")),
        "interpretation": {
            "source_membership": (
                "all retrieval-eligible overlay records assigned to a current level, "
                "before split-specific direct heldout filtering"
            ),
            "indexed_representative_cards": (
                "compact references to exact representative cards materialized in "
                "each split index; join to source membership for the card fields"
            ),
            "per_query_selected_cards": (
                "not included; read active_evidence/new_card_ids in each run's prepared.json"
            ),
        },
        "tasks": {},
    }
    for task in selected_tasks:
        task_contract = contract["tasks"][task]
        task_dir = output_dir / task
        config_module = importlib.import_module(_TASK_CONFIG_MODULES[task])
        config_path = Path(str(config_module.__file__)).resolve()
        overlay = overlays[task] / "records.parquet"
        catalog_dir = artifact_root / str(task_contract["catalog"])
        catalog_manifest = catalog_dir / "manifest.json"
        task_report: dict[str, Any] = {
            "level_description_source": str(config_path.relative_to(rebuild.PROJECT_ROOT)),
            "level_description_source_sha256": sha256_file(config_path),
            "source_membership": export_source_membership(
                task=task,
                records_path=local_input(overlay),
                catalog_manifest=catalog_manifest,
                output_path=task_dir / "source_record_level_membership.parquet",
            ),
            "indices": {},
        }
        indexed_card_paths: list[Path] = []
        for split_scheme in selected_splits:
            index_contract = task_contract["indices"][split_scheme]
            evidence_path = (
                artifact_root
                / str(index_contract["path"])
                / "assay_molecule_evidence.jsonl"
            )
            indexed_output = (
                task_dir / split_scheme / "indexed_representative_cards.parquet"
            )
            task_report["indices"][split_scheme] = export_indexed_cards(
                task=task,
                split_scheme=split_scheme,
                evidence_path=local_input(evidence_path),
                catalog_manifest=catalog_manifest,
                output_path=indexed_output,
            )
            indexed_card_paths.append(Path(task_report["indices"][split_scheme]["path"]))
        task_report["card_link_validation"] = validate_card_links(
            Path(task_report["source_membership"]["path"]),
            indexed_card_paths,
        )
        report["tasks"][task] = task_report
    portable = _relative_dataset_paths(report, output_dir)
    write_json_atomic(output_dir / "manifest.json", portable)
    (output_dir / "README.md").write_text(_readme(), encoding="utf-8")
    return portable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=rebuild.DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--splits", nargs="+", choices=("scaffold", "random"), default=["scaffold", "random"]
    )
    args = parser.parse_args(argv)
    report = export_dataset(
        artifact_root=args.artifact_root,
        records_root=args.records_root,
        output_dir=args.output_dir,
        tasks=args.tasks,
        splits=args.splits,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
