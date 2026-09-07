"""Build record-family membership plus physical-assay provenance for Starling.

``first_level`` is an assay-level ordering and coverage-summary field, not a
unique family assignment.  Each source group keeps its own family level in
``source_families`` so mechanism-tagged indexes can expose individual records
only at their proper level. Physical assay identity remains useful for card
provenance, exact deduplication, and diversity-aware card selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.assay_catalog import assay_id, assay_unit
from tools.chembl_tool.common.build_runtime import (
    sha256_file,
    local_input,
    local_workdir,
    publish_file,
    write_jsonl_local,
    build_signature,
    reusable,
)


VERSION = "starling_physical_assay_family_catalog.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/family_catalogs"
)
MECHANISM_OUTPUT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/"
    "family_catalogs_mechanism_tagged_v1"
)
TASKS = {
    "ames": {
        "records": "data/starling_data/ames/canonical_v1/records.parquet",
        "config_module": "tools.chembl_tool.tasks.ames.experiment_config",
        "config_name": "STARLING",
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "output_name": "ames",
    },
    "bbb_martins": {
        "records": "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/bbb_source_family_purity_v6/records.parquet",
        "config_module": "tools.chembl_tool.tasks.bbb_martins.experiment_config",
        "config_name": "STARLING_SOURCE_PURITY",
        "output_root": str(MECHANISM_OUTPUT_ROOT),
        "output_name": "bbb_martins_source_purity_v6",
    },
    "bioavailability_ma": {
        "records": "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/bioavailability_source_family_purity_legacy_record_supported_v2_vote_pure_v1/records.parquet",
        "config_module": "tools.chembl_tool.tasks.bioavailability_ma.experiment_config",
        "config_name": "STARLING",
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "output_name": "bioavailability_ma_legacy_record_supported_v2_vote_pure_v1",
    },
    "skin_reaction": {
        "records": "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/source_family_purity_v5/skin_reaction/records.parquet",
        "config_module": "tools.chembl_tool.tasks.skin_reaction.experiment_config",
        "config_name": "STARLING",
        "output_root": str(MECHANISM_OUTPUT_ROOT),
        "output_name": "skin_reaction_source_purity_v5",
    },
    "clintox": {
        "records": "outputs/paper/starling_assay_relevance_all_v1/clintox/assay_catalog.jsonl",
        "config_module": "tools.chembl_tool.tasks.clintox.experiment_config",
        "config_name": "STARLING",
        "output_root": str(DEFAULT_OUTPUT_ROOT),
    },
}

_CLINTOX_SOURCE_LEVELS = {
    "clinical_trial_failure": 1,
    "nonclinical_in_vivo_toxicity": 3,
    "organ_specific_toxicity": 4,
    "genotoxicity_carcinogenicity": 5,
    "cellular_stress": 6,
    "general_cytotoxicity": 7,
    "off_target_ddi_exposure": 8,
}


def _build_clintox_catalog(
    catalog_path: Path,
    config_name: str = "STARLING",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = getattr(
        importlib.import_module(TASKS["clintox"]["config_module"]), config_name
    )
    source_rows = [json.loads(line) for line in catalog_path.open() if line.strip()]
    rows = []
    for source in source_rows:
        level = _CLINTOX_SOURCE_LEVELS[str(source["source_id"])]
        family = config.mechanism_groups[level - 1]
        rows.append(
            {
                "catalog_version": VERSION,
                "task": "clintox",
                "assay_id": str(source["assay_id"]),
                "assay_context": str(source["assay_context"]),
                "selection_rank": 0,
                "first_level": level,
                "first_family_id": family.output_group_id,
                "first_endpoint_group": family.family_key,
                "family_levels": [level],
                "family_ids": [family.output_group_id],
                "family_endpoint_groups": [family.family_key],
                "source_groups": list(family.source_group_ids),
                "source_families": [
                    {
                        "source_group_id": source_group,
                        "level": level,
                        "family_id": family.output_group_id,
                        "endpoint_group": family.family_key,
                    }
                    for source_group in family.source_group_ids
                ],
                "record_count": int(source.get("record_count") or 0),
            }
        )
    clinical_family = config.mechanism_groups[1]
    rows.append(
        {
            "catalog_version": VERSION,
            "task": "clintox",
            "assay_id": "STARLING_CLINTOX_CLINICAL_CONTEXT",
            "assay_context": "clinical human safety context",
            "selection_rank": 0,
            "first_level": 2,
            "first_family_id": clinical_family.output_group_id,
            "first_endpoint_group": clinical_family.family_key,
            "family_levels": [2],
            "family_ids": [clinical_family.output_group_id],
            "family_endpoint_groups": [clinical_family.family_key],
            "source_groups": list(clinical_family.source_group_ids),
            "source_families": [
                {
                    "source_group_id": source_group,
                    "level": 2,
                    "family_id": clinical_family.output_group_id,
                    "endpoint_group": clinical_family.family_key,
                }
                for source_group in clinical_family.source_group_ids
            ],
            "record_count": 0,
        }
    )
    rows.sort(key=lambda row: (row["first_level"], row["assay_id"]))
    for selection_rank, row in enumerate(rows, start=1):
        row["selection_rank"] = selection_rank
    level_counts = Counter(int(row["first_level"]) for row in rows)
    cumulative = 0
    levels = []
    for level, family in enumerate(config.mechanism_groups, start=1):
        cumulative += level_counts[level]
        levels.append(
            {
                "level": level,
                "family_id": family.output_group_id,
                "endpoint_group": family.family_key,
                "source_groups": list(family.source_group_ids),
                "new_physical_assays": level_counts[level],
                "cumulative_physical_assays": cumulative,
            }
        )
    return rows, {
        "catalog_version": VERSION,
        "task": "clintox",
        "records": str(catalog_path.resolve()),
        "records_sha256": sha256_file(catalog_path),
        "assay_definition": "source-native assay field; one explicit clinical-context unit",
        "overlap_policy": "source-native assay ids are disjoint; assign each to one family level",
        "n_allowed_source_records": sum(
            int(row.get("record_count") or 0) for row in rows
        ),
        "n_physical_assays": len(rows),
        "n_multi_family_assays": 0,
        "levels": levels,
        "overlap_patterns": {
            str(level): level_counts[level] for level in sorted(level_counts)
        },
        "llm_visible": False,
    }


def build_catalog(
    task: str,
    records_path: Path,
    *,
    config_name: str = "STARLING",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task == "clintox":
        return _build_clintox_catalog(records_path, config_name)
    config = getattr(importlib.import_module(TASKS[task]["config_module"]), config_name)
    frame = pq.read_table(
        local_input(records_path),
        columns=[
            "group_id",
            "canonical_assay_context",
            "canonical_endpoint_name",
            "retrieval_eligible",
        ],
    ).to_pandas()
    frame = frame.loc[frame["retrieval_eligible"].eq(True)].copy()  # noqa: E712
    available_groups = set(frame["group_id"].dropna().astype(str))
    levels = []
    source_group_to_levels: dict[str, list[int]] = defaultdict(list)
    for level, spec in enumerate(config.mechanism_groups, start=1):
        source_groups = spec.resolve(available_groups)
        if not source_groups:
            raise ValueError(
                f"{task} family {spec.family_key} resolves to no source groups"
            )
        levels.append(
            {
                "level": level,
                "family_id": spec.output_group_id,
                "endpoint_group": spec.family_key,
                "source_groups": list(source_groups),
            }
        )
        for source_group in source_groups:
            source_group_to_levels[source_group].append(level)

    ambiguous_source_groups = {
        source_group: family_levels
        for source_group, family_levels in source_group_to_levels.items()
        if len(family_levels) != 1
    }
    if ambiguous_source_groups:
        raise ValueError(
            f"{task} source groups map to multiple progressive families: "
            f"{ambiguous_source_groups}"
        )

    allowed_groups = set(source_group_to_levels)
    frame = frame.loc[frame["group_id"].isin(allowed_groups)].copy()
    units = [
        assay_unit(context, endpoint)[0]
        for context, endpoint in zip(
            frame["canonical_assay_context"],
            frame["canonical_endpoint_name"],
            strict=True,
        )
    ]
    frame["assay_context"] = units
    frame["assay_id"] = [assay_id(task, unit) for unit in units]

    rows = []
    overlap_patterns: Counter[tuple[int, ...]] = Counter()
    grouped = {}
    for stable_assay_id, context, source_group in zip(
        frame["assay_id"], frame["assay_context"], frame["group_id"], strict=True
    ):
        if stable_assay_id not in grouped:
            grouped[stable_assay_id] = [str(context), set(), 0]
        entry = grouped[stable_assay_id]
        entry[1].add(str(source_group))
        entry[2] += 1
    for stable_assay_id, (context, group_set, record_count) in grouped.items():
        source_groups = sorted(group_set)
        family_levels = sorted(
            {
                level
                for source_group in source_groups
                for level in source_group_to_levels[source_group]
            }
        )
        overlap_patterns[tuple(family_levels)] += 1
        earliest = family_levels[0]
        source_families = [
            {
                "source_group_id": source_group,
                "level": level,
                "family_id": levels[level - 1]["family_id"],
                "endpoint_group": levels[level - 1]["endpoint_group"],
            }
            for source_group in source_groups
            for level in source_group_to_levels[source_group]
        ]
        rows.append(
            {
                "catalog_version": VERSION,
                "task": task,
                "assay_id": str(stable_assay_id),
                "assay_context": context,
                "selection_rank": 0,
                "first_level": earliest,
                "first_family_id": levels[earliest - 1]["family_id"],
                "first_endpoint_group": levels[earliest - 1]["endpoint_group"],
                "family_levels": family_levels,
                "family_ids": [
                    levels[level - 1]["family_id"] for level in family_levels
                ],
                "family_endpoint_groups": [
                    levels[level - 1]["endpoint_group"] for level in family_levels
                ],
                "source_groups": source_groups,
                "source_families": source_families,
                "record_count": record_count,
            }
        )
    rows.sort(key=lambda row: (row["first_level"], row["assay_id"]))
    for selection_rank, row in enumerate(rows, start=1):
        row["selection_rank"] = selection_rank

    level_counts = Counter(int(row["first_level"]) for row in rows)
    cumulative = 0
    for level in levels:
        cumulative += level_counts[level["level"]]
        level["new_physical_assays"] = level_counts[level["level"]]
        level["cumulative_physical_assays"] = cumulative
    manifest = {
        "catalog_version": VERSION,
        "task": task,
        "config_name": config_name,
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "assay_definition": (
            "canonical assay context; canonical endpoint is used only as fallback "
            "when assay context is missing"
        ),
        "overlap_policy": "assign each physical assay to its earliest cumulative family level",
        "progressive_overlap_policy": (
            "source records retain their own family levels; first_level is only "
            "assay catalog ordering and coverage summary in the v8 progressive view"
        ),
        "family_assignment_unit": "source_record",
        "assay_identity_role": "provenance, exact deduplication, and card diversity",
        "first_level_role": (
            "legacy cumulative-family assignment; catalog ordering and physical-assay "
            "coverage summary only in the v8 progressive view"
        ),
        "n_allowed_source_records": int(len(frame)),
        "n_physical_assays": len(rows),
        "n_multi_family_assays": sum(len(row["family_levels"]) > 1 for row in rows),
        "levels": levels,
        "overlap_patterns": {
            "+".join(map(str, pattern)): count
            for pattern, count in sorted(overlap_patterns.items())
        },
        "llm_visible": False,
    }
    return rows, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS)
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Override the task-specific current output root. Without this flag, "
            "each task writes to the exact catalog root used by the current runner."
        ),
    )
    parser.add_argument(
        "--records",
        help="Override the records parquet when exactly one task is selected.",
    )
    parser.add_argument(
        "--output-name",
        help="Override the output directory name when exactly one task is selected.",
    )
    parser.add_argument(
        "--config-name",
        default="",
        help=(
            "Override the task-specific current SourceExperimentConfig attribute. "
            "Without this flag, each task uses its current catalog config."
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if (args.records or args.output_name) and len(args.tasks) != 1:
        parser.error("--records/--output-name require exactly one --tasks value")
    for task in args.tasks:
        records_path = Path(args.records or TASKS[task]["records"])
        config_name = args.config_name or str(TASKS[task]["config_name"])
        config_module = importlib.import_module(TASKS[task]["config_module"])
        output_root = Path(args.output_root or TASKS[task]["output_root"])
        output_dir = output_root / str(
            args.output_name or TASKS[task].get("output_name") or task
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = output_dir / "family_assays.jsonl"
        manifest_path = output_dir / "manifest.json"
        signature = build_signature(
            [records_path, Path(__file__), Path(config_module.__file__)],
            {"task": task, "config_name": config_name},
        )
        if reusable(manifest_path, signature, [(catalog_path, "catalog_sha256")]):
            print(f"[{task}] reused verified family catalog", flush=True)
            continue
        rows, manifest = build_catalog(task, records_path, config_name=config_name)
        with local_workdir() as staging:
            local_catalog = staging / catalog_path.name
            write_jsonl_local(local_catalog, rows, workers=min(args.workers, 16))
            catalog_hash = publish_file(local_catalog, catalog_path)
        manifest.update(
            {
                "catalog": str(catalog_path.resolve()),
                "catalog_sha256": catalog_hash,
                "build_inputs": signature,
            }
        )
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({"task": task, **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
