"""Inventory record feature coverage and LLM workload before bulk extraction.

One streaming pass per task. The default writes counts and bounded stratified
samples only; --write-features also writes compressed normalized record inputs.
Numeric feature vectors are encoded separately from these reusable inputs.
No model requests are sent by this command.
"""

from __future__ import annotations

import argparse
import gc
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tools.chembl_tool.common.record_diversity import (
    IDENTITY_FIELDS,
    SCIENTIFIC_FIELDS,
    VERSION,
    prepare_record,
)

REGISTRY = Path("tools/chembl_tool/paper_experiments/current_starling_retrieval.json")
META = [
    "canonical_record_id",
    "source_id",
    "source_row_number",
    "molecule_id",
    "molecule_name",
    "canonical_smiles",
    "group_id",
    "retrieval_eligible",
    "raw_record_json",
    "reviewed_record_json",
    "needs_more_context",
    "support_text",
]


def scan_task(job):
    task, output, sample_limit, write_features, molecule_filter, filter_field = job
    started = time.monotonic()
    from tools.chembl_tool.common import record_diversity

    code_sha256 = hashlib.sha256(
        Path(record_diversity.__file__).read_bytes()
    ).hexdigest()
    registry = json.loads(REGISTRY.read_text())
    spec = registry["tasks"][task]
    artifact = Path(registry["artifact_root"])
    path = (
        Path(spec["source_records"])
        if spec.get("overlay_is_canonical")
        else artifact / spec["overlay"] / "records.parquet"
    )
    catalog = artifact / spec["catalog"] / "manifest.json"
    level_rows = json.loads(catalog.read_text())["levels"]
    levels = {g: int(row["level"]) for row in level_rows for g in row["source_groups"]}
    # Optional frozen retrieval scope: task -> subset -> level -> molecule values.
    scopes = {}
    if molecule_filter:
        scopes = {
            subset: {
                int(level): set(molecules) for level, molecules in selected.items()
            }
            for subset, selected in json.loads(Path(molecule_filter).read_text())[
                task
            ].items()
        }
    selected_by_level = defaultdict(set)
    for selected in scopes.values():
        for level, molecules in selected.items():
            selected_by_level[level].update(molecules)
    seen_by_level = defaultdict(set)
    subset_counts = {subset: defaultdict(Counter) for subset in scopes}
    level_values = {
        level: (
            pa.array([g for g, value in levels.items() if value == level]),
            pa.array(sorted(molecules), type=pa.string()),
        )
        for level, molecules in selected_by_level.items()
    }
    parquet = pq.ParquetFile(path)
    columns = [
        c
        for c in parquet.schema_arrow.names
        if c in SCIENTIFIC_FIELDS or c in IDENTITY_FIELDS or c in META
    ]
    raw_tables = {}
    stats, samples = {}, defaultdict(list)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    for batch in parquet.iter_batches(
        batch_size=4096, columns=columns, use_threads=False
    ):
        table = pa.Table.from_batches([batch])
        table = table.filter(
            pc.and_(
                pc.fill_null(table["retrieval_eligible"], False),
                pc.is_in(table["group_id"], value_set=pa.array(list(levels))),
            )
        )
        if molecule_filter:
            mask = pa.array([False] * table.num_rows, type=pa.bool_())
            for groups, molecules in level_values.values():
                mask = pc.or_(
                    mask,
                    pc.and_(
                        pc.is_in(table["group_id"], value_set=groups),
                        pc.is_in(table[filter_field], value_set=molecules),
                    ),
                )
            table = table.filter(mask)
        if not table.num_rows:
            continue
        rows = table.to_pylist()
        raw_rows = [None] * len(rows)
        if task == "ames":
            positions = defaultdict(list)
            for i, row in enumerate(rows):
                positions[row["source_id"]].append(i)
            for source, indices in positions.items():
                if source not in raw_tables:
                    raw_path = Path("data/starling_data/ames/raw_v1") / (
                        source + ".parquet"
                    )
                    names = pq.read_schema(raw_path).names
                    raw_tables[source] = pq.read_table(
                        raw_path,
                        columns=[
                            c
                            for c in names
                            if c in SCIENTIFIC_FIELDS
                            or c in IDENTITY_FIELDS
                            or c in META
                        ],
                    )
                ordinals = [int(rows[i]["source_row_number"]) for i in indices]
                data = (
                    raw_tables[source]
                    .take(pa.array(ordinals, type=pa.int64()))
                    .to_pylist()
                )
                for i, native in zip(indices, data):
                    raw_rows[i] = native
        feature_rows = []
        for row, raw in zip(rows, raw_rows):
            prepared = prepare_record(row, raw)
            level = levels[row["group_id"]]
            if molecule_filter:
                seen_by_level[level].add(row[filter_field])
                for subset, selected in scopes.items():
                    if row[filter_field] in selected.get(level, set()):
                        subset_counts[subset][level]["rows"] += 1
                        subset_counts[subset][level][prepared["route"]] += 1
            source = row["source_id"]
            key = f"L{level}|{source}"
            if key not in stats:
                stats[key] = {
                    "rows": 0,
                    "routes": Counter(),
                    "reasons": Counter(),
                    "groups": Counter(),
                    "fields": Counter(),
                    "outcomes": Counter(),
                    "input_chars": 0,
                }
            st = stats[key]
            st["rows"] += 1
            st["routes"][prepared["route"]] += 1
            st["reasons"].update(
                prepared["llm_reasons"] if prepared["needs_llm"] else []
            )
            st["groups"].update(g for g, values in prepared["groups"].items() if values)
            st["fields"].update(
                k for values in prepared["groups"].values() for k in values
            )
            st["outcomes"].update(prepared["outcome_kinds"])
            if prepared["needs_llm"]:
                st["input_chars"] += len(
                    json.dumps(
                        prepared["input"], ensure_ascii=False, separators=(",", ":")
                    )
                )
            # Bottom-k hashes are order-independent; keep both routes for false-negative review.
            bucket = key + "|" + prepared["route"]
            rank = hashlib.blake2b(
                row["canonical_record_id"].encode(), digest_size=8
            ).hexdigest()
            pool = samples[bucket]
            if len(pool) < sample_limit or rank < pool[-1][0]:
                pool.append(
                    (
                        rank,
                        {"task": task, "level": level, "source_id": source, **prepared},
                    )
                )
                pool.sort(key=lambda x: x[0])
                del pool[sample_limit:]
            if write_features:
                feature_rows.append(
                    {
                        "record_id": prepared["record_id"],
                        "source_id": source,
                        "level": level,
                        "route": prepared["route"],
                        "payload_json": json.dumps(
                            prepared, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
        if feature_rows:
            t = pa.Table.from_pylist(feature_rows)
            if writer is None:
                writer = pq.ParquetWriter(
                    output / (task + ".features.parquet"), t.schema, compression="zstd"
                )
            writer.write_table(t)
    if writer is not None:
        writer.close()
    report = {
        "version": VERSION,
        "implementation_sha256": code_sha256,
        "task": task,
        "records": str(path),
        "records_sha256": spec["overlay_records_sha256"],
        "catalog": str(catalog),
        "scope": (
            "eligible source records for frozen selected molecule-level pairs; union counted once"
            if molecule_filter
            else "all eligible mapped source records before split heldout filtering; each row counted once"
        ),
        "molecule_filter": str(molecule_filter) if molecule_filter else None,
        "molecule_filter_field": filter_field,
        "molecule_filter_sha256": hashlib.sha256(
            Path(molecule_filter).read_bytes()
        ).hexdigest()
        if molecule_filter
        else None,
        "subset_counts": subset_counts,
        "selected_molecules_by_level": {
            level: len(molecules) for level, molecules in selected_by_level.items()
        },
        "unmatched_selected_values": {
            level: sorted(molecules - seen_by_level[level])
            for level, molecules in selected_by_level.items()
            if molecules - seen_by_level[level]
        },
        "counts_are_routing_estimates_not_semantic_ground_truth": True,
        "strata": stats,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "samples": [value for pool in samples.values() for _, value in pool],
    }
    (output / (task + ".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "task": task,
        "rows": sum(s["rows"] for s in stats.values()),
        "llm_candidates": sum(s["routes"]["llm_candidate"] for s in stats.values()),
        "structured_candidates": sum(
            s["routes"]["structured_candidate"] for s in stats.values()
        ),
        "unresolved_no_text": sum(
            s["routes"]["unresolved_no_text"] for s in stats.values()
        ),
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--molecule-filter",
        type=Path,
        help="JSON: task -> subset -> level -> selected molecule values",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "bbb_martins",
            "bioavailability_ma",
            "skin_reaction",
            "ames",
            "dili",
            "carcinogens",
        ],
    )
    parser.add_argument(
        "--molecule-filter-field",
        choices=["canonical_smiles", "molecule_id"],
        default="canonical_smiles",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--samples-per-stratum-route", type=int, default=2)
    parser.add_argument("--write-features", action="store_true")
    args = parser.parse_args()
    if args.samples_per_stratum_route < 1:
        parser.error("sample count must be positive")
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=gc.disable) as pool:
        rows = list(
            pool.map(
                scan_task,
                [
                    (
                        t,
                        str(args.output_dir),
                        args.samples_per_stratum_route,
                        args.write_features,
                        str(args.molecule_filter) if args.molecule_filter else None,
                        args.molecule_filter_field,
                    )
                    for t in args.tasks
                ],
            )
        )
    summary = {
        "version": VERSION,
        "tasks": rows,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "source_rows": sum(r["rows"] for r in rows),
        "llm_candidate_rows": sum(r["llm_candidates"] for r in rows),
        "deduplication": "not applied: LLM candidate rows are a conservative request upper bound before exact-input caching",
        "model_calls": 0,
        "features_written": args.write_features,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
