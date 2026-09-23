"""Compare pinned reranker prompts with sampled, published Starling dataset rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from predict.utils.json import sha256_file

from .score_tdc_ranked_retrieval import DirectL1PromptRenderer
from .v27_safety import MODELS, SafetyV27PromptRenderer
from .v27_skin import SkinV27PromptRenderer, TRAINING_MODEL


SPLITS = ("train", "validation_ranking", "test_ranking")
SAFETY_NAMES = {"ames": "Ames", "dili": "DILI", "carcinogens": "Carcinogens"}
STARLING = Path(__file__).resolve().parents[4] / "starling_assay_transfer"


def _sample(paths: list[Path], count: int) -> list[dict]:
    groups = [
        (path, group)
        for path in paths
        for group in range(pq.ParquetFile(path).metadata.num_row_groups)
    ]
    if not groups:
        raise ValueError("Published dataset has no row groups")
    group_count = min(len(groups), 32, count)
    chosen = [groups[index * len(groups) // group_count] for index in range(group_count)]
    per_group = (count + group_count - 1) // group_count
    output = []
    for path, group in chosen:
        table = pq.ParquetFile(path).read_row_group(group)
        take = min(per_group, table.num_rows)
        positions = [index * table.num_rows // take for index in range(take)]
        output.extend(table.take(pa.array(positions)).to_pylist())
    return output[:count]


def _training_record(task: str, row: dict, smiles: str) -> dict:
    payload = json.loads(row["prompt_payload_json"])
    if task == "skin_reaction":
        return {
            "task_id": task, "progressive_level": row["level"],
            "source_id": row["source_id"], "canonical_smiles": smiles,
            "canonical_endpoint_name": row["endpoint"],
            "source_fields": payload,
            "display_measurement_text": row["canonical_measurement_text"],
            "display_unit_text": row["canonical_unit_text"],
        }
    return {
        "task_id": task, "progressive_level": row["level"],
        "canonical_smiles": smiles, "source_fields": payload,
        "canonical_endpoint_name": row["endpoint"],
        "canonical_pair_fields_json": row["pair_fields_json"],
        "canonical_measurement_text": payload.get("measurement_text"),
        "canonical_unit_text": payload.get("unit_text"),
        "canonical_measurement_scale_id": row["canonical_measurement_scale_id"],
        "canonical_category_id": row["category_id"],
        "measurement_kind": row["measurement_kind"],
    }


def audit_direct(task: str, root: Path, per_split: int) -> dict:
    dataset = root / "assay_transfer/context_conditioned/artifacts/v10/hf/v10_3_tdc" / SAFETY_NAMES[task] / "mixed_canonical"
    renderer = DirectL1PromptRenderer(task, "tdc-v1")
    coverage = Counter()
    manifest = dataset / "manifest.json"
    inputs = {str(manifest.relative_to(root)): sha256_file(manifest)}
    checked = 0
    for split in SPLITS:
        path = dataset / split / "data.parquet"
        direct_rows = [row for row in _sample([path], per_split * 3)
                       if row["source_id"] == f"tdc_{task}_v1"][:per_split]
        if len(direct_rows) != per_split:
            raise ValueError(f"Direct dataset sample is incomplete: {task}/{split}")
        for row in direct_rows:
            actual = renderer.render(
                {"drug": row["retrieval_smiles"], "Y": row["retrieval_gold_Y"]},
                {"drug": row["query_smiles"]},
            )
            if actual != row["prompt"]:
                raise ValueError(f"Direct training prompt differs: {task}/{split}/{row['pair_id']}")
            coverage[(split, str(row["retrieval_gold_Y"]))] += 1
            checked += 1
    if {label for _, label in coverage} != {"0", "1"}:
        raise ValueError(f"Direct audit missed a TDC outcome stratum: {task}")
    return {"model": task, "family": "tdc_direct", "checked": checked,
            "template_sha256": renderer.template_hash, "projection_sha256": renderer.projection_hash,
            "coverage": {"/".join(key): value for key, value in sorted(coverage.items())},
            "inputs": inputs}


def audit_record(task: str, root: Path, per_split: int) -> dict:
    record_root = root / "assay_transfer/record_level/artifacts/v27"
    dataset = (record_root / "skin_reaction/hf/combined" if task == "skin_reaction"
               else record_root / "general_safety" / task / "hf")
    records_path = (record_root / "skin_reaction/splits/records.parquet" if task == "skin_reaction"
                    else record_root / "general_safety" / task / "splits/records.parquet")
    renderer = (SkinV27PromptRenderer(training_template=True) if task == "skin_reaction"
                else SafetyV27PromptRenderer(task))
    sampled = {}
    manifest = dataset / "manifest.json"
    split_manifest = records_path.parent / "manifest.json"
    inputs = {
        str(manifest.relative_to(root)): sha256_file(manifest),
        str(split_manifest.relative_to(root)): sha256_file(split_manifest),
    }
    for split in SPLITS:
        paths = sorted((dataset / split).glob("*.parquet")) if task != "skin_reaction" else [dataset / split / "data.parquet"]
        if not paths:
            raise FileNotFoundError(dataset / split)
        sampled[split] = _sample(paths, per_split)
    ids = sorted({str(row[f"{role}_record_id"])
                  for rows in sampled.values() for row in rows for role in ("retrieval", "query")})
    sidecars = [records_path]
    if task == "skin_reaction":
        sidecars += [records_path.with_name("test_records.parquet"),
                     records_path.with_name("test_gate_records.parquet")]
    records = {}
    for sidecar in sidecars:
        for row in pq.read_table(sidecar, filters=[("record_id", "in", ids)]).to_pylist():
            key = str(row["record_id"])
            prompt_keys = ("level", "source_id", "endpoint", "prompt_payload_json",
                           "canonical_measurement_text", "canonical_unit_text")
            if key in records and any(records[key].get(name) != row.get(name) for name in prompt_keys):
                raise ValueError(f"Conflicting training record sidecars: {task}/{key}")
            records.setdefault(key, row)
    if set(records) != set(ids):
        raise ValueError(f"Training record join is incomplete: {task}")
    coverage = Counter()
    checked = 0
    for split, rows in sampled.items():
        for row in rows:
            known = _training_record(task, records[str(row["retrieval_record_id"])], row["retrieval_smiles"])
            query = _training_record(task, records[str(row["query_record_id"])], row["query_smiles"])
            actual = renderer.render_pair(known, query)
            if actual != row["prompt"]:
                raise ValueError(f"Record training prompt differs: {task}/{split}/{row['pair_id']}")
            adapted = renderer.render(known, row["query_smiles"])
            if not adapted or row["query_smiles"] not in adapted:
                raise ValueError(f"Indirect query-copy prompt is invalid: {task}/{split}/{row['pair_id']}")
            hidden_query = adapted.split("Experiment B (query measurement hidden)", 1)[1]
            if "Known reported measurement:" in hidden_query or "Measurement text:" in hidden_query:
                raise ValueError(f"Result leaked into hidden query: {task}/{split}/{row['pair_id']}")
            coverage[(split, str(query["progressive_level"]), str(records[str(row["query_record_id"])]["source_id"]))] += 1
            checked += 1
    return {"model": task, "family": "record_v27", "checked": checked,
            "template_sha256": renderer.template_hash, "projection_sha256": renderer.projection_hash,
            "coverage": {"/".join(key): value for key, value in sorted(coverage.items())},
            "inputs": inputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-root", type=Path, default=STARLING)
    parser.add_argument("--per-split", type=int, default=128)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.per_split <= 341:
        parser.error("--per-split must yield between 3 and 1023 examples per model")
    root = args.starling_root.resolve()
    results = [audit_direct(task, root, args.per_split) for task in SAFETY_NAMES]
    results += [audit_record(task, root, args.per_split) for task in (*SAFETY_NAMES, "skin_reaction")]
    report = {
        "schema_version": "assay_transfer_training_prompt_audit.v1",
        "sampling": "evenly_spaced_files_row_groups_and_rows.v1",
        "starling_root": str(root),
        "models": {**MODELS, "skin_reaction": TRAINING_MODEL},
        "results": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        table = ["family\tmodel\tchecked\tstrata\ttemplate_sha256\tprojection_sha256"]
        table.extend("\t".join(map(str, (
            row["family"], row["model"], row["checked"], len(row["coverage"]),
            row["template_sha256"], row["projection_sha256"],
        ))) for row in results)
        (args.report.parent / "prompt_audit.tsv").write_text("\n".join(table) + "\n", encoding="utf-8")
        (args.report.parent / "report.md").write_text(
            "# V27 and TDC direct prompt audit\n\n"
            f"Compared {sum(row['checked'] for row in results)} published dataset prompts "
            "byte-for-byte with the cache renderers. All comparisons passed. "
            "The separate indirect inference adaptation copies candidate fields "
            "allowed by the training query-visibility filter onto the hidden-query "
            "side. This does not make them the query's own assay context.\n\n"
            "See `prompt_audit.tsv` for counts and `prompt_audit.json` for the "
            "sampled datasets, hashes, and observed strata.\n",
            encoding="utf-8",
        )
    print(json.dumps({"checked": sum(row["checked"] for row in results),
                      "models": len(results), "report": str(args.report) if args.report else None},
                     sort_keys=True))


if __name__ == "__main__":
    main()
