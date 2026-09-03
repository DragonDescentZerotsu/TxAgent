"""Build bounded, model-free ClinTox Stage-2 review inputs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.clustered_auxiliary_mapping import (
    DEFAULT_NULL_LIKE,
    cluster_values,
    embed_values,
)
from tools.chembl_tool.common.starling.build_endpoint_unit_profile import (
    MIN_PROFILE_ROWS,
    profile_lookup,
    render_profile_block,
)
from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_scalar,
    file_sha256,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
)
from tools.chembl_tool.tasks.clintox.starling_measurement_resolution import (
    DEFAULT_CLEANED_RECORDS,
    DEFAULT_PROFILE_PATH,
    SOURCE_IDS,
    prompt_row_fields,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import POLICY
from tools.chembl_tool.tasks.clintox.starling_source import DEFAULT_DATA_ROOT


OUTPUT_ROOT = Path(__file__).resolve().parent / "data_processing/stage2_review_v1"
DEFAULT_GOLD = OUTPUT_ROOT / "measurement_resolution_gold.jsonl"
GOLD_CANDIDATES = 50
CLUSTER_FREQUENT = 500
CLUSTER_TAIL = 1_500
CLUSTER_TARGET = 50
CLUSTER_MAX = 75
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", re.I)
_NAMESPACES = {
    "human_clinical_toxicity": ("toxicity_category", "clinical_context"),
    "nonclinical_in_vivo_toxicity": ("evidence_type", "animal_context"),
    "organ_specific_toxicity": ("toxicity_endpoint", "organ_system"),
    "genotoxicity_carcinogenicity": ("endpoint", "evidence_category", "assay_type"),
    "cellular_stress": ("stress_endpoint", "evidence_basis"),
    "general_cytotoxicity": ("endpoint_type", "assay_method"),
    "off_target_ddi_exposure": ("result_metric", "evidence_type"),
}
_DIAGNOSTIC_FIELDS = {
    "human_clinical_toxicity": (
        "molecule_name", "smiles", "toxicity_outcome", "clinical_context",
        "qualifying_conditions",
    ),
    "nonclinical_in_vivo_toxicity": (
        "molecule_name", "smiles", "observed_effect", "animal_context",
        "exposure_context",
    ),
    "organ_specific_toxicity": (
        "molecule_name", "smiles", "organ_system", "effect_status",
        "evidence_context", "biological_system",
    ),
    "genotoxicity_carcinogenicity": (
        "molecule_name", "smiles", "evidence_category", "assay_type",
        "study_context", "biological_system",
    ),
    "cellular_stress": (
        "molecule_name", "smiles", "evidence_basis", "biological_model",
        "mechanistic_effect",
    ),
    "general_cytotoxicity": (
        "molecule_name", "smiles", "assay_method", "cell_model",
        "test_concentration", "exposure_time_h",
    ),
    "off_target_ddi_exposure": (
        "molecule_name", "smiles", "evidence_type", "target_or_endpoint",
        "assay_context", "target_identifier",
    ),
}
_STRATA = (
    "scientific_notation",
    "range_or_censored",
    "percent_or_ratio",
    "multi_quantity",
    "numeric_prose",
    "other",
)
_STRATUM_QUOTA = dict(zip(_STRATA, (8, 8, 8, 8, 9, 9), strict=True))


def build_gold_candidates(cleaned_path: Path, output_path: Path) -> dict[str, Any]:
    columns = {
        "cleaned_record_id",
        "source_id",
        "canonical_endpoint_name",
        "measurement_resolution_route",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "support_text",
    }
    available = set(pq.read_schema(cleaned_path).names)
    heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    overall: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    candidate_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(cleaned_path).iter_batches(
        columns=sorted(columns & available), batch_size=50_000
    ):
        for row in batch.to_pylist():
            source = str(row.get("source_id") or "")
            route_bucket = str(row.get("measurement_resolution_route") or "")
            if source not in SOURCE_IDS or route_bucket not in {
                "accept",
                "categorical",
                "extract",
                "reject",
            }:
                continue
            record_id = str(row["cleaned_record_id"])
            stratum = _measurement_stratum(row.get("measurement_text"))
            payload = {
                "audit_case_id": f"clintox:{record_id}",
                "source_id": source,
                "stratum": stratum,
                "route_bucket": route_bucket,
                "input": {
                    field: row.get(field) for field in prompt_row_fields(source)
                },
                "expected": None,
                "review_status": "pending_manual_label",
            }
            rank = int(hashlib.sha256(record_id.encode()).hexdigest(), 16)
            _keep_smallest(heaps[(source, stratum)], rank, record_id, payload, 32)
            _keep_smallest(overall[source], rank, record_id, payload, 300)
            candidate_counts[source] += 1
            stratum_counts[f"{source}:{stratum}"] += 1

    selected: list[dict[str, Any]] = []
    for source in SOURCE_IDS:
        source_rows: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for stratum in _STRATA:
            rows = _heap_rows(heaps[(source, stratum)])[: _STRATUM_QUOTA[stratum]]
            source_rows.extend(rows)
            selected_ids.update(row["audit_case_id"] for row in rows)
        for row in _heap_rows(overall[source]):
            if len(source_rows) >= GOLD_CANDIDATES:
                break
            if row["audit_case_id"] not in selected_ids:
                source_rows.append(row)
                selected_ids.add(row["audit_case_id"])
        if len(source_rows) != GOLD_CANDIDATES:
            raise ValueError(f"{source} produced only {len(source_rows)} gold candidates")
        selected.extend(sorted(source_rows, key=lambda row: row["audit_case_id"]))

    manifest = {
        "corpus_version": "clintox_measurement_resolution_gold_candidates.v1",
        "task_id": "clintox",
        "cases": len(selected),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in selected).items())),
        "route_counts": dict(sorted(Counter(row["route_bucket"] for row in selected).items())),
        "candidate_source_counts": dict(sorted(candidate_counts.items())),
        "candidate_stratum_counts": dict(sorted(stratum_counts.items())),
        "selection": "50_per_source_measurement_form_stratified_stable_hash",
        "labelled_before_model_run": False,
        "status": "pending_manual_label; not a gold fixture",
        "cleaned_records": {
            "path": str(cleaned_path),
            "sha256": file_sha256(cleaned_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest


def build_endpoint_review_cards(
    cleaned_path: Path,
    candidates_path: Path,
    profile_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Render the frozen endpoint view used for blind gold review."""
    lines = [json.loads(line) for line in candidates_path.read_text().splitlines() if line]
    candidate_manifest, candidates = lines[0], lines[1:]
    candidate_ids = {
        row["audit_case_id"].split(":", 1)[1] for row in candidates
    }
    columns = {
        "cleaned_record_id", "source_id", "canonical_endpoint_name",
        *(field for fields in _DIAGNOSTIC_FIELDS.values() for field in fields),
        *(field for source in SOURCE_IDS for field in prompt_row_fields(source)),
        *(field for spec in POLICY.controlled_measurements for field in spec.input_fields),
    }
    available = set(pq.read_schema(cleaned_path).names)
    records: dict[str, dict[str, Any]] = {}
    for batch in pq.ParquetFile(cleaned_path).iter_batches(
        columns=sorted(columns & available), batch_size=50_000
    ):
        for record in batch.to_pylist():
            record_id = str(record.get("cleaned_record_id") or "")
            if record_id in candidate_ids:
                if record_id in records:
                    raise ValueError(f"duplicate cleaned_record_id {record_id}")
                records[record_id] = record
    missing = candidate_ids - records.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold candidates are absent from Stage 01")

    profile = json.loads(profile_path.read_text())
    profiles = profile_lookup(profile)
    controlled_fields = {
        spec.source_id: spec.input_fields for spec in POLICY.controlled_measurements
    }
    cards: list[dict[str, Any]] = []
    for candidate in candidates:
        record_id = candidate["audit_case_id"].split(":", 1)[1]
        record = records[record_id]
        source = candidate["source_id"]
        endpoint = str(record.get("canonical_endpoint_name") or "missing_endpoint")
        entry = profiles.get((source, endpoint))
        profile_block = (
            render_profile_block(entry, exclude_ids=candidate_ids)
            if entry and int(entry.get("resolved_rows") or 0) >= MIN_PROFILE_ROWS
            else f"[endpoint: {endpoint} - no reliable deterministic summary]"
        )
        cards.append(
            {
                "audit_case_id": candidate["audit_case_id"],
                "source_id": source,
                "canonical_endpoint_name": endpoint,
                "resolver_input": candidate["input"],
                "endpoint_profile_block": profile_block,
                "controlled_inputs": {
                    field: record.get(field)
                    for field in controlled_fields.get(source, ())
                },
                "diagnostic_context": {
                    field: record.get(field) for field in _DIAGNOSTIC_FIELDS[source]
                },
            }
        )

    manifest = {
        "corpus_version": "clintox_endpoint_review_cards.v1",
        "task_id": "clintox",
        "cases": len(cards),
        "candidate_ledger_sha256": file_sha256(candidates_path),
        "cleaned_records_sha256": file_sha256(cleaned_path),
        "endpoint_profile_sha256": file_sha256(profile_path),
        "all_candidate_ids_excluded_from_profile_examples": True,
        "observed_route_hidden": True,
        "resolver_labels_must_use_resolver_input_and_endpoint_profile_only": True,
        "diagnostic_context_is_scope_audit_only": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest


def score_gold_replay(gold_path: Path, mapping_path: Path, output_path: Path) -> dict[str, Any]:
    """Score a frozen extraction replay without changing its gold labels."""
    lines = [json.loads(line) for line in gold_path.read_text().splitlines() if line]
    gold_manifest, all_cases = lines[0], lines[1:]
    cases = [row for row in all_cases if row["route_bucket"] == "extract"]
    gold = {row["audit_case_id"].split(":", 1)[1]: row for row in cases}
    predictions = {
        str(row["cleaned_record_id"]): row
        for row in pq.read_table(mapping_path).to_pylist()
    }
    if gold.keys() != predictions.keys():
        raise ValueError("gold replay IDs do not exactly match current extraction gold")

    scored: list[dict[str, Any]] = []
    for record_id, case in gold.items():
        prediction = predictions[record_id]
        expected = case["expected"]
        returned = json.loads(prediction["measurements_json"])
        status_exact = prediction["status"] == expected["status"]
        expected_pairs = [
            {"measurement": row["measurement"], "unit": row["unit"]}
            for row in expected["measurements"]
        ]
        raw_pair_exact = status_exact and returned == expected_pairs
        canonical_exact = status_exact
        if canonical_exact and expected["status"] == "ok":
            if len(returned) != len(expected["measurements"]):
                canonical_exact = False
            else:
                for observed, wanted in zip(returned, expected["measurements"], strict=True):
                    pair = normalize_measurement_and_unit(
                        observed["measurement"], observed["unit"], task="clintox"
                    )
                    scalar = parse_point_measurement(pair.canonical_measurement).value
                    if (
                        scalar is None
                        or not math.isclose(
                            scalar,
                            float(wanted["expected_scalar"]),
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        or pair.canonical_unit != wanted["expected_canonical_unit"]
                    ):
                        canonical_exact = False
                        break
        error_class = None
        text = str(case["input"].get("measurement_text") or "")
        if not canonical_exact:
            if expected["status"] == "unsure" and prediction["status"] == "ok" and re.search(
                r"(?:\bto\b|\brange\b|[<>≤≥]|\d\s*[-–]\s*\d)", text, re.I
            ):
                error_class = "bound_or_range_promoted"
            elif expected["status"] == "unavailable" and prediction["status"] == "ok":
                error_class = "condition_or_other_endpoint_promoted"
            elif expected["status"] == prediction["status"] == "ok" and "10" in text:
                error_class = "scientific_notation_pair_mismatch"
            else:
                error_class = f"status_or_pair:{expected['status']}->{prediction['status']}"
        scored.append(
            {
                "audit_case_id": case["audit_case_id"],
                "source_id": case["source_id"],
                "expected": expected,
                "observed": {
                    "status": prediction["status"],
                    "measurements": returned,
                },
                "status_exact": status_exact,
                "raw_pair_exact": raw_pair_exact,
                "canonical_exact": canonical_exact,
                "error_class": error_class,
            }
        )

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        return {
            "n": total,
            "status_exact": sum(row["status_exact"] for row in rows),
            "status_accuracy": sum(row["status_exact"] for row in rows) / total,
            "raw_pair_exact": sum(row["raw_pair_exact"] for row in rows),
            "raw_pair_accuracy": sum(row["raw_pair_exact"] for row in rows) / total,
            "canonical_exact": sum(row["canonical_exact"] for row in rows),
            "canonical_accuracy": sum(row["canonical_exact"] for row in rows) / total,
        }

    per_source = {
        source: metrics([row for row in scored if row["source_id"] == source])
        for source in sorted({row["source_id"] for row in scored})
    }
    error_counts = Counter(
        row["error_class"] for row in scored if row["error_class"] is not None
    )
    replay_manifest_path = mapping_path.with_suffix(".manifest.json")
    replay_manifest = (
        json.loads(replay_manifest_path.read_text())
        if replay_manifest_path.is_file()
        else {}
    )
    payload = {
        "version": "clintox_measurement_resolution_gold_replay.v1",
        "gold_path": str(gold_path),
        "gold_sha256": file_sha256(gold_path),
        "mapping_path": str(mapping_path),
        "mapping_sha256": file_sha256(mapping_path),
        "gold_corpus_version": gold_manifest["corpus_version"],
        "replay": {
            "model": replay_manifest.get("model"),
            "prompt_version": replay_manifest.get("prompt", {}).get("prompt_version"),
            "prompt_sha256": replay_manifest.get("prompt", {}).get("rendered_sha256"),
            "rejected_rows": replay_manifest.get("rejected_rows"),
        },
        "overall": metrics(scored),
        "per_source": per_source,
        "error_class_counts": dict(sorted(error_counts.items())),
        "prompt_gate": {
            "overall_canonical_at_least_95pct": metrics(scored)["canonical_accuracy"] >= 0.95,
            "every_source_canonical_at_least_90pct": all(
                row["canonical_accuracy"] >= 0.90 for row in per_source.values()
            ),
            "zero_bound_range_condition_or_exponent_errors": not any(
                error_counts.get(name, 0)
                for name in (
                    "bound_or_range_promoted",
                    "condition_or_other_endpoint_promoted",
                    "scientific_notation_pair_mismatch",
                )
            ),
        },
        "mismatches": [row for row in scored if not row["canonical_exact"]],
    }
    payload["prompt_gate"]["passed"] = all(payload["prompt_gate"].values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return payload


def build_cluster_pilots(data_root: Path, output_path: Path, *, device: str) -> dict[str, Any]:
    namespaces: list[dict[str, Any]] = []
    for source, fields in _NAMESPACES.items():
        parquet_path = data_root / source / "extractions.parquet"
        for field in fields:
            counts: Counter[str] = Counter()
            for batch in pq.ParquetFile(parquet_path).iter_batches(
                columns=[field], batch_size=100_000
            ):
                for value in batch.column(0).to_pylist():
                    cleaned = clean_scalar(value)
                    if (
                        cleaned is not None
                        and str(cleaned).strip().casefold() not in DEFAULT_NULL_LIKE
                    ):
                        counts[str(cleaned).strip()] += 1
            frequent = [value for value, _ in counts.most_common(CLUSTER_FREQUENT)]
            frequent_set = set(frequent)
            tail = sorted(
                (value for value in counts if value not in frequent_set),
                key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
            )[:CLUSTER_TAIL]
            values = frequent + tail
            embeddings = embed_values(
                values,
                device=device,
                progress_label=f"clintox/{source}/{field}",
            )
            clusters = cluster_values(
                values,
                embeddings,
                target_size=CLUSTER_TARGET,
                max_size=CLUSTER_MAX,
            )
            ranked = sorted(
                clusters,
                key=lambda cluster: (
                    -sum(counts[value] for value in cluster.values),
                    cluster.cluster_id,
                ),
            )
            review_ids = {cluster.cluster_id for cluster in ranked[:10]}
            namespaces.append(
                {
                    "namespace": f"{source}.{field}",
                    "source_id": source,
                    "input_field": field,
                    "distinct_values": len(counts),
                    "pilot_values": len(values),
                    "frequent_values": len(frequent),
                    "stable_tail_values": len(tail),
                    "cluster_count": len(clusters),
                    "review_cluster_ids": sorted(review_ids),
                    "clusters": [
                        {
                            "cluster_id": cluster.cluster_id,
                            "review_with_gpt": cluster.cluster_id in review_ids,
                            "values": list(cluster.values),
                            "value_counts": [counts[value] for value in cluster.values],
                        }
                        for cluster in clusters
                    ],
                }
            )
    payload = {
        "version": "clintox_stage2_cluster_pilot.v1",
        "status": "model_free_clusters_ready; gpt_review_pending",
        "sampling": {"frequent": CLUSTER_FREQUENT, "stable_tail": CLUSTER_TAIL},
        "clustering": {"target_size": CLUSTER_TARGET, "max_size": CLUSTER_MAX},
        "namespaces": namespaces,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _measurement_stratum(value: Any) -> str:
    text = str(value or "")
    lowered = text.casefold()
    numbers = _NUMBER.findall(text)
    if re.search(r"(?:e[+-]?\d+|10\s*\^|×\s*10|x\s*10)", lowered):
        return "scientific_notation"
    if re.search(r"(?:\bto\b|\brange\b|[<>≤≥~]|\d\s*[-–]\s*\d)", lowered):
        return "range_or_censored"
    if "%" in text or re.search(r"\b(?:ratio|fold|times)\b", lowered):
        return "percent_or_ratio"
    if len(numbers) > 1 or ";" in text:
        return "multi_quantity"
    if numbers and re.search(r"[a-z]", lowered):
        return "numeric_prose"
    return "other"


def _keep_smallest(heap, rank: int, record_id: str, row: dict[str, Any], limit: int) -> None:
    item = (-rank, record_id, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _heap_rows(heap) -> list[dict[str, Any]]:
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-records", type=Path, default=DEFAULT_CLEANED_RECORDS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--endpoint-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--score-gold-replay", type=Path)
    parser.add_argument("--gold-fixture", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args(argv)
    if args.score_gold_replay:
        payload = score_gold_replay(
            args.gold_fixture,
            args.score_gold_replay,
            args.score_gold_replay.with_suffix(".metrics.json"),
        )
        print(json.dumps({key: value for key, value in payload.items() if key != "mismatches"}, indent=2))
        return 0
    candidates_path = args.output_root / "measurement_resolution_gold_candidates.jsonl"
    build_gold_candidates(
        args.cleaned_records,
        candidates_path,
    )
    build_endpoint_review_cards(
        args.cleaned_records,
        candidates_path,
        args.endpoint_profile,
        args.output_root / "measurement_resolution_endpoint_review.jsonl",
    )
    if not args.gold_only:
        build_cluster_pilots(
            args.data_root,
            args.output_root / "cluster_pilot.json",
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
