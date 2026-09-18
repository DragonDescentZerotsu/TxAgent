"""Build mandatory, prompt-hidden diagnostics for one progressive run.

The input is a completed condition root containing prepared and output stages.
This module indexes the identifiers the renderer exposed, measures explicit
private-reasoning references, and reports the binary-label mix of L1. It never
changes a prompt, response, retrieval selection, or immutable cache.
"""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from functools import lru_cache

import pyarrow.parquet as pq

from predict.harnesses.progressive.prompt import prompt_assets, split_prompt_version
from predict.harnesses.progressive.references import (
    CARD_RE,
    GROUP_RE,
    MOLECULE_RE,
    SCHEMA_VERSION as REFERENCE_SCHEMA_VERSION,
    build_reference_index,
    match_references,
    validate_prompt_index,
)
from predict.harnesses.progressive.state import card_alias_maps
from predict.utils.json import canonical_json_bytes, sha256_file, write_json_atomic


ARTIFACT_VERSION = "progressive_run_diagnostics.v2"
COMPLETE_OUTPUT_STATUSES = {"ok", "carried_forward", "reused", "reused_none"}
MIX_FIELDS = ["task", "query_index", "benchmark_row_id", "actual_selected_k",
              "unique_parent_count", "repeated_parent_card_count", "n_label_0",
              "n_label_1", "minority_label", "minority_share", "unanimous"]
MIX_SUMMARY_FIELDS = ["task", "n_queries", "mean_actual_selected_k",
                      "mean_unique_parent_count", "mean_repeated_parent_card_count",
                      "mean_label_0_fraction", "mean_label_1_fraction",
                      "mean_minority_share", "unanimous_fraction"]
FILTER_FIELDS = ["task", "query_index", "benchmark_row_id", "level", "candidate_count",
                 "selected_count", "selected_parent_count", "selected_semantic_bucket_count",
                 "mean_morgan_similarity", "repeated_parent_records",
                 "overlap_semantic_top25", "overlap_control25"]
FILTER_SUMMARY_FIELDS = ["task", "level", "n_queries", "mean_selected_count",
                         "mean_selected_parent_count", "mean_selected_semantic_bucket_count",
                         "mean_morgan_similarity", "mean_repeated_parent_records",
                         "mean_overlap_semantic_top25", "mean_overlap_control25"]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _input_sha256(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(path)


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _filter_diagnostics(root: Path) -> tuple[list[dict], list[dict], set[Path]]:
    rows, paths = [], set()
    for path in sorted(root.glob("*/queries/query_idx*/levels/level_*/filter/selection.json")):
        receipt = _read(path)
        if receipt.get("status") != "ok":
            raise ValueError(f"invalid record-filter selection: {path}")
        rows.append({field: receipt.get(field, "") for field in FILTER_FIELDS})
        paths.update(candidate for candidate in (
            path, path.with_name("request.json"), path.with_name("output.json")
        ) if candidate.is_file())
    summary = []
    for task, level in sorted({(row["task"], row["level"]) for row in rows}):
        group = [row for row in rows if (row["task"], row["level"]) == (task, level)]
        mean = lambda field: sum(float(row[field]) for row in group) / len(group)
        summary.append({"task": task, "level": level, "n_queries": len(group),
            "mean_selected_count": mean("selected_count"),
            "mean_selected_parent_count": mean("selected_parent_count"),
            "mean_selected_semantic_bucket_count": mean("selected_semantic_bucket_count"),
            "mean_morgan_similarity": mean("mean_morgan_similarity"),
            "mean_repeated_parent_records": mean("repeated_parent_records"),
            "mean_overlap_semantic_top25": mean("overlap_semantic_top25"),
            "mean_overlap_control25": mean("overlap_control25")})
    return rows, summary, paths


def _context_labels(
    manifest_path: Path,
    benchmark_row_id: str,
    analogs: list[tuple[str, Mapping[str, Any]]],
) -> tuple[list[tuple[int, str, str]], list[Path]]:
    manifest = _read(manifest_path)
    database = manifest_path.with_name(str(manifest.get("database") or "retrieval.sqlite3"))
    uri = f"file:{database}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        query = connection.execute(
            "SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?",
            (benchmark_row_id,),
        ).fetchone()
        if query is None:
            raise ValueError(f"query absent from L1 context cache: {benchmark_row_id}")
        candidates = list(connection.execute(
            """SELECT g.external_context_id,g.parent_id,g.parent_smiles,
                      g.condition_group,g.gold_label
               FROM candidates AS c JOIN gold_contexts AS g USING(context_key)
               WHERE c.query_id=?""",
            (int(query[0]),),
        ))
    resolved = []
    used: set[str] = set()
    for _, analog in analogs:
        condition = str(analog.get("selected_condition") or "no_reported_external_condition")
        matches = [row for row in candidates
                   if str(row["parent_smiles"]) == str(analog.get("canonical_smiles"))
                   and str(row["condition_group"]) == condition
                   and str(row["external_context_id"]) not in used]
        if len(matches) != 1:
            raise ValueError("cannot uniquely recover selected L1 context label")
        row = matches[0]
        used.add(str(row["external_context_id"]))
        resolved.append((int(row["gold_label"]), str(row["external_context_id"]), str(row["parent_id"])))
    return resolved, [manifest_path, database]


def _ranking_labels(
    task: str,
    subset: str,
    benchmark_row_id: str,
    mode: str,
    analogs: list[tuple[str, Mapping[str, Any]]],
    parent_ids: list[str] | None = None,
    rankings: Path | None = None,
) -> tuple[list[tuple[int, str, str]], list[Path]]:
    rankings = rankings or (
        Path("predict/retrieval/cache/assay_reranking/archive/v9_direct_gold_morgan100")
        / task / "scaffold" / subset / "rankings.parquet"
    ).resolve()
    rows = _ranking_rows(rankings).get(benchmark_row_id, [])
    if not rows:
        raise ValueError(f"query absent from pinned L1 ranking: {benchmark_row_id}")
    resolved = []
    for index, (_, analog) in enumerate(analogs):
        matches = [row for row in rows if (
            row["retrieval_molecule_identity_key"] == parent_ids[index]
            if parent_ids is not None
            else row["retrieval_smiles"] == analog.get("canonical_smiles")
        )]
        if not matches:
            raise ValueError("cannot recover selected L1 parent from pinned ranking")
        assay_selected = isinstance(analog.get("assay_transfer_panel_rank"), int)
        if mode in {"assay-transfer", "joint"} and (mode != "joint" or assay_selected):
            row = min(matches, key=lambda item: (int(item["model_rank"]), item["retrieval_record_id"]))
        else:
            row = min(matches, key=lambda item: (
                int(item["retrieval_parent_rank"]),
                int(item["retrieval_parent_context_index"]),
                item["retrieval_record_id"],
            ))
        resolved.append((
            int(row["retrieval_gold_Y"]),
            str(row["retrieval_record_id"]),
            str(row["retrieval_molecule_identity_key"]),
        ))
    return resolved, [rankings]


@lru_cache(maxsize=None)
def _ranking_rows(rankings: Path) -> dict[str, list[dict[str, Any]]]:
    columns = [
        "query_record_id", "retrieval_record_id", "retrieval_molecule_identity_key",
        "retrieval_smiles", "retrieval_gold_Y", "retrieval_parent_rank",
        "retrieval_parent_context_index", "model_rank",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pq.read_table(rankings, columns=columns).to_pylist():
        grouped.setdefault(str(row["query_record_id"]), []).append(row)
    return grouped


def _cache_matched_parent_ids(
    manifest_path: Path,
    benchmark_row_id: str,
    analogs: list[tuple[str, Mapping[str, Any]]],
) -> tuple[list[str], list[Path], Path | None]:
    manifest = _read(manifest_path)
    database = manifest_path.with_name(str(manifest.get("database") or "retrieval.sqlite3"))
    uri = f"file:{database}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        query = connection.execute(
            "SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?",
            (benchmark_row_id,),
        ).fetchone()
        if query is None:
            raise ValueError(f"query absent from cache-matched database: {benchmark_row_id}")
        rows = list(connection.execute(
            """SELECT a.parent_id,r.external_record_id
               FROM assignments AS a JOIN records AS r USING(record_key)
               WHERE a.query_id=? AND a.pool='fixed' AND a.level='L1'""",
            (int(query[0]),),
        ))
    by_record = {str(row["external_record_id"]): str(row["parent_id"]) for row in rows}
    parent_ids = []
    for _, analog in analogs:
        record_ids = [
            str(card.get("_canonical_record_id") or "")
            for card in (analog.get("cards") or {}).values()
        ]
        matches = {by_record[record_id] for record_id in record_ids if record_id in by_record}
        if len(matches) != 1:
            raise ValueError("cannot recover one cache-matched parent from visible L1 records")
        parent_ids.append(matches.pop())
    audit = manifest.get("legacy_selection_audit") or {}
    ranking = audit.get("l1_ranking") or audit.get("v9") or {}
    ranking_path = ranking.get("rankings")
    return parent_ids, [manifest_path, database], (Path(ranking_path) if ranking_path else None)


def _ranked_parent_ids(
    manifest_path: Path,
    benchmark_row_id: str,
    analogs: list[tuple[str, Mapping[str, Any]]],
) -> tuple[list[str], list[Path], Path]:
    manifest = _read(manifest_path)
    rankings = manifest_path.with_name(str(manifest["rankings_database"]))
    evidence = (manifest_path.parent / str(manifest["evidence_database"])).resolve()
    with sqlite3.connect(f"file:{rankings}?mode=ro&immutable=1", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("ATTACH DATABASE ? AS evidence", (str(evidence),))
        query = connection.execute(
            "SELECT query_id FROM benchmark_queries WHERE benchmark_row_id=?",
            (benchmark_row_id,),
        ).fetchone()
        if query is None:
            raise ValueError(f"query absent from ranked retrieval database: {benchmark_row_id}")
        rows = list(connection.execute(
            """SELECT r.parent_id,e.external_record_id
               FROM rankings AS r JOIN evidence.records AS e USING(source_row_uid)
               WHERE r.query_id=? AND r.pool='fixed' AND r.level='L1'""",
            (int(query[0]),),
        ))
    by_record = {str(row["external_record_id"]): str(row["parent_id"]) for row in rows}
    parent_ids = []
    for _, analog in analogs:
        record_ids = [
            str(card.get("_canonical_record_id") or "")
            for card in (analog.get("cards") or {}).values()
        ]
        matches = {by_record[record_id] for record_id in record_ids if record_id in by_record}
        if len(matches) != 1:
            raise ValueError("cannot recover one ranked-retrieval parent from visible L1 records")
        parent_ids.append(matches.pop())
    l1_rankings = Path(manifest["inputs"]["l1_rankings"]["path"])
    return parent_ids, [manifest_path, rankings, evidence], l1_rankings


def _l1_labels(
    root: Path,
    task: str,
    prepared: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> tuple[list[tuple[int, str, str]], list[Path]]:
    analogs = sorted(
        (prepared.get("active_evidence") or {}).items(),
        key=lambda item: (int(item[1].get("_selection_rank") or 0), item[0]),
    )
    embedded = [analog.get("_diagnostic_label") for _, analog in analogs]
    if embedded and all(label in {0, 1} for label in embedded):
        return [(
            int(analog["_diagnostic_label"]),
            str(analog["_diagnostic_context_id"]),
            str(analog["_diagnostic_parent_id"]),
        ) for _, analog in analogs], []
    policy = (experiment.get("retrieval_policy") or {}).get(task) or prepared.get("retrieval_policy") or {}
    contract = str(policy.get("selection_contract") or "")
    if contract.startswith("l1_context_"):
        return _context_labels(
            Path(policy["cache_manifest"]),
            str(prepared["benchmark_row_id"]),
            analogs,
        )
    parent_ids = None
    rankings = None
    sources: list[Path] = []
    if contract == "ranked_evidence_retrieval.v1":
        parent_ids, sources, rankings = _ranked_parent_ids(
            Path(policy["cache_manifest"]),
            str(prepared["benchmark_row_id"]),
            analogs,
        )
    elif contract.startswith("cache_matched_retrieval.v"):
        parent_ids, sources, rankings = _cache_matched_parent_ids(
            Path(policy["cache_manifest"]),
            str(prepared["benchmark_row_id"]),
            analogs,
        )
    labels, ranking_sources = _ranking_labels(
        task,
        str(experiment.get("evaluation_subset") or "valid"),
        str(prepared["benchmark_row_id"]),
        str(experiment.get("reranking") or prepared.get("l1_ranking") or "morgan"),
        analogs,
        parent_ids,
        rankings,
    )
    return labels, [*sources, *ranking_sources]


def _legacy_layout(prompt_version: str) -> str:
    base, _ = split_prompt_version(prompt_version)
    if base == "reranked_progressive_l1_context_l2_semantic_v1":
        return "semantic_l2"
    if base == "reranked_progressive_l1_context_l2_weighted_v1":
        return "weighted_l2_legacy"
    return "l1"


def _stage_reference_index(
    prepared_path: Path,
    prepared: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    request_path = prepared_path.with_name("request.json")
    request = _read(request_path) if request_path.is_file() else {}
    saved = request.get("reasoning_reference_index")
    prompt_version = str(
        prepared.get("prompt_version")
        or experiment.get("prompt_version")
        or experiment.get("assay_transfer_prompt_version")
        or experiment.get("prompt_profile")
        or ""
    )
    declared = (
        prompt_assets(prompt_version)["settings"].get("reasoning_reference_contract")
        if prompt_version else None
    )
    if isinstance(saved, list):
        expected = request.get("reasoning_reference_index_sha256")
        actual = hashlib.sha256(canonical_json_bytes(saved)).hexdigest()
        if expected != actual or request.get("reasoning_reference_contract") != declared:
            raise ValueError(f"invalid saved reasoning-reference index: {request_path}")
        return saved, "request", actual
    if declared:
        raise ValueError(f"contracted prompt lacks a saved reference index: {request_path}")
    active = prepared.get("active_evidence") or {}
    aliases, _ = card_alias_maps(active)
    index = build_reference_index(
        active,
        card_id_to_alias=aliases,
        current_level=int(prepared["level"]),
        layout=_legacy_layout(prompt_version),
    )
    messages = request.get("messages")
    if isinstance(messages, list) and len(messages) > 1:
        validate_prompt_index(str(messages[1].get("content") or ""), index)
    return (
        index,
        "compatibility_reconstruction",
        hashlib.sha256(canonical_json_bytes(index)).hexdigest(),
    )


def build_run_diagnostics(root: str | Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """Write the complete diagnostic bundle and fail on missing measurement inputs."""
    root = Path(root).resolve()
    experiment_path = root / "experiment_manifest.json"
    if not experiment_path.is_file():
        raise ValueError(f"missing experiment manifest: {experiment_path}")
    experiment = _read(experiment_path)
    evidence_rows: list[dict[str, Any]] = []
    mix_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    reference_receipts: list[dict[str, Any]] = []
    stage_artifact_receipts: list[dict[str, Any]] = []
    provenance_paths: set[Path] = {experiment_path}
    prepared_paths = sorted(root.glob("*/queries/query_idx*/levels/level_*/prepared.json"))
    if not prepared_paths:
        raise ValueError(f"no progressive stages found: {root}")
    for prepared_path in prepared_paths:
        prepared = _read(prepared_path)
        task = str(prepared["task"])
        level = int(prepared["level"])
        query_index = int(prepared["query_index"])
        output_path = prepared_path.with_name("output.json")
        output = _read(output_path) if output_path.is_file() else {}
        gap_status = ""
        if not output_path.is_file():
            gap_status = "missing_output"
        elif output.get("status") not in COMPLETE_OUTPUT_STATUSES:
            gap_status = "non_complete_output"
        if gap_status and not allow_partial:
            message = (
                "missing model output"
                if gap_status == "missing_output"
                else "non-complete model output"
            )
            raise ValueError(f"{message} for diagnostics: {output_path}")
        request_path = prepared_path.with_name("request.json")
        stage_artifact_receipts.append({
            "stage": prepared_path.parent.relative_to(root).as_posix(),
            "prepared_sha256": sha256_file(prepared_path),
            "request_sha256": sha256_file(request_path) if request_path.is_file() else None,
            "output_sha256": sha256_file(output_path) if output_path.is_file() else None,
        })
        units, index_source, index_sha256 = _stage_reference_index(
            prepared_path, prepared, experiment
        )
        reference_receipts.append({
            "stage": prepared_path.parent.relative_to(root).as_posix(),
            "source": index_source,
            "sha256": index_sha256,
        })
        for row in units:
            row.update(gold_label="", context_id="", parent_id="")
        if level == 1:
            labels, sources = _l1_labels(root, task, prepared, experiment)
            provenance_paths.update(sources)
            analog_units = [
                row for row in units
                if row["unit_kind"] in {"molecule", "context_conditioned_molecule"}
            ]
            if len(labels) != len(analog_units):
                raise ValueError("L1 label count differs from visible molecule count")
            for row, (label, context_id, parent_id) in zip(analog_units, labels):
                row.update(gold_label=label, context_id=context_id, parent_id=parent_id)
            counts = Counter(label for label, _, _ in labels)
            selected_k = len(labels)
            unique_parents = len({parent_id for _, _, parent_id in labels})
            mix_rows.append({
                "task": task,
                "query_index": query_index,
                "benchmark_row_id": prepared.get("benchmark_row_id"),
                "actual_selected_k": selected_k,
                "unique_parent_count": unique_parents,
                "repeated_parent_card_count": selected_k - unique_parents,
                "n_label_0": counts[0],
                "n_label_1": counts[1],
                "minority_label": "tie" if counts[0] == counts[1] else (0 if counts[0] < counts[1] else 1),
                "minority_share": min(counts[0], counts[1]) / selected_k,
                "unanimous": int(min(counts[0], counts[1]) == 0),
            })
        for row in units:
            evidence_rows.append({
                "task": task,
                "query_index": query_index,
                "level": level,
                "reference_contract_version": REFERENCE_SCHEMA_VERSION,
                "reference_index_source": index_source,
                "reference_index_sha256": index_sha256,
                **row,
            })
        llm = output.get("llm") if isinstance(output.get("llm"), Mapping) else {}
        reasoning = llm.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise ValueError(f"reasoning_content is not text: {output_path}")
        measurement_status = gap_status or "measured"
        if not gap_status and not prepared.get("should_call_model") and not reasoning:
            measurement_status = "not_applicable"
        elif not gap_status and not reasoning.strip():
            if not allow_partial:
                raise ValueError(f"missing private reasoning for called stage: {output_path}")
            measurement_status = "missing_reasoning"
        if measurement_status in {"missing_output", "non_complete_output", "missing_reasoning"}:
            gap_rows.append({
                "task": task,
                "query_index": query_index,
                "level": level,
                "stage": prepared_path.parent.relative_to(root).as_posix(),
                "status": measurement_status,
                "output_status": output.get("status", ""),
            })
        referenced, occurrence_count, unknown, ambiguous, ambiguous_occurrences = (
            match_references(reasoning, units)
        )
        unit_by_id = {str(row["stable_id"]): row for row in units}
        molecule_ids = {
            str(row["stable_id"]) for row in units
            if row["unit_kind"] in {"molecule", "context_conditioned_molecule"}
        }
        group_ids = {
            str(row["stable_id"]) for row in units if row["unit_kind"] == "evidence_group"
        }
        record_ids = {
            str(row["stable_id"]) for row in units if row["unit_kind"] == "record"
        }
        all_ids = set(unit_by_id)
        new_ids = {str(row["stable_id"]) for row in units if row["is_new"]}
        molecule_labels = Counter(
            str(row["visible_id"]).casefold() for row in units
            if row["unit_kind"] in {"molecule", "context_conditioned_molecule"}
        )
        ambiguous_visible = sorted(
            {str(row["visible_id"]) for row in units
             if molecule_labels[str(row["visible_id"]).casefold()] > 1}
        )
        measurable = measurement_status == "measured"
        molecule_coverage_defined = measurable and not ambiguous_visible
        molecule_or_group_ids = molecule_ids | group_ids
        references = lambda ids: len(referenced & ids)
        coverage_rows.append({
            "task": task,
            "query_index": query_index,
            "benchmark_row_id": prepared.get("benchmark_row_id"),
            "level": level,
            "measurement_status": measurement_status,
            "reference_contract_version": REFERENCE_SCHEMA_VERSION,
            "reference_index_source": index_source,
            "reference_index_sha256": index_sha256,
            "visible_molecule_count": len(molecule_ids),
            "referenced_molecule_count": references(molecule_ids),
            "molecule_coverage_status": (
                "not_applicable" if not measurable else
                "defined" if molecule_coverage_defined else
                "undefined_ambiguous_visible_ids"
            ),
            "molecule_coverage": (
                references(molecule_ids) / len(molecule_ids)
                if molecule_ids and molecule_coverage_defined else ""
            ),
            "visible_evidence_group_count": len(group_ids),
            "referenced_evidence_group_count": references(group_ids),
            "evidence_group_coverage": (
                references(group_ids) / len(group_ids)
                if group_ids and measurable else ""
            ),
            "visible_molecule_or_group_count": len(molecule_or_group_ids),
            "referenced_molecule_or_group_count": references(molecule_or_group_ids),
            "molecule_or_group_coverage": (
                references(molecule_or_group_ids) / len(molecule_or_group_ids)
                if molecule_coverage_defined and molecule_or_group_ids else ""
            ),
            "visible_record_count": len(record_ids),
            "referenced_record_count": references(record_ids),
            "record_coverage": (
                references(record_ids) / len(record_ids)
                if record_ids and measurable else ""
            ),
            "visible_unit_count": len(all_ids),
            "referenced_unit_count": len(referenced),
            "unit_coverage": (
                len(referenced) / len(all_ids)
                if all_ids and molecule_coverage_defined else ""
            ),
            "new_unit_count": len(new_ids),
            "referenced_new_unit_count": references(new_ids),
            "new_unit_coverage": (
                references(new_ids) / len(new_ids)
                if new_ids and molecule_coverage_defined else ""
            ),
            "reference_occurrence_count": occurrence_count,
            "ambiguous_reference_occurrence_count": ambiguous_occurrences,
            "referenced_visible_ids": ",".join(sorted(
                str(unit_by_id[stable_id]["visible_id"]) for stable_id in referenced
            )),
            "unknown_reference_tokens": ",".join(unknown),
            "ambiguous_reference_tokens": ",".join(ambiguous),
            "ambiguous_visible_ids": ",".join(ambiguous_visible),
        })

    mix_summary = []
    for task in sorted({row["task"] for row in mix_rows}):
        rows = [row for row in mix_rows if row["task"] == task]
        mix_summary.append({
            "task": task,
            "n_queries": len(rows),
            "mean_actual_selected_k": sum(row["actual_selected_k"] for row in rows) / len(rows),
            "mean_unique_parent_count": sum(
                row["unique_parent_count"] for row in rows
            ) / len(rows),
            "mean_repeated_parent_card_count": sum(
                row["repeated_parent_card_count"] for row in rows
            ) / len(rows),
            "mean_label_0_fraction": sum(row["n_label_0"] / row["actual_selected_k"] for row in rows) / len(rows),
            "mean_label_1_fraction": sum(row["n_label_1"] / row["actual_selected_k"] for row in rows) / len(rows),
            "mean_minority_share": sum(row["minority_share"] for row in rows) / len(rows),
            "unanimous_fraction": sum(row["unanimous"] for row in rows) / len(rows),
        })
    coverage_summary = []
    keys = sorted({(row["task"], row["level"], row["measurement_status"]) for row in coverage_rows})
    for task, level, status in keys:
        rows = [row for row in coverage_rows
                if (row["task"], row["level"], row["measurement_status"]) == (task, level, status)]
        measured = status == "measured"

        def mean(field: str) -> float | str:
            values = [float(row[field]) for row in rows if row[field] != ""]
            return sum(values) / len(values) if measured and values else ""

        coverage_summary.append({
            "task": task,
            "level": level,
            "measurement_status": status,
            "n_queries": len(rows),
            "n_molecule_coverage_defined": sum(
                row["molecule_coverage_status"] == "defined" for row in rows
            ),
            "n_molecule_coverage_undefined": sum(
                row["molecule_coverage_status"] == "undefined_ambiguous_visible_ids"
                for row in rows
            ),
            "mean_molecule_coverage": mean("molecule_coverage"),
            "mean_evidence_group_coverage": mean("evidence_group_coverage"),
            "mean_molecule_or_group_coverage": mean("molecule_or_group_coverage"),
            "mean_record_coverage": mean("record_coverage"),
            "mean_unit_coverage": mean("unit_coverage"),
            "mean_new_unit_coverage": mean("new_unit_coverage"),
        })

    filter_rows, filter_summary, filter_paths = _filter_diagnostics(root)
    provenance_paths.update(filter_paths)
    tables = {
        "visible_evidence.tsv": (evidence_rows, [
            "task", "query_index", "level", "reference_contract_version",
            "reference_index_source", "reference_index_sha256", "unit_kind",
            "visible_id", "stable_id", "source_analog_id", "containing_unit_id",
            "first_visible_level", "is_new", "prompt_heading", "gold_label",
            "context_id", "parent_id",
        ]),
        "neighborhood_label_mix.per_query.tsv": (mix_rows, MIX_FIELDS),
        "neighborhood_label_mix.summary.tsv": (mix_summary, MIX_SUMMARY_FIELDS),
        "reasoning_reference_coverage.per_query.tsv": (coverage_rows, list(coverage_rows[0])),
        "reasoning_reference_coverage.summary.tsv": (coverage_summary, list(coverage_summary[0])),
    }
    if filter_rows:
        tables["record_filter.per_query.tsv"] = (filter_rows, FILTER_FIELDS)
        tables["record_filter.summary.tsv"] = (filter_summary, FILTER_SUMMARY_FIELDS)
    if gap_rows:
        tables["diagnostic_gaps.tsv"] = (gap_rows, list(gap_rows[0]))
    else:
        (root / "diagnostic_gaps.tsv").unlink(missing_ok=True)
    for name, (rows, fields) in tables.items():
        _write_tsv(root / name, rows, fields)
    report = [
        "# Progressive run diagnostics", "",
        "L1 label mix uses the frozen label of the exact condition context that admitted each visible molecule.",
        "Reasoning coverage counts case-insensitive, exact `Molecule N`, `Evidence group N`, and `Cxx` references in private reasoning. A valid 0/K coverage result is allowed.",
        "", "| task | queries | mean minority share | unanimous |", "|---|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['task']} | {row['n_queries']} | {row['mean_minority_share']:.4f} | {row['unanimous_fraction']:.4f} |"
        for row in mix_summary
    )
    if not mix_summary:
        report.append("| N/A (no L1 stages) | 0 | N/A | N/A |")
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outputs = [*tables, "report.md"]
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "status": "partial" if gap_rows else "complete",
        "coverage_contract": {
            "stream": "llm.reasoning_content",
            "explicit_reference_patterns": [MOLECULE_RE.pattern, GROUP_RE.pattern, CARD_RE.pattern],
            "matching": "case_insensitive_exact_visible_id",
            "required_mentions": False,
            "omitted_visible_units_are_valid": True,
            "missing_reasoning_or_visible_index_is_invalid": True,
        },
        "label_contract": "frozen label of the selected molecule-condition context",
        "inputs": {
            str(path): _input_sha256(path, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(provenance_paths)
        },
        "outputs": {name: sha256_file(root / name) for name in outputs},
        "reference_indices_sha256": hashlib.sha256(
            canonical_json_bytes(reference_receipts)
        ).hexdigest(),
        "reference_index_sources": dict(Counter(
            row["source"] for row in reference_receipts
        )),
        "stage_artifacts": {
            "count": len(stage_artifact_receipts),
            "receipt_sha256": hashlib.sha256(
                canonical_json_bytes(stage_artifact_receipts)
            ).hexdigest(),
            "receipt_fields": [
                "stage", "prepared_sha256", "request_sha256", "output_sha256"
            ],
        },
        "n_l1_queries": len(mix_rows),
        "n_record_filter_queries": len(filter_rows),
        "n_stage_outputs": len(coverage_rows),
        "n_measured_stage_outputs": sum(
            row["measurement_status"] in {"measured", "not_applicable"}
            for row in coverage_rows
        ),
        "n_diagnostic_gaps": len(gap_rows),
    }
    manifest_path = root / "diagnostics_manifest.json"
    write_json_atomic(manifest_path, manifest)
    run_path = root / "run.json"
    if run_path.is_file():
        run = _read(run_path)
        run["diagnostics"] = manifest_path.name
        run["diagnostics_schema_version"] = ARTIFACT_VERSION
        run["diagnostics_manifest_sha256"] = sha256_file(manifest_path)
        write_json_atomic(run_path, run)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_run_diagnostics(args.run_root, allow_partial=args.allow_partial)
    print(json.dumps({
        "status": manifest["status"],
        "l1_queries": manifest["n_l1_queries"],
        "stage_outputs": manifest["n_stage_outputs"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
