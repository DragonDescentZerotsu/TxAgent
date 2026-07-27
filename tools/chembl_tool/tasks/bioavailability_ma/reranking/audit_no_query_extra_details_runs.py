"""Audit corrected assay-transfer runs and compare them with historical v6.5 runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
from typing import Any


EXPECTED_PROFILE = "v6_5_query_context_copy_no_extra_details"
EXPECTED_QUERY_CONTEXT_POLICY = (
    "copy_retrieval_assay_context_except_extra_details_value_hidden.v1"
)
EXPECTED_GROUP_IDS = {
    "Observed.direct_oral_bioavailability",
    "Observed.oral_auc_cmax_exposure",
    "Fa.absorption_solubility_permeability",
    "Fg.gut_wall_efflux_intestinal_metabolism",
    "Fh.hepatic_clearance_metabolic_stability",
}
TRANSPORT_FAILURE_PATTERN = re.compile(
    r"(APIConnectionError|APITimeoutError|RateLimitError|HTTP [45]\d\d|"
    r"transport (?:error|failure)|timed out)",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    batches = _mapping(args.batch)
    previous = _mapping(args.previous)
    instruction_provenance = _instruction_provenance(
        Path(args.expected_group_instructions_file)
    )
    audits = {
        str(k): _audit_batch(
            k,
            path,
            instruction_provenance=instruction_provenance,
            require_instruction_provenance=args.require_group_instruction_provenance,
            expect_matched_reuse=args.expect_matched_reuse,
            expected_group_output_schema=args.expected_group_output_schema,
        )
        for k, path in sorted(batches.items())
    }
    comparisons = {
        str(k): _compare_batches(Path(path), Path(previous[k]))
        for k, path in sorted(batches.items())
        if k in previous
    }
    within_condition_comparisons = {
        f"k{left}_to_k{right}": _compare_condition_predictions(
            left,
            Path(batches[left]),
            right,
            Path(batches[right]),
        )
        for left, right in itertools.combinations(sorted(batches), 2)
    }
    missing_previous = sorted(set(batches) - set(previous))
    matched_replay_failures = {
        k: {
            "retrieval_mismatch_indices": comparison[
                "retrieval_mismatch_indices"
            ],
            "single_llm_mismatch_indices": comparison[
                "single_llm_mismatch_indices"
            ],
        }
        for k, comparison in comparisons.items()
        if comparison["retrieval_mismatch_indices"]
        or comparison["single_llm_mismatch_indices"]
    }
    status = (
        "pass"
        if (
            all(not audit["violations"] for audit in audits.values())
            and (not args.expect_matched_reuse or not matched_replay_failures)
        )
        else "fail"
    )
    summary = {
        "status": status,
        "profile": EXPECTED_PROFILE,
        "query_context_policy": EXPECTED_QUERY_CONTEXT_POLICY,
        "group_prompt_instructions": instruction_provenance,
        "group_output_schema": args.expected_group_output_schema,
        "expect_matched_reuse": args.expect_matched_reuse,
        "matched_replay_failures": matched_replay_failures,
        "batches": audits,
        "historical_comparisons": comparisons,
        "within_condition_comparisons": within_condition_comparisons,
        "k_without_historical_comparator": missing_previous,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_flips(
        out_dir / "prediction_flips.tsv",
        comparisons,
        within_condition_comparisons,
    )
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"status": status, "out_dir": str(out_dir)}, indent=2))
    return 0 if status == "pass" else 1


def _audit_batch(
    k: int,
    batch_dir: Path,
    *,
    instruction_provenance: dict[str, Any],
    require_instruction_provenance: bool,
    expect_matched_reuse: bool,
    expected_group_output_schema: str,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    manifest = _read_json(batch_dir / "manifest.json")
    metrics = _read_json(batch_dir / "metrics.json")
    expected_manifest = {
        "n_items": 64,
        "parallelism": 32,
        "group_workers": 5,
        "retrieval_source": "starling_in_distribution",
        "experiment_mode": "full_mechanism",
        "neighbor_identity_policy": "parent_disjoint",
        "retrieval_reranker": "assay_transfer",
        "enable_assay_transfer_scores": True,
        "assay_transfer_min_score": 0.5,
        "assay_transfer_template_profile": EXPECTED_PROFILE,
        "group_prompt_format": "assay_transfer_tool",
        "group_output_schema": expected_group_output_schema,
        "top_k_per_group": k,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            violations.append(
                {
                    "reason": "batch_manifest_mismatch",
                    "field": field,
                    "expected": expected,
                    "observed": manifest.get(field),
                }
            )
    instruction_hash = str(instruction_provenance["sha256"])
    if (
        require_instruction_provenance
        or manifest.get("group_prompt_instructions_sha256")
    ) and manifest.get("group_prompt_instructions_sha256") != instruction_hash:
        violations.append(
            {
                "reason": "batch_group_instruction_provenance",
                "expected": instruction_hash,
                "observed": manifest.get("group_prompt_instructions_sha256"),
            }
        )
    if expect_matched_reuse:
        for field in (
            "single_analysis_source_batch",
            "retrieval_replay_source_batch",
            "rerank_preflight_source_batch",
        ):
            source = str(manifest.get(field) or "")
            if not source or f"__k{k}__" not in source:
                violations.append(
                    {
                        "reason": "matched_reuse_batch_source",
                        "field": field,
                        "observed": source,
                    }
                )
    for field, expected in {
        "n_total": 64,
        "n_successful": 64,
        "n_failed_runs": 0,
    }.items():
        if metrics.get(field) != expected:
            violations.append(
                {
                    "reason": "batch_metrics_mismatch",
                    "field": field,
                    "expected": expected,
                    "observed": metrics.get(field),
                }
            )
    preflight = manifest.get("rerank_cache_preflight") or {}
    provenance = preflight.get("provenance") or {}
    if provenance.get("template_profile") != EXPECTED_PROFILE:
        violations.append({"reason": "preflight_template_profile"})
    if provenance.get("query_context_policy") != EXPECTED_QUERY_CONTEXT_POLICY:
        violations.append({"reason": "preflight_query_context_policy"})
    if (preflight.get("cache_version_validation") or {}).get("status") != "pass":
        violations.append({"reason": "cache_version_validation"})
    if preflight.get("cache_quick_check") != "ok":
        violations.append({"reason": "cache_quick_check"})

    group_sizes: Counter[int] = Counter()
    transferability: dict[str, Counter[str]] = {}
    usefulness: dict[str, Counter[str]] = {}
    confidence: dict[str, Counter[str]] = {}
    scores: list[float] = []
    similarities: list[float] = []
    repeated_record_entries = 0
    group_tool_calls = 0
    group_tool_results = 0
    single_property_results = 0
    final_tool_calls = 0
    final_tool_results = 0
    transport_failure_lines: list[str] = []
    run_dirs = sorted((batch_dir / "runs").glob("*_idx*"))
    if len(run_dirs) != 64:
        violations.append(
            {"reason": "run_count", "expected": 64, "observed": len(run_dirs)}
        )
    for run_dir in run_dirs:
        run_manifest = _read_json(run_dir / "manifest.json")
        query_index = int(run_manifest["query_index"])
        if run_manifest.get("group_tools_enabled") is not False:
            violations.append(
                {"reason": "group_tools_enabled", "query_index": query_index}
            )
        if run_manifest.get("assay_transfer_template_profile") != EXPECTED_PROFILE:
            violations.append(
                {"reason": "run_template_profile", "query_index": query_index}
            )
        if run_manifest.get("group_output_schema") != expected_group_output_schema:
            violations.append(
                {
                    "reason": "run_group_output_schema",
                    "query_index": query_index,
                    "expected": expected_group_output_schema,
                    "observed": run_manifest.get("group_output_schema"),
                }
            )
        schema_provenance = run_manifest.get("group_output_schema_provenance") or {}
        if (
            schema_provenance.get("profile") != expected_group_output_schema
            or not schema_provenance.get("schema_sha256")
        ):
            violations.append(
                {
                    "reason": "run_group_output_schema_provenance",
                    "query_index": query_index,
                }
            )
        run_provenance = run_manifest.get("assay_transfer_prompt_provenance") or {}
        if run_provenance.get("query_context_policy") != EXPECTED_QUERY_CONTEXT_POLICY:
            violations.append(
                {"reason": "run_query_context_policy", "query_index": query_index}
            )
        if (
            require_instruction_provenance
            or run_manifest.get("group_prompt_instructions_sha256")
        ) and run_manifest.get("group_prompt_instructions_sha256") != instruction_hash:
            violations.append(
                {
                    "reason": "run_group_instruction_provenance",
                    "query_index": query_index,
                    "expected": instruction_hash,
                    "observed": run_manifest.get(
                        "group_prompt_instructions_sha256"
                    ),
                }
            )
        if expect_matched_reuse:
            for field in (
                "single_analysis_source_run_dir",
                "retrieval_replay_source_run_dir",
            ):
                source = str(run_manifest.get(field) or "")
                if not source or f"__k{k}__" not in source:
                    violations.append(
                        {
                            "reason": "matched_reuse_run_source",
                            "query_index": query_index,
                            "field": field,
                            "observed": source,
                        }
                    )

        single = _read_json(run_dir / "single_molecule_reasoning_output.json")
        single_llm = single.get("llm") or {}
        property_results = [
            result
            for result in single_llm.get("tool_results") or []
            if result.get("tool_name") == "molecule_properties"
            and result.get("status") == "ok"
        ]
        single_property_results += len(property_results)
        if single.get("status") != "ok" or len(property_results) != 1:
            violations.append(
                {
                    "reason": "single_molecule_properties_contract",
                    "query_index": query_index,
                    "successful_property_results": len(property_results),
                }
            )

        retrieval = _read_json(run_dir / "retrieval.json")
        retrieval_groups = {
            group["group_id"]: group for group in retrieval.get("groups") or []
        }
        if set(retrieval_groups) != EXPECTED_GROUP_IDS:
            violations.append(
                {
                    "reason": "retrieval_group_ids",
                    "query_index": query_index,
                    "observed": sorted(retrieval_groups),
                }
            )
        group_outputs = _read_jsonl(run_dir / "group_reasoning_outputs.jsonl")
        if {row["group_id"] for row in group_outputs} != {
            group_id
            for group_id, group in retrieval_groups.items()
            if group.get("neighbors")
        }:
            violations.append(
                {"reason": "group_output_coverage", "query_index": query_index}
            )
        for output in group_outputs:
            group_id = output["group_id"]
            llm = output.get("llm") or {}
            group_tool_calls += len(llm.get("tool_calls") or [])
            group_tool_results += len(llm.get("tool_results") or [])
            if output.get("status") != "ok":
                violations.append(
                    {
                        "reason": "group_status",
                        "query_index": query_index,
                        "group_id": group_id,
                    }
                )
            messages = llm.get("messages") or []
            system = next(
                (str(item.get("content") or "") for item in messages if item.get("role") == "system"),
                "",
            )
            user = next(
                (str(item.get("content") or "") for item in messages if item.get("role") == "user"),
                "",
            )
            if (
                "No tools are available for this branch." not in system
                or "Use the supplied assay-transfer likelihoods" not in system
                or "You may call" in system
            ):
                violations.append(
                    {
                        "reason": "group_system_prompt",
                        "query_index": query_index,
                        "group_id": group_id,
                    }
                )
            for instruction in instruction_provenance["instructions"]:
                if instruction not in user:
                    violations.append(
                        {
                            "reason": "group_instruction_prompt",
                            "query_index": query_index,
                            "group_id": group_id,
                            "missing": instruction,
                        }
                    )
            if expected_group_output_schema == "assay-transfer":
                if (
                    '"assay_transfer_assessment": "string"' not in user
                    or '"bioavailability_implications": [' not in user
                    or '"record_rank": "integer or null"' not in user
                    or '"molecule_chembl_id": "string"' in user
                    or '"evidence_direction":' in user
                    or '"transferability":' in user
                ):
                    violations.append(
                        {
                            "reason": "assay_transfer_group_output_schema_prompt",
                            "query_index": query_index,
                            "group_id": group_id,
                        }
                    )
            group = retrieval_groups[group_id]
            neighbors = group.get("neighbors") or []
            group_sizes[len(neighbors)] += 1
            molecule_ids = [
                str(row.get("molecule_chembl_id") or "") for row in neighbors
            ]
            repeated_record_entries += len(molecule_ids) - len(set(molecule_ids))
            for neighbor in neighbors:
                score = float(neighbor["transfer_selection_score"])
                similarity = float(neighbor["similarity"])
                scores.append(score)
                similarities.append(similarity)
                if score + 1e-12 < 0.5:
                    violations.append(
                        {
                            "reason": "score_below_threshold",
                            "query_index": query_index,
                            "group_id": group_id,
                            "score": score,
                        }
                    )
                winning = neighbor.get("transfer_winning_record") or {}
                endpoint = str(winning.get("canonical_endpoint_key") or "")
                if not endpoint or f"- endpoint: {endpoint}" not in user:
                    violations.append(
                        {
                            "reason": "canonical_endpoint_prompt",
                            "query_index": query_index,
                            "group_id": group_id,
                            "endpoint": endpoint,
                        }
                    )
            content = llm.get("content") or {}
            if (
                expected_group_output_schema == "assay-transfer"
                and _contains_field_name(content, "molecule_chembl_id")
            ):
                violations.append(
                    {
                        "reason": "group_output_contains_molecule_chembl_id",
                        "query_index": query_index,
                        "group_id": group_id,
                    }
                )
            transferability.setdefault(group_id, Counter())[
                str(content.get("transferability") or "missing")
            ] += 1
            usefulness.setdefault(group_id, Counter())[
                str(content.get("useful_for_bioavailability_reasoning", "missing")).lower()
            ] += 1
            confidence.setdefault(group_id, Counter())[
                str(content.get("confidence") or "missing")
            ] += 1

        final = _read_json(run_dir / "final_reasoning_output.json")
        final_llm = final.get("llm") or {}
        final_tool_calls += len(final_llm.get("tool_calls") or [])
        final_tool_results += len(final_llm.get("tool_results") or [])
        if final.get("status") != "ok":
            violations.append(
                {"reason": "final_status", "query_index": query_index}
            )

    for log_path in sorted((batch_dir / "logs").glob("*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if TRANSPORT_FAILURE_PATTERN.search(line):
                transport_failure_lines.append(f"{log_path.name}: {line[:500]}")
    if group_tool_calls or group_tool_results:
        violations.append(
            {
                "reason": "group_tools_observed",
                "tool_calls": group_tool_calls,
                "tool_results": group_tool_results,
            }
        )
    if final_tool_calls or final_tool_results:
        violations.append(
            {
                "reason": "final_tools_observed",
                "tool_calls": final_tool_calls,
                "tool_results": final_tool_results,
            }
        )
    if transport_failure_lines:
        violations.append(
            {
                "reason": "transport_failures",
                "count": len(transport_failure_lines),
                "examples": transport_failure_lines[:10],
            }
        )
    return {
        "batch_dir": str(batch_dir),
        "metrics": metrics,
        "n_runs": len(run_dirs),
        "n_single_molecule_properties_results": single_property_results,
        "n_group_tool_calls": group_tool_calls,
        "n_group_tool_results": group_tool_results,
        "n_final_tool_calls": final_tool_calls,
        "n_final_tool_results": final_tool_results,
        "n_transport_failure_lines": len(transport_failure_lines),
        "group_size_distribution": dict(sorted(group_sizes.items())),
        "n_repeated_record_entries_within_group": repeated_record_entries,
        "score_diagnostics": _diagnostics(scores),
        "similarity_diagnostics": _diagnostics(similarities),
        "group_transferability_distribution": {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(transferability.items())
        },
        "group_usefulness_distribution": {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(usefulness.items())
        },
        "group_confidence_distribution": {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(confidence.items())
        },
        "violations": violations,
    }


def _contains_field_name(value: Any, field_name: str) -> bool:
    if isinstance(value, dict):
        return field_name in value or any(
            _contains_field_name(nested, field_name) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_field_name(nested, field_name) for nested in value)
    return False


def _compare_batches(current: Path, previous: Path) -> dict[str, Any]:
    current_metrics = _read_json(current / "metrics.json")
    previous_metrics = _read_json(previous / "metrics.json")
    current_predictions = _predictions(current)
    previous_predictions = _predictions(previous)
    overlap = sorted(set(current_predictions) & set(previous_predictions))
    flips = [
        {
            "query_index": index,
            "label": current_predictions[index].get("label"),
            "previous_prediction": previous_predictions[index].get("pred_label"),
            "current_prediction": current_predictions[index].get("pred_label"),
        }
        for index in overlap
        if current_predictions[index].get("pred_label")
        != previous_predictions[index].get("pred_label")
    ]
    previous_scores, previous_similarities, previous_transferability = (
        _batch_retrieval_and_transferability(previous)
    )
    current_scores, current_similarities, current_transferability = (
        _batch_retrieval_and_transferability(current)
    )
    retrieval_mismatch_indices: list[int] = []
    single_llm_mismatch_indices: list[int] = []
    for index in overlap:
        current_run = current / "runs" / f"{current.name}_idx{index:05d}"
        previous_run = previous / "runs" / f"{previous.name}_idx{index:05d}"
        current_retrieval = _read_json(current_run / "retrieval.json")
        previous_retrieval = _read_json(previous_run / "retrieval.json")
        if (
            current_retrieval.get("query") != previous_retrieval.get("query")
            or current_retrieval.get("groups") != previous_retrieval.get("groups")
        ):
            retrieval_mismatch_indices.append(index)
        current_single = _read_json(
            current_run / "single_molecule_reasoning_output.json"
        )
        previous_single = _read_json(
            previous_run / "single_molecule_reasoning_output.json"
        )
        if current_single.get("llm") != previous_single.get("llm"):
            single_llm_mismatch_indices.append(index)
    return {
        "current_batch": str(current),
        "previous_batch": str(previous),
        "n_paired": len(overlap),
        "n_prediction_flips": len(flips),
        "prediction_flips": flips,
        "n_retrieval_exact_matches": len(overlap)
        - len(retrieval_mismatch_indices),
        "retrieval_mismatch_indices": retrieval_mismatch_indices,
        "n_single_llm_exact_matches": len(overlap)
        - len(single_llm_mismatch_indices),
        "single_llm_mismatch_indices": single_llm_mismatch_indices,
        "accuracy_delta": round(
            float(current_metrics["accuracy"]) - float(previous_metrics["accuracy"]),
            6,
        ),
        "macro_f1_delta": round(
            float(current_metrics["macro_f1"]) - float(previous_metrics["macro_f1"]),
            6,
        ),
        "current_metrics": current_metrics,
        "previous_metrics": previous_metrics,
        "current_score_diagnostics": _diagnostics(current_scores),
        "previous_score_diagnostics": _diagnostics(previous_scores),
        "current_similarity_diagnostics": _diagnostics(current_similarities),
        "previous_similarity_diagnostics": _diagnostics(previous_similarities),
        "current_group_transferability_distribution": current_transferability,
        "previous_group_transferability_distribution": previous_transferability,
    }


def _compare_condition_predictions(
    left_k: int,
    left: Path,
    right_k: int,
    right: Path,
) -> dict[str, Any]:
    left_metrics = _read_json(left / "metrics.json")
    right_metrics = _read_json(right / "metrics.json")
    left_predictions = _predictions(left)
    right_predictions = _predictions(right)
    overlap = sorted(set(left_predictions) & set(right_predictions))
    flips = [
        {
            "query_index": index,
            "label": right_predictions[index].get("label"),
            "previous_prediction": left_predictions[index].get("pred_label"),
            "current_prediction": right_predictions[index].get("pred_label"),
        }
        for index in overlap
        if left_predictions[index].get("pred_label")
        != right_predictions[index].get("pred_label")
    ]
    return {
        "from_k": left_k,
        "to_k": right_k,
        "from_batch": str(left),
        "to_batch": str(right),
        "n_paired": len(overlap),
        "n_prediction_flips": len(flips),
        "prediction_flips": flips,
        "accuracy_delta": round(
            float(right_metrics["accuracy"]) - float(left_metrics["accuracy"]), 6
        ),
        "macro_f1_delta": round(
            float(right_metrics["macro_f1"]) - float(left_metrics["macro_f1"]), 6
        ),
    }


def _batch_retrieval_and_transferability(
    batch: Path,
) -> tuple[list[float], list[float], dict[str, dict[str, int]]]:
    scores: list[float] = []
    similarities: list[float] = []
    transferability: dict[str, Counter[str]] = {}
    for run_dir in sorted((batch / "runs").glob("*_idx*")):
        retrieval = _read_json(run_dir / "retrieval.json")
        for group in retrieval.get("groups") or []:
            for neighbor in group.get("neighbors") or []:
                scores.append(float(neighbor["transfer_selection_score"]))
                similarities.append(float(neighbor["similarity"]))
        for output in _read_jsonl(run_dir / "group_reasoning_outputs.jsonl"):
            transferability.setdefault(output["group_id"], Counter())[
                str(((output.get("llm") or {}).get("content") or {}).get("transferability") or "missing")
            ] += 1
    return (
        scores,
        similarities,
        {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(transferability.items())
        },
    )


def _diagnostics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _predictions(batch: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["query_index"]): row
        for row in _read_jsonl(batch / "predictions.jsonl")
    }


def _write_flips(
    path: Path,
    historical_comparisons: dict[str, dict[str, Any]],
    within_condition_comparisons: dict[str, dict[str, Any]],
) -> None:
    rows = [
        {
            "comparison": f"historical_k{k}_to_corrected_k{k}",
            "from_k": int(k),
            "to_k": int(k),
            **row,
        }
        for k, comparison in historical_comparisons.items()
        for row in comparison["prediction_flips"]
    ]
    rows.extend(
        {
            "comparison": label,
            "from_k": comparison["from_k"],
            "to_k": comparison["to_k"],
            **row,
        }
        for label, comparison in within_condition_comparisons.items()
        for row in comparison["prediction_flips"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparison",
                "from_k",
                "to_k",
                "query_index",
                "label",
                "previous_prediction",
                "current_prediction",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Corrected Assay-Transfer Group-Reasoning Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Template profile: `{summary['profile']}`",
        f"- Query-context policy: `{summary['query_context_policy']}`",
        f"- Group-instruction SHA-256: `{summary['group_prompt_instructions']['sha256']}`",
        f"- Group-output schema: `{summary['group_output_schema']}`",
        f"- Matched retrieval/single reuse: `{summary['expect_matched_reuse']}`",
        "",
        "| k | successful | accuracy | macro-F1 | group tools | single property results | transport failures | violations |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, audit in sorted(summary["batches"].items(), key=lambda item: int(item[0])):
        metrics = audit["metrics"]
        lines.append(
            f"| {k} | {metrics.get('n_successful')} | {metrics.get('accuracy')} | "
            f"{metrics.get('macro_f1')} | {audit['n_group_tool_calls']} / "
            f"{audit['n_group_tool_results']} | "
            f"{audit['n_single_molecule_properties_results']} | "
            f"{audit['n_transport_failure_lines']} | {len(audit['violations'])} |"
        )
    lines.extend(
        [
            "",
            "## Historical comparisons",
            "",
            "| k | paired | retrieval exact | single prior exact | flips | accuracy delta | macro-F1 delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k, comparison in sorted(
        summary["historical_comparisons"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            f"| {k} | {comparison['n_paired']} | "
            f"{comparison['n_retrieval_exact_matches']} | "
            f"{comparison['n_single_llm_exact_matches']} | "
            f"{comparison['n_prediction_flips']} | {comparison['accuracy_delta']} | "
            f"{comparison['macro_f1_delta']} |"
        )
    if summary["k_without_historical_comparator"]:
        lines.extend(
            [
                "",
                "No matching historical run was available for k="
                + ", ".join(str(k) for k in summary["k_without_historical_comparator"])
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "## Within-condition k comparisons",
            "",
            "| comparison | paired | flips | accuracy delta | macro-F1 delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, comparison in sorted(summary["within_condition_comparisons"].items()):
        lines.append(
            f"| {label} | {comparison['n_paired']} | "
            f"{comparison['n_prediction_flips']} | {comparison['accuracy_delta']} | "
            f"{comparison['macro_f1_delta']} |"
        )
    lines.extend(
        [
            "",
            "The machine-readable summary includes threshold, group-size, repeated-record, "
            "score/similarity, prompt, tool, provenance, prediction-flip, and group "
            "transferability diagnostics. No performance improvement was required.",
            "",
        ]
    )
    return "\n".join(lines)


def _mapping(values: list[str]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected K=PATH, received {value!r}")
        output[int(key)] = Path(path)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instruction_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    text = raw.decode("utf-8")
    instructions = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not instructions:
        raise ValueError(f"No instruction lines in {resolved}")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "instruction_count": len(instructions),
        "instructions": instructions,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        help="Corrected batch mapping in K=PATH form; repeat for each k.",
    )
    parser.add_argument(
        "--previous",
        action="append",
        default=[],
        help="Historical batch mapping in K=PATH form; repeat when available.",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/paper/molecular_evidence_agent/validation_parent_disjoint/"
            "bioavailability_ma/v6_5_no_query_extra_details/analysis"
        ),
    )
    parser.add_argument(
        "--expected-group-instructions-file",
        default=(
            "tools/chembl_tool/tasks/bioavailability_ma/"
            "prompt_instructions/assay_transfer_tool.txt"
        ),
    )
    parser.add_argument(
        "--require-group-instruction-provenance",
        action="store_true",
        help="Require the expected instruction-file hash in batch and run manifests.",
    )
    parser.add_argument(
        "--expected-group-output-schema",
        choices=["legacy", "assay-transfer"],
        default="legacy",
    )
    parser.add_argument(
        "--expect-matched-reuse",
        action="store_true",
        help="Require matched-k retrieval replay and single-analysis source provenance.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
