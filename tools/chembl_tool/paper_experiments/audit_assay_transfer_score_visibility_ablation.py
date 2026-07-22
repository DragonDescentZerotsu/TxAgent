"""Audit assay-transfer score-hidden batches against visible and Morgan controls."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
from typing import Any

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import classify_molecule_relation
from tools.chembl_tool.paper_experiments.audit_assay_transfer_reranked_validation import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    EXPECTED_SCORING_CONTRACT,
    _audit_cache,
    _compare,
    _read_json,
    _read_jsonl,
)


SCORE_SEMANTICS = "best compatible source measurement transfers"
FORBIDDEN_PROMPT_FIELDS = {
    "assay_transfer_score",
    "assay_transfer_score_policy",
    "transfer_selection_score",
    "transfer_selection_rank",
    "transfer_winning_record_id",
    "transfer_scored_record_count",
}
FORBIDDEN_RELATIONS = {"exact_record", "same_connectivity_variant", "same_parent"}
SOFT_MODEL = "jiosephlee/assay-transfer-tool-soft"
SOFT_REVISION = "5fc06af66b490575e8eb32231d96aeada26794c6"
SOFT_V6_5_MODEL = "jiosephlee/assay-transfer-tool-soft-v6.5"
SOFT_V6_5_REVISION = "3572110c78aae7ac6de4be41b5ac530094573719"
SOFT_V6_5_PROFILE = "v6_5_query_context_copy"
SOFT_V6_5_TEMPLATE_HASH = "e30f995988db7214cae4b170a2c36f3a5fd61b6bee6188b39ce54377fa67f5bd"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    hidden = dict(_label_path(value) for value in args.hidden_batch)
    visible = dict(_label_path(value) for value in args.visible_batch)
    baselines = dict(_label_path(value) for value in args.baseline)
    if set(hidden) != set(visible) or set(hidden) != set(baselines):
        raise SystemExit("Hidden, visible, and baseline labels must match exactly")

    soft = dict(_label_path(value) for value in args.soft_batch)
    if soft and set(soft) != set(hidden):
        raise SystemExit("Soft and existing condition labels must match exactly")
    if bool(soft) != bool(args.soft_cache):
        raise SystemExit("--soft-batch and --soft-cache must be supplied together")

    soft_v6_5 = dict(_label_path(value) for value in args.soft_v6_5_batch)
    if soft_v6_5 and set(soft_v6_5) != {"sim0p3_k3", "sim0p3_k5", "sim0p3_k7"}:
        raise SystemExit("v6.5 batches must use labels sim0p3_k3, sim0p3_k5, and sim0p3_k7")
    if bool(soft_v6_5) != bool(args.soft_v6_5_cache):
        raise SystemExit("--soft-v6-5-batch and --soft-v6-5-cache must be supplied together")

    cache_audit = _audit_cache(Path(args.cache))
    soft_cache_audit = _audit_soft_cache(Path(args.soft_cache)) if args.soft_cache else None
    soft_v6_5_cache_audit = (
        _audit_soft_cache(
            Path(args.soft_v6_5_cache),
            model=SOFT_V6_5_MODEL,
            revision=SOFT_V6_5_REVISION,
            score_count=9824,
            template_profile=SOFT_V6_5_PROFILE,
            template_hash=SOFT_V6_5_TEMPLATE_HASH,
        )
        if args.soft_v6_5_cache
        else None
    )
    audits = []
    for label in hidden:
        audit = _audit_condition(
            label,
            hidden_dir=Path(hidden[label]),
            visible_dir=Path(visible[label]),
            baseline_dir=Path(baselines[label]),
        )
        audits.append(audit)
        (Path(hidden[label]) / "score_visibility_ablation_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    soft_audits = []
    for label, path in soft.items():
        audit = _audit_soft_condition(label, Path(path))
        soft_audits.append(audit)
        (Path(path) / "soft_score_hidden_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    soft_v6_5_audits = []
    for label, path in soft_v6_5.items():
        audit = _audit_soft_condition(
            label,
            Path(path),
            model=SOFT_V6_5_MODEL,
            revision=SOFT_V6_5_REVISION,
            template_profile=SOFT_V6_5_PROFILE,
            template_hash=SOFT_V6_5_TEMPLATE_HASH,
        )
        soft_v6_5_audits.append(audit)
        (Path(path) / "soft_v6_5_score_hidden_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    status = "pass" if (
        cache_audit["audit_status"] == "pass"
        and all(item["audit_status"] == "pass" for item in audits)
        and (soft_cache_audit is None or soft_cache_audit["audit_status"] == "pass")
        and all(item["audit_status"] == "pass" for item in soft_audits)
        and (soft_v6_5_cache_audit is None or soft_v6_5_cache_audit["audit_status"] == "pass")
        and all(item["audit_status"] == "pass" for item in soft_v6_5_audits)
    ) else "fail"
    summary = {
        "status": status,
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "scoring_contract": EXPECTED_SCORING_CONTRACT,
        "cache": str(Path(args.cache)),
        "cache_audit": cache_audit,
        "conditions": audits,
        "soft_model": SOFT_MODEL if soft else None,
        "soft_model_revision": SOFT_REVISION if soft else None,
        "soft_cache": str(Path(args.soft_cache)) if args.soft_cache else None,
        "soft_cache_audit": soft_cache_audit,
        "soft_conditions": soft_audits,
        "soft_v6_5_model": SOFT_V6_5_MODEL if soft_v6_5 else None,
        "soft_v6_5_model_revision": SOFT_V6_5_REVISION if soft_v6_5 else None,
        "soft_v6_5_cache": str(Path(args.soft_v6_5_cache)) if args.soft_v6_5_cache else None,
        "soft_v6_5_cache_audit": soft_v6_5_cache_audit,
        "soft_v6_5_conditions": soft_v6_5_audits,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    _write_tsv(out_dir / "condition_results.tsv", summary)
    print(json.dumps({"status": status, "out_dir": str(out_dir)}, indent=2))
    return 0 if status == "pass" else 1


def _audit_soft_cache(
    cache_path: Path,
    *,
    model: str = SOFT_MODEL,
    revision: str = SOFT_REVISION,
    score_count: int = 45057,
    template_profile: str = "",
    template_hash: str = "",
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    version_path = cache_path.parent / "VERSION.json"
    version = _read_json(version_path) if version_path.is_file() else {}
    for field, expected in {
        "status": "complete",
        "model": model,
        "model_revision": revision,
        "scoring_contract_version": EXPECTED_SCORING_CONTRACT,
        "n_prompt_scores": score_count,
    }.items():
        if version.get(field) != expected:
            violations.append({"reason": "soft_cache_version_mismatch", "field": field, "observed": version.get(field)})
    try:
        with sqlite3.connect(f"file:{cache_path.resolve()}?mode=ro", uri=True) as connection:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            n_rows = int(connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0])
            model_rows = int(connection.execute(
                "SELECT COUNT(*) FROM prompt_scores WHERE model=? AND model_revision=?",
                (model, revision),
            ).fetchone()[0])
    except (sqlite3.Error, TypeError, IndexError) as exc:
        quick_check, n_rows, model_rows = "error", 0, 0
        violations.append({"reason": "unreadable_soft_cache", "error": str(exc)})
    if template_profile and version.get("template_profile") != template_profile:
        violations.append({"reason": "soft_cache_template_profile", "observed": version.get("template_profile")})
    if template_hash and version.get("template_hash") != template_hash:
        violations.append({"reason": "soft_cache_template_hash", "observed": version.get("template_hash")})
    if quick_check != "ok" or n_rows != score_count or model_rows != score_count:
        violations.append({"reason": "soft_cache_content", "quick_check": quick_check, "rows": n_rows, "model_rows": model_rows})
    return {
        "audit_status": "pass" if not violations else "fail",
        "version_path": str(version_path),
        "n_prompt_scores": n_rows,
        "quick_check": quick_check,
        "violations": violations,
    }


def _audit_soft_condition(
    label: str,
    batch_dir: Path,
    *,
    model: str = SOFT_MODEL,
    revision: str = SOFT_REVISION,
    template_profile: str = "",
    template_hash: str = "",
) -> dict[str, Any]:
    manifest = _read_json(batch_dir / "manifest.json")
    metrics = _read_json(batch_dir / "metrics.json")
    top_k = int(manifest["top_k_per_group"])
    floor = float(manifest["min_similarity"])
    violations: list[dict[str, Any]] = []
    relations: Counter[str] = Counter()
    n_neighbors = 0
    n_groups_below_k = 0
    for field, expected in {
        "experiment_mode": "full_mechanism",
        "retrieval_source": "starling",
        "retrieval_reranker": "assay_transfer",
        "neighbor_identity_policy": "parent_disjoint",
        "enable_assay_transfer_scores": False,
        "llm_neighbor_score_policy": "",
    }.items():
        if manifest.get(field) != expected:
            violations.append({"reason": "soft_manifest_mismatch", "field": field, "observed": manifest.get(field)})
    provenance = (manifest.get("rerank_cache_preflight") or {}).get("provenance") or {}
    for field, expected in {
        "model": model,
        "model_revision": revision,
        "scoring_contract_version": EXPECTED_SCORING_CONTRACT,
    }.items():
        if provenance.get(field) != expected:
            violations.append({"reason": "soft_provenance_mismatch", "field": field, "observed": provenance.get(field)})
    if template_profile:
        for field, expected in {
            "assay_transfer_template_profile": template_profile,
        }.items():
            if manifest.get(field) != expected:
                violations.append({"reason": "soft_manifest_mismatch", "field": field, "observed": manifest.get(field)})
        for field, expected in {"template_profile": template_profile, "template_hash": template_hash}.items():
            if provenance.get(field) != expected:
                violations.append({"reason": "soft_provenance_mismatch", "field": field, "observed": provenance.get(field)})
    for field, expected in {"n_total": 64, "n_successful": 64, "n_failed_runs": 0}.items():
        if metrics.get(field) != expected:
            violations.append({"reason": "incomplete_soft_batch", "field": field, "observed": metrics.get(field)})

    runs = _runs_by_index(batch_dir)
    if set(runs) != set(range(64)):
        violations.append({"reason": "soft_run_coverage", "observed": len(runs)})
    for index, run_dir in sorted(runs.items()):
        retrieval = _read_json(run_dir / "retrieval.json")
        query_identity = normalize_molecule_identity(retrieval["query"]["canonical_smiles"])
        for group in retrieval.get("groups") or []:
            neighbors = group.get("neighbors") or []
            if len(neighbors) > top_k:
                violations.append({"reason": "over_k", "query_index": index, "group": group["group_id"]})
            n_groups_below_k += len(neighbors) < top_k
            selection = group.get("transfer_neighbor_selection") or {}
            if int(selection.get("n_selected", -1)) != len(neighbors):
                violations.append({"reason": "invalid_below_k_attribution", "query_index": index, "group": group["group_id"]})
            previous = float("inf")
            for neighbor in neighbors:
                n_neighbors += 1
                relation = classify_molecule_relation(
                    query_identity, normalize_molecule_identity(neighbor["canonical_smiles"])
                ).value
                relations[relation] += 1
                similarity = float(neighbor["similarity"])
                score = float(neighbor["transfer_selection_score"])
                if relation in FORBIDDEN_RELATIONS:
                    violations.append({"reason": relation, "query_index": index, "group": group["group_id"]})
                if similarity < floor:
                    violations.append({"reason": "below_similarity_floor", "query_index": index, "similarity": similarity})
                if not 0.0 <= score <= 1.0 or score > previous:
                    violations.append({"reason": "invalid_soft_rerank_order", "query_index": index, "group": group["group_id"]})
                previous = score
        prompt_result = _audit_hidden_prompts(run_dir, retrieval)
        violations.extend({"query_index": index, **item} for item in prompt_result["violations"])
    return {
        "label": label,
        "batch_dir": str(batch_dir),
        "audit_status": "pass" if not violations else "fail",
        "metrics": metrics,
        "top_k_per_group": top_k,
        "min_similarity": floor,
        "n_neighbors": n_neighbors,
        "n_groups_below_k": n_groups_below_k,
        "neighbor_relation_counts": dict(sorted(relations.items())),
        "violations": violations,
    }


def _audit_condition(
    label: str, *, hidden_dir: Path, visible_dir: Path, baseline_dir: Path
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    hidden_manifest = _read_json(hidden_dir / "manifest.json")
    visible_manifest = _read_json(visible_dir / "manifest.json")
    hidden_metrics = _read_json(hidden_dir / "metrics.json")
    for field, expected in {
        "n_total": 64,
        "n_successful": 64,
        "n_failed_runs": 0,
    }.items():
        if hidden_metrics.get(field) != expected:
            violations.append({"reason": "incomplete_batch", "field": field, "observed": hidden_metrics.get(field)})
    for field, expected in {
        "experiment_mode": "full_mechanism",
        "retrieval_source": "starling",
        "retrieval_reranker": "assay_transfer",
        "neighbor_identity_policy": "parent_disjoint",
        "enable_assay_transfer_scores": False,
        "llm_neighbor_score_policy": "",
    }.items():
        if hidden_manifest.get(field) != expected:
            violations.append({"reason": "hidden_manifest_mismatch", "field": field, "observed": hidden_manifest.get(field)})
    for field in (
        "top_k_per_group",
        "min_similarity",
        "rerank_raw_pool_size",
        "rerank_candidate_size",
        "rerank_catalog",
        "rerank_cache",
        "rerank_candidate_manifest",
        "assay_transfer_model_revision",
    ):
        if hidden_manifest.get(field) != visible_manifest.get(field):
            violations.append({
                "reason": "paired_manifest_mismatch",
                "field": field,
                "hidden": hidden_manifest.get(field),
                "visible": visible_manifest.get(field),
            })
    provenance = (hidden_manifest.get("rerank_cache_preflight") or {}).get("provenance") or {}
    for field, expected in {
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "scoring_contract_version": EXPECTED_SCORING_CONTRACT,
    }.items():
        if provenance.get(field) != expected:
            violations.append({"reason": "provenance_mismatch", "field": field, "observed": provenance.get(field)})

    hidden_runs = _runs_by_index(hidden_dir)
    visible_runs = _runs_by_index(visible_dir)
    if set(hidden_runs) != set(range(64)) or set(visible_runs) != set(range(64)):
        violations.append({"reason": "run_coverage", "hidden": len(hidden_runs), "visible": len(visible_runs)})
    n_groups = 0
    n_neighbors = 0
    n_hidden_group_prompts = 0
    for index in sorted(set(hidden_runs) & set(visible_runs)):
        hidden_retrieval = _read_json(hidden_runs[index] / "retrieval.json")
        visible_retrieval = _read_json(visible_runs[index] / "retrieval.json")
        hidden_signature = _retrieval_signature(hidden_retrieval)
        visible_signature = _retrieval_signature(visible_retrieval)
        if hidden_signature != visible_signature:
            violations.append({"reason": "retrieval_not_identical", "query_index": index})
        n_groups += len(hidden_signature)
        n_neighbors += sum(len(group[1]) for group in hidden_signature)
        prompt_result = _audit_hidden_prompts(hidden_runs[index], hidden_retrieval)
        n_hidden_group_prompts += prompt_result["n_group_prompts"]
        violations.extend({"query_index": index, **item} for item in prompt_result["violations"])

    visible_audit_path = visible_dir / "reranked_retrieval_audit.json"
    if not visible_audit_path.is_file() or _read_json(visible_audit_path).get("audit_status") != "pass":
        violations.append({"reason": "visible_control_audit_not_passed"})

    hidden_vs_morgan = _compare(hidden_dir, baseline_dir)
    hidden_vs_visible = _compare(hidden_dir, visible_dir)
    return {
        "label": label,
        "audit_status": "pass" if not violations else "fail",
        "hidden_batch": str(hidden_dir),
        "visible_batch": str(visible_dir),
        "morgan_batch": str(baseline_dir),
        "top_k_per_group": hidden_manifest.get("top_k_per_group"),
        "min_similarity": hidden_manifest.get("min_similarity"),
        "n_retrievals_identical": 64 - sum(item.get("reason") == "retrieval_not_identical" for item in violations),
        "n_groups": n_groups,
        "n_neighbors": n_neighbors,
        "n_hidden_group_prompts": n_hidden_group_prompts,
        "hidden_metrics": hidden_metrics,
        "visible_metrics": _read_json(visible_dir / "metrics.json"),
        "morgan_metrics": _read_json(baseline_dir / "metrics.json"),
        "hidden_vs_morgan": hidden_vs_morgan,
        "hidden_vs_visible": hidden_vs_visible,
        "violations": violations,
    }


def _runs_by_index(batch_dir: Path) -> dict[int, Path]:
    result = {}
    for path in (batch_dir / "runs").glob("*_idx*"):
        if path.is_dir():
            result[int(path.name.rsplit("idx", 1)[1])] = path
    return result


def _retrieval_signature(retrieval: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (str(group["group_id"]), group.get("neighbors") or [])
        for group in retrieval.get("groups") or []
    ]


def _audit_hidden_prompts(run_dir: Path, retrieval: dict[str, Any]) -> dict[str, Any]:
    group_ids = {str(group["group_id"]) for group in retrieval.get("groups") or [] if group.get("neighbors")}
    violations: list[dict[str, Any]] = []
    n_group_prompts = 0
    for trace in _read_jsonl(run_dir / "trace_messages.jsonl"):
        task = str(trace.get("task") or "")
        user_text = "\n".join(
            str(message.get("content") or "")
            for message in trace.get("messages") or []
            if message.get("role") == "user"
        )
        if task in group_ids:
            n_group_prompts += 1
        if task in group_ids or task == "final":
            leaked = sorted(field for field in FORBIDDEN_PROMPT_FIELDS if field in user_text)
            if leaked:
                violations.append({"reason": "score_field_leak", "task": task, "fields": leaked})
            if SCORE_SEMANTICS in user_text:
                violations.append({"reason": "score_semantics_leak", "task": task})
    if n_group_prompts != len(group_ids):
        violations.append({"reason": "group_prompt_count", "expected": len(group_ids), "observed": n_group_prompts})
    return {"n_group_prompts": n_group_prompts, "violations": violations}


def _report(summary: dict[str, Any]) -> str:
    soft_by_label = {item["label"]: item for item in summary.get("soft_conditions") or []}
    soft_v6_5_by_label = {item["label"]: item for item in summary.get("soft_v6_5_conditions") or []}
    lines = [
        "# Assay-transfer score-visibility ablation",
        "",
        f"Audit status: **{summary['status']}**. Hidden runs preserve assay-transfer ranking while removing scores from LLM prompts.",
        "",
    ]
    if soft_by_label:
        lines.extend([
            f"Soft reranker: `{summary['soft_model']}@{summary['soft_model_revision']}`. Its scores are also hidden from the reasoning model.",
            "",
        ])
    for prefix, title in (("sim0p3", "With similarity floor 0.30"), ("no_floor", "Without a similarity floor")):
        header = "| k | Morgan accuracy | Hidden accuracy | Visible accuracy |"
        separator = "|---:|---:|---:|---:|"
        if soft_by_label:
            header += " Soft accuracy |"
            separator += "---:|"
        header += " Morgan Macro-F1 | Hidden Macro-F1 | Visible Macro-F1 |"
        separator += "---:|---:|---:|"
        if soft_by_label:
            header += " Soft Macro-F1 |"
            separator += "---:|"
        header += " Δ hidden vs Morgan F1 | Δ hidden vs visible F1 | Flips vs visible | Audit |"
        separator += "---:|---:|---:|---|"
        lines.extend([f"## {title}", "", header, separator])
        rows = sorted(
            (item for item in summary["conditions"] if item["label"].startswith(prefix)),
            key=lambda item: int(item["top_k_per_group"]),
        )
        for item in rows:
            morgan = item["morgan_metrics"]
            hidden = item["hidden_metrics"]
            visible = item["visible_metrics"]
            values = [
                str(item["top_k_per_group"]), f"{morgan['accuracy']:.4f}", f"{hidden['accuracy']:.4f}",
                f"{visible['accuracy']:.4f}",
            ]
            soft_item = soft_by_label.get(item["label"])
            if soft_by_label:
                values.append(f"{soft_item['metrics']['accuracy']:.4f}" if soft_item else "")
            values.extend([f"{morgan['macro_f1']:.4f}", f"{hidden['macro_f1']:.4f}", f"{visible['macro_f1']:.4f}"])
            if soft_by_label:
                values.append(f"{soft_item['metrics']['macro_f1']:.4f}" if soft_item else "")
            values.extend([
                f"{item['hidden_vs_morgan']['macro_f1_delta']:+.4f}",
                f"{item['hidden_vs_visible']['macro_f1_delta']:+.4f}",
                str(item["hidden_vs_visible"]["n_prediction_flips"]),
                item["audit_status"] if soft_item is None else f"{item['audit_status']}/{soft_item['audit_status']}",
            ])
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.append("The score-visible and score-hidden outputs are independent validation generations; only the frozen retrieval-independent single-molecule analysis was reused.\n")
    return "\n".join(lines)


def _write_tsv(path: Path, summary: dict[str, Any]) -> None:
    soft_by_label = {item["label"]: item for item in summary.get("soft_conditions") or []}
    soft_v6_5_by_label = {item["label"]: item for item in summary.get("soft_v6_5_conditions") or []}
    fields = [
        "setting",
        "k",
        "min_similarity",
        "morgan_accuracy",
        "morgan_macro_f1",
        "score_hidden_accuracy",
        "score_hidden_macro_f1",
        "soft_score_hidden_accuracy",
        "soft_score_hidden_macro_f1",
        "soft_v6_5_score_hidden_accuracy",
        "soft_v6_5_score_hidden_macro_f1",
        "score_visible_accuracy",
        "score_visible_macro_f1",
        "hidden_vs_morgan_accuracy_delta",
        "hidden_vs_morgan_macro_f1_delta",
        "hidden_vs_visible_accuracy_delta",
        "hidden_vs_visible_macro_f1_delta",
        "prediction_flips_vs_morgan",
        "prediction_flips_vs_visible",
        "successful",
        "total",
        "failed",
        "retrievals_identical",
        "audit_status",
    ]
    ordered = sorted(
        summary["conditions"],
        key=lambda item: (0 if item["label"].startswith("sim0p3") else 1, int(item["top_k_per_group"])),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in ordered:
            hidden = item["hidden_metrics"]
            visible = item["visible_metrics"]
            morgan = item["morgan_metrics"]
            soft = soft_by_label.get(item["label"])
            soft_v6_5 = soft_v6_5_by_label.get(item["label"])
            writer.writerow({
                "setting": "similarity_floor_0.30" if item["label"].startswith("sim0p3") else "no_similarity_floor",
                "k": item["top_k_per_group"],
                "min_similarity": item["min_similarity"],
                "morgan_accuracy": morgan["accuracy"],
                "morgan_macro_f1": morgan["macro_f1"],
                "score_hidden_accuracy": hidden["accuracy"],
                "score_hidden_macro_f1": hidden["macro_f1"],
                "soft_score_hidden_accuracy": soft["metrics"]["accuracy"] if soft else "",
                "soft_score_hidden_macro_f1": soft["metrics"]["macro_f1"] if soft else "",
                "soft_v6_5_score_hidden_accuracy": soft_v6_5["metrics"]["accuracy"] if soft_v6_5 else "",
                "soft_v6_5_score_hidden_macro_f1": soft_v6_5["metrics"]["macro_f1"] if soft_v6_5 else "",
                "score_visible_accuracy": visible["accuracy"],
                "score_visible_macro_f1": visible["macro_f1"],
                "hidden_vs_morgan_accuracy_delta": item["hidden_vs_morgan"]["accuracy_delta"],
                "hidden_vs_morgan_macro_f1_delta": item["hidden_vs_morgan"]["macro_f1_delta"],
                "hidden_vs_visible_accuracy_delta": item["hidden_vs_visible"]["accuracy_delta"],
                "hidden_vs_visible_macro_f1_delta": item["hidden_vs_visible"]["macro_f1_delta"],
                "prediction_flips_vs_morgan": item["hidden_vs_morgan"]["n_prediction_flips"],
                "prediction_flips_vs_visible": item["hidden_vs_visible"]["n_prediction_flips"],
                "successful": hidden["n_successful"],
                "total": hidden["n_total"],
                "failed": hidden["n_failed_runs"],
                "retrievals_identical": item["n_retrievals_identical"],
                "audit_status": item["audit_status"],
            })


def _label_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit(f"Expected LABEL=PATH, found {value!r}")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-batch", action="append", required=True)
    parser.add_argument("--visible-batch", action="append", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--soft-batch", action="append", default=[])
    parser.add_argument("--soft-cache")
    parser.add_argument("--soft-v6-5-batch", action="append", default=[])
    parser.add_argument("--soft-v6-5-cache")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
