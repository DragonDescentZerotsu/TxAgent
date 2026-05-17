"""Audit ClinTox reasoning batch errors and evidence patterns."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.tasks.clintox.constants import (
    CLINTOX_NEGATIVE_PREDICTION,
    CLINTOX_POSITIVE_PREDICTION,
)


DEFAULT_INPUT_JSONL = "data/processed/ClinTox/test.jsonl"
DEFAULT_BATCH_DIR = "outputs/chembl_tool/tasks/clintox/reasoning/batches/clintox_full_prompt_v7_final_only_from_v2"

EVIDENCE_CATEGORIES = {
    "clinical_or_human": [
        "human",
        "trial",
        "withdraw",
        "black box",
        "boxed warning",
        "postmarketing",
    ],
    "dili_hepatic": [
        "dili",
        "hepatotoxic",
        "liver",
        "hepatic",
        "bsep",
        "cholestasis",
        "bilirubin",
    ],
    "cardiac_ion_channel": [
        "herg",
        "qt",
        "ikr",
        "5-ht2b",
        "valvulopathy",
        "cardiac",
        "ion channel",
        "calcium channel",
        "sodium channel",
        "cav",
    ],
    "cytotoxic_oncology": [
        "cytotoxic",
        "cytotoxicity",
        "antitumor",
        "alkylating",
        "antimetabolite",
        "myelosuppression",
        "marrow",
        "dna",
        "genotoxic",
        "mitotic",
    ],
    "cns_or_narrow_ti": [
        "barbiturate",
        "gaba",
        "respiratory depression",
        "cns depression",
        "sedative",
        "seizure",
        "neurotox",
        "narrow therapeutic",
    ],
    "endocrine_or_repro": [
        "estrogen",
        "androgen",
        "endocrine",
        "uterine",
        "reproductive",
        "aromatase",
    ],
    "mitochondrial_or_redox": [
        "mitochondrial",
        "ros",
        "redox",
        "oxidative",
        "quinone",
    ],
    "ddi_or_exposure": [
        "cyp",
        "transporter",
        "oatp",
        "oct",
        "mate",
        "drug interaction",
        "ddi",
        "exposure",
    ],
    "evidence_sparse_or_weak": [
        "no direct",
        "absence of",
        "insufficient",
        "no coherent",
        "no usable",
        "weak analog",
        "low transferability",
        "evidence gap",
    ],
}

FINAL_TEXT_FIELDS = [
    "final_summary",
    "main_reasons",
]

CONTEXT_TEXT_FIELDS = [
    "final_summary",
    "main_reasons",
    "single_molecule_assessment",
    "clinical_or_human_safety_assessment",
    "in_vivo_toxicology_assessment",
    "organ_safety_pharmacology_assessment",
    "genotoxicity_or_carcinogenicity_assessment",
    "cell_stress_and_cytotoxicity_assessment",
    "offtarget_ddi_or_exposure_assessment",
    "conflicting_evidence",
    "evidence_gaps",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    batch_dir = Path(args.batch_dir)
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_rows = _read_jsonl(Path(args.input_jsonl))
    rows = _load_prediction_rows(batch_dir)
    fill_rows = [_load_prediction_rows(Path(path)) for path in args.fill_missing_from]
    rows = _merge_missing_predictions(rows, fill_rows)

    audit_rows = [_build_audit_row(row, input_rows) for row in rows]
    metrics = _compute_metrics(audit_rows)
    group_summary, group_rows = _summarize_groups(audit_rows)
    category_summary = _summarize_categories(audit_rows)
    connectivity_conflicts = [] if args.skip_connectivity_conflicts else _find_connectivity_label_conflicts(input_rows)

    summary = {
        "batch_dir": str(batch_dir),
        "input_jsonl": args.input_jsonl,
        "fill_missing_from": args.fill_missing_from,
        "metrics": metrics,
        "category_summary": category_summary,
        "group_summary": group_summary,
        "connectivity_label_conflicts": connectivity_conflicts,
    }

    _write_json(out_dir / "audit_summary.json", summary)
    _write_error_cases(out_dir / "error_cases.csv", audit_rows)
    _write_group_rows(out_dir / "group_direction_by_confusion.csv", group_rows)
    _write_report(out_dir / "report.md", summary, audit_rows)
    print(out_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default=DEFAULT_BATCH_DIR)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--fill-missing-from",
        action="append",
        default=[],
        help="Optional batch directory whose predictions fill missing/error predictions by query_index.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-connectivity-conflicts", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_prediction_rows(batch_dir: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(batch_dir / "predictions.jsonl")
    return {int(row["query_index"]): row for row in rows}


def _merge_missing_predictions(
    base_rows: dict[int, dict[str, Any]],
    fill_batches: list[dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged = dict(base_rows)
    for fill_rows in fill_batches:
        for query_index, fill_row in fill_rows.items():
            current = merged.get(query_index)
            if current is None or _prediction_to_label(current.get("clintox_prediction")) is None:
                if _prediction_to_label(fill_row.get("clintox_prediction")) is not None:
                    patched = dict(fill_row)
                    patched["filled_from_batch"] = str(fill_row.get("run_dir", "")).split("/runs/", 1)[0]
                    merged[query_index] = patched
    return [merged[index] for index in sorted(merged)]


def _build_audit_row(row: dict[str, Any], input_rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_index = int(row["query_index"])
    label = _parse_label(row.get("label"))
    if label is None and 0 <= query_index < len(input_rows):
        label = _parse_label(input_rows[query_index].get("Y"))
    prediction = str(row.get("clintox_prediction") or "missing")
    pred_label = _prediction_to_label(prediction)
    confusion = _confusion_name(label, pred_label)
    final_content = _read_final_content(row.get("final_reasoning_output"))
    final_text = _text_from_fields(final_content, FINAL_TEXT_FIELDS, fallback=row.get("final_summary", ""))
    context_text = _text_from_fields(final_content, CONTEXT_TEXT_FIELDS, fallback=row.get("final_summary", ""))
    categories = _match_categories(final_text)
    context_categories = _match_categories(context_text)
    group_stats = _group_stats(row.get("source_run_dir") or row.get("run_dir"))
    return {
        "query_index": query_index,
        "smiles": row.get("smiles") or (input_rows[query_index].get("drug") if query_index < len(input_rows) else ""),
        "label": label,
        "prediction": prediction,
        "pred_label": pred_label,
        "confusion": confusion,
        "confidence": row.get("confidence") or final_content.get("confidence"),
        "correct": label is not None and pred_label is not None and label == pred_label,
        "status": row.get("status"),
        "run_id": row.get("run_id"),
        "run_dir": row.get("run_dir"),
        "source_run_dir": row.get("source_run_dir"),
        "filled_from_batch": row.get("filled_from_batch", ""),
        "final_summary": final_content.get("final_summary") or row.get("final_summary", ""),
        "categories": categories,
        "context_categories": context_categories,
        "group_stats": group_stats,
    }


def _parse_label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if label in (0, 1) else None


def _prediction_to_label(prediction: Any) -> int | None:
    normalized = str(prediction or "").strip().lower()
    if normalized == CLINTOX_POSITIVE_PREDICTION:
        return 1
    if normalized == CLINTOX_NEGATIVE_PREDICTION:
        return 0
    return None


def _confusion_name(label: int | None, pred_label: int | None) -> str:
    if label is None:
        return "unlabeled"
    if pred_label is None:
        return "missing"
    if label == 1 and pred_label == 1:
        return "TP"
    if label == 1 and pred_label == 0:
        return "FN"
    if label == 0 and pred_label == 1:
        return "FP"
    return "TN"


def _read_final_content(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload.get("llm"), dict) and isinstance(payload["llm"].get("content"), dict):
        return payload["llm"]["content"]
    return payload if isinstance(payload, dict) else {}


def _text_from_fields(content: dict[str, Any], fields: list[str], fallback: str = "") -> str:
    parts = [fallback]
    for field in fields:
        value = content.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(part for part in parts if part).lower()


def _match_categories(text: str) -> list[str]:
    matches = [name for name, keywords in EVIDENCE_CATEGORIES.items() if any(keyword in text for keyword in keywords)]
    return matches or ["uncategorized"]


def _group_stats(run_dir_value: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": 0,
        "ok": 0,
        "errors": 0,
        "useful": 0,
        "directions": Counter(),
        "confidence": Counter(),
        "transferability": Counter(),
        "groups": Counter(),
        "error_messages": Counter(),
    }
    if not run_dir_value:
        return stats
    path = Path(str(run_dir_value)) / "group_reasoning_outputs.jsonl"
    if not path.exists():
        return stats
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                group = json.loads(line)
            except json.JSONDecodeError:
                stats["errors"] += 1
                continue
            stats["total"] += 1
            if group.get("status") != "ok":
                stats["errors"] += 1
                stats["error_messages"][str(group.get("error") or group.get("status") or "unknown")[:180]] += 1
                continue
            stats["ok"] += 1
            content = _group_content(group)
            if content.get("useful_for_clintox_reasoning") is True:
                stats["useful"] += 1
                stats["directions"][str(content.get("evidence_direction") or "missing")] += 1
                stats["confidence"][str(content.get("confidence") or "missing")] += 1
                stats["transferability"][str(content.get("transferability") or "missing")] += 1
                stats["groups"][str(group.get("group_id") or content.get("group_id") or "missing")] += 1
    return stats


def _group_content(group: dict[str, Any]) -> dict[str, Any]:
    llm = group.get("llm")
    if isinstance(llm, dict) and isinstance(llm.get("content"), dict):
        return llm["content"]
    return group


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["label"] in (0, 1)]
    confusion = Counter(row["confusion"] for row in evaluable)
    correct = confusion["TP"] + confusion["TN"]
    per_class = {
        "0": _class_metrics(confusion["TN"], confusion["FN"], confusion["FP"]),
        "1": _class_metrics(confusion["TP"], confusion["FP"], confusion["FN"]),
    }
    return {
        "n_total": len(rows),
        "n_evaluable": len(evaluable),
        "accuracy": _safe_div(correct, len(evaluable)),
        "macro_f1": round((per_class["0"]["f1"] + per_class["1"]["f1"]) / 2, 6),
        "positive_class_precision": per_class["1"]["precision"],
        "positive_class_recall": per_class["1"]["recall"],
        "positive_class_f1": per_class["1"]["f1"],
        "confusion_matrix": {
            "tn": confusion["TN"],
            "fp": confusion["FP"],
            "fn": confusion["FN"],
            "tp": confusion["TP"],
            "missing": confusion["missing"],
        },
        "prediction_distribution": dict(Counter(row["prediction"] for row in evaluable)),
        "per_class": per_class,
    }


def _class_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _safe_div(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _summarize_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_confusion: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        confusion = row["confusion"]
        group_stats = row["group_stats"]
        by_confusion[confusion]["molecules"]["n"] += 1
        for field in ("total", "ok", "errors", "useful"):
            by_confusion[confusion]["counts"][field] += int(group_stats.get(field, 0))
        for field in ("directions", "confidence", "transferability", "groups", "error_messages"):
            by_confusion[confusion][field].update(group_stats.get(field, {}))
    for confusion, sections in sorted(by_confusion.items()):
        for field in ("directions", "confidence", "transferability", "groups", "error_messages"):
            for key, count in sections[field].most_common():
                csv_rows.append({"confusion": confusion, "section": field, "value": key, "count": count})
    summary = {
        confusion: {
            section: dict(counter.most_common(30)) for section, counter in sections.items()
        }
        for confusion, sections in sorted(by_confusion.items())
    }
    return summary, csv_rows


def _summarize_categories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for confusion in sorted({row["confusion"] for row in rows}):
        counter = Counter()
        examples: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            if row["confusion"] != confusion:
                continue
            for category in row["categories"]:
                counter[category] += 1
                if len(examples[category]) < 10:
                    examples[category].append(row["query_index"])
        out[confusion] = {
            "counts": dict(counter.most_common()),
            "examples": {key: value for key, value in examples.items()},
        }
    return out


def _find_connectivity_label_conflicts(input_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover - depends on local environment
        return [{"status": "rdkit_unavailable", "error": str(exc)}]

    by_connectivity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(input_rows):
        smiles = row.get("drug") or row.get("smiles") or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            continue
        inchi_key = Chem.MolToInchiKey(mol)
        connectivity = inchi_key.split("-", 1)[0]
        by_connectivity[connectivity].append(
            {"query_index": index, "label": _parse_label(row.get("Y")), "smiles": smiles, "inchi_key": inchi_key}
        )
    conflicts = []
    for connectivity, items in sorted(by_connectivity.items()):
        labels = {item["label"] for item in items}
        if len(labels) > 1:
            conflicts.append({"connectivity_inchi_key": connectivity, "items": items})
    return conflicts


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _write_error_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_index",
        "label",
        "prediction",
        "pred_label",
        "confusion",
        "confidence",
        "status",
        "categories",
        "group_total",
        "group_errors",
        "group_useful",
        "top_group_directions",
        "run_id",
        "source_run_dir",
        "final_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["confusion"] not in {"FP", "FN", "missing"}:
                continue
            group_stats = row["group_stats"]
            writer.writerow(
                {
                    "query_index": row["query_index"],
                    "label": row["label"],
                    "prediction": row["prediction"],
                    "pred_label": row["pred_label"],
                    "confusion": row["confusion"],
                    "confidence": row["confidence"],
                    "status": row["status"],
                    "categories": ";".join(row["categories"]),
                    "group_total": group_stats.get("total", 0),
                    "group_errors": group_stats.get("errors", 0),
                    "group_useful": group_stats.get("useful", 0),
                    "top_group_directions": json.dumps(dict(group_stats.get("directions", {}).most_common(8)), ensure_ascii=False),
                    "run_id": row["run_id"],
                    "source_run_dir": row["source_run_dir"],
                    "final_summary": row["final_summary"],
                }
            )


def _write_group_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["confusion", "section", "value", "count"])
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# ClinTox Reasoning Batch Audit",
        "",
        f"- batch_dir: `{summary['batch_dir']}`",
        f"- input_jsonl: `{summary['input_jsonl']}`",
        f"- fill_missing_from: `{json.dumps(summary['fill_missing_from'], ensure_ascii=False)}`",
        f"- accuracy: {metrics['accuracy']}",
        f"- macro_f1: {metrics['macro_f1']}",
        f"- positive_class_precision: {metrics['positive_class_precision']}",
        f"- positive_class_recall: {metrics['positive_class_recall']}",
        f"- positive_class_f1: {metrics['positive_class_f1']}",
        f"- confusion_matrix: `{json.dumps(metrics['confusion_matrix'], ensure_ascii=False)}`",
        f"- prediction_distribution: `{json.dumps(metrics['prediction_distribution'], ensure_ascii=False)}`",
        "",
        "## Category Summary",
        "",
    ]
    for confusion in ("FN", "FP", "TP", "TN", "missing"):
        item = summary["category_summary"].get(confusion)
        if not item:
            continue
        lines.append(f"### {confusion}")
        lines.append("")
        lines.append(f"- counts: `{json.dumps(item['counts'], ensure_ascii=False)}`")
        lines.append(f"- examples: `{json.dumps(item['examples'], ensure_ascii=False)}`")
        lines.append("")

    lines.extend(["## Group Evidence Summary", ""])
    for confusion in ("FN", "FP", "TP", "TN", "missing"):
        item = summary["group_summary"].get(confusion)
        if not item:
            continue
        lines.append(f"### {confusion}")
        lines.append("")
        lines.append(f"- counts: `{json.dumps(item.get('counts', {}), ensure_ascii=False)}`")
        lines.append(f"- directions: `{json.dumps(item.get('directions', {}), ensure_ascii=False)}`")
        lines.append(f"- confidence: `{json.dumps(item.get('confidence', {}), ensure_ascii=False)}`")
        lines.append(f"- transferability: `{json.dumps(item.get('transferability', {}), ensure_ascii=False)}`")
        lines.append(f"- top_groups: `{json.dumps(dict(list(item.get('groups', {}).items())[:12]), ensure_ascii=False)}`")
        lines.append("")

    conflicts = summary["connectivity_label_conflicts"]
    lines.extend(["## Connectivity Label Conflicts", ""])
    if conflicts:
        for conflict in conflicts:
            if conflict.get("status") == "rdkit_unavailable":
                lines.append(f"- RDKit unavailable: `{conflict.get('error')}`")
                continue
            pairs = ", ".join(f"idx{item['query_index']}:Y={item['label']}" for item in conflict["items"])
            lines.append(f"- `{conflict['connectivity_inchi_key']}`: {pairs}")
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Error Cases", ""])
    lines.append("| index | type | label | prediction | confidence | group_total | group_errors | group_useful | categories | summary |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["confusion"] not in {"FP", "FN", "missing"}:
            continue
        group_stats = row["group_stats"]
        summary_text = str(row["final_summary"]).replace("\n", " ")[:260]
        lines.append(
            f"| {row['query_index']} | {row['confusion']} | {row['label']} | {row['prediction']} | "
            f"{row['confidence']} | {group_stats.get('total', 0)} | {group_stats.get('errors', 0)} | "
            f"{group_stats.get('useful', 0)} | {','.join(row['categories'])} | {summary_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
