"""Build Starling BBB evidence libraries and neighbor indices.

The Starling BBB dataset can be used as a Tier 1 replacement source for BBB
analog retrieval. Two modes are supported:

* qualitative: use Starling labels and text, but do not expose quantitative
  fields to the reasoning prompt.
* all: use the same qualitative context plus parsed quantitative metric/value
  fields when present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rdkit import Chem

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)


SOURCE_DATASET = "starling-labs/BBB"
DEFAULT_SPLIT = "train"
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling"
GROUP_ID = "Tier 1.starling_direct_bbb_evidence"
ENDPOINT_GROUP = "starling_direct_bbb_evidence"
INDEX_VERSION_PREFIX = "bbb_martins_starling_bbb_neighbor_index"
EVIDENCE_FILENAME = "starling_bbb_evidence.jsonl"
INDEX_FILENAME = "starling_bbb_neighbor_index.pkl"
META_FILENAME = "starling_bbb_neighbor_index.meta.json"


SUPPORT_LABELS = {
    "permeable",
    "good_penetration",
    "increased_permeability",
    "high_permeability",
    "passive_diffusion",
    "influx_substrate",
    "not_subject_to_efflux",
}
RISK_LABELS = {
    "poor_penetration",
    "impermeable",
    "low_permeability",
    "restricted",
    "efflux_substrate",
    "efflux_limited",
}
CONTEXT_LABELS = {"transporter_mediated"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    include_numerical = args.mode == "all"
    out_dir = ensure_dir(Path(args.out_root) / args.mode)

    records = _load_hf_records(args.dataset, args.split)
    evidence_rows, stats = build_starling_bbb_evidence_rows(
        records,
        include_numerical=include_numerical,
        max_record_examples=args.max_record_examples,
    )
    print(
        f"[build_starling_bbb_evidence_library] mode={args.mode} "
        f"evidence molecules={len(evidence_rows):,} kept source rows={stats['n_source_rows_kept']:,}",
        flush=True,
    )

    index = build_neighbor_index(
        evidence_rows,
        index_version=f"{INDEX_VERSION_PREFIX}.{args.mode}.v1",
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "starling_bbb",
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "group_id": GROUP_ID,
        "include_numerical": include_numerical,
        "exact_query_exclusion": True,
    }

    evidence_path = out_dir / EVIDENCE_FILENAME
    index_path = out_dir / INDEX_FILENAME
    meta_path = out_dir / META_FILENAME
    _write_jsonl(evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        import pickle

        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "index_version": f"{INDEX_VERSION_PREFIX}.{args.mode}.v1",
        "source_dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "group_id": GROUP_ID,
        "include_numerical": include_numerical,
        **stats,
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "n_groups": len(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "exact_query_exclusion": True,
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {
            "evidence_jsonl": str(evidence_path),
            "index_pkl": str(index_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_starling_bbb_evidence_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    include_numerical: bool,
    max_record_examples: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_cache: dict[str, str] = {}
    stats: Counter[str] = Counter()

    for source_index, raw_record in enumerate(records):
        stats["n_source_rows"] += 1
        record = dict(raw_record)
        smiles = str(record.get("smiles") or "").strip()
        canonical_smiles = canonical_cache.get(smiles)
        if canonical_smiles is None:
            canonical_smiles = _canonicalize_smiles(smiles)
            canonical_cache[smiles] = canonical_smiles
        if not canonical_smiles:
            stats["n_invalid_smiles"] += 1
            continue

        has_qualitative = _has_qualitative_signal(record)
        has_numerical = _has_numerical_signal(record)
        if not has_qualitative and not (include_numerical and has_numerical):
            stats["n_rows_without_selected_signal"] += 1
            continue

        record["_source_index"] = source_index
        record["_canonical_smiles"] = canonical_smiles
        rows_by_smiles[canonical_smiles].append(record)
        stats["n_source_rows_kept"] += 1
        if has_qualitative:
            stats["n_rows_with_qualitative_signal"] += 1
        if has_numerical:
            stats["n_rows_with_numerical_signal"] += 1

    evidence_rows = [
        _summarize_molecule(
            smiles,
            rows,
            include_numerical=include_numerical,
            max_record_examples=max_record_examples,
        )
        for smiles, rows in sorted(rows_by_smiles.items())
    ]
    stats["n_unique_source_smiles"] = len(rows_by_smiles)
    stats["n_molecules_with_numerical_evidence"] = sum(
        any(_has_numerical_signal(row) for row in rows) for rows in rows_by_smiles.values()
    )
    stats["n_molecules_with_qualitative_evidence"] = sum(
        any(_has_qualitative_signal(row) for row in rows) for rows in rows_by_smiles.values()
    )
    return evidence_rows, dict(stats)


def _summarize_molecule(
    smiles: str,
    rows: list[dict[str, Any]],
    *,
    include_numerical: bool,
    max_record_examples: int,
) -> dict[str, Any]:
    molecule_id = "STARLING_BBB_" + hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16].upper()
    labels = _unique_text(
        label
        for row in rows
        for label in (row.get("bbb_permeability_label"), row.get("bbb_transport_label"))
    )
    label_counter = Counter(_normalized_label(label) for label in labels if _normalized_label(label))
    support_count = sum(label_counter[label] for label in SUPPORT_LABELS)
    risk_count = sum(label_counter[label] for label in RISK_LABELS)
    context_count = sum(label_counter[label] for label in CONTEXT_LABELS)
    direction = _evidence_direction(support_count, risk_count, context_count)

    numerical_rows = [row for row in rows if _has_numerical_signal(row)]
    qualitative_rows = [row for row in rows if _has_qualitative_signal(row)]
    examples = _representative_examples(rows, include_numerical=include_numerical, limit=max_record_examples)
    numerical_examples = (
        _representative_examples(numerical_rows, include_numerical=True, limit=max_record_examples)
        if include_numerical
        else []
    )
    standard_type = "Starling BBB evidence"
    standard_value = ""
    standard_units = ""
    if include_numerical and numerical_rows:
        metric_counts = Counter(_clean_text(row.get("quant_metric")) for row in numerical_rows if row.get("quant_metric"))
        unit_counts = Counter(_clean_text(row.get("quant_units")) for row in numerical_rows if row.get("quant_units"))
        standard_type = "Starling BBB quantitative and qualitative evidence"
        if metric_counts:
            standard_type += ": " + ", ".join(metric for metric, _ in metric_counts.most_common(3))
        standard_value = "; ".join(
            _quant_value_text(row)
            for row in numerical_examples[:3]
            if _quant_value_text(row)
        )
        standard_units = ", ".join(unit for unit, _ in unit_counts.most_common(3))
    elif labels:
        standard_type += ": " + ", ".join(labels[:4])

    activity_comment = _activity_comment(
        rows,
        labels=labels,
        include_numerical=include_numerical,
        numerical_rows=numerical_rows,
        direction=direction,
    )
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "assay_chembl_id": "STARLING_BBB",
        "assay_tier": "Tier 1",
        "endpoint_group": ENDPOINT_GROUP,
        "group_id": GROUP_ID,
        "standard_type": standard_type,
        "standard_relation": "",
        "standard_value": standard_value,
        "standard_units": standard_units,
        "pchembl_value": "",
        "activity_comment": activity_comment,
        "data_validity_comment": "",
        "assay_description": _record_examples_text(examples, include_numerical=include_numerical, max_chars=5000),
        "target_pref_name": "blood-brain barrier permeability and brain exposure",
        "target_genes": "",
        "organism": "; ".join(_unique_text(row.get("species") for row in rows)[:8]),
        "confidence_score": "",
        "relationship_type": "",
        "evidence_direction": direction,
        "evidence_strength": "strong" if include_numerical and numerical_rows else "moderate",
        "endpoint_group_reason": "Starling literature-derived BBB molecule evidence used as Tier 1 retrieval source.",
        "evidence_source": SOURCE_DATASET,
        "source_record_count": len(rows),
        "source_qualitative_record_count": len(qualitative_rows),
        "source_numeric_record_count": len(numerical_rows),
        "source_permeability_labels": _unique_text(row.get("bbb_permeability_label") for row in rows),
        "source_transport_labels": _unique_text(row.get("bbb_transport_label") for row in rows),
        "source_pmids": _unique_text(row.get("pmid") for row in rows)[:20],
        "source_support_texts": _unique_text(row.get("support_text") for row in rows)[:max_record_examples],
        "source_record_examples": examples,
        "source_numerical_examples": numerical_examples,
        "source_numerical_included_in_llm_text": include_numerical,
    }


def _activity_comment(
    rows: list[dict[str, Any]],
    *,
    labels: list[str],
    include_numerical: bool,
    numerical_rows: list[dict[str, Any]],
    direction: str,
) -> str:
    parts = [
        f"Starling BBB literature summary over {len(rows)} source records.",
        f"Qualitative labels: {', '.join(labels) if labels else 'not specified'}.",
        f"Direction summary: {direction}.",
    ]
    if include_numerical:
        parts.append(f"Quantitative records retained: {len(numerical_rows)}.")
    else:
        parts.append("Quantitative metric/value fields intentionally omitted for this no-numerical retrieval variant.")
    return " ".join(parts)


def _record_examples_text(
    examples: list[dict[str, Any]],
    *,
    include_numerical: bool,
    max_chars: int,
) -> str:
    blocks = []
    for example in examples:
        fields = [
            f"permeability_label: {example.get('bbb_permeability_label') or 'not specified'}",
            f"transport_label: {example.get('bbb_transport_label') or 'not specified'}",
            f"assay_model: {example.get('assay_model') or 'not specified'}",
            f"species: {example.get('species') or 'not specified'}",
            f"qualifying_conditions: {example.get('qualifying_conditions') or 'not specified'}",
            f"extra_details: {example.get('extra_details') or 'not specified'}",
        ]
        if include_numerical:
            fields.append(f"quantitative_metric: {example.get('quant_metric') or 'not specified'}")
            fields.append(f"quantitative_value: {example.get('quant_value') or 'not specified'}")
            fields.append(f"quantitative_units: {example.get('quant_units') or 'not specified'}")
        fields.append(f"support_text: {example.get('support_text') or 'not specified'}")
        blocks.append("\n".join(fields))
    text = "\n\n".join(blocks)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _representative_examples(
    rows: list[dict[str, Any]],
    *,
    include_numerical: bool,
    limit: int,
) -> list[dict[str, Any]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in sorted(rows, key=lambda item: int(item.get("_source_index") or 0)):
        key = (
            _clean_text(row.get("bbb_permeability_label")),
            _clean_text(row.get("bbb_transport_label")),
            _clean_text(row.get("quant_metric")) if include_numerical else "",
            _clean_text(row.get("quant_value")) if include_numerical else "",
            _clean_text(row.get("support_text")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return [_example_row(row, include_numerical=include_numerical) for row in _evenly_spaced_items(unique_rows, limit)]


def _example_row(row: Mapping[str, Any], *, include_numerical: bool) -> dict[str, Any]:
    output = {
        "source_index": row.get("_source_index", ""),
        "pmid": _clean_text(row.get("pmid")),
        "support_text": _clean_text(row.get("support_text")),
        "bbb_permeability_label": _clean_text(row.get("bbb_permeability_label")),
        "bbb_transport_label": _clean_text(row.get("bbb_transport_label")),
        "assay_model": _clean_text(row.get("assay_model")),
        "species": _clean_text(row.get("species")),
        "qualifying_conditions": _clean_text(row.get("qualifying_conditions")),
        "extra_details": _clean_text(row.get("extra_details")),
    }
    if include_numerical:
        output.update(
            {
                "quant_metric": _clean_text(row.get("quant_metric")),
                "quant_value": _clean_text(row.get("quant_value")),
                "quant_units": _clean_text(row.get("quant_units")),
            }
        )
    return output


def _evidence_direction(support_count: int, risk_count: int, context_count: int) -> str:
    if support_count > risk_count:
        return "supports_bbb_crossing"
    if risk_count > support_count:
        return "argues_against_bbb_crossing"
    if context_count:
        return "transporter_or_context_dependent"
    return "neutral_or_unclear"


def _has_qualitative_signal(row: Mapping[str, Any]) -> bool:
    return bool(_clean_text(row.get("bbb_permeability_label")) or _clean_text(row.get("bbb_transport_label")))


def _has_numerical_signal(row: Mapping[str, Any]) -> bool:
    return bool(_clean_text(row.get("quant_metric")) or _clean_text(row.get("quant_value")))


def _quant_value_text(row: Mapping[str, Any]) -> str:
    metric = _clean_text(row.get("quant_metric"))
    value = _clean_text(row.get("quant_value"))
    units = _clean_text(row.get("quant_units"))
    if not (metric or value):
        return ""
    return " ".join(part for part in [metric, value, units] if part)


def _normalized_label(value: Any) -> str:
    return _clean_text(value).lower().replace("-", "_").replace(" ", "_")


def _unique_text(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _evenly_spaced_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[index] for index in dict.fromkeys(indices)]


def _canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _load_hf_records(dataset: str, split: str) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(dataset, split=split)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=SOURCE_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--mode", choices=["qualitative", "all"], default="qualitative")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
