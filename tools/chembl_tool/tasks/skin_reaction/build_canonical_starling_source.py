"""Build auditable, mutually exclusive direct and AOP Skin Starling evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Mapping

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    AOP_PARTITION,
    AOP_RECORDS_PATH,
    CANONICAL_SOURCE_DIR,
    CANONICAL_VERSION,
    DEDUP_AUDIT_PATH,
    DIRECT_PARTITION,
    DIRECT_RECORDS_PATH,
    MANIFEST_PATH,
    PARTITION_AUDIT_PATH,
    RAW_AOP_PATH,
    RAW_DIRECT_PATH,
    REJECT_PARTITION,
    PartitionDecision,
    classify_aop_source_record,
    classify_direct_source_record,
    normalized_direct_label,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    import pandas as pd

    direct_path = Path(args.direct_source)
    aop_path = Path(args.aop_source)
    direct_frame = pd.read_parquet(direct_path)
    aop_frame = pd.read_parquet(aop_path)
    outputs = build_canonical_frames(direct_frame, aop_frame)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "direct_records": out_dir / DIRECT_RECORDS_PATH.name,
        "aop_records": out_dir / AOP_RECORDS_PATH.name,
        "partition_audit": out_dir / PARTITION_AUDIT_PATH.name,
        "dedup_audit": out_dir / DEDUP_AUDIT_PATH.name,
    }
    for name, path in paths.items():
        outputs[name].to_parquet(path, index=False)

    manifest = {
        "contract_version": CANONICAL_VERSION,
        "source_inputs": {
            "direct": {"path": str(direct_path), "sha256": sha256_file(direct_path)},
            "aop": {"path": str(aop_path), "sha256": sha256_file(aop_path)},
        },
        "partition_contract": {
            "direct": "validated final sensitization/contact-allergy outcomes only",
            "aop": "experimental MIE, KE2, KE3, or KE4 evidence only",
            "excluded": [
                "phototoxicity/photoallergy/photoirritation",
                "irritation/corrosion",
                "prediction-only or in-silico evidence",
                "integrated or unresolved endpoints",
            ],
            "mutually_exclusive": True,
            "raw_inputs_immutable": True,
            "identity_normalizer": IDENTITY_NORMALIZER_VERSION,
        },
        "stats": outputs["stats"],
        "paths": {},
    }
    for name, path in paths.items():
        manifest["paths"][name] = str(path)
        manifest["paths"][f"{name}_sha256"] = sha256_file(path)
    manifest_path = out_dir / MANIFEST_PATH.name
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_canonical_frames(direct_frame: Any, aop_frame: Any) -> dict[str, Any]:
    """Return inspectable canonical frames without mutating either raw input."""
    import pandas as pd

    accepted: dict[str, list[dict[str, Any]]] = {
        DIRECT_PARTITION: [],
        AOP_PARTITION: [],
    }
    audit: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    invalid_smiles = 0

    sources = (
        ("direct_skin_reaction", direct_frame, classify_direct_source_record),
        ("sensitization_aop", aop_frame, classify_aop_source_record),
    )
    for source_name, frame, classifier in sources:
        for source_index, row in enumerate(frame.to_dict(orient="records")):
            decision = classifier(row)
            identity = normalize_molecule_identity(_text(row.get("SMILES")))
            if decision.partition != REJECT_PARTITION and (
                identity.status != "ok" or not identity.parent_inchi_key
            ):
                decision = PartitionDecision(REJECT_PARTITION, "invalid_or_missing_smiles")
                invalid_smiles += 1
            audit_row = {
                "source_partition": source_name,
                "source_index": source_index,
                "source_record_id": _source_record_id(source_name, source_index),
                "pmid": _text(row.get("pmid")),
                "extraction_id": _text(row.get("extraction_id")),
                "smiles": _text(row.get("SMILES")),
                "parent_inchi_key": identity.parent_inchi_key,
                "partition": decision.partition,
                "partition_reason": decision.reason,
                "canonical_aop_event": decision.aop_event,
                "deduplicated": False,
                "deduplicated_into": "",
            }
            audit.append(audit_row)
            decisions[f"{source_name}:{decision.partition}:{decision.reason}"] += 1
            if decision.partition == REJECT_PARTITION:
                continue
            normalized = (
                _to_direct_record(row, source_name, source_index, identity.parent_inchi_key)
                if decision.partition == DIRECT_PARTITION
                else _to_aop_record(
                    row,
                    source_name,
                    source_index,
                    identity.parent_inchi_key,
                    decision.aop_event,
                )
            )
            accepted[decision.partition].append(normalized)

    direct_rows, direct_dedup = _deduplicate(accepted[DIRECT_PARTITION], DIRECT_PARTITION)
    aop_rows, aop_dedup = _deduplicate(accepted[AOP_PARTITION], AOP_PARTITION)
    duplicate_map = {
        row["duplicate_source_record_id"]: row["retained_source_record_id"]
        for row in [*direct_dedup, *aop_dedup]
    }
    for row in audit:
        retained = duplicate_map.get(row["source_record_id"])
        if retained:
            row["deduplicated"] = True
            row["deduplicated_into"] = retained

    if len(audit) != len(direct_frame) + len(aop_frame):
        raise AssertionError("canonical partition audit does not reconcile to raw inputs")
    if any(row["aop_event"] not in {"MIE_protein_binding", "KE2_keratinocyte_activation", "KE3_dendritic_cell_activation", "KE4_T_cell_activation"} for row in aop_rows):
        raise AssertionError("non-key-event evidence leaked into canonical AOP")
    direct_ids = {row["source_record_id"] for row in direct_rows}
    aop_ids = {row["source_record_id"] for row in aop_rows}
    if direct_ids & aop_ids:
        raise AssertionError("source record leaked into both direct and AOP")

    stats = {
        "n_raw_direct_rows": len(direct_frame),
        "n_raw_aop_rows": len(aop_frame),
        "n_partition_audit_rows": len(audit),
        "n_canonical_direct_before_dedup": len(accepted[DIRECT_PARTITION]),
        "n_canonical_direct_records": len(direct_rows),
        "n_canonical_aop_before_dedup": len(accepted[AOP_PARTITION]),
        "n_canonical_aop_records": len(aop_rows),
        "n_direct_duplicates_removed": len(direct_dedup),
        "n_aop_duplicates_removed": len(aop_dedup),
        "n_invalid_or_missing_smiles": invalid_smiles,
        "decision_counts": dict(sorted(decisions.items())),
        "canonical_direct_source_counts": dict(
            sorted(Counter(row["source_partition"] for row in direct_rows).items())
        ),
        "canonical_aop_source_counts": dict(
            sorted(Counter(row["source_partition"] for row in aop_rows).items())
        ),
        "canonical_aop_event_counts": dict(
            sorted(Counter(row["aop_event"] for row in aop_rows).items())
        ),
        "partition_reconciles": len(audit) == len(direct_frame) + len(aop_frame),
        "direct_aop_source_record_overlap": len(direct_ids & aop_ids),
    }
    return {
        "direct_records": pd.DataFrame(direct_rows),
        "aop_records": pd.DataFrame(aop_rows),
        "partition_audit": pd.DataFrame(audit),
        "dedup_audit": pd.DataFrame([*direct_dedup, *aop_dedup]),
        "stats": stats,
    }


def _to_direct_record(
    row: Mapping[str, Any], source: str, source_index: int, parent_key: str
) -> dict[str, Any]:
    if source == "direct_skin_reaction":
        result = {
            "pmid": _text(row.get("pmid")),
            "extraction_id": _text(row.get("extraction_id")),
            "confidence": _number(row.get("confidence")),
            "paragraph_idx": _number(row.get("paragraph_idx")),
            "support_text": _text(row.get("support_text")),
            "outcome_label": normalized_direct_label(row.get("outcome_label"))
            or _text(row.get("outcome_label")).lower(),
            "reaction_type": _text(row.get("reaction_type")),
            "assay_or_test": _text(row.get("assay_or_test")),
            "species_or_population": _text(row.get("species_or_population")),
            "dose_or_concentration": _text(row.get("dose_or_concentration")),
            "positive_count": _text(row.get("positive_count")),
            "total_tested": _text(row.get("total_tested")),
            "effect_metric": _text(row.get("effect_metric")),
            "extra_details": _text(row.get("extra_details")),
            "SMILES": _text(row.get("SMILES")),
        }
    else:
        result = {
            "pmid": _text(row.get("pmid")),
            "extraction_id": _text(row.get("extraction_id")),
            "confidence": _number(row.get("confidence")),
            "paragraph_idx": _number(row.get("paragraph_idx")),
            "support_text": _text(row.get("support_text")),
            "outcome_label": normalized_direct_label(row.get("result_label")),
            "reaction_type": "sensitization",
            "assay_or_test": _text(row.get("assay_type")),
            "species_or_population": _text(row.get("experimental_conditions")),
            "dose_or_concentration": "",
            "positive_count": "",
            "total_tested": "",
            "effect_metric": " ".join(
                value
                for value in (
                    _text(row.get("result_value")),
                    _text(row.get("result_unit")),
                )
                if value
            ),
            "extra_details": _join_nonempty(
                row.get("endpoint_or_target"),
                row.get("qualifying_conditions"),
                row.get("extra_details"),
            ),
            "SMILES": _text(row.get("SMILES")),
        }
    result.update(_provenance(source, source_index, row, parent_key))
    return result


def _to_aop_record(
    row: Mapping[str, Any],
    source: str,
    source_index: int,
    parent_key: str,
    aop_event: str,
) -> dict[str, Any]:
    if source == "sensitization_aop":
        result = {
            "paragraph_idx": _number(row.get("paragraph_idx")),
            "support_text": _text(row.get("support_text")),
            "global_identifier": _text(row.get("global_identifier")),
            "assay_type": _text(row.get("assay_type")),
            "aop_event": aop_event,
            "endpoint_or_target": _text(row.get("endpoint_or_target")),
            "result_label": _text(row.get("result_label")),
            "result_value": _text(row.get("result_value")),
            "result_unit": _text(row.get("result_unit")),
            "experimental_conditions": _text(row.get("experimental_conditions")),
            "qualifying_conditions": _text(row.get("qualifying_conditions")),
            "extra_details": _text(row.get("extra_details")),
            "confidence": _number(row.get("confidence")),
            "needs_more_context": _text(row.get("needs_more_context")),
            "pmid": _text(row.get("pmid")),
            "extraction_id": _text(row.get("extraction_id")),
            "SMILES": _text(row.get("SMILES")),
        }
    else:
        result = {
            "paragraph_idx": _number(row.get("paragraph_idx")),
            "support_text": _text(row.get("support_text")),
            "global_identifier": "",
            "assay_type": _text(row.get("assay_or_test")),
            "aop_event": aop_event,
            "endpoint_or_target": "",
            "result_label": normalized_direct_label(row.get("outcome_label")) or _text(
                row.get("outcome_label")
            ).lower(),
            "result_value": _text(row.get("effect_metric")),
            "result_unit": "",
            "experimental_conditions": _join_nonempty(
                row.get("species_or_population"), row.get("dose_or_concentration")
            ),
            "qualifying_conditions": "",
            "extra_details": _text(row.get("extra_details")),
            "confidence": _number(row.get("confidence")),
            "needs_more_context": "",
            "pmid": _text(row.get("pmid")),
            "extraction_id": _text(row.get("extraction_id")),
            "SMILES": _text(row.get("SMILES")),
        }
    result.update(_provenance(source, source_index, row, parent_key))
    return result


def _provenance(
    source: str, source_index: int, row: Mapping[str, Any], parent_key: str
) -> dict[str, Any]:
    return {
        "source_partition": source,
        "source_index": source_index,
        "source_record_id": _source_record_id(source, source_index),
        "source_extraction_id": _text(row.get("extraction_id")),
        "parent_inchi_key": parent_key,
        "canonical_contract_version": CANONICAL_VERSION,
    }


def _deduplicate(
    rows: list[dict[str, Any]], partition: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        endpoint = row.get("outcome_label") if partition == DIRECT_PARTITION else row.get("aop_event")
        result = row.get("outcome_label") if partition == DIRECT_PARTITION else row.get("result_label")
        key = (
            _text(row.get("parent_inchi_key")),
            _normalized_text(row.get("pmid")),
            _normalized_text(row.get("support_text")),
            _normalized_text(endpoint),
            _normalized_text(result),
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            retained.append(row)
            continue
        duplicates.append(
            {
                "partition": partition,
                "retained_source_record_id": existing["source_record_id"],
                "duplicate_source_record_id": row["source_record_id"],
                "parent_inchi_key": row["parent_inchi_key"],
                "pmid": row.get("pmid", ""),
                "endpoint": endpoint,
                "result": result,
                "dedup_reason": "same_parent_pmid_support_endpoint_result",
            }
        )
    return retained, duplicates


def _source_record_id(source: str, source_index: int) -> str:
    return f"{source}:{source_index}"


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).lower()).strip()


def _join_nonempty(*values: Any) -> str:
    return " | ".join(value for value in (_text(item) for item in values) if value)


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return value


def _number(value: Any) -> float | None:
    value = _clean(value)
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    value = _clean(value)
    return str(value).strip() if value != "" else ""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-source", default=str(RAW_DIRECT_PATH))
    parser.add_argument("--aop-source", default=str(RAW_AOP_PATH))
    parser.add_argument("--out-dir", default=str(CANONICAL_SOURCE_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
