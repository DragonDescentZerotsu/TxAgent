"""Export every upstream record for frozen Bioavailability gold parents.

The canonical claim is the benchmark vote unit.  Upstream HF/local records are
kept separately so a cross-source duplicate remains inspectable without being
mistaken for two votes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    DIRECT_CLAIMS_PATH,
    DIRECT_REPORT_TYPES,
    DIRECT_SOURCE_ROWS_PATH,
    HF_SNAPSHOT_PATH,
    MANIFEST_PATH,
    RAW_LOCAL_SOURCE_PATH,
    classify_local_record,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_benchmark import (
    load_label_decisions,
)


AUDIT_VERSION = "bioavailability_gold_record_audit.v1"
DEFAULT_BENCHMARK_DIR = Path(
    "data/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/gold_record_audit_v1"
)
SPLITS = ("train", "valid", "test")

AUDIT_FIELDS = (
    "audit_canonical_claim_id",
    "audit_claim_decision",
    "audit_filter_reason",
    "audit_candidate_label",
    "audit_vote_direction",
    "audit_used_in_gold_vote",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = {
        "benchmark_dir": Path(args.benchmark_dir),
        "claims": Path(args.claims),
        "direct_source_rows": Path(args.direct_source_rows),
        "hf_snapshot": Path(args.hf_snapshot),
        "local_source": Path(args.local_source),
        "canonical_manifest": Path(args.canonical_manifest),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parents = load_gold_parents(paths["benchmark_dir"])
    parent_by_key = {
        str(row["molecule_identity_key"]): row for row in parents.to_dict(orient="records")
    }
    claims = pd.read_parquet(paths["claims"])
    claim_decisions = annotate_claim_decisions(
        claims,
        parent_by_key,
        manifest_path=paths["canonical_manifest"],
        source_path=paths["claims"],
    )
    source_audit = _source_audit_by_id(claim_decisions)
    direct_source = annotate_direct_source_rows(
        pd.read_parquet(paths["direct_source_rows"]), parent_by_key, source_audit
    )
    hf_records = annotate_hf_upstream_records(
        pd.read_parquet(paths["hf_snapshot"]), parent_by_key, source_audit
    )
    local_records = annotate_local_upstream_records(
        pd.read_parquet(paths["local_source"]), parent_by_key, source_audit
    )

    validation = validate_export(
        parents=parents,
        claims=claim_decisions,
        direct_source=direct_source,
        hf_records=hf_records,
        local_records=local_records,
    )
    frames = {
        "gold_parents": parents,
        "canonical_claim_decisions": claim_decisions,
        "canonical_source_records": direct_source,
        "hf_upstream_records": hf_records,
        "local_upstream_records": local_records,
    }
    output_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        output_paths[name] = path

    manifest = build_manifest(paths, output_paths, frames, validation)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


def load_gold_parents(benchmark_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in SPLITS:
        path = benchmark_dir / f"{split}_molecule_labels.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = str(row["molecule_identity_key"])
                if key in seen:
                    raise AssertionError(f"duplicate gold parent across splits: {key}")
                seen.add(key)
                rows.append(
                    {
                        "molecule_identity_key": key,
                        "parent_smiles": str(row["drug"]),
                        "gold_label": int(row["Y"]),
                        "split": split,
                        "gold_label_decision": str(row["label_decision"]),
                        "gold_source_record_count": int(row["source_record_count"]),
                        "gold_label_0_votes": int(row["label_counts"].get("0", 0)),
                        "gold_label_1_votes": int(row["label_counts"].get("1", 0)),
                        "gold_majority_record_count": int(row["majority_record_count"]),
                        "gold_minority_record_count": int(row["minority_record_count"]),
                        "gold_agreement_fraction": float(row["agreement_fraction"]),
                        "gold_bemis_murcko_scaffold": str(
                            row.get("bemis_murcko_scaffold") or ""
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["split", "molecule_identity_key"], kind="stable"
    ).reset_index(drop=True)


def annotate_claim_decisions(
    claims: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    *,
    manifest_path: Path = MANIFEST_PATH,
    source_path: Path = DIRECT_CLAIMS_PATH,
) -> pd.DataFrame:
    decisions, _ = load_label_decisions(
        source_path=source_path,
        manifest_path=manifest_path,
    )
    decision_rows = list(decisions)
    if len(decision_rows) != len(claims):
        raise AssertionError("claim/decision row count mismatch")

    selected: list[dict[str, Any]] = []
    for (_, source_row), decision in zip(claims.iterrows(), decision_rows, strict=True):
        key = str(source_row["parent_identity_key"])
        parent = parent_by_key.get(key)
        if parent is None:
            continue
        row = source_row.to_dict()
        row.update(_parent_audit_fields(parent))
        row["audit_canonical_claim_id"] = str(source_row["canonical_claim_id"])
        if decision.record is None:
            row.update(
                {
                    "audit_claim_decision": "filtered_by_gold_rule",
                    "audit_filter_reason": str(decision.reason or "unspecified_rejection"),
                    "audit_candidate_label": None,
                    "audit_vote_direction": "not_a_vote",
                    "audit_used_in_gold_vote": False,
                }
            )
        else:
            label = int(decision.record.label)
            row.update(
                {
                    "audit_claim_decision": "used_as_gold_vote",
                    "audit_filter_reason": "",
                    "audit_candidate_label": label,
                    "audit_vote_direction": (
                        "supports_final_gold_label"
                        if label == int(parent["gold_label"])
                        else "opposes_final_gold_label"
                    ),
                    "audit_used_in_gold_vote": True,
                }
            )
        selected.append(row)
    return pd.DataFrame(selected).sort_values(
        ["parent_identity_key", "canonical_claim_id"], kind="stable"
    ).reset_index(drop=True)


def _source_audit_by_id(claims: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in claims.to_dict(orient="records"):
        source_ids = _string_list(row.get("source_record_ids"))
        representative_id = str(row.get("source_record_id") or "")
        for source_id in source_ids:
            if source_id in result:
                raise AssertionError(f"source record maps to multiple claims: {source_id}")
            result[source_id] = {
                field: row.get(field) for field in AUDIT_FIELDS
            } | {
                "audit_claim_n_source_records": int(row["n_source_records"]),
                "audit_cross_source_deduplicated": bool(
                    row["cross_source_deduplicated"]
                ),
                "audit_source_record_is_claim_representative": source_id
                == representative_id,
            }
    return result


def annotate_direct_source_rows(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    selected = frame[
        frame["parent_identity_key"].astype(str).isin(parent_by_key)
    ].copy()
    missing = sorted(set(selected["source_record_id"].astype(str)) - set(source_audit))
    if missing:
        raise AssertionError(f"canonical source records missing claim mapping: {missing[:3]}")
    return _attach_source_audit(selected, parent_by_key, source_audit)


def annotate_hf_upstream_records(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    selected = _select_upstream_parents(frame, parent_by_key)
    selected["audit_source_origin"] = "hf"
    selected["audit_upstream_source_index"] = selected["source_index"].astype(int)
    selected["audit_source_record_id"] = selected["source_index"].map(
        lambda value: f"hf:{int(value)}"
    )
    selected = _attach_upstream_decisions(
        selected,
        parent_by_key,
        source_audit,
        exclusion=lambda row: (
            "non_direct_oral_bioavailability_report_type",
            str(row.get("bioavailability_report_type") or "").strip().lower(),
        ),
    )
    unexpected = selected[
        (selected["audit_pipeline_status"] == "filtered_before_canonical_claim")
        & selected["bioavailability_report_type"].astype(str).str.lower().isin(DIRECT_REPORT_TYPES)
    ]
    if len(unexpected):
        raise AssertionError("valid HF direct rows unexpectedly missing canonical mapping")
    return selected


def annotate_local_upstream_records(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    indexed = frame.copy()
    indexed.insert(0, "audit_upstream_source_index", range(len(indexed)))
    selected = _select_upstream_parents(indexed, parent_by_key)
    selected["audit_source_origin"] = "local"
    selected["audit_source_record_id"] = selected.apply(
        lambda row: (
            f"local:{int(row['audit_upstream_source_index'])}:"
            f"{_text(row.get('extraction_id')) or 'row'}"
        ),
        axis=1,
    )

    def local_exclusion(row: Mapping[str, Any]) -> tuple[str, str]:
        partition, reason = classify_local_record(row)
        return reason, partition

    selected = _attach_upstream_decisions(
        selected, parent_by_key, source_audit, exclusion=local_exclusion
    )
    unexpected = selected[
        (selected["audit_pipeline_status"] == "filtered_before_canonical_claim")
        & (selected["audit_precanonical_partition"] == "canonical_absolute_bioavailability")
    ]
    if len(unexpected):
        raise AssertionError("valid local absolute rows unexpectedly missing canonical mapping")
    return selected


def _select_upstream_parents(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    cache: dict[str, tuple[str, str] | None] = {}

    def resolve(smiles_value: Any) -> tuple[str, str] | None:
        smiles = _text(smiles_value)
        if smiles not in cache:
            identity = normalize_molecule_identity(smiles)
            cache[smiles] = (
                (identity.parent_smiles, identity.parent_inchi_key or identity.parent_smiles)
                if identity.status == "ok" and identity.parent_smiles
                else None
            )
        return cache[smiles]

    identities = [resolve(value) for value in frame["smiles"]]
    keys = [identity[1] if identity else "" for identity in identities]
    mask = pd.Series(keys, index=frame.index).isin(parent_by_key)
    selected = frame.loc[mask].copy()
    selected["audit_parent_smiles"] = [
        identities[position][0]
        for position, keep in enumerate(mask.to_numpy())
        if keep and identities[position] is not None
    ]
    selected["audit_molecule_identity_key"] = [
        keys[position] for position, keep in enumerate(mask.to_numpy()) if keep
    ]
    return selected


def _attach_upstream_decisions(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Mapping[str, Any]],
    *,
    exclusion: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row in frame.to_dict(orient="records"):
        key = str(source_row["audit_molecule_identity_key"])
        row = dict(source_row)
        row.update(_parent_audit_fields(parent_by_key[key]))
        source_id = str(row["audit_source_record_id"])
        if source_id in source_audit:
            row.update(source_audit[source_id])
            row["audit_pipeline_status"] = (
                "used_via_canonical_claim"
                if row["audit_used_in_gold_vote"]
                else "filtered_by_gold_rule_via_canonical_claim"
            )
            row["audit_precanonical_filter_reason"] = ""
            row["audit_precanonical_partition"] = "canonical_direct_claim"
        else:
            reason, partition = exclusion(row)
            row.update(
                {
                    "audit_pipeline_status": "filtered_before_canonical_claim",
                    "audit_precanonical_filter_reason": reason,
                    "audit_precanonical_partition": partition,
                    "audit_canonical_claim_id": "",
                    "audit_claim_decision": "not_a_canonical_claim",
                    "audit_filter_reason": "",
                    "audit_candidate_label": None,
                    "audit_vote_direction": "not_a_vote",
                    "audit_used_in_gold_vote": False,
                    "audit_claim_n_source_records": 0,
                    "audit_cross_source_deduplicated": False,
                    "audit_source_record_is_claim_representative": False,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["audit_molecule_identity_key", "audit_upstream_source_index"], kind="stable"
    ).reset_index(drop=True)


def _attach_source_audit(
    frame: pd.DataFrame,
    parent_by_key: Mapping[str, Mapping[str, Any]],
    source_audit: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for source_row in frame.to_dict(orient="records"):
        row = dict(source_row)
        row.update(_parent_audit_fields(parent_by_key[str(row["parent_identity_key"])]))
        row.update(source_audit[str(row["source_record_id"])])
        row["audit_pipeline_status"] = (
            "used_via_canonical_claim"
            if row["audit_used_in_gold_vote"]
            else "filtered_by_gold_rule_via_canonical_claim"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["parent_identity_key", "source_record_id"], kind="stable"
    ).reset_index(drop=True)


def _parent_audit_fields(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_gold_label": int(parent["gold_label"]),
        "audit_gold_split": str(parent["split"]),
        "audit_gold_label_decision": str(parent["gold_label_decision"]),
    }


def validate_export(
    *,
    parents: pd.DataFrame,
    claims: pd.DataFrame,
    direct_source: pd.DataFrame,
    hf_records: pd.DataFrame,
    local_records: pd.DataFrame,
) -> dict[str, Any]:
    vote_claims = claims[claims["audit_used_in_gold_vote"]]
    expected_votes = int(parents["gold_source_record_count"].sum())
    if len(vote_claims) != expected_votes:
        raise AssertionError(f"accepted vote mismatch: {len(vote_claims)} != {expected_votes}")

    vote_counts = (
        vote_claims.groupby(["parent_identity_key", "audit_candidate_label"])
        .size()
        .to_dict()
    )
    for parent in parents.to_dict(orient="records"):
        key = str(parent["molecule_identity_key"])
        for label in (0, 1):
            actual = int(vote_counts.get((key, label), 0))
            expected = int(parent[f"gold_label_{label}_votes"])
            if actual != expected:
                raise AssertionError(
                    f"parent vote mismatch for {key}, label={label}: {actual} != {expected}"
                )

    expected_source_ids = {
        source_id
        for value in claims["source_record_ids"]
        for source_id in _string_list(value)
    }
    direct_source_ids = set(direct_source["source_record_id"].astype(str))
    if direct_source_ids != expected_source_ids:
        raise AssertionError("canonical claim/source-record provenance does not reconcile")
    upstream_ids = set(hf_records["audit_source_record_id"].astype(str)) | set(
        local_records["audit_source_record_id"].astype(str)
    )
    if not expected_source_ids.issubset(upstream_ids):
        raise AssertionError("canonical source records are missing from upstream export")

    return {
        "status": "passed",
        "n_gold_parents": len(parents),
        "n_canonical_claims_for_gold_parents": len(claims),
        "n_vote_claims": len(vote_claims),
        "n_filtered_canonical_claims": len(claims) - len(vote_claims),
        "n_canonical_source_records": len(direct_source),
        "n_hf_upstream_records": len(hf_records),
        "n_local_upstream_records": len(local_records),
        "n_upstream_records": len(hf_records) + len(local_records),
        "n_upstream_records_used_via_claim": int(
            hf_records["audit_used_in_gold_vote"].sum()
            + local_records["audit_used_in_gold_vote"].sum()
        ),
        "claim_filter_reason_counts": dict(
            sorted(
                Counter(
                    str(value)
                    for value in claims.loc[
                        ~claims["audit_used_in_gold_vote"], "audit_filter_reason"
                    ]
                ).items()
            )
        ),
        "hf_pipeline_status_counts": _counts(hf_records["audit_pipeline_status"]),
        "local_pipeline_status_counts": _counts(local_records["audit_pipeline_status"]),
        "vote_count_matches_frozen_parent_artifacts": True,
        "claim_source_ids_match_canonical_source_rows": True,
        "canonical_source_ids_present_in_upstream_exports": True,
    }


def build_manifest(
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_lineage": "record_supported_v2.bioavailability_canonical_direct.v2",
        "scope": (
            "all HF Oral_Bioavailability and local Oral_AUC-Cmax_Exposure upstream "
            "records whose normalized parent has a frozen gold label"
        ),
        "vote_contract": {
            "vote_unit": "canonical_claim_id",
            "used_marker": "audit_claim_decision=used_as_gold_vote",
            "filtered_marker": "audit_claim_decision=filtered_by_gold_rule",
            "cross_source_note": (
                "multiple upstream rows may share one canonical_claim_id and contribute "
                "to one vote; never count upstream rows as independent votes"
            ),
        },
        "validation": dict(validation),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in inputs.items()
            if path.is_file()
        }
        | {"benchmark_dir": str(inputs["benchmark_dir"])},
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": len(frames[name]),
            }
            for name, path in outputs.items()
        },
    }


def _render_readme(manifest: Mapping[str, Any]) -> str:
    validation = manifest["validation"]
    return f"""# Bioavailability gold-record audit v1

This export is tied to `{manifest['dataset_lineage']}` and does not modify the
frozen benchmark.

## Which table answers which question

- `gold_parents.parquet`: one row per final labeled parent.
- `canonical_claim_decisions.parquet`: **authoritative vote table**. One row is
  one `canonical_claim_id`; `audit_claim_decision` says whether the claim voted
  or was filtered, and `audit_filter_reason` gives the exact rule reason.
- `canonical_source_records.parquet`: normalized HF/local source rows underlying
  the canonical claims. Cross-source duplicates share a claim ID.
- `hf_upstream_records.parquet`: every original HF row for the labeled parents,
  including rows excluded before canonicalization.
- `local_upstream_records.parquet`: every original local extraction row for the
  labeled parents, including AUC/Cmax, relative, and ambiguous records excluded
  before canonicalization.

Do not count `hf_upstream_records` or `local_upstream_records` rows as votes.
Count unique rows in `canonical_claim_decisions` where
`audit_claim_decision == "used_as_gold_vote"`.

## Frozen reconciliation

- Gold parents: {validation['n_gold_parents']:,}
- Canonical claims for those parents: {validation['n_canonical_claims_for_gold_parents']:,}
- Claims used as votes: {validation['n_vote_claims']:,}
- Canonical claims filtered by gold rules: {validation['n_filtered_canonical_claims']:,}
- Original HF/local rows exported: {validation['n_upstream_records']:,}
- Validation: `{validation['status']}`
"""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--claims", default=str(DIRECT_CLAIMS_PATH))
    parser.add_argument("--direct-source-rows", default=str(DIRECT_SOURCE_ROWS_PATH))
    parser.add_argument("--hf-snapshot", default=str(HF_SNAPSHOT_PATH))
    parser.add_argument("--local-source", default=str(RAW_LOCAL_SOURCE_PATH))
    parser.add_argument("--canonical-manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
