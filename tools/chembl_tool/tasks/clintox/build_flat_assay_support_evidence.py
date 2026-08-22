"""Materialize representative support records for ClinTox physical assays.

This is the source-native bridge needed before the shared compact-summary and
assay-family retrieval stages.  It preserves at most three deterministic
representative records per physical assay and molecule; it does not create or
modify ClinTox gold labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.tasks.clintox.build_flat_assay_catalog import SOURCE_SPECS


VERSION = "clintox_source_native_assay_support_evidence.v1"
VALUE_UNIT_FIELDS = {
    "cellular_stress": ("", ""),
    "general_cytotoxicity": ("result_value", "result_unit"),
    "genotoxicity_carcinogenicity": ("result_direction", ""),
    "nonclinical_in_vivo_toxicity": ("endpoint_value", "endpoint_unit"),
    "off_target_ddi_exposure": ("result_value", "result_unit"),
    "organ_specific_toxicity": ("quantitative_result", ""),
}
SPECIES_FIELDS = {
    "cellular_stress": "biological_model",
    "general_cytotoxicity": "cell_model",
    "genotoxicity_carcinogenicity": "biological_system",
    "nonclinical_in_vivo_toxicity": "animal_context",
    "off_target_ddi_exposure": "assay_context",
    "organ_specific_toxicity": "biological_system",
}


def _parent_key(smiles: Any) -> str:
    identity = normalize_molecule_identity(_clean(smiles))
    return identity.parent_connectivity_key or identity.parent_inchi_key


def _evidence_row(
    *,
    assay_id: str,
    molecule_id: str,
    canonical_smiles: str,
    assay_context: str,
    source_id: str,
    examples: list[dict[str, str]],
    source_record_count: int,
) -> dict[str, Any]:
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": canonical_smiles,
        "assay_chembl_id": assay_id,
        "assay_tier": "Assay",
        "endpoint_group": "assay_ranked_evidence",
        "group_id": f"Assay.{assay_id}",
        "standard_type": "; ".join(
            dict.fromkeys(item["endpoint_type"] for item in examples if item["endpoint_type"])
        ),
        "standard_value": "; ".join(
            dict.fromkeys(item["reported_value"] for item in examples if item["reported_value"])
        ),
        "standard_units": "; ".join(
            dict.fromkeys(item["reported_units"] for item in examples if item["reported_units"])
        ),
        "target_pref_name": assay_context,
        "evidence_source": f"Starling ClinTox {source_id}",
        "source_record_count": source_record_count,
        "source_record_examples": examples,
        "evidence_scope": {"assay_context": [assay_context]},
        "uncertainty": [],
    }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _representatives(group: pd.DataFrame, *, limit: int = 3) -> list[dict[str, str]]:
    examples = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in group.to_dict(orient="records"):
        example = {
            "endpoint_type": _clean(row.get("endpoint_type")),
            "reported_value": _clean(row.get("reported_value")),
            "reported_units": _clean(row.get("reported_units")),
            "assay_context": _clean(row.get("assay_context")),
            "species_context": _clean(row.get("species_context")),
            "qualifying_conditions": _clean(row.get("qualifying_conditions")),
            "support_text": _clean(row.get("support_text")),
        }
        key = (
            example["endpoint_type"],
            example["reported_value"],
            example["reported_units"],
            example["support_text"],
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        if len(examples) >= limit:
            break
    return examples


def _raw_source_rows(
    *,
    source_root: Path,
    source_id: str,
    membership: pd.DataFrame,
    assay_context_by_id: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    spec = SOURCE_SPECS[source_id]
    value_field, unit_field = VALUE_UNIT_FIELDS[source_id]
    species_field = SPECIES_FIELDS[source_id]
    columns = list(
        dict.fromkeys(
            [
                "support_text",
                "SMILES",
                "confidence",
                "qualifying_conditions",
                spec["endpoint_field"],
                species_field,
                *([value_field] if value_field else []),
                *([unit_field] if unit_field else []),
            ]
        )
    )
    raw = pd.read_parquet(source_root / source_id / "extractions.parquet", columns=columns)
    raw = raw.reset_index(names="source_row_number")
    selected = membership.loc[membership["source_id"].eq(source_id)].copy()
    joined = selected.merge(raw, on="source_row_number", how="inner", validate="many_to_one")
    joined["endpoint_type"] = joined[spec["endpoint_field"]]
    joined["reported_value"] = joined[value_field] if value_field else ""
    joined["reported_units"] = joined[unit_field] if unit_field else ""
    joined["species_context"] = joined[species_field]
    joined["assay_context"] = joined["assay_id"].map(assay_context_by_id)
    joined["confidence"] = pd.to_numeric(joined["confidence"], errors="coerce")
    joined = joined.sort_values(
        ["assay_id", "molecule_id", "confidence", "source_row_number"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="stable",
    )
    for (stable_assay_id, molecule_id), group in joined.groupby(
        ["assay_id", "molecule_id"], sort=False
    ):
        assay_context = _clean(group["assay_context"].iloc[0])
        yield _evidence_row(
            assay_id=str(stable_assay_id),
            molecule_id=str(molecule_id),
            canonical_smiles=_clean(group["SMILES"].iloc[0]),
            assay_context=assay_context,
            source_id=source_id,
            examples=_representatives(group),
            source_record_count=len(group),
        )


def _direct_rows(
    *,
    direct_path: Path,
    membership: pd.DataFrame,
    assay_context_by_id: Mapping[str, str],
    heldout_parent_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    direct = pd.read_parquet(direct_path).copy()
    selected = membership.loc[membership["source_id"].eq("clinical_trial_failure")].copy()
    joined = selected.merge(
        direct,
        on="source_row_number",
        how="inner",
        validate="many_to_one",
    )
    joined["endpoint_type"] = joined["toxicity_outcome"]
    joined["reported_value"] = ""
    joined["reported_units"] = ""
    joined["species_context"] = "human clinical"
    joined["qualifying_conditions"] = ""
    joined["assay_context"] = joined["assay_id"].map(assay_context_by_id)
    joined["confidence"] = pd.to_numeric(joined["confidence"], errors="coerce")
    joined["parent_key"] = joined["SMILES"].map(_parent_key)
    heldout_mask = joined["parent_key"].isin(heldout_parent_keys)
    n_before = len(joined)
    joined = joined.loc[~heldout_mask].copy()
    joined = joined.sort_values(
        ["assay_id", "molecule_id", "confidence", "source_row_number"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="stable",
    )
    rows = []
    for (stable_assay_id, molecule_id), group in joined.groupby(
        ["assay_id", "molecule_id"], sort=False
    ):
        assay_context = _clean(group["assay_context"].iloc[0])
        rows.append(
            _evidence_row(
                assay_id=str(stable_assay_id),
                molecule_id=str(molecule_id),
                canonical_smiles=_clean(group["SMILES"].iloc[0]),
                assay_context=assay_context,
                source_id="clinical_trial_failure",
                examples=_representatives(group),
                source_record_count=len(group),
            )
        )
    return rows, {
        "n_direct_records_before_heldout_filter": n_before,
        "n_direct_heldout_records_excluded": int(heldout_mask.sum()),
        "n_direct_records_after_heldout_filter": len(joined),
        "n_direct_heldout_records_after_filter": int(
            joined["parent_key"].isin(heldout_parent_keys).sum()
        ),
    }


def _clinical_context_rows(evidence_path: Path) -> Iterator[dict[str, Any]]:
    with evidence_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("group_id") != "Mechanism.clinical_human_safety":
                continue
            examples = []
            for source in (row.get("source_record_examples") or [])[:3]:
                context = source.get("context") if isinstance(source.get("context"), Mapping) else {}
                examples.append(
                    {
                        "endpoint_type": _clean(source.get("endpoint_type")),
                        "reported_value": _clean(source.get("reported_value")),
                        "reported_units": _clean(source.get("reported_units")),
                        "assay_context": "clinical human safety context",
                        "species_context": "human clinical",
                        "qualifying_conditions": _clean(context.get("clinical_context")),
                        "support_text": _clean(source.get("support_text")),
                    }
                )
            yield _evidence_row(
                assay_id="STARLING_CLINTOX_CLINICAL_CONTEXT",
                molecule_id=str(row.get("molecule_chembl_id") or ""),
                canonical_smiles=_clean(row.get("canonical_smiles")),
                assay_context="clinical human safety context",
                source_id="clinical_human_safety",
                examples=examples,
                source_record_count=int(row.get("source_record_count") or len(examples)),
            )


def build(
    *,
    source_root: Path,
    membership_path: Path,
    catalog_path: Path,
    direct_path: Path,
    clinical_evidence_path: Path,
    heldout_molecules_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    membership = pd.read_parquet(membership_path)
    catalog = [json.loads(line) for line in catalog_path.open() if line.strip()]
    assay_context_by_id = {str(row["assay_id"]): str(row["assay_context"]) for row in catalog}
    heldout_parent_keys = {
        key
        for row in read_jsonl(heldout_molecules_path)
        if (key := _parent_key(row.get("drug")))
    }
    counts: dict[str, int] = {}
    with atomic_output_path(output_path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for source_id in SOURCE_SPECS:
                count = 0
                for row in _raw_source_rows(
                    source_root=source_root,
                    source_id=source_id,
                    membership=membership,
                    assay_context_by_id=assay_context_by_id,
                ):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                counts[source_id] = count
            direct_rows, direct_filter_stats = _direct_rows(
                direct_path=direct_path,
                membership=membership,
                assay_context_by_id=assay_context_by_id,
                heldout_parent_keys=heldout_parent_keys,
            )
            for row in direct_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["clinical_trial_failure"] = len(direct_rows)
            clinical_count = 0
            for row in _clinical_context_rows(clinical_evidence_path):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                clinical_count += 1
            counts["clinical_human_safety"] = clinical_count
    return {
        "artifact_type": VERSION,
        "source_root": str(source_root.resolve()),
        "membership": str(membership_path.resolve()),
        "membership_sha256": sha256_file(membership_path),
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "direct": str(direct_path.resolve()),
        "direct_sha256": sha256_file(direct_path),
        "clinical_evidence": str(clinical_evidence_path.resolve()),
        "clinical_evidence_sha256": sha256_file(clinical_evidence_path),
        "heldout_molecules": str(heldout_molecules_path.resolve()),
        "heldout_molecules_sha256": sha256_file(heldout_molecules_path),
        "n_heldout_parent_identities": len(heldout_parent_keys),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "max_representative_records_per_assay_molecule": 3,
        "rows_by_source": counts,
        "n_rows": sum(counts.values()),
        "reference_pool": "direct_only_heldout_filtered",
        "neighbor_identity_policy_default": "scaffold_disjoint",
        **direct_filter_stats,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/starling_data/clintox/raw_v1")
    parser.add_argument(
        "--membership",
        default="outputs/paper/starling_assay_relevance_all_v1/clintox/record_assay_membership.parquet",
    )
    parser.add_argument(
        "--catalog",
        default="outputs/paper/starling_assay_relevance_all_v1/clintox/assay_catalog.jsonl",
    )
    parser.add_argument(
        "--direct",
        default="data/starling_data/clintox/canonical_retrieval_v1/strict_direct_claims.parquet",
    )
    parser.add_argument(
        "--clinical-evidence",
        default=(
            "outputs/chembl_tool/tasks/clintox/evidence_library/"
            "starling_clinical_trial_failure_v1/starling_clintox_evidence.jsonl"
        ),
    )
    parser.add_argument(
        "--heldout-molecules",
        default=(
            "data/processed_clintox_clinical_trial_failure_v1/ClinTox/"
            "scaffold/heldout_molecule_labels.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper/starling_conditioned_assay_family_curve_v1/clintox",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "assay_support_evidence.jsonl"
    manifest = build(
        source_root=Path(args.source_root),
        membership_path=Path(args.membership),
        catalog_path=Path(args.catalog),
        direct_path=Path(args.direct),
        clinical_evidence_path=Path(args.clinical_evidence),
        heldout_molecules_path=Path(args.heldout_molecules),
        output_path=output_path,
    )
    write_json_atomic(output_dir / "assay_support_evidence_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
