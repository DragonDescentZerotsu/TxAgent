"""Build the deterministic 140-row send_v2 versus superseded-source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization.cleaning import standardize_smiles
from tools.chembl_tool.tasks.clintox.starling_source import (
    DEFAULT_DATA_ROOT,
    SOURCE_SPECS,
)


AUDIT_VERSION = "clintox_send_v2_source_delta_review.v1"
DEFAULT_OLD_ROOT = Path("data/starling_data/clintox")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data_processing/source_delta_v2"
OLD_SOURCE_PATHS = {
    "human_clinical_toxicity": "clintox_base_v1/extractions.parquet",
    "nonclinical_in_vivo_toxicity": "raw_v1/nonclinical_in_vivo_toxicity/extractions.parquet",
    "organ_specific_toxicity": "raw_v1/organ_specific_toxicity/extractions.parquet",
    "genotoxicity_carcinogenicity": "raw_v1/genotoxicity_carcinogenicity/extractions.parquet",
    "cellular_stress": "raw_v1/cellular_stress/extractions.parquet",
    "general_cytotoxicity": "raw_v1/general_cytotoxicity/extractions.parquet",
    "off_target_ddi_exposure": "raw_v1/off_target_ddi_exposure/extractions.parquet",
}
SEMANTIC_FIELDS = {
    "human_clinical_toxicity": ("toxicity_category", "outcome_measure"),
    "nonclinical_in_vivo_toxicity": ("evidence_type", "endpoint_value", "endpoint_unit"),
    "organ_specific_toxicity": ("toxicity_endpoint", "effect_status", "quantitative_result"),
    "genotoxicity_carcinogenicity": ("endpoint", "result_direction"),
    "cellular_stress": ("stress_endpoint", "effect_direction"),
    "general_cytotoxicity": ("endpoint_type", "result_value", "result_unit"),
    "off_target_ddi_exposure": ("target_or_endpoint", "result_metric", "result_value", "result_unit"),
}


def _key_frame(path: Path) -> pd.DataFrame:
    frame = pq.read_table(
        path, columns=["pmid", "extraction_id", "SMILES"]
    ).to_pandas()
    frame["record_key"] = list(
        zip(frame["pmid"].fillna("").astype(str), frame["extraction_id"].fillna("").astype(str))
    )
    if frame["record_key"].duplicated().any():
        raise ValueError(f"duplicate PMID/extraction_id keys in {path}")
    return frame


def _delta_types(new: pd.DataFrame, old: pd.DataFrame) -> pd.Series:
    old_smiles = dict(zip(old["record_key"], old["SMILES"].fillna("").astype(str)))
    result: list[str] = []
    validity: dict[str, bool] = {}
    for key, value in zip(new["record_key"], new["SMILES"].fillna("").astype(str)):
        if value not in validity:
            validity[value] = bool(standardize_smiles(value)[0])
        prior = old_smiles.get(key)
        if not validity[value]:
            result.append("invalid_source_smiles")
        elif prior is None:
            result.append("new_record")
        elif not prior.strip():
            result.append("direct_smiles_backfill")
        elif prior != value:
            result.append("changed_smiles")
        else:
            result.append("retained_record")
    return pd.Series(result, index=new.index)


def _stable_rank(source_id: str, key: tuple[str, str]) -> str:
    return hashlib.sha256(f"{source_id}|{key[0]}|{key[1]}".encode()).hexdigest()


def _select_indices(source_id: str, frame: pd.DataFrame, n: int = 20) -> list[int]:
    working = frame.copy()
    working["rank"] = [
        _stable_rank(source_id, key) for key in working["record_key"]
    ]
    strata = (
        "new_record", "direct_smiles_backfill", "changed_smiles",
        "invalid_source_smiles", "retained_record",
    )
    groups = {
        name: working[working["delta_type"] == name].sort_values("rank").index.tolist()
        for name in strata
    }
    selected: list[int] = []
    while len(selected) < n and any(groups.values()):
        for name in strata:
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop(0))
    if len(selected) != n:
        raise ValueError(f"could not select {n} delta rows for {source_id}")
    return selected


def _source_sample(source_id: str, old_root: Path, new_root: Path) -> pd.DataFrame:
    old_path = old_root / OLD_SOURCE_PATHS[source_id]
    new_path = new_root / source_id / "extractions.parquet"
    old = _key_frame(old_path)
    new = _key_frame(new_path)
    new["delta_type"] = _delta_types(new, old)
    indices = _select_indices(source_id, new)
    columns = ["support_text", "pmid", "extraction_id", "SMILES", *SEMANTIC_FIELDS[source_id]]
    selected = pq.read_table(new_path, columns=columns).take(indices).to_pandas()
    old_smiles = dict(zip(old["record_key"], old["SMILES"].fillna("").astype(str)))
    selected.insert(0, "source_id", source_id)
    selected.insert(1, "delta_type", new.loc[indices, "delta_type"].tolist())
    selected.insert(2, "old_SMILES", [old_smiles.get(new.loc[index, "record_key"]) for index in indices])
    selected.insert(0, "manual_review_version", AUDIT_VERSION)
    selected.insert(0, "manual_review_notes", "")
    selected.insert(0, "manual_review_status", "pending")
    return selected


def build_audit(
    *, old_root: str | Path = DEFAULT_OLD_ROOT, new_root: str | Path = DEFAULT_DATA_ROOT
) -> tuple[dict[str, Any], pd.DataFrame]:
    sample = pd.concat(
        [_source_sample(spec.source_id, Path(old_root), Path(new_root)) for spec in SOURCE_SPECS],
        ignore_index=True,
    )
    if len(sample) != 140:
        raise ValueError("source delta audit must contain exactly 140 rows")
    summary = {
        "audit_version": AUDIT_VERSION,
        "status": "pending_manual_review",
        "sample_records": len(sample),
        "sampling_policy": "20 deterministic rows per source, round-robin across release delta types",
        "source_counts": sample["source_id"].value_counts().sort_index().to_dict(),
        "delta_type_counts": sample["delta_type"].value_counts().sort_index().to_dict(),
    }
    return summary, sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", default=str(DEFAULT_OLD_ROOT))
    parser.add_argument("--new-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    summary, sample = build_audit(old_root=args.old_root, new_root=args.new_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / "manual_audit.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
