"""Build the first source-local Skin L2-L3 semantic and readout release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "skin_source_local_semantic_readout"
LEVELS = {2, 3}
CATEGORICAL_INDEX = {
    "direct_skin_reaction": 5,
    "phototoxicity_irritation_local_damage": 5,
}


def _stable_id(prefix: str, *values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _identities(level: int, pair_key: str) -> tuple[str, str, str, str]:
    values = json.loads(pair_key)
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"invalid Skin pair bucket key: {pair_key}")
    source, endpoint, unit, *context = map(str, values)
    categorical = CATEGORICAL_INDEX.get(source)
    semantic_context = list(context)
    scale = "__none__"
    if categorical is not None:
        position = categorical - 3
        scale = semantic_context.pop(position)
    semantic = _stable_id("sb", level, source, endpoint, *semantic_context)
    readout = _stable_id("rb", semantic, unit, scale)
    atom = _stable_id("atom", level, source, pair_key)
    return source, semantic, readout, atom


def _record_map(records_path: Path, levels_path: Path) -> pd.DataFrame:
    records = pd.read_parquet(records_path)[
        ["canonical_record_id", "source_row_uid", "source_id", "pair_bucket_key"]
    ]
    levels = pd.read_parquet(levels_path)[["source_row_uid", "level"]]
    merged = records.merge(levels, on="source_row_uid", validate="one_to_one")
    merged = merged[merged.level.isin(LEVELS)].copy()
    identities = [
        _identities(int(row.level), str(row.pair_bucket_key))
        for row in merged.itertuples(index=False)
    ]
    columns = ["parsed_source_id", "semantic_bucket_id", "readout_bucket_id", "atom_id"]
    merged[columns] = pd.DataFrame(identities, index=merged.index)
    if not (merged.source_id.astype(str) == merged.parsed_source_id).all():
        raise ValueError("Skin pair bucket source disagrees with its record")
    return merged.drop(columns="parsed_source_id")


def _write_maps(records: pd.DataFrame, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    semantic = records[
        ["level", "source_id", "semantic_bucket_id", "atom_id"]
    ].drop_duplicates("atom_id")
    semantic.insert(2, "source_semantic_bucket_id", semantic.semantic_bucket_id)
    readout = records[
        ["level", "source_id", "semantic_bucket_id", "atom_id", "readout_bucket_id"]
    ].drop_duplicates("atom_id")
    readout["readout_depth"] = 1
    readout["termination_reason"] = "source_local_canonical_identity"
    paths = {
        "semantic_bucket_map": output / "semantic_bucket_map.parquet",
        "readout_bucket_map": output / "readout_bucket_map.parquet",
        "record_readout_bucket_map": output / "record_readout_bucket_map.parquet",
    }
    semantic.sort_values(["level", "source_id", "semantic_bucket_id", "atom_id"]).to_parquet(paths["semantic_bucket_map"], index=False)
    readout.sort_values(["level", "semantic_bucket_id", "readout_bucket_id"]).to_parquet(paths["readout_bucket_map"], index=False)
    records.sort_values(["level", "source_row_uid"]).to_parquet(paths["record_readout_bucket_map"], index=False)
    return paths


def build(
    records_path: Path,
    levels_path: Path,
    output: Path,
    published_records_path: Path | None = None,
    published_levels_path: Path | None = None,
    evidence_library_version: str = "v10_main_universe_v1",
) -> dict:
    records = _record_map(records_path, levels_path)
    paths = _write_maps(records, output)
    readouts = records.groupby("readout_bucket_id").semantic_bucket_id.nunique()
    if records.source_row_uid.duplicated().any() or (readouts != 1).any():
        raise ValueError("Skin semantic/readout identity contract failed")
    manifest = {
        "version": f"{VERSION}.{evidence_library_version}",
        "task": "skin_reaction",
        "evidence_library_version": evidence_library_version,
        "levels": ["L2", "L3"],
        "policy": "source-local endpoint and assay context; readout adds atomic unit and scale",
        "inputs": {
            "stage3_records": {
                "path": str(published_records_path or records_path),
                "sha256": file_sha256(records_path),
            },
            "level_mapping": {
                "path": str(published_levels_path or levels_path),
                "sha256": file_sha256(levels_path),
            },
        },
        "counts": {
            "records": len(records),
            "atoms": int(records.atom_id.nunique()),
            "semantic_buckets": int(records.semantic_bucket_id.nunique()),
            "readout_buckets": int(records.readout_bucket_id.nunique()),
        },
        "outputs": {
            name: {"path": path.name, "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "validations": {"complete_mapped_l2_l3_coverage": True, "one_semantic_parent_per_readout": True},
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-records", type=Path)
    parser.add_argument("--published-levels", type=Path)
    parser.add_argument(
        "--evidence-library-version", default="v10_main_universe_v1"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.records,
                args.levels,
                args.output,
                args.published_records,
                args.published_levels,
                args.evidence_library_version,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
