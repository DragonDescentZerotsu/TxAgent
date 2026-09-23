"""Build an unselected AMES endpoint-family successor from the frozen map.

The policy merges only named, closely related endpoint concepts within an
existing source/level semantic parent. It copies every atom and record exactly
once, records parent-score lineage, and deliberately assigns no new weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd

from predict.utils.json import sha256_file, write_json_atomic
from semantic_buckets.artifacts import REPOSITORY_ROOT


SOURCE = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/source_local_semantic_v5/"
    "ames_dili_carcinogens_main_universe_v3_frozen_queued_boundary_20260922/"
    "ames/semantic_run"
)
WEIGHTS = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/semantic_weight_two_pass_v4/"
    "ames_morgan100_official_two_pass_v6_cards_task_level_wavefront_"
    "luna_three_of_three_resume_v3_20260922/weights.parquet"
)
POLICY = REPOSITORY_ROOT / "semantic_buckets/policies/ames_endpoint_family_merge_candidate_v1.json"
OUTPUT = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/ames_endpoint_family_split_candidate_v1_20260923"
)


def _family_lookup() -> dict[str, str]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("status") != "candidate_unreviewed":
        raise ValueError("Endpoint-family policy is not a candidate")
    lookup = {}
    for group in policy["groups"]:
        for endpoint in group["endpoint_concepts"]:
            if endpoint in lookup:
                raise ValueError(f"Endpoint appears in two families: {endpoint}")
            lookup[endpoint] = group["family"]
    return lookup


def _child_id(parent: str, family: str) -> str:
    value = json.dumps([parent, family], ensure_ascii=False, separators=(",", ":"))
    return "sbaf_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _candidate_atoms() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atoms = pd.read_parquet(SOURCE / "input_atoms.parquet")
    original = pd.read_parquet(SOURCE / "semantic_bucket_map.parquet")
    if len(atoms) != len(original) or atoms.atom_id.duplicated().any():
        raise ValueError("Frozen AMES atom map is incomplete")
    atom_values = atoms[["atom_id", "record_count", "values_json"]].copy()
    atom_values["endpoint"] = atom_values.values_json.map(
        lambda value: json.loads(value)["canonical_endpoint_concept"])
    joined = original.merge(atom_values, on="atom_id", validate="one_to_one")
    lookup = _family_lookup()
    joined["endpoint_family"] = joined.endpoint.map(lambda value: lookup.get(value, value))
    family_count = joined.groupby("semantic_bucket_id").endpoint_family.nunique()
    joined["parent_semantic_bucket_id"] = joined.semantic_bucket_id
    joined["candidate_semantic_bucket_id"] = [
        parent if family_count[parent] == 1 else _child_id(parent, family)
        for parent, family in zip(joined.semantic_bucket_id, joined.endpoint_family)
    ]
    if (joined.groupby("candidate_semantic_bucket_id")[["level", "source_id", "endpoint_family"]]
            .nunique().gt(1).any().any()):
        raise ValueError("Candidate crosses a source, level, or endpoint family")
    candidate = original.copy()
    candidate["source_semantic_bucket_id"] = joined.candidate_semantic_bucket_id.to_numpy()
    candidate["semantic_bucket_id"] = joined.candidate_semantic_bucket_id.to_numpy()
    return atoms, candidate, joined


def _candidate_records(candidate: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_parquet(SOURCE / "record_semantic_bucket_map.parquet")
    mapped = records.drop(columns=["semantic_bucket_id"]).merge(
        candidate[["atom_id", "semantic_bucket_id"]], on="atom_id",
        how="left", validate="many_to_one")
    if (len(mapped) != len(records) or mapped.semantic_bucket_id.isna().any()
            or mapped.duplicated(["canonical_record_id", "source_row_uid", "level"]).any()
            or set(mapped.source_row_uid) != set(records.source_row_uid)):
        raise ValueError("Candidate does not preserve exact physical-record coverage")
    return mapped.sort_values(["level", "source_row_uid"])


def _lineage(joined: pd.DataFrame) -> pd.DataFrame:
    weights = pd.read_parquet(WEIGHTS, columns=["level", "semantic_bucket_id", "weight"])
    summary = joined.groupby([
        "parent_semantic_bucket_id", "candidate_semantic_bucket_id", "level", "source_id",
        "endpoint_family"], as_index=False).agg(
        atom_count=("atom_id", "size"), record_count=("record_count", "sum"),
        endpoint_concepts=("endpoint", lambda values: json.dumps(sorted(set(values)))),
    )
    summary = summary.merge(weights.rename(columns={
        "semantic_bucket_id": "parent_semantic_bucket_id", "weight": "parent_weight",
    }), on=["level", "parent_semantic_bucket_id"], validate="many_to_one")
    if summary.parent_weight.isna().any():
        raise ValueError("A successor child has no auditable parent weight")
    return summary.sort_values(["level", "parent_semantic_bucket_id", "endpoint_family"])


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    atoms, candidate, joined = _candidate_atoms()
    records = _candidate_records(candidate)
    lineage = _lineage(joined)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ames_endpoint_family.", dir=OUTPUT.parent) as temporary:
        stage = Path(temporary)
        candidate.to_parquet(stage / "semantic_bucket_map.parquet", index=False)
        records.to_parquet(stage / "record_semantic_bucket_map.parquet", index=False)
        lineage.to_parquet(stage / "bucket_lineage.parquet", index=False)
        lineage.to_csv(stage / "bucket_lineage.tsv", sep="\t", index=False)
        mixed = joined.groupby("parent_semantic_bucket_id").endpoint.nunique().gt(1)
        document = {
            "schema_version": "ames_endpoint_family_candidate.v1",
            "status": "candidate_unreviewed_unweighted", "task": "ames",
            "evidence_library_version": "v10_main_universe_v3",
            "counts": {
                "atoms": len(atoms), "records": len(records),
                "original_buckets": joined.parent_semantic_bucket_id.nunique(),
                "mixed_endpoint_parents": int(mixed.sum()),
                "candidate_buckets": candidate.semantic_bucket_id.nunique(),
                "replacement_groups": len(lineage[lineage.parent_semantic_bucket_id.isin(mixed[mixed].index)]),
            },
            "inputs": {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in (
                SOURCE / "input_atoms.parquet", SOURCE / "semantic_bucket_map.parquet",
                SOURCE / "record_semantic_bucket_map.parquet", WEIGHTS, POLICY)},
            "files": {name: sha256_file(stage / name) for name in (
                "semantic_bucket_map.parquet", "record_semantic_bucket_map.parquet",
                "bucket_lineage.parquet", "bucket_lineage.tsv")},
            "weight_policy": "No child scores assigned; parent scores are audit anchors only",
        }
        write_json_atomic(stage / "manifest.json", document)
        stage.rename(OUTPUT)
    return document


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
