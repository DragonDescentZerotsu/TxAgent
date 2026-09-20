"""Run the reviewed source-local semantic-parent loop for a registered library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_semantic_v4.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    ROOT
    / "semantic_buckets/provenance/source_local_semantic_v4/"
    "skin_main_universe_v5_semantic_parent_v1_20260919"
)
ENDPOINTS = tuple(
    {
        "name": host,
        "base_url": f"http://{host}:{port}/v1",
        "provider": "local",
        "credential_env": "",
        "max_inflight": 128,
    }
    for host, port in (
        ("dgx005", 50001),
        ("dgx011", 50001),
        ("dgx020", 50002),
        ("dgx027", 50001),
    )
)
PAIR_COLUMNS = {
    "direct_skin_reaction": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_assay_or_test", "canonical_species_or_population",
        "canonical_measurement_scale_id",
    ),
    "sensitization_aop": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_aop_event", "canonical_assay_type", "canonical_species_context",
    ),
    "skin_exposure": (
        "canonical_endpoint_concept", "canonical_unit_text",
        "canonical_study_design", "canonical_species_context",
    ),
}
REFINEMENT_COLUMNS = {
    "direct_skin_reaction": (
        "canonical_endpoint_concept", "canonical_assay_or_test",
        "canonical_measurement_scale_id",
    ),
    "sensitization_aop": (
        "canonical_endpoint_concept", "canonical_aop_event", "canonical_assay_type",
    ),
    "skin_exposure": ("canonical_endpoint_concept", "canonical_study_design"),
}
COLUMN_DESCRIPTIONS = {
    "canonical_endpoint_concept": "normalized biological endpoint",
    "canonical_assay_or_test": "normalized assay or test identity",
    "canonical_measurement_scale_id": "measurement or response representation",
    "canonical_aop_event": "adverse-outcome-pathway event",
    "canonical_assay_type": "normalized sensitization assay type",
    "canonical_study_design": "normalized exposure study design",
}


def _paths(run_root: Path) -> dict[str, Path]:
    release = ROOT / "data/evidence_libraries/skin_reaction/v10_main_universe_v5"
    return {
        "release": release,
        "records": release / "03_pair_buckets/records.parquet",
        "levels": release / "level_mapping/records.parquet",
        "input": run_root / "input/record_relevance_map.parquet",
        "input_manifest": run_root / "input/manifest.json",
        "review": run_root / "prompt_review",
        "run": run_root / "semantic_run",
    }


def build_input(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    current = (paths["release"].parent / "CURRENT").read_text().strip()
    if current != paths["release"].name:
        raise ValueError(f"Skin CURRENT is {current}, expected {paths['release'].name}")
    records = pd.read_parquet(paths["records"])[
        ["canonical_record_id", "source_row_uid", "source_id", "pair_bucket_key"]
    ]
    levels = pd.read_parquet(paths["levels"])[
        ["canonical_record_id", "source_row_uid", "level"]
    ]
    joined = records.merge(
        levels, on=["canonical_record_id", "source_row_uid"], validate="one_to_one"
    )
    joined = joined[joined.level.isin([2, 3])].copy()
    endpoints = joined.pair_bucket_key.map(lambda value: str(json.loads(value)[1]))
    joined["level"] = joined.level.map(lambda value: f"L{int(value)}")
    joined["relevance_bucket"] = [
        core._canonical_json({"canonical_endpoint_concept": value}) for value in endpoints
    ]
    joined["node_key"] = joined.relevance_bucket
    joined = joined.sort_values(["level", "source_id", "source_row_uid"])
    if len(joined) != 24_494 or joined.source_row_uid.duplicated().any():
        raise ValueError("Skin L2/L3 physical-record contract changed")
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(paths["input"], index=False)
    manifest = {
        "version": f"{VERSION}.input.v1", "status": "complete",
        "task": "skin_reaction", "evidence_release": paths["release"].name,
        "initial_parent_policy": "level + source_id + canonical_endpoint_concept",
        "inputs": {
            "stage3_records": {"path": str(paths["records"]), "sha256": file_sha256(paths["records"])},
            "level_mapping": {"path": str(paths["levels"]), "sha256": file_sha256(paths["levels"])},
        },
        "output": {"path": paths["input"].name, "sha256": file_sha256(paths["input"])},
        "counts": {"records": len(joined), "initial_parents": joined.groupby(
            ["level", "source_id", "node_key"], sort=False
        ).ngroups},
    }
    write_json_atomic(paths["input_manifest"], manifest)
    return manifest


def configure(run_root: Path) -> dict[str, Path]:
    paths = _paths(run_root)
    observed = set(pd.read_parquet(paths["input"], columns=["source_id"]).source_id)
    unknown = observed - set(PAIR_COLUMNS)
    if unknown:
        raise ValueError(f"unregistered Skin semantic sources: {sorted(unknown)}")
    core.VERSION = VERSION
    core.BATCH_SEED_VERSION = VERSION
    core.TASK_NAME = "Skin_Reaction"
    core.BASE_URL = ENDPOINTS[0]["base_url"]
    core.ENDPOINTS = ENDPOINTS
    core.RECORD_MAP = paths["input"]
    core.RECORDS = paths["records"]
    core.PROMPT_ROOT = Path(__file__).with_name("prompts") / "source_local_semantic_v4"
    core.DEFAULT_REVIEW = paths["review"]
    core.DEFAULT_OUTPUT = paths["run"]
    core.LEVELS = ("L2", "L3")
    core.EXPECTED_RECORD_COUNT = 24_494
    core.EXPECTED_ATOM_COUNT = 4_491
    core.PAIR_COLUMNS = {key: PAIR_COLUMNS[key] for key in sorted(observed)}
    core.REFINEMENT_COLUMNS = {key: REFINEMENT_COLUMNS[key] for key in sorted(observed)}
    core.PROMPT_DIMENSION_COLUMNS = core.REFINEMENT_COLUMNS
    core.INITIAL_COLUMNS = {
        key: ("canonical_endpoint_concept",) for key in sorted(observed)
    }
    core.SAMPLE_CARD_COLUMNS = core.REFINEMENT_COLUMNS
    core.COLUMN_DESCRIPTIONS = COLUMN_DESCRIPTIONS
    core.PAIR_KEY_OPTIONAL_TRAILING_COLUMNS = {}
    core.SELECTOR_PROFILE_LIMIT = None
    core.SELECTOR_VALUE_LIMIT = None
    core.MAX_SOURCE_ROUNDS = 3
    core.MERGE_BATCH_SIZE = 40
    core.LOW_MAX_TOKENS = 8_192
    core.HIGH_MAX_TOKENS = 131_072
    core.INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = False
    core.INCLUDE_DOWNSTREAM_PROMPT_REVIEW = False
    core.SOURCE_LOCAL_FINAL_STATUS = "awaiting_agentic_review"
    core.PROMPT_REVIEW_TITLE = "Skin source-local semantic v4 prompt review"
    return paths


def prepare(run_root: Path) -> dict[str, Any]:
    build_input(run_root)
    paths = configure(run_root)
    return core.prepare_prompt_review(paths["review"])


def run(run_root: Path, approved_review_sha256: str) -> dict[str, Any]:
    paths = configure(run_root)
    manifest = core.run_semantic(
        paths["run"], review_manifest_path=paths["review"] / "manifest.json",
        approved_review_sha256=approved_review_sha256, parallelism=128,
    )
    manifest.update({
        "task": "skin_reaction", "evidence_release": paths["release"].name,
        "cross_source_merging": False, "publication_status": "candidate_unselected",
    })
    write_json_atomic(paths["run"] / "manifest.json", manifest)
    return manifest


def _validate_candidate(paths: dict[str, Path]) -> tuple[Path, pd.DataFrame]:
    candidate_path = paths["run"] / "source_semantic_bucket_map.parquet"
    candidate = pd.read_parquet(candidate_path)
    atoms = pd.read_parquet(paths["run"] / "input_atoms.parquet")
    if candidate.atom_id.duplicated().any() or len(candidate) != core.EXPECTED_ATOM_COUNT:
        raise ValueError("semantic candidate does not cover every atom exactly once")
    if set(candidate.atom_id) != set(atoms.atom_id):
        raise ValueError("semantic candidate atom universe differs from its frozen input")
    identity = candidate[["atom_id", "level", "source_id"]].merge(
        atoms[["atom_id", "level", "source_id"]], on="atom_id", validate="one_to_one",
        suffixes=("_candidate", "_input"),
    )
    if (identity.level_candidate != identity.level_input).any() or (
        identity.source_id_candidate != identity.source_id_input
    ).any():
        raise ValueError("semantic candidate changed an atom's source or level")
    return candidate_path, candidate


def _record_map(paths: dict[str, Path], semantic: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_parquet(paths["input"])
    records["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in records[["level", "source_id", "pair_bucket_key"]]
        .itertuples(index=False, name=None)
    ]
    mapped = records[
        ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id"]
    ].merge(semantic[["atom_id", "semantic_bucket_id"]], on="atom_id", validate="many_to_one")
    if len(mapped) != core.EXPECTED_RECORD_COUNT or mapped.source_row_uid.duplicated().any():
        raise ValueError("established semantic map does not cover every scoped record exactly once")
    return mapped.sort_values(["level", "source_row_uid"])


def _write_semantic_manifest(
    paths: dict[str, Path], review_path: Path, semantic: pd.DataFrame,
    record_map: pd.DataFrame,
) -> Path:
    run = paths["run"]
    manifest_path = run / "semantic_bucket_map_manifest.json"
    manifest = {
        "version": f"{VERSION}.established.v1", "status": "complete_reviewed",
        "task": "skin_reaction", "evidence_release": paths["release"].name,
        "levels": list(core.LEVELS), "cross_source_merging": False,
        "semantic_bucket_id_policy": "source_semantic_bucket_id",
        "agentic_review": {"path": str(review_path), "sha256": file_sha256(review_path)},
        "source_semantic_bucket_map_sha256": file_sha256(run / "source_semantic_bucket_map.parquet"),
        "semantic_bucket_map_sha256": file_sha256(run / "semantic_bucket_map.parquet"),
        "record_semantic_bucket_map_sha256": file_sha256(run / "record_semantic_bucket_map.parquet"),
        "input_atoms_sha256": file_sha256(run / "input_atoms.parquet"),
        "atom_count": len(semantic), "record_count": len(record_map),
        "semantic_bucket_count": int(semantic.semantic_bucket_id.nunique()),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def establish(run_root: Path, review_path: Path) -> dict[str, Any]:
    paths = configure(run_root)
    manifest_path = paths["run"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "awaiting_agentic_review":
        raise ValueError("semantic candidate is not awaiting agentic review")
    candidate_path, candidate = _validate_candidate(paths)
    review = json.loads(review_path.read_text())
    expected = {"version", "reviewer", "candidate_map_sha256", "decision", "rationale", "audit"}
    if set(review) != expected or review["version"] != f"{VERSION}.agentic_review.v1":
        raise ValueError("invalid semantic review schema")
    if review["decision"] != "approve":
        raise ValueError("semantic review did not approve establishment")
    if review["candidate_map_sha256"] != file_sha256(candidate_path):
        raise ValueError("semantic review targets another candidate map")

    semantic = candidate.copy()
    semantic["semantic_bucket_id"] = semantic.source_semantic_bucket_id
    semantic = semantic[
        ["level", "source_id", "source_semantic_bucket_id", "semantic_bucket_id", "atom_id"]
    ].sort_values(["level", "source_id", "semantic_bucket_id", "atom_id"])
    record_map = _record_map(paths, semantic)

    semantic_path = paths["run"] / "semantic_bucket_map.parquet"
    record_path = paths["run"] / "record_semantic_bucket_map.parquet"
    freeze_path = paths["run"] / "semantic_bucket_map_manifest.json"
    if semantic_path.exists() or record_path.exists() or freeze_path.exists():
        raise FileExistsError("established semantic outputs already exist")
    semantic.to_parquet(semantic_path, index=False)
    record_map.to_parquet(record_path, index=False)
    semantic_manifest = _write_semantic_manifest(paths, review_path, semantic, record_map)
    manifest.update({
        "status": "complete_reviewed", "publication_status": "reviewed_unselected",
        "agentic_review": {"path": str(review_path), "sha256": file_sha256(review_path)},
        "semantic_bucket_map": {"path": semantic_path.name, "sha256": file_sha256(semantic_path),
                                "rows": len(semantic), "semantic_bucket_count": semantic.semantic_bucket_id.nunique()},
        "record_semantic_bucket_map": {"path": record_path.name, "sha256": file_sha256(record_path),
                                       "rows": len(record_map)},
        "semantic_bucket_map_manifest": {
            "path": semantic_manifest.name, "sha256": file_sha256(semantic_manifest),
        },
    })
    write_json_atomic(manifest_path, manifest)
    return manifest


def status(run_root: Path) -> dict[str, Any]:
    paths = _paths(run_root)
    result: dict[str, Any] = {"run_root": str(run_root)}
    if (paths["run"] / "manifest.json").is_file():
        result["manifest"] = json.loads((paths["run"] / "manifest.json").read_text())
    database = paths["run"] / "requests.sqlite3"
    if database.is_file():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT count(*) total, sum(status='complete') complete, "
            "sum(status='failed') failed, sum(status='pending') pending FROM requests"
        ).fetchone()
        result["requests"] = dict(zip(("total", "complete", "failed", "pending"), row))
        connection.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "establish", "status"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_root)
    elif args.command == "run":
        if not args.approved_review_sha256:
            parser.error("run requires --approved-review-sha256")
        result = run(args.run_root, args.approved_review_sha256)
    elif args.command == "establish":
        if not args.review:
            parser.error("establish requires --review")
        result = establish(args.run_root, args.review)
    else:
        result = status(args.run_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
