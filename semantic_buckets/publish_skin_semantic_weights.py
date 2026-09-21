"""Publish the completed Skin semantic-family weights for evidence v6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT / "releases/skin_reaction"
RELEASE_ID = "v10_main_universe_v6"
RUN_ID = "skin_morgan100_official_two_pass_v1_20260919"
SEMANTIC_ID = "source_local_semantic_parent_v1_20260919"
WORLD_ID = "skin_morgan_top100_valid_test_l2_l3_v1"
SEMANTIC_ROOT = ROOT / (
    "provenance/source_local_semantic_v4/"
    "skin_main_universe_v5_semantic_parent_v1_20260919/semantic_run"
)
WORLD_ROOT = ROOT / "provenance/retrieval_worlds" / WORLD_ID
RUN_ROOT = ROOT / "provenance/semantic_weight_two_pass_v1" / RUN_ID
EVIDENCE_ROOT = ROOT.parent / "data/evidence_libraries/skin_reaction"
FINAL_ROOT = RELEASE_ROOT / RELEASE_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o664)


def _evidence_binding() -> dict[str, Any]:
    if (EVIDENCE_ROOT / "CURRENT").read_text(encoding="utf-8").strip() != RELEASE_ID:
        raise ValueError("Skin evidence CURRENT is not v10_main_universe_v6")
    names = (
        "03_pair_buckets/records.parquet",
        "03_pair_buckets/pair_bucket_records.parquet",
        "02_canonicalized/source_contract.json",
        "level_mapping/records.parquet",
        "level_mapping/assay_transfer_record_eligibility.parquet",
    )
    pairs = {}
    for name in names:
        old, current = EVIDENCE_ROOT / "v10_main_universe_v5" / name, EVIDENCE_ROOT / RELEASE_ID / name
        old_hash, current_hash = _sha256(old), _sha256(current)
        if old_hash != current_hash:
            raise ValueError(f"evidence input changed between v5 and v6: {name}")
        pairs[name] = {"v5": old_hash, "v6": current_hash}
    return {
        "version": "skin_weight_evidence_binding.v1",
        "v5_release_manifest_sha256": _sha256(EVIDENCE_ROOT / "v10_main_universe_v5/manifest.json"),
        "v6_release_manifest_sha256": _sha256(EVIDENCE_ROOT / f"{RELEASE_ID}/manifest.json"),
        "scientific_inputs": pairs,
    }


def _verify_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_manifest = json.loads((RUN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete_unselected":
        raise ValueError("official Skin two-pass run is not complete")
    semantic_manifest = json.loads(
        (SEMANTIC_ROOT / "semantic_bucket_map_manifest.json").read_text(encoding="utf-8")
    )
    world_manifest = json.loads((WORLD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if semantic_manifest.get("status") != "complete_reviewed":
        raise ValueError("reviewed Skin semantic generation is not complete")
    if world_manifest.get("status") != "complete":
        raise ValueError("Skin Morgan retrieval world is not complete")
    input_paths = {
        "semantic_manifest": SEMANTIC_ROOT / "semantic_bucket_map_manifest.json",
        "semantic_map": SEMANTIC_ROOT / "semantic_bucket_map.parquet",
        "input_atoms": SEMANTIC_ROOT / "input_atoms.parquet",
        "sample_cards": SEMANTIC_ROOT / "sample_cards.json.gz",
        "world_manifest": WORLD_ROOT / "manifest.json",
        "world": WORLD_ROOT / "semantic_bucket_world.parquet",
    }
    for name, path in input_paths.items():
        if _sha256(path) != run_manifest["inputs"][name]["sha256"]:
            raise ValueError(f"official run input changed: {name}")
    required = ("weights.parquet", "semantic_bucket_rankings.parquet")
    for name in required:
        if _sha256(RUN_ROOT / name) != run_manifest["artifacts"][name]:
            raise ValueError(f"official run artifact changed: {name}")
    weights = pd.read_parquet(RUN_ROOT / "weights.parquet")
    world = pd.read_parquet(WORLD_ROOT / "semantic_bucket_world.parquet")
    semantic = pd.read_parquet(SEMANTIC_ROOT / "semantic_bucket_map.parquet")
    record_map = SEMANTIC_ROOT / "record_semantic_bucket_map.parquet"
    if _sha256(record_map) != semantic_manifest["record_semantic_bucket_map_sha256"]:
        raise ValueError("reviewed record semantic map changed")
    if len(weights) != 1244 or weights.semantic_bucket_id.duplicated().any():
        raise ValueError("official weights do not cover exactly 1,244 families")
    if set(weights.semantic_bucket_id) != set(world.semantic_bucket_id):
        raise ValueError("weights and Morgan world bucket sets differ")
    if len(semantic) != 4491 or semantic.semantic_bucket_id.nunique() != 1316:
        raise ValueError("reviewed semantic map counts changed")
    binding = _evidence_binding()
    return weights, world, semantic, {"run_manifest": run_manifest, "binding": binding}


def _rank_columns(weights: pd.DataFrame) -> pd.DataFrame:
    frame = weights.copy()
    frame["weight_rationale"] = frame.pop("rationale")
    frame["task_id"] = frame["task"]
    frame["weight_rank"] = frame.pop("final_rank").astype(int)
    frame["level_rank"] = frame["weight_rank"]
    sizes = frame.groupby("level").semantic_bucket_id.transform("size")
    frame["weight_percentile"] = (sizes - frame.weight_rank) / (sizes - 1).clip(lower=1) * 100
    frame["level_percentile"] = frame["weight_percentile"]
    frame["assignment_method"] = "llm_calibration"
    frame["source_epoch"] = "official_two_pass_complete"
    frame["source_run_id"] = RUN_ID
    return frame


def _record_rankings(mapping: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "task_id", "task", "level", "semantic_bucket_id", "weight", "weight_rank",
        "weight_percentile", "level_rank", "level_percentile", "weight_rationale",
        "assignment_method", "source_epoch", "source_run_id",
    ]
    records = mapping.merge(
        weights[keep], on=["level", "semantic_bucket_id"], how="inner", validate="many_to_one"
    )
    if len(records) != int(mapping.semantic_bucket_id.isin(set(weights.semantic_bucket_id)).sum()):
        raise ValueError("record ranking join dropped or duplicated weighted records")
    return records.sort_values(["level", "weight_rank", "source_row_uid"], kind="stable")


def _write_sources(stage: Path) -> dict[str, str]:
    generation = stage / "generations" / SEMANTIC_ID
    for name in (
        "semantic_bucket_map.parquet", "semantic_bucket_map_manifest.json",
        "record_semantic_bucket_map.parquet", "agentic_review.json", "manifest.json",
    ):
        _copy(SEMANTIC_ROOT / name, generation / ("semantic_run_manifest.json" if name == "manifest.json" else name))
    return {
        "semantic_map": f"generations/{SEMANTIC_ID}/semantic_bucket_map.parquet",
        "semantic_map_manifest": f"generations/{SEMANTIC_ID}/semantic_bucket_map_manifest.json",
        "record_semantic_bucket_map": f"generations/{SEMANTIC_ID}/record_semantic_bucket_map.parquet",
    }


def _write_weighting(stage: Path, weights: pd.DataFrame) -> dict[str, str]:
    weighting = stage / "weighting" / RUN_ID
    weighting.mkdir(parents=True)
    weights.to_parquet(weighting / "semantic_bucket_weights.parquet", index=False)
    weights.to_csv(weighting / "semantic_bucket_weights.tsv", sep="\t", index=False)
    rankings = weights[
        ["task_id", "task", "level", "semantic_bucket_id", "weight", "weight_rank",
         "weight_percentile", "level_rank", "level_percentile", "weight_rationale",
         "assignment_method", "source_epoch", "source_run_id", "pass1_order_rank"]
    ].sort_values(["level", "weight_rank", "semantic_bucket_id"])
    rankings.to_parquet(weighting / "semantic_bucket_rankings.parquet", index=False)
    rankings.to_csv(weighting / "semantic_bucket_rankings.tsv", sep="\t", index=False)
    _copy(RUN_ROOT / "manifest.json", weighting / "source_run_manifest.json")
    prefix = f"weighting/{RUN_ID}/"
    return {
        "semantic_bucket_weights": prefix + "semantic_bucket_weights.parquet",
        "semantic_bucket_rankings": prefix + "semantic_bucket_rankings.parquet",
    }


def _write_records(stage: Path, records: pd.DataFrame) -> str:
    weighting = stage / "weighting" / RUN_ID
    records.to_parquet(weighting / "record_relevance_rankings.parquet", index=False)
    return f"weighting/{RUN_ID}/record_relevance_rankings.parquet"


def _manifest(stage: Path, weights: pd.DataFrame, semantic: pd.DataFrame,
              world: pd.DataFrame, mapping: pd.DataFrame, selected: dict[str, str],
              provenance: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for key, relative in selected.items():
        files[key] = {"path": relative, "sha256": _sha256(stage / relative)}
    uncovered = set(semantic.semantic_bucket_id) - set(weights.semantic_bucket_id)
    return {
        "schema_version": "semantic_buckets.release.v1",
        "status": "complete_reviewed",
        "task": "skin_reaction",
        "evidence_library_version": RELEASE_ID,
        "semantic_generation": SEMANTIC_ID,
        "weight_generation": RUN_ID,
        "weight_scope": "morgan_top100_valid_test_l2_l3",
        "semantic_mode": "semantic_family_native",
        "selected": selected,
        "files": files,
        "counts": {
            "semantic_atoms": len(semantic), "semantic_buckets": semantic.semantic_bucket_id.nunique(),
            "weighted_buckets": len(weights), "uncovered_buckets": len(uncovered),
            "record_assignments": len(mapping),
            "weighted_record_assignments": int(mapping.semantic_bucket_id.isin(
                set(weights.semantic_bucket_id)
            ).sum()),
            "world_records": int(world.record_count.sum()),
        },
        "upstream": {
            "semantic_run_manifest_sha256": _sha256(SEMANTIC_ROOT / "manifest.json"),
            "world_manifest_sha256": _sha256(WORLD_ROOT / "manifest.json"),
            "world_sha256": _sha256(WORLD_ROOT / "semantic_bucket_world.parquet"),
        },
        "provenance": provenance,
    }


def _validate_stage(stage: Path) -> dict[str, Any]:
    document = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    for item in document["files"].values():
        path = stage / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"published artifact changed: {path}")
    weights = pd.read_parquet(stage / document["selected"]["semantic_bucket_weights"])
    records = pd.read_parquet(stage / document["selected"]["record_relevance_rankings"])
    if weights.semantic_bucket_id.duplicated().any() or records.empty:
        raise ValueError("published weight artifacts are invalid")
    if records.weight.isna().any() or records.semantic_bucket_id.nunique() != len(weights):
        raise ValueError("published record rankings have incomplete weight coverage")
    return document


def build() -> dict[str, Any]:
    if FINAL_ROOT.exists():
        raise FileExistsError(FINAL_ROOT)
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{RELEASE_ID}.", dir=RELEASE_ROOT))
    try:
        raw_weights, world, semantic, provenance = _verify_sources()
        mapping = pd.read_parquet(SEMANTIC_ROOT / "record_semantic_bucket_map.parquet")
        weights = _rank_columns(raw_weights)
        selected = _write_sources(stage)
        selected.update(_write_weighting(stage, weights))
        records = _record_rankings(mapping, weights)
        selected["record_relevance_rankings"] = _write_records(stage, records)
        uncovered = semantic[~semantic.semantic_bucket_id.isin(set(weights.semantic_bucket_id))]
        audit = stage / "audit"
        audit.mkdir()
        uncovered.to_parquet(audit / "uncovered_semantic_bucket_atoms.parquet", index=False)
        uncovered.to_csv(audit / "uncovered_semantic_bucket_atoms.tsv", sep="\t", index=False)
        selected["uncovered_semantic_bucket_atoms"] = "audit/uncovered_semantic_bucket_atoms.parquet"
        selected["uncovered_semantic_bucket_atoms_tsv"] = "audit/uncovered_semantic_bucket_atoms.tsv"
        provenance_dir = stage / "provenance"
        provenance_dir.mkdir()
        write_json_atomic(provenance_dir / "evidence_input_binding.json", provenance["binding"])
        provenance["evidence_input_binding"] = "provenance/evidence_input_binding.json"
        document = _manifest(stage, weights, semantic, world, mapping, selected, provenance)
        write_json_atomic(stage / "manifest.json", document)
        _validate_stage(stage)
        stage.replace(FINAL_ROOT)
        return json.loads((FINAL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    args = parser.parse_args()
    print(json.dumps(build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
