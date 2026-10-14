"""Selected semantic releases and task-native evidence inputs."""

from __future__ import annotations
from predict.harnesses.branches.manifests import canonical_location
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any
import pandas as pd
from predict.utils.json import sha256_file
from collections.abc import Mapping
import pyarrow.parquet as pq
from .storage import _canonical_json
from .storage import _stable_id

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parent / "releases"
TASK_ALIASES = {
    "bbb": "bbb_martins",
    "bbb_martins": "bbb_martins",
    "oral": "bioavailability_ma",
    "bioavailability_ma": "bioavailability_ma",
    "skin": "skin_reaction",
    "skin_reaction": "skin_reaction",
    "ames": "ames",
    "dili": "dili",
    "carcinogens": "carcinogens",
}


@dataclass(frozen=True)
class SemanticBucketArtifacts:
    """Selected semantic/readout artifacts for one evidence-library release."""

    task: str
    release: str
    root: Path
    manifest: Path
    semantic_map: Path
    semantic_map_manifest: Path
    readout_bucket_map: Path | None
    record_readout_bucket_map: Path | None
    record_semantic_bucket_map: Path | None
    semantic_bucket_rankings: Path | None
    record_relevance_rankings: Path | None
    retrieval_eligibility: Path | None
    semantic_bucket_weights: Path | None
    central_ledger: Path | None = None


def normalize_task(task: str) -> str:
    try:
        return TASK_ALIASES[task]
    except KeyError as exc:
        raise ValueError(f"unsupported semantic-bucket task: {task}") from exc


def resolve_release(task: str, release: str = "v10") -> str:
    task = normalize_task(task)
    if release != "CURRENT":
        return release
    current = REPOSITORY_ROOT / "data" / "evidence_libraries" / task / "CURRENT"
    selected = current.read_text(encoding="utf-8").strip()
    if not selected or "/" in selected or "\\" in selected:
        raise ValueError(f"invalid evidence-library CURRENT pointer: {current}")
    return selected


def semantic_bucket_root(task: str, release: str = "v10") -> Path:
    task = normalize_task(task)
    return RELEASE_ROOT / task / resolve_release(task, release)


def generation_root(task: str, generation: str, release: str = "v10") -> Path:
    return semantic_bucket_root(task, release) / "generations" / generation


def eligibility_root(task: str, generation: str, release: str = "v10") -> Path:
    return semantic_bucket_root(task, release) / "eligibility" / generation


def _selected_path(root: Path, selected: dict[str, Any], key: str) -> Path:
    value = selected.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"semantic-bucket manifest has no selected.{key}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"semantic-bucket manifest path escapes its release: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_semantic_bucket_artifacts(
    task: str, release: str = "CURRENT"
) -> SemanticBucketArtifacts:
    """Resolve the reviewed artifact selection recorded by a release manifest."""
    task = normalize_task(task)
    release = resolve_release(task, release)
    root = semantic_bucket_root(task, release)
    manifest_path = root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "semantic_buckets.release.v1":
        raise ValueError(f"unsupported semantic-bucket manifest: {manifest_path}")
    if (
        document.get("task") != task
        or document.get("evidence_library_version") != release
    ):
        raise ValueError(f"semantic-bucket manifest identity mismatch: {manifest_path}")
    selected = document.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(
            f"semantic-bucket manifest has no selected mapping: {manifest_path}"
        )
    eligibility = selected.get("retrieval_eligibility")
    eligibility_path = (
        _selected_path(root, selected, "retrieval_eligibility") if eligibility else None
    )
    weights = selected.get("semantic_bucket_weights")
    weights_path = (
        _selected_path(root, selected, "semantic_bucket_weights") if weights else None
    )
    readout = selected.get("readout_bucket_map")
    record_readout = selected.get("record_readout_bucket_map")
    record_semantic = selected.get("record_semantic_bucket_map")
    if not record_readout and (not record_semantic):
        raise ValueError(
            f"semantic-bucket manifest has no record mapping: {manifest_path}"
        )
    return SemanticBucketArtifacts(
        task=task,
        release=release,
        root=root,
        manifest=manifest_path,
        semantic_map=_selected_path(root, selected, "semantic_map"),
        semantic_map_manifest=_selected_path(root, selected, "semantic_map_manifest"),
        readout_bucket_map=_selected_path(root, selected, "readout_bucket_map")
        if readout
        else None,
        record_readout_bucket_map=_selected_path(
            root, selected, "record_readout_bucket_map"
        )
        if record_readout
        else None,
        record_semantic_bucket_map=_selected_path(
            root, selected, "record_semantic_bucket_map"
        )
        if record_semantic
        else None,
        semantic_bucket_rankings=_selected_path(
            root, selected, "semantic_bucket_rankings"
        )
        if selected.get("semantic_bucket_rankings")
        else None,
        record_relevance_rankings=_selected_path(
            root, selected, "record_relevance_rankings"
        )
        if selected.get("record_relevance_rankings")
        else None,
        retrieval_eligibility=eligibility_path,
        semantic_bucket_weights=weights_path,
        central_ledger=_selected_path(root, selected, "central_ledger")
        if selected.get("central_ledger")
        else None,
    )


def load_reviewed_record_weights(
    task: str, release: str = "CURRENT"
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load selected record weights, preserving explicit provisional provenance."""
    task = normalize_task(task)
    release = resolve_release(task, release)
    manifest = semantic_bucket_root(task, release) / "manifest.json"
    return _cached_record_weights(task, release, sha256_file(manifest))


def load_skin_candidate_snapshot(
    manifest_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Load an immutable, unselected Skin projection with its map and weight lineage."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "semantic_retrieval_mean_weights.v2"
        or manifest.get("status") != "candidate_unreviewed"
        or manifest.get("provisional_uids") != 0
    ):
        raise ValueError("Skin candidate projection is incomplete")
    inputs = manifest["inputs"]
    pins = {str(manifest_path): sha256_file(manifest_path)}
    for key in ("ledger", "weights", "retrieval_uids"):
        entry = inputs[key]
        source = canonical_location(entry["path"]).resolve()
        if sha256_file(source) != entry["sha256"]:
            raise ValueError(f"Skin candidate {key} changed")
        pins[str(source)] = entry["sha256"]
    ledger_path = canonical_location(inputs["ledger"]["path"])
    ledger_manifest_path = ledger_path.parent / "manifest.json"
    ledger_manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))
    if (
        ledger_manifest.get("task") != "skin_reaction"
        or ledger_manifest.get("status") != "candidate_unreviewed"
        or ledger_manifest["ledger"]["sha256"] != inputs["ledger"]["sha256"]
    ):
        raise ValueError("Skin candidate ledger differs")
    pins[str(ledger_manifest_path.resolve())] = sha256_file(ledger_manifest_path)
    for key in ("stage3", "levels", "record_map", "semantic_manifest"):
        entry = ledger_manifest["inputs"][key]
        source = canonical_location(entry["path"]).resolve()
        if sha256_file(source) != entry["sha256"]:
            raise ValueError(f"Skin candidate {key} changed")
        pins[str(source)] = entry["sha256"]
    weights_manifest_path = (
        canonical_location(inputs["weights"]["path"]).parent / "manifest.json"
    )
    weights_manifest = json.loads(weights_manifest_path.read_text(encoding="utf-8"))
    if (
        weights_manifest.get("state") != "complete_unselected"
        or weights_manifest.get("weights_sha256") != inputs["weights"]["sha256"]
        or weights_manifest["inputs"]["record_map"]["sha256"]
        != ledger_manifest["inputs"]["record_map"]["sha256"]
    ):
        raise ValueError("Skin candidate weights differ from the ledger map")
    pins[str(weights_manifest_path.resolve())] = sha256_file(weights_manifest_path)
    projection = manifest["projection"]
    table = manifest_path.with_name(projection["path"])
    if sha256_file(table) != projection["sha256"]:
        raise ValueError("Skin candidate projection changed")
    pins[str(table.resolve())] = projection["sha256"]
    frame = pd.read_parquet(table)
    if (
        len(frame) != projection["rows"]
        or frame.duplicated(["source_row_uid", "level"]).any()
        or frame.semantic_bucket_id.isna().any()
        or (not frame.weight.between(0, 1).all())
        or (set(frame.weight_source) != {"candidate_family_mean"})
    ):
        raise ValueError("Skin candidate projection rows are invalid")
    expected = pd.read_parquet(
        canonical_location(inputs["retrieval_uids"]["path"]),
        columns=["source_row_uid", "level"],
    )
    if set(zip(frame.source_row_uid, frame.level)) != set(
        zip(expected.source_row_uid, expected.level)
    ):
        raise ValueError("Skin candidate projection does not cover retrieval UIDs")
    rows = {
        (str(row.source_row_uid), str(row.level)): {
            "semantic_bucket_id": str(row.semantic_bucket_id),
            "semantic_weight": float(row.weight),
            "semantic_weight_source": "candidate_family_mean",
        }
        for row in frame.itertuples(index=False)
    }
    return (rows, pins)


def load_dili_candidate_snapshot(
    snapshot_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Load the complete unselected DILI V2 calibration for its pinned retrieval UIDs."""
    snapshot_path = snapshot_path.resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("status") != "complete_unselected":
        raise ValueError("DILI candidate snapshot is incomplete")
    entry = snapshot["tasks"]["dili"]
    calibration_path = (
        canonical_location(entry["calibrated_weights"]["path"]).parent / "manifest.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("state") != "complete_unselected"
        or calibration["inputs"]["record_map"]["sha256"]
        != entry["candidate_record_map"]["sha256"]
        or calibration["weights_sha256"] != entry["calibrated_weights"]["sha256"]
    ):
        raise ValueError("DILI candidate calibration differs from the snapshot")
    pins = {
        str(snapshot_path): sha256_file(snapshot_path),
        str(calibration_path): sha256_file(calibration_path),
    }
    sources = {
        "record_map": entry["candidate_record_map"],
        "weights": entry["calibrated_weights"],
        "retrieval_uids": calibration["inputs"]["retrieval_uids"],
    }
    for source in sources.values():
        path = canonical_location(source["path"]).resolve()
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"DILI candidate input changed: {path}")
        pins[str(path)] = source["sha256"]
    uids = pd.read_parquet(
        canonical_location(sources["retrieval_uids"]["path"]),
        columns=["source_row_uid", "level"],
    )
    mapping = pd.read_parquet(
        canonical_location(sources["record_map"]["path"]),
        columns=["source_row_uid", "level", "semantic_bucket_id"],
    )
    weights = pd.read_parquet(
        sources["weights"]["path"], columns=["level", "semantic_bucket_id", "weight"]
    )
    frame = mapping.merge(
        weights, on=["level", "semantic_bucket_id"], how="left", validate="many_to_one"
    )
    covered = uids.merge(
        frame, on=["source_row_uid", "level"], how="left", validate="one_to_one"
    )
    if (
        covered.semantic_bucket_id.isna().any()
        or covered.weight.isna().any()
        or (not frame.weight.dropna().between(0, 1).all())
    ):
        raise ValueError("DILI candidate retrieval weights are incomplete")
    rows = {
        (str(row.source_row_uid), str(row.level)): {
            "semantic_bucket_id": str(row.semantic_bucket_id),
            "semantic_weight": float(row.weight),
            "semantic_weight_source": "candidate_calibrated_v2",
        }
        for row in frame.dropna(subset=["weight"]).itertuples(index=False)
    }
    extension_path = (
        REPOSITORY_ROOT
        / "data/caches/semantic_buckets/tdc_v2_dili_parent_disjoint_v2/projection/manifest.json"
    )
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    if (
        extension.get("status") != "candidate_unreviewed"
        or extension.get("schema_version") != "semantic_retrieval_mean_weights.v2"
    ):
        raise ValueError("TDC DILI semantic extension is incomplete")
    table = extension_path.with_name(extension["projection"]["path"])
    for path, digest in (
        (extension_path, sha256_file(extension_path)),
        (table, extension["projection"]["sha256"]),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"TDC DILI semantic extension changed: {path}")
        pins[str(path.resolve())] = digest
    fallback = pd.read_parquet(table)
    if (
        len(fallback) != extension["projection"]["rows"]
        or fallback.duplicated(["source_row_uid", "level"]).any()
        or (not fallback.weight.between(0, 1).all())
    ):
        raise ValueError("TDC DILI semantic extension rows are invalid")
    for row in fallback.itertuples(index=False):
        rows.setdefault(
            (str(row.source_row_uid), str(row.level)),
            {
                "semantic_bucket_id": ""
                if pd.isna(row.semantic_bucket_id)
                else str(row.semantic_bucket_id),
                "semantic_weight": float(row.weight),
                "semantic_weight_source": str(row.weight_source),
            },
        )
    return (rows, pins)


@lru_cache(maxsize=1)
def _cached_record_weights(
    task: str, release: str, manifest_sha256: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    artifacts = resolve_semantic_bucket_artifacts(task, release)
    document = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    rankings = artifacts.record_relevance_rankings
    if document.get("status") != "complete_reviewed" or rankings is None:
        raise ValueError(f"No reviewed semantic weights selected for {task}/{release}")
    entry = document.get("files", {}).get("record_relevance_rankings", {})
    if (
        sha256_file(artifacts.manifest) != manifest_sha256
        or not rankings.is_file()
        or (not entry.get("sha256"))
    ):
        raise ValueError(f"Selected semantic rankings changed for {task}/{release}")
    if artifacts.central_ledger is not None:
        projection_path = rankings.parent / "manifest.json"
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        selected_files = document.get("files", {})
        expected_projection = selected_files.get("weight_projection_manifest", {}).get(
            "sha256"
        )
        if (
            projection.get("status")
            not in {"complete_reviewed", "complete_reviewed_with_provisional_mean"}
            or projection.get("schema_version")
            not in {
                "semantic_retrieval_mean_weights.v1",
                "semantic_retrieval_mean_weights.v2",
            }
            or (not expected_projection)
            or (sha256_file(projection_path) != expected_projection)
            or (
                projection.get("inputs", {}).get("ledger", {}).get("sha256")
                != selected_files.get("central_ledger", {}).get("sha256")
            )
            or (
                projection.get("inputs", {}).get("weights", {}).get("sha256")
                != selected_files.get("semantic_bucket_weights", {}).get("sha256")
            )
            or (projection.get("projection", {}).get("sha256") != entry.get("sha256"))
        ):
            raise ValueError(
                f"Selected mean-weight projection is not ledger-bound: {task}/{release}"
            )
    rows = {}
    ranking_rows = pd.read_parquet(rankings)
    has_source = "weight_source" in ranking_rows
    if (
        artifacts.central_ledger is not None
        and projection["schema_version"] == "semantic_retrieval_mean_weights.v2"
        and (not has_source)
    ):
        raise ValueError(
            f"Selected projection lacks weight provenance: {task}/{release}"
        )
    for row in ranking_rows.itertuples(index=False):
        key = (str(row.source_row_uid), str(row.level))
        weight = float(row.weight)
        if key in rows or not math.isfinite(weight) or (not 0 <= weight <= 1):
            raise ValueError(
                f"Invalid selected semantic weight: {task}/{release}/{key}"
            )
        source = str(row.weight_source) if has_source else "reviewed_family_mean"
        bucket = (
            None if pd.isna(row.semantic_bucket_id) else str(row.semantic_bucket_id)
        )
        if (bucket is None) != (source == "provisional_task_level_mean"):
            raise ValueError(
                f"Invalid provisional semantic assignment: {task}/{release}/{key}"
            )
        rows[key] = {
            "semantic_bucket_id": bucket,
            "semantic_weight": weight,
            "semantic_weight_source": source,
        }
    if (
        artifacts.central_ledger is not None
        and projection["schema_version"] == "semantic_retrieval_mean_weights.v2"
        and (
            sum(
                (
                    row["semantic_weight_source"] == "provisional_task_level_mean"
                    for row in rows.values()
                )
            )
            != projection["provisional_uids"]
        )
    ):
        raise ValueError(f"Selected provisional count changed: {task}/{release}")
    return (
        rows,
        {
            "release": release,
            "release_manifest": str(artifacts.manifest),
            "release_manifest_sha256": manifest_sha256,
            "record_relevance_rankings": str(rankings),
            "record_relevance_rankings_sha256": entry["sha256"],
            "weighted_record_count": len(rows),
        },
    )


def load_record_bucket_map(
    task: str, release: str = "CURRENT", *, eligible_only: bool = False
) -> pd.DataFrame:
    """Load record-to-semantic/readout assignments, optionally eligibility-filtered."""
    artifacts = resolve_semantic_bucket_artifacts(task, release)
    mapping_path = (
        artifacts.record_semantic_bucket_map or artifacts.record_readout_bucket_map
    )
    if mapping_path is None:
        raise ValueError(f"no record mapping is selected for {artifacts.task}")
    mapping = pd.read_parquet(mapping_path)
    if not eligible_only:
        return mapping
    if artifacts.retrieval_eligibility is None:
        raise ValueError(f"no retrieval eligibility is selected for {artifacts.task}")
    eligibility = pd.read_parquet(
        artifacts.retrieval_eligibility,
        columns=[
            "canonical_record_id",
            "source_row_uid",
            "level",
            "semantic_bucket_id",
            "retrieval_eligible",
        ],
    )
    keys = ["canonical_record_id", "source_row_uid", "level", "semantic_bucket_id"]
    if eligibility.duplicated(keys).any():
        raise ValueError(
            f"retrieval eligibility repeats record assignments for {artifacts.task}"
        )
    joined = mapping.merge(eligibility, on=keys, how="inner", validate="one_to_one")
    if len(joined) != len(eligibility):
        raise ValueError(
            f"retrieval eligibility references unmapped records for {artifacts.task}"
        )
    return joined[joined["retrieval_eligible"]].reset_index(drop=True)


def load_central_ledger(task: str, release: str = "CURRENT") -> pd.DataFrame:
    """Load the selected physical-UID census, including explicitly pending rows."""
    artifacts = resolve_semantic_bucket_artifacts(task, release)
    if artifacts.central_ledger is None:
        raise ValueError(f"no central semantic ledger is selected for {artifacts.task}")
    return pd.read_parquet(artifacts.central_ledger)


def validate_column_contracts(*, policy: Mapping) -> None:
    """Keep semantic decisions inside each source's pair-bucket schema."""
    if set(policy["REFINEMENT_COLUMNS"]) != set(policy["PAIR_COLUMNS"]):
        raise ValueError("refinement-column sources do not match pair-bucket sources")
    for source, pair_columns in policy["PAIR_COLUMNS"].items():
        pair = set(pair_columns)
        refinement = set(policy["REFINEMENT_COLUMNS"][source])
        prompt = set(policy["PROMPT_DIMENSION_COLUMNS"][source])
        initial = set(policy["INITIAL_COLUMNS"][source])
        if not refinement <= pair:
            raise ValueError(f"{source} refinement columns are not pair-bucket columns")
        if not prompt <= refinement:
            raise ValueError(f"{source} prompt dimensions are not refinement columns")
        if not initial <= refinement:
            raise ValueError(f"{source} initial columns are not refinement columns")


def parse_pair_bucket(
    source: str, pair_bucket_key: str, *, policy: Mapping
) -> dict[str, str]:
    values = json.loads(pair_bucket_key)
    columns = policy["PAIR_COLUMNS"][source]
    if not isinstance(values, list) or values[:1] != [source]:
        raise ValueError(f"invalid {source} pair bucket: {pair_bucket_key}")
    minimum = len(columns) - policy["PAIR_KEY_OPTIONAL_TRAILING_COLUMNS"].get(source, 0)
    if len(values) - 1 not in {minimum, len(columns)}:
        raise ValueError(
            f"{source} pair bucket has {len(values) - 1} values; expected {minimum} or {len(columns)}"
        )
    parsed = {
        column: str(value) for column, value in zip(columns, values[1:], strict=False)
    }
    for column in columns[len(values) - 1 :]:
        parsed[column] = "__not_present_in_pair_key__"
    return parsed


def _load_atoms(*, policy: Mapping) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    validate_column_contracts(policy=policy)
    mapping = pq.read_table(policy["RECORD_MAP"]).to_pandas()
    mapping = mapping[mapping["level"].isin(policy["LEVELS"])].copy()
    if len(mapping) != policy["EXPECTED_RECORD_COUNT"]:
        raise ValueError(f"unexpected level-scoped record count: {len(mapping)}")
    if mapping["source_row_uid"].duplicated().any():
        raise ValueError("source_row_uid is not unique in the level-scoped mapping")
    key_columns = [
        "level",
        "source_id",
        "pair_bucket_key",
        "node_key",
        "relevance_bucket",
    ]
    counts = (
        mapping.groupby(key_columns, sort=True, dropna=False)
        .size()
        .rename("record_count")
    )
    representatives = (
        mapping.sort_values("canonical_record_id")
        .drop_duplicates(key_columns)[
            key_columns + ["source_row_uid", "canonical_record_id"]
        ]
        .set_index(key_columns)
    )
    atoms = counts.to_frame().join(representatives).reset_index()
    if (
        policy["EXPECTED_ATOM_COUNT"] is not None
        and len(atoms) != policy["EXPECTED_ATOM_COUNT"]
    ):
        raise ValueError(f"unexpected level-scoped pair-bucket count: {len(atoms)}")
    record_columns = sorted(
        {
            "source_row_uid",
            "canonical_record_id",
            *(
                column
                for columns in policy["SAMPLE_CARD_COLUMNS"].values()
                for column in columns
            ),
        }
    )
    records = pq.read_table(policy["RECORDS"], columns=record_columns).to_pandas()
    records = records.drop_duplicates("source_row_uid").set_index("source_row_uid")
    sample_cards: dict[str, dict[str, Any]] = {}
    atom_rows = []
    for row in atoms.itertuples(index=False):
        atom = _stable_id("atom", row.level, row.source_id, row.pair_bucket_key)
        values = parse_pair_bucket(row.source_id, row.pair_bucket_key, policy=policy)
        source_record = records.loc[row.source_row_uid].to_dict()
        card = {
            column: source_record[column]
            for column in policy["SAMPLE_CARD_COLUMNS"][row.source_id]
            if pd.notna(source_record.get(column))
            and str(source_record[column]).strip()
        }
        sample_cards[atom] = {
            "canonical_record_id": row.canonical_record_id,
            **(
                {"pair_bucket_key": row.pair_bucket_key}
                if policy["INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS"]
                else {}
            ),
            **card,
        }
        atom_rows.append(
            {
                "atom_id": atom,
                "level": row.level,
                "source_id": row.source_id,
                "pair_bucket_key": row.pair_bucket_key,
                "initial_node_key": row.node_key,
                "initial_relevance_bucket": row.relevance_bucket,
                "record_count": int(row.record_count),
                "values_json": _canonical_json(values),
            }
        )
    return (pd.DataFrame(atom_rows), sample_cards)
