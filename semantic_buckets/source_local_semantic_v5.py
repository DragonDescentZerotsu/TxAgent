"""Run semantic-only source-local refinement for a completed V10 release."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v10 import build_current_release
from data.processing.evidence_library.versions.v10.task_registry import import_task_module
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "source_local_semantic_v5.v1"
TASK_NAMES = {"ames": "AMES", "dili": "DILI", "carcinogens": "Carcinogens"}
PROMPT_ROOT = Path(__file__).with_name("prompts") / "source_local_semantic_v4"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_ENDPOINTS = (
    "http://dgx017:50001/v1",
    "http://dgx020:50002/v1",
)
DESCRIPTIONS = {
    "canonical_endpoint_name": "normalized biological endpoint",
    "canonical_endpoint_concept": "reviewed biological endpoint concept",
    "canonical_measurement_scale_id": "measurement or response representation",
    "canonical_assay_context": "normalized assay, readout, or experimental context",
}


@dataclass(frozen=True)
class Workflow:
    task: str
    release_root: Path
    run_root: Path
    model: str
    endpoints: tuple[dict[str, Any], ...]
    contract: Any
    pair_columns: dict[str, tuple[str, ...]]
    refinement_columns: dict[str, tuple[str, ...]]
    endpoint_columns: dict[str, str]
    retrieval_index: Path

    @property
    def records(self) -> Path:
        return self.release_root / "03_pair_buckets/records.parquet"

    @property
    def canonical_records(self) -> Path:
        return self.release_root / "02_canonicalized/records.parquet"

    @property
    def levels(self) -> Path:
        return self.release_root / "level_mapping/records.parquet"

    @property
    def input(self) -> Path:
        return self.run_root / "input/record_relevance_map.parquet"

    @property
    def review(self) -> Path:
        return self.run_root / "prompt_review"

    @property
    def run(self) -> Path:
        return self.run_root / "semantic_run"


def _contract(task: str) -> Any:
    if task not in TASK_NAMES:
        raise ValueError(f"unsupported semantic task: {task}")
    downstream = import_task_module(task, "build_starling_downstream_artifacts")
    contract = downstream.get_spec().policy.record_contract
    if contract is None:
        raise ValueError(f"{task} has no canonical record contract")
    return contract


def _semantic_columns(
    contract: Any,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    pair_columns: dict[str, tuple[str, ...]] = {}
    refinements: dict[str, tuple[str, ...]] = {}
    for source, spec in contract.pair_buckets.items():
        profile = contract.sources[source]
        endpoint = (
            "canonical_endpoint_concept"
            if "canonical_endpoint_concept" in profile.canonical_output_fields
            else "canonical_endpoint_name"
        )
        columns = (
            endpoint,
            "canonical_unit_text",
            *spec.additional_dimensions,
        )
        semantic = tuple(
            column
            for column in columns
            if column == endpoint
            or (
                column != "canonical_unit_text"
                and "species" not in column
                and "population" not in column
            )
        )
        if semantic[:1] != (endpoint,) or len(semantic) < 2:
            raise ValueError(f"{source} lacks endpoint-first semantic dimensions")
        pair_columns[source] = columns
        refinements[source] = semantic
    return pair_columns, refinements


def _endpoint_specs(urls: tuple[str, ...], max_inflight: int) -> tuple[dict[str, Any], ...]:
    if max_inflight < 1:
        raise ValueError("max_inflight must be positive")
    specs = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid endpoint URL: {url}")
        specs.append(
            {
                "name": parsed.netloc,
                "base_url": url.rstrip("/"),
                "provider": "local",
                "credential_env": "",
                "max_inflight": max_inflight,
            }
        )
    return tuple(specs)


def workflow(
    task: str,
    release_root: Path,
    run_root: Path,
    *,
    model: str = DEFAULT_MODEL,
    endpoint_urls: tuple[str, ...] = DEFAULT_ENDPOINTS,
    max_inflight: int = 128,
    retrieval_index: Path,
) -> Workflow:
    contract = _contract(task)
    pair_columns, refinements = _semantic_columns(contract)
    return Workflow(
        task=task,
        release_root=release_root.resolve(),
        run_root=run_root.resolve(),
        model=model,
        endpoints=_endpoint_specs(endpoint_urls, max_inflight),
        contract=contract,
        pair_columns=pair_columns,
        refinement_columns=refinements,
        endpoint_columns={
            source: columns[0] for source, columns in pair_columns.items()
        },
        retrieval_index=retrieval_index.resolve(),
    )


def _validated_release(config: Workflow) -> dict[str, Any]:
    result = build_current_release._validate_release(config.task, config.release_root)
    required = (config.records, config.canonical_records, config.levels)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"semantic inputs are missing: {missing}")
    return result


def _joined_records(config: Workflow) -> pd.DataFrame:
    records = pd.read_parquet(config.records)[
        ["canonical_record_id", "source_row_uid", "source_id", "pair_bucket_key"]
    ]
    levels = pd.read_parquet(config.levels)[
        ["canonical_record_id", "source_row_uid", "level"]
    ]
    keys = ["canonical_record_id", "source_row_uid"]
    if records.duplicated(keys).any() or levels.duplicated(keys).any():
        raise ValueError("semantic inputs repeat canonical-record/UID identity")
    joined = records.merge(levels, on=keys, validate="one_to_one")
    if len(joined) != len(records) or len(joined) != len(levels):
        raise ValueError("Stage 3 and level mapping do not have exact UID coverage")
    unknown = set(joined.source_id) - set(config.pair_columns)
    if unknown:
        raise ValueError(f"unregistered semantic sources: {sorted(unknown)}")
    return joined


def _endpoint_from_key(config: Workflow, source: str, value: str) -> str:
    parsed = json.loads(value)
    expected = config.pair_columns[source]
    if not isinstance(parsed, list) or parsed[:1] != [source]:
        raise ValueError(f"invalid pair bucket for {source}")
    if len(parsed) != len(expected) + 1:
        raise ValueError(f"{source} pair bucket has an unexpected shape")
    return str(parsed[1])


def _retrieval_uids(config: Workflow) -> set[str]:
    index = json.loads(config.retrieval_index.read_text(encoding="utf-8"))
    if index.get("status") != "complete" or index.get("task_id") != config.task:
        raise ValueError("retrieval index is incomplete or belongs to another task")
    selected: set[str] = set()
    for subset in ("valid", "test"):
        for level, entry in index["splits"][subset]["levels"].items():
            if level == "L1":
                continue
            manifest_path = config.retrieval_index.parent / entry["manifest"]
            if file_sha256(manifest_path) != entry["manifest_sha256"]:
                raise ValueError(f"retrieval manifest hash changed: {subset}/{level}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            database = manifest_path.with_name(str(manifest["database"]))
            if file_sha256(database) != manifest["database_sha256"]:
                raise ValueError(f"retrieval database hash changed: {subset}/{level}")
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                selected.update(str(row[0]) for row in connection.execute(
                    "SELECT DISTINCT item_id FROM rankings"
                ))
    if not selected:
        raise ValueError("retrieval index selects no L2+ UIDs")
    return selected


def build_input(config: Workflow) -> dict[str, Any]:
    release_receipt = _validated_release(config)
    joined = _joined_records(config)
    joined = joined[pd.to_numeric(joined.level, errors="raise") >= 2].copy()
    selected = _retrieval_uids(config)
    available = set(joined.source_row_uid)
    if not selected <= available:
        raise ValueError(f"retrieval scope has {len(selected - available)} unknown UIDs")
    joined = joined[joined.source_row_uid.isin(selected)].copy()
    if joined.empty:
        raise ValueError("semantic refinement has no L2+ records")
    endpoints = [
        _endpoint_from_key(config, source, key)
        for source, key in joined[["source_id", "pair_bucket_key"]].itertuples(
            index=False, name=None
        )
    ]
    joined["level"] = joined.level.map(lambda value: f"L{int(value)}")
    joined["relevance_bucket"] = [
        core._canonical_json({config.endpoint_columns[source]: value})
        for source, value in zip(joined.source_id, endpoints, strict=True)
    ]
    joined["node_key"] = joined.relevance_bucket
    joined = joined.sort_values(["level", "source_id", "source_row_uid"])
    config.input.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(config.input, index=False)
    manifest = _input_manifest(config, joined, release_receipt)
    write_json_atomic(config.input.parent / "manifest.json", manifest)
    return manifest


def _input_manifest(
    config: Workflow, rows: pd.DataFrame, release_receipt: dict[str, Any]
) -> dict[str, Any]:
    atoms = rows.groupby(
        ["level", "source_id", "pair_bucket_key"], sort=False
    ).ngroups
    parents = rows.groupby(["level", "source_id", "node_key"], sort=False).ngroups
    return {
        "version": f"{VERSION}.input.v1",
        "status": "complete",
        "task": config.task,
        "evidence_release": config.release_root.name,
        "initial_parent_policy": "level + source_id + reviewed endpoint column",
        "release_validation": release_receipt,
        "inputs": {
            "stage3_records": {"path": str(config.records), "sha256": file_sha256(config.records)},
            "canonical_records": {
                "path": str(config.canonical_records),
                "sha256": file_sha256(config.canonical_records),
            },
            "level_mapping": {"path": str(config.levels), "sha256": file_sha256(config.levels)},
            "retrieval_index": {
                "path": str(config.retrieval_index),
                "sha256": file_sha256(config.retrieval_index),
            },
        },
        "output": {"path": str(config.input), "sha256": file_sha256(config.input)},
        "counts": {"records": len(rows), "atoms": atoms, "initial_parents": parents},
    }


def _column_descriptions(config: Workflow) -> dict[str, str]:
    columns = {column for values in config.refinement_columns.values() for column in values}
    return {
        column: DESCRIPTIONS.get(
            column, column.removeprefix("canonical_").replace("_", " ")
        )
        for column in columns
    }


def configure_core(config: Workflow) -> dict[str, Any]:
    manifest = json.loads((config.input.parent / "manifest.json").read_text())
    levels = tuple(sorted(pd.read_parquet(config.input, columns=["level"]).level.unique()))
    sources = sorted(pd.read_parquet(config.input, columns=["source_id"]).source_id.unique())
    core.VERSION = f"{VERSION}.{config.task}"
    core.BATCH_SEED_VERSION = core.VERSION
    core.TASK_NAME = TASK_NAMES[config.task]
    core.MODEL = config.model
    core.BASE_URL = config.endpoints[0]["base_url"]
    core.ENDPOINTS = config.endpoints
    core.RECORD_MAP = config.input
    core.RECORDS = config.canonical_records
    core.PROMPT_ROOT = PROMPT_ROOT
    core.DEFAULT_REVIEW = config.review
    core.DEFAULT_OUTPUT = config.run
    core.LEVELS = levels
    core.EXPECTED_RECORD_COUNT = int(manifest["counts"]["records"])
    core.EXPECTED_ATOM_COUNT = int(manifest["counts"]["atoms"])
    core.PAIR_COLUMNS = {source: config.pair_columns[source] for source in sources}
    core.REFINEMENT_COLUMNS = {
        source: config.refinement_columns[source] for source in sources
    }
    _configure_semantic_policy(config, sources)
    return manifest


def _configure_semantic_policy(config: Workflow, sources: list[str]) -> None:
    core.PROMPT_DIMENSION_COLUMNS = core.REFINEMENT_COLUMNS
    core.INITIAL_COLUMNS = {
        source: (config.endpoint_columns[source],) for source in sources
    }
    core.SAMPLE_CARD_COLUMNS = core.REFINEMENT_COLUMNS
    core.COLUMN_DESCRIPTIONS = _column_descriptions(config)
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
    core.PROMPT_REVIEW_TITLE = f"{TASK_NAMES[config.task]} semantic-only prompt review"


def prepare(config: Workflow) -> dict[str, Any]:
    build_input(config)
    configure_core(config)
    return core.prepare_prompt_review(config.review)


def run(config: Workflow, approved_review_sha256: str) -> dict[str, Any]:
    configure_core(config)
    parallelism = int(config.endpoints[0]["max_inflight"])
    manifest = core.run_semantic(
        config.run,
        review_manifest_path=config.review / "manifest.json",
        approved_review_sha256=approved_review_sha256,
        parallelism=parallelism,
    )
    manifest.update(
        {
            "task": config.task,
            "evidence_release": config.release_root.name,
            "cross_source_merging": False,
            "readout_buckets_built": False,
            "publication_status": "candidate_unselected",
        }
    )
    write_json_atomic(config.run / "manifest.json", manifest)
    return manifest


def _validated_candidate(config: Workflow) -> tuple[Path, pd.DataFrame]:
    candidate_path = config.run / "source_semantic_bucket_map.parquet"
    candidate = pd.read_parquet(candidate_path)
    atoms = pd.read_parquet(config.run / "input_atoms.parquet")
    if candidate.atom_id.duplicated().any() or len(candidate) != len(atoms):
        raise ValueError("semantic candidate does not cover every atom exactly once")
    if set(candidate.atom_id) != set(atoms.atom_id):
        raise ValueError("semantic candidate atom universe differs from frozen input")
    identity = candidate[["atom_id", "level", "source_id"]].merge(
        atoms[["atom_id", "level", "source_id"]],
        on="atom_id",
        validate="one_to_one",
        suffixes=("_candidate", "_input"),
    )
    if (identity.level_candidate != identity.level_input).any() or (
        identity.source_id_candidate != identity.source_id_input
    ).any():
        raise ValueError("semantic candidate changed an atom's source or level")
    return candidate_path, candidate


def _record_map(config: Workflow, semantic: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_parquet(config.input)
    records["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in records[
            ["level", "source_id", "pair_bucket_key"]
        ].itertuples(index=False, name=None)
    ]
    columns = ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id"]
    mapped = records[columns].merge(
        semantic[["atom_id", "semantic_bucket_id"]],
        on="atom_id",
        validate="many_to_one",
    )
    if len(mapped) != len(records) or mapped.source_row_uid.duplicated().any():
        raise ValueError("semantic map lacks exact scoped-record coverage")
    return mapped.sort_values(["level", "source_row_uid"])


def _review(config: Workflow, review_path: Path, candidate_path: Path) -> dict[str, Any]:
    review = json.loads(review_path.read_text())
    expected = {"version", "reviewer", "candidate_map_sha256", "decision", "rationale", "audit"}
    if set(review) != expected or review["version"] != f"{VERSION}.agentic_review.v1":
        raise ValueError("invalid semantic review schema")
    if review["decision"] != "approve":
        raise ValueError("semantic review did not approve establishment")
    if review["candidate_map_sha256"] != file_sha256(candidate_path):
        raise ValueError("semantic review targets another candidate map")
    return review


def establish(config: Workflow, review_path: Path) -> dict[str, Any]:
    configure_core(config)
    manifest_path = config.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "awaiting_agentic_review":
        raise ValueError("semantic candidate is not awaiting agentic review")
    candidate_path, candidate = _validated_candidate(config)
    _review(config, review_path, candidate_path)
    semantic = candidate.copy()
    semantic["semantic_bucket_id"] = semantic.source_semantic_bucket_id
    columns = ["level", "source_id", "source_semantic_bucket_id", "semantic_bucket_id", "atom_id"]
    semantic = semantic[columns].sort_values(columns[:-1] + ["atom_id"])
    records = _record_map(config, semantic)
    outputs = _write_established(config, semantic, records, review_path)
    manifest.update(outputs)
    write_json_atomic(manifest_path, manifest)
    return manifest


def _write_established(
    config: Workflow,
    semantic: pd.DataFrame,
    records: pd.DataFrame,
    review_path: Path,
) -> dict[str, Any]:
    semantic_path = config.run / "semantic_bucket_map.parquet"
    record_path = config.run / "record_semantic_bucket_map.parquet"
    freeze_path = config.run / "semantic_bucket_map_manifest.json"
    if any(path.exists() for path in (semantic_path, record_path, freeze_path)):
        raise FileExistsError("established semantic outputs already exist")
    semantic.to_parquet(semantic_path, index=False)
    records.to_parquet(record_path, index=False)
    frozen = _semantic_manifest(config, semantic, records, review_path)
    write_json_atomic(freeze_path, frozen)
    return {
        "status": "complete_reviewed",
        "publication_status": "reviewed_unselected",
        "readout_buckets_built": False,
        "agentic_review": {"path": str(review_path), "sha256": file_sha256(review_path)},
        "semantic_bucket_map": {
            "path": semantic_path.name,
            "sha256": file_sha256(semantic_path),
            "rows": len(semantic),
            "semantic_bucket_count": semantic.semantic_bucket_id.nunique(),
        },
        "record_semantic_bucket_map": {
            "path": record_path.name,
            "sha256": file_sha256(record_path),
            "rows": len(records),
        },
        "semantic_bucket_map_manifest": {
            "path": freeze_path.name,
            "sha256": file_sha256(freeze_path),
        },
    }


def _semantic_manifest(
    config: Workflow,
    semantic: pd.DataFrame,
    records: pd.DataFrame,
    review_path: Path,
) -> dict[str, Any]:
    return {
        "version": f"{VERSION}.established.v1",
        "status": "complete_reviewed",
        "task": config.task,
        "evidence_release": config.release_root.name,
        "levels": sorted(semantic.level.unique()),
        "cross_source_merging": False,
        "readout_buckets_built": False,
        "semantic_bucket_id_policy": "source_semantic_bucket_id",
        "agentic_review": {"path": str(review_path), "sha256": file_sha256(review_path)},
        "source_semantic_bucket_map_sha256": file_sha256(
            config.run / "source_semantic_bucket_map.parquet"
        ),
        "semantic_bucket_map_sha256": file_sha256(config.run / "semantic_bucket_map.parquet"),
        "record_semantic_bucket_map_sha256": file_sha256(
            config.run / "record_semantic_bucket_map.parquet"
        ),
        "input_atoms_sha256": file_sha256(config.run / "input_atoms.parquet"),
        "atom_count": len(semantic),
        "record_count": len(records),
        "semantic_bucket_count": int(semantic.semantic_bucket_id.nunique()),
    }


def status(config: Workflow) -> dict[str, Any]:
    result: dict[str, Any] = {"run_root": str(config.run_root)}
    manifest_path = config.run / "manifest.json"
    if manifest_path.is_file():
        result["manifest"] = json.loads(manifest_path.read_text())
    database = config.run / "requests.sqlite3"
    if database.is_file():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT count(*), sum(status='complete'), sum(status='failed'), "
            "sum(status='pending') FROM requests"
        ).fetchone()
        connection.close()
        result["requests"] = dict(zip(("total", "complete", "failed", "pending"), row))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "establish", "status"))
    parser.add_argument("--task", required=True, choices=tuple(TASK_NAMES))
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--review", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    config = workflow(
        args.task,
        args.release_root,
        args.run_root,
        model=args.model,
        endpoint_urls=tuple(args.endpoints or DEFAULT_ENDPOINTS),
        max_inflight=args.max_inflight,
        retrieval_index=args.retrieval_index,
    )
    if args.command == "prepare":
        result = prepare(config)
    elif args.command == "run":
        if not args.approved_review_sha256:
            parser.error("run requires --approved-review-sha256")
        result = run(config, args.approved_review_sha256)
    elif args.command == "establish":
        if not args.review:
            parser.error("establish requires --review")
        result = establish(config, args.review)
    else:
        result = status(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
