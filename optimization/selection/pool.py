"""Pinned candidate pools, exact assay scores and ledger-bound record weights."""

from __future__ import annotations
import json
import argparse
import pandas as pd
import math
import hashlib
from pathlib import Path
from typing import Any, Mapping
from predict.utils.json import sha256_file
import csv
import sqlite3
import tempfile
import pyarrow.parquet as pq
import yaml
from optimization.selection.record_scores import load_scores as load_dili_scores
from predict.retrieval.assay_reranking.build_tdc_score_overlay import (
    BUNDLE,
    _level,
)
from data.processing.paths import REPO_ROOT
from predict.retrieval.assay_reranking.ranked_uid_retrieval import hydrate_uids
from predict.retrieval.assay_reranking.artifact_bundle import DEFAULT_CACHE_SET, CACHE_SET_SCHEMA, cache_task
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from predict.retrieval.selection import assert_disjoint, indexed_manifests
from optimization.semantic_buckets.inputs import (
    load_reviewed_record_weights,
    load_skin_candidate_snapshot,
    load_dili_candidate_snapshot,
    resolve_semantic_bucket_artifacts,
)
from predict.tasks.bbb_martins.prediction_filter import prediction_only


def coverage_band(scored: int, total: int) -> str:
    if total < 1 or not 0 <= scored <= total:
        raise ValueError("Invalid score coverage counts")
    return (
        "high"
        if 10 * scored >= 7 * total
        else "low"
        if 5 * scored <= 2 * total
        else "middle"
    )


def load_coverage(
    path: Path,
    *,
    task: str,
    queries: Mapping[str, Any] | list[str],
    direct_index: Path | None = None,
    later_index: Path | None = None,
    cache_bundle: Path | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("queries") or {}
    if (
        document.get("schema_version") != "gold_mixed_score_coverage.v1"
        or document.get("status") != "complete"
        or document.get("task_id") != task
        or (set(rows) != set(queries))
    ):
        raise ValueError("Gold score coverage queries or identity differ")
    for name, source in (
        ("direct_index", direct_index),
        ("later_index", later_index),
        ("cache_bundle", cache_bundle),
    ):
        if source is not None:
            entry = document.get("inputs", {}).get(name) or {}
            if entry.get("path") != str(source.resolve()) or entry.get(
                "sha256"
            ) != sha256_file(source):
                raise ValueError(f"Gold score coverage {name} changed")
    for query_id, row in rows.items():
        direct_total = int(row["direct_total"])
        direct_scored = int(row["direct_scored"])
        indirect_total = int(row["indirect_total"])
        indirect_scored = int(row["indirect_scored"])
        scored, total = (direct_scored + indirect_scored, direct_total + indirect_total)
        if (
            direct_total < 100
            or not 0 <= direct_scored <= direct_total
            or (not 0 <= indirect_scored <= indirect_total)
            or (row.get("scored_candidates") != scored)
            or (row.get("candidate_count") != total)
            or (row.get("coverage_band") != coverage_band(scored, total))
        ):
            raise ValueError(f"Gold score coverage counts differ: {query_id}")
    return document


SCHEMA = "tdc_mixed_selection"
DILI_SCHEMA = "tdc_dili_geomean_mixed_selection"
BBB_FILTERED_SCHEMA = "tdc_bbb_filtered_selection"
DIRECT_FIELDS = (
    "benchmark_row_id",
    "selection_rank",
    "context_id",
    "parent_id",
    "gold_label",
    "assay_transfer_score",
    "score_key",
)
INDIRECT_FIELDS = (
    "benchmark_row_id",
    "selection_rank",
    "level",
    "source_row_uid",
    "parent_id",
    "assay_transfer_score",
    "score_key",
    "semantic_bucket_id",
    "semantic_weight",
    "semantic_weight_source",
)
AVAILABLE_COVERAGE_FIELDS = (
    "benchmark_row_id",
    "scored_candidates",
    "candidate_count",
    "direct_scored",
    "indirect_scored",
)
FILTER_FIELDS = ("source_row_uid", "prediction_only")


def _scores(
    connection: sqlite3.Connection, query_id: str
) -> dict[tuple[str, str], tuple[str, float, str]]:
    return {
        (level, item): (parent, float(score), key)
        for level, item, parent, score, key in connection.execute(
            "SELECT level,item_id,parent_id,score,score_key FROM scores WHERE query_id=?",
            (query_id,),
        )
    }


def _semantic_ledger(
    task: str, extension: Path | None = None, snapshot: Path | None = None
) -> tuple[dict, dict[str, str]]:
    if snapshot is not None:
        if task not in {"skin_reaction", "dili"} or extension is not None:
            raise ValueError("Candidate semantic snapshot is unsupported")
        loader = (
            load_dili_candidate_snapshot
            if task == "dili"
            else load_skin_candidate_snapshot
        )
        return loader(snapshot)
    artifacts = resolve_semantic_bucket_artifacts(task, "CURRENT")
    selected = json.loads(artifacts.manifest.read_text())
    if (
        artifacts.central_ledger is None
        or selected.get("provisional_retrieval_weight_policy")
        != "mean_of_reviewed_semantic_buckets_in_task_level"
    ):
        raise ValueError(
            f"TDC-v2 optimization requires the selected central ledger: {task}"
        )
    required = (
        "central_ledger",
        "semantic_bucket_weights",
        "record_relevance_rankings",
        "weight_projection_manifest",
        "pending_uid_worklist",
    )
    pins = {}
    for key in required:
        entry = selected["files"].get(key) or {}
        path = artifacts.root / str(entry.get("path") or "")
        if not path.is_file() or not entry.get("sha256"):
            raise ValueError(f"Selected semantic ledger input changed: {task}/{key}")
        pins[str(path.resolve())] = entry["sha256"]
    projection_path = artifacts.record_relevance_rankings.parent / "manifest.json"
    if sha256_file(projection_path) != pins[str(projection_path.resolve())]:
        raise ValueError(f"Selected semantic projection manifest changed: {task}")
    projection = json.loads(projection_path.read_text())
    if (
        projection.get("schema_version") != "semantic_retrieval_mean_weights.v2"
        or projection.get("status") != "complete_reviewed_with_provisional_mean"
        or projection["inputs"]["ledger"]["sha256"]
        != pins[str(artifacts.central_ledger)]
        or (
            projection["inputs"]["weights"]["sha256"]
            != pins[str(artifacts.semantic_bucket_weights)]
        )
    ):
        raise ValueError(f"TDC-v2 semantic projection is not ledger-bound: {task}")
    semantics, audit = load_reviewed_record_weights(task, "CURRENT")
    pins[audit["release_manifest"]] = audit["release_manifest_sha256"]
    if extension is not None:
        extension = extension.resolve()
        extra = json.loads(extension.read_text())
        expected = extra.get("inputs") or {}
        table = extension.with_name(extra["projection"]["path"])
        if (
            extra.get("schema_version") != "semantic_retrieval_mean_weights.v2"
            or extra.get("status") != "candidate_unreviewed"
            or (expected["ledger"]["sha256"] != pins[str(artifacts.central_ledger)])
            or (
                expected["weights"]["sha256"]
                != pins[str(artifacts.semantic_bucket_weights)]
            )
            or (sha256_file(table) != extra["projection"]["sha256"])
            or (
                sha256_file(Path(expected["retrieval_uids"]["path"]))
                != expected["retrieval_uids"]["sha256"]
            )
        ):
            raise ValueError("Semantic extension is not pinned to the selected ledger")
        rows = pq.read_table(table).to_pylist()
        if len(rows) != extra["projection"]["rows"]:
            raise ValueError("Semantic extension row count differs")
        for row in rows:
            key = (str(row["source_row_uid"]), str(row["level"]))
            if key in semantics:
                raise ValueError(
                    f"Semantic extension overlaps selected projection: {key}"
                )
            semantics[key] = {
                "semantic_bucket_id": row["semantic_bucket_id"],
                "semantic_weight": float(row["weight"]),
                "semantic_weight_source": row["weight_source"],
            }
        pins[str(extension)] = sha256_file(extension)
        pins[str(table)] = extra["projection"]["sha256"]
        pins[expected["retrieval_uids"]["path"]] = expected["retrieval_uids"]["sha256"]
    return (semantics, pins)


def prepare_bbb_filter(
    subset: str, target: Path, *, cache_bundle: Path | None = None, benchmark: str = "tdc"
) -> Path:
    from predict.harnesses.branches.flat import _source_semantic_card

    "Classify the frozen BBB indirect universe once before profile optimization."
    if subset not in {"valid", "test"} or target.exists():
        raise ValueError("Expected a valid/test split and a new filter directory")
    cache_bundle = cache_bundle or DEFAULT_CACHE_SET
    bundle = yaml.safe_load(cache_bundle.read_text())
    if bundle.get("schema_version") == CACHE_SET_SCHEMA:
        spec = cache_task(cache_bundle, "bbb_martins", benchmark)
        index_path = (REPO_ROOT / spec["retrieval"]["later"]).resolve()
    elif bundle.get("benchmark") == "tdc" and "tasks" in bundle:
        later = bundle["tasks"]["bbb_martins"]["retrieval"]["later"]
        index_path = (REPO_ROOT / later).resolve()
    else:
        later = bundle["caches"]["bbb_martins"]["later"]
        index_path = (cache_bundle.parent / later).resolve()
    index = json.loads(index_path.read_text())
    evidence = index["evidence"]
    projection = (index_path.parent / evidence["manifest"]).resolve()
    if sha256_file(projection) != evidence["manifest_sha256"]:
        raise ValueError("BBB evidence projection changed")
    inputs = {
        str(cache_bundle.resolve()): sha256_file(cache_bundle),
        str(index_path): sha256_file(index_path),
        str(projection): evidence["manifest_sha256"],
    }
    uids = set()
    for level in index["splits"][subset]["levels"]:
        if level == "L1":
            continue
        database, _ = _level(index_path, subset, level, inputs)
        with sqlite3.connect(
            f"file:{database.resolve()}?mode=ro&immutable=1", uri=True
        ) as reader:
            uids.update(
                (
                    uid
                    for (uid,) in reader.execute(
                        "SELECT DISTINCT item_id FROM rankings"
                    )
                )
            )
    rows = hydrate_uids(projection, uids)
    decisions = []
    for uid in sorted(uids):
        card, _ = _source_semantic_card(
            json.loads(rows[uid]["payload"]), card_id=uid, transfer_likelihood=None
        )
        decisions.append(
            {
                "source_row_uid": uid,
                "prediction_only": int(prediction_only({"card": card})),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".bbb_prediction_filter.", dir=target.parent
    ) as scratch:
        stage = Path(scratch) / target.name
        stage.mkdir()
        pd.DataFrame(decisions, columns=FILTER_FIELDS).to_csv(
            stage / "eligibility.tsv", sep="\t", index=False
        )
        write_json_atomic(
            stage / "manifest.json",
            {
                "schema_version": "bbb_prediction_only_filter.v1",
                "status": "complete",
                "task_id": "bbb_martins",
                "subset": subset,
                "source_commit": "95cfb7c57efd810c5a5cb5143200055302b6bb0b",
                "predicate_sha256": sha256_file(
                    Path(prediction_only.__code__.co_filename)
                ),
                "inputs": inputs,
                "files": {
                    "eligibility.tsv": {
                        "sha256": sha256_file(stage / "eligibility.tsv"),
                        "rows": len(decisions),
                    }
                },
                "excluded_count": sum((row["prediction_only"] for row in decisions)),
            },
        )
        stage.rename(target)
    return target / "manifest.json"


def _bbb_filter(
    path: Path, subset: str, inputs: dict[str, str]
) -> tuple[set[str], set[str]]:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != "bbb_prediction_only_filter.v1"
        or manifest.get("status") != "complete"
        or manifest.get("subset") != subset
        or (
            manifest.get("predicate_sha256")
            != sha256_file(Path(prediction_only.__code__.co_filename))
        )
    ):
        raise ValueError("BBB prediction filter differs")
    for source, digest in manifest["inputs"].items():
        if inputs.get(source) != digest:
            raise ValueError(f"BBB filter candidate input differs: {source}")
    table = path.with_name("eligibility.tsv")
    if sha256_file(table) != manifest["files"]["eligibility.tsv"]["sha256"]:
        raise ValueError("BBB filter decisions changed")
    with table.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(FILTER_FIELDS):
            raise ValueError("BBB filter table schema differs")
        decisions = list(reader)
    if len(decisions) != manifest["files"]["eligibility.tsv"]["rows"]:
        raise ValueError("BBB filter table count differs")
    covered = {row["source_row_uid"] for row in decisions}
    excluded = {
        row["source_row_uid"] for row in decisions if row["prediction_only"] == "1"
    }
    if (
        len(covered) != len(decisions)
        or any((row["prediction_only"] not in {"0", "1"} for row in decisions))
        or len(excluded) != manifest["excluded_count"]
    ):
        raise ValueError("BBB filter decisions are invalid")
    return (covered, excluded)


CARD_IDENTITY = "canonical_context_molecule.v1"


def load_exact_scores(path: Path | None, *, task: str) -> tuple[dict, dict]:
    """Read frozen query/evidence scores; absence never becomes an inferred score."""
    if path is None:
        return ({}, {})
    path = path.resolve()
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("task", manifest.get("task_id")) != task
    ):
        raise ValueError("exact score snapshot task or completion differs")
    pins = {str(path): sha256_file(path)}
    for name, digest in manifest.get("inputs", {}).items():
        if isinstance(digest, dict):
            source = Path(digest["path"])
            expected = digest["sha256"]
        else:
            source = Path(name)
            expected = digest
        if sha256_file(source) != expected:
            raise ValueError("exact score snapshot source changed")
        pins[str(source.resolve())] = expected
    database = path.with_name(manifest["database"])
    if sha256_file(database) != manifest["database_sha256"]:
        raise ValueError("exact score snapshot database changed")
    pins[str(database)] = manifest["database_sha256"]
    with sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT query_id,level,item_id,parent_id,score,score_key FROM scores"
        ).fetchall()
    scores = {}
    for query, level, item, parent, score, key in rows:
        identity = (str(query), str(level), str(item))
        if (
            identity in scores
            or not key
            or (not math.isfinite(score))
            or (not 0 <= score <= 1)
        ):
            raise ValueError("exact scores repeat identities or contain invalid values")
        scores[identity] = (str(parent), float(score), str(key))
    if len(scores) != manifest.get("score_count", len(scores)):
        raise ValueError("exact score snapshot count differs")
    return (scores, dict(manifest=manifest, pins=pins))


def attach_exact_scores(
    rows: list[dict], *, query_id: str, level: str, scores: dict
) -> None:
    for row in rows:
        value = scores.get((query_id, level, str(row["item_id"])))
        if value is not None and value[0] != row["parent_id"]:
            raise ValueError("exact score parent identity differs")
        row["assay_transfer_score"] = value[1] if value is not None else None
        row["score_key"] = value[2] if value is not None else None


def load_direct(
    release_index: Path,
    *,
    task: str,
    subset: str,
    queries: dict,
    scores: dict | None = None,
) -> tuple[dict, dict]:
    index = json.loads(release_index.read_text())
    if index.get("context_identity") != CARD_IDENTITY:
        raise ValueError(
            "direct search requires a fresh canonical card cache; parent-grouped caches cannot be reused"
        )
    manifest_path = indexed_manifests(release_index, subset, ("L1",))["L1"]
    manifest = json.loads(manifest_path.read_text())
    database = manifest_path.with_name(manifest["database"])
    output = {}
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro&immutable=1", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        for query_id, drug in queries.items():
            registered = connection.execute(
                "SELECT drug FROM queries WHERE benchmark_row_id=?", (query_id,)
            ).fetchone()
            if registered is None or registered["drug"] != drug:
                raise ValueError("canonical card query identity differs")
            rows = connection.execute(
                "SELECT r.*,c.gold_label,c.canonical_smiles FROM rankings r JOIN contexts c ON c.context_id=r.morgan_context_id WHERE benchmark_row_id=? ORDER BY morgan_rank",
                (query_id,),
            ).fetchall()
            output[query_id] = []
            for row in rows:
                assert_disjoint(
                    drug, row["parent_smiles"], manifest["neighbor_identity_policy"]
                )
                if row["gold_label"] not in (0, 1):
                    raise ValueError(
                        "canonical direct card must have an authoritative binary label"
                    )
                output[query_id].append(
                    dict(
                        item_id=row["morgan_context_id"],
                        canonical_smiles=row["canonical_smiles"],
                        parent_id=row["parent_id"],
                        parent_smiles=row["parent_smiles"],
                        gold_label=row["gold_label"],
                        morgan_similarity=row["morgan_similarity"],
                    )
                )
            attach_exact_scores(
                output[query_id], query_id=query_id, level="L1", scores=scores or {}
            )
    return (
        output,
        dict(
            release_index=str(release_index.resolve()),
            release_index_sha256=sha256_file(release_index),
            l1_manifest=str(manifest_path),
            l1_manifest_sha256=sha256_file(manifest_path),
            database=str(database),
            database_sha256=manifest["database_sha256"],
        ),
    )


def load_indirect(
    release_index: Path,
    *,
    task: str,
    subset: str,
    queries: dict,
    scores: dict | None = None,
    semantic_extension: Path | None = None,
    semantic_snapshot: Path | None = None,
    dili_score_cache: Path | None = None,
    prediction_filter: Path | None = None,
    cache_bundle: Path | None = None,
    exclude_disjoint_violations: bool = False,
) -> tuple[dict, dict]:
    from predict.retrieval.assay_reranking.ranked_uid_retrieval import (
        load_ranked_universe,
    )

    semantics, semantic_pins = _semantic_ledger(
        task, semantic_extension, semantic_snapshot
    )
    index = json.loads(release_index.read_text())
    levels = tuple(index["splits"][subset]["levels"])
    levels = tuple((level for level in levels if level != "L1"))
    ranked, projection, audit = load_ranked_universe(
        release_index, task=task, subset=subset, levels=levels, queries=queries
    )
    manifests = indexed_manifests(release_index, subset, levels)
    tool_scores, tool_audit = (
        load_dili_scores(dili_score_cache) if dili_score_cache else ({}, {})
    )
    if dili_score_cache and (
        task != "dili"
        or tool_audit["release_index_sha256"] != sha256_file(release_index)
    ):
        raise ValueError("DILI scores are bound to a different candidate release")
    excluded = set()
    filter_pins = {}
    if prediction_filter:
        if task != "bbb_martins":
            raise ValueError("prediction-only evidence filtering is BBB-specific")
        if cache_bundle is None:
            raise ValueError("prediction filtering requires the candidate cache bundle")
        candidate_pins = {
            str(cache_bundle.resolve()): sha256_file(cache_bundle),
            str(release_index.resolve()): audit["release_index_sha256"],
            str(projection.resolve()): audit["evidence_manifest_sha256"],
        }
        for level in levels:
            _level(release_index.resolve(), subset, level, candidate_pins)
        covered, excluded = _bbb_filter(prediction_filter, subset, candidate_pins)
        filter_pins = {
            str(path.resolve()): sha256_file(path)
            for path in (
                prediction_filter, prediction_filter.with_name("eligibility.tsv")
            )
        }
        if any(
            (
                str(row["item_id"]) not in covered
                for by_query in ranked.values()
                for rows in by_query.values()
                for row in rows
            )
        ):
            raise ValueError("prediction filter omits candidate UIDs")
    from predict.retrieval.policies import decide_candidate, normalize_molecule_identity

    disjoint_exclusions = []
    output = {query: [] for query in queries}
    for level, by_query in ranked.items():
        manifest = json.loads(manifests[level].read_text())
        for query_id, rows in by_query.items():
            for row in rows:
                uid = str(row["item_id"])
                if exclude_disjoint_violations:
                    decision = decide_candidate(normalize_molecule_identity(queries[query_id]),
                        {"canonical_smiles": row["parent_smiles"]}, manifest.get("neighbor_identity_policy", "parent_disjoint"))
                    if decision.excluded:
                        disjoint_exclusions.append(dict(benchmark_row_id=query_id, level=level,
                                                       source_row_uid=uid, reason=decision.reason))
                        continue
                assert_disjoint(
                    queries[query_id],
                    row["parent_smiles"],
                    manifest.get("neighbor_identity_policy", "parent_disjoint"),
                )
                if uid in excluded:
                    continue
                semantic = semantics.get((uid, level))
                if semantic is None:
                    raise ValueError(
                        f"ledger projection omits candidate UID: {uid}/{level}"
                    )
                candidate = {**row, **semantic, "level": level}
                if dili_score_cache:
                    if uid not in tool_scores:
                        raise ValueError(f"DILI scores omit candidate UID: {uid}")
                    candidate["dili_tool_score"] = tool_scores[uid]
                output[query_id].append(candidate)
    for query_id, rows in output.items():
        if len({str(row["item_id"]) for row in rows}) != len(rows):
            raise ValueError("joint indirect pool repeats physical UIDs")
        if scores is not None:
            for level in levels:
                attach_exact_scores(
                    [row for row in rows if row["level"] == level],
                    query_id=query_id,
                    level=level,
                    scores=scores,
                )
    return (
        output,
        dict(
            later_release_index=str(release_index.resolve()),
            later_release_index_sha256=sha256_file(release_index),
            cache=audit,
            semantics=dict(pinned_inputs=semantic_pins),
            dili_tool_scores=tool_audit,
            prediction_filter=filter_pins,
            disjoint_candidate_filter=dict(enabled=exclude_disjoint_violations,
                                            exclusions=disjoint_exclusions),
        ),
    )


def prepare_l1_cache(
    task: str,
    train_path: Path,
    query_paths: dict[str, Path],
    output: Path,
    *,
    capacity: int = 100,
    identity_policy: str = "scaffold_disjoint",
    membership_path: Path | None = None,
    membership_projection: Path | None = None,
) -> Path:
    """Build a fresh full-molecule card cache; parents participate only in filtering."""
    import pandas as pd
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from predict.retrieval.policies import normalize_molecule_identity
    from predict.retrieval.assay_reranking.build_ranked_uid_retrieval import (
        _schema,
        _write_complete,
    )
    from predict.retrieval.assay_reranking.ranked_uid_retrieval import SCHEMA_VERSION

    if identity_policy != "scaffold_disjoint":
        raise ValueError("Direct L1 construction requires scaffold-disjoint candidates")
    if output.exists() or capacity < 1:
        raise ValueError("L1 cache requires a fresh root and positive card capacity")
    train = read_jsonl(train_path)
    if not train or len({row["benchmark_row_id"] for row in train}) != len(train):
        raise ValueError("canonical cards are empty or repeat card identities")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    prepared = []
    for card in train:
        full = card.get("molecule_identity", {}).get("canonical_smiles") or card.get(
            "drug"
        )
        if not full or card["Y"] not in (0, 1):
            raise ValueError(
                "canonical card lacks full-molecule identity or authoritative binary label"
            )
        molecule = Chem.MolFromSmiles(full)
        if molecule is None:
            raise ValueError("canonical card full molecule cannot be fingerprinted")
        identity = normalize_molecule_identity(full)
        prepared.append(
            (
                str(card["benchmark_row_id"]),
                full,
                identity,
                generator.GetFingerprint(molecule),
                card,
            )
        )
    memberships = {}
    if membership_path:
        for row in pq.read_table(
            membership_path,
            columns=["benchmark_row_id", "source_row_uid", "vote_label"],
        ).to_pylist():
            memberships.setdefault(str(row["benchmark_row_id"]), []).append(row)
    if membership_projection:
        if membership_path:
            raise ValueError("declare one physical membership source")
        for record in pq.read_table(membership_projection).to_pylist():
            payload = json.loads(record["payload"])
            memberships.setdefault(str(payload["benchmark_row_id"]), []).append(
                dict(
                    source_row_uid=record["source_row_uid"],
                    vote_label=payload["Y"],
                    payload=payload,
                    external_record_id=record["external_record_id"],
                )
            )
    sources = {
        str(train_path.resolve()): sha256_file(train_path),
        **{str(path.resolve()): sha256_file(path) for path in query_paths.values()},
    }
    if membership_path:
        sources[str(membership_path.resolve())] = sha256_file(membership_path)
    if membership_projection:
        sources[str(membership_projection.resolve())] = sha256_file(
            membership_projection
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".canonical_l1.", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        projection = root / "evidence"
        projection.mkdir()
        records = {}
        contexts = []
        for card_id, full, identity, fingerprint, card in prepared:
            parent = identity.parent_inchi_key or identity.parent_smiles
            members = memberships.get(card_id)
            if members is None and (membership_path or membership_projection):
                raise ValueError(f"physical membership omits canonical card: {card_id}")
            if members is None:
                members = [
                    dict(source_row_uid=str(uid), vote_label=card["Y"])
                    for uid in card["source_record_ids"]
                ]
            if not members:
                raise ValueError("canonical card has no physical members")
            uids = []
            for member in members:
                uid = str(member["source_row_uid"])
                uids.append(uid)
                if uid in records:
                    raise ValueError(
                        "physical L1 voter belongs to multiple canonical cards"
                    )
                payload = dict(
                    label_source="canonical_context_cards.v1",
                    family_key="canonical_training_labels",
                    source_id="canonical_train",
                    measurement_kind="categorical_label",
                    progressive_level="L1",
                    canonical_smiles=full,
                    source_canonical_smiles=full,
                    source_row_uid=uid,
                    record_id=uid,
                    benchmark_row_id=card_id,
                    condition_group=card.get("condition_group", ""),
                    Y=int(member["vote_label"]),
                    source_fields=dict(label=int(member["vote_label"])),
                    source_contract=dict(
                        contract_version="canonical_context_cards.v1",
                        source_or_simply_cleaned={"label": True},
                    ),
                )
                if member.get("payload"):
                    payload = {**member["payload"], "canonical_smiles": full}
                records[uid] = dict(
                    source_row_uid=uid,
                    external_record_id=member.get("external_record_id", uid),
                    parent_id=parent,
                    parent_smiles=identity.parent_smiles,
                    level="L1",
                    payload=json.dumps(payload, sort_keys=True),
                )
            contexts.append((card_id, full, parent, identity.parent_smiles, card, uids))
        pd.DataFrame(records.values()).to_parquet(
            projection / "records.parquet", index=False
        )
        evidence = dict(
            schema_version="ranked_evidence_projection.v1",
            status="complete",
            task_id=task,
            context_identity=CARD_IDENTITY,
            records="records.parquet",
            records_sha256=sha256_file(projection / "records.parquet"),
            record_count=len(records),
            inputs=sources,
        )
        evidence["content_id"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode()
        ).hexdigest()
        write_json_atomic(projection / "VERSION.json", evidence)
        splits = {}
        for subset, path in query_paths.items():
            target = root / "scaffold" / subset / "L1"
            target.mkdir(parents=True)
            database = target / "rankings.sqlite3"
            connection = sqlite3.connect(database)
            _schema(connection, "L1")
            connection.execute(
                'ALTER TABLE contexts ADD COLUMN canonical_smiles TEXT NOT NULL DEFAULT ""'
            )
            connection.execute(
                'ALTER TABLE rankings ADD COLUMN canonical_smiles TEXT NOT NULL DEFAULT ""'
            )
            for card_id, full, parent, parent_smiles, card, uids in contexts:
                connection.execute(
                    "INSERT INTO contexts VALUES (?,?,?,?,?)",
                    (
                        card_id,
                        parent,
                        card.get("condition_group", ""),
                        int(card["Y"]),
                        full,
                    ),
                )
                connection.executemany(
                    "INSERT INTO context_records VALUES (?,?,?)",
                    [(card_id, uid, rank) for rank, uid in enumerate(uids, 1)],
                )
            counts = {}
            for query in read_jsonl(path):
                query_id = str(query["benchmark_row_id"])
                drug = str(query["drug"])
                full = query.get("molecule_identity", {}).get(
                    "canonical_smiles"
                ) or query.get("drug")
                if not full:
                    raise ValueError("query lacks canonical full-molecule identity")
                identity = normalize_molecule_identity(full)
                query_mol = Chem.MolFromSmiles(full)
                if query_mol is None:
                    raise ValueError("query full molecule cannot be fingerprinted")
                connection.execute(
                    "INSERT INTO queries VALUES (?,?,?,?)",
                    (
                        query_id,
                        drug,
                        identity.parent_inchi_key or identity.parent_smiles,
                        identity.parent_smiles,
                    ),
                )
                similarities = DataStructs.BulkTanimotoSimilarity(
                    generator.GetFingerprint(query_mol),
                    [entry[3] for entry in prepared],
                )
                ordered = sorted(
                    zip(prepared, similarities), key=lambda pair: (-pair[1], pair[0][0])
                )
                chosen = []
                for item, similarity in ordered:
                    card_id, card_full, card_identity, _, card = item
                    try:
                        assert_disjoint(
                            full, card_identity.parent_smiles, identity_policy
                        )
                    except ValueError:
                        continue
                    chosen.append((item, float(similarity)))
                    if len(chosen) == capacity:
                        break
                if len(chosen) != capacity:
                    raise ValueError(
                        f"{query_id} has fewer than {capacity} eligible canonical cards"
                    )
                member_count = 0
                context_lookup = {row[0]: row[-1] for row in contexts}
                for rank, (item, similarity) in enumerate(chosen, 1):
                    card_id, card_full, card_identity, _, card = item
                    uids = context_lookup[card_id]
                    member_count += len(uids)
                    parent = (
                        card_identity.parent_inchi_key or card_identity.parent_smiles
                    )
                    connection.execute(
                        "INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            query_id,
                            card_id,
                            parent,
                            card_identity.parent_smiles,
                            similarity,
                            rank,
                            1,
                            rank,
                            None,
                            None,
                            card_id,
                            None,
                            len(uids),
                            None,
                            None,
                            card_full,
                        ),
                    )
                counts[query_id] = dict(
                    candidate_parents=len(
                        {item[2].parent_smiles for item, _ in chosen}
                    ),
                    candidate_cards=capacity,
                    morgan_candidate_records=member_count,
                    assay_candidate_records=0,
                )
            identity = dict(
                task_id=task,
                subset=subset,
                level="L1",
                pool="fixed",
                capacity=capacity,
                parent_capacity=capacity,
                context_identity=CARD_IDENTITY,
                neighbor_identity_policy=identity_policy,
                parent_identity_usage="filtering_only",
                query_count=len(counts),
                stored_rows=len(counts) * capacity,
                query_counts=counts,
                inputs=sources,
                model=None,
                scoring_contract_version="exact_query_card.v1",
            )
            manifest = _write_complete(
                connection,
                database,
                task=task,
                subset=subset,
                level="L1",
                identity=identity,
                target=target,
            )
            connection.close()
            splits[subset] = dict(
                levels={
                    "L1": dict(
                        manifest=str((target / "VERSION.json").relative_to(root)),
                        manifest_sha256=sha256_file(target / "VERSION.json"),
                        content_id=manifest["content_id"],
                    )
                }
            )
        index = dict(
            schema_version="ranked_uid_task_release_index.v1",
            selection_contract=SCHEMA_VERSION,
            status="complete",
            task_id=task,
            profile="canonical_context_molecule_l1_v1",
            context_identity=CARD_IDENTITY,
            parent_identity_usage="filtering_only",
            parent_capacity=capacity,
            card_capacity=capacity,
            levels_independent=True,
            ranking_modes=["morgan"],
            assay_transfer_status="not_computed",
            morgan_fingerprint=dict(
                radius=2, bits=2048, include_chirality=True, similarity="Tanimoto"
            ),
            neighbor_identity_policy_by_level={"L1": identity_policy},
            evidence=dict(
                manifest="evidence/VERSION.json",
                manifest_sha256=sha256_file(projection / "VERSION.json"),
                content_id=evidence["content_id"],
                record_count=evidence["record_count"],
            ),
            splits=splits,
        )
        write_json_atomic(root / "RELEASE_INDEX.json", index)
        root.rename(output)
    return output / "RELEASE_INDEX.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a fresh canonical L1 context-card cache without model calls."
    )
    parser.add_argument("action", choices=("prepare-l1",))
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument(
        "--query-path", action="append", required=True, metavar="SUBSET=PATH"
    )
    parser.add_argument("--membership-path", type=Path)
    parser.add_argument("--membership-projection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument(
        "--identity-policy",
        choices=("scaffold_disjoint",),
        default="scaffold_disjoint",
    )
    args = parser.parse_args()
    queries = {}
    for binding in args.query_path:
        subset, separator, path = binding.partition("=")
        if not separator or subset in queries or subset not in {"valid", "test"}:
            parser.error("query paths require unique valid/test=PATH bindings")
        queries[subset] = Path(path)
    print(
        prepare_l1_cache(
            args.task,
            args.train_path,
            queries,
            args.output,
            capacity=args.capacity,
            identity_policy=args.identity_policy,
            membership_path=args.membership_path,
            membership_projection=args.membership_projection,
        )
    )
