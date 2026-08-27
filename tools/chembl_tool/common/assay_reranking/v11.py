"""V11 prompt, catalog, and append-only cache contracts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.starling.normalization.measurements import (
    canonicalize_endpoint,
)


ASSET_ROOT = Path(__file__).with_name("assets") / "v11"
PROJECTION_PATH = ASSET_ROOT / "prompt_projection.json"
MODEL_REGISTRY_PATH = ASSET_ROOT / "models.json"
SOURCE_MANIFEST_PATH = ASSET_ROOT / "SOURCE.json"
PROFILE_NAME = "v11_with_categorical"
TEMPLATE_PROFILE = "v11_query_context_copy"
QUERY_CONTEXT_POLICY = "copy_retrieval_assay_context_value_hidden.v11"
BACKBONE_DTYPE = "bfloat16"
LOGIT_EXTRACTION_DTYPE = "float32"
SCORING_CONTRACT_VERSION = (
    "assay_transfer_chat_first_divergent_token_bf16_backbone_fp32_head.v2"
)
CATALOG_SCHEMA_VERSION = "txagent_assay_transfer_catalog.v11"
CANDIDATE_SCHEMA_VERSION = "txagent_assay_transfer_candidates.v11"
CACHE_SCHEMA_VERSION = "txagent_assay_transfer_cache.v11"
COMPACT_CACHE_SCHEMA_VERSION = "txagent_assay_transfer_compact_cache.v1"
DEMAND_SCHEMA_VERSION = "txagent_assay_transfer_prompt_demand.v11"


class AssayTransferCacheMiss(RuntimeError):
    """A frozen reasoning run requested a score absent from its cache."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_immutable_revision(revision: str) -> str:
    value = str(revision or "").strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Model revision must be an immutable 40-character SHA: {revision!r}")
    return value


def model_profile(task_id: str) -> dict[str, Any]:
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    try:
        profile = dict(registry["tasks"][task_id])
    except KeyError as exc:
        raise ValueError(f"Unknown v11 assay-transfer task: {task_id}") from exc
    if not profile.get("model") or not profile.get("revision"):
        raise ValueError(f"V11 assay-transfer model is not configured for {task_id}")
    profile["revision"] = require_immutable_revision(str(profile["revision"]))
    return profile


def default_cache_paths(
    task_id: str, *, split: str = "scaffold", subset: str = "valid"
) -> dict[str, str]:
    root = (
        Path("outputs/chembl_tool/tasks")
        / task_id
        / "evidence_library/assay_transfer_rerank"
        / PROFILE_NAME
        / split
        / subset
    )
    return {
        "catalog": str(root / "catalog.jsonl"),
        "candidate_manifest": str(root / "candidate_manifest.jsonl"),
        "cache": str(root / "scores.sqlite3"),
        "version": str(root / "VERSION.json"),
    }


def load_projection() -> dict[str, Any]:
    projection = json.loads(PROJECTION_PATH.read_text(encoding="utf-8"))
    if projection.get("schema_version") != "assay_transfer_prompt_projection.v11":
        raise ValueError("Unexpected v11 prompt-projection schema")
    if projection.get("record_contract_version") != "starling_record_contract.v7":
        raise ValueError("V11 prompts require starling_record_contract.v7")
    return projection


def verify_vendored_assets() -> dict[str, str]:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    observed = {name: file_sha256(ASSET_ROOT / name) for name in manifest["files"]}
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in manifest["files"].items()
        if observed[name] != expected
    }
    if mismatches:
        raise ValueError(f"Vendored v11 asset hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return observed


def template_bundle_hash(task_id: str) -> str:
    projection = load_projection()
    task = projection["tasks"][task_id]
    names = ["_macros.jinja", *(source["template"] for source in task["sources"].values())]
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ASSET_ROOT / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class V11PromptRenderer:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.projection = load_projection()
        if task_id not in self.projection["tasks"]:
            raise ValueError(f"V11 projection has no task {task_id}")
        self.template_hash = template_bundle_hash(task_id)
        self.projection_hash = file_sha256(PROJECTION_PATH)
        self.environment = Environment(
            loader=FileSystemLoader(str(ASSET_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        if str(record.get("task_id") or "") != self.task_id:
            raise ValueError("V11 prompt record belongs to another task")
        source_id = str(record.get("source_id") or "")
        try:
            config = self.projection["tasks"][self.task_id]["sources"][source_id]
        except KeyError as exc:
            raise ValueError(f"Unknown v11 source {self.task_id}/{source_id}") from exc
        query = {field: _clean(record.get(field)) for field in config["both"]}
        query["smiles"] = query_smiles
        retrieval_fields = [*config["both"], *config.get("retrieval_only", [])]
        retrieval = {field: _clean(record.get(field)) for field in retrieval_fields}
        query.update(_clean(config.get("constants", {})))
        retrieval.update(_clean(config.get("constants", {})))
        retrieval.update(_clean(config.get("retrieval_only_constants", {})))
        return self.environment.get_template(config["template"]).render(
            query=query, retrieval=retrieval
        ).strip()

    def canonical_endpoint_key(self, record: Mapping[str, Any]) -> str:
        """Resolve the exact endpoint used by this record's scoring prompt."""
        source_id = str(record.get("source_id") or "")
        try:
            config = self.projection["tasks"][self.task_id]["sources"][source_id]
        except KeyError as exc:
            raise ValueError(f"Unknown v11 source {self.task_id}/{source_id}") from exc
        constants = config.get("constants") or {}
        endpoint = constants.get("endpoint_name", record.get("endpoint_name"))
        if endpoint in (None, ""):
            # Some accepted BBB literature claims state only the mechanism
            # family (for example direct BBB evidence) and have no finer
            # endpoint name. Treat those claims as one deterministic unknown
            # endpoint within that source family so multi-record expansion
            # remains endpoint-distinct without inventing assay semantics.
            return f"unspecified_endpoint:{source_id}"
        return canonicalize_endpoint(endpoint)


@dataclass(frozen=True)
class PromptTask:
    cache_key: str
    prompt_hash: str
    prompt: str
    task_id: str
    query_smiles: str
    group_id: str
    molecule_id: str
    record_id: str
    model: str
    model_revision: str
    scoring_contract_version: str
    template_hash: str
    projection_hash: str


@dataclass(frozen=True)
class PromptScore:
    cache_key: str
    logp_transfer: float
    logp_not_transfer: float
    transfer_probability: float


def prompt_task_to_dict(task: PromptTask) -> dict[str, Any]:
    return asdict(task)


def prompt_task_from_dict(payload: Mapping[str, Any]) -> PromptTask:
    return PromptTask(**dict(payload))


def build_prompt_task(
    renderer: V11PromptRenderer,
    record: Mapping[str, Any],
    *,
    query_smiles: str,
    group_id: str,
    molecule_id: str,
    model: str,
    model_revision: str,
) -> PromptTask:
    prompt = renderer.render(record, query_smiles)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    key_payload = {
        "prompt_hash": prompt_hash,
        "model": model,
        "model_revision": require_immutable_revision(model_revision),
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "template_hash": renderer.template_hash,
        "projection_hash": renderer.projection_hash,
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PromptTask(
        cache_key=cache_key,
        prompt_hash=prompt_hash,
        prompt=prompt,
        task_id=renderer.task_id,
        query_smiles=query_smiles,
        group_id=group_id,
        molecule_id=molecule_id,
        record_id=str(record["record_id"]),
        model=model,
        model_revision=require_immutable_revision(model_revision),
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
        projection_hash=renderer.projection_hash,
    )


class V11ScoreCache:
    def __init__(self, path: str | Path, *, mode: str):
        if mode not in {"read_only", "read_write"}:
            raise ValueError(f"Unsupported cache mode: {mode}")
        self.path = Path(path)
        self.mode = mode
        if mode == "read_only":
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self.connection = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=60.0
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=60.0)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self._create_schema()
        self.connection.row_factory = sqlite3.Row
        self._validate_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prompt_scores (
                cache_key TEXT PRIMARY KEY,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                scoring_contract_version TEXT NOT NULL,
                template_hash TEXT NOT NULL,
                projection_hash TEXT NOT NULL,
                logp_transfer REAL NOT NULL,
                logp_not_transfer REAL NOT NULL,
                transfer_probability REAL NOT NULL,
                score_origin TEXT NOT NULL DEFAULT 'inference',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS prompt_scores_prompt_hash
                ON prompt_scores(prompt_hash);
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES ('schema_version', ?)",
            (CACHE_SCHEMA_VERSION,),
        )
        self.connection.commit()

    def _validate_schema(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT value FROM cache_metadata WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"Invalid v11 score cache: {self.path}") from exc
        if row is None or row[0] != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported v11 score cache schema: {self.path}")

    def close(self) -> None:
        self.connection.close()

    def lookup(self, tasks: Iterable[PromptTask]) -> dict[str, PromptScore]:
        task_list = list(tasks)
        output: dict[str, PromptScore] = {}
        for offset in range(0, len(task_list), 500):
            keys = [task.cache_key for task in task_list[offset : offset + 500]]
            if not keys:
                continue
            placeholders = ",".join("?" for _ in keys)
            query = (
                "SELECT cache_key, logp_transfer, logp_not_transfer, transfer_probability "
                f"FROM prompt_scores WHERE cache_key IN ({placeholders})"
            )
            for row in self.connection.execute(query, keys):
                output[row["cache_key"]] = PromptScore(
                    cache_key=row["cache_key"],
                    logp_transfer=float(row["logp_transfer"]),
                    logp_not_transfer=float(row["logp_not_transfer"]),
                    transfer_probability=float(row["transfer_probability"]),
                )
        return output

    def write_batch(
        self, tasks: Iterable[PromptTask], scores: Iterable[PromptScore]
    ) -> int:
        if self.mode != "read_write":
            raise PermissionError("V11 score cache is read-only")
        task_by_key = {task.cache_key: task for task in tasks}
        rows = []
        for score in scores:
            task = task_by_key.get(score.cache_key)
            if task is None:
                raise ValueError(f"Score has no matching task: {score.cache_key}")
            rows.append((
                task.cache_key, task.prompt_hash, task.model, task.model_revision,
                task.scoring_contract_version, task.template_hash, task.projection_hash,
                score.logp_transfer, score.logp_not_transfer,
                score.transfer_probability, "inference",
            ))
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO prompt_scores(
                cache_key, prompt_hash, model, model_revision,
                scoring_contract_version, template_hash, projection_hash,
                logp_transfer, logp_not_transfer, transfer_probability, score_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)


class V11Catalog:
    def __init__(self, path: str | Path, *, task_id: str):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), "")
            self.metadata = json.loads(first) if first else {}
            self.records = [json.loads(line) for line in handle if line.strip()]
        if self.metadata.get("record_type") != "catalog_metadata":
            raise ValueError(f"Invalid v11 catalog: {self.path}")
        if self.metadata.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise ValueError("Unsupported v11 catalog schema")
        if self.metadata.get("task_id") != task_id:
            raise ValueError("V11 catalog task mismatch")
        self.by_id = {str(row["record_id"]): row for row in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("V11 catalog repeats a record ID")

    def records_by_id(self, record_ids: Iterable[str]) -> list[dict[str, Any]]:
        output = []
        for record_id in record_ids:
            if str(record_id) not in self.by_id:
                raise ValueError(f"Candidate manifest references missing record {record_id}")
            output.append(self.by_id[str(record_id)])
        return output


class V11CandidateManifest:
    def __init__(self, path: str | Path, *, task_id: str):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), "")
            self.metadata = json.loads(first) if first else {}
            rows = [json.loads(line) for line in handle if line.strip()]
        if self.metadata.get("record_type") != "manifest_metadata":
            raise ValueError(f"Invalid v11 candidate manifest: {self.path}")
        if self.metadata.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("Unsupported v11 candidate-manifest schema")
        if self.metadata.get("task_id") != task_id:
            raise ValueError("V11 candidate-manifest task mismatch")
        self.sha256 = file_sha256(self.path)
        self._records: dict[tuple[str, str, str], tuple[str, ...]] = {}
        for row in rows:
            query_smiles = str(row["query_smiles"])
            group_id = str(row["group_id"])
            for candidate in row.get("candidates") or []:
                key = (query_smiles, group_id, str(candidate["molecule_id"]))
                values = tuple(str(value) for value in candidate.get("record_ids") or [])
                if key in self._records and self._records[key] != values:
                    raise ValueError(f"Conflicting v11 candidate join: {key}")
                self._records[key] = values

    def record_ids(self, query_smiles: str, group_id: str, molecule_id: str) -> tuple[str, ...]:
        key = (query_smiles, group_id, molecule_id)
        if key not in self._records:
            raise ValueError(f"Candidate absent from frozen v11 manifest: {key}")
        return self._records[key]


class V11CompactScoreCache:
    """Read-only finalized cache keyed directly by retrieval identities."""

    def __init__(self, path: str | Path, *, task_id: str):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(
            f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=60.0
        )
        self.connection.row_factory = sqlite3.Row
        metadata = {
            str(row["key"]): json.loads(str(row["value"]))
            for row in self.connection.execute("SELECT key, value FROM cache_metadata")
        }
        if metadata.get("schema_version") != COMPACT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported compact v11 cache schema: {self.path}")
        if metadata.get("task_id") != task_id:
            raise ValueError("Compact v11 cache task mismatch")
        if metadata.get("status") != "complete":
            raise ValueError("Compact v11 cache is not finalized")
        self.metadata = metadata

    def close(self) -> None:
        self.connection.close()

    def records_for_candidates(
        self, query_smiles: str, group_id: str, molecule_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        ids = list(dict.fromkeys(str(value) for value in molecule_ids))
        output = {value: [] for value in ids}
        for offset in range(0, len(ids), 400):
            chunk = ids[offset : offset + 400]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            sql = f"""
                SELECT m.molecule_chembl_id, r.external_record_id, r.payload,
                       s.transfer_probability
                FROM assignments a
                JOIN queries q USING(query_id)
                JOIN groups_dim g USING(group_key)
                JOIN molecules m USING(molecule_key)
                JOIN records r USING(record_key)
                JOIN scores s USING(score_key)
                WHERE q.query_smiles = ? AND g.group_id = ?
                  AND m.molecule_chembl_id IN ({placeholders})
                ORDER BY m.molecule_chembl_id, r.external_record_id
            """
            for row in self.connection.execute(sql, [query_smiles, group_id, *chunk]):
                payload = json.loads(str(row["payload"]))
                output[str(row["molecule_chembl_id"])].append(
                    {
                        "record_id": str(row["external_record_id"]),
                        "transfer_probability": float(row["transfer_probability"]),
                        "payload": payload,
                    }
                )
        return output


class V11CachedAssayReranker:
    name = "assay_transfer"

    def __init__(
        self,
        *,
        task_id: str,
        catalog_path: str | Path | None = None,
        candidate_manifest_path: str | Path | None = None,
        cache_path: str | Path,
        cache_mode: str = "read_only",
        model: str | None = None,
        model_revision: str | None = None,
        allow_missing: bool = False,
    ):
        profile = model_profile(task_id)
        self.task_id = task_id
        self.model = str(model or profile["model"])
        self.model_revision = require_immutable_revision(str(model_revision or profile["revision"]))
        self.renderer = V11PromptRenderer(task_id)
        self.compact_cache: V11CompactScoreCache | None = None
        self.catalog: V11Catalog | None = None
        self.candidate_manifest: V11CandidateManifest | None = None
        if catalog_path and candidate_manifest_path:
            self.catalog = V11Catalog(catalog_path, task_id=task_id)
            self.candidate_manifest = V11CandidateManifest(candidate_manifest_path, task_id=task_id)
            self.cache: V11ScoreCache | V11CompactScoreCache = V11ScoreCache(
                cache_path, mode=cache_mode
            )
            observed_metadata = self.catalog.metadata
        else:
            if cache_mode != "read_only":
                raise ValueError("Compact v11 caches are runtime read-only")
            self.compact_cache = V11CompactScoreCache(cache_path, task_id=task_id)
            self.cache = self.compact_cache
            observed_metadata = self.compact_cache.metadata
        self.allow_missing = allow_missing
        expected = {
            "model": self.model,
            "model_revision": self.model_revision,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": self.renderer.template_hash,
            "projection_hash": self.renderer.projection_hash,
        }
        mismatches = {
            key: {"expected": value, "observed": observed_metadata.get(key)}
            for key, value in expected.items()
            if observed_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"V11 cache provenance mismatch: {json.dumps(mismatches, sort_keys=True)}")

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": PROFILE_NAME,
            "task_id": self.task_id,
            "model": self.model,
            "model_revision": self.model_revision,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_profile": TEMPLATE_PROFILE,
            "template_hash": self.renderer.template_hash,
            "projection_hash": self.renderer.projection_hash,
            "query_context_policy": QUERY_CONTEXT_POLICY,
            "candidate_contract": (
                self.compact_cache.metadata.get("candidate_contract")
                if self.compact_cache else "tanimoto_raw_pool_then_identity_exclusion.v1"
            ),
        }

    def _tasks(
        self, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> dict[str, list[PromptTask]]:
        if self.catalog is None or self.candidate_manifest is None:
            raise RuntimeError("Prompt tasks are unavailable for compact finalized caches")
        output: dict[str, list[PromptTask]] = {}
        for candidate in candidates:
            molecule_id = str(candidate["molecule_chembl_id"])
            record_ids = self.candidate_manifest.record_ids(query_smiles, group_id, molecule_id)
            records = self.catalog.records_by_id(record_ids)
            output[molecule_id] = [
                build_prompt_task(
                    self.renderer,
                    record,
                    query_smiles=query_smiles,
                    group_id=group_id,
                    molecule_id=molecule_id,
                    model=self.model,
                    model_revision=self.model_revision,
                )
                for record in records
            ]
        return output

    def rerank_records(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.compact_cache is not None:
            return self._rerank_compact(
                query_smiles=query_smiles, group_id=group_id, candidates=candidates
            )
        tasks = self._tasks(query_smiles, group_id, candidates)
        flat = [task for values in tasks.values() for task in values]
        cached = self.cache.lookup(flat)
        missing = [task for task in flat if task.cache_key not in cached]
        if missing and not self.allow_missing:
            raise AssayTransferCacheMiss(
                f"V11 cache is missing {len(missing)} of {len(flat)} scores for {group_id}"
            )
        by_molecule = {str(row["molecule_chembl_id"]): row for row in candidates}
        output = []
        for molecule_id, values in tasks.items():
            candidate = by_molecule[molecule_id]
            for task in values:
                score = cached.get(task.cache_key)
                if score is None:
                    continue
                assert self.catalog is not None
                record = self.catalog.by_id[task.record_id]
                canonical_endpoint_key = self.renderer.canonical_endpoint_key(record)
                winning_record = _with_resolved_measurement_display(
                    candidate,
                    task.record_id,
                    {
                        "record_id": task.record_id,
                        "canonical_endpoint_key": canonical_endpoint_key,
                        "canonical_smiles": record.get("canonical_smiles"),
                        "measurement_kind": record.get("measurement_kind"),
                        "training_measurement_kind_supported": record.get(
                            "training_measurement_kind_supported"
                        ),
                        "source_contract": record.get("source_contract"),
                        "source_fields": record.get("source_fields"),
                    },
                )
                output.append({
                    **candidate,
                    "transfer_selection_score": score.transfer_probability,
                    "transfer_winning_record_id": task.record_id,
                    "transfer_winning_record": winning_record,
                    "transfer_scored_record_count": 1,
                })
        output.sort(key=lambda row: (
            -float(row["transfer_selection_score"]),
            -float(row.get("similarity") or 0.0),
            str(row.get("molecule_chembl_id") or ""),
            str(row.get("transfer_winning_record_id") or ""),
        ))
        for rank, row in enumerate(output, start=1):
            row["transfer_selection_rank"] = rank
        return output

    def _rerank_compact(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        assert self.compact_cache is not None
        by_molecule = {str(row["molecule_chembl_id"]): row for row in candidates}
        found = self.compact_cache.records_for_candidates(
            query_smiles, group_id, by_molecule
        )
        output = []
        missing = []
        for molecule_id, candidate in by_molecule.items():
            rows = found.get(molecule_id) or []
            if not rows:
                missing.append(molecule_id)
                continue
            for scored in rows:
                record = _with_resolved_measurement_display(
                    candidate,
                    str(scored["record_id"]),
                    scored["payload"],
                )
                output.append(
                    {
                        **candidate,
                        "transfer_selection_score": scored["transfer_probability"],
                        "transfer_winning_record_id": scored["record_id"],
                        "transfer_winning_record": record,
                        "transfer_scored_record_count": 1,
                    }
                )
        if missing and not self.allow_missing:
            raise AssayTransferCacheMiss(
                f"Compact v11 cache is missing {len(missing)} of "
                f"{len(by_molecule)} candidates for {group_id}"
            )
        output.sort(
            key=lambda row: (
                -float(row["transfer_selection_score"]),
                -float(row.get("similarity") or 0.0),
                str(row.get("molecule_chembl_id") or ""),
                str(row.get("transfer_winning_record_id") or ""),
            )
        )
        for rank, row in enumerate(output, start=1):
            row["transfer_selection_rank"] = rank
        return output

    def rerank(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.rerank_records(
            query_smiles=query_smiles, group_id=group_id, candidates=candidates
        )


def _with_resolved_measurement_display(
    candidate: Mapping[str, Any],
    record_id: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the hydrated display pair without changing a frozen score cache."""
    output = dict(record)
    if output.get("resolved_measurement_display"):
        return output
    for evidence in candidate.get("evidence_rows") or []:
        record_ids = list(evidence.get("_representative_record_ids") or [])
        if record_id not in record_ids:
            continue
        examples = list((evidence.get("minimal_evidence") or {}).get("examples") or [])
        position = record_ids.index(record_id)
        if position < len(examples) and examples[position].get(
            "resolved_measurement_display"
        ):
            output["resolved_measurement_display"] = examples[position][
                "resolved_measurement_display"
            ]
        break
    return output


def probability_from_logits(logit_a: float, logit_b: float) -> float:
    maximum = max(logit_a, logit_b)
    a = math.exp(logit_a - maximum)
    b = math.exp(logit_b - maximum)
    return a / (a + b)


def preflight_cache_coverage(
    *,
    task_id: str,
    catalog_path: str = "",
    candidate_manifest_path: str = "",
    cache_path: str,
    model: str,
    model_revision: str,
    expected_score_count: int = 0,
    cache_version_path: str = "",
    indices: list[int] | None = None,
    **_: Any,
) -> dict[str, Any]:
    reranker = V11CachedAssayReranker(
        task_id=task_id,
        catalog_path=catalog_path,
        candidate_manifest_path=candidate_manifest_path,
        cache_path=cache_path,
        cache_mode="read_only",
        model=model,
        model_revision=model_revision,
    )
    try:
        quick_check = str(reranker.cache.connection.execute("PRAGMA quick_check").fetchone()[0])
        score_table = "scores" if reranker.compact_cache is not None else "prompt_scores"
        count = int(reranker.cache.connection.execute(f"SELECT COUNT(*) FROM {score_table}").fetchone()[0])
        if quick_check != "ok":
            raise ValueError(f"V11 cache quick_check failed: {quick_check}")
        if expected_score_count and count != expected_score_count:
            raise ValueError(f"V11 score count mismatch: expected={expected_score_count}, observed={count}")
        version = {}
        if cache_version_path:
            version = json.loads(Path(cache_version_path).read_text(encoding="utf-8"))
            if version.get("status") != "complete":
                raise ValueError("V11 cache VERSION status is not complete")
            if version.get("scoring_contract_version") != SCORING_CONTRACT_VERSION:
                raise ValueError("V11 cache VERSION scoring contract mismatch")
            if version.get("backbone_dtype") != BACKBONE_DTYPE:
                raise ValueError("V11 cache VERSION backbone dtype mismatch")
            if version.get("logit_extraction_dtype") != LOGIT_EXTRACTION_DTYPE:
                raise ValueError("V11 cache VERSION logit extraction dtype mismatch")
            if int(version.get("n_prompt_scores") or 0) != count:
                raise ValueError("V11 cache VERSION score count mismatch")
        return {
            "status": "complete",
            "n_queries": len(indices or []),
            "cache_quick_check": quick_check,
            "n_cache_rows": count,
            "provenance": reranker.provenance(),
            "cache_version_validation": (
                {"status": "pass", "path": cache_version_path}
                if cache_version_path else {"status": "not_requested"}
            ),
        }
    finally:
        reranker.cache.close()
