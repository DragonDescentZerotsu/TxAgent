"""Catalog, prompt, and SQLite contracts for cached assay-transfer reranking."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ASSAY_TRANSFER_MODEL = "jiosephlee/assay-transfer-tool"
ASSAY_TRANSFER_MODEL_REVISION = "e7b694d1a3d52f5e50bc55ab73f3a07542ff69eb"
SCORING_CONTRACT_VERSION = "assay_transfer_chat_first_divergent_token_logits.v1"
CACHE_SCHEMA_VERSION = "assay_transfer_rerank_flat_cache.v2"
CATALOG_SCHEMA_VERSION = "txagent_assay_transfer_catalog.v1"
DEFAULT_CATALOG = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "assay_transfer_rerank/flat_v2/catalog.jsonl"
)
DEFAULT_CACHE = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "assay_transfer_rerank/flat_v2/scores.sqlite3"
)
TEMPLATE_DIR = Path(__file__).with_name("assay_transfer_templates")
TEMPLATE_BY_CONCEPT = {
    "oral_bioavailability": "oral_bioavailability_intern_mcqa_v3.jinja",
    "oral_exposure": "oral_exposure_intern_mcqa_v3.jinja",
    "Fa": "Fa_intern_mcqa_v3.jinja",
    "Fg": "Fg_intern_mcqa_v3.jinja",
    "Fh": "Fh_intern_mcqa_v3.jinja",
}
CONCEPT_BY_GROUP = {
    "Observed.direct_oral_bioavailability": "oral_bioavailability",
    "Observed.oral_auc_cmax_exposure": "oral_exposure",
    "Fa.absorption_solubility_permeability": "Fa",
    "Fg.gut_wall_efflux_intestinal_metabolism": "Fg",
    "Fh.hepatic_clearance_metabolic_stability": "Fh",
}


class AssayTransferCacheMiss(RuntimeError):
    """Raised when strict cached retrieval cannot score every candidate record."""


@dataclass(frozen=True)
class PromptTask:
    cache_key: str
    prompt_hash: str
    prompt: str
    query_smiles: str
    group_id: str
    molecule_id: str
    record_id: str
    model: str
    model_revision: str
    scoring_contract_version: str
    template_hash: str
    catalog_version: str


@dataclass(frozen=True)
class PromptScore:
    cache_key: str
    logp_transfer: float
    logp_not_transfer: float
    transfer_probability: float


def template_bundle_hash(template_dir: Path = TEMPLATE_DIR) -> str:
    digest = hashlib.sha256()
    for concept, filename in sorted(TEMPLATE_BY_CONCEPT.items()):
        path = template_dir / filename
        digest.update(concept.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class AssayTransferCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata, self.records = load_catalog(self.path)
        self.catalog_version = str(self.metadata["catalog_version"])
        self.template_hash = str(self.metadata["template_hash"])
        current_template_hash = template_bundle_hash()
        if self.template_hash != current_template_hash:
            raise ValueError(
                "Assay-transfer catalog template hash does not match vendored templates: "
                f"catalog={self.template_hash}, current={current_template_hash}"
            )
        self._by_concept_smiles: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._by_record_id: dict[str, dict[str, Any]] = {}
        for record in self.records:
            record_id = str(record["record_id"])
            if record_id in self._by_record_id:
                raise ValueError(f"Duplicate assay-transfer catalog record ID: {record_id}")
            self._by_record_id[record_id] = record
            concept = str(record["assay_concept"])
            expected_template_id = Path(TEMPLATE_BY_CONCEPT[concept]).stem
            if record.get("template_id") != expected_template_id:
                raise ValueError(
                    f"Catalog record {record.get('record_id')} has template_id={record.get('template_id')!r}; "
                    f"expected {expected_template_id!r}"
                )
            key = (str(record["assay_concept"]), str(record["canonical_smiles"]))
            self._by_concept_smiles.setdefault(key, []).append(record)
        for records in self._by_concept_smiles.values():
            records.sort(key=lambda row: str(row["record_id"]))

    def compatible_records(self, group_id: str, canonical_smiles: str) -> list[dict[str, Any]]:
        concept = CONCEPT_BY_GROUP.get(group_id)
        if not concept:
            raise ValueError(f"Assay-transfer reranking is not enabled for evidence family: {group_id}")
        return list(self._by_concept_smiles.get((concept, canonical_smiles), ()))

    def records_by_id(self, record_ids: Iterable[str]) -> list[dict[str, Any]]:
        output = []
        for record_id in record_ids:
            try:
                output.append(self._by_record_id[str(record_id)])
            except KeyError as error:
                raise ValueError(f"Candidate manifest references unknown record ID: {record_id}") from error
        return output


class AssayTransferCandidateManifest:
    """Exact condition/query/candidate-to-record joins for a flat catalog."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or rows[0].get("record_type") != "manifest_metadata":
            raise ValueError(f"Invalid assay-transfer candidate manifest: {self.path}")
        self.metadata = rows[0]
        self.condition_id = str(self.metadata.get("condition_id") or "")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._record_ids: dict[tuple[str, str, str], tuple[str, ...]] = {}
        for row in rows[1:]:
            if row.get("record_type") != "candidate_group":
                raise ValueError(f"Invalid candidate-group row in manifest: {self.path}")
            query_smiles = str(row["query_smiles"])
            group_id = str(row["group_id"])
            for candidate in row.get("candidates") or []:
                key = (query_smiles, group_id, str(candidate["molecule_id"]))
                record_ids = tuple(sorted(set(str(value) for value in candidate["record_ids"])))
                previous = self._record_ids.setdefault(key, record_ids)
                if previous != record_ids:
                    raise ValueError(f"Conflicting duplicate candidate join in manifest: {key}")

    def record_ids(self, query_smiles: str, group_id: str, molecule_id: str) -> tuple[str, ...]:
        key = (query_smiles, group_id, molecule_id)
        if key not in self._record_ids:
            raise ValueError(
                "Candidate is not present in the frozen assay-transfer condition manifest: "
                f"condition={self.condition_id}, group={group_id}, molecule={molecule_id}"
            )
        return self._record_ids[key]


def load_catalog(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Assay-transfer catalog does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        header_line = next((line for line in handle if line.strip()), "")
        metadata = json.loads(header_line) if header_line else {}
        records = [json.loads(line) for line in handle if line.strip()]
    if metadata.get("record_type") != "catalog_metadata":
        raise ValueError(f"Invalid assay-transfer catalog header: {path}")
    if metadata.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported assay-transfer catalog schema: {metadata.get('schema_version')}")
    if any(row.get("record_type") != "assay_record" for row in records):
        raise ValueError(f"Invalid assay-transfer catalog record: {path}")
    return metadata, records


class AssayTransferPromptRenderer:
    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.template_dir = template_dir
        self.template_hash = template_bundle_hash(template_dir)
        self.environment = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(self, record: dict[str, Any], query_smiles: str) -> str:
        concept = str(record["assay_concept"])
        filename = TEMPLATE_BY_CONCEPT.get(concept)
        if not filename:
            raise ValueError(f"Unknown assay-transfer concept: {concept}")
        context = {key: _display(value) for key, value in (record.get("template_context") or {}).items()}
        values = {
            **context,
            "metric_type": str(record["metric_type"]),
            "threshold_display": str(record["threshold_display"]),
            "endpoint_subtype": str(record["endpoint_subtype"]),
            "unit_basis": str(record["unit_basis"]),
            "retrieved_original_smiles": str(record["original_smiles"]),
            "query_original_smiles": query_smiles,
            "value_display": str(record["value_display"]),
        }
        for field in _TEMPLATE_CONTEXT_FIELDS:
            values.setdefault(field, "not specified")
        return self.environment.get_template(filename).render(**values)


class AssayTransferScoreCache:
    """SQLite cache with writes deliberately available only through explicit calls."""

    def __init__(self, path: str | Path, *, mode: str):
        if mode not in {"read_only", "read_write"}:
            raise ValueError(f"Unsupported assay-transfer cache mode: {mode}")
        self.path = Path(path)
        self.mode = mode
        if mode == "read_only":
            if not self.path.exists():
                raise FileNotFoundError(f"Assay-transfer score cache does not exist: {self.path}")
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

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prompt_scores (
                cache_key TEXT PRIMARY KEY,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                scoring_contract_version TEXT NOT NULL,
                template_hash TEXT NOT NULL,
                logp_transfer REAL NOT NULL,
                logp_not_transfer REAL NOT NULL,
                transfer_probability REAL NOT NULL,
                score_origin TEXT NOT NULL DEFAULT 'inference',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(
                    prompt_hash, model, model_revision,
                    scoring_contract_version, template_hash
                )
            );
            CREATE INDEX IF NOT EXISTS prompt_scores_prompt_hash ON prompt_scores(prompt_hash);
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
        except sqlite3.OperationalError as error:
            raise ValueError(f"Invalid assay-transfer cache schema: {self.path}") from error
        if row is None or row[0] != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported assay-transfer cache schema: {self.path}")

    def lookup(self, tasks: Iterable[PromptTask]) -> dict[str, PromptScore]:
        task_list = list(tasks)
        if not task_list:
            return {}
        output: dict[str, PromptScore] = {}
        for offset in range(0, len(task_list), 500):
            keys = [task.cache_key for task in task_list[offset : offset + 500]]
            placeholders = ",".join("?" for _ in keys)
            for row in self.connection.execute(
                f"SELECT cache_key, logp_transfer, logp_not_transfer, transfer_probability "
                f"FROM prompt_scores WHERE cache_key IN ({placeholders})",
                keys,
            ):
                output[row["cache_key"]] = PromptScore(
                    cache_key=row["cache_key"],
                    logp_transfer=float(row["logp_transfer"]),
                    logp_not_transfer=float(row["logp_not_transfer"]),
                    transfer_probability=float(row["transfer_probability"]),
                )
        return output

    def write_batch(
        self,
        tasks: Iterable[PromptTask],
        scores: Iterable[PromptScore],
        *,
        score_origin: str = "inference",
    ) -> int:
        if self.mode != "read_write":
            raise PermissionError("Assay-transfer cache is read-only")
        task_list = list(tasks)
        task_by_key = {task.cache_key: task for task in task_list}
        if len(task_by_key) != len(task_list):
            raise ValueError("Assay-transfer write batch contains duplicate task keys")
        rows = []
        for score in scores:
            if score.cache_key not in task_by_key:
                raise ValueError(f"Assay-transfer score has no matching task: {score.cache_key}")
            task = task_by_key[score.cache_key]
            rows.append(
                (
                    task.cache_key,
                    task.prompt_hash,
                    task.model,
                    task.model_revision,
                    task.scoring_contract_version,
                    task.template_hash,
                    score.logp_transfer,
                    score.logp_not_transfer,
                    score.transfer_probability,
                    score_origin,
                )
            )
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO prompt_scores(
                cache_key, prompt_hash, model, model_revision,
                scoring_contract_version, template_hash,
                logp_transfer, logp_not_transfer, transfer_probability, score_origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
        # Verify durably-stored payloads with set-based lookups rather than one
        # SELECT per row: at multi-million-row scale the per-row round trip on the
        # single writer connection starves the inference workers.
        observed_by_key: dict[str, tuple[Any, ...]] = {}
        keys = [row[0] for row in rows]
        for offset in range(0, len(keys), 500):
            chunk = keys[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            for observed in self.connection.execute(
                f"""
                SELECT cache_key, prompt_hash, model, model_revision, scoring_contract_version,
                       template_hash, logp_transfer, logp_not_transfer, transfer_probability
                FROM prompt_scores WHERE cache_key IN ({placeholders})
                """,
                chunk,
            ):
                observed_by_key[observed[0]] = tuple(observed[1:])
        for row in rows:
            if observed_by_key.get(row[0]) != row[1:-1]:
                raise ValueError(
                    "Conflicting assay-transfer payload for append-only score key "
                    f"{row[0]}"
                )
        # Callers only submit cache misses, so a successful batch normally inserts
        # every row. Returning the number of requested durable rows also makes an
        # idempotent retry harmless after a parent-process interruption.
        return len(rows)


class AssayTransferCachedReranker:
    name = "assay_transfer"

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        cache_path: str | Path,
        cache_mode: str = "read_only",
        model: str = ASSAY_TRANSFER_MODEL,
        model_revision: str = ASSAY_TRANSFER_MODEL_REVISION,
        allow_missing: bool = False,
        candidate_manifest_path: str | Path | None = None,
    ):
        self.model = model
        self.model_revision = require_immutable_revision(model_revision)
        self.catalog = AssayTransferCatalog(catalog_path)
        self.candidate_manifest = (
            AssayTransferCandidateManifest(candidate_manifest_path)
            if candidate_manifest_path
            else None
        )
        if (
            str(self.catalog.metadata.get("source_mode") or "").startswith("flat_")
            and self.candidate_manifest is None
        ):
            raise ValueError(
                "Flat assay-transfer catalogs require an exact --rerank-candidate-manifest"
            )
        if (
            self.candidate_manifest is not None
            and self.candidate_manifest.metadata.get("catalog_version") != self.catalog.catalog_version
        ):
            raise ValueError("Assay-transfer manifest and catalog versions do not match")
        self.cache = AssayTransferScoreCache(cache_path, mode=cache_mode)
        self.renderer = AssayTransferPromptRenderer()
        self.allow_missing = allow_missing
        self.missing_tasks: dict[str, PromptTask] = {}
        self.seen_tasks: dict[str, PromptTask] = {}

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "model_revision": self.model_revision,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "template_hash": self.renderer.template_hash,
            "catalog_version": self.catalog.catalog_version,
            "condition_id": (
                self.candidate_manifest.condition_id if self.candidate_manifest is not None else ""
            ),
            "candidate_manifest_sha256": (
                self.candidate_manifest.sha256 if self.candidate_manifest is not None else ""
            ),
        }

    def tasks_for_candidates(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> dict[str, list[PromptTask]]:
        tasks_by_molecule: dict[str, list[PromptTask]] = {}
        for candidate in candidates:
            molecule_id = str(candidate["molecule_chembl_id"])
            canonical_smiles = str(candidate["canonical_smiles"])
            if self.candidate_manifest is None:
                records = self.catalog.compatible_records(group_id, canonical_smiles)
            else:
                record_ids = self.candidate_manifest.record_ids(
                    query_smiles, group_id, molecule_id
                )
                records = self.catalog.records_by_id(record_ids)
                wrong_smiles = [
                    str(record["record_id"])
                    for record in records
                    if str(record["canonical_smiles"]) != canonical_smiles
                ]
                if wrong_smiles:
                    raise ValueError(
                        f"Candidate manifest record/molecule mismatch for {molecule_id}: {wrong_smiles}"
                    )
            tasks_by_molecule[molecule_id] = [
                build_prompt_task(
                    renderer=self.renderer,
                    record=record,
                    query_smiles=query_smiles,
                    group_id=group_id,
                    molecule_id=molecule_id,
                    model=self.model,
                    model_revision=self.model_revision,
                    catalog_version=self.catalog.catalog_version,
                )
                for record in records
            ]
        return tasks_by_molecule

    def rerank(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tasks_by_molecule = self.tasks_for_candidates(
            query_smiles=query_smiles, group_id=group_id, candidates=candidates
        )
        all_tasks = [task for tasks in tasks_by_molecule.values() for task in tasks]
        self.seen_tasks.update((task.cache_key, task) for task in all_tasks)
        cached = self.cache.lookup(all_tasks)
        missing = [task for task in all_tasks if task.cache_key not in cached]
        self.missing_tasks.update((task.cache_key, task) for task in missing)
        if missing and not self.allow_missing:
            raise AssayTransferCacheMiss(
                f"Assay-transfer cache is missing {len(missing)} of {len(all_tasks)} prompt scores "
                f"for family {group_id}; run precompute_assay_transfer_rerank first"
            )

        output = []
        for candidate in candidates:
            molecule_id = str(candidate["molecule_chembl_id"])
            available = [
                (cached[task.cache_key], task)
                for task in tasks_by_molecule[molecule_id]
                if task.cache_key in cached
            ]
            if available:
                best_score, best_task = sorted(
                    available,
                    key=lambda item: (-item[0].transfer_probability, item[1].record_id),
                )[0]
                transfer_probability = best_score.transfer_probability
                winning_record_id = best_task.record_id
            else:
                transfer_probability = -1.0
                winning_record_id = ""
            output.append(
                {
                    **candidate,
                    "transfer_selection_score": transfer_probability,
                    "transfer_winning_record_id": winning_record_id,
                    "transfer_scored_record_count": len(available),
                }
            )
        output.sort(
            key=lambda row: (
                -float(row["transfer_selection_score"]),
                -float(row.get("similarity") or 0.0),
                str(row.get("molecule_chembl_id") or ""),
            )
        )
        for rank, row in enumerate(output, start=1):
            row["transfer_selection_rank"] = rank
        return output


def build_prompt_task(
    *,
    renderer: AssayTransferPromptRenderer,
    record: dict[str, Any],
    query_smiles: str,
    group_id: str,
    molecule_id: str,
    model: str,
    model_revision: str,
    catalog_version: str,
) -> PromptTask:
    prompt = renderer.render(record, query_smiles)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_key = flat_score_key(
        prompt_hash=prompt_hash,
        model=model,
        model_revision=model_revision,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
    )
    return PromptTask(
        cache_key=cache_key,
        prompt_hash=prompt_hash,
        prompt=prompt,
        query_smiles=query_smiles,
        group_id=group_id,
        molecule_id=molecule_id,
        record_id=str(record["record_id"]),
        model=model,
        model_revision=model_revision,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
        catalog_version=catalog_version,
    )


def flat_score_key(
    *,
    prompt_hash: str,
    model: str,
    model_revision: str,
    scoring_contract_version: str,
    template_hash: str,
) -> str:
    """Return the retrieval-agnostic identity of one model scoring prompt."""
    key_payload = {
        "prompt_hash": prompt_hash,
        "model": model,
        "model_revision": model_revision,
        "scoring_contract_version": scoring_contract_version,
        "template_hash": template_hash,
    }
    return hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def probability_from_log_likelihoods(logp_transfer: float, logp_not_transfer: float) -> float:
    maximum = max(logp_transfer, logp_not_transfer)
    a = math.exp(logp_transfer - maximum)
    b = math.exp(logp_not_transfer - maximum)
    return a / (a + b)


def require_immutable_revision(revision: str) -> str:
    revision = str(revision or "").strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(f"Assay-transfer model revision must be an immutable 40-character SHA: {revision!r}")
    return revision


def prompt_task_to_dict(task: PromptTask) -> dict[str, Any]:
    return asdict(task)


def prompt_task_from_dict(payload: dict[str, Any]) -> PromptTask:
    return PromptTask(**payload)


def _display(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "not specified"


_TEMPLATE_CONTEXT_FIELDS = (
    "species_or_population",
    "report_or_statistic_type",
    "dose",
    "study_or_assay_system",
    "measured_process",
    "biological_context",
    "medium",
    "formulation_or_solid_form",
    "transporter_or_enzyme",
    "substrate_status",
    "intestinal_site",
    "molecular_form",
    "enzyme_or_pathway",
    "qualifying_conditions",
    "comparator",
    "extra_details",
)
