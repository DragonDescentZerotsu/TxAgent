"""Offline Morgan-100 pool membership and shared scores for pinned BBB/oral tools.

Legacy caches are read-only reuse sources. Pool membership is separate from
prompt identity: one model/prompt score can serve several pools and records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pyarrow.parquet as pq
from rdkit import DataStructs

from . import runtime, v9, v24_1_levels as bbb, v25_oral_levels as oral
from predict.retrieval.policies import (
    IDENTITY_NORMALIZER_VERSION,
    decide_candidate,
    normalize_molecule_identity,
    standardize_smiles_and_fp,
)

MODULES = {bbb.TASK_ID: bbb, oral.TASK_ID: oral}
POOLS = ("tool-accepted", "tool-compatible", "all")
POOL_SIZE = 100
SCHEMA = "assay_transfer_three_pools.v1"
ROOT = runtime.cache_profile_root("recent_models_three_pools_morgan100_v1")
TRAINING_ROOT = Path(__file__).resolve().parents[3].parent / "starling_assay_transfer"
ARCHIVE_ROOT = Path(__file__).resolve().parents[3] / "data/legacy/artifacts/evidence_libraries/v10_before_level_mapping_compatibility_20260908"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "manifest.json"
MAPPING_ARCHIVE_MANIFEST = ARCHIVE_ROOT / "source_uid_levels/manifest.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_source(item):
    """Resolve a pinned historical input without weakening its content check."""
    path = Path(item["path"]).resolve()
    if path.is_file() and runtime.file_sha256(path) == item["sha256"]:
        return path
    for manifest in (ARCHIVE_MANIFEST, MAPPING_ARCHIVE_MANIFEST):
        receipt = json.loads(manifest.read_text())["files"].get(str(path), {})
        archived = Path(receipt.get("path", ""))
        if (receipt.get("sha256") == item["sha256"] and archived.is_file()
                and runtime.file_sha256(archived) == item["sha256"]):
            return archived
    raise ValueError(f"pinned source changed without a verified archive: {path}")


def paths(task, subset, root=None, gold_release="v1"):
    if task not in MODULES or subset not in {"valid", "test"}:
        raise ValueError("unsupported task or split")
    root = Path(root) if root else ROOT / task / "scaffold" / subset
    task_name = "BBB_Martins" if task == "bbb_martins" else "Bioavailability_Ma"
    gold_root = (MODULES[task].GOLD_ROOT if gold_release == "v1" else
                 Path("data/gold_labels") / task_name / gold_release / "scaffold")
    return {"root": root, "cache": root / "scores.sqlite3", "version": root / "VERSION.json",
            "journals": root / ".scores", "queries": gold_root
            / f"{subset}_molecule_condition_labels.jsonl"}


def membership(record):
    return tuple(p for p in POOLS if p == "all" or
                 (p == "tool-accepted" and record["tool_accepted"]) or
                 (p == "tool-compatible" and record["has_scalar"]))


class Renderer:
    """Keep legacy scalar rendering; add a source-text/no-unit branch only."""
    def __init__(self, task):
        self.module = MODULES[task]
        self.legacy = bbb.V241PromptRenderer() if task == bbb.TASK_ID else oral.V25OralPromptRenderer()
        self.template_hash = self.legacy.template_hash
        self.text_projection_hash = digest({"base": self.legacy.projection_hash,
                                           "non_scalar": "source_measurement_without_unit.v1"})

    def render(self, record, query):
        if record["has_scalar"]:
            return self.legacy.render(record, query), self.legacy.projection_hash
        m = self.module
        payload = {k: v for k, v in record["source_fields"].items() if k not in m.HIDDEN and k != "unit_text"}
        if m is bbb:
            payload = {**{k: v for k, v in payload.items() if k not in m.RESULT_TAIL},
                       **{k: payload[k] for k in m.RESULT_TAIL if k in payload}}
        known_smiles = str(payload.get("source_smiles") or payload.get("smiles") or record["canonical_smiles"])
        prompt = self.legacy.environment.get_template("prompt.jinja").render(
            known_smiles=known_smiles, query_smiles=query,
            known_fields=m._fields(payload, known=True),
            query_fields=m._fields({k: v for k, v in payload.items() if k != "molecule_name"}, known=False),
        ).strip()
        return prompt, self.text_projection_hash

    def prompt_task(self, record, query):
        prompt, projection = self.render(record, query)
        spec = self.module.MODELS[record["progressive_level"]]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        identity = dict(prompt_hash=prompt_hash, model=spec["model"], model_revision=spec["revision"],
                        template_hash=self.template_hash, projection_hash=projection,
                        scoring_contract_version=runtime.SCORING_CONTRACT_VERSION)
        return runtime.PromptTask(cache_key=digest(identity), prompt=prompt, task_id=self.module.TASK_ID,
                                  query_smiles=query, group_id=record["progressive_level"],
                                  molecule_id=record.get("parent_id", ""), record_id=record["record_id"], **identity)


def load_source(task, *, current_release=False):
    from huggingface_hub import snapshot_download

    m = MODULES[task]
    accepted, files = {}, {}
    for level, profile in m.MODELS.items():
        root = Path(snapshot_download(profile["dataset"], repo_type="dataset",
                                      revision=profile["dataset_revision"], local_files_only=True))
        release = json.loads((root / "manifest.json").read_text())
        for name in ("records", "source_contract", "uid_levels", "gold_validation", "gold_test"):
            verified_source(release["source_snapshot"][name])
        buckets = json.loads((root / "calibration.json").read_text())["accepted_buckets"]
        accepted[level] = {key for key, value in buckets.items() if value.get("level") == level}
        if not accepted[level] or (m is oral and
                {buckets[key].get("measurement_kind") for key in accepted[level]} != m.MEASUREMENT_KINDS[level]):
            raise ValueError(f"unexpected released buckets: {task}/{level}")
        files[f"{level}_dataset_calibration"] = root / "calibration.json"
        files[f"{level}_dataset_manifest"] = root / "manifest.json"
    source_version = ROOT / task / "scaffold/valid/VERSION.json"
    if current_release:
        source_version = Path()
    if source_version.is_file():
        source_cache = json.loads(source_version.read_text())
        if source_cache.get("status") != "complete" or source_cache.get("task_id") != task:
            raise ValueError("invalid three-pool source cache")
        records_path = verified_source(source_cache["inputs"]["records"])
        mapping_path = verified_source(source_cache["inputs"]["mapping"])
        source_contract = verified_source(source_cache["inputs"]["source_contract"])
        files["candidate_source_cache"] = source_version
    else:
        records_path, mapping_path, source_contract = m.STAGE3, m.LEVEL_MAPPING, m.SOURCE_CONTRACT
    contract = json.loads(source_contract.read_text())
    expected_contract = "source_column_contract.v1" if m is bbb else "source_column_contract.v2"
    if contract.get("contract_version") != expected_contract:
        raise ValueError(f"unexpected source contract: {task}")
    fields, source_union = {}, set()
    for source, spec in contract["sources"].items():
        fields[source] = [name for name, value in spec["normalized_artifact_columns"].items()
                          if value.get("source_or_simply_cleaned") is True]
        source_union.update(fields[source])
    mapping = {}
    for r in pq.read_table(mapping_path, columns=["source_row_uid", "level", "family_key"]).to_pylist():
        uid = r["source_row_uid"]
        if uid in mapping:
            raise ValueError(f"duplicate mapping UID: {uid}")
        mapping[uid] = (f"L{int(r['level'])}", r["family_key"])
    if source_version.is_file():
        expected_mapping_hash = source_cache["inputs"]["mapping"]["sha256"]
    else:
        expected_mapping_hash = m.LEVEL_MAPPING_RECEIPT["sha256"]
    if runtime.file_sha256(mapping_path) != expected_mapping_hash:
        raise ValueError("UID mapping hash mismatch")
    core = {"source_row_uid", "canonical_record_id", "canonical_smiles", "source_id", "pair_bucket_key",
            "assay_transfer_eligible", "measurement_kind", "finite_scalar_value", "canonical_pair_fields_json",
            "canonical_measurement_scale_id", "canonical_category_id", "canonical_measurement_text",
            "canonical_unit_text", "canonical_transporter_identifier"}
    columns = sorted((core | source_union) & set(pq.read_schema(records_path).names))
    records, identities, issues = {}, {}, []
    for batch in pq.ParquetFile(records_path).iter_batches(batch_size=10000, columns=columns):
        for raw in batch.to_pylist():
            mapped = mapping.get(raw["source_row_uid"])
            if mapped is None:
                issues.append((raw["canonical_record_id"], "unmapped"))
                continue
            level, family = mapped
            if level not in m.MODELS:
                continue
            smiles = str(raw["canonical_smiles"] or "")
            if smiles not in identities:
                identities[smiles] = normalize_molecule_identity(smiles)
            identity = identities[smiles]
            if identity.status != "ok" or not identity.parent_smiles:
                issues.append((raw["canonical_record_id"], "invalid_parent"))
                continue
            key = str(raw["canonical_record_id"])
            if key in records:
                raise ValueError(f"duplicate record: {key}")
            bucket = json.dumps([*json.loads(raw["pair_bucket_key"]), level], separators=(",", ":"))
            scalar = raw["finite_scalar_value"]
            records[key] = {**{k: raw.get(k) for k in core}, "record_id": key, "task_id": task,
                "progressive_level": level, "level_family": family,
                "canonical_smiles": identity.parent_smiles, "parent_id": identity.parent_inchi_key or identity.parent_smiles,
                "identity_smiles_forms": sorted({smiles, identity.parent_smiles}),
                "source_fields": {f: raw.get(f) for f in fields[raw["source_id"]]},
                "tool_accepted": bucket in accepted[level],
                "has_scalar": scalar is not None and math.isfinite(scalar)}
    files.update(records=records_path, mapping=mapping_path, source_contract=source_contract,
                 builder=Path(__file__), legacy_builder=Path(m.__file__),
                 runtime=Path(runtime.__file__), template=m.PROMPT_ROOT / "prompt.jinja")
    files.update(identity_normalizer=Path(sys.modules[normalize_molecule_identity.__module__].__file__),
                 level_mapping_manifest=m.LEVEL_MANIFEST)
    if not current_release:
        files.update(legacy_archive_manifest=ARCHIVE_MANIFEST,
                     mapping_archive_manifest=MAPPING_ARCHIVE_MANIFEST)
    return records, {k: {"path": str(p.resolve()), "sha256": runtime.file_sha256(p)} for k, p in files.items()}, issues


def audit_models(task, records):
    """Compare actual released examples with their training and copied renderers."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    sys.path.insert(0, str(TRAINING_ROOT))
    from assay_transfer.record_level.v20 import hf as training
    from assay_transfer.record_level.v23_2 import hf as training_bbb
    from assay_transfer.record_level.v24 import oral as training_oral
    m, renderer = MODULES[task], Renderer(task)
    report = {}
    for level, model in m.MODELS.items():
        root = Path(snapshot_download(model["dataset"], repo_type="dataset", revision=model["dataset_revision"], local_files_only=True))
        manifest = json.loads((root / "manifest.json").read_text())
        for name in ("records", "source_contract", "uid_levels", "gold_validation", "gold_test"):
            item = manifest["source_snapshot"][name]
            verified_source(item)
        counts = Counter()
        example_task = None
        for split in ("train", "validation_ranking", "test_ranking"):
            columns = ["prompt", "retrieval_record_id", "query_record_id", "retrieval_smiles", "query_smiles",
                       "retrieval_source_payload_json", "query_source_payload_json", "source_id", "measurement_kind"]
            for batch in pq.ParquetFile(root / split / "data.parquet").iter_batches(batch_size=2048, columns=columns):
                for row in batch.to_pylist():
                    stratum = (split, row["source_id"], row["measurement_kind"])
                    if counts[stratum] >= 16:
                        continue
                    projected = []
                    for role in ("retrieval", "query"):
                        record = records[row[f"{role}_record_id"].removeprefix(task + ":")]
                        payload = json.loads(row[f"{role}_source_payload_json"])
                        if m is bbb:
                            payload = training_bbb.prompt_payload({**record,
                                "pair_fields_json": record["canonical_pair_fields_json"],
                                "category_id": record["canonical_category_id"]}, payload)
                            fields_fn = training.projection.fields
                        else:
                            payload = training_oral.prompt_payload(record, payload)
                            fields_fn = training_oral.fields
                        projected.append((payload, fields_fn))
                    known = {"source_smiles": row["retrieval_smiles"], "_known_fields": projected[0][1](projected[0][0], True)}
                    query = {"source_smiles": row["query_smiles"], "_query_fields": projected[1][1](projected[1][0], False)}
                    if training._prompt(known, query) != row["prompt"]:
                        raise ValueError(f"released training prompt mismatch: {level}/{stratum}")
                    ref = records[row["retrieval_record_id"].removeprefix(task + ":")]
                    example_task = renderer.prompt_task(ref, row["query_smiles"])
                    copied = {k: v for k, v in projected[0][0].items() if k != "molecule_name"}
                    query["_query_fields"] = projected[0][1](copied, False)
                    if training._prompt(known, query) != example_task.prompt:
                        raise ValueError(f"copied-context training prompt mismatch: {level}/{stratum}")
                    counts[stratum] += 1
        snapshot = Path(runtime.resolve_model_snapshot(model["model"], model["revision"], local_files_only=True))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
        prefixes, a, b = runtime._answer_prefixes(tokenizer, [example_task])
        if not prefixes[0] or a[0] == b[0]:
            raise ValueError("invalid answer serialization")
        report[level] = {"model": model, "examples_checked": sum(counts.values()),
            "strata": {"/".join(k): v for k, v in counts.items()}, "prompt_mismatches": 0,
            "tokenizer_files": {p.name: runtime.file_sha256(p) for p in snapshot.iterdir()
                                if p.name.startswith(("tokenizer", "chat_template", "special_tokens"))},
            "answer_token_ids": [a[0], b[0]], "template_hash": renderer.template_hash,
            "scalar_projection_hash": renderer.legacy.projection_hash,
            "non_scalar_projection_hash": renderer.text_projection_hash}
        print(f"{task} {level}: {sum(counts.values())} released/copied prompts verified", flush=True)
    return report


def reusable_scores(task, subset):
    m, renderer = MODULES[task], Renderer(task)
    sources = [m.cache_paths(subset)]
    if m is oral:
        sources += [oral.cache_paths(subset, single_level=level) for level in ("L2", "L3")]
    scores, receipts = {}, []
    for p in sources:
        version = json.loads(p["version"].read_text())
        if (version["status"] != "complete" or version["task_id"] != task or version["subset"] != subset
                or version["template_hash"] != renderer.template_hash
                or version["scoring_contract_version"] != runtime.SCORING_CONTRACT_VERSION
                or any(spec != m.MODELS[level] for level, spec in version["models"].items())):
            raise ValueError(f"incompatible reuse source: {p['root']}")
        if runtime.file_sha256(p["cache"]) != version["cache_sha256"]:
            raise ValueError("reuse source hash mismatch")
        code_receipts = {"builder", "legacy_builder", "runtime"}
        resolved_inputs = {name: str(verified_source(item)) for name, item in version["inputs"].items()
                           if name not in code_receipts}
        c = sqlite3.connect(f"file:{p['cache']}?mode=ro", uri=True)
        if c.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("reuse database is corrupt")
        stored = {k: {**json.loads(v), "has_scalar": True} for k, v in c.execute("SELECT record_key,payload FROM records")}
        if version["projection_hash"] != renderer.legacy.projection_hash:
            # The old oral projection differs only in query molecule-name hiding.
            old_projection = {"always_hidden": sorted(oral.ALWAYS_HIDDEN),
                "query_result_fields": sorted(oral.QUERY_RESULT_FIELDS),
                "release_hidden": sorted(oral.HIDDEN),
                "source_field_policy": "source_column_contract.v2:source_or_simply_cleaned"}
            if (m is not oral or version["projection_hash"] != digest(old_projection)
                    or any(oral._clean(r["source_fields"].get(name)) is not None
                           for r in stored.values() for name in oral.QUERY_IDENTITY_FIELDS)):
                raise ValueError("reuse projection changes rendered prompt content")
        count = 0
        for query, record_key, probability in c.execute(
            "SELECT q.query_smiles,a.record_key,s.transfer_probability FROM assignments a "
            "JOIN queries q USING(query_id) JOIN scores s USING(score_key)"
        ):
            key = renderer.prompt_task(stored[record_key], query).cache_key
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid reused probability")
            if key in scores and scores[key] != probability:
                raise ValueError(f"conflicting reused score: {key}")
            scores[key] = probability
            count += 1
        c.close()
        receipts.append({"version": str(p["version"].resolve()), "version_sha256": runtime.file_sha256(p["version"]),
                         "cache": str(p["cache"].resolve()), "cache_sha256": version["cache_sha256"],
                         "verified_input_paths": resolved_inputs, "assignments": count,
                         "historical_code_receipts": {name: version["inputs"][name]
                                                      for name in code_receipts if name in version["inputs"]},
                         "source_projection_hash": version["projection_hash"],
                         "target_projection_hash": renderer.legacy.projection_hash,
                         "projection_rendering_unchanged": True})
        print(f"{task}/{subset}: checked {count} reusable assignments from {p['root'].name}", flush=True)
    return scores, receipts


def l1_reference(task, subset, lineage="v9"):
    root = v9.ranking_cache_dir(task, subset=subset, pool_size=100, lineage=lineage)
    version = json.loads((root / "VERSION.json").read_text())
    if version["status"] != "complete" or version["model"] != v9.model_profile(task, lineage):
        raise ValueError("L1 checkpoint mismatch")
    if version["prompt_assets"] != v9.verify_vendored_assets():
        raise ValueError("L1 prompt assets changed")
    if version["rankings_sha256"] != runtime.file_sha256(root / "rankings.parquet"):
        raise ValueError("L1 cache hash mismatch")
    return {"version": str((root / "VERSION.json").resolve()), "sha256": runtime.file_sha256(root / "VERSION.json"),
            "rankings_sha256": version["rankings_sha256"], "model": version["model"], "pool_size": 100}


def reuse_previous_three_pool(c, task, subset):
    """Fill exact prompt keys from the immutable complete Morgan-100 predecessor."""
    m, renderer = MODULES[task], Renderer(task)
    root = ROOT / task / "scaffold" / subset
    version_path, cache = root / "VERSION.json", root / "scores.sqlite3"
    version = json.loads(version_path.read_text())
    if (version.get("status") != "complete" or version.get("task_id") != task
            or version.get("subset") != subset or version.get("models") != m.MODELS
            or version.get("scoring_contract_version") != runtime.SCORING_CONTRACT_VERSION
            or runtime.file_sha256(cache) != version.get("cache_sha256")):
        raise ValueError(f"incompatible previous three-pool cache: {root}")
    for level, audit in version["model_prompt_audit"].items():
        if (audit["model"] != m.MODELS[level] or audit["template_hash"] != renderer.template_hash
                or audit["scalar_projection_hash"] != renderer.legacy.projection_hash
                or audit["non_scalar_projection_hash"] != renderer.text_projection_hash):
            raise ValueError(f"previous three-pool prompt identity changed: {task}/{level}")
    code_receipts = {"builder", "legacy_builder", "runtime"}
    verified = {name: str(verified_source(item)) for name, item in version["inputs"].items()
                if name not in code_receipts}
    prior = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    if prior.execute("PRAGMA quick_check").fetchone()[0] != "ok" or prior.execute(
            "SELECT COUNT(*) FROM scores WHERE transfer_probability IS NULL "
            "OR transfer_probability < 0 OR transfer_probability > 1").fetchone()[0]:
        raise ValueError("previous three-pool score database is invalid")
    prior.close()
    c.execute("ATTACH DATABASE ? AS prior", (str(cache),))
    try:
        c.execute("UPDATE scores SET transfer_probability=(SELECT p.transfer_probability FROM prior.scores p "
                  "WHERE p.cache_key=scores.cache_key), origin='exact_three_pool_cache_reuse' "
                  "WHERE transfer_probability IS NULL AND EXISTS "
                  "(SELECT 1 FROM prior.scores p WHERE p.cache_key=scores.cache_key)")
        reused = c.execute("SELECT changes()").fetchone()[0]
        c.commit()
    finally:
        c.execute("DETACH DATABASE prior")
    return {"version": str(version_path.resolve()), "version_sha256": runtime.file_sha256(version_path),
            "cache": str(cache.resolve()), "cache_sha256": version["cache_sha256"],
            "verified_input_paths": verified, "scores": reused,
            "historical_code_receipts": {name: version["inputs"][name]
                                         for name in code_receipts if name in version["inputs"]},
            "reuse_contract": "exact_cache_key_from_complete_three_pool_predecessor.v1"}


def previous_model_audit(task):
    """Reuse the immutable training/prompt audit when current V10 dropped old rows."""
    m, renderer = MODULES[task], Renderer(task)
    path = ROOT / task / "scaffold/valid/VERSION.json"
    version = json.loads(path.read_text())
    if version.get("status") != "complete" or version.get("models") != m.MODELS:
        raise ValueError(f"incompatible prior model audit: {path}")
    audit = version.get("model_prompt_audit") or {}
    for level, spec in m.MODELS.items():
        row = audit.get(level) or {}
        if (row.get("model") != spec or row.get("template_hash") != renderer.template_hash
                or row.get("scalar_projection_hash") != renderer.legacy.projection_hash
                or row.get("non_scalar_projection_hash") != renderer.text_projection_hash):
            raise ValueError(f"prior model prompt audit changed: {task}/{level}")
    return audit, {"path": str(path.resolve()), "sha256": runtime.file_sha256(path)}


def prepare(
    task, subset, root=None, gold_release="v1", l1_lineage="v9", *,
    identity_policy="scaffold_disjoint", current_release=False,
):
    p, m = paths(task, subset, root, gold_release), MODULES[task]
    if p["cache"].exists() or p["version"].exists():
        raise FileExistsError(f"cache already exists: {p['root']}")
    records, inputs, issues = (
        load_source(task, current_release=True) if current_release else load_source(task)
    )
    invalid = [item for item in issues if item[1] == "invalid_parent"]
    if invalid:
        raise ValueError(f"mapped records require an explicit identity decision: {invalid}")
    if issues:
        raise ValueError(f"rebuilt V10 has unmapped records: {issues}")
    if current_release:
        audit, prior_audit = previous_model_audit(task)
        inputs["prior_model_prompt_audit"] = prior_audit
    else:
        audit = audit_models(task, records)
    renderer = Renderer(task)
    # Render every source record before selecting candidates; no hidden failures.
    for r in records.values():
        renderer.render(r, "CCO")
    previous_cache = (ROOT / task / "scaffold" / subset / "VERSION.json").is_file()
    reused, reuse_sources = ({}, []) if previous_cache else reusable_scores(task, subset)
    l1 = l1_reference(task, subset, l1_lineage)
    frozen = bbb._read_jsonl(p["queries"])
    inputs["queries"] = {"path": str(p["queries"].resolve()), "sha256": runtime.file_sha256(p["queries"])}
    parent_forms, grouped = defaultdict(set), defaultdict(lambda: defaultdict(list))
    for key, r in records.items():
        parent_forms[r["parent_id"]].update(r.get("identity_smiles_forms", [r["canonical_smiles"]]))
        for pool in membership(r):
            grouped[pool, r["progressive_level"]][r["parent_id"]].append(key)
    parents = sorted(parent_forms)
    fps = [standardize_smiles_and_fp(normalize_molecule_identity(sorted(parent_forms[parent])[0]).parent_smiles)[2]
           for parent in parents]
    if any(fp is None for fp in fps):
        raise ValueError("invalid parent fingerprint")
    packed = np.stack([np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8) for fp in fps])
    bit_counts = bbb.POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    p["root"].mkdir(parents=True)
    c = sqlite3.connect(p["cache"])
    c.executescript("""
        PRAGMA journal_mode=DELETE;
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query_smiles TEXT UNIQUE NOT NULL);
        CREATE TABLE benchmark_queries(benchmark_row_id TEXT PRIMARY KEY, drug TEXT NOT NULL, query_id INTEGER NOT NULL);
        CREATE TABLE records(record_key INTEGER PRIMARY KEY, external_record_id TEXT UNIQUE NOT NULL,
                             parent_id TEXT NOT NULL, level TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE scores(score_key INTEGER PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, level TEXT NOT NULL,
                            projection_hash TEXT NOT NULL, transfer_probability REAL, origin TEXT);
        CREATE TABLE prompts(score_key INTEGER PRIMARY KEY, prompt TEXT NOT NULL);
        CREATE TABLE assignments(query_id INTEGER NOT NULL, pool TEXT NOT NULL, level TEXT NOT NULL,
                                 record_key INTEGER NOT NULL, score_key INTEGER NOT NULL,
                                 PRIMARY KEY(query_id,pool,level,record_key)) WITHOUT ROWID;
    """)
    record_keys, score_keys, query_keys = {}, {}, {}
    pool_counts = Counter()
    selected_pools = 0
    try:
        for row in frozen:
            query = row["molecule_identity"]["parent_smiles"]
            identity = normalize_molecule_identity(row["drug"])
            if identity.parent_inchi_key != row["molecule_identity"]["parent_inchi_key"]:
                raise ValueError("frozen query identity changed")
            qid = query_keys.setdefault(query, len(query_keys) + 1)
            c.execute("INSERT OR IGNORE INTO queries VALUES (?,?)", (qid, query))
            c.execute("INSERT INTO benchmark_queries VALUES (?,?,?)", (row["benchmark_row_id"], row["drug"], qid))
            if c.execute("SELECT 1 FROM assignments WHERE query_id=? LIMIT 1", (qid,)).fetchone():
                continue
            qfp = standardize_smiles_and_fp(query)[2]
            sims = bbb._similarities(qfp, packed, bit_counts)
            independent = DataStructs.BulkTanimotoSimilarity(qfp, fps)
            ranked = sorted(range(len(parents)), key=lambda i: (-float(sims[i]), parents[i]))
            if ranked != sorted(range(len(parents)), key=lambda i: (-independent[i], parents[i])):
                raise ValueError("independent Morgan ordering mismatch")
            selected = {key: [] for key in grouped}
            for index in ranked:
                parent = parents[index]
                relevant = [key for key in grouped if len(selected[key]) < POOL_SIZE and parent in grouped[key]]
                if not relevant or any(decide_candidate(identity, {"canonical_smiles": s}, identity_policy).excluded
                                       for s in parent_forms[parent]):
                    continue
                for key in relevant:
                    selected[key].append(parent)
                if all(len(v) == POOL_SIZE for v in selected.values()):
                    break
            if len(selected) != len(POOLS) * len(m.MODELS) or any(len(v) != POOL_SIZE for v in selected.values()):
                raise ValueError(f"insufficient pool coverage for {row['benchmark_row_id']}")
            for (pool, level), selected_parents in selected.items():
                expected_records = {key for parent in selected_parents for key in grouped[pool, level][parent]}
                for key in sorted(expected_records):
                    r = records[key]
                    if decide_candidate(identity, r, identity_policy).excluded:
                        raise ValueError("expanded record violates the identity policy")
                    if key not in record_keys:
                        record_keys[key] = len(record_keys) + 1
                        c.execute("INSERT INTO records VALUES (?,?,?,?,?)", (record_keys[key], key, r["parent_id"], level,
                                  json.dumps(r, sort_keys=True, ensure_ascii=False)))
                    prompt_task = renderer.prompt_task(r, query)
                    ck = prompt_task.cache_key
                    if ck not in score_keys:
                        sk = score_keys[ck] = len(score_keys) + 1
                        probability = reused.get(ck)
                        c.execute("INSERT INTO scores VALUES (?,?,?,?,?,?)", (sk, ck, level, prompt_task.projection_hash,
                                  probability, "exact_cache_reuse" if probability is not None else None))
                        c.execute("INSERT INTO prompts VALUES (?,?)", (sk, prompt_task.prompt))
                    c.execute("INSERT INTO assignments VALUES (?,?,?,?,?)", (qid, pool, level, record_keys[key], score_keys[ck]))
                observed = {k for (k,) in c.execute(
                    "SELECT r.external_record_id FROM assignments a JOIN records r USING(record_key) "
                    "WHERE query_id=? AND pool=? AND a.level=?", (qid, pool, level))}
                if expected_records != observed:
                    raise ValueError("record expansion mismatch")
                pool_counts[pool, level] += len(expected_records)
                selected_pools += 1
            c.commit()
            print(f"{task}/{subset}: {len(query_keys)} query parents; {len(score_keys)} unique prompts", flush=True)
        if previous_cache:
            reuse_sources.append(reuse_previous_three_pool(c, task, subset))
        counts = {"/".join((level, origin or "pending")): n for level, origin, n in c.execute(
                  "SELECT level,origin,COUNT(*) FROM scores GROUP BY level,origin")}
        c.execute("CREATE INDEX score_level ON scores(level,score_key)")
        c.execute("CREATE INDEX assignment_score ON assignments(score_key)")
        c.commit()
    finally:
        c.close()
    version = {"schema_version": SCHEMA, "status": "prepared", "task_id": task, "subset": subset,
        "models": m.MODELS, "pools": list(POOLS), "morgan_pool_size": POOL_SIZE,
        "gold_release": gold_release, "l1_lineage": l1_lineage,
        "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
        "record_eligibility_tag_filters_membership": False,
        "neighbor_identity_policy": f"all_parent_forms_{identity_policy}",
        "evidence_release": "current_v10" if current_release else "pinned_historical_v10",
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "scoring_contract_version": runtime.SCORING_CONTRACT_VERSION, "inputs": inputs,
        "model_prompt_audit": audit, "l1_shared_cache": l1, "reuse_sources": reuse_sources,
        "source_issues": issues,
        "n_query_rows": len(frozen), "n_query_parents": len(query_keys), "n_unique_scores": len(score_keys),
        "pool_assignment_counts": {"/".join(k): v for k, v in pool_counts.items()}, "score_counts": counts,
        "preparation_audit": {"source_records_rendered": len(records), "independent_morgan_pools": selected_pools,
                              "record_expansion_mismatches": 0, "record_identity_overlaps": 0},
        "runtime": bbb._runtime_metadata()}
    bbb._write_json(p["version"], version)
    return version


def verify_inputs(version):
    if version["schema_version"] != SCHEMA or version["models"] != MODULES[version["task_id"]].MODELS:
        raise ValueError("incompatible three-pool contract")
    code_receipts = {"builder", "legacy_builder", "runtime"}
    for name, item in version["inputs"].items():
        if name in code_receipts:
            continue
        if runtime.file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"cache input changed: {item['path']}")


def score(task, subset, level, device, num_shards=1, shard_index=0, batch_size=128, root=None):
    if level not in MODULES[task].MODELS or not 0 <= shard_index < num_shards or batch_size < 1:
        raise ValueError("invalid scoring arguments")
    p = paths(task, subset, root)
    version = json.loads(p["version"].read_text())
    verify_inputs(version)
    if version["status"] != "prepared":
        raise ValueError("cache is not prepared")
    c = sqlite3.connect(f"file:{p['cache']}?mode=ro", uri=True)
    rows = c.execute("SELECT cache_key,prompt,projection_hash FROM scores JOIN prompts USING(score_key) "
                     "WHERE level=? AND transfer_probability IS NULL ORDER BY LENGTH(prompt),cache_key", (level,)).fetchall()
    c.close()
    rows = rows[shard_index::num_shards]
    p["journals"].mkdir(exist_ok=True)
    journal = p["journals"] / f"{level}-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    done = bbb._read_jsonl(journal) if journal.exists() else []
    if [r["cache_key"] for r in done] != [r[0] for r in rows[:len(done)]]:
        raise ValueError("journal is not the exact completed shard prefix")
    spec = MODULES[task].MODELS[level]
    renderer = Renderer(task)
    snapshot = runtime.resolve_model_snapshot(spec["model"], spec["revision"], local_files_only=True)
    model, tokenizer = runtime.load_model(snapshot, device=device)
    with journal.open("a") as handle:
        for offset in range(len(done), len(rows), batch_size):
            batch = [runtime.PromptTask(cache_key=key, prompt=prompt, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                       task_id=task, query_smiles="", group_id=level, molecule_id="", record_id="", model=spec["model"],
                       model_revision=spec["revision"], scoring_contract_version=runtime.SCORING_CONTRACT_VERSION,
                       template_hash=renderer.template_hash, projection_hash=projection)
                     for key, prompt, projection in rows[offset:offset + batch_size]]
            for result in runtime.score_prompt_batch(model, tokenizer, batch, device=device):
                handle.write(json.dumps({"cache_key": result.cache_key, "transfer_probability": result.transfer_probability}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"{task}/{subset}/{level}/{shard_index}: {offset + len(batch)}/{len(rows)}", flush=True)
    return {"status": "complete", "level": level, "shard_index": shard_index, "scores": len(rows)}


def finalize(task, subset, root=None):
    p = paths(task, subset, root)
    version = json.loads(p["version"].read_text())
    verify_inputs(version)
    if version["status"] != "prepared":
        raise ValueError("cache is not prepared")
    c = sqlite3.connect(p["cache"])
    try:
        c.execute("CREATE TEMP TABLE fresh(cache_key TEXT PRIMARY KEY, probability REAL NOT NULL)")
        for journal in sorted(p["journals"].glob("*.jsonl")):
            for row in bbb._read_jsonl(journal):
                value = row["transfer_probability"]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid fresh probability")
                c.execute("INSERT INTO fresh VALUES (?,?)", (row["cache_key"], value))
        missing = c.execute("SELECT cache_key FROM scores WHERE transfer_probability IS NULL EXCEPT SELECT cache_key FROM fresh").fetchone()
        extra = c.execute("SELECT cache_key FROM fresh EXCEPT SELECT cache_key FROM scores WHERE transfer_probability IS NULL").fetchone()
        if missing or extra:
            raise ValueError(f"incomplete/conflicting journals: missing={missing}, extra={extra}")
        c.execute("UPDATE scores SET transfer_probability=(SELECT probability FROM fresh WHERE fresh.cache_key=scores.cache_key), "
                  "origin='fresh_inference' WHERE transfer_probability IS NULL")
        if c.execute("SELECT COUNT(*) FROM scores WHERE transfer_probability IS NULL").fetchone()[0]:
            raise ValueError("missing probabilities")
        c.commit()
        c.execute("DROP TABLE prompts")
        c.commit()
        c.execute("VACUUM")
        if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("final cache integrity failure")
        counts = {"/".join((level, origin)): n for level, origin, n in c.execute(
                  "SELECT level,origin,COUNT(*) FROM scores GROUP BY level,origin")}
    finally:
        c.close()
    version.update(status="complete", score_counts=counts, cache_sha256=runtime.file_sha256(p["cache"]), cache_integrity="ok")
    bbb._write_json(p["version"], version)
    for journal in p["journals"].glob("*.jsonl"):
        journal.unlink()
    if p["journals"].exists():
        p["journals"].rmdir()
    return version


def load_top_ranked_records(queries, *, task, subset, pool, levels=None, limit=50):
    if pool not in POOLS or limit < 1:
        raise ValueError("invalid pool or record limit")
    levels = tuple(levels or MODULES[task].MODELS)
    if not levels or not set(levels) <= set(MODULES[task].MODELS):
        raise ValueError("invalid levels")
    p = paths(task, subset)
    version = json.loads(p["version"].read_text())
    verify_inputs(version)
    if version["status"] != "complete" or runtime.file_sha256(p["cache"]) != version["cache_sha256"]:
        raise ValueError("cache is incomplete or changed")
    c = sqlite3.connect(f"file:{p['cache']}?mode=ro", uri=True)
    output = {}
    try:
        for query_id, drug in queries.items():
            row = c.execute("SELECT query_id,drug FROM benchmark_queries WHERE benchmark_row_id=?", (str(query_id),)).fetchone()
            if row is None or row[1] != drug:
                raise ValueError(f"query differs from frozen ledger: {query_id}")
            output[query_id] = {}
            for level in levels:
                selected = c.execute("SELECT r.payload,s.transfer_probability FROM assignments a JOIN records r USING(record_key) "
                    "JOIN scores s USING(score_key) WHERE query_id=? AND pool=? AND a.level=? "
                    "ORDER BY s.transfer_probability DESC,r.external_record_id LIMIT ?", (row[0], pool, level, limit)).fetchall()
                if len(selected) != limit:
                    raise ValueError(f"insufficient scored records: {query_id}/{level}")
                output[query_id][level] = [{**json.loads(payload), "transfer_probability": score} for payload, score in selected]
    finally:
        c.close()
    return output, {"pool": pool, "version": str(p["version"].resolve()), "cache_sha256": version["cache_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score", "finalize"))
    parser.add_argument("--task", choices=tuple(MODULES), required=True)
    parser.add_argument("--subset", choices=("valid", "test"), required=True)
    parser.add_argument("--level")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--gold-release", choices=("v1", "v2"), default="v1")
    parser.add_argument("--l1-lineage", choices=v9.DIRECT_LINEAGES, default="v9")
    parser.add_argument(
        "--identity-policy", choices=("parent_disjoint", "scaffold_disjoint"),
        default="scaffold_disjoint",
    )
    parser.add_argument("--current-release", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.task, args.subset, args.cache_root, args.gold_release, args.l1_lineage,
            identity_policy=args.identity_policy, current_release=args.current_release,
        )
    elif args.command == "finalize":
        result = finalize(args.task, args.subset, args.cache_root)
    else:
        result = score(
            args.task, args.subset, args.level, args.device, args.num_shards,
            args.shard_index, args.batch_size, args.cache_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
