#!/usr/bin/env python3
"""Audit whether stored Stage-01 structures match subjects named in evidence text.

Discovery mode only inventories rows that may need later review.  Audit mode is
a proposal generator.  Neither edits Stage-01 records or reviewed ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq
import requests
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.llm_client import openai_client


REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_VERSION = "smiles_identity_audit.v2"
PROMPT_VERSION = "smiles_identity_subject_extraction.v2"
NAME_EXTRACTION_VERSION = "missing_molecule_name_extraction.v5"
NAME_EXTRACTION_PROMPT_VERSION = "missing_molecule_name_extraction.v5"
NAME_EXTRACTION_REASONING_EFFORT = "xhigh"
NAME_EXTRACTION_TIMEOUT_S = 14400
NAME_SMILES_COMPARISON_VERSION = "name_smiles_comparison.v1"
NAME_EXTRACTION_MODEL = "gpt-5.6-luna"
DEFAULT_MODEL = "gpt-5.4-mini"
TASK_CONFIG = {
    "bbb_martins": {
        "sample": REPO_ROOT
        / "tools/chembl_tool/tasks/bbb_martins/data_processing/"
        "source_value_cleaning_v1/smiles_identity_sample_audit_20260829.json",
        "sample_field": "sampled_rows",
    },
    "bioavailability_ma": {
        "sample": REPO_ROOT
        / "tools/chembl_tool/tasks/bioavailability_ma/data_processing/"
        "source_value_cleaning_v1/smiles_identity_sample_audit_20260829.json",
        "sample_field": "classification_ledger",
    },
    "skin_reaction": {
        "sample": REPO_ROOT
        / "tools/chembl_tool/tasks/skin_reaction/data_processing/"
        "source_value_cleaning_v1/smiles_identity_sample_audit_20260829.json",
        "sample_field": "ledger",
    },
}


class ExtractedSubject(BaseModel):
    id: str
    status: Literal[
        "single_explicit_subject",
        "multiple_compounds_ambiguous",
        "subject_not_explicit",
    ]
    molecule_name: str | None
    evidence_span: str | None


class ExtractionResponse(BaseModel):
    items: list[ExtractedSubject]


class NameExtractionItem(BaseModel):
    id: str
    status: Literal["explicit", "ambiguous", "absent"]
    molecule_name: str | None


class NameExtractionResponse(BaseModel):
    items: list[NameExtractionItem]


class ReviewedMismatch(BaseModel):
    id: str
    decision: Literal[
        "confirmed_mismatch",
        "same_identity_or_form",
        "ambiguous_keep",
    ]
    evidence_span: str | None
    rationale: str


class ReviewResponse(BaseModel):
    items: list[ReviewedMismatch]


EXTRACTION_SYSTEM_PROMPT = """You audit molecular evidence rows.
For each item, extract the exact molecule name for the indexed experimental subject.
Use only the supplied row. Do not use outside chemical knowledge. Do not select a
comparator, inhibitor, pretreatment, vehicle, metabolite, formulation ingredient,
or assay reagent unless the row explicitly identifies it as the measured subject.
The endpoint and measurement describe what was read out; treatment language alone
does not identify the indexed structure. If the row names a treatment plus an assay
marker, parent drug plus metabolite, substrate plus inhibitor, or otherwise multiple
plausible indexed chemicals, mark multiple_compounds_ambiguous unless the supplied
fields themselves uniquely identify one. Copy the shortest verbatim name span. If
the subject is not explicit, mark subject_not_explicit. Return exactly one result
per input id."""


REVIEW_SYSTEM_PROMPT = """You audit a possible molecule/structure mismatch.
Use the supplied row text, extracted subject span, and PubChem lookup provenance.
confirmed_mismatch means the row clearly measures the named subject and the stored
structure belongs to a different chemical. same_identity_or_form covers salts,
solvates, stereochemical variants, mixtures, aliases, or otherwise compatible
representations. ambiguous_keep means the row does not justify intervention.
PubChem lookup by a short name can select the wrong synonym and is not by itself
proof of a mismatch. If the row names multiple chemicals and the stored structure
could represent any analyte, marker, treatment, parent, or metabolite in the row,
choose ambiguous_keep. Be conservative. Return exactly one result per input id."""


NAME_EXTRACTION_PROMPT = """Extract the experimental subject molecule from each evidence row.
Return one result per id with status explicit, ambiguous, or absent and molecule_name.
Copy molecule_name verbatim from the text only when one measured subject is explicit.
Use endpoint_name and measurement_text to identify what the support_text measures, but
copy molecule_name only from support_text.
The subject is the molecule whose concentration, permeability, flux, clearance,
AUC, response, or other outcome is measured. A treatment or condition does not
become the subject merely because it changes that outcome. For "X increased the
permeability of Y", choose Y; if Y is unnamed, use absent. When formulations of
one named drug are compared but the drug outcome is measured, choose that drug.
Do not select a comparator, inhibitor, vehicle, formulation ingredient, metabolite,
or assay reagent unless it is itself the measured subject. Use ambiguous when
multiple molecules each have the measured outcome or the subject remains unclear.
For explicit, molecule_name must be one exact contiguous substring copied from the
support text, including its original spelling and punctuation. Never correct a name,
infer an unstated molecule, or return placeholders such as "unclear". If no exact
molecule span is present, use absent. molecule_name must be null unless status is
explicit. Generic words such as "drug", "compound", or "formulation", and phrases
describing a dose, Cmax, AUC, or other measurement are not molecule names. Prefer the
named molecule directly tied to the primary measured outcome over inhibitors,
excipients, and compounds mentioned only as context. Multiple formulations of the
same named drug still have that drug as the subject. If an excipient changes an
unnamed drug metric, the subject is absent, not the excipient."""


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "complete":
                rows[str(row["request_id"])] = row
    return rows


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _policy(task: str) -> Any:
    return importlib.import_module(
        f"tools.chembl_tool.tasks.{task}.starling_policy"
    ).POLICY


def _stage1_path(task: str) -> Path:
    return (
        REPO_ROOT
        / f"outputs/chembl_tool/tasks/{task}/evidence_library/"
        "starling_normalized_v7/01_cleaned/records.parquet"
    )


def _pilot_rows(task: str, stage1_path: Path, sample_path: Path) -> list[dict[str, Any]]:
    config = TASK_CONFIG[task]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))[
        config["sample_field"]
    ]
    wanted = {
        (str(row.get("source_id") or ""), int(row.get("source_row_number") or 0)): row
        for row in sample
    }
    requested_columns = (
        "cleaned_record_id",
        "source_id",
        "source_row_number",
        "source_record_id",
        "endpoint_name",
        "measurement_text",
        "support_text",
        "source_smiles",
        "canonical_smiles",
        "structure_status",
        "smiles",
    )
    available = set(pq.read_schema(stage1_path).names)
    has_persisted_source_smiles = "source_smiles" in available
    no_name_sources = sorted(
        source_id
        for source_id, profile in _policy(task).record_contract.sources.items()
        if "molecule_name" not in profile.source_columns
    )
    frame = pd.read_parquet(
        stage1_path,
        columns=[field for field in requested_columns if field in available],
        filters=[("source_id", "in", no_name_sources)],
    )
    stored_column = "source_smiles" if has_persisted_source_smiles else "smiles"
    stored = frame[stored_column]
    frame = frame.loc[stored.notna() & stored.astype(str).str.strip().ne("")].copy()
    if not has_persisted_source_smiles:
        frame["source_smiles"] = frame["smiles"]
    inventory = json.loads(
        (stage1_path.parents[1] / "00_source/source_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    frame.insert(0, "id", frame["cleaned_record_id"].astype(str))
    frame.insert(2, "source_sha256", frame["source_id"].map(inventory["source_hashes"]))
    wanted_index = pd.MultiIndex.from_tuples(wanted)
    row_index = pd.MultiIndex.from_frame(frame[["source_id", "source_row_number"]])
    frame = frame.loc[row_index.isin(wanted_index)]
    output: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        key = (str(row["source_id"]), int(row["source_row_number"]))
        output.append(
            {
                "id": str(row.get("id") or ""),
                "task_id": task,
                "source_id": str(row.get("source_id") or ""),
                "source_sha256": str(row.get("source_sha256") or ""),
                "source_row_number": int(row.get("source_row_number") or 0),
                "source_record_id": str(row.get("source_record_id") or ""),
                "endpoint_name": _text(row.get("endpoint_name")),
                "measurement_text": _text(row.get("measurement_text")),
                "support_text": _text(row.get("support_text")),
                "source_smiles": _text(row.get("source_smiles")),
                "canonical_smiles": _text(row.get("canonical_smiles")),
                "structure_status": _text(row.get("structure_status")),
                "manual_classification": str(
                    wanted[key].get("classification") or ""
                ),
            }
        )
    return sorted(output, key=lambda row: row["id"])


def _candidate_frame(task: str, stage1_path: Path) -> tuple[pd.DataFrame, list[str]]:
    requested_columns = (
        "cleaned_record_id",
        "source_id",
        "source_row_number",
        "source_record_id",
        "endpoint_name",
        "measurement_text",
        "support_text",
        "molecule_name",
        "source_smiles",
        "canonical_smiles",
        "structure_status",
        "smiles",
    )
    available = set(pq.read_schema(stage1_path).names)
    has_persisted_source_smiles = "source_smiles" in available
    frame = pd.read_parquet(
        stage1_path,
        columns=[field for field in requested_columns if field in available],
    )
    if "molecule_name" not in frame:
        frame["molecule_name"] = None
    if "canonical_smiles" not in frame:
        frame["canonical_smiles"] = None
    if "support_text" not in frame:
        frame["support_text"] = None
    names = frame["molecule_name"]
    structures = frame["canonical_smiles"]
    support = frame["support_text"]
    frame = frame.loc[
        frame["structure_status"].eq("resolved")
        & structures.notna()
        & structures.astype(str).str.strip().ne("")
        & (names.isna() | names.astype(str).str.strip().eq(""))
        & support.notna()
        & support.astype(str).str.strip().ne("")
    ].copy()
    if not has_persisted_source_smiles:
        frame["source_smiles"] = frame["smiles"]
    inventory = json.loads(
        (stage1_path.parents[1] / "00_source/source_inventory.json").read_text(
            encoding="utf-8"
        )
    )
    source_hashes = inventory["source_hashes"]
    frame.insert(0, "id", frame["cleaned_record_id"].astype(str))
    frame.insert(1, "task_id", task)
    frame.insert(3, "source_sha256", frame["source_id"].map(source_hashes))
    frame.insert(4, "candidate_reason", "resolved_structure_missing_molecule_name")
    if frame["source_sha256"].isna().any():
        raise ValueError("candidate source is absent from the Stage-00 inventory")
    frame = frame.drop(columns=["smiles"], errors="ignore")
    source_ids = sorted(frame["source_id"].astype(str).unique())
    return frame.sort_values("id", kind="stable").reset_index(drop=True), source_ids


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _request_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("id", "source_id", "endpoint_name", "measurement_text", "support_text")
    return [{field: row.get(field) for field in fields} for row in rows]


def _call_extraction(client: OpenAI, rows: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    aliases = {f"r{index:03d}": row["id"] for index, row in enumerate(rows)}
    request_rows = _request_rows(rows)
    for alias, row in zip(aliases, request_rows, strict=True):
        row["id"] = alias
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "none"},
        store=False,
        input=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"items": request_rows}, ensure_ascii=False),
            },
        ],
        text_format=ExtractionResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("extraction response did not contain parsed output")
    items = [item.model_dump() for item in parsed.items]
    _validate_aliases(items, aliases)
    for item in items:
        item["id"] = aliases[item["id"]]
    return items, _usage(response)


def _call_name_extraction(
    client: OpenAI,
    rows: list[dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    aliases = {f"r{index:03d}": row["id"] for index, row in enumerate(rows)}
    request_rows = [
        {
            "id": alias,
            "endpoint_name": row.get("endpoint_name"),
            "measurement_text": row.get("measurement_text"),
            "support_text": row["support_text"],
        }
        for alias, row in zip(aliases, rows, strict=True)
    ]
    response = client.responses.parse(
        model=model,
        reasoning={"effort": NAME_EXTRACTION_REASONING_EFFORT},
        store=False,
        input=[
            {"role": "system", "content": NAME_EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"items": request_rows}, ensure_ascii=False),
            },
        ],
        text_format=NameExtractionResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("name extraction response did not contain parsed output")
    items = [item.model_dump() for item in parsed.items]
    _validate_aliases(items, aliases)
    for item in items:
        item["id"] = aliases[item["id"]]
    return items, _usage(response)


def _call_review(client: OpenAI, rows: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fields = (
        "id",
        "source_id",
        "endpoint_name",
        "measurement_text",
        "support_text",
        "source_smiles",
        "extracted_molecule_name",
        "extraction_evidence_span",
        "pubchem_title",
        "pubchem_smiles",
        "stored_parent_inchi_key",
        "pubchem_parent_inchi_key",
    )
    aliases = {f"r{index:03d}": row["id"] for index, row in enumerate(rows)}
    request_rows = [{field: row.get(field) for field in fields} for row in rows]
    for alias, row in zip(aliases, request_rows, strict=True):
        row["id"] = alias
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "none"},
        store=False,
        input=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"items": request_rows},
                    ensure_ascii=False,
                ),
            },
        ],
        text_format=ReviewResponse,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("review response did not contain parsed output")
    items = [item.model_dump() for item in parsed.items]
    _validate_aliases(items, aliases)
    for item in items:
        item["id"] = aliases[item["id"]]
    return items, _usage(response)


def _validate_aliases(items: list[dict[str, Any]], aliases: dict[str, str]) -> None:
    actual = [str(item.get("id") or "") for item in items]
    if len(actual) != len(set(actual)) or set(actual) != set(aliases):
        raise ValueError("response aliases do not exactly match requested aliases")


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _run_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    workers: int,
    model: str,
    prompt_version: str,
    log_path: Path,
    call: Any,
    client: OpenAI,
    request_namespace: str = "",
    request_variant: str = "",
    endpoint_clients: list[tuple[OpenAI, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], int]:
    cached = _load_jsonl(log_path)
    endpoints = endpoint_clients or [(client, request_namespace)]
    planned: list[tuple[str, list[dict[str, Any]], int]] = []
    required_request_ids: set[str] = set()
    for batch_index, batch in enumerate(_chunks(rows, batch_size)):
        endpoint_index = batch_index % len(endpoints)
        namespace = endpoints[endpoint_index][1]
        request_payload = {
            "audit_version": AUDIT_VERSION,
            "prompt_version": prompt_version,
            "model": model,
            "rows": batch,
        }
        if namespace:
            request_payload["request_namespace"] = namespace
        if request_variant:
            request_payload["request_variant"] = request_variant
        request_id = _json_hash(request_payload)
        required_request_ids.add(request_id)
        if request_id not in cached:
            planned.append((request_id, batch, endpoint_index))
    lock = threading.Lock()
    executors = [ThreadPoolExecutor(max_workers=workers) for _ in endpoints]
    try:
        futures = {
            executors[endpoint_index].submit(
                _call_with_retries,
                call,
                endpoints[endpoint_index][0],
                batch,
                model,
            ): (
                request_id,
                batch,
                endpoint_index,
            )
            for request_id, batch, endpoint_index in planned
        }
        for future in as_completed(futures):
            request_id, batch, endpoint_index = futures[future]
            items, usage = future.result()
            namespace = endpoints[endpoint_index][1]
            row = {
                "request_id": request_id,
                "status": "complete",
                "model": model,
                "request_namespace": namespace or None,
                "request_variant": request_variant or None,
                "prompt_version": prompt_version,
                "item_count": len(batch),
                "usage": usage,
                "items": items,
            }
            _append_jsonl(log_path, row, lock)
            cached[request_id] = row
    finally:
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)
    results: dict[str, dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    used_requests = 0
    needed_ids = {row["id"] for row in rows}
    for request_id, entry in cached.items():
        if request_id not in required_request_ids:
            continue
        used_requests += 1
        for key in usage:
            usage[key] += int(entry.get("usage", {}).get(key, 0) or 0)
        for item in entry["items"]:
            results[str(item["id"])] = {
                **item,
                "_request_namespace": entry.get("request_namespace"),
            }
    if set(results) != needed_ids:
        raise ValueError("completed batches do not cover every requested row")
    return results, usage, used_requests


def _call_with_retries(
    call: Any,
    client: OpenAI,
    batch: list[dict[str, Any]],
    model: str,
    attempts: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    for attempt in range(attempts):
        try:
            return call(client, batch, model)
        except RateLimitError as error:
            if getattr(error, "code", None) in {
                "credit_balance_exhausted",
                "insufficient_quota",
            }:
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 30))
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


class PubChemCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.next_request_at = 0.0
        self.pending_writes = 0
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS lookup (name TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def resolve(self, name: str, timeout_s: float) -> dict[str, Any]:
        key = " ".join(name.casefold().split())
        with self.lock:
            cached = self.connection.execute(
                "SELECT payload FROM lookup WHERE name = ?", (key,)
            ).fetchone()
        if cached:
            return json.loads(cached[0])
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name, safe='')}/property/Title,CanonicalSMILES,"
            "IsomericSMILES,InChIKey/JSON"
        )
        response = None
        for attempt in range(3):
            try:
                with self.lock:
                    delay = max(0.0, self.next_request_at - time.monotonic())
                    self.next_request_at = max(
                        self.next_request_at, time.monotonic()
                    ) + 0.2
                if delay:
                    time.sleep(delay)
                session = getattr(self.local, "session", None)
                if session is None:
                    session = self.local.session = requests.Session()
                response = session.get(url, timeout=timeout_s)
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            except requests.RequestException:
                response = None
            time.sleep(2**attempt)
        if response is None or response.status_code in {429, 500, 502, 503, 504}:
            return {"status": "lookup_error", "query_name": name}
        if response.status_code in {400, 404, 422}:
            payload = {
                "status": "invalid_query" if response.status_code != 404 else "not_found",
                "query_name": name,
            }
        else:
            response.raise_for_status()
            properties = response.json().get("PropertyTable", {}).get("Properties", [])
            if len(properties) != 1:
                payload = {
                    "status": "ambiguous" if properties else "not_found",
                    "query_name": name,
                    "result_count": len(properties),
                }
            else:
                item = properties[0]
                smiles = (
                    item.get("SMILES")
                    or item.get("IsomericSMILES")
                    or item.get("ConnectivitySMILES")
                    or item.get("CanonicalSMILES")
                )
                payload = {
                    "status": "resolved" if smiles else "no_structure",
                    "query_name": name,
                    "cid": item.get("CID"),
                    "title": item.get("Title"),
                    "smiles": smiles,
                    "inchi_key": item.get("InChIKey"),
                }
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO lookup(name, payload) VALUES (?, ?)",
                (key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            self.pending_writes += 1
            if self.pending_writes >= 50:
                self.connection.commit()
                self.pending_writes = 0
        return payload

    def close(self) -> None:
        with self.lock:
            self.connection.commit()
            self.connection.close()


def _compare(row: dict[str, Any], extraction: dict[str, Any], pubchem: dict[str, Any]) -> dict[str, Any]:
    result = {**row, **{
        "extraction_status": extraction["status"],
        "extracted_molecule_name": extraction.get("molecule_name"),
        "extraction_evidence_span": extraction.get("evidence_span"),
        "pubchem_status": pubchem.get("status"),
        "pubchem_query_name": pubchem.get("query_name"),
        "pubchem_cid": pubchem.get("cid"),
        "pubchem_title": pubchem.get("title"),
        "pubchem_smiles": pubchem.get("smiles"),
    }}
    if pubchem.get("status") != "resolved":
        result["identity_comparison"] = "unresolved_keep"
        return result
    stored_smiles = str(row.get("canonical_smiles") or row.get("source_smiles") or "")
    if "*" in stored_smiles:
        result["identity_comparison"] = "generic_or_mixture_keep"
        return result
    stored = normalize_molecule_identity(
        stored_smiles
    )
    resolved = normalize_molecule_identity(str(pubchem["smiles"]))
    result.update(
        {
            "stored_standard_inchi_key": stored.standard_inchi_key,
            "stored_parent_inchi_key": stored.parent_inchi_key,
            "pubchem_standard_inchi_key": resolved.standard_inchi_key,
            "pubchem_parent_inchi_key": resolved.parent_inchi_key,
        }
    )
    if stored.standard_inchi_key and stored.standard_inchi_key == resolved.standard_inchi_key:
        comparison = "exact_identity"
    elif (
        stored.parent_connectivity_key
        and stored.parent_connectivity_key == resolved.parent_connectivity_key
    ):
        comparison = "same_parent_or_form"
    elif set(stored.component_parent_inchi_keys) & set(
        resolved.component_parent_inchi_keys
    ):
        comparison = "same_parent_or_form"
    elif _same_named_metal_component(stored_smiles, str(pubchem["smiles"])):
        comparison = "same_parent_or_form"
    elif _same_heavy_atom_graph(
        stored_smiles,
        str(pubchem["smiles"]),
    ):
        comparison = "same_parent_or_form"
    elif _same_molecular_formula(stored_smiles, str(pubchem["smiles"])):
        comparison = "same_formula_isomer_keep"
    elif stored.status != "ok":
        comparison = "stored_structure_invalid"
    else:
        comparison = "different_parent"
    result["identity_comparison"] = comparison
    return result


def _same_heavy_atom_graph(left: str, right: str) -> bool:
    with rdBase.BlockLogs():
        first = Chem.MolFromSmiles(left, sanitize=False)
        second = Chem.MolFromSmiles(right, sanitize=False)
        if first is None or second is None:
            return False
        if (
            first.GetNumAtoms() != second.GetNumAtoms()
            or first.GetNumBonds() != second.GetNumBonds()
        ):
            return False
        match = rdFMCS.FindMCS(
            [first, second],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=2,
        )
    return bool(
        not match.canceled
        and match.numAtoms == first.GetNumAtoms()
        and match.numBonds == first.GetNumBonds()
    )


def _same_molecular_formula(left: str, right: str) -> bool:
    with rdBase.BlockLogs():
        first = Chem.MolFromSmiles(left)
        second = Chem.MolFromSmiles(right)
    return bool(
        first is not None
        and second is not None
        and CalcMolFormula(first) == CalcMolFormula(second)
    )


def _same_named_metal_component(left: str, right: str) -> bool:
    metals = {
        "Al",
        "Ca",
        "Co",
        "Cr",
        "Cu",
        "Fe",
        "K",
        "Li",
        "Mg",
        "Mn",
        "Na",
        "Ni",
        "Sn",
        "Zn",
    }
    with rdBase.BlockLogs():
        first = Chem.MolFromSmiles(left, sanitize=False)
        second = Chem.MolFromSmiles(right, sanitize=False)
    if first is None or second is None:
        return False
    for single, mixture in ((first, second), (second, first)):
        if single.GetNumAtoms() != 1:
            continue
        symbol = single.GetAtomWithIdx(0).GetSymbol()
        if symbol in metals and any(
            atom.GetSymbol() == symbol for atom in mixture.GetAtoms()
        ):
            return True
    return False


def _pubchem_lookup_name(name: str) -> str | None:
    value = " ".join(str(name).split()).strip(" ,;:")
    if not value:
        return None
    if value.casefold() in {
        "dog",
        "dogs",
        "human",
        "humans",
        "man",
        "men",
        "mouse",
        "mice",
        "monkey",
        "monkeys",
        "rat",
        "rats",
        "rabbit",
        "rabbits",
    }:
        return None
    if re.match(
        r"^(?:compound|drug|agent|test compound)\s*[-#]?\s*[A-Z0-9-]+\b",
        value,
        re.I,
    ):
        return None
    if " " not in value and len(value) <= 12 and (
        any(character.isdigit() for character in value)
        or (value.isupper() and len(value) <= 8)
    ):
        return None
    value = re.sub(r"^\[[^]]*\]", "", value).strip()
    value = re.sub(r"\s*\([A-Z][A-Z0-9-]{1,9}\)\s*$", "", value).strip()
    value = re.sub(
        r"\s+(?:nano(?:crystals?|particles?)|tablets?|formulations?)$",
        "",
        value,
        flags=re.I,
    ).strip()
    return value or None


def _treatment_readout_ambiguity(row: dict[str, Any], name: str) -> bool:
    support = str(row.get("support_text") or "").casefold()
    endpoint = str(row.get("endpoint_name") or "").casefold()
    subject = name.casefold()
    if subject in endpoint:
        return False
    escaped = re.escape(subject)
    condition_pattern = (
        rf"(?:^with .{{0,40}}{escaped}|{escaped}\s*\([^)]*"
        rf"(?:nm|[µμu]m|mm|mg|g/kg)[^)]*\).{{0,50}}"
        rf"(?:decreas|increas|inhibit|induc)|administration of {escaped}.{{0,80}}"
        rf"(?:decreas|increas|inhibit|induc)|challenge with .{{0,80}}"
        rf"sensitized with {escaped})"
    )
    if re.search(condition_pattern, support):
        return True
    marker_endpoint = any(
        marker in endpoint
        for marker in ("evans blue", "lucifer yellow", "fluorescein", "dye")
    )
    treatment_pattern = (
        rf"(?:{escaped}.{{0,30}}treatment|treat(?:ed|ment).{{0,30}}{escaped}|"
        rf"receiv(?:e|ed|ing).{{0,30}}{escaped})"
    )
    return marker_endpoint and bool(re.search(treatment_pattern, support))


def _normalized_name(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _verbatim_name_span(name: Any, support_text: str) -> str | None:
    name = _text(name)
    if not name:
        return None
    dash = "-‐‑‒–—−"
    pattern = "".join(
        f"[{re.escape(dash)}]"
        if character in dash
        else r"\s+"
        if character.isspace()
        else re.escape(character)
        for character in name
    )
    match = re.search(pattern, support_text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _support_id(
    support_text: str, endpoint_name: str | None, measurement_text: str | None
) -> str:
    return _json_hash(
        {
            "support_text": support_text,
            "endpoint_name": endpoint_name,
            "measurement_text": measurement_text,
        }
    )


def _stratified_sample(
    frame: pd.DataFrame,
    size: int,
    *,
    strata: tuple[str, ...],
    seed: str,
    minimum_per_stratum: int = 1,
) -> pd.DataFrame:
    if size >= len(frame):
        return frame.copy()
    work = frame.copy()
    work["_sample_hash"] = work["id"].astype(str).map(
        lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
    )
    groups = list(work.groupby(list(strata), dropna=False, sort=True))
    counts = [len(group) for _, group in groups]
    quotas = [min(count, minimum_per_stratum) for count in counts]
    if sum(quotas) > size:
        quotas = [0] * len(groups)
    remaining = size - sum(quotas)
    while remaining:
        capacity = [count - quota for count, quota in zip(counts, quotas, strict=True)]
        total_capacity = sum(capacity)
        if not total_capacity:
            break
        raw = [remaining * value / total_capacity for value in capacity]
        additions = [min(capacity[index], int(value)) for index, value in enumerate(raw)]
        if not any(additions):
            order = sorted(
                range(len(groups)),
                key=lambda index: (raw[index], capacity[index]),
                reverse=True,
            )
            additions[order[0]] = 1
        for index, addition in enumerate(additions):
            quotas[index] += addition
            remaining -= addition
    selected = [
        group.sort_values("_sample_hash", kind="stable").head(quota)
        for (_, group), quota in zip(groups, quotas, strict=True)
        if quota
    ]
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values("_sample_hash", kind="stable")
        .drop(columns="_sample_hash")
        .reset_index(drop=True)
    )


def _control_frame(task: str, stage1_path: Path) -> pd.DataFrame:
    fields = (
        "cleaned_record_id",
        "source_id",
        "source_row_number",
        "source_record_id",
        "molecule_name",
        "support_text",
        "canonical_smiles",
        "structure_status",
    )
    available = set(pq.read_schema(stage1_path).names)
    frame = pd.read_parquet(
        stage1_path, columns=[field for field in fields if field in available]
    )
    names = frame["molecule_name"]
    support = frame["support_text"]
    smiles = frame["canonical_smiles"]
    frame = frame.loc[
        frame["structure_status"].eq("resolved")
        & smiles.notna()
        & smiles.astype(str).str.strip().ne("")
        & names.notna()
        & names.astype(str).str.strip().ne("")
        & support.notna()
        & support.astype(str).str.strip().ne("")
    ].copy()
    literal = [
        _high_confidence_control_subject(str(name), str(text))
        for name, text in zip(frame["molecule_name"], frame["support_text"], strict=True)
    ]
    frame = frame.loc[literal].copy()
    frame.insert(0, "id", task + ":control:" + frame["cleaned_record_id"].astype(str))
    frame.insert(1, "task_id", task)
    frame["expected_name"] = frame["molecule_name"].astype(str).str.strip()
    return frame.reset_index(drop=True)


def _high_confidence_control_subject(name: str, support_text: str) -> bool:
    name = name.strip()
    if (
        len(name) < 3
        or name.casefold().startswith("smiles:")
        or not re.search(r"[A-Za-z]", name)
        or re.search(
            r"(?i)\b(?:compar(?:ed|ison)\s+(?:with|to)|greater\s+than|"
            r"higher\s+than|lower\s+than|respectively|metabolite)\b",
            support_text,
        )
    ):
        return False
    escaped = re.escape(name)
    return bool(
        re.search(
            rf"(?i)^(?:the\s+)?{escaped}\s+"
            r"(?:showed|exhibited|demonstrated|displayed|had|has|"
            r"achieved|produced|gave|yielded)\b",
            support_text.strip(),
        )
        or re.search(
            rf"(?i)\b(?:oral\s+|absolute\s+|systemic\s+)?"
            rf"bioavailability\s+of\s+{escaped}\s+(?:was|is|of)\b",
            support_text,
        )
    )


def _name_extraction_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    candidates = pd.concat(
        [_candidate_frame(task, _stage1_path(task))[0] for task in TASK_CONFIG],
        ignore_index=True,
    )
    candidates["candidate_id"] = candidates["id"].astype(str)
    candidates["id"] = (
        candidates["task_id"].astype(str)
        + ":target:"
        + candidates["candidate_id"]
    )
    if args.name_extraction_scope == "full":
        candidates["pilot_kind"] = "target"
        return candidates, len(candidates)
    targets = _stratified_sample(
        candidates,
        args.pilot_target_rows,
        strata=("task_id", "source_id"),
        seed="missing-name-target-v1",
        minimum_per_stratum=5,
    )
    targets["pilot_kind"] = "target"
    controls = pd.concat(
        [_control_frame(task, _stage1_path(task)) for task in TASK_CONFIG],
        ignore_index=True,
    )
    controls = _stratified_sample(
        controls,
        args.pilot_control_rows,
        strata=("task_id", "source_id"),
        seed="missing-name-control-v1",
        minimum_per_stratum=5,
    )
    controls["pilot_kind"] = "control"
    return pd.concat([targets, controls], ignore_index=True, sort=False), len(candidates)


def _read_completed_name_results(
    path: Path,
    default_model: str,
    default_base_url: str,
    default_prompt_version: str,
    default_reasoning_effort: str,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    return {
        str(row["support_id"]): {
            "id": str(row["support_id"]),
            "status": str(row["status"]),
            "molecule_name": _text(row.get("molecule_name")),
            "model": str(row.get("extraction_model") or default_model),
            "base_url": str(row.get("extraction_base_url") or default_base_url),
            "prompt_version": str(
                row.get("extraction_prompt_version") or default_prompt_version
            ),
            "reasoning_effort": str(
                row.get("extraction_reasoning_effort") or default_reasoning_effort
            ),
        }
        for row in frame.to_dict(orient="records")
    }


def _manual_review_gate(sample_path: Path, review_path: Path) -> dict[str, Any]:
    sample_ids = set(_load_jsonl_rows(sample_path, "id"))
    if not review_path.is_file():
        return {
            "status": "pending",
            "review_path": str(review_path),
            "expected_rows": len(sample_ids),
        }
    if review_path.suffix == ".json":
        review = json.loads(review_path.read_text(encoding="utf-8"))
        with sample_path.open("rb") as handle:
            sample_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if review.get("sample_sha256") != sample_sha256:
            raise ValueError("manual review does not match the current sample")
        default = str(review.get("default_decision") or "")
        decisions = {identifier: default for identifier in sample_ids}
        for exception in review.get("exceptions", []):
            decisions[str(exception["id"])] = str(exception["decision"])
    else:
        reviews = _load_jsonl_rows(review_path, "id")
        decisions = {
            key: str(row.get("decision") or "") for key, row in reviews.items()
        }
    allowed = {"correct", "incorrect_status", "wrong_subject"}
    complete = set(decisions) == sample_ids
    valid = set(decisions.values()) <= allowed
    correct = sum(value == "correct" for value in decisions.values())
    wrong_subject = sum(value == "wrong_subject" for value in decisions.values())
    return {
        "status": "pass"
        if complete and valid and correct >= 98 and wrong_subject == 0
        else "fail",
        "review_path": str(review_path),
        "expected_rows": len(sample_ids),
        "reviewed_rows": len(decisions),
        "valid_decisions": valid,
        "correct_rows": correct,
        "wrong_subject_rows": wrong_subject,
    }


def _load_jsonl_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row[key])] = row
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _run_name_extraction(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir) / "name_extraction"
    scope_dir = output_root / args.name_extraction_scope
    pilot_dir = output_root / "pilot"
    configured_base_urls = args.name_extraction_base_url
    if not isinstance(configured_base_urls, list):
        configured_base_urls = [configured_base_urls]
    base_urls = [
        (base_url or "https://api.openai.com/v1").rstrip("/")
        for base_url in configured_base_urls
    ]
    gate_dir = Path(args.name_extraction_gate_dir or pilot_dir)
    manual_review_path = Path(
        args.pilot_manual_review or gate_dir / "manual_review.json"
    )
    if args.name_extraction_scope == "full":
        pilot_manifest_path = gate_dir / "manifest.json"
        if not pilot_manifest_path.is_file():
            raise ValueError("the 1,000-row pilot must complete before the full run")
        pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
        if pilot_manifest.get("automatic_gate_status") != "pass":
            raise ValueError("the pilot automatic gate did not pass")
        manual_gate = _manual_review_gate(
            gate_dir / "manual_review_sample.jsonl", manual_review_path
        )
        if manual_gate["status"] != "pass":
            raise ValueError(f"the pilot manual gate did not pass: {manual_gate}")
    else:
        manual_gate = {"status": "pending"}

    rows, all_candidate_rows = _name_extraction_rows(args)
    if len(rows) == 0:
        raise ValueError("no rows are eligible for molecule-name extraction")
    rows = rows.copy()
    rows["support_text"] = rows["support_text"].astype(str)
    for field in ("endpoint_name", "measurement_text"):
        rows[field] = rows[field].map(_text)
    rows["support_id"] = [
        _support_id(support, endpoint, measurement)
        for support, endpoint, measurement in zip(
            rows["support_text"],
            rows["endpoint_name"],
            rows["measurement_text"],
            strict=True,
        )
    ]
    prompt_fields = ["support_text", "endpoint_name", "measurement_text"]
    unique_support = rows[["support_id", *prompt_fields]].drop_duplicates()
    if unique_support["support_id"].duplicated().any():
        raise ValueError("name-extraction prompt hash collision detected")

    reused: dict[str, dict[str, Any]] = {}
    legacy_results: dict[str, dict[str, Any]] = {}
    if args.name_extraction_scope == "full":
        reused = _read_completed_name_results(
            gate_dir / "unique_support_extractions.parquet",
            args.name_extraction_model,
            base_urls[0],
            NAME_EXTRACTION_PROMPT_VERSION,
            NAME_EXTRACTION_REASONING_EFFORT,
        )
        legacy_results = _read_completed_name_results(
            pilot_dir / "unique_support_extractions.parquet",
            NAME_EXTRACTION_MODEL,
            "https://api.openai.com/v1",
            "missing_molecule_name_extraction.v2",
            "none",
        )
    log_path = scope_dir / "extraction_batches.jsonl"
    for entry in _load_jsonl(log_path).values():
        if args.name_extraction_scope == "pilot":
            continue
        destination = (
            reused
            if entry["prompt_version"] == NAME_EXTRACTION_PROMPT_VERSION
            else legacy_results
        )
        for item in entry["items"]:
            destination[str(item["id"])] = {
                **item,
                "model": str(entry["model"]),
                "base_url": str(
                    entry.get("request_namespace")
                    or (
                        "https://api.openai.com/v1"
                        if entry["model"] == NAME_EXTRACTION_MODEL
                        else "unknown"
                    )
                ),
                "prompt_version": str(entry["prompt_version"]),
                "reasoning_effort": str(
                    entry.get("request_variant")
                    or (
                        NAME_EXTRACTION_REASONING_EFFORT
                        if entry["prompt_version"] == NAME_EXTRACTION_PROMPT_VERSION
                        else "none"
                    )
                ).rsplit("reasoning_effort=", 1)[-1],
            }
    legacy_contexts_reused = 0
    legacy_multi_context_support_texts_rerun = 0
    if legacy_results:
        context_counts = unique_support.groupby("support_text").size()
        singleton_support = set(context_counts[context_counts.eq(1)].index)
        for row in unique_support.itertuples(index=False):
            legacy_id = hashlib.sha256(row.support_text.encode("utf-8")).hexdigest()
            if row.support_text in singleton_support and legacy_id in legacy_results:
                context_id = str(row.support_id)
                if context_id not in reused:
                    reused[context_id] = {**legacy_results[legacy_id], "id": context_id}
                    legacy_contexts_reused += 1
        legacy_multi_context_support_texts_rerun = sum(
            hashlib.sha256(text.encode("utf-8")).hexdigest() in legacy_results
            for text in context_counts[context_counts.gt(1)].index
        )
    call_rows = [
        {
            "id": str(row.support_id),
            "support_text": str(row.support_text),
            "endpoint_name": _text(row.endpoint_name),
            "measurement_text": _text(row.measurement_text),
        }
        for row in unique_support.itertuples(index=False)
        if str(row.support_id) not in reused
    ]
    results = dict(reused)
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    request_count = 0
    if call_rows:
        endpoint_clients = [
            (
                openai_client(
                    base_url=base_url,
                    keys_path=args.keys_path,
                    unauthenticated=args.name_extraction_no_auth,
                    max_connections=args.workers,
                    timeout_s=NAME_EXTRACTION_TIMEOUT_S,
                ),
                base_url,
            )
            for base_url in base_urls
        ]
        completed_before = len(_load_jsonl(log_path))
        called, usage, request_count = _run_batches(
            call_rows,
            batch_size=args.extraction_batch_size,
            workers=args.workers,
            model=args.name_extraction_model,
            prompt_version=NAME_EXTRACTION_PROMPT_VERSION,
            log_path=log_path,
            call=_call_name_extraction,
            client=endpoint_clients[0][0],
            request_namespace=base_urls[0],
            request_variant=(
                f"reasoning_effort={NAME_EXTRACTION_REASONING_EFFORT}"
            ),
            endpoint_clients=endpoint_clients,
        )
        for item in called.values():
            item["model"] = args.name_extraction_model
            item["base_url"] = item.pop("_request_namespace", base_urls[0])
            item["prompt_version"] = NAME_EXTRACTION_PROMPT_VERSION
            item["reasoning_effort"] = NAME_EXTRACTION_REASONING_EFFORT
        new_request_count = len(_load_jsonl(log_path)) - completed_before
        results.update(called)
    else:
        new_request_count = 0
    required_support_ids = set(unique_support["support_id"].astype(str))
    results = {key: value for key, value in results.items() if key in required_support_ids}
    if set(results) != required_support_ids:
        raise ValueError("completed extraction does not cover every unique support text")

    unique_output = unique_support.copy()
    unique_output["status"] = unique_output["support_id"].map(
        lambda key: results[str(key)]["status"]
    )
    unique_output["raw_molecule_name"] = unique_output["support_id"].map(
        lambda key: results[str(key)].get("molecule_name")
    )
    unique_output["extraction_model"] = unique_output["support_id"].map(
        lambda key: results[str(key)]["model"]
    )
    unique_output["extraction_base_url"] = unique_output["support_id"].map(
        lambda key: results[str(key)]["base_url"]
    )
    unique_output["extraction_prompt_version"] = unique_output["support_id"].map(
        lambda key: results[str(key)]["prompt_version"]
    )
    unique_output["extraction_reasoning_effort"] = unique_output["support_id"].map(
        lambda key: results[str(key)]["reasoning_effort"]
    )
    unique_output["molecule_name"] = [
        _verbatim_name_span(name, support)
        if status == "explicit"
        else None
        for status, name, support in zip(
            unique_output["status"],
            unique_output["raw_molecule_name"],
            unique_output["support_text"],
            strict=True,
        )
    ]
    invalid_explicit = unique_output["status"].eq("explicit") & unique_output[
        "molecule_name"
    ].isna()
    invalid_explicit_downgrades = int(invalid_explicit.sum())
    unique_output.loc[invalid_explicit, "status"] = "absent"
    typography_repairs = int(
        (
            unique_output["molecule_name"].notna()
            & (
                unique_output["raw_molecule_name"].fillna("")
                != unique_output["molecule_name"].fillna("")
            )
        ).sum()
    )
    _write_parquet(scope_dir / "unique_support_extractions.parquet", unique_output)

    row_output = rows.merge(
        unique_output.drop(columns=prompt_fields),
        on="support_id",
        how="left",
        validate="many_to_one",
    )
    row_output = row_output.rename(
        columns={"status": "extraction_status", "molecule_name_y": "extracted_molecule_name"}
    )
    if "molecule_name_x" in row_output:
        row_output = row_output.rename(columns={"molecule_name_x": "stage1_molecule_name"})
    elif "molecule_name" in row_output:
        row_output = row_output.rename(columns={"molecule_name": "extracted_molecule_name"})
    _write_parquet(scope_dir / "row_extractions.parquet", row_output)

    explicit = row_output["extraction_status"].eq("explicit")
    names = row_output["extracted_molecule_name"]
    verbatim = [
        not pd.isna(name) and bool(name) and str(name) in str(text)
        for name, text in zip(names, row_output["support_text"], strict=True)
    ]
    status_name_consistent = (explicit & names.notna()) | (~explicit & names.isna())
    cardinality_complete = row_output["extraction_status"].notna().all()
    explicit_verbatim = bool(pd.Series(verbatim, index=row_output.index)[explicit].all())
    status_name_consistent = bool(status_name_consistent.all())

    controls = row_output.loc[row_output["pilot_kind"].eq("control")]
    control_accuracy = None
    if len(controls):
        control_correct = (
            controls["extraction_status"].eq("explicit")
            & controls.apply(
                lambda row: _normalized_name(row["extracted_molecule_name"])
                == _normalized_name(row["expected_name"]),
                axis=1,
            )
        )
        control_accuracy = float(control_correct.mean())
    automatic_pass = (
        cardinality_complete
        and explicit_verbatim
        and status_name_consistent
        and (control_accuracy is None or control_accuracy >= 0.95)
    )

    if args.name_extraction_scope == "pilot":
        review_candidates = row_output.loc[row_output["pilot_kind"].eq("target")]
        review_sample = _stratified_sample(
            review_candidates,
            args.pilot_manual_review_rows,
            strata=("task_id", "source_id"),
            seed="missing-name-manual-review-v1",
        )
        _write_jsonl(
            scope_dir / "manual_review_sample.jsonl",
            [
                {
                    "id": str(row["id"]),
                    "task_id": str(row["task_id"]),
                    "source_id": str(row["source_id"]),
                    "source_row_number": int(row["source_row_number"]),
                    "endpoint_name": _text(row["endpoint_name"]),
                    "measurement_text": _text(row["measurement_text"]),
                    "support_text": str(row["support_text"]),
                    "extraction_status": str(row["extraction_status"]),
                    "extracted_molecule_name": _text(row["extracted_molecule_name"]),
                }
                for row in review_sample.to_dict(orient="records")
            ],
        )
        manual_gate = _manual_review_gate(
            scope_dir / "manual_review_sample.jsonl", manual_review_path
        )

    manifest = {
        "name_extraction_version": NAME_EXTRACTION_VERSION,
        "scope": args.name_extraction_scope,
        "quality_gate_dir": str(gate_dir),
        "model": args.name_extraction_model,
        "base_url": base_urls[0] if len(base_urls) == 1 else base_urls,
        "reasoning_effort": NAME_EXTRACTION_REASONING_EFFORT,
        "request_timeout_s": NAME_EXTRACTION_TIMEOUT_S,
        "store": False,
        "prompt_fields": ["id", *prompt_fields],
        "all_candidate_rows": all_candidate_rows,
        "selected_rows": len(row_output),
        "target_rows": int(row_output["pilot_kind"].eq("target").sum()),
        "control_rows": int(row_output["pilot_kind"].eq("control").sum()),
        "unique_support_texts": int(rows["support_text"].nunique()),
        "unique_prompt_contexts": len(unique_output),
        "reused_prompt_contexts": len(required_support_ids & set(reused)),
        "legacy_singleton_contexts_reused": legacy_contexts_reused,
        "legacy_multi_context_support_texts_rerun": (
            legacy_multi_context_support_texts_rerun
        ),
        "model_counts": {
            str(key): int(value)
            for key, value in unique_output["extraction_model"]
            .value_counts()
            .sort_index()
            .items()
        },
        "provider_counts": {
            f"{model} @ {base_url}": int(value)
            for (model, base_url), value in unique_output.groupby(
                ["extraction_model", "extraction_base_url"]
            ).size().sort_index().items()
        },
        "prompt_version_counts": {
            str(key): int(value)
            for key, value in unique_output["extraction_prompt_version"]
            .value_counts()
            .sort_index()
            .items()
        },
        "reasoning_effort_counts": {
            str(key): int(value)
            for key, value in unique_output["extraction_reasoning_effort"]
            .value_counts()
            .sort_index()
            .items()
        },
        "requests_covering_scope": request_count,
        "new_requests_submitted": new_request_count,
        "checkpoint_requests_reused": request_count - new_request_count,
        "verbatim_typography_repairs": typography_repairs,
        "invalid_explicit_downgrades": invalid_explicit_downgrades,
        "usage": usage,
        "automatic_gates": {
            "response_cardinality_complete": bool(cardinality_complete),
            "explicit_names_verbatim": explicit_verbatim,
            "names_null_unless_explicit": status_name_consistent,
            "hidden_control_normalized_exact_accuracy": control_accuracy,
            "hidden_control_minimum_accuracy": 0.95,
        },
        "automatic_gate_status": "pass" if automatic_pass else "fail",
        "manual_gate": manual_gate,
        "writes_stage1_or_reviewed_ledgers": False,
    }
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _run_discovery(args: argparse.Namespace, stage1_path: Path) -> None:
    frame, source_ids = _candidate_frame(args.task, stage1_path)
    output_dir = Path(args.output_dir) / args.task / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidates.parquet"
    temporary = candidate_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, candidate_path)
    with candidate_path.open("rb") as handle:
        candidate_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "audit_version": AUDIT_VERSION,
        "mode": "candidate_discovery",
        "task_id": args.task,
        "stage1_path": str(stage1_path),
        "stage1_rows": pq.ParquetFile(stage1_path).metadata.num_rows,
        "candidate_definition": {
            "structure_status": "resolved",
            "canonical_smiles_nonempty": True,
            "stage1_molecule_name_blank": True,
            "support_text_nonempty": True,
        },
        "source_ids": source_ids,
        "candidate_rows": len(frame),
        "source_counts": {
            str(key): int(value)
            for key, value in frame["source_id"].value_counts().sort_index().items()
        },
        "structure_status_counts": {
            str(key): int(value)
            for key, value in frame["structure_status"]
            .fillna("missing")
            .value_counts()
            .sort_index()
            .items()
        },
        "future_extraction_batches_at_configured_size": (
            frame["support_text"].nunique() + args.extraction_batch_size - 1
        )
        // args.extraction_batch_size,
        "unique_support_texts": int(frame["support_text"].nunique()),
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha256,
        "external_requests": 0,
        "writes_stage1_or_reviewed_ledgers": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _run_name_smiles_comparison(args: argparse.Namespace) -> None:
    extraction_path = Path(
        args.name_extractions
        or Path(args.output_dir) / "name_extraction/full/row_extractions.parquet"
    )
    extraction_manifest_path = extraction_path.parent / "manifest.json"
    if not extraction_path.is_file() or not extraction_manifest_path.is_file():
        raise ValueError("completed name-extraction Parquet and manifest are required")
    extraction_manifest = json.loads(
        extraction_manifest_path.read_text(encoding="utf-8")
    )
    if extraction_manifest.get("automatic_gate_status") != "pass":
        raise ValueError("name extraction did not pass its automatic gate")
    rows = pd.read_parquet(extraction_path)
    if len(rows) != int(extraction_manifest["selected_rows"]):
        raise ValueError("name-extraction row count does not match its manifest")
    explicit = rows.loc[
        rows["extraction_status"].eq("explicit")
        & rows["extracted_molecule_name"].notna()
        & rows["extracted_molecule_name"].astype(str).str.strip().ne("")
    ].copy()

    lookup_queries: dict[str, str] = {}
    for row in explicit.to_dict(orient="records"):
        name = str(row["extracted_molecule_name"])
        query = _pubchem_lookup_name(name)
        if query and not _treatment_readout_ambiguity(row, name):
            lookup_queries.setdefault(" ".join(query.casefold().split()), query)
    cache = PubChemCache(Path(args.pubchem_cache))
    try:
        resolved: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.pubchem_workers) as executor:
            futures = {
                executor.submit(cache.resolve, query, args.pubchem_timeout_s): key
                for key, query in sorted(lookup_queries.items())
            }
            for completed, future in enumerate(as_completed(futures), 1):
                resolved[futures[future]] = future.result()
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        f"PubChem lookups complete: {completed:,}/{len(futures):,}",
                        flush=True,
                    )
    finally:
        cache.close()

    compared: list[dict[str, Any]] = []
    for row in explicit.to_dict(orient="records"):
        name = str(row["extracted_molecule_name"])
        query = _pubchem_lookup_name(name)
        if _treatment_readout_ambiguity(row, name):
            pubchem = {"status": "treatment_readout_ambiguous", "query_name": name}
        elif query:
            pubchem = resolved[" ".join(query.casefold().split())]
        else:
            pubchem = {"status": "ambiguous_name", "query_name": name}
        compared.append(
            _compare(
                row,
                {
                    "status": "explicit",
                    "molecule_name": name,
                    "evidence_span": name,
                },
                pubchem,
            )
        )

    comparison_frame = pd.DataFrame(compared)
    conflict_frame = comparison_frame.loc[
        comparison_frame["identity_comparison"].isin(
            {"different_parent", "stored_structure_invalid"}
        )
    ].copy()
    output_dir = Path(args.output_dir) / "name_smiles_comparison/v1"
    comparison_path = output_dir / "comparisons.parquet"
    conflict_path = output_dir / "conflict_candidates.parquet"
    _write_parquet(comparison_path, comparison_frame)
    _write_parquet(conflict_path, conflict_frame)
    hashes: dict[str, str] = {}
    for label, path in {
        "name_extractions": extraction_path,
        "comparisons": comparison_path,
        "conflict_candidates": conflict_path,
    }.items():
        with path.open("rb") as handle:
            hashes[label] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "comparison_version": NAME_SMILES_COMPARISON_VERSION,
        "name_extraction_path": str(extraction_path),
        "pubchem_cache": str(args.pubchem_cache),
        "pubchem_workers": args.pubchem_workers,
        "pubchem_max_requests_per_second": 5,
        "input_rows": len(rows),
        "explicit_name_rows": len(explicit),
        "unique_pubchem_queries": len(lookup_queries),
        "lookup_status_counts": _counts(compared, "pubchem_status"),
        "comparison_counts": _counts(compared, "identity_comparison"),
        "conflict_candidate_rows": len(conflict_frame),
        "conflict_classes": ["different_parent", "stored_structure_invalid"],
        "artifacts": {
            "comparisons": str(comparison_path),
            "conflict_candidates": str(conflict_path),
        },
        "sha256": hashes,
        "llm_requests": 0,
        "writes_stage1_or_reviewed_ledgers": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> None:
    if getattr(args, "find_name_smiles_conflicts", False):
        if args.task != "all":
            raise ValueError("--find-name-smiles-conflicts requires --task all")
        _run_name_smiles_comparison(args)
        return
    if getattr(args, "name_extraction_scope", None):
        _run_name_extraction(args)
        return
    task = args.task
    if task == "all":
        raise ValueError("--task all is only valid with --name-extraction-scope")
    config = TASK_CONFIG[task]
    stage1_path = Path(args.stage1 or _stage1_path(task))
    if args.discover_only:
        _run_discovery(args, stage1_path)
        return
    sample_path = Path(args.sample or config["sample"])
    output_dir = Path(args.output_dir) / task / "pilot"
    rows = _pilot_rows(task, stage1_path, sample_path)
    if not rows:
        raise ValueError(f"no eligible pilot rows for {task}")
    client = openai_client(keys_path=args.keys_path)
    extractions, extraction_usage, extraction_requests = _run_batches(
        rows,
        batch_size=args.extraction_batch_size,
        workers=args.workers,
        model=args.model,
        prompt_version=PROMPT_VERSION,
        log_path=output_dir / "extraction_batches.jsonl",
        call=_call_extraction,
        client=client,
    )
    cache = PubChemCache(Path(args.pubchem_cache))
    try:
        compared: list[dict[str, Any]] = []
        for row in rows:
            extraction = extractions[row["id"]]
            if extraction["status"] != "single_explicit_subject" or not extraction.get("molecule_name"):
                pubchem = {"status": "not_attempted"}
            else:
                lookup_name = _pubchem_lookup_name(extraction["molecule_name"])
                if _treatment_readout_ambiguity(row, extraction["molecule_name"]):
                    pubchem = {
                        "status": "treatment_readout_ambiguous",
                        "query_name": extraction["molecule_name"],
                    }
                elif lookup_name:
                    pubchem = cache.resolve(lookup_name, args.pubchem_timeout_s)
                else:
                    pubchem = {
                        "status": "ambiguous_name",
                        "query_name": extraction["molecule_name"],
                    }
            compared.append(_compare(row, extraction, pubchem))
    finally:
        cache.close()
    review_input = [
        row
        for row in compared
        if row["identity_comparison"] in {"different_parent", "stored_structure_invalid"}
    ]
    reviews: dict[str, dict[str, Any]] = {}
    review_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    review_requests = 0
    if review_input:
        reviews, review_usage, review_requests = _run_batches(
            review_input,
            batch_size=args.review_batch_size,
            workers=args.workers,
            model=args.model,
            prompt_version=PROMPT_VERSION + ".review",
            log_path=output_dir / "review_batches.jsonl",
            call=_call_review,
            client=client,
        )
    proposals: list[dict[str, Any]] = []
    for row in review_input:
        review = reviews[row["id"]]
        row.update(
            {
                "review_decision": review["decision"],
                "review_evidence_span": review.get("evidence_span"),
                "review_rationale": review["rationale"],
            }
        )
        if review["decision"] == "confirmed_mismatch":
            proposals.append(
                {
                    "proposal_id": _json_hash(
                        {"audit_version": AUDIT_VERSION, "id": row["id"], "after": row["pubchem_smiles"]}
                    ),
                    "proposal_type": "smiles_repair",
                    "status": "proposal_only",
                    "task_id": task,
                    "cleaned_record_id": row["id"],
                    "source_id": row["source_id"],
                    "source_sha256": row["source_sha256"],
                    "source_row_number": row["source_row_number"],
                    "source_record_id": row["source_record_id"],
                    "before": {"smiles": row["source_smiles"]},
                    "after": {"smiles": row["pubchem_smiles"]},
                    "subject_name": row["extracted_molecule_name"],
                    "pubchem_cid": row["pubchem_cid"],
                    "evidence_span": review.get("evidence_span"),
                    "rationale": review["rationale"],
                }
            )
    _write_jsonl(output_dir / "row_audit.jsonl", compared)
    _write_jsonl(output_dir / "proposals.jsonl", proposals)
    manual_mismatches = {
        row["id"]
        for row in compared
        if row["manual_classification"].startswith("confirmed_mismatch")
    }
    proposed_ids = {row["cleaned_record_id"] for row in proposals}
    high_match_ids = {
        row["id"] for row in compared if row["manual_classification"] == "high_confidence_match"
    }
    manifest = {
        "audit_version": AUDIT_VERSION,
        "mode": "pilot",
        "task_id": task,
        "model": args.model,
        "reasoning_effort": "none",
        "store": False,
        "stage1_path": str(stage1_path),
        "sample_path": str(sample_path),
        "eligible_rows": len(rows),
        "extraction_requests": extraction_requests,
        "review_requests": review_requests,
        "usage": {
            key: extraction_usage[key] + review_usage[key]
            for key in extraction_usage
        },
        "comparison_counts": _counts(compared, "identity_comparison"),
        "review_counts": _counts(compared, "review_decision"),
        "proposal_count": len(proposals),
        "acceptance": {
            "manual_confirmed_mismatches": len(manual_mismatches),
            "manual_repaired_mismatches_identity_verified": sum(
                row["id"] in manual_mismatches
                and row["identity_comparison"] in {"exact_identity", "same_parent_or_form"}
                for row in compared
            ),
            "manual_repaired_mismatches_proposed": len(manual_mismatches & proposed_ids),
            "high_confidence_matches_proposed": len(high_match_ids & proposed_ids),
            "response_cardinality_complete": len(extractions) == len(rows),
        },
        "writes_stage1_or_reviewed_ledgers": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "not_reviewed")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*sorted(TASK_CONFIG), "all"], required=True)
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/chembl_tool/smiles_identity_audit_v2",
    )
    parser.add_argument(
        "--pubchem-cache",
        type=Path,
        default=REPO_ROOT
        / "outputs/chembl_tool/smiles_identity_audit_v2/pubchem_cache.sqlite",
    )
    parser.add_argument("--keys-path", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--name-extraction-model", default=NAME_EXTRACTION_MODEL)
    parser.add_argument(
        "--name-extraction-base-url",
        action="append",
        help="repeat to distribute extraction continuously across endpoints",
    )
    parser.add_argument("--name-extraction-no-auth", action="store_true")
    parser.add_argument("--name-extraction-gate-dir", type=Path)
    parser.add_argument(
        "--name-extraction-scope",
        choices=("pilot", "full"),
        help="extract missing molecule names only; never edit Stage 1",
    )
    parser.add_argument("--pilot-target-rows", type=int, default=800)
    parser.add_argument("--pilot-control-rows", type=int, default=200)
    parser.add_argument("--pilot-manual-review-rows", type=int, default=100)
    parser.add_argument("--pilot-manual-review", type=Path)
    parser.add_argument("--name-extractions", type=Path)
    parser.add_argument(
        "--find-name-smiles-conflicts",
        action="store_true",
        help="compare completed extracted names with stored structures via PubChem",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="write the complete deterministic candidate inventory without external calls",
    )
    parser.add_argument("--extraction-batch-size", type=int, default=50)
    parser.add_argument("--review-batch-size", type=int, default=20)
    parser.add_argument("--pubchem-timeout-s", type=float, default=30.0)
    parser.add_argument("--pubchem-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if min(
        args.workers,
        args.extraction_batch_size,
        args.review_batch_size,
        args.pilot_target_rows,
        args.pilot_control_rows,
        args.pilot_manual_review_rows,
        args.pubchem_workers,
    ) < 1:
        parser.error("workers and batch sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
