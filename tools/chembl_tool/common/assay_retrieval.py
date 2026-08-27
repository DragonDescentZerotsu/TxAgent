"""Relevance-ranked flat assay retrieval for normalized Starling evidence.

The assay rank is used only to choose a cumulative prefix.  It is not copied
into evidence rows or the LLM-visible group payload.  Within every selected
assay, retrieval keeps the top-k structurally similar molecules above the
frozen Morgan threshold, then merges repeated molecules across assays into one
flat reasoning branch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterable

import pandas as pd

from tools.chembl_tool.common.evidence_contract import (
    ASSAY_COMPACT_PROMPT_PROFILE,
    ASSAY_COMPACT_V2_PROMPT_PROFILE,
    ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
    ASSAY_RAW_CARD_PROMPT_PROFILE,
    attach_minimal_evidence,
)
from tools.chembl_tool.common.experiment_retrieval import flatten_retrieval_groups
from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.reasoning_payload import (
    EXTERNAL_CONDITION_RENDERER_VERSION,
    attach_external_condition,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
)
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    retrieve_neighbors,
)
from tools.chembl_tool.common.starling.assay_catalog import assay_id, assay_unit

INDEX_VERSION = "starling_assay_ranked_morgan.v1"
RAW_CARD_INDEX_VERSION = "starling_assay_ranked_morgan.raw_v3"
MECHANISM_TAGGED_INDEX_VERSION = "starling_assay_ranked_morgan.mechanism_tagged_v4"
FAMILY_MOLECULE_VIEW_VERSION = "starling_family_molecule_prefix_view.v1"
FLAT_GROUP_ID = "Flat.assay_ranked_evidence"
FAMILY_MOLECULE_GROUP_PREFIX = "Flat.progressive_family_level_"
DEFAULT_RETRIEVAL_PREFIXES = (10, 100, 400)
DIRECT_ONLY_HELDOUT_FILTERED = "direct_only_heldout_filtered"
RECORD_CARD_SELECTION_VERSION = "assay_spanning_even_then_round_robin.v1"
MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION = "mechanism_diverse_informative.v1"
RECORD_CARD_SELECTIONS = (
    RECORD_CARD_SELECTION_VERSION,
    MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION,
)


def geometric_assay_prefixes(
    total_assays: int,
    *,
    start: int = 5,
    multiplier: int = 4,
) -> tuple[int, ...]:
    """Return a geometric prefix schedule whose final point is the full catalog."""
    if total_assays <= 0:
        raise ValueError("total_assays must be positive")
    if start <= 0:
        raise ValueError("start must be positive")
    if multiplier <= 1:
        raise ValueError("multiplier must be greater than one")
    prefixes: list[int] = []
    value = min(start, total_assays)
    while value < total_assays:
        prefixes.append(value)
        value *= multiplier
    prefixes.append(total_assays)
    return tuple(prefixes)


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _unique_text(values: Iterable[Any], *, limit: int) -> list[str]:
    output: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in output:
            output.append(text)
            if len(output) >= limit:
                break
    return output


def _truncate_text(value: Any, limit: int) -> str:
    text = _clean(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_counts(values: Iterable[Any], *, limit: int = 8) -> str:
    counts = Counter(_clean(value) or "unspecified" for value in values)
    return "; ".join(
        f"{value} ({count})" if count > 1 else value
        for value, count in counts.most_common(limit)
    )


def _parent_key(smiles: str) -> str:
    identity = normalize_molecule_identity(smiles)
    return identity.parent_inchi_key or identity.parent_smiles


def _assay_selection_rank(row: dict[str, Any]) -> int:
    value = row.get("selection_rank", row.get("relevance_rank"))
    if value in (None, ""):
        raise ValueError("assay catalog row lacks selection_rank or relevance_rank")
    return int(value)


def _load_allowed_parent_keys(path: Path, smiles_field: str) -> tuple[set[str], int]:
    rows = read_jsonl(path)
    keys = {_parent_key(str(row.get(smiles_field) or "")) for row in rows}
    keys.discard("")
    if not keys:
        raise ValueError(
            f"{path} contains no usable molecules in field {smiles_field!r}"
        )
    return keys, len(rows)


def _filter_heldout_direct_records(
    records: pd.DataFrame,
    *,
    heldout_molecules_path: Path,
    heldout_smiles_field: str,
    filter_source_id: str,
    filter_scope_field: str = "",
    filter_scope_value: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove held-out parents only from the benchmark-defining direct source."""
    if bool(filter_scope_field) != bool(filter_scope_value):
        raise ValueError(
            "filter_scope_field and filter_scope_value must be set together"
        )
    if not filter_source_id and not filter_scope_field:
        raise ValueError(
            "filter_source_id or filter_scope_field is required with "
            "heldout_molecules_path"
        )

    heldout_keys, n_heldout_rows = _load_allowed_parent_keys(
        heldout_molecules_path,
        heldout_smiles_field,
    )
    smiles_to_key = {
        smiles: _parent_key(smiles)
        for smiles in records["canonical_smiles"].dropna().astype(str).unique()
    }
    record_keys = records["canonical_smiles"].astype(str).map(smiles_to_key)
    direct_scope = pd.Series(True, index=records.index)
    if filter_source_id:
        direct_scope &= records["source_id"].astype(str).eq(filter_source_id)
    if filter_scope_field:
        direct_scope &= records[filter_scope_field].astype(str).eq(filter_scope_value)
    heldout = record_keys.isin(heldout_keys)
    excluded = direct_scope & heldout
    retained = records.loc[~excluded].copy()

    retained_keys = retained["canonical_smiles"].astype(str).map(smiles_to_key)
    retained_direct_scope = pd.Series(True, index=retained.index)
    if filter_source_id:
        retained_direct_scope &= retained["source_id"].astype(str).eq(filter_source_id)
    if filter_scope_field:
        retained_direct_scope &= (
            retained[filter_scope_field].astype(str).eq(filter_scope_value)
        )
    overlap_after = retained_direct_scope & retained_keys.isin(heldout_keys)
    if overlap_after.any():
        raise AssertionError(
            f"{int(overlap_after.sum())} held-out direct records remain after filtering"
        )

    excluded_keys = set(record_keys.loc[excluded])
    retained_heldout_nondirect = heldout & ~direct_scope
    return retained, {
        "reference_pool": DIRECT_ONLY_HELDOUT_FILTERED,
        "direct_only_heldout_filtered": True,
        "heldout_molecules": str(heldout_molecules_path.resolve()),
        "heldout_smiles_field": heldout_smiles_field,
        "n_heldout_rows": n_heldout_rows,
        "n_heldout_parent_identities": len(heldout_keys),
        "filter_source_id": filter_source_id,
        "filter_scope_field": filter_scope_field,
        "filter_scope_value": filter_scope_value,
        "n_records_before_heldout_filter": len(records),
        "n_direct_heldout_records_excluded": int(excluded.sum()),
        "n_matched_heldout_parent_identities": len(excluded_keys),
        "n_records_after_heldout_filter": len(retained),
        "n_heldout_nondirect_records_retained": int(retained_heldout_nondirect.sum()),
        "n_direct_heldout_records_after_filter": 0,
    }


def _source_family_map(
    assay_metadata: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not assay_metadata:
        return {}
    return {
        str(item.get("source_group_id") or ""): dict(item)
        for item in assay_metadata.get("source_families") or []
        if isinstance(item, dict) and str(item.get("source_group_id") or "")
    }


def _representative_records(
    group: pd.DataFrame,
    *,
    limit: int,
    assay_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ordered = group.sort_values(
        ["confidence", "canonical_record_id"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    )
    examples = []
    seen: set[tuple[str, str, str, str]] = set()
    family_by_group = _source_family_map(assay_metadata)
    for row in ordered.to_dict(orient="records"):
        family = family_by_group.get(str(row.get("group_id") or ""), {})
        example = {
            "endpoint_type": _clean(row.get("canonical_endpoint_name")),
            "reported_value": _clean(row.get("canonical_measurement_text")),
            "reported_units": _clean(row.get("canonical_unit_text")),
            "assay_context": _clean(row.get("canonical_assay_context")),
            "species_context": _clean(row.get("canonical_species_context")),
            "qualifying_conditions": _clean(row.get("qualifying_conditions")),
            "support_text": _clean(row.get("support_text")),
            "evidence_family": _clean(family.get("endpoint_group")),
            "evidence_family_level": family.get("level", ""),
        }
        key = (
            example["endpoint_type"],
            example["reported_value"],
            example["reported_units"],
            example["support_text"],
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
    if not family_by_group:
        return examples[:limit]

    selected = []
    selected_families: set[str] = set()
    for example in examples:
        family = str(example.get("evidence_family") or "")
        if family and family not in selected_families:
            selected.append(example)
            selected_families.add(family)
            if len(selected) >= limit:
                return selected
    selected_keys = {id(example) for example in selected}
    selected.extend(example for example in examples if id(example) not in selected_keys)
    return selected[:limit]


def _aggregate_assay_molecule(
    task: str,
    assay_context: str,
    stable_assay_id: str,
    group: pd.DataFrame,
    *,
    max_record_examples: int,
    max_support_text_chars: int,
    assay_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    molecule_id = str(group["molecule_id"].iloc[0])
    canonical_smiles = str(group["canonical_smiles"].iloc[0])
    examples = _representative_records(
        group,
        limit=max_record_examples,
        assay_metadata=assay_metadata,
    )
    support_texts = [
        _truncate_text(text, max_support_text_chars)
        for text in _unique_text(
            (example["support_text"] for example in examples),
            limit=max_record_examples,
        )
    ]
    # Preserve the frozen top-1280 replay serialization so extending the index
    # does not invalidate completed LLM results.  support_text is intentionally
    # retained here as well as in source_support_texts for artifact parity.
    prompt_examples = [dict(example) for example in examples]
    source_names = _unique_text(group["source_name"].tolist(), limit=8)
    molecule_names = _unique_text(group["molecule_name"].tolist(), limit=8)
    species = _unique_text(group["canonical_species_context"].tolist(), limit=8)
    qualifying_conditions = _unique_text(
        group["qualifying_conditions"].tolist(), limit=8
    )
    endpoint_summary = _compact_counts(group["canonical_endpoint_name"].tolist())
    measurement_summary = _compact_counts(group["canonical_measurement_text"].tolist())
    units_summary = _compact_counts(group["canonical_unit_text"].tolist())
    row = {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": canonical_smiles,
        "assay_chembl_id": stable_assay_id,
        "assay_tier": "Assay",
        "endpoint_group": "assay_ranked_evidence",
        "group_id": f"Assay.{stable_assay_id}",
        "standard_type": endpoint_summary,
        "standard_relation": "",
        "standard_value": measurement_summary,
        "standard_units": units_summary,
        "assay_description": "\n\n---\n\n".join(support_texts),
        "activity_comment": (
            f"Starling assay-level summary over {len(group)} records; "
            f"canonical_assay_context={assay_context}"
        ),
        "target_pref_name": assay_context,
        "organism": "; ".join(species),
        "confidence_score": float(group["confidence"].median())
        if group["confidence"].notna().any()
        else "",
        "evidence_source": " + ".join(source_names) or "Starling",
        "evidence_role": "unspecified",
        "transferability": "not_assessed",
        "source_molecule_names": molecule_names,
        "source_record_count": int(len(group)),
        "source_record_examples": prompt_examples,
        "source_support_texts": support_texts,
        "evidence_scope": {
            "assay_context": [assay_context],
            "species_context": species,
            "qualifying_conditions": qualifying_conditions,
        },
        "uncertainty": (
            ["some_source_rows_missing_support_text"]
            if group["support_text"].map(_clean).eq("").any()
            else []
        ),
        "assay_retrieval": {
            "task": task,
            "assay_id": stable_assay_id,
            "assay_context": assay_context,
            **(
                {
                    "first_level": assay_metadata.get("first_level"),
                    "first_family_id": assay_metadata.get("first_family_id"),
                    "first_endpoint_group": assay_metadata.get("first_endpoint_group"),
                    "family_levels": assay_metadata.get("family_levels") or [],
                    "family_ids": assay_metadata.get("family_ids") or [],
                    "family_endpoint_groups": assay_metadata.get(
                        "family_endpoint_groups"
                    )
                    or [],
                }
                if assay_metadata
                else {}
            ),
        },
    }
    return attach_minimal_evidence(row)


def build_assay_evidence_rows(
    *,
    task: str,
    records_path: Path,
    membership_path: Path | None,
    ranked_assays_path: Path,
    max_record_examples: int = 3,
    max_support_text_chars: int = 0,
    max_assays: int = 0,
    allowed_molecules_path: Path | None = None,
    allowed_smiles_field: str = "drug",
    heldout_molecules_path: Path | None = None,
    heldout_smiles_field: str = "drug",
    filter_source_id: str = "",
    filter_scope_field: str = "",
    filter_scope_value: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = [
        "canonical_record_id",
        "molecule_id",
        "canonical_smiles",
        "canonical_endpoint_name",
        "canonical_measurement_text",
        "canonical_unit_text",
        "canonical_assay_context",
        "canonical_species_context",
        "qualifying_conditions",
        "support_text",
        "confidence",
        "source_name",
        "molecule_name",
        "group_id",
    ]
    if membership_path is None:
        columns.append("retrieval_eligible")
    if heldout_molecules_path is not None:
        if "source_id" not in columns:
            columns.append("source_id")
        if filter_scope_field and filter_scope_field not in columns:
            columns.append(filter_scope_field)
    ranked_all = sorted(
        read_jsonl(ranked_assays_path),
        key=_assay_selection_rank,
    )
    ranked = ranked_all[:max_assays] if max_assays > 0 else ranked_all
    selected_assay_ids = {str(row["assay_id"]) for row in ranked}
    family_catalog = any(row.get("first_level") not in (None, "") for row in ranked_all)
    allowed_source_groups = {
        str(source_group)
        for row in ranked_all
        for source_group in row.get("source_groups") or []
        if str(source_group)
    }
    if family_catalog and not allowed_source_groups:
        raise ValueError("family assay catalog contains no source_groups")

    records = pq.read_table(records_path, columns=columns).to_pandas()
    if membership_path is not None:
        member_ids = set(
            pq.read_table(membership_path, columns=["canonical_record_id"])
            .column("canonical_record_id")
            .to_pylist()
        )
        records = records.loc[records["canonical_record_id"].isin(member_ids)].copy()
        membership_stats = {
            "record_selection": "stage07_membership",
            "membership": str(membership_path.resolve()),
        }
    else:
        records = records.loc[records["retrieval_eligible"].eq(True)].copy()  # noqa: E712
        membership_stats = {
            "record_selection": "stage03_retrieval_eligible",
            "membership": "",
        }
    n_membership_records = len(records)
    n_membership_molecules = int(records["molecule_id"].nunique())
    if family_catalog:
        records = records.loc[records["group_id"].isin(allowed_source_groups)].copy()
    if allowed_molecules_path is not None and heldout_molecules_path is not None:
        raise ValueError(
            "allowed_molecules_path and heldout_molecules_path are mutually exclusive"
        )
    heldout_stats: dict[str, Any] = {}
    if heldout_molecules_path is not None:
        records, heldout_stats = _filter_heldout_direct_records(
            records,
            heldout_molecules_path=heldout_molecules_path,
            heldout_smiles_field=heldout_smiles_field,
            filter_source_id=filter_source_id,
            filter_scope_field=filter_scope_field,
            filter_scope_value=filter_scope_value,
        )
    allowed_stats: dict[str, Any] = {}
    if allowed_molecules_path is not None:
        allowed_keys, n_allowed_rows = _load_allowed_parent_keys(
            allowed_molecules_path,
            allowed_smiles_field,
        )
        smiles_to_key = {
            smiles: _parent_key(smiles)
            for smiles in records["canonical_smiles"].dropna().astype(str).unique()
        }
        record_keys = records["canonical_smiles"].astype(str).map(smiles_to_key)
        records = records.loc[record_keys.isin(allowed_keys)].copy()
        retained_keys = {
            smiles_to_key[str(smiles)]
            for smiles in records["canonical_smiles"].dropna().astype(str).unique()
        }
        unexpected = retained_keys - allowed_keys
        if unexpected:
            raise AssertionError(
                f"non-train parent leakage remains for {len(unexpected)} identities"
            )
        allowed_stats = {
            "allowed_molecules": str(allowed_molecules_path.resolve()),
            "allowed_smiles_field": allowed_smiles_field,
            "n_allowed_rows": n_allowed_rows,
            "n_allowed_parent_identities": len(allowed_keys),
            "n_membership_records_before_allowed_filter": n_membership_records,
            "n_membership_molecules_before_allowed_filter": n_membership_molecules,
            "n_records_after_allowed_filter": len(records),
            "n_molecules_after_allowed_filter": int(records["molecule_id"].nunique()),
            "n_retained_parent_identities": len(retained_keys),
            "n_non_allowed_records_excluded": n_membership_records - len(records),
            "train_molecule_only": True,
            "n_unexpected_parent_identities": 0,
        }
    units = [
        assay_unit(context, endpoint)[0]
        for context, endpoint in zip(
            records["canonical_assay_context"],
            records["canonical_endpoint_name"],
            strict=True,
        )
    ]
    records["assay_context"] = units
    records["assay_id"] = [assay_id(task, unit) for unit in units]
    records = records.loc[records["assay_id"].isin(selected_assay_ids)].copy()
    ranking_by_id = {str(row["assay_id"]): row for row in ranked}
    missing_ids = sorted(set(records["assay_id"]) - set(ranking_by_id))
    if missing_ids:
        raise ValueError(
            f"Ranked assay catalog is missing {len(missing_ids)} record assay ids"
        )

    evidence_rows = []
    for (stable_assay_id, molecule_id), group in records.groupby(
        ["assay_id", "molecule_id"], sort=True
    ):
        assay_context = str(group["assay_context"].iloc[0])
        evidence_rows.append(
            _aggregate_assay_molecule(
                task,
                assay_context,
                str(stable_assay_id),
                group,
                max_record_examples=max_record_examples,
                max_support_text_chars=max_support_text_chars,
                assay_metadata=ranking_by_id[str(stable_assay_id)],
            )
        )
    ranking = [
        {
            "assay_id": str(row["assay_id"]),
            "assay_context": str(row["assay_context"]),
            "group_id": f"Assay.{row['assay_id']}",
            **{
                field: row[field]
                for field in (
                    "first_level",
                    "first_family_id",
                    "first_endpoint_group",
                    "family_levels",
                    "family_ids",
                    "family_endpoint_groups",
                    "source_groups",
                    "source_families",
                )
                if field in row
            },
        }
        for row in ranked
    ]
    stats = {
        "n_source_records": int(len(records)),
        "n_ranked_assays_total": len(ranked_all),
        "n_assays": len(ranking),
        "n_assay_molecule_rows": len(evidence_rows),
        "n_molecules": int(records["molecule_id"].nunique()),
        **membership_stats,
        **heldout_stats,
        **allowed_stats,
        "family_catalog": family_catalog,
        "allowed_source_groups": sorted(allowed_source_groups),
    }
    return evidence_rows, ranking, stats


def build_assay_index(
    *,
    task: str,
    records_path: Path,
    membership_path: Path | None,
    ranked_assays_path: Path,
    workers: int = 1,
    max_record_examples: int = 3,
    max_support_text_chars: int = 0,
    max_assays: int = 0,
    allowed_molecules_path: Path | None = None,
    allowed_smiles_field: str = "drug",
    heldout_molecules_path: Path | None = None,
    heldout_smiles_field: str = "drug",
    filter_source_id: str = "",
    filter_scope_field: str = "",
    filter_scope_value: str = "",
    evidence_prompt_profile: str = ASSAY_COMPACT_PROMPT_PROFILE,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    evidence_rows, ranking, stats = build_assay_evidence_rows(
        task=task,
        records_path=records_path,
        membership_path=membership_path,
        ranked_assays_path=ranked_assays_path,
        max_record_examples=max_record_examples,
        max_support_text_chars=max_support_text_chars,
        max_assays=max_assays,
        allowed_molecules_path=allowed_molecules_path,
        allowed_smiles_field=allowed_smiles_field,
        heldout_molecules_path=heldout_molecules_path,
        heldout_smiles_field=heldout_smiles_field,
        filter_source_id=filter_source_id,
        filter_scope_field=filter_scope_field,
        filter_scope_value=filter_scope_value,
    )
    index_version = {
        ASSAY_COMPACT_PROMPT_PROFILE: INDEX_VERSION,
        ASSAY_RAW_CARD_PROMPT_PROFILE: RAW_CARD_INDEX_VERSION,
        ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE: MECHANISM_TAGGED_INDEX_VERSION,
    }.get(evidence_prompt_profile)
    if index_version is None:
        raise ValueError(
            f"Unsupported new assay index prompt profile: {evidence_prompt_profile}"
        )
    index = build_neighbor_index(
        evidence_rows,
        index_version=index_version,
        workers=workers,
        progress_every=10000,
    )
    index["assay_ranking"] = ranking
    index["source"] = {
        "type": "starling_assay_ranked_flat",
        "task": task,
        "index_version": index_version,
        "ranked_assays_sha256": sha256_file(ranked_assays_path),
        "relevance_scores_visible_to_llm": False,
        "evidence_prompt_profile": evidence_prompt_profile,
        "support_text_policy": (
            "complete_representative_support"
            if evidence_prompt_profile
            in (
                ASSAY_RAW_CARD_PROMPT_PROFILE,
                ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
            )
            else "legacy_compact_excerpt"
        ),
    }
    return index, evidence_rows, stats


def retrieve_assay_prefix(
    query_smiles: str,
    index: dict[str, Any],
    *,
    assay_prefix: int,
    assay_start: int = 0,
    top_k_per_assay: int = 3,
    min_similarity: float = 0.3,
    neighbor_identity_policy: str = "parent_disjoint",
    max_record_cards_per_molecule: int = 0,
    max_neighbor_molecules: int = 0,
    record_card_selection: str = RECORD_CARD_SELECTION_VERSION,
) -> dict[str, Any]:
    if assay_prefix <= 0:
        raise ValueError("assay_prefix must be positive")
    if assay_start < 0:
        raise ValueError("assay_start must be non-negative")
    if max_record_cards_per_molecule < 0:
        raise ValueError("max_record_cards_per_molecule must be non-negative")
    if max_neighbor_molecules < 0:
        raise ValueError("max_neighbor_molecules must be non-negative")
    if record_card_selection not in RECORD_CARD_SELECTIONS:
        raise ValueError(f"Unsupported record-card selection: {record_card_selection}")
    ranking = list(index.get("assay_ranking") or [])
    selected = ranking[assay_start : assay_start + assay_prefix]
    selected_group_ids = [str(row["group_id"]) for row in selected]
    native = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=top_k_per_assay,
        min_similarity=min_similarity,
        groups=selected_group_ids,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    if native.get("status") != "ok":
        return native
    return _assemble_assay_result(
        native,
        index=index,
        selected=selected,
        top_k_per_assay=top_k_per_assay,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        assay_start=assay_start,
        max_record_cards_per_molecule=max_record_cards_per_molecule,
        max_neighbor_molecules=max_neighbor_molecules,
        record_card_selection=record_card_selection,
    )


def _evenly_spaced_indices(n_items: int, n_selected: int) -> list[int]:
    if n_selected <= 0 or n_items <= 0:
        return []
    if n_selected >= n_items:
        return list(range(n_items))
    if n_selected == 1:
        return [0]
    return [index * (n_items - 1) // (n_selected - 1) for index in range(n_selected)]


def _record_card_key(row: dict[str, Any], example: Any) -> str:
    if isinstance(example, dict):
        return json.dumps(example, sort_keys=True, ensure_ascii=False, default=str)
    fallback = {
        "standard_type": row.get("standard_type"),
        "standard_value": row.get("standard_value"),
        "standard_units": row.get("standard_units"),
        "assay_description": row.get("assay_description"),
        "minimal_evidence": row.get("minimal_evidence"),
    }
    return json.dumps(fallback, sort_keys=True, ensure_ascii=False, default=str)


def _cap_neighbor_record_cards_even(
    neighbor: dict[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    """Keep a deterministic, assay-spanning sample of prompt-visible cards."""
    if limit <= 0:
        return neighbor
    rows = [row for row in neighbor.get("evidence_rows") or [] if isinstance(row, dict)]
    candidates: list[list[Any]] = []
    for row in rows:
        examples = [
            example
            for example in row.get("source_record_examples") or []
            if isinstance(example, dict)
        ][:3]
        candidates.append(examples or [None])

    selected: list[tuple[int, Any]] = []
    seen: set[str] = set()

    def add(row_index: int, example: Any) -> None:
        if len(selected) >= limit:
            return
        key = _record_card_key(rows[row_index], example)
        if key not in seen:
            seen.add(key)
            selected.append((row_index, example))

    first_card_rows = [index for index, values in enumerate(candidates) if values]
    for position in _evenly_spaced_indices(
        len(first_card_rows), min(limit, len(first_card_rows))
    ):
        row_index = first_card_rows[position]
        add(row_index, candidates[row_index][0])
    for example_index in (1, 2):
        for row_index in first_card_rows:
            if example_index < len(candidates[row_index]):
                add(row_index, candidates[row_index][example_index])
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for row_index in first_card_rows:
            for example in candidates[row_index]:
                add(row_index, example)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

    selected_by_row: dict[int, list[dict[str, Any]]] = {}
    fallback_rows: set[int] = set()
    for row_index, example in selected:
        if example is None:
            fallback_rows.add(row_index)
        else:
            selected_by_row.setdefault(row_index, []).append(example)
    retained_rows = []
    for row_index, row in enumerate(rows):
        if row_index not in selected_by_row and row_index not in fallback_rows:
            continue
        retained = deepcopy(row)
        if row_index in selected_by_row:
            retained["source_record_examples"] = selected_by_row[row_index]
        retained_rows.append(retained)

    output = dict(neighbor)
    output["evidence_rows"] = retained_rows
    output["n_evidence_rows"] = len(retained_rows)
    output["n_prompt_record_cards"] = len(selected)
    output["prompt_record_card_limit"] = limit
    return output


_EMPTY_MEASUREMENTS = {
    "",
    "na",
    "n/a",
    "none",
    "not reported",
    "not specified",
    "unknown",
    "unspecified",
}


def _informative_card_key(
    row: dict[str, Any],
    example: dict[str, Any] | None,
) -> tuple[int, ...]:
    example = example or {}
    value = _clean(example.get("reported_value")).lower()
    unit = _clean(example.get("reported_units")).lower()
    endpoint = _clean(example.get("endpoint_type"))
    support = _clean(example.get("support_text"))
    context_count = sum(
        bool(_clean(example.get(field)))
        for field in ("assay_context", "species_context", "qualifying_conditions")
    )
    has_value = value not in _EMPTY_MEASUREMENTS
    has_unit = unit not in _EMPTY_MEASUREMENTS
    has_numeric_value = bool(re.search(r"[-+]?\d", value))
    confidence = row.get("confidence_score")
    try:
        confidence_rank = round(float(confidence) * 1000)
    except (TypeError, ValueError):
        confidence_rank = -1
    return (
        int(has_value and bool(support)),
        int(has_numeric_value),
        int(has_value),
        int(bool(support)),
        int(bool(endpoint)),
        int(has_unit),
        context_count,
        confidence_rank,
    )


def _mechanism_family(
    row: dict[str, Any],
    example: dict[str, Any] | None,
) -> tuple[str, int]:
    example = example or {}
    family = _clean(example.get("evidence_family"))
    level = example.get("evidence_family_level")
    assay_metadata = row.get("assay_retrieval") or {}
    if not family and isinstance(assay_metadata, dict):
        family = _clean(assay_metadata.get("first_endpoint_group"))
        level = assay_metadata.get("first_level")
    try:
        normalized_level = int(level)
    except (TypeError, ValueError):
        normalized_level = 10**9
    return family or "unclassified", normalized_level


def _cap_neighbor_record_cards_mechanism_aware(
    neighbor: dict[str, Any],
    *,
    limit: int,
) -> dict[str, Any]:
    """Keep direct-first, informative cards while preserving family diversity."""
    rows = [row for row in neighbor.get("evidence_rows") or [] if isinstance(row, dict)]
    candidates = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        examples = [
            example
            for example in row.get("source_record_examples") or []
            if isinstance(example, dict)
        ][:3] or [None]
        for example in examples:
            key = _record_card_key(row, example)
            if key in seen:
                continue
            seen.add(key)
            family, family_level = _mechanism_family(row, example)
            endpoint = _clean((example or {}).get("endpoint_type")) or _clean(
                row.get("standard_type")
            )
            candidates.append(
                {
                    "row_index": row_index,
                    "example": example,
                    "key": key,
                    "family": family,
                    "family_level": family_level,
                    "endpoint": endpoint,
                    "information": _informative_card_key(row, example),
                }
            )
    if not candidates:
        return _cap_neighbor_record_cards_even(neighbor, limit=limit)

    def informative_order(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            *(-value for value in candidate["information"]),
            candidate["family_level"],
            candidate["key"],
        )

    selected = []
    selected_keys: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        if len(selected) < limit and candidate["key"] not in selected_keys:
            selected.append(candidate)
            selected_keys.add(candidate["key"])

    # The first card is the closest-to-target available family, with record
    # informativeness breaking ties. Subsequent passes add new families and
    # endpoints before redundant cards.
    add(
        min(
            candidates,
            key=lambda candidate: (
                candidate["family_level"],
                informative_order(candidate),
            ),
        )
    )
    best_by_family: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=informative_order):
        best_by_family.setdefault(candidate["family"], candidate)
    for candidate in sorted(best_by_family.values(), key=informative_order):
        add(candidate)
    selected_endpoints = {candidate["endpoint"] for candidate in selected}
    for candidate in sorted(candidates, key=informative_order):
        if candidate["endpoint"] not in selected_endpoints:
            add(candidate)
            selected_endpoints.add(candidate["endpoint"])
    for candidate in sorted(candidates, key=informative_order):
        add(candidate)

    selected_by_row: dict[int, list[dict[str, Any]]] = {}
    fallback_rows: set[int] = set()
    for candidate in selected:
        row_index = int(candidate["row_index"])
        example = candidate["example"]
        if example is None:
            fallback_rows.add(row_index)
        else:
            selected_by_row.setdefault(row_index, []).append(example)
    retained_rows = []
    for row_index, row in enumerate(rows):
        if row_index not in selected_by_row and row_index not in fallback_rows:
            continue
        retained = deepcopy(row)
        if row_index in selected_by_row:
            retained["source_record_examples"] = selected_by_row[row_index]
        retained_rows.append(retained)

    output = dict(neighbor)
    output["evidence_rows"] = retained_rows
    output["n_evidence_rows"] = len(retained_rows)
    output["n_prompt_record_cards"] = len(selected)
    output["prompt_record_card_limit"] = limit
    output["prompt_record_families"] = sorted(
        {candidate["family"] for candidate in selected}
    )
    return output


def _cap_neighbor_record_cards(
    neighbor: dict[str, Any],
    *,
    limit: int,
    selection: str = RECORD_CARD_SELECTION_VERSION,
) -> dict[str, Any]:
    if limit <= 0:
        return neighbor
    if selection == RECORD_CARD_SELECTION_VERSION:
        return _cap_neighbor_record_cards_even(neighbor, limit=limit)
    if selection == MECHANISM_AWARE_RECORD_CARD_SELECTION_VERSION:
        return _cap_neighbor_record_cards_mechanism_aware(neighbor, limit=limit)
    raise ValueError(f"Unsupported record-card selection: {selection}")


def _filter_neighbor_cards_by_family_level(
    neighbor: dict[str, Any],
    *,
    max_level: int,
) -> dict[str, Any] | None:
    """Delay each tagged record card until its biological family is available."""
    retained_rows = []
    for row in neighbor.get("evidence_rows") or []:
        retained = deepcopy(row)
        examples = [
            example
            for example in row.get("source_record_examples") or []
            if isinstance(example, dict)
        ]
        if examples:
            visible = []
            for example in examples:
                try:
                    level = int(example.get("evidence_family_level"))
                except (TypeError, ValueError):
                    level = 0
                if not level or level <= max_level:
                    visible.append(example)
            if not visible:
                continue
            retained["source_record_examples"] = visible
        else:
            assay_metadata = row.get("assay_retrieval") or {}
            try:
                level = int(assay_metadata.get("first_level"))
            except (AttributeError, TypeError, ValueError):
                level = 0
            if level and level > max_level:
                continue
        retained_rows.append(retained)
    if not retained_rows:
        return None
    output = dict(neighbor)
    output["evidence_rows"] = retained_rows
    output["n_evidence_rows"] = len(retained_rows)
    return output


def _neighbor_min_family_level(neighbor: dict[str, Any]) -> int:
    levels = []
    for row in neighbor.get("evidence_rows") or []:
        examples = [
            example
            for example in row.get("source_record_examples") or []
            if isinstance(example, dict)
        ]
        raw_levels = [example.get("evidence_family_level") for example in examples]
        if not raw_levels:
            raw_levels = [(row.get("assay_retrieval") or {}).get("first_level")]
        for raw_level in raw_levels:
            try:
                levels.append(int(raw_level))
            except (TypeError, ValueError):
                continue
    return min(levels, default=10**9)


def _assemble_assay_result(
    native: dict[str, Any],
    *,
    index: dict[str, Any],
    selected: list[dict[str, Any]],
    top_k_per_assay: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    assay_start: int,
    max_record_cards_per_molecule: int,
    max_neighbor_molecules: int,
    record_card_selection: str,
) -> dict[str, Any]:
    source = index.get("source") or {}
    evidence_prompt_profile = str(source.get("evidence_prompt_profile") or "")
    if not evidence_prompt_profile:
        evidence_prompt_profile = (
            ASSAY_COMPACT_V2_PROMPT_PROFILE
            if source.get("support_summary_cache")
            else ASSAY_COMPACT_PROMPT_PROFILE
        )
    groups_with_hits = [group for group in native["groups"] if group.get("neighbors")]
    flat = flatten_retrieval_groups(groups_with_hits)
    max_visible_family_level = max(
        (int(row.get("first_level") or 0) for row in selected),
        default=0,
    )
    if (
        evidence_prompt_profile == ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
        and max_visible_family_level
    ):
        visible_neighbors = [
            _filter_neighbor_cards_by_family_level(
                neighbor,
                max_level=max_visible_family_level,
            )
            for neighbor in flat["neighbors"]
        ]
        flat["neighbors"] = [
            neighbor for neighbor in visible_neighbors if neighbor is not None
        ]
        # Similarity remains primary. At an exact Morgan tie, keep evidence
        # closer to the task outcome before a more distant source.
        flat["neighbors"].sort(
            key=lambda neighbor: (
                -float(neighbor.get("similarity") or 0),
                _neighbor_min_family_level(neighbor),
                str(neighbor.get("molecule_chembl_id") or ""),
            )
        )
        for rank, neighbor in enumerate(flat["neighbors"], start=1):
            neighbor["rank"] = rank
    n_candidate_neighbor_molecules = len(flat["neighbors"])
    if max_neighbor_molecules:
        flat["neighbors"] = flat["neighbors"][:max_neighbor_molecules]
        flat["n_candidate_molecules_before_cap"] = n_candidate_neighbor_molecules
        flat["n_candidate_molecules"] = len(flat["neighbors"])
    if max_record_cards_per_molecule:
        flat["neighbors"] = [
            _cap_neighbor_record_cards(
                neighbor,
                limit=max_record_cards_per_molecule,
                selection=record_card_selection,
            )
            for neighbor in flat["neighbors"]
        ]
    flat.update(
        {
            "group_id": FLAT_GROUP_ID,
            "tier": "Flat",
            "endpoint_group": "assay_ranked_evidence",
            "evidence_prompt_profile": evidence_prompt_profile,
            "n_selected_assays": len(selected),
            "n_assays_with_neighbors": len(groups_with_hits),
            "n_assay_neighbor_slots": sum(
                len(group["neighbors"]) for group in groups_with_hits
            ),
            "n_unique_neighbor_molecules": len(flat["neighbors"]),
            "n_candidate_neighbor_molecules_before_cap": n_candidate_neighbor_molecules,
        }
    )
    if max_record_cards_per_molecule:
        flat.update(
            {
                "n_prompt_record_cards": sum(
                    int(neighbor.get("n_prompt_record_cards") or 0)
                    for neighbor in flat["neighbors"]
                ),
                "max_record_cards_per_molecule": max_record_cards_per_molecule,
            }
        )
    coverage = {
        "n_groups": 1,
        "n_groups_with_neighbors": int(bool(flat["neighbors"])),
        "n_neighbors_total": len(flat["neighbors"]),
        "n_candidate_neighbor_molecules_before_cap": n_candidate_neighbor_molecules,
        "n_selected_assays": len(selected),
        "n_assays_with_neighbors": len(groups_with_hits),
        "n_assay_neighbor_slots": flat["n_assay_neighbor_slots"],
        "min_similarity": min_similarity,
        "top_k_per_assay": top_k_per_assay,
    }
    if max_record_cards_per_molecule:
        coverage.update(
            {
                "n_prompt_record_cards": flat["n_prompt_record_cards"],
                "max_record_cards_per_molecule": max_record_cards_per_molecule,
            }
        )
    if max_neighbor_molecules:
        coverage["max_neighbor_molecules"] = max_neighbor_molecules
    experiment = {
        "mode": "assay_flat",
        "source": "starling_assay_ranked_flat",
        "assay_prefix": len(selected),
        "top_k_per_assay": top_k_per_assay,
        "min_similarity": min_similarity,
        "neighbor_identity_policy": neighbor_identity_policy,
        "selected_assay_ids": [row["assay_id"] for row in selected],
        "relevance_scores_visible_to_llm": False,
    }
    if assay_start:
        experiment["assay_start"] = assay_start
    if max_record_cards_per_molecule:
        experiment["max_record_cards_per_molecule"] = max_record_cards_per_molecule
        experiment["record_card_selection"] = record_card_selection
    if max_neighbor_molecules:
        experiment["max_neighbor_molecules"] = max_neighbor_molecules
    if max_visible_family_level:
        experiment["max_visible_evidence_family_level"] = max_visible_family_level
    return {
        "status": "ok",
        "evidence_source": dict(index.get("source") or {}),
        "experiment": experiment,
        "query": native["query"],
        "groups": [flat] if flat["neighbors"] else [],
        "coverage": coverage,
    }


def retrieve_assay_prefixes(
    query_smiles: str,
    index: dict[str, Any],
    *,
    prefixes: Iterable[int],
    assay_start: int = 0,
    top_k_per_assay: int = 3,
    min_similarity: float = 0.3,
    neighbor_identity_policy: str = "parent_disjoint",
    max_record_cards_per_molecule: int = 0,
    max_neighbor_molecules: int = 0,
    record_card_selection: str = RECORD_CARD_SELECTION_VERSION,
) -> dict[int, dict[str, Any]]:
    normalized = sorted({int(prefix) for prefix in prefixes})
    if not normalized or normalized[0] <= 0:
        raise ValueError("prefixes must contain positive integers")
    if assay_start < 0:
        raise ValueError("assay_start must be non-negative")
    if max_record_cards_per_molecule < 0:
        raise ValueError("max_record_cards_per_molecule must be non-negative")
    if max_neighbor_molecules < 0:
        raise ValueError("max_neighbor_molecules must be non-negative")
    if record_card_selection not in RECORD_CARD_SELECTIONS:
        raise ValueError(f"Unsupported record-card selection: {record_card_selection}")
    ranking = list(index.get("assay_ranking") or [])
    largest = min(normalized[-1], max(0, len(ranking) - assay_start))
    selected_largest = ranking[assay_start : assay_start + largest]
    native = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=top_k_per_assay,
        min_similarity=min_similarity,
        groups=[str(row["group_id"]) for row in selected_largest],
        neighbor_identity_policy=neighbor_identity_policy,
    )
    if native.get("status") != "ok":
        return {prefix: native for prefix in normalized}
    native_groups = list(native["groups"])
    results = {}
    for prefix in normalized:
        selected = ranking[assay_start : assay_start + prefix]
        prefix_native = dict(native)
        prefix_native["groups"] = native_groups[: len(selected)]
        results[prefix] = _assemble_assay_result(
            prefix_native,
            index=index,
            selected=selected,
            top_k_per_assay=top_k_per_assay,
            min_similarity=min_similarity,
            neighbor_identity_policy=neighbor_identity_policy,
            assay_start=assay_start,
            max_record_cards_per_molecule=max_record_cards_per_molecule,
            max_neighbor_molecules=max_neighbor_molecules,
            record_card_selection=record_card_selection,
        )
    return results


def build_family_molecule_prefix_view(
    index: dict[str, Any],
    *,
    levels: Iterable[int],
) -> dict[str, Any]:
    """Build cumulative record-family pools without per-assay neighbor gating.

    The source index remains assay-aware so every card keeps its physical-assay
    provenance.  This view changes only candidate generation: representative
    source records are exposed at their own ``evidence_family_level``, merged by
    molecule, and made cumulative across the requested levels.
    """
    normalized = sorted({int(level) for level in levels})
    if not normalized or normalized != list(range(1, normalized[-1] + 1)):
        raise ValueError("levels must be contiguous positive integers starting at 1")
    profile = str((index.get("source") or {}).get("evidence_prompt_profile") or "")
    if profile != ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE:
        raise ValueError(
            "family-molecule prefix retrieval requires a mechanism-tagged index"
        )

    molecules = list(index.get("molecules") or [])
    molecule_index = {
        str(row.get("molecule_chembl_id") or ""): position
        for position, row in enumerate(molecules)
    }
    virtual_groups = {
        level: f"{FAMILY_MOLECULE_GROUP_PREFIX}{level}" for level in normalized
    }
    group_to_molecule_indices: dict[str, list[int]] = {
        group_id: [] for group_id in virtual_groups.values()
    }
    evidence_by_molecule_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    n_missing_family_level = 0
    n_visible_examples_by_level = Counter()
    n_source_assays_by_level: dict[int, set[str]] = {
        level: set() for level in normalized
    }

    for raw_molecule_id, assay_groups in (
        index.get("evidence_by_molecule_group") or {}
    ).items():
        molecule_id = str(raw_molecule_id)
        position = molecule_index.get(molecule_id)
        if position is None:
            raise ValueError(f"evidence references unknown molecule: {molecule_id}")
        cumulative_rows: dict[int, list[dict[str, Any]]] = {
            level: [] for level in normalized
        }
        for rows in assay_groups.values():
            for row in rows:
                examples = [
                    dict(example)
                    for example in row.get("source_record_examples") or []
                    if isinstance(example, dict)
                ]
                tagged: list[tuple[int, dict[str, Any]]] = []
                for example in examples:
                    try:
                        family_level = int(example.get("evidence_family_level") or 0)
                    except (TypeError, ValueError):
                        family_level = 0
                    if family_level <= 0:
                        n_missing_family_level += 1
                        continue
                    tagged.append((family_level, example))
                if not tagged:
                    continue
                assay_key = str(
                    row.get("assay_chembl_id")
                    or (row.get("assay_retrieval") or {}).get("assay_id")
                    or row.get("group_id")
                    or ""
                )
                for level in normalized:
                    visible = [
                        example
                        for family_level, example in tagged
                        if family_level <= level
                    ]
                    if not visible:
                        continue
                    retained = dict(row)
                    retained["source_record_examples"] = visible
                    cumulative_rows[level].append(retained)
                    n_visible_examples_by_level[level] += len(visible)
                    if assay_key:
                        n_source_assays_by_level[level].add(assay_key)

        molecule_groups: dict[str, list[dict[str, Any]]] = {}
        for level, rows in cumulative_rows.items():
            if not rows:
                continue
            group_id = virtual_groups[level]
            group_to_molecule_indices[group_id].append(position)
            molecule_groups[group_id] = rows
        if molecule_groups:
            evidence_by_molecule_group[molecule_id] = molecule_groups

    if n_missing_family_level:
        raise ValueError(
            "mechanism-tagged index contains representative records without a "
            f"family level: {n_missing_family_level}"
        )

    view = dict(index)
    view["group_to_molecule_indices"] = group_to_molecule_indices
    view["evidence_by_molecule_group"] = evidence_by_molecule_group
    view["family_molecule_prefix_view"] = {
        "version": FAMILY_MOLECULE_VIEW_VERSION,
        "levels": normalized,
        "group_ids": {str(level): virtual_groups[level] for level in normalized},
        "candidate_generation": "global_molecule_similarity_within_cumulative_record_families",
        "per_assay_neighbor_cap": None,
        "n_candidate_molecules_by_level": {
            str(level): len(group_to_molecule_indices[virtual_groups[level]])
            for level in normalized
        },
        "n_source_assays_by_level": {
            str(level): len(n_source_assays_by_level[level]) for level in normalized
        },
        "n_visible_representative_records_by_level": {
            str(level): int(n_visible_examples_by_level[level])
            for level in normalized
        },
    }
    view["source"] = {
        **dict(index.get("source") or {}),
        "family_molecule_prefix_view": FAMILY_MOLECULE_VIEW_VERSION,
    }
    return view


def retrieve_family_molecule_prefixes(
    query_smiles: str,
    index: dict[str, Any],
    *,
    levels: Iterable[int],
    min_similarity: float = 0.3,
    neighbor_identity_policy: str = "parent_disjoint",
) -> dict[int, dict[str, Any]]:
    """Retrieve every eligible molecule globally within each family prefix."""
    normalized = sorted({int(level) for level in levels})
    metadata = dict(index.get("family_molecule_prefix_view") or {})
    if metadata.get("version") != FAMILY_MOLECULE_VIEW_VERSION:
        raise ValueError("index is not a family-molecule prefix view")
    if normalized != [int(level) for level in metadata.get("levels") or []]:
        raise ValueError("requested levels do not match the prepared prefix view")
    group_ids = [str(metadata["group_ids"][str(level)]) for level in normalized]
    native = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=max(1, len(index.get("molecules") or [])),
        min_similarity=min_similarity,
        groups=group_ids,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    if native.get("status") != "ok":
        return {level: native for level in normalized}

    results: dict[int, dict[str, Any]] = {}
    for level, group in zip(normalized, native.get("groups") or [], strict=True):
        neighbors = list(group.get("neighbors") or [])
        flat = {
            **dict(group),
            "group_id": FLAT_GROUP_ID,
            "tier": "Flat",
            "endpoint_group": "progressive_family_evidence",
            "evidence_prompt_profile": str(
                (index.get("source") or {}).get("evidence_prompt_profile") or ""
            ),
            "n_unique_neighbor_molecules": len(neighbors),
            "n_candidate_neighbor_molecules_before_budget": len(neighbors),
            "n_source_assays": int(metadata["n_source_assays_by_level"][str(level)]),
            "n_visible_representative_records": int(
                metadata["n_visible_representative_records_by_level"][str(level)]
            ),
        }
        coverage = {
            "n_groups": 1,
            "n_groups_with_neighbors": int(bool(neighbors)),
            "n_neighbors_total": len(neighbors),
            "n_candidate_neighbor_molecules_before_budget": len(neighbors),
            "n_source_assays": flat["n_source_assays"],
            "n_visible_representative_records": flat[
                "n_visible_representative_records"
            ],
            "min_similarity": min_similarity,
            "candidate_generation": metadata["candidate_generation"],
            "per_assay_neighbor_cap": None,
        }
        results[level] = {
            "status": "ok",
            "evidence_source": dict(index.get("source") or {}),
            "retrieval_policy": dict(native.get("retrieval_policy") or {}),
            "experiment": {
                "mode": "progressive_family_molecule_flat",
                "source": "starling_mechanism_tagged_family_molecule_view",
                "family_level": level,
                "candidate_generation": metadata["candidate_generation"],
                "per_assay_neighbor_cap": None,
                "min_similarity": min_similarity,
                "neighbor_identity_policy": neighbor_identity_policy,
                "relevance_scores_visible_to_llm": False,
            },
            "query": dict(native.get("query") or {}),
            "groups": [flat] if neighbors else [],
            "coverage": coverage,
        }
    return results


def _build_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index, evidence_rows, stats = build_assay_index(
        task=args.task,
        records_path=Path(args.records),
        membership_path=Path(args.membership) if args.membership else None,
        ranked_assays_path=Path(args.ranked_assays),
        workers=args.workers,
        max_record_examples=args.max_record_examples,
        max_support_text_chars=args.max_support_text_chars,
        max_assays=args.max_assays,
        allowed_molecules_path=(
            Path(args.allowed_molecules_jsonl) if args.allowed_molecules_jsonl else None
        ),
        allowed_smiles_field=args.allowed_smiles_field,
        heldout_molecules_path=(
            Path(args.heldout_molecules_jsonl) if args.heldout_molecules_jsonl else None
        ),
        heldout_smiles_field=args.heldout_smiles_field,
        filter_source_id=args.filter_source_id,
        filter_scope_field=args.filter_scope_field,
        filter_scope_value=args.filter_scope_value,
        evidence_prompt_profile=args.evidence_prompt_profile,
    )
    index_path = output_dir / "assay_neighbor_index.pkl"
    evidence_path = output_dir / "assay_molecule_evidence.jsonl"
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_jsonl_atomic(evidence_path, evidence_rows)
    manifest = {
        "index_version": str(
            (index.get("source") or {}).get("index_version") or INDEX_VERSION
        ),
        "task": args.task,
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(Path(args.records)),
        "membership": str(Path(args.membership).resolve()) if args.membership else "",
        "membership_sha256": sha256_file(Path(args.membership))
        if args.membership
        else "",
        "ranked_assays": str(Path(args.ranked_assays).resolve()),
        "ranked_assays_sha256": sha256_file(Path(args.ranked_assays)),
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "evidence": str(evidence_path.resolve()),
        "evidence_sha256": sha256_file(evidence_path),
        "top_k_per_assay_default": 3,
        "min_similarity_default": 0.3,
        "neighbor_identity_policy_default": args.neighbor_identity_policy_default,
        "relevance_scores_visible_to_llm": False,
        "max_assays_materialized": args.max_assays,
        "max_record_examples": args.max_record_examples,
        "max_support_text_chars": args.max_support_text_chars,
        "evidence_prompt_profile": args.evidence_prompt_profile,
        "support_text_policy": str(
            (index.get("source") or {}).get("support_text_policy") or ""
        ),
        "allowed_molecules_jsonl_sha256": (
            sha256_file(Path(args.allowed_molecules_jsonl))
            if args.allowed_molecules_jsonl
            else ""
        ),
        "heldout_molecules_jsonl_sha256": (
            sha256_file(Path(args.heldout_molecules_jsonl))
            if args.heldout_molecules_jsonl
            else ""
        ),
        **stats,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(
        f"[{args.task}] built assay index: assays={stats['n_assays']:,} "
        f"molecules={stats['n_molecules']:,} assay_molecule_rows={stats['n_assay_molecule_rows']:,}"
    )


def _build_preaggregated_command(args: argparse.Namespace) -> None:
    """Build a raw-card index from an already filtered assay evidence artifact."""
    evidence_path = Path(args.evidence_jsonl)
    source_manifest_path = Path(args.source_manifest)
    ranked_assays_path = Path(args.ranked_assays)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(read_jsonl(ranked_assays_path), key=_assay_selection_rank)
    family_by_assay = {str(row["assay_id"]): row for row in ranked}
    evidence_rows = read_jsonl(evidence_path)
    for row in evidence_rows:
        assay_key = str(
            row.get("assay_chembl_id")
            or (row.get("assay_retrieval") or {}).get("assay_id")
            or ""
        )
        assay_metadata = family_by_assay.get(assay_key, {})
        raw_examples = []
        for original in (row.get("source_record_examples") or [])[:3]:
            example = dict(original)
            example.pop("support_summary", None)
            if args.evidence_prompt_profile == ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE:
                example["evidence_family"] = assay_metadata.get(
                    "first_endpoint_group", ""
                )
                example["evidence_family_level"] = assay_metadata.get("first_level", "")
            raw_examples.append(example)
        row["source_record_examples"] = raw_examples
        if args.evidence_prompt_profile == ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE:
            row.setdefault("assay_retrieval", {}).update(
                {
                    field: assay_metadata.get(field)
                    for field in (
                        "first_level",
                        "first_family_id",
                        "first_endpoint_group",
                        "family_levels",
                        "family_ids",
                        "family_endpoint_groups",
                    )
                }
            )

    ranking = [
        {
            "assay_id": str(row["assay_id"]),
            "assay_context": str(row["assay_context"]),
            "group_id": f"Assay.{row['assay_id']}",
            **{
                field: row[field]
                for field in (
                    "first_level",
                    "first_family_id",
                    "first_endpoint_group",
                    "family_levels",
                    "family_ids",
                    "family_endpoint_groups",
                    "source_groups",
                    "source_families",
                )
                if field in row
            },
        }
        for row in ranked
    ]
    index_version = (
        MECHANISM_TAGGED_INDEX_VERSION
        if args.evidence_prompt_profile == ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
        else RAW_CARD_INDEX_VERSION
    )
    index = build_neighbor_index(
        evidence_rows,
        index_version=index_version,
        workers=args.workers,
        progress_every=10000,
    )
    index["assay_ranking"] = ranking
    index["source"] = {
        "type": "starling_assay_ranked_flat",
        "task": args.task,
        "index_version": index_version,
        "ranked_assays_sha256": sha256_file(ranked_assays_path),
        "relevance_scores_visible_to_llm": False,
        "evidence_prompt_profile": args.evidence_prompt_profile,
        "support_text_policy": "complete_representative_support",
    }

    raw_evidence_path = output_dir / "assay_molecule_evidence.jsonl"
    index_path = output_dir / "assay_neighbor_index.pkl"
    write_jsonl_atomic(raw_evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **source_manifest,
        "index_version": index_version,
        "task": args.task,
        "ranked_assays": str(ranked_assays_path.resolve()),
        "ranked_assays_sha256": sha256_file(ranked_assays_path),
        "index": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "evidence": str(raw_evidence_path.resolve()),
        "evidence_sha256": sha256_file(raw_evidence_path),
        "preaggregated_evidence": str(evidence_path.resolve()),
        "preaggregated_evidence_sha256": sha256_file(evidence_path),
        "preaggregated_manifest": str(source_manifest_path.resolve()),
        "preaggregated_manifest_sha256": sha256_file(source_manifest_path),
        "evidence_prompt_profile": args.evidence_prompt_profile,
        "support_text_policy": "complete_representative_support",
        "neighbor_identity_policy_default": args.neighbor_identity_policy_default,
        "n_assays": len(ranking),
        "n_assay_molecule_rows": len(evidence_rows),
        "n_molecules": len(index.get("molecules") or []),
        "n_evidence_molecule_ids": len(
            {str(row.get("molecule_chembl_id") or "") for row in evidence_rows}
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(
        f"[{args.task}] built preaggregated raw assay index: assays={len(ranking):,} "
        f"molecules={manifest['n_molecules']:,} assay_molecule_rows={len(evidence_rows):,}"
    )


def _retrieve_command(args: argparse.Namespace) -> None:
    with Path(args.index).open("rb") as handle:
        index = pickle.load(handle)
    results = retrieve_assay_prefixes(
        args.query_smiles,
        index,
        prefixes=args.prefixes,
        assay_start=args.assay_start,
        top_k_per_assay=args.top_k_per_assay,
        min_similarity=args.min_similarity,
        neighbor_identity_policy=args.neighbor_identity_policy,
        max_record_cards_per_molecule=args.max_record_cards_per_molecule,
        max_neighbor_molecules=args.max_neighbor_molecules,
        record_card_selection=args.record_card_selection,
    )
    output_dir = Path(args.output_dir)
    for prefix, result in results.items():
        prefix_dir = output_dir / f"top_{prefix}"
        prefix_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(prefix_dir / "retrieval.json", result)
    print(f"wrote {len(results)} strictly nested assay-prefix retrieval artifacts")


def _materialize_batch_command(args: argparse.Namespace) -> None:
    with Path(args.index).open("rb") as handle:
        index = pickle.load(handle)
    records = read_jsonl(Path(args.input_jsonl))
    start = max(0, int(args.start))
    stop = len(records) if args.limit <= 0 else min(len(records), start + args.limit)
    prefixes = sorted({int(prefix) for prefix in args.prefixes})
    output_root = Path(args.output_root)
    batch_dirs = {
        prefix: output_root / f"assay_flat_top{prefix}" for prefix in prefixes
    }
    for batch_dir in batch_dirs.values():
        (batch_dir / "runs").mkdir(parents=True, exist_ok=True)
    for query_index in range(start, stop):
        query_smiles = str(records[query_index].get(args.smiles_field) or "")
        results = retrieve_assay_prefixes(
            query_smiles,
            index,
            prefixes=prefixes,
            assay_start=args.assay_start,
            top_k_per_assay=args.top_k_per_assay,
            min_similarity=args.min_similarity,
            neighbor_identity_policy=args.neighbor_identity_policy,
            max_record_cards_per_molecule=args.max_record_cards_per_molecule,
            max_neighbor_molecules=args.max_neighbor_molecules,
            record_card_selection=args.record_card_selection,
        )
        for prefix, result in results.items():
            if args.condition_field:
                attach_external_condition(result, records[query_index])
            batch_dir = batch_dirs[prefix]
            run_dir = batch_dir / "runs" / f"{batch_dir.name}_idx{query_index:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(run_dir / "retrieval.json", result)
        if (
            query_index - start + 1
        ) % args.progress_every == 0 or query_index + 1 == stop:
            print(f"materialized {query_index - start + 1:,}/{stop - start:,} queries")
    for prefix, batch_dir in batch_dirs.items():
        manifest = {
            "artifact_type": "assay_retrieval_replay_batch.v1",
            "input_jsonl": str(Path(args.input_jsonl).resolve()),
            "input_jsonl_sha256": sha256_file(Path(args.input_jsonl)),
            "index": str(Path(args.index).resolve()),
            "index_sha256": sha256_file(Path(args.index)),
            "smiles_field": args.smiles_field,
            "indices": list(range(start, stop)),
            "n_items": stop - start,
            "assay_prefix": prefix,
            "assay_start": args.assay_start,
            "top_k_per_assay": args.top_k_per_assay,
            "min_similarity": args.min_similarity,
            "neighbor_identity_policy": args.neighbor_identity_policy,
            "max_record_cards_per_molecule": args.max_record_cards_per_molecule,
            "max_neighbor_molecules": args.max_neighbor_molecules,
            "record_card_selection": (
                args.record_card_selection if args.max_record_cards_per_molecule else ""
            ),
            "flat_group_id": FLAT_GROUP_ID,
            "relevance_scores_visible_to_llm": False,
            "condition_field": args.condition_field,
            "condition_renderer": (
                EXTERNAL_CONDITION_RENDERER_VERSION if args.condition_field else ""
            ),
            "condition_visibility": (
                "flat_evidence_and_final_only" if args.condition_field else "none"
            ),
        }
        write_json_atomic(batch_dir / "manifest.json", manifest)
    print(f"wrote {len(batch_dirs)} replay batches under {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-index")
    build.add_argument("--task", required=True)
    build.add_argument("--records", required=True)
    build.add_argument(
        "--membership",
        default="",
        help=(
            "Optional Stage 07 record-membership parquet. If omitted, use all "
            "Stage 03 rows with retrieval_eligible=true."
        ),
    )
    build.add_argument("--ranked-assays", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument(
        "--neighbor-identity-policy-default",
        choices=("parent_disjoint", "scaffold_disjoint"),
        default="parent_disjoint",
        help="Document the intended query-time identity policy in the index manifest.",
    )
    build.add_argument("--max-record-examples", type=int, default=3)
    build.add_argument(
        "--max-support-text-chars",
        type=int,
        default=0,
        help="Truncate each retained representative support excerpt; 0 keeps it whole.",
    )
    build.add_argument(
        "--max-assays",
        type=int,
        default=0,
        help="Materialize only this ranked prefix; 0 materializes the entire catalog.",
    )
    build.add_argument(
        "--allowed-molecules-jsonl",
        default="",
        help="Optional hard allowlist; only molecules whose normalized parent occurs here are indexed.",
    )
    build.add_argument("--allowed-smiles-field", default="drug")
    build.add_argument(
        "--heldout-molecules-jsonl",
        default="",
        help=(
            "Optional held-out parent union. Matching parents are removed only from "
            "the declared direct source/scope."
        ),
    )
    build.add_argument("--heldout-smiles-field", default="drug")
    build.add_argument("--filter-source-id", default="")
    build.add_argument("--filter-scope-field", default="")
    build.add_argument("--filter-scope-value", default="")
    build.add_argument(
        "--evidence-prompt-profile",
        choices=(
            ASSAY_COMPACT_PROMPT_PROFILE,
            ASSAY_RAW_CARD_PROMPT_PROFILE,
            ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
        ),
        default=ASSAY_COMPACT_PROMPT_PROFILE,
        help=(
            "LLM evidence view recorded in the index; mechanism-tagged v4 keeps "
            "complete support and exposes biological family provenance per card."
        ),
    )
    build.set_defaults(func=_build_command)

    preaggregated = subparsers.add_parser("build-preaggregated-index")
    preaggregated.add_argument("--task", required=True)
    preaggregated.add_argument("--evidence-jsonl", required=True)
    preaggregated.add_argument("--source-manifest", required=True)
    preaggregated.add_argument("--ranked-assays", required=True)
    preaggregated.add_argument("--output-dir", required=True)
    preaggregated.add_argument("--workers", type=int, default=1)
    preaggregated.add_argument(
        "--evidence-prompt-profile",
        choices=(
            ASSAY_RAW_CARD_PROMPT_PROFILE,
            ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE,
        ),
        default=ASSAY_RAW_CARD_PROMPT_PROFILE,
    )
    preaggregated.add_argument(
        "--neighbor-identity-policy-default",
        choices=("parent_disjoint", "scaffold_disjoint"),
        default="parent_disjoint",
    )
    preaggregated.set_defaults(func=_build_preaggregated_command)

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--index", required=True)
    retrieve.add_argument("--query-smiles", required=True)
    retrieve.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=list(DEFAULT_RETRIEVAL_PREFIXES),
        help="Ad-hoc prefix sizes; formal curves use the task-specific geometric launcher.",
    )
    retrieve.add_argument("--top-k-per-assay", type=int, default=3)
    retrieve.add_argument("--assay-start", type=int, default=0)
    retrieve.add_argument("--max-record-cards-per-molecule", type=int, default=0)
    retrieve.add_argument("--max-neighbor-molecules", type=int, default=0)
    retrieve.add_argument(
        "--record-card-selection",
        choices=RECORD_CARD_SELECTIONS,
        default=RECORD_CARD_SELECTION_VERSION,
    )
    retrieve.add_argument("--min-similarity", type=float, default=0.3)
    retrieve.add_argument("--neighbor-identity-policy", default="parent_disjoint")
    retrieve.add_argument("--output-dir", required=True)
    retrieve.set_defaults(func=_retrieve_command)

    materialize = subparsers.add_parser("materialize-batch")
    materialize.add_argument("--index", required=True)
    materialize.add_argument("--input-jsonl", required=True)
    materialize.add_argument("--smiles-field", default="drug")
    materialize.add_argument(
        "--condition-field",
        default="",
        help="Optional benchmark condition field rendered into flat/final prompts only.",
    )
    materialize.add_argument("--prefixes", type=int, nargs="+", required=True)
    materialize.add_argument("--top-k-per-assay", type=int, default=3)
    materialize.add_argument("--assay-start", type=int, default=0)
    materialize.add_argument("--max-record-cards-per-molecule", type=int, default=0)
    materialize.add_argument("--max-neighbor-molecules", type=int, default=0)
    materialize.add_argument(
        "--record-card-selection",
        choices=RECORD_CARD_SELECTIONS,
        default=RECORD_CARD_SELECTION_VERSION,
    )
    materialize.add_argument("--min-similarity", type=float, default=0.3)
    materialize.add_argument("--neighbor-identity-policy", default="parent_disjoint")
    materialize.add_argument("--start", type=int, default=0)
    materialize.add_argument("--limit", type=int, default=0)
    materialize.add_argument("--progress-every", type=int, default=10)
    materialize.add_argument("--output-root", required=True)
    materialize.set_defaults(func=_materialize_batch_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
