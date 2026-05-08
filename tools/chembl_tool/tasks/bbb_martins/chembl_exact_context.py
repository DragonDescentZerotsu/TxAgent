"""Exact ChEMBL molecule context and shared-assay enrichment for BBB reasoning."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.sqlite import connect_sqlite
from tools.chembl_tool.common.text import normalize_text


DEFAULT_CHEMBL_SQLITE = "tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db"

QUERY_ACTIVITY_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "data_validity_comment",
    "activity_comment",
    "standard_text_value",
    "action_type",
]


def enrich_retrieval_with_chembl_context(
    retrieval: dict[str, Any],
    index: dict[str, Any],
    *,
    chembl_sqlite: str | Path = DEFAULT_CHEMBL_SQLITE,
    max_exact_bbb_rows: int = 30,
    max_shared_per_neighbor: int = 8,
) -> dict[str, Any]:
    """Attach exact-query ChEMBL context and query-vs-neighbor shared assay rows.

    The enrichment is deliberately scoped: it does not change retrieval groups,
    neighbor ranking, or BBB endpoint grouping. It only compares query activities
    against assays already present in retrieved neighbor evidence.
    """
    query = retrieval.get("query") or {}
    inchi_key = str(query.get("standard_inchi_key") or "").strip()
    if not inchi_key:
        retrieval["query_chembl_context"] = {"status": "not_available", "reason": "missing query InChIKey"}
        return retrieval

    try:
        conn = connect_sqlite(chembl_sqlite)
    except FileNotFoundError:
        retrieval["query_chembl_context"] = {
            "status": "not_available",
            "reason": f"ChEMBL SQLite not found: {chembl_sqlite}",
        }
        return retrieval

    try:
        exact_matches = lookup_exact_molecule(conn, inchi_key)
        if not exact_matches:
            retrieval["query_chembl_context"] = {"status": "not_found", "standard_inchi_key": inchi_key}
            return retrieval

        selected = exact_matches[0]
        context = {
            "status": "found",
            "standard_inchi_key": inchi_key,
            "selected_molecule_chembl_id": selected.get("molecule_chembl_id", ""),
            "selected_molregno": selected.get("molregno"),
            "exact_matches": exact_matches,
            "bbb_relevant_evidence_rows": exact_bbb_evidence_rows(
                index,
                str(selected.get("molecule_chembl_id") or ""),
                max_rows=max_exact_bbb_rows,
            ),
        }
        retrieval["query_chembl_context"] = context

        assay_ids = sorted(_neighbor_assay_ids(retrieval))
        if not assay_ids:
            return retrieval

        query_activities = load_query_activities_for_assays(conn, int(selected["molregno"]), assay_ids)
        attach_shared_assay_context(retrieval, query_activities, max_per_neighbor=max_shared_per_neighbor)
        return retrieval
    finally:
        conn.close()


def lookup_exact_molecule(conn: sqlite3.Connection, standard_inchi_key: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          md.chembl_id AS molecule_chembl_id,
          md.molregno AS molregno,
          cs.canonical_smiles,
          cs.standard_inchi_key,
          cp.mw_freebase,
          cp.alogp,
          cp.hba,
          cp.hbd,
          cp.psa,
          cp.rtb,
          cp.num_ro5_violations,
          cp.full_mwt,
          cp.aromatic_rings,
          cp.heavy_atoms,
          cp.qed_weighted,
          cp.full_molformula,
          cp.np_likeness_score
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        LEFT JOIN compound_properties cp ON cs.molregno = cp.molregno
        WHERE cs.standard_inchi_key = ?
        ORDER BY md.chembl_id
        LIMIT 5
        """,
        (standard_inchi_key,),
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def exact_bbb_evidence_rows(index: dict[str, Any], molecule_chembl_id: str, *, max_rows: int = 30) -> list[dict[str, Any]]:
    by_group = (index.get("evidence_by_molecule_group") or {}).get(molecule_chembl_id) or {}
    rows: list[dict[str, Any]] = []
    for group_id in sorted(by_group):
        for row in by_group[group_id]:
            rows.append(row)
            if len(rows) >= max_rows:
                return rows
    return rows


def load_query_activities_for_assays(
    conn: sqlite3.Connection,
    molregno: int,
    assay_chembl_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    by_assay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in _chunks([assay_id for assay_id in assay_chembl_ids if assay_id], 500):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT
              act.activity_id,
              a.chembl_id AS assay_chembl_id,
              act.standard_type,
              act.standard_relation,
              act.standard_value,
              act.standard_units,
              act.pchembl_value,
              act.data_validity_comment,
              act.activity_comment,
              act.standard_text_value,
              act.action_type
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            WHERE act.molregno = ?
              AND a.chembl_id IN ({placeholders})
            ORDER BY a.chembl_id, act.standard_type, act.activity_id
            """,
            (molregno, *chunk),
        ).fetchall()
        for row in rows:
            item = _row_to_dict(row)
            by_assay[str(item.get("assay_chembl_id") or "")].append(item)
    return dict(by_assay)


def attach_shared_assay_context(
    retrieval: dict[str, Any],
    query_activities_by_assay: dict[str, list[dict[str, Any]]],
    *,
    max_per_neighbor: int = 8,
) -> None:
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            same_endpoint: list[dict[str, Any]] = []
            different_endpoint: list[dict[str, Any]] = []
            for neighbor_row in neighbor.get("evidence_rows") or []:
                assay_id = str(neighbor_row.get("assay_chembl_id") or "")
                query_rows = query_activities_by_assay.get(assay_id) or []
                if not query_rows:
                    continue
                neighbor_endpoint = _endpoint_key(neighbor_row)
                for query_row in query_rows:
                    comparison = _comparison_card(query_row, neighbor_row)
                    if _endpoint_key(query_row) == neighbor_endpoint:
                        same_endpoint.append(comparison)
                    else:
                        different_endpoint.append(comparison)
            neighbor["shared_assay_context"] = {
                "same_endpoint_activity": same_endpoint[:max_per_neighbor],
                "same_assay_different_endpoint_activity": different_endpoint[:max_per_neighbor],
                "n_same_endpoint_activity": len(same_endpoint),
                "n_same_assay_different_endpoint_activity": len(different_endpoint),
            }


def _comparison_card(query_row: dict[str, Any], neighbor_row: dict[str, Any]) -> dict[str, Any]:
    query_units = str(query_row.get("standard_units") or "")
    neighbor_units = str(neighbor_row.get("standard_units") or "")
    return {
        "assay_chembl_id": neighbor_row.get("assay_chembl_id", ""),
        "standard_type_match": _endpoint_key(query_row) == _endpoint_key(neighbor_row),
        "units_match": normalize_text(query_units) == normalize_text(neighbor_units),
        "query_activity": _activity_card(query_row),
        "neighbor_activity": _activity_card(neighbor_row),
    }


def _activity_card(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in QUERY_ACTIVITY_FIELDS if field in row}


def _endpoint_key(row: dict[str, Any]) -> str:
    return normalize_text(str(row.get("standard_type") or ""))


def _neighbor_assay_ids(retrieval: dict[str, Any]) -> set[str]:
    assay_ids: set[str] = set()
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            for row in neighbor.get("evidence_rows") or []:
                assay_id = str(row.get("assay_chembl_id") or "").strip()
                if assay_id:
                    assay_ids.add(assay_id)
    return assay_ids


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
