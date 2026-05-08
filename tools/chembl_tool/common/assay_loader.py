"""Assay-level loaders shared by ChEMBL task modules."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .sqlite import get_table_columns, table_exists


def load_activity_summary(
    conn: sqlite3.Connection,
    assay_ids: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Aggregate activity counts and standard types by assay_id."""
    where_sql = "WHERE assay_id IS NOT NULL"
    params: list[int] = []
    if assay_ids is not None:
        if not assay_ids:
            return {}
        placeholders = ",".join("?" for _ in assay_ids)
        where_sql = f"WHERE assay_id IN ({placeholders})"
        params = assay_ids
    query = """
        SELECT
          assay_id,
          COUNT(*) AS n_activities,
          COUNT(DISTINCT molregno) AS n_unique_molecules,
          GROUP_CONCAT(DISTINCT LOWER(standard_type)) AS standard_types
        FROM activities
        {where_sql}
        GROUP BY assay_id
    """.format(where_sql=where_sql)
    summaries: dict[int, dict[str, Any]] = {}
    for row in conn.execute(query, params):
        standard_types = []
        if row["standard_types"]:
            standard_types = sorted({item for item in row["standard_types"].split(",") if item})
        summaries[int(row["assay_id"])] = {
            "n_activities": int(row["n_activities"] or 0),
            "n_unique_molecules": int(row["n_unique_molecules"] or 0),
            "standard_types": standard_types,
        }
    return summaries


def load_target_annotations(conn: sqlite3.Connection) -> dict[int, dict[str, list[str]]]:
    """Aggregate component gene symbols and synonyms by target id."""
    required = {"target_components", "component_sequences", "component_synonyms"}
    if not all(table_exists(conn, table) for table in required):
        return {}

    query = """
        SELECT
          tc.tid,
          cs.accession,
          cs.description,
          csy.component_synonym,
          csy.syn_type
        FROM target_components tc
        LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
        LEFT JOIN component_synonyms csy ON tc.component_id = csy.component_id
    """
    by_tid: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"genes": set(), "synonyms": set(), "accessions": set(), "component_descriptions": set()}
    )
    for row in conn.execute(query):
        tid = row["tid"]
        if tid is None:
            continue
        entry = by_tid[int(tid)]
        if row["accession"]:
            entry["accessions"].add(str(row["accession"]))
        if row["description"]:
            entry["component_descriptions"].add(str(row["description"]))
        synonym = row["component_synonym"]
        syn_type = row["syn_type"]
        if synonym:
            if syn_type == "GENE_SYMBOL":
                entry["genes"].add(str(synonym).upper())
            else:
                entry["synonyms"].add(str(synonym))

    return {
        tid: {key: sorted(values) for key, values in entry.items()}
        for tid, entry in by_tid.items()
    }


def iter_assay_metadata(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Stream assay metadata joined with target and optional document fields."""
    assays_cols = get_table_columns(conn, "assays")
    target_cols = get_table_columns(conn, "target_dictionary")
    docs_join = table_exists(conn, "docs") and "doc_id" in assays_cols
    docs_cols = get_table_columns(conn, "docs") if docs_join else set()

    assay_selects = _selects(
        "a",
        assays_cols,
        {
            "assay_id": "assay_id",
            "doc_id": "doc_id",
            "assay_chembl_id": "chembl_id",
            "description": "description",
            "assay_type": "assay_type",
            "assay_test_type": "assay_test_type",
            "assay_category": "assay_category",
            "assay_cell_type": "assay_cell_type",
            "assay_tissue": "assay_tissue",
            "assay_organism": "assay_organism",
            "confidence_score": "confidence_score",
            "relationship_type": "relationship_type",
            "tid": "tid",
        },
    )
    target_selects = _selects(
        "td",
        target_cols,
        {
            "target_chembl_id": "chembl_id",
            "target_pref_name": "pref_name",
            "target_type": "target_type",
            "target_organism": "organism",
        },
    )
    doc_selects = []
    if docs_join:
        for column in ("title", "abstract", "pubmed_id", "doi"):
            if column in docs_cols:
                doc_selects.append(f"d.{column} AS doc_{column}")
            else:
                doc_selects.append(f"NULL AS doc_{column}")
    else:
        doc_selects = [f"NULL AS doc_{column}" for column in ("title", "abstract", "pubmed_id", "doi")]

    join_docs_sql = "LEFT JOIN docs d ON a.doc_id = d.doc_id" if docs_join else ""
    query = f"""
        SELECT
          {", ".join(assay_selects + target_selects + doc_selects)}
        FROM assays a
        LEFT JOIN target_dictionary td ON a.tid = td.tid
        {join_docs_sql}
    """
    for row in conn.execute(query):
        yield dict(row)


def merge_assay_record(
    metadata: dict[str, Any],
    activity_summary: dict[str, Any] | None,
    target_annotation: dict[str, list[str]] | None,
) -> dict[str, Any]:
    """Merge assay metadata with pre-aggregated activity and target fields."""
    row = dict(metadata)
    summary = activity_summary or {}
    annotation = target_annotation or {}
    row["n_activities"] = int(summary.get("n_activities") or 0)
    row["n_unique_molecules"] = int(summary.get("n_unique_molecules") or 0)
    row["standard_types"] = list(summary.get("standard_types") or [])
    row["target_genes"] = list(annotation.get("genes") or [])
    row["target_synonyms"] = list(annotation.get("synonyms") or [])
    row["target_accessions"] = list(annotation.get("accessions") or [])
    row["component_descriptions"] = list(annotation.get("component_descriptions") or [])
    return row


def _selects(alias: str, columns: set[str], requested: dict[str, str]) -> list[str]:
    expressions = []
    for output_name, source_name in requested.items():
        if source_name in columns:
            expressions.append(f"{alias}.{source_name} AS {output_name}")
        else:
            expressions.append(f"NULL AS {output_name}")
    return expressions
