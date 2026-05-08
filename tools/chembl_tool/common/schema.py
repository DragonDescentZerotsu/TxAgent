"""Schema fallback helpers for ChEMBL SQLite tables."""

from __future__ import annotations

import sqlite3

from .sqlite import get_table_columns, table_exists


def require_any_columns(
    conn: sqlite3.Connection,
    table: str,
    candidates: list[str],
) -> list[str]:
    """Return available candidate columns, raising only if the table is absent."""
    if not table_exists(conn, table):
        raise ValueError(f"Required ChEMBL table is missing: {table}")
    columns = get_table_columns(conn, table)
    return [column for column in candidates if column in columns]


def build_select_columns(
    conn: sqlite3.Connection,
    table: str,
    requested: dict[str, str | None],
    alias: str | None = None,
) -> list[str]:
    """Build SELECT expressions with NULL fallback for missing columns.

    `requested` maps output column name to source column name. If the source is
    None, the output column name is also used as the source column name.
    """
    columns = get_table_columns(conn, table)
    prefix = f"{alias}." if alias else ""
    expressions: list[str] = []
    for output_name, source_name in requested.items():
        source = source_name or output_name
        if source in columns:
            expressions.append(f"{prefix}{source} AS {output_name}")
        else:
            expressions.append(f"NULL AS {output_name}")
    return expressions
