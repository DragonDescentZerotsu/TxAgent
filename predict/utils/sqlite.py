"""SQLite helpers for ChEMBL task workflows."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    """Open a ChEMBL SQLite database with row access by column name."""
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not table_exists(conn, table_name):
        return set()
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def read_sql(
    conn: sqlite3.Connection,
    query: str,
    params: dict[str, Any] | tuple[Any, ...] | None = None,
) -> list[sqlite3.Row]:
    return list(conn.execute(query, params or {}))
