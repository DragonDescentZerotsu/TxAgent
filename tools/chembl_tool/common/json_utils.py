"""Small JSON parsing helpers shared by OpenAI-compatible reasoning clients."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and byte-size audits."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_content(content: str) -> Any:
    """Parse a JSON response without raising, so validation can request a retry."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as first_error:
        text = str(content or "")
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as nested_error:
                return {
                    "unparsed_text": text,
                    "parse_error": str(nested_error),
                }
        return {"unparsed_text": text, "parse_error": str(first_error)}


@contextmanager
def atomic_output_path(path: Path):
    """Yield a same-directory temporary path and publish it only on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        yield temporary
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Serialize one JSON value without exposing a partial destination file."""
    with atomic_output_path(path) as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Serialize JSONL without exposing a partially rewritten branch file."""
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
