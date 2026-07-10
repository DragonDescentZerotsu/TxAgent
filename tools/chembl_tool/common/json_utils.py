"""Small JSON parsing helpers shared by OpenAI-compatible reasoning clients."""

from __future__ import annotations

import json
from typing import Any


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
