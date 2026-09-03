"""Render the versioned progressive prompt without changing its wire text."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined


TEMPLATE_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )


def render_progressive_messages(
    *, system_role: str, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rendered = _environment().get_template("progressive.jinja").render(
        system_role=system_role,
        user_content=json.dumps(payload, ensure_ascii=False),
    )
    messages = json.loads(rendered)
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("progressive prompt must render exactly two messages")
    return messages

