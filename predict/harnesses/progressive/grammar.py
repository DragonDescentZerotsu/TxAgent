"""Render version-owned progressive reasoning grammars."""

from __future__ import annotations

from typing import Any, Mapping

from predict.harnesses.progressive.prompt import (
    TEMPLATE_DIR,
    prompt_asset_path,
    prompt_assets,
    template_environment,
)


def render_reasoning_grammar(
    prompt_version: str,
    active: Mapping[str, Mapping[str, Any]],
    *,
    card_id_to_alias: Mapping[str, str],
    task: str | None = None,
) -> str | None:
    """Render the optional private-reasoning grammar for visible molecules."""
    template_name = prompt_assets(prompt_version)["settings"].get(
        "reasoning_grammar_template"
    )
    if not template_name:
        return None

    molecules = []
    for context in sorted(active.values(), key=lambda row: int(row["_selection_rank"])):
        cards = sorted(
            (context.get("cards") or {}).values(),
            key=lambda row: (int(row["first_seen_level"]), str(row["card_id"])),
        )
        aliases = [card_id_to_alias[str(card["card_id"])] for card in cards]
        if not aliases:
            raise ValueError("reasoning grammar requires at least one record per molecule")
        molecules.append(
            {"number": len(molecules) + 1, "record_ids": ",".join(aliases)}
        )
    if not molecules:
        raise ValueError("reasoning grammar requires at least one visible molecule")

    path = prompt_asset_path(prompt_version, str(template_name)).relative_to(TEMPLATE_DIR)
    return template_environment().get_template(str(path)).render(
        molecules=molecules, task=task
    )
